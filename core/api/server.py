import json
import logging
import asyncio
import os
import shutil
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Request, Query, UploadFile, File, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from agent_core.iic.core.event_handler import EventHandler as IICEventHandler

# from pydantic import BaseModel, Field # BaseModel, Field are no longer needed as SessionRequest will be removed

# Remove imports for get_session, remove_session as they are part of the old session management
# create_session is still needed, but its invocation will change
from .session import (
    create_session,
    pending_websocket_sessions,
    active_runs_store,
    active_event_managers,
    initialize_session_security,
    refresh_session_tokens,
    validate_session_jwt,
)
from api.events import SessionEventManager
from agent_core.iic.core.iic_handlers import list_projects, get_project, create_project, delete_project, update_project, update_run_meta, delete_run, update_run_name, move_iic
# Import server_manager startup/shutdown functions
from agent_core.services.server_manager import lifespan_manager
# Import the new message handler registry
from .message_handlers import MESSAGE_HANDLERS
# Import metadata related modules
from .metadata import fetch_metadata, MetadataResponse
# Import connection manager for resilient WebSocket handling
from .connection_manager import connection_manager, ConnectionState

# Configure logging (can be handled by run_server.py, or called here as well)
logger = logging.getLogger(__name__)

# Global shutdown event to signal WebSocket connections to close gracefully
shutdown_event = asyncio.Event()

# Create FastAPI application
app = FastAPI(title="PocketFlow Search Agent API", lifespan=lifespan_manager)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3800",
        "http://127.0.0.1:3800",
        "http://localhost:8000",  # Add FastAPI port
        "http://127.0.0.1:8000",
        "http://localhost:8800",
        "http://127.0.0.1:8800",
    ],  # Allow frontend development environment and FastAPI access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)

# Frontend file configuration
frontend_path = "frontend/out"

if os.path.exists(frontend_path):
    # Redirect root path to /webview/
    @app.get("/", include_in_schema=False)
    async def redirect_to_webview():
        """Redirect root path to the frontend application"""
        return RedirectResponse(url="/webview/", status_code=302)

    # Handle all requests under the /webview/ path
    @app.get("/webview/{file_path:path}", include_in_schema=False)
    async def serve_webview_files(file_path: str = ""):
        """Handle frontend files and routes under the /webview/ path"""
        # If the path is empty, default to returning index.html
        if not file_path or file_path == "/":
            file_path = "index.html"

        # Try to find the corresponding file
        full_file_path = os.path.join(frontend_path, file_path)

        # If the file exists, return it directly
        if os.path.isfile(full_file_path):
            return FileResponse(full_file_path)

        # If it's a directory, try to return index.html from within it
        if os.path.isdir(full_file_path):
            index_in_dir = os.path.join(full_file_path, "index.html")
            if os.path.isfile(index_in_dir):
                return FileResponse(index_in_dir)

        # New: If the file is not found, try adding the .html extension
        if not file_path.endswith('.html'):
            html_file_path = os.path.join(frontend_path, file_path + ".html")
            if os.path.isfile(html_file_path):
                return FileResponse(html_file_path)

        # If nothing is found, check for a 404.html
        error_404_path = os.path.join(frontend_path, "404.html")
        if os.path.isfile(error_404_path):
            return FileResponse(error_404_path, status_code=404)

        # If there isn't even a 404.html, return a simple 404
        raise HTTPException(status_code=404, detail="Page not found")


# SessionRequest Pydantic model is removed as POST /session now expects an empty body.

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint handler with resilient connection management.

    Features:
    - Heartbeat (ping/pong) to detect stale connections
    - Reconnection grace period for temporary disconnects
    - Event buffering during disconnection
    """

    # 1. Verify if the session_id is valid and not in use
    if session_id not in pending_websocket_sessions:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or expired session ID")
        logger.warning("websocket_connection_rejected", extra={"session_id": session_id, "reason": "invalid_expired_session"})
        return

    # Remove from the pending list, marking it as used
    del pending_websocket_sessions[session_id]
    logger.debug("websocket_session_validated", extra={"session_id": session_id})


    # 3. Create a new SessionEventManager instance for this WebSocket connection
    #    and store it in websocket.state.event_manager
    #    session_id is passed to SessionEventManager mainly for logging purposes.
    event_manager = SessionEventManager(session_id=session_id)
    websocket.state.event_manager = event_manager
    websocket.state.session_id = session_id  # Store session_id for message handlers

    # 4. Initialize the list of tasks associated with this WebSocket connection
    websocket.state.active_run_tasks: Dict[str, asyncio.Task] = {} # type: ignore
    websocket.state.active_run_id: Optional[str] = None  # Track active run for reconnection
    websocket.state.websocket = websocket  # Store reference for reconnection handlers

    try:
        await websocket.accept()
        iic_eh = IICEventHandler()
        # Attach the event_manager to websocket.state, making it available throughout the connection lifecycle
        websocket.state.event_manager.connect(websocket)
        websocket.state.event_manager.attach(iic_eh.on_message)
        # Add the event_manager to the global list
        active_event_managers.append(event_manager)

        # Register connection with connection manager and start heartbeat
        connection_manager.register_connection(session_id, websocket, event_manager)
        await connection_manager.start_heartbeat(session_id)

        logger.info("websocket_connection_accepted", extra={"session_id": session_id})

        # Main message processing loop
        while True:
            # Check if server is shutting down
            if shutdown_event.is_set():
                logger.info("websocket_closing_due_to_shutdown", extra={"session_id": session_id})
                await websocket.close(code=status.WS_1001_GOING_AWAY, reason="Server shutting down")
                break

            try:
                # Use wait_for with timeout to periodically check shutdown event
                message_json_str = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=5.0  # Check for shutdown every 5 seconds
                )
                message = json.loads(message_json_str)
                message_type = message.get("type")
                message_data = message.get("data", {})

                logger.debug("websocket_message_received", extra={"session_id": session_id, "message_type": message_type, "raw_preview": message_json_str[:200]})

                # Handle heartbeat pong responses
                if message_type == "pong":
                    connection_manager.handle_pong(session_id)
                    continue

                handler = MESSAGE_HANDLERS.get(message_type)
                if handler:
                    # All handlers now expect (websocket.state, data) as parameters
                    await handler(websocket.state, message_data)
                else:
                    logger.warning("websocket_unknown_message_type", extra={"session_id": session_id, "message_type": message_type})
                    if hasattr(websocket.state, 'event_manager') and websocket.state.event_manager:
                        await websocket.state.event_manager.emit_error(run_id=None, agent_id="System", error_message=f"Unknown message type: {message_type}")

            except asyncio.TimeoutError:
                # This block is used to periodically check completed tasks
                done_run_ids = []
                active_tasks_loop = websocket.state.active_run_tasks if hasattr(websocket.state, 'active_run_tasks') else {}

                if not active_tasks_loop:
                    continue

                for run_id_iter, task_iter in list(active_tasks_loop.items()):
                    if task_iter.done():
                        done_run_ids.append(run_id_iter)
                        try:
                            exc = task_iter.exception()
                            if exc:
                                logger.error("run_task_ended_with_exception", extra={"run_id": run_id_iter, "session_id": session_id, "error": str(exc)}, exc_info=exc)
                                # Error events should be sent by the business flow internally or by handle_start_run's exception block
                            else:
                                logger.info("run_task_completed_normally", extra={"run_id": run_id_iter, "session_id": session_id})
                                # Successful completion events should be sent by the business flow internally
                        except asyncio.CancelledError:
                            logger.info("run_task_cancelled", extra={"run_id": run_id_iter, "session_id": session_id, "reason": "timeout_loop"})
                        except Exception as e_check:
                            logger.error("run_task_completion_check_error", extra={"run_id": run_id_iter, "session_id": session_id, "error": str(e_check)}, exc_info=True)

                for run_id_to_remove in done_run_ids:
                    if run_id_to_remove in active_tasks_loop:
                        del active_tasks_loop[run_id_to_remove]
                        logger.info("run_task_removed", extra={"run_id": run_id_to_remove, "session_id": session_id, "reason": "completed_cancelled"})
                    # Also remove from global store if it's truly finished and not just task done (e.g. cancelled)
                    # This cleanup might be better handled by the flow itself or when stop_run is called.
                    # For now, just removing from websocket.state.active_run_tasks.
                    # if run_id_to_remove in active_runs_store:
                    #     del active_runs_store[run_id_to_remove]
                    #     logger.info(f"Removed run context {run_id_to_remove} from active_runs_store.")
                continue

            except WebSocketDisconnect:
                logger.info("websocket_disconnected", extra={"session_id": session_id})
                break

            except json.JSONDecodeError as json_err:
                logger.error("websocket_invalid_json", extra={"session_id": session_id, "raw_message": message_json_str, "error": str(json_err)})
                if hasattr(websocket.state, 'event_manager') and websocket.state.event_manager:
                    await websocket.state.event_manager.emit_error(run_id=None, agent_id="System", error_message="Invalid JSON message received.")

            except Exception as e:
                logger.error("websocket_unexpected_error", extra={"session_id": session_id, "error": str(e)}, exc_info=True)
                if hasattr(websocket.state, 'event_manager') and websocket.state.event_manager:
                    await websocket.state.event_manager.emit_error(run_id=None, agent_id="System", error_message=f"Unexpected WebSocket processing error: {e}")
                break
    finally:
        # Use connection manager to handle disconnection with grace period
        # This will NOT immediately cancel tasks - they continue running during grace period
        runs_in_grace = connection_manager.unregister_connection(session_id)

        active_tasks_at_close = websocket.state.active_run_tasks if hasattr(websocket.state, 'active_run_tasks') else {}

        if runs_in_grace:
            # Runs are in grace period - tasks keep running, events will be buffered
            logger.info("websocket_disconnect_grace_period_started", extra={
                "session_id": session_id,
                "runs_in_grace": runs_in_grace,
                "active_task_count": len(active_tasks_at_close)
            })
            # Register tasks with connection manager so they can be tracked
            for run_id in runs_in_grace:
                run_state = connection_manager.get_run_state(run_id)
                if run_state and run_id in active_tasks_at_close:
                    run_state.tasks[run_id] = active_tasks_at_close[run_id]
        else:
            # No active runs in grace period, proceed with immediate cleanup (legacy behavior)
            logger.info("websocket_cleanup_active_runs", extra={"session_id": session_id, "active_run_count": len(active_tasks_at_close)})

            for run_id_final, task_final in list(active_tasks_at_close.items()):
                if not task_final.done():
                    logger.info("websocket_cancelling_run_task", extra={"run_id": run_id_final, "session_id": session_id, "phase": "final_cleanup"})
                    task_final.cancel()
                    try:
                        await task_final
                    except asyncio.CancelledError:
                        logger.info("run_task_cancelled_success", extra={"run_id": run_id_final, "session_id": session_id, "phase": "final_cleanup"})
                    except Exception as e_final_cancel:
                        logger.error("run_task_cancellation_error", extra={"run_id": run_id_final, "session_id": session_id, "phase": "final_cleanup", "error": str(e_final_cancel)}, exc_info=True)

                # Clean up run_context from global store for this run_id
                if run_id_final in active_runs_store:
                    del active_runs_store[run_id_final]
                    logger.info("run_context_removed", extra={"session_id": session_id, "run_id": run_id_final, "phase": "final_cleanup"})
                else:
                    logger.warning("run_context_not_found", extra={"session_id": session_id, "run_id": run_id_final, "phase": "final_cleanup", "possible_cause": "already_removed_by_stop_run"})

            if hasattr(websocket.state, 'active_run_tasks'):
                websocket.state.active_run_tasks.clear()

        # Remove event_manager from the global list upon disconnect
        if event_manager in active_event_managers:
            active_event_managers.remove(event_manager)

        if hasattr(websocket.state, 'event_manager') and websocket.state.event_manager:
            await websocket.state.event_manager.disconnect()

        logger.info("websocket_handling_finished", extra={"session_id": session_id})


# --- Session Security Models ---

class SessionCreateRequest(BaseModel):
    """Request body for session creation."""
    project_id: str = "default"


class RefreshTokenRequest(BaseModel):
    """Request body for token refresh."""
    refresh_token: str


@app.post("/session")
async def create_new_session(
    response: Response,
    body: Optional[SessionCreateRequest] = None,
    __Secure_Fgp: Optional[str] = Cookie(None, alias="__Secure-Fgp"),
):
    """Creates a new secure WebSocket connection credential with JWT.

    This endpoint creates:
    - A unique session_id
    - A signed JWT token bound to a fingerprint cookie
    - A refresh token for silent token renewal

    The fingerprint cookie is set automatically (HttpOnly, Secure, SameSite=Strict).
    If an existing fingerprint cookie is present (reconnection), it will be reused.

    Returns:
        session_id: Unique session identifier
        jwt_token: Signed JWT for WebSocket authentication
        refresh_token: Token for silent refresh before JWT expiry
        expires_in: Seconds until JWT expires
    """
    try:
        project_id = body.project_id if body else "default"
        result = await create_session(
            response=response,
            project_id=project_id,
            existing_fingerprint=__Secure_Fgp,  # Reuse fingerprint on reconnection
        )
        return result
    except Exception as e:
        logger.error("session_credential_creation_failed", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create session credential: {str(e)}")


@app.post("/session/refresh")
async def refresh_session(
    body: RefreshTokenRequest,
    __Secure_Fgp: Optional[str] = Cookie(None, alias="__Secure-Fgp"),
):
    """Refresh session tokens using a refresh token.

    Implements token rotation: each refresh token can only be used once.
    Token reuse (replaying an old refresh token) is detected and triggers
    revocation of all session tokens as a security measure.

    Requires:
    - refresh_token in request body
    - __Secure-Fgp cookie (automatically sent by browser)

    Returns:
        jwt_token: New signed JWT
        refresh_token: New refresh token (old one is invalidated)
        expires_in: Seconds until new JWT expires
    """
    result = await refresh_session_tokens(
        refresh_token=body.refresh_token,
        fingerprint_cookie=__Secure_Fgp,
    )

    if not result:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired refresh token"
        )

    return {
        "jwt_token": result.jwt_token,
        "refresh_token": result.refresh_token,
        "expires_in": result.expires_in,
    }


# --- Connection Resilience Endpoints ---

@app.get("/run/{run_id}/status")
async def get_run_connection_status(run_id: str):
    """Get the connection status of a run.

    Returns whether the run is active, in grace period, or terminated,
    and whether it can be reconnected to.
    """
    run_state = connection_manager.get_run_state(run_id)

    if not run_state:
        return {
            "run_id": run_id,
            "exists": False,
            "can_reconnect": False,
            "message": "Run not found or already terminated"
        }

    return {
        "run_id": run_id,
        "exists": True,
        "state": run_state.state.value,
        "can_reconnect": connection_manager.can_reconnect(run_id),
        "connected_at": run_state.connected_at.isoformat() if run_state.connected_at else None,
        "disconnected_at": run_state.disconnected_at.isoformat() if run_state.disconnected_at else None,
        "grace_period_expires": run_state.grace_period_expires.isoformat() if run_state.grace_period_expires else None,
        "buffered_events": len(run_state.event_buffer),
        "last_checkpoint": run_state.last_checkpoint.isoformat() if run_state.last_checkpoint else None
    }


@app.get("/runs/active")
async def get_active_runs():
    """Get all active runs (connected or in grace period)."""
    active_run_ids = connection_manager.get_active_run_ids()
    runs_info = []

    for run_id in active_run_ids:
        run_state = connection_manager.get_run_state(run_id)
        if run_state:
            runs_info.append({
                "run_id": run_id,
                "state": run_state.state.value,
                "session_id": run_state.session_id,
                "can_reconnect": connection_manager.can_reconnect(run_id)
            })

    return {
        "active_runs": runs_info,
        "stats": connection_manager.get_stats()
    }


@app.get("/runs/grace-period")
async def get_runs_in_grace_period():
    """Get all runs currently in reconnection grace period."""
    grace_period_runs = connection_manager.get_runs_in_grace_period()
    runs_info = []

    for run_id in grace_period_runs:
        run_state = connection_manager.get_run_state(run_id)
        if run_state:
            runs_info.append({
                "run_id": run_id,
                "session_id": run_state.session_id,
                "disconnected_at": run_state.disconnected_at.isoformat() if run_state.disconnected_at else None,
                "grace_period_expires": run_state.grace_period_expires.isoformat() if run_state.grace_period_expires else None,
                "buffered_events": len(run_state.event_buffer)
            })

    return {
        "runs_in_grace_period": runs_info,
        "count": len(runs_info)
    }


# --- Project Management Endpoints ---
@app.get("/projects")
async def get_projects():
    """List all projects."""
    return list_projects()

@app.get("/project/{project_id}")
async def get_project_details(project_id: str):
    """Get details of a specific project."""
    return get_project(project_id)

@app.post("/project")
async def create_new_project(request: Request):
    """Create a new project."""
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' in request body.")
    return create_project(name)

@app.post("/project/{project_id}/upload")
async def upload_file_to_project(project_id: str, file: UploadFile = File(...)):
    """
    Uploads a file to a specific project's 'assets' subdirectory.
    The file monitor will automatically detect and index it.
    """
    from agent_core.iic.core.iic_handlers import get_iic_dir

    project_path_str = get_iic_dir(project_id)
    if not project_path_str or not os.path.isdir(project_path_str):
        raise HTTPException(status_code=404, detail=f"Project with ID '{project_id}' not found.")

    assets_dir = os.path.join(project_path_str, "assets")
    try:
        os.makedirs(assets_dir, exist_ok=True)
    except OSError as e:
        logger.error("assets_directory_creation_failed", extra={"assets_dir": assets_dir, "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not create assets directory on the server.")

    # Simple security check to prevent filename from containing path traversal characters
    # os.path.basename will remove path information, providing a layer of protection
    safe_filename = os.path.basename(file.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_location = os.path.join(assets_dir, safe_filename)

    if os.path.exists(file_location):
        raise HTTPException(status_code=409, detail=f"File '{safe_filename}' already exists in this project's assets.")

    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        logger.info("file_upload_success", extra={"filename": safe_filename, "project_id": project_id, "file_location": file_location})
        return {"info": f"File '{safe_filename}' uploaded successfully to project '{project_id}'."}
    except Exception as e:
        logger.error("file_upload_save_failed", extra={"file_location": file_location, "error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")

@app.put("/project/{project_id}")
async def update_project_meta(project_id: str, request: Request):
    """Update the meta block of a project with the provided key-value pairs."""
    try:
        updates = await request.json()
        if not isinstance(updates, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return update_project(project_id, updates)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating project: {str(e)}")

@app.put("/run/{run_id}")
async def update_run_metadata(run_id: str, request: Request):
    """Update the meta block of a run with the provided key-value pairs."""
    try:
        updates = await request.json()
        if not isinstance(updates, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return await update_run_meta(run_id, updates)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating run metadata: {str(e)}")

@app.delete("/project/{project_id}")
async def delete_existing_project(project_id: str):
    """Delete a project (soft delete)."""
    return await delete_project(project_id)

@app.delete("/run/{run_id}")
async def delete_existing_run(run_id: str):
    """Delete a run (soft delete)."""
    return await delete_run(run_id)

@app.put("/run/{run_id}/name")
async def update_existing_run_name(run_id: str, request: Request):
    """Rename a run file at the filesystem level."""
    try:
        body = await request.json()
        new_name = body.get("new_name")
        if not new_name:
            raise HTTPException(status_code=400, detail="Missing 'new_name' in request body.")
        return await update_run_name(run_id, new_name)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error renaming run: {str(e)}")

@app.post("/run/move")
async def move_run_between_projects(request: Request):
    """Move a run from one project to another."""
    try:
        body = await request.json()
        run_id = body.get("run_id")
        from_project_id = body.get("from_project_id")
        to_project_id = body.get("to_project_id")

        # Validate required fields
        if not run_id:
            raise HTTPException(status_code=400, detail="Missing 'run_id' in request body.")
        if not from_project_id:
            raise HTTPException(status_code=400, detail="Missing 'from_project_id' in request body.")
        if not to_project_id:
            raise HTTPException(status_code=400, detail="Missing 'to_project_id' in request body.")

        # Prevent moving to the same project
        if from_project_id == to_project_id:
            raise HTTPException(status_code=400, detail="Source and destination projects cannot be the same.")

        return move_iic(run_id, from_project_id, to_project_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error moving run: {str(e)}")

# --- Report Download Endpoint ---
@app.get("/api/reports/{project_id}/{filename}")
async def download_report(project_id: str, filename: str):
    """Serve a final report file for download.
    
    Reports are saved when Principal completes each epoch.
    URL format: /api/reports/{project_id}/{run_id}_epoch{N}.md
    """
    # Security: validate both project_id and filename to prevent path traversal
    safe_project_id = os.path.basename(project_id)
    if not safe_project_id or safe_project_id != project_id or '..' in project_id:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename or '..' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Construct path to report file
    report_path = os.path.join("projects", safe_project_id, "reports", safe_filename)
    
    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Serve the file with appropriate headers for download
    return FileResponse(
        path=report_path,
        filename=safe_filename,
        media_type="text/markdown"
    )

# --- Metadata Endpoint ---
@app.get("/metadata", response_model=MetadataResponse)
async def get_metadata(url: str = Query(..., description="The URL for which to fetch metadata")):
    """Fetches OpenGraph and other metadata information for the specified URL

    Args:
        url: The web page URL to fetch information from

    Returns:
        MetadataResponse: A response containing title, description, image, favicon, and domain
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        metadata = await fetch_metadata(url)
        return metadata
    except Exception as e:
        logger.error("metadata_fetch_failed", extra={"url": url, "error": str(e)})
        raise HTTPException(status_code=500, detail="Failed to fetch metadata")


