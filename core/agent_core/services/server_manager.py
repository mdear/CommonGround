import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mcp.client.session as mcp_session
from fastapi import FastAPI
from mcp.client.session_group import ClientSessionGroup, StdioServerParameters, StreamableHttpParameters
from .file_monitor import start_file_monitoring
from ..iic.core.iic_handlers import BASE_DIR
from ..config.app_config import get_native_mcp_servers
# --- START: Modified code ---
# Import initialize_registry from tool_registry
from ..framework.tool_registry import initialize_registry
# Import session security initialization
from api.session import initialize_session_security
# --- END: Modified code ---

logger = logging.getLogger(__name__)

# --- Global variables ---
# --- START: Modified code - Completely remove global SESSION_GROUP ---
# SESSION_GROUP: Optional[ClientSessionGroup] = None
# --- END: Modified code ---
MCP_SESSION_POOL: Optional[asyncio.Queue] = None

# --- START: Modified code - Remove initialize_session_group_and_tools and shutdown_session_group ---
# async def initialize_session_group_and_tools(): ...
# async def shutdown_session_group(): ...
# --- END: Modified code ---

@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """
    [Modified] FastAPI's lifespan manager, implementing a better "discover and pool" logic.
    """
    global MCP_SESSION_POOL

    # Re-configure logging after uvicorn potentially modified it
    try:
        from ..config.logging_config import setup_global_logging
        import os
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_file = os.getenv("LOG_FILE", None)
        setup_global_logging(log_level, log_file)

        # Additional suppression for MCP-related warnings
        import logging
        logging.getLogger("mcp").setLevel(logging.ERROR)
        logging.getLogger("mcp.client").setLevel(logging.ERROR)
        logging.getLogger("mcp.server").setLevel(logging.ERROR)

        logger.info("logging_reconfigured_in_lifespan", extra={"description": "Logging reconfigured in lifespan manager"})
    except Exception as e:
        logger.error("logging_reconfiguration_failed", extra={"description": "Failed to reconfigure logging", "error": str(e)})

    logger.info("application_startup_begin", extra={"description": "Initializing resources via lifespan manager"})

    # Initialize session security (JWT + fingerprint cookies)
    try:
        initialize_session_security()
        logger.info("session_security_initialized", extra={"description": "Session security manager initialized"})
    except Exception as e:
        logger.error("session_security_init_failed", extra={"description": "Failed to initialize session security", "error": str(e)})
        raise

    # 1. Initialize an empty session pool
    MCP_SESSION_POOL = asyncio.Queue()
    logger.info("mcp_session_pool_initialized", extra={"description": "MCP Session Pool initialized"})

    # 2. Create a session group specifically for tool discovery
    logger.info("tool_discovery_session_create_begin", extra={"description": "Creating a session group for initial tool discovery"})
    discovery_session_group = await initialize_mcp_session_for_context()

    # 3. Use this session group to initialize the tool registry
    if discovery_session_group:
        logger.info("tool_registry_init_begin", extra={"description": "Passing discovery session to initialize the tool registry"})
        await initialize_registry(discovery_session_group, "agent_core/nodes/custom_nodes")

        # 4. [Core] Put the session group used for discovery directly into the pool as the first available resource
        await release_mcp_session_to_pool(discovery_session_group)
        logger.info("tool_discovery_session_pooled", extra={"description": "Tool discovery session has been successfully pooled for reuse"})
    else:
        logger.warning("tool_discovery_session_failed", extra={"description": "Failed to create discovery session group. No native MCP tools will be registered"})
        # Even if discovery fails, continue to initialize an empty registry
        await initialize_registry(None, "agent_core/nodes/custom_nodes")

    logger.info("file_monitor_start", extra={"description": "Starting file monitor", "directory": BASE_DIR})
    start_file_monitoring(BASE_DIR, loop=asyncio.get_running_loop())
    # Core of the Lifespan Manager: yield control to the FastAPI application
    yield

    # When the application shuts down, this code will be executed
    logger.info("application_shutdown_begin", extra={"description": "Cleaning up resources via lifespan manager"})

    # Signal all WebSocket connections to close gracefully
    try:
        from api.server import shutdown_event
        shutdown_event.set()
        logger.info("shutdown_event_signaled", extra={"description": "Signaled all WebSocket connections to close"})
        # Give WebSockets a moment to close gracefully
        await asyncio.sleep(2)
    except Exception as e:
        logger.warning("shutdown_event_signal_failed", extra={"error": str(e)})

    # 5. When the application shuts down, clean up all remaining sessions in the pool
    if MCP_SESSION_POOL:
        logger.info("mcp_session_pool_cleanup_begin", extra={"description": "Closing idle MCP sessions from the pool", "session_count": MCP_SESSION_POOL.qsize()})
        while not MCP_SESSION_POOL.empty():
            try:
                session_to_close = MCP_SESSION_POOL.get_nowait()
                try:
                    await session_to_close.__aexit__(None, None, None)
                except Exception:
                    logger.warning("mcp_session_graceful_exit_failed", extra={"description": "DUE to MCP SDK Limitation - this session can not gracefully exit. Accumulating too many this error might drain your resource, but it should be fine as far as you are not running this as a service"})
            except asyncio.QueueEmpty:
                break
            # This outer exception handler is now only for get_nowait()
            except Exception as e:
                logger.error("mcp_session_pool_shutdown_error", extra={"description": "Error retrieving session from pool during shutdown", "error": str(e)}, exc_info=True)
        logger.info("mcp_session_pool_cleanup_complete", extra={"description": "All idle MCP sessions from the pool have been closed"})


    logger.info("application_shutdown_complete", extra={"description": "Application shutdown complete"})

# --- START: Modified code - Remove get_session_group ---
# def get_session_group() -> Optional[ClientSessionGroup]:
#     return SESSION_GROUP
# --- END: Modified code ---

async def initialize_mcp_session_for_context() -> Optional[ClientSessionGroup]:
    """
    Creates a new ClientSessionGroup instance for a single context and connects to all servers.
    Returns a connected ClientSessionGroup instance, or None on failure.
    """
    logger.info("mcp_session_group_init_begin", extra={"description": "Initializing a new ClientSessionGroup for a specific agent context"})

    session_group = ClientSessionGroup()

    native_mcp_servers = get_native_mcp_servers()
    server_connections = []
    for name, conf in native_mcp_servers.items():
        transport = conf.get("transport")
        params = None
        if transport == "stdio":
            params = StdioServerParameters(command=conf["command"], args=conf["args"])
        elif transport == "http":
            params = StreamableHttpParameters(url=conf["url"])
        else:
            logger.warning("mcp_unsupported_transport", extra={"description": "Unsupported transport type for server", "transport": transport, "server_name": name})
            continue
        server_connections.append({"name": name, "params": params})

    connect_tasks = [session_group.connect_to_server(conn["params"]) for conn in server_connections]
    results = await asyncio.gather(*connect_tasks, return_exceptions=True)

    for i, result_or_exc in enumerate(results):
        server_name = server_connections[i]["name"]
        if isinstance(result_or_exc, Exception):
            logger.error("mcp_server_connection_failed", extra={"description": "Context-specific MCP connection failed", "server_name": server_name, "error": str(result_or_exc)}, exc_info=result_or_exc)
            return None
        else:
            result_or_exc.server_name_from_config = server_name
            logger.info("mcp_server_connection_success", extra={"description": "Context-specific MCP connection successful", "server_name": server_name})

    return session_group

# --- START: New code - Session pool management functions ---
async def acquire_mcp_session_from_pool(max_retries=3) -> Optional[ClientSessionGroup]: # Add retries in case the pool is full of bad connections
    """
    (Modified) Acquires a healthy MCP session group from the global pool.
    Performs a health check before returning the session.
    """
    if MCP_SESSION_POOL is None:
        logger.error("mcp_session_pool_not_initialized", extra={"description": "MCP Session Pool is not initialized"})
        return None

    for attempt in range(max_retries):
        try:
            # 1. Try to get a session from the pool
            session_group = MCP_SESSION_POOL.get_nowait()
            logger.info("mcp_session_acquired_from_pool", extra={"description": "Acquired a session from pool, performing health check", "pool_size": MCP_SESSION_POOL.qsize()})

            # 2. Perform a health check on each session
            is_healthy = True
            if session_group.sessions: # Ensure the session_group contains sessions
                # We only need to ping one session to confirm if the underlying process for the whole group is alive
                # Usually, one group corresponds to one process
                try:
                    # Use asyncio.wait_for with a short timeout to prevent ping from blocking
                    await asyncio.wait_for(session_group.sessions[0].send_ping(), timeout=3.0)
                    logger.info("mcp_session_health_check_passed", extra={"description": "Health check PASSED. Returning session to caller"})
                    return session_group # Healthy, return the session
                except Exception as e:
                    logger.warning("mcp_session_health_check_failed", extra={"description": "Health check FAILED for a pooled session. Discarding it", "error": str(e)})
                    is_healthy = False
                    # Make sure to close this invalid session group to release resources
                    try:
                        await session_group.__aexit__(None, None, None)
                    except Exception:
                        logger.warning("mcp_session_cleanup_limitation", extra={"description": "DUE to MCP SDK Limitation - this session can not gracefully exit. Accumulating too many this error might drain your resource, but it should be fine as far as you are not running this as a service"})

            if not is_healthy:
                # If not healthy, continue the loop to try and get the next one
                continue

        except asyncio.QueueEmpty:
            # 3. If the pool is empty, create a new session
            logger.info("mcp_session_pool_empty_creating_new", extra={"description": "MCP session pool is empty. Creating a new session"})
            return await initialize_mcp_session_for_context()

    # If a healthy session cannot be found after multiple loops
    logger.error("mcp_session_acquire_failed_creating_fallback", extra={"description": "Failed to acquire a healthy MCP session after retries. Creating a new one as a last resort", "max_retries": max_retries})
    return await initialize_mcp_session_for_context()

async def release_mcp_session_to_pool(session_group: ClientSessionGroup):
    """
    Releases a used MCP session group back to the global pool.
    """
    if MCP_SESSION_POOL is None:
        logger.error("mcp_session_pool_not_initialized_release", extra={"description": "MCP Session Pool is not initialized! Cannot release session"})
        return
    if session_group:
        await MCP_SESSION_POOL.put(session_group)
        logger.info("mcp_session_released_to_pool", extra={"description": "MCP session released back to the pool", "pool_size": MCP_SESSION_POOL.qsize()})
# --- END: New code ---


async def reconnect_mcp_server(session_group: ClientSessionGroup, server_name: str) -> bool:
    """
    Attempts to reconnect a specific MCP server within a session group.

    This handles the case where a server connection was lost (e.g., server restart,
    network error) and needs to be re-established without recreating the entire session group.

    Args:
        session_group: The ClientSessionGroup containing the dead connection
        server_name: The name of the server to reconnect (from mcp.json config)

    Returns:
        True if reconnection succeeded, False otherwise
    """
    logger.info("mcp_server_reconnect_begin", extra={
        "description": "Attempting to reconnect to MCP server",
        "server_name": server_name
    })

    try:
        # 1. Get the server configuration
        native_mcp_servers = get_native_mcp_servers()
        if server_name not in native_mcp_servers:
            logger.error("mcp_server_reconnect_config_not_found", extra={
                "description": "Server configuration not found for reconnection",
                "server_name": server_name
            })
            return False

        conf = native_mcp_servers[server_name]
        transport = conf.get("transport")

        # 2. Build connection parameters
        params = None
        if transport == "stdio":
            params = StdioServerParameters(command=conf["command"], args=conf["args"])
        elif transport == "http":
            params = StreamableHttpParameters(url=conf["url"])
        else:
            logger.error("mcp_server_reconnect_unsupported_transport", extra={
                "description": "Unsupported transport type for reconnection",
                "transport": transport,
                "server_name": server_name
            })
            return False

        # 3. Find and disconnect the dead session for this server
        dead_session = None
        for session in list(session_group.sessions):
            # Check if this session belongs to the server we want to reconnect
            if hasattr(session, 'server_name_from_config') and session.server_name_from_config == server_name:
                dead_session = session
                break

        if dead_session:
            try:
                await session_group.disconnect_from_server(dead_session)
                logger.info("mcp_server_dead_session_disconnected", extra={
                    "description": "Dead session disconnected",
                    "server_name": server_name
                })
            except Exception as e:
                # Even if disconnect fails, continue with reconnection
                logger.warning("mcp_server_disconnect_warning", extra={
                    "description": "Warning during dead session disconnect (continuing anyway)",
                    "server_name": server_name,
                    "error": str(e)
                })

        # 4. Connect to the server with fresh connection
        new_session = await session_group.connect_to_server(params)
        new_session.server_name_from_config = server_name

        logger.info("mcp_server_reconnect_success", extra={
            "description": "Successfully reconnected to MCP server",
            "server_name": server_name
        })
        return True

    except Exception as e:
        logger.error("mcp_server_reconnect_failed", extra={
            "description": "Failed to reconnect to MCP server",
            "server_name": server_name,
            "error": str(e)
        }, exc_info=True)
        return False

