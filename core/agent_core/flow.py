import logging
import asyncio
from pocketflow import AsyncFlow
from .nodes.base_agent_node import AgentNode # <--- Import AgentNode
from .services.server_manager import acquire_mcp_session_from_pool, release_mcp_session_to_pool
from .utils.debug_helper import dump_state_to_file
from .framework.tool_registry import connect_tools_to_node
import uuid # <--- Import uuid module
import os

from datetime import datetime, timezone # Added as per design doc

# Note: Logging is now configured globally in run_server.py via setup_global_logging()
# This basicConfig call has been removed to avoid conflicts with structured logging
logger = logging.getLogger(__name__)

# Create custom ProjectFlow class
class ProjectFlow(AsyncFlow):
    """Project-specific flow class that extends AsyncFlow and adds custom preparation logic"""

async def run_principal_async(principal_context: dict):
    """Asynchronously run the Principal flow of the agent"""
    logger = logging.getLogger(__name__)
    turn_manager = principal_context['refs']['run']['runtime'].get('turn_manager')

    # ... (Session recording code remains unchanged) ...
    team_state = principal_context['refs']['team']
    new_session_record = {
        "session_id": f"principal_session_{len(team_state.get('principal_execution_sessions', [])) + 1}",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "termination_reason": None
    }
    team_state.setdefault("principal_execution_sessions", []).append(new_session_record)
    logger.info("principal_flow_session_started", extra={"session_record": new_session_record})

    # ==================== Final modification starts ====================
    # --- START: Modified code - Using session pool ---
    principal_mcp_session_group = await acquire_mcp_session_from_pool()
    # --- END: Modified code ---

    if principal_mcp_session_group:
        principal_context.setdefault("runtime_objects", {})["mcp_session_group"] = principal_mcp_session_group
    else:
        logger.error("principal_mcp_session_acquisition_failed", extra={"mcp_tools_available": False}, exc_info=True)
    
    # Get required information from principal_context
    question = principal_context['refs']['team'].get("question")
    events = principal_context['refs']['run']['runtime']['event_manager']
    current_run_id = principal_context['meta']['run_id']

    if not question:
        error_msg = f"Run {current_run_id}: Question not found in principal_context refs for run_principal_async."
        logger.error("principal_question_not_found", extra={"run_id": current_run_id, "error_message": error_msg}, exc_info=True)
        if events: await events.emit_error(run_id=current_run_id, agent_id="System", error_message=error_msg)
        raise ValueError(error_msg)
    
    from .state.management import validate_context_state, update_context_activity 
    if not validate_context_state(principal_context): 
        raise ValueError("Principal context state is invalid: missing required fields")
    update_context_activity(principal_context)

    principal_node = AgentNode(
        profile_id="Principal", 
        agent_id_override="Principal", 
        shared_for_init=principal_context,
        parent_agent_id=principal_context['meta'].get("parent_agent_id")
    )
    
    project_flow = ProjectFlow(start=principal_node)
    principal_node - "default" >> principal_node
    connect_tools_to_node(principal_node, context=principal_context) 

    logger.debug("principal_flow_initialized", extra={"run_id": current_run_id})

    try:
        logger.debug("principal_async_run_starting", extra={"run_id": current_run_id})
        await project_flow.run_async(principal_context)
        logger.debug("principal_async_run_completed", extra={"run_id": current_run_id})
        
        if os.getenv("STATE_DUMP", "").lower() == "true":
            await dump_state_to_file(principal_context['refs']['run'])
            
        final_state = principal_context["state"]
        if "final_result_package" in final_state:
            return final_state["final_result_package"]
        else:
            return {
                "status": "COMPLETED_WITH_ERROR",
                "final_summary": "Principal flow concluded unexpectedly without a final result package.",
                "terminating_tool": None,
                "error_details": final_state.get("error_message", "Unknown reason for unexpected termination."),
                "deliverables": None
            }
    except asyncio.CancelledError:
        # ... (CancelledError handling logic remains unchanged) ...
        logger.info("principal_flow_cancelled", extra={"run_id": current_run_id})
        if turn_manager:
            turn_manager.cancel_current_turn(principal_context)
        
        if events:
            await events.emit_turns_sync(principal_context)
        return {
            "status": "CANCELLED",
            "final_summary": "Principal flow was cancelled by user or system.",
            "terminating_tool": None, "error_details": None, "deliverables": None
        }
    except Exception as e:
        # ... (Exception handling logic remains unchanged) ...
        error_msg = f"Run {current_run_id}: Error in Principal flow: {str(e)}"
        logger.error("principal_flow_error", extra={"run_id": current_run_id, "error_message": str(e)}, exc_info=True)
        if events:
            await events.emit_error(run_id=current_run_id, agent_id="System", error_message=error_msg)
        return {
            "status": "COMPLETED_WITH_ERROR",
            "final_summary": f"Principal flow failed with an exception: {error_msg}",
            "terminating_tool": None,
            "error_details": error_msg,
            "deliverables": None
        }
    finally:
        # --- START: Modified code - Return MCP session to the pool ---
        mcp_session_to_release = principal_context.get("runtime_objects", {}).get("mcp_session_group")
        if mcp_session_to_release:
            await release_mcp_session_to_pool(mcp_session_to_release)
            del principal_context["runtime_objects"]["mcp_session_group"] # Remove reference from context
        # --- END: Modified code ---

        # Removed all mcp_session_group cleanup logic
        run_context_from_refs = principal_context['refs']['run']
        completion_event = run_context_from_refs['runtime'].get("principal_completion_event")
        if completion_event:
            team_state_from_refs = principal_context['refs']['team']
            if "is_principal_flow_running" in team_state_from_refs:
                team_state_from_refs["is_principal_flow_running"] = False

            completion_event.set()
            logger.info("principal_completion_event_set", extra={"run_id": current_run_id})
        else:
            logger.warning("principal_completion_event_not_found", extra={"run_id": current_run_id})
        
        if events:
            try:
                logger.debug("principal_flow_finished", extra={"run_id": current_run_id, "websocket_active": True})
            except Exception as e_finally:
                logger.error("principal_finally_block_error", extra={"run_id": current_run_id, "error_message": str(e_finally)}, exc_info=True)
    # ==================== End of final modification ====================

async def run_associate_async(associate_context: dict):
    """Asynchronously run the agent's flow for the search agent. This flow is responsible for executing single-step tasks assigned by the PrincipalNode"""
    logger = logging.getLogger(__name__)

    # ... (dispatch_history recording code remains unchanged) ...
    team_state_from_refs = associate_context['refs']['team']
    executing_associate_id_for_history = associate_context.get("meta", {}).get("agent_id")
    
    if executing_associate_id_for_history:
        history_entry_to_update = next((
            h for h in reversed(team_state_from_refs.get("dispatch_history", []))
            if h.get("dispatch_id") == executing_associate_id_for_history and h.get("status") == "LAUNCHING"
        ), None)

        if history_entry_to_update:
            history_entry_to_update["start_timestamp"] = datetime.now(timezone.utc).isoformat()
            history_entry_to_update["status"] = "RUNNING"
            logger.info("associate_flow_started", extra={"executing_associate_id": executing_associate_id_for_history, "status": "RUNNING"})
        else:
            logger.error("associate_dispatch_history_not_found", extra={"executing_associate_id": executing_associate_id_for_history, "expected_status": "LAUNCHING"}, exc_info=True)
    else:
        logger.error("associate_id_not_found_in_params", extra={"initial_params_available": False}, exc_info=True)


    # ==================== Start of final modification ====================
    # --- START: Modified code - Use session pool ---
    associate_mcp_session_group = await acquire_mcp_session_from_pool()
    # --- END: Modified code ---
    if associate_mcp_session_group:
        associate_context.setdefault("runtime_objects", {})["mcp_session_group"] = associate_mcp_session_group
    else:
        # Continue even if it fails, allowing tools that depend on MCP to fail upon invocation
        executing_associate_id_for_log = associate_context.get("meta", {}).get("agent_id", "UnknownAssociate")
        logger.error("associate_mcp_session_acquisition_failed", extra={"executing_associate_id": executing_associate_id_for_log}, exc_info=True)

    # Get required information from associate_context
    events = associate_context['refs']['run']['runtime']['event_manager']
    current_run_id = associate_context['meta']['run_id']
    
    associate_meta = associate_context.get("meta", {})
    executing_associate_id = associate_meta.get("agent_id", f"Associate_Unknown_{uuid.uuid4().hex[:4]}")
    profile_logical_name_used = associate_meta.get("profile_logical_name", "UnknownProfile")
    profile_instance_id_used = associate_meta.get("profile_instance_id", "UnknownProfileInstance")
    
    from .state.management import validate_context_state, update_context_activity
    if not validate_context_state(associate_context): 
        logger.warning("associate_context_state_invalid", extra={"executing_associate_id": executing_associate_id, "run_id": current_run_id})
    update_context_activity(associate_context)

    logger.debug("associate_flow_initializing", extra={"run_id": current_run_id, "executing_associate_id": executing_associate_id})

    associate_node_instance = AgentNode(
        profile_id=profile_logical_name_used,
        agent_id_override=executing_associate_id,
        profile_instance_id_override=profile_instance_id_used,
        shared_for_init=associate_context,
        parent_agent_id=associate_context['meta'].get("parent_agent_id")
    )
    
    associate_workflow = AsyncFlow(start=associate_node_instance)
    associate_node_instance - "default" >> associate_node_instance
    connect_tools_to_node(associate_node_instance, context=associate_context)
    logger.debug("associate_flow_configured", extra={"run_id": current_run_id, "executing_associate_id": executing_associate_id})

    try:
        logger.debug("associate_async_run_starting", extra={"run_id": current_run_id, "executing_associate_id": executing_associate_id})
        await associate_workflow.run_async(associate_context)
        logger.debug("associate_async_run_completed", extra={"run_id": current_run_id, "executing_associate_id": executing_associate_id})
        
        if os.getenv("STATE_DUMP", "").lower() == "true":
            await dump_state_to_file(associate_context['refs']['run'])
            
        final_state = associate_context.get("state", {})
        if "deliverables" not in final_state:
            final_state["deliverables"] = {}
            logger.debug("associate_deliverables_initialized", extra={"executing_associate_id": executing_associate_id})
        
        # ===============================================================
        # HANDLE CONTEXT BUDGET HANDBACK
        # If agent exceeded context budget, update dispatch history with handback
        # ===============================================================
        deliverables = final_state.get("deliverables", {})
        if deliverables.get("status") == "CONTEXT_BUDGET_EXCEEDED":
            team_state = associate_context['refs']['team']
            handback_data = deliverables.get("_handback", {})
            
            # Find and update the dispatch history entry
            for entry in reversed(team_state.get("dispatch_history", [])):
                if entry.get("dispatch_id") == executing_associate_id:
                    entry["status"] = "PARTIAL_HANDBACK"
                    entry["handback"] = handback_data
                    entry["end_timestamp"] = datetime.now(timezone.utc).isoformat()
                    entry["termination_reason"] = "context_budget_exceeded"
                    
                    logger.info("dispatcher_history_updated_handback", extra={
                        "dispatch_id": executing_associate_id,
                        "new_status": "PARTIAL_HANDBACK",
                        "kb_tokens_available": handback_data.get("kb_token_count", 0),
                        "tool_calls_completed": len(handback_data.get("tool_calls_completed", []))
                    })
                    break
            
            # Notify Principal of handback if context available
            try:
                from .framework.context_budget_handback import (
                    ContextBudgetHandback,
                    notify_principal_of_handback
                )
                
                # Find Principal context through refs
                principal_context = associate_context.get('refs', {}).get('run', {}).get('principal_context')
                if principal_context and handback_data:
                    handback = ContextBudgetHandback.from_dict(handback_data)
                    notify_principal_of_handback(principal_context, handback)
                    logger.info("principal_notified_of_handback", extra={
                        "executing_associate_id": executing_associate_id,
                        "principal_inbox_items": len(principal_context.get("state", {}).get("inbox", []))
                    })
            except Exception as e:
                logger.warning("principal_handback_notification_failed", extra={
                    "executing_associate_id": executing_associate_id,
                    "error": str(e)
                })
        # ===============================================================
        
        return associate_context

    except asyncio.CancelledError:
        # ... (CancelledError handling logic remains unchanged) ...
        cancel_msg = f"Associate flow (Agent ID: {executing_associate_id}, Run ID: {current_run_id}) was cancelled."
        logger.info("associate_flow_cancelled", extra={"executing_associate_id": executing_associate_id, "run_id": current_run_id})
        
        # Update dispatch history with cancellation info
        team_state_from_refs = associate_context['refs']['team']
        if history_entry := next((h for h in reversed(team_state_from_refs.get("dispatch_history", [])) if h.get("dispatch_id") == executing_associate_id), None):
            history_entry["_flow_cancelled"] = True
            history_entry["_flow_cancelled_at"] = datetime.now(timezone.utc).isoformat()
        
        final_state = associate_context.setdefault("state", {})
        final_state["error_message"] = cancel_msg
        final_state.setdefault("deliverables", {})["error"] = "Flow was cancelled."
        return associate_context
    except Exception as e:
        # Enhanced exception handling with structured logging
        error_msg = f"Associate flow error (Agent ID: {executing_associate_id}, Run ID: {current_run_id}): {str(e)}"
        logger.error("associate_flow_error", extra={
            "executing_associate_id": executing_associate_id, 
            "run_id": current_run_id, 
            "error_message": str(e),
            "error_type": type(e).__name__,
            "module_id": associate_meta.get("module_id"),
            "profile": profile_logical_name_used,
        }, exc_info=True)
        
        # Update dispatch history with detailed error info
        team_state_from_refs = associate_context['refs']['team']
        if history_entry := next((h for h in reversed(team_state_from_refs.get("dispatch_history", [])) if h.get("dispatch_id") == executing_associate_id), None):
            history_entry["_flow_error"] = True
            history_entry["_flow_error_at"] = datetime.now(timezone.utc).isoformat()
            history_entry["_flow_error_type"] = type(e).__name__
            history_entry["_flow_error_message"] = str(e)[:500]  # Truncate for storage
        
        final_state = associate_context.setdefault("state", {})
        final_state["error_message"] = error_msg
        final_state.setdefault("deliverables", {})["error"] = f"Flow execution failed: {str(e)}"
        return associate_context
    finally:
        # --- START: Modified code - Return MCP session to the pool ---
        mcp_session_to_release = associate_context.get("runtime_objects", {}).get("mcp_session_group")
        if mcp_session_to_release:
            await release_mcp_session_to_pool(mcp_session_to_release)
            del associate_context["runtime_objects"]["mcp_session_group"]
        # --- END: Modified code ---
        
        # Instrumentation: Track flow finalization in dispatch history
        team_state_from_refs = associate_context.get('refs', {}).get('team', {})
        if history_entry := next((h for h in reversed(team_state_from_refs.get("dispatch_history", [])) if h.get("dispatch_id") == executing_associate_id), None):
            history_entry["_finally_block_executed"] = True
            history_entry["_finally_block_at"] = datetime.now(timezone.utc).isoformat()
        
        # Removed all mcp_session_group cleanup logic
        executing_associate_id_for_log = associate_context.get("meta", {}).get("agent_id", "UnknownAssociate")
        current_run_id_for_log = associate_context.get('meta', {}).get('run_id', 'UnknownRun')
        logger.debug("associate_flow_finished", extra={"run_id": current_run_id_for_log, "executing_associate_id": executing_associate_id_for_log, "websocket_active": True})
    # ==================== End of final modification ====================

async def run_partner_interaction_async(partner_context: dict):
    """Asynchronously run the Partner Agent's interaction flow"""
    logger = logging.getLogger(__name__)
    turn_manager = partner_context['refs']['run']['runtime'].get('turn_manager')

    events = partner_context['refs']['run']['runtime']['event_manager']
    current_run_id = partner_context['meta']['run_id']
    initial_user_query = partner_context["state"].get("question") 

    logger.info("partner_interaction_flow_initializing", extra={"run_id": current_run_id, "initial_query_preview": str(initial_user_query)[:100]})

    # ==================== Start of final modification ====================
    # --- START: Modified code - Use session pool ---
    partner_mcp_session_group = await acquire_mcp_session_from_pool()
    # --- END: Modified code ---
    if partner_mcp_session_group:
        partner_context.setdefault("runtime_objects", {})["mcp_session_group"] = partner_mcp_session_group
    else:
        logger.error("partner_mcp_session_acquisition_failed", extra={"mcp_tools_available": False}, exc_info=True)

    from .state.management import validate_context_state, update_context_activity
    if not validate_context_state(partner_context):
        logger.warning("partner_context_state_invalid", extra={"run_id": current_run_id})

    partner_agent_id = "Partner"
    partner_node = AgentNode(
        profile_id="Partner", 
        agent_id_override=partner_agent_id, 
        shared_for_init=partner_context,
        parent_agent_id=partner_context['meta'].get("parent_agent_id")
    )
    
    partner_flow = ProjectFlow(start=partner_node)
    partner_node - "default" >> partner_node
    connect_tools_to_node(partner_node, context=partner_context) 

    logger.debug("partner_interaction_flow_initialized", extra={"run_id": current_run_id, "partner_agent_id": partner_agent_id})

    try:
        while True:
            # ... (Session and event setup code remains unchanged) ...
            update_context_activity(partner_context)
            
            tasks_to_await = []
            user_input_event = partner_context['runtime_objects'].get("new_user_input_event")
            user_input_task = None
            if isinstance(user_input_event, asyncio.Event):
                user_input_task = asyncio.create_task(user_input_event.wait())
                tasks_to_await.append(user_input_task)

            principal_completion_event = partner_context['runtime_objects'].get("active_principal_completion_event")
            principal_completion_task = None
            if isinstance(principal_completion_event, asyncio.Event):
                principal_completion_task = asyncio.create_task(principal_completion_event.wait())
                tasks_to_await.append(principal_completion_task)
            
            if not tasks_to_await:
                logger.error("partner_no_events_to_wait", extra={"run_id": current_run_id, "partner_agent_id": partner_agent_id}, exc_info=True)
                break

            logger.info("partner_waiting_for_events", extra={"run_id": current_run_id, "partner_agent_id": partner_agent_id, "has_user_input_event": user_input_event is not None, "has_principal_completion_event": principal_completion_event is not None})
            
            done, pending = await asyncio.wait(tasks_to_await, return_when=asyncio.FIRST_COMPLETED)

            for task in pending:
                task.cancel()
            
            if user_input_task and user_input_task in done:
                user_input_event.clear()
                logger.debug("partner_reactivated_by_user_input", extra={"run_id": current_run_id, "partner_agent_id": partner_agent_id})
            
            elif principal_completion_task and principal_completion_task in done:
                logger.info("partner_reactivated_by_principal_completion", extra={"run_id": current_run_id, "partner_agent_id": partner_agent_id})
                if "active_principal_completion_event" in partner_context['runtime_objects']:
                    del partner_context['runtime_objects']["active_principal_completion_event"]
                logger.info("partner_completion_event_processed", extra={"run_id": current_run_id})
            
            logger.debug("partner_thinking_loop_starting", extra={"run_id": current_run_id, "partner_agent_id": partner_agent_id})
            await partner_flow.run_async(partner_context) 
            logger.debug("partner_thinking_loop_completed", extra={"run_id": current_run_id, "partner_agent_id": partner_agent_id})
    except asyncio.CancelledError:
        # ... (CancelledError handling logic remains unchanged) ...
        logger.info("partner_interaction_flow_cancelled", extra={"run_id": current_run_id})
        if turn_manager:
            turn_manager.cancel_current_turn(partner_context)
        
        if events:
            await events.emit_turns_sync(partner_context)
        raise
    except Exception as e:
        # ... (Exception handling logic remains unchanged) ...
        error_msg = f"Run {current_run_id}: Error in Partner interaction flow: {str(e)}"
        logger.error("partner_interaction_flow_error", extra={"run_id": current_run_id, "error_message": str(e)}, exc_info=True)
        if events:
            await events.emit_error(run_id=current_run_id, agent_id=partner_agent_id, error_message=error_msg)
        raise
    finally:
        # --- START: Modified code - Return MCP session to the pool ---
        mcp_session_to_release = partner_context.get("runtime_objects", {}).get("mcp_session_group")
        if mcp_session_to_release:
            await release_mcp_session_to_pool(mcp_session_to_release)
            del partner_context["runtime_objects"]["mcp_session_group"]
        # --- END: Modified code ---
        # Removed all mcp_session_group cleanup logic
        partner_agent_id_for_log = partner_context.get('meta', {}).get('agent_id', 'Partner')
        current_run_id_for_log = partner_context.get('meta', {}).get('run_id', 'UnknownRun')
        logger.info("partner_task_terminating", extra={"run_id": current_run_id_for_log, "partner_agent_id": partner_agent_id_for_log})
    # ==================== End of final modification ====================
