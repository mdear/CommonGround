# File path: nodes/custom_nodes/dispatcher_node.py

import logging
import asyncio
import copy
import uuid
from datetime import datetime, timezone # Added timezone
from typing import List, Dict, Any, Optional, Tuple
from pocketflow import AsyncParallelBatchNode # Ensure this is the correct base class
from ...framework.tool_registry import tool_registry
# from nodes.base_agent_node import AgentNode # Not directly used here for instantiation
from ...state.management import _create_flow_specific_state_template
from ...framework.profile_utils import get_active_profile_by_name
from ...framework.handover_service import HandoverService
# Content selection for budget-aware inheritance
from ...utils.content_selection import (
    compute_inheritance_budget_chars,
    select_inherited_content_with_hydration,
    format_inherited_content_for_briefing,
    STRATEGY_LLM_SUMMARY,
    STRATEGY_NEWEST_FIRST,
    STRATEGY_EMPTY
)
from ...framework.context_budget_guardian import get_model_context_limit
from ...llm.config_resolver import LLMConfigResolver

logger = logging.getLogger(__name__)

# Constants for dispatch health monitoring
DISPATCH_TIMEOUT_SECONDS = 600  # 10 minutes - if a dispatch is RUNNING longer, it may be stuck


def detect_stuck_dispatches(team_state: Dict, timeout_seconds: int = DISPATCH_TIMEOUT_SECONDS) -> List[Dict]:
    """
    Detects dispatches that are stuck in RUNNING or LAUNCHING state for longer than the timeout.
    
    This helps identify silent failures where Associate subflows crashed without proper cleanup.
    
    Args:
        team_state: The shared team_state dictionary.
        timeout_seconds: How long a dispatch can be RUNNING before being considered stuck.
        
    Returns:
        List of stuck dispatch entries.
    """
    dispatch_history = team_state.get("dispatch_history", [])
    if not dispatch_history:
        return []
    
    stuck = []
    now = datetime.now(timezone.utc)
    
    for entry in dispatch_history:
        status = entry.get("status", "").upper()
        if status in ["RUNNING", "LAUNCHING"]:
            # Check start time
            start_time_str = entry.get("start_timestamp") or entry.get("_dispatch_started_at")
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    elapsed = (now - start_time).total_seconds()
                    if elapsed > timeout_seconds:
                        stuck.append(entry)
                        logger.warning("stuck_dispatch_detected", extra={
                            "dispatch_id": entry.get("dispatch_id"),
                            "module_id": entry.get("module_id"),
                            "status": status,
                            "elapsed_seconds": elapsed,
                            "start_timestamp": start_time_str,
                            "_subcontext_created": entry.get("_subcontext_created", "unknown"),
                            "_associate_flow_started": entry.get("_associate_flow_started", "unknown"),
                            "_associate_flow_completed": entry.get("_associate_flow_completed", "unknown"),
                        })
                except (ValueError, TypeError) as e:
                    logger.debug("stuck_dispatch_time_parse_error", extra={
                        "dispatch_id": entry.get("dispatch_id"),
                        "error": str(e)
                    })
    
    return stuck


def get_dispatch_health_report(team_state: Dict) -> Dict[str, Any]:
    """
    Generates a health report for all dispatches in team_state.
    
    Returns:
        Dict with health metrics and anomalies detected.
    """
    dispatch_history = team_state.get("dispatch_history", [])
    
    report = {
        "total_dispatches": len(dispatch_history),
        "by_status": {},
        "anomalies": [],
        "lifecycle_incomplete": [],
    }
    
    for entry in dispatch_history:
        status = entry.get("status", "UNKNOWN")
        report["by_status"][status] = report["by_status"].get(status, 0) + 1
        
        # Check for lifecycle anomalies (instrumentation fields)
        subcontext_created = entry.get("_subcontext_created", None)
        flow_started = entry.get("_associate_flow_started", None)
        flow_completed = entry.get("_associate_flow_completed", None)
        
        # Detect incomplete lifecycles
        if subcontext_created is not None:  # Has instrumentation
            issues = []
            if subcontext_created and not flow_started:
                issues.append("subcontext_created_but_flow_not_started")
            if flow_started and not flow_completed:
                issues.append("flow_started_but_not_completed")
            if entry.get("_critical_error"):
                issues.append(f"critical_error: {entry.get('_critical_error_type', 'unknown')}")
            if entry.get("_flow_error"):
                issues.append(f"flow_error: {entry.get('_flow_error_type', 'unknown')}")
            
            if issues:
                report["lifecycle_incomplete"].append({
                    "dispatch_id": entry.get("dispatch_id"),
                    "module_id": entry.get("module_id"),
                    "status": status,
                    "issues": issues,
                })
    
    # Detect stuck dispatches
    stuck = detect_stuck_dispatches(team_state)
    if stuck:
        report["anomalies"].append({
            "type": "stuck_dispatches",
            "count": len(stuck),
            "dispatch_ids": [s.get("dispatch_id") for s in stuck]
        })
    
    return report


DESCRIPTION = """
Called by the Principal to validate and assign a Work Module to an Associate Agent for execution.
  - assignments: List of assignments to be made. Each assignment targets **one** Work Module.
    - A module can only be assigned to one Associate Agent at a time.
"""

@tool_registry(
    name="dispatch_submodules",
    # Associate the tool with our newly created protocol
    handover_protocol="principal_to_associate_briefing",
    description=DESCRIPTION,
    parameters={
        "type": "object",
        "properties": {
            # Only control parameters are kept here now
            "assignments": {
                "type": "array",
                "description": "List of work module assignments. Each assignment targets one pending ('pending' or 'pending_review' status) work module.",
                "items": {
                    "type": "object",
                    "properties": {
                        # --- Control Parameters (not defined by protocol) ---
                        "agent_profile_logical_name": {
                            "type": "string",
                            "description": "The logical name of the Associate Agent Profile to use for this module."
                        },
                        "assigned_role_name": {
                            "type": "string",
                            "description": "The role name assigned for this execution, e.g., 'Market_Researcher'."
                        }
                        # --- Context Parameters (will be auto-merged by handover_protocol) ---
                        # "module_id_to_assign", "assignment_specific_instructions", "inherit_messages_from"
                        # have been removed as they are now defined by principal_to_associate_briefing.yaml
                    },
                    # The "required" list will also be auto-merged from the handover_protocol
                    "required": ["agent_profile_logical_name", "assigned_role_name"]
                }
            },
            # --- Context Parameters (moved to protocol) ---
            # "shared_context_for_all_assignments" is now moved to the handover protocol
        },
        "required": ["assignments"]
    },
    default_knowledge_item_type="DISPATCH_SUBMODULES_RESULT"
)
class DispatcherNode(AsyncParallelBatchNode):
    """
    DispatcherNode: Validates and assigns tasks in dispatchable states (not 'deleted', 'deprecated', 'completed') from `team_state.plan` to Associate Agents,
    and updates their status.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug("dispatcher_node_initialized")

    async def _preselect_inherited_content(
        self,
        inherit_messages_from: List[str],
        work_modules: Dict[str, Any],
        run_context: Dict,
        target_profile_logical_name: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Pre-select content from source modules within a computed budget.

        This method implements budget-aware content inheritance to prevent
        new Associates from being "born over-budget".

        Algorithm:
        1. Resolve target agent's context limit from its LLM config
        2. Compute per-source budget: (limit * 0.40) / num_sources
        3. For each source, use two-tier selection:
           - Tier 1: Use deliverables.primary_summary if it fits
           - Tier 2: Fall back to newest-first message selection
        4. Hydrate messages BEFORE selection to get accurate sizing

        Args:
            inherit_messages_from: List of source module IDs
            work_modules: Dict of all work modules
            run_context: Global run context (for KB and config access)
            target_profile_logical_name: Profile name of the spawning agent

        Returns:
            Tuple of (preselected_messages, selection_metadata)
        """
        if not inherit_messages_from:
            return [], {"skipped": True, "reason": "no_sources"}

        # Get Knowledge Base for hydration
        kb = run_context.get("runtime", {}).get("knowledge_base")

        # Resolve target agent's context limit
        agent_profiles_store = run_context.get("config", {}).get("agent_profiles_store", {})
        shared_llm_configs = run_context.get("config", {}).get("shared_llm_configs_ref", {})

        target_context_limit = 200000  # Default conservative limit

        try:
            target_profile = get_active_profile_by_name(agent_profiles_store, target_profile_logical_name)
            if target_profile:
                llm_config_ref = target_profile.get("llm_config_ref")
                if llm_config_ref and shared_llm_configs:
                    resolver = LLMConfigResolver(shared_llm_configs)
                    resolved_config = resolver.resolve(target_profile)
                    model_name = resolved_config.get("model", "")
                    target_context_limit = get_model_context_limit(model_name, resolved_config)
        except Exception as e:
            logger.warning("dispatcher_context_limit_resolution_failed", extra={
                "profile": target_profile_logical_name,
                "error": str(e),
                "fallback": target_context_limit
            })

        # Compute per-source budget
        num_sources = len(inherit_messages_from)
        per_source_budget = compute_inheritance_budget_chars(target_context_limit, num_sources)

        logger.info("dispatcher_preselect_started", extra={
            "inherit_from": inherit_messages_from,
            "target_context_limit": target_context_limit,
            "per_source_budget_chars": per_source_budget
        })

        # Process each source module
        all_preselected = []
        selection_metadata = {
            "target_context_limit": target_context_limit,
            "per_source_budget_chars": per_source_budget,
            "sources": {}
        }

        for source_module_id in inherit_messages_from:
            source_module = work_modules.get(source_module_id)
            if not source_module:
                logger.warning("dispatcher_preselect_source_not_found", extra={
                    "source_module_id": source_module_id
                })
                selection_metadata["sources"][source_module_id] = {
                    "error": "module_not_found"
                }
                continue

            # Get the most recent context_archive entry
            context_archive = source_module.get("context_archive", [])
            if not context_archive:
                logger.debug("dispatcher_preselect_no_archive", extra={
                    "source_module_id": source_module_id
                })
                selection_metadata["sources"][source_module_id] = {
                    "error": "no_context_archive"
                }
                continue

            latest_archive = context_archive[-1]

            # Perform selection with hydration
            try:
                selected_content, content_metadata = await select_inherited_content_with_hydration(
                    context_archive_entry=latest_archive,
                    budget_chars=per_source_budget,
                    knowledge_base=kb,
                    source_id=source_module_id
                )

                # Format for briefing injection
                formatted_messages = format_inherited_content_for_briefing(
                    content=selected_content,
                    metadata=content_metadata,
                    source_id=source_module_id
                )

                all_preselected.extend(formatted_messages)
                selection_metadata["sources"][source_module_id] = content_metadata

                logger.info("dispatcher_preselect_source_complete", extra={
                    "source_module_id": source_module_id,
                    "strategy": content_metadata.get("strategy"),
                    "chars_used": content_metadata.get("chars_used", 0),
                    "items_selected": content_metadata.get("items_selected", 0)
                })

            except Exception as e:
                logger.error("dispatcher_preselect_source_failed", extra={
                    "source_module_id": source_module_id,
                    "error": str(e)
                }, exc_info=True)
                selection_metadata["sources"][source_module_id] = {
                    "error": str(e)
                }

        total_chars = sum(
            meta.get("chars_used", 0)
            for meta in selection_metadata["sources"].values()
            if isinstance(meta, dict) and "chars_used" in meta
        )
        selection_metadata["total_chars_selected"] = total_chars
        selection_metadata["total_messages_selected"] = len(all_preselected)

        logger.info("dispatcher_preselect_complete", extra={
            "total_chars": total_chars,
            "total_messages": len(all_preselected),
            "sources_processed": len(inherit_messages_from)
        })

        return all_preselected, selection_metadata

    async def prep_async(self, shared: Dict) -> List[Dict]:
        logger.debug("dispatcher_prep_async_started")
        # shared is principal's SubContext
        principal_state = shared["state"]
        principal_team_state = shared['refs']['team']
        work_modules = principal_team_state.get("work_modules", {})
        agent_profiles_store = shared['refs']['run']['config'].get("agent_profiles_store")
        current_tool_call_id = principal_state.get("current_tool_call_id", f"dtcid_unknown_{uuid.uuid4().hex[:4]}")

        if not agent_profiles_store:
            logger.error("dispatcher_prep_agent_profiles_missing")
            principal_state["_temp_dispatcher_prep_failures"] = [{"error": "Agent profiles store not available."}]
            return []
        if not isinstance(work_modules, dict):
            logger.error("dispatcher_prep_work_modules_malformed")
            principal_state["_temp_dispatcher_prep_failures"] = [{"error": "Work modules are malformed or missing."}]
            return []

        dispatch_action = principal_state.get("current_action", {})
        assignments_input = dispatch_action.get("assignments", [])
        shared_context_for_all = dispatch_action.get("shared_context_for_all_assignments", "")

        tasks_for_parallel_exec = []
        failed_assignments_at_prep = []
        assigned_module_ids_in_this_call = set()

        for assign_idx, assignment_item in enumerate(assignments_input):
            # Defensive: LLM may return malformed data (e.g., strings instead of dicts)
            if not isinstance(assignment_item, dict):
                err_msg = f"Assignment at index {assign_idx} is not a dict (got {type(assignment_item).__name__}). Raw value: {str(assignment_item)[:100]}"
                logger.warning("dispatcher_prep_malformed_assignment", extra={"index": assign_idx, "type": type(assignment_item).__name__, "error_message": err_msg})
                failed_assignments_at_prep.append({"input": str(assignment_item)[:100], "reason": err_msg})
                continue

            module_id = assignment_item.get("module_id_to_assign")
            agent_profile_logical_name = assignment_item.get("agent_profile_logical_name")
            assigned_role_name = assignment_item.get("assigned_role_name")

            # [Requirement 2.1] Prevent duplicate dispatch in a single call
            if module_id in assigned_module_ids_in_this_call:
                err_msg = f"Duplicate assignment for module_id '{module_id}' in a single call."
                logger.warning("dispatcher_prep_duplicate_assignment", extra={"module_id": module_id, "error_message": err_msg})
                failed_assignments_at_prep.append({"input": assignment_item, "reason": err_msg})
                continue
            if module_id:
                assigned_module_ids_in_this_call.add(module_id)

            module_to_assign = work_modules.get(module_id)
            if not module_to_assign:
                failed_assignments_at_prep.append({"input": assignment_item, "reason": f"Work Module ID '{module_id}' not found."})
                continue

            # [Requirement 2.2] Allow dispatching from 'pending' or 'pending_review'
            current_module_status = module_to_assign.get("status", "unknown").lower()
            if current_module_status not in ["pending", "pending_review"]:
                err_msg = f"Work Module '{module_id}' has status '{current_module_status}', but must be 'pending' or 'pending_review' to be dispatched."
                failed_assignments_at_prep.append({"input": assignment_item, "reason": err_msg})
                continue

            actual_profile_details = get_active_profile_by_name(agent_profiles_store, agent_profile_logical_name)
            if not actual_profile_details:
                failed_assignments_at_prep.append({"input": assignment_item, "reason": f"Profile '{agent_profile_logical_name}' not found or inactive."})
                continue

            assignment_package = {
                "original_assignment_input": assignment_item,
                "resolved_profile_instance_id": actual_profile_details.get("profile_id"),
                "resolved_profile_logical_name": agent_profile_logical_name,
                "assigned_role_name": assigned_role_name,
                "module_to_execute": copy.deepcopy(module_to_assign),
                "shared_context_for_all_assignments": shared_context_for_all,
                "executing_associate_id": f"Assoc_{agent_profile_logical_name.replace('Associate_', '')[:10]}_{module_id.replace('WM_', '')}",
                "dispatch_tool_call_id_ref": current_tool_call_id,
                "shared_for_exec_context": shared
            }
            tasks_for_parallel_exec.append(assignment_package)

        principal_state["_temp_dispatcher_prep_failures"] = failed_assignments_at_prep
        logger.info("dispatcher_prep_async_completed", extra={
            "valid_assignments": len(tasks_for_parallel_exec),
            "failed_prep_checks": len(failed_assignments_at_prep)
        })
        return tasks_for_parallel_exec

    async def exec_async(self, assignment_package: Dict) -> Dict:
        parent_context = assignment_package.pop("shared_for_exec_context") # This is Principal's SubContext
        run_context_global = parent_context['refs']['run'] # This is the global RunContext
        team_state_global = parent_context['refs']['team'] # This is the global TeamState
        events = run_context_global['runtime'].get("event_manager")
        run_id = run_context_global['meta'].get("run_id")

        if "_ongoing_associate_tasks" not in run_context_global['sub_context_refs']:
            run_context_global['sub_context_refs']["_ongoing_associate_tasks"] = {}

        module_to_execute = assignment_package["module_to_execute"]
        module_id = module_to_execute["module_id"]
        executing_associate_id = assignment_package["executing_associate_id"]
        profile_logical_name = assignment_package["resolved_profile_logical_name"]

        logger.info("dispatcher_exec_assignment_started", extra={
            "module_id": module_id,
            "profile_logical_name": profile_logical_name,
            "executing_associate_id": executing_associate_id
        })

        module_to_update = copy.deepcopy(team_state_global.get("work_modules", {}).get(module_id))
        if not module_to_update:
            return {"error": f"Module {module_id} not found at execution time."}

        start_time_iso = datetime.now(timezone.utc).isoformat()
        module_to_update["status"] = "ongoing"
        module_to_update["updated_at"] = start_time_iso
        module_to_update.setdefault("assignee_history", []).append({
            "dispatch_id": executing_associate_id, "agent_id": executing_associate_id,
            "started_at": start_time_iso, "ended_at": None, "outcome": "running"
        })
        team_state_global["work_modules"][module_id] = module_to_update
        if events and hasattr(events, "emit_work_module_updated"):
            await events.emit_work_module_updated(run_id, module_to_update)

        try:
            from ...events.event_triggers import trigger_view_model_update
            await trigger_view_model_update(parent_context, "kanban_view")
            logger.info("dispatcher_kanban_view_update_triggered", extra={"module_id": module_id, "new_status": "ongoing"})
        except Exception as e_trigger:
            logger.error("dispatcher_kanban_view_update_failed", extra={"module_id": module_id, "error": str(e_trigger)}, exc_info=True)

        history_entry = {
            "dispatch_id": executing_associate_id, "dispatch_tool_call_id_ref": assignment_package["dispatch_tool_call_id_ref"],
            "module_id": module_id, "profile_logical_name": profile_logical_name, "start_timestamp": None,
            "end_timestamp": None, "status": "LAUNCHING", "final_summary": None, "error_details": None,
            # Instrumentation: Track dispatch lifecycle for debugging silent failures
            "_dispatch_started_at": datetime.now(timezone.utc).isoformat(),
            "_subcontext_created": False,
            "_associate_flow_started": False,
            "_associate_flow_completed": False,
        }
        team_state_global.setdefault("dispatch_history", []).append(history_entry)
        logger.info("dispatcher_history_entry_added", extra={"executing_associate_id": executing_associate_id, "status": "LAUNCHING"})

        # --- Budget-Aware Content Pre-Selection ---
        # If inherit_messages_from is specified, pre-select content within budget
        # BEFORE calling HandoverService to ensure budget compliance
        original_assignment = assignment_package.get("original_assignment_input", {})
        inherit_messages_from = original_assignment.get("inherit_messages_from", [])
        preselected_messages = []
        preselection_metadata = {}

        if inherit_messages_from:
            try:
                preselected_messages, preselection_metadata = await self._preselect_inherited_content(
                    inherit_messages_from=inherit_messages_from,
                    work_modules=team_state_global.get("work_modules", {}),
                    run_context=run_context_global,
                    target_profile_logical_name=profile_logical_name
                )
                logger.info("dispatcher_content_preselection_complete", extra={
                    "module_id": module_id,
                    "total_chars": preselection_metadata.get("total_chars_selected", 0),
                    "total_messages": len(preselected_messages)
                })
            except Exception as e:
                logger.error("dispatcher_content_preselection_failed", extra={
                    "module_id": module_id,
                    "error": str(e)
                }, exc_info=True)
                # Continue with empty preselected content - let HandoverService use fallback
        # --- End Budget-Aware Content Pre-Selection ---

        try:
            # Build a temporary source_context to simulate the state when the Principal calls the tool
            # Inject pre-selected content into parameters for HandoverService to use
            enhanced_parameters = original_assignment.copy()
            if preselected_messages:
                enhanced_parameters["_preselected_inherited_messages"] = preselected_messages
                enhanced_parameters["_preselection_metadata"] = preselection_metadata

            temp_source_context_for_handover = {
                "state": {
                    "current_action": {
                        # Place the current assignment's parameters into current_action.parameters
                        "parameters": enhanced_parameters
                    }
                },
                "refs": parent_context["refs"],
                "meta": parent_context["meta"]
            }

            # Call HandoverService
            inbox_item_data = await HandoverService.execute(
                "principal_to_associate_briefing",
                temp_source_context_for_handover
            )

        except Exception as e:
            logger.error("dispatcher_handover_service_failed", extra={"executing_associate_id": executing_associate_id, "error": str(e)}, exc_info=True)
            return {"error": f"Failed to prepare context handover: {e}"}

        associate_sub_context_state = _create_flow_specific_state_template()
        associate_sub_context_state.setdefault("inbox", []).append({
            "item_id": f"inbox_{uuid.uuid4().hex[:8]}",
            "source": inbox_item_data["source"],
            "payload": inbox_item_data["payload"],
            "consumption_policy": "consume_on_read",
            "metadata": {"created_at": datetime.now(timezone.utc).isoformat()}
        })

        principal_last_turn_id = parent_context['state'].get("last_turn_id")
        associate_sub_context_state['last_turn_id'] = principal_last_turn_id
        logger.debug("dispatcher_last_turn_id_passed", extra={"last_turn_id": principal_last_turn_id, "executing_associate_id": executing_associate_id})

        principal_agent_id = parent_context['meta'].get("agent_id")
        assigned_role_name = assignment_package.get("assigned_role_name")

        associate_sub_context: Dict[str, Any] = {
            "meta": {
                "run_id": run_id,
                "agent_id": executing_associate_id,
                "parent_agent_id": principal_agent_id,
                "assigned_role_name": assigned_role_name,
                "module_id": module_id,
                "module_description": module_to_execute.get("description"),
                "profile_logical_name": profile_logical_name,
                "profile_instance_id": assignment_package["resolved_profile_instance_id"],
                "dispatch_tool_call_id_ref": assignment_package["dispatch_tool_call_id_ref"],
            },
            "state": associate_sub_context_state,
            "runtime_objects": {},
            "refs": { "run": run_context_global, "team": team_state_global }
        }

        logger.info("dispatcher_associate_starting", extra={"executing_associate_id": executing_associate_id})

        # Update history entry to track subcontext creation
        if history_entry_to_update := next((h for h in team_state_global.get("dispatch_history", []) if h.get("dispatch_id") == executing_associate_id), None):
            history_entry_to_update["_subcontext_created"] = True
            history_entry_to_update["_subcontext_created_at"] = datetime.now(timezone.utc).isoformat()

        completed_associate_context = None
        associate_exec_status = "error"
        last_turn_id = None
        new_messages_from_associate = []
        try:
            if run_context_global:
                run_context_global['sub_context_refs']["_ongoing_associate_tasks"][executing_associate_id] = associate_sub_context
                logger.info("dispatcher_associate_task_registered", extra={"executing_associate_id": executing_associate_id})
            
            # Update history entry to track flow start
            if history_entry_to_update := next((h for h in team_state_global.get("dispatch_history", []) if h.get("dispatch_id") == executing_associate_id), None):
                history_entry_to_update["_associate_flow_started"] = True
                history_entry_to_update["_associate_flow_started_at"] = datetime.now(timezone.utc).isoformat()
            
            from ...flow import run_associate_async
            completed_associate_context = await run_associate_async(associate_sub_context)

            final_associate_state = completed_associate_context.get("state", {})
            last_turn_id = final_associate_state.get("last_turn_id")

            if not final_associate_state.get("error_message"):
                associate_exec_status = "success"
        except Exception as e:
            logger.error("dispatcher_associate_critical_error", extra={
                "executing_associate_id": executing_associate_id, 
                "error": str(e),
                "error_type": type(e).__name__,
                "module_id": module_id,
                "profile": profile_logical_name,
            }, exc_info=True)
            
            # Update history entry to track the failure point
            if history_entry_to_update := next((h for h in team_state_global.get("dispatch_history", []) if h.get("dispatch_id") == executing_associate_id), None):
                history_entry_to_update["_critical_error"] = True
                history_entry_to_update["_critical_error_at"] = datetime.now(timezone.utc).isoformat()
                history_entry_to_update["_critical_error_type"] = type(e).__name__
                history_entry_to_update["_critical_error_message"] = str(e)[:500]  # Truncate for storage
            
            if completed_associate_context is None: completed_associate_context = {}
            final_associate_state = completed_associate_context.setdefault("state", {})
            final_associate_state["error_message"] = f"Dispatcher critical error: {str(e)}"
            final_associate_state.setdefault("deliverables", {})["error"] = f"Dispatcher critical error: {str(e)}"

        finally:
            end_time_iso = datetime.now(timezone.utc).isoformat()
            final_outcome = "completed_success" if associate_exec_status == "success" else "completed_error"
            
            # Update history entry to track flow completion
            if history_entry_to_update := next((h for h in team_state_global.get("dispatch_history", []) if h.get("dispatch_id") == executing_associate_id), None):
                history_entry_to_update["_associate_flow_completed"] = True
                history_entry_to_update["_associate_flow_completed_at"] = end_time_iso
                history_entry_to_update["_final_outcome"] = final_outcome

            final_associate_state = completed_associate_context.get("state", {}) if completed_associate_context else {}
            deliverables_from_associate = final_associate_state.get("deliverables", {})
            error_details_from_associate = final_associate_state.get("error_message")

            all_messages = final_associate_state.get("messages", [])

            # Filter out messages that are marked as not for handover (e.g., initial briefings).
            # The msg.get("_internal", {}) ensures safe access even if the _internal key doesn't exist.
            new_messages_from_associate = [
                msg for msg in all_messages if not msg.get("_internal", {}).get("_no_handover")
            ]

            logger.info("dispatcher_messages_extracted", extra={"executing_associate_id": executing_associate_id, "new_message_count": len(new_messages_from_associate)})

            history_entry_to_update = next((h for h in team_state_global.get("dispatch_history", []) if h.get("dispatch_id") == executing_associate_id), None)
            if history_entry_to_update:
                status_map = {"success": "COMPLETED_SUCCESS", "error": "COMPLETED_ERROR"}
                history_entry_to_update["status"] = status_map.get(associate_exec_status, "COMPLETED_ERROR")
                history_entry_to_update["end_timestamp"] = end_time_iso
                history_entry_to_update["error_details"] = error_details_from_associate
                if deliverables_from_associate:
                    summary = ", ".join(deliverables_from_associate.keys())
                    history_entry_to_update["final_summary"] = f"Deliverables: {summary}"
                logger.info("dispatcher_history_updated", extra={"executing_associate_id": executing_associate_id, "new_status": history_entry_to_update['status']})

            history_list = module_to_update.get("assignee_history", [])
            entry_to_update = next((h for h in reversed(history_list) if h.get("dispatch_id") == executing_associate_id and h.get("outcome") == "running"), None)
            if entry_to_update:
                entry_to_update["ended_at"] = end_time_iso
                entry_to_update["outcome"] = final_outcome

            module_to_update.setdefault("context_archive", []).append({
                "dispatch_id": executing_associate_id, "archived_at": end_time_iso,
                "messages": final_associate_state.get("messages", []), "deliverables": deliverables_from_associate
            })

            # Propagate deliverables to canonical work_modules[].deliverables field for easy access
            # This ensures Principal/Partner can access deliverables without parsing context_archive
            if deliverables_from_associate:
                module_to_update["deliverables"] = deliverables_from_associate
                logger.info("deliverables_propagated_to_module", extra={
                    "module_id": module_id,
                    "dispatch_id": executing_associate_id,
                    "deliverable_keys": list(deliverables_from_associate.keys()) if isinstance(deliverables_from_associate, dict) else "non-dict"
                })

            module_to_update["status"] = "pending_review"
            module_to_update["review_info"] = {
                "trigger": "associate_completed" if associate_exec_status == "success" else "associate_failed",
                "message": "Associate completed its work." if associate_exec_status == "success" else "Associate failed with an exception.",
                "error_details": error_details_from_associate
            }
            module_to_update["updated_at"] = end_time_iso
            team_state_global["work_modules"][module_id] = module_to_update
            if events and hasattr(events, "emit_work_module_updated"):
                await events.emit_work_module_updated(run_id, module_to_update)

            if run_context_global and executing_associate_id in run_context_global['sub_context_refs'].get("_ongoing_associate_tasks", {}):
                del run_context_global['sub_context_refs']["_ongoing_associate_tasks"][executing_associate_id]
                logger.info("dispatcher_associate_task_deregistered", extra={"executing_associate_id": executing_associate_id})

        return {
            "executing_associate_id": executing_associate_id,
            "module_id": module_id,
            "agent_profile_logical_name_used": profile_logical_name,
            "status_of_associate_execution": associate_exec_status,
            "deliverables_from_associate": deliverables_from_associate,
            "error_detail_from_associate": error_details_from_associate,
            "last_turn_id": last_turn_id,
            "new_messages_from_associate": new_messages_from_associate
        }

    async def post_async(self, shared: Dict, prep_res: List[Dict], exec_res_list: List[Dict]):
        principal_state = shared["state"]
        dispatch_tool_call_id = principal_state.get("current_tool_call_id", f"dtcid_unknown_{uuid.uuid4().hex[:4]}")

        logger.debug("dispatcher_post_async_aggregating", extra={"execution_count": len(exec_res_list), "dispatch_tool_call_id": dispatch_tool_call_id})

        failed_assignments_from_prep = principal_state.pop("_temp_dispatcher_prep_failures", [])

        num_launched_modules = len(exec_res_list)
        num_successful_executions = sum(1 for res in exec_res_list if res.get("status_of_associate_execution") == "success")
        num_failed_executions = num_launched_modules - num_successful_executions
        num_prep_failures = len(failed_assignments_from_prep)
        original_assignments_requested_count = num_launched_modules + num_prep_failures

        overall_dispatch_op_status = "TOTAL_FAILURE"
        if original_assignments_requested_count == 0:
            overall_dispatch_op_status = "NO_ASSIGNMENTS_REQUESTED"
        elif num_launched_modules > 0:
            if num_successful_executions == num_launched_modules:
                overall_dispatch_op_status = "SUCCESS" if num_prep_failures == 0 else "PARTIAL_SUCCESS_SOME_PREP_FAILED"
            elif num_successful_executions > 0:
                overall_dispatch_op_status = "PARTIAL_SUCCESS_ASSOCIATES_SOME_FAILED" if num_prep_failures == 0 else "PARTIAL_SUCCESS_MIXED_RESULTS"
            else:
                overall_dispatch_op_status = "TOTAL_FAILURE_ASSOCIATES_ALL_FAILED" if num_prep_failures == 0 else "TOTAL_FAILURE_PREP_AND_ASSOC_FAILED"
        elif num_prep_failures > 0 and num_launched_modules == 0 :
             overall_dispatch_op_status = "TOTAL_FAILURE_ALL_PREP_FAILED"

        dispatch_op_message = (
            f"Dispatch operation concluded for {original_assignments_requested_count} requested assignment(s). "
            f"{num_launched_modules} module(s) were dispatched. "
            f"Of those, {num_successful_executions} completed successfully and are now 'pending_review'. "
            f"{num_failed_executions} failed and are also 'pending_review' for analysis. "
            f"{num_prep_failures} assignment(s) failed pre-check and were not dispatched."
        )

        llm_output_assignment_results = []
        for exec_result in exec_res_list:
            if "error" in exec_result: continue
            llm_output_assignment_results.append({
                "module_id": exec_result.get("module_id"),
                "associate_id": exec_result.get("executing_associate_id"),
                "execution_status": exec_result.get("status_of_associate_execution"),
                "deliverables": exec_result.get("deliverables_from_associate", {}),
                "error_details": exec_result.get("error_detail_from_associate"),
                "new_messages_from_associate": exec_result.get("new_messages_from_associate", [])
            })

        final_tool_output_content = {
            "status": overall_dispatch_op_status,
            "message": dispatch_op_message,
            "assignment_execution_results": llm_output_assignment_results,
            "failed_preparation_details": failed_assignments_from_prep
        }

        # --- Create Aggregation Turn (Conditionally) ---
        if exec_res_list:
            team_state_from_refs_post = shared['refs']['team']
            run_id_from_meta_post = shared['meta']['run_id']
            turn_manager = shared['refs']['run']['runtime'].get('turn_manager')

            # Find the Turn that initiated this dispatch
            dispatch_turn = turn_manager._get_turn_by_id(team_state_from_refs_post, principal_state.get("current_turn_id")) if turn_manager else None

            if dispatch_turn and turn_manager:
                last_turn_ids_of_subflows = [res.get("last_turn_id") for res in exec_res_list if res.get("last_turn_id")]

                # Call TurnManager to create the aggregation turn
                aggregation_turn_id = turn_manager.create_aggregation_turn(
                    team_state=team_state_from_refs_post,
                    run_id=run_id_from_meta_post,
                    dispatch_turn=dispatch_turn,
                    last_turn_ids_of_subflows=last_turn_ids_of_subflows,
                    dispatch_tool_call_id=dispatch_tool_call_id,
                    aggregation_summary=f"{num_successful_executions}/{num_launched_modules} successful."
                )

                # Pass the "baton" to the new aggregation turn
                principal_state['last_turn_id'] = aggregation_turn_id
                logger.debug("dispatcher_relay_baton_passed", extra={"aggregation_turn_id": aggregation_turn_id})
            else:
                logger.error("dispatcher_aggregation_turn_creation_failed", extra={"dispatch_tool_call_id": dispatch_tool_call_id, "reason": "Could not find dispatch_turn or turn_manager"})
        else:
            # If exec_res_list is empty, it means all tasks failed in the preparation phase, so no aggregation turn is created.
            # last_turn_id remains unchanged; the next Principal turn will connect directly to the current dispatch_turn.
            logger.info("dispatcher_aggregation_turn_skipped", extra={"dispatch_tool_call_id": dispatch_tool_call_id, "reason": "no_subtasks_executed"})
        # --- End Aggregation Turn ---

        tool_result_payload = {
            "tool_name": self._tool_info["name"],
            "tool_call_id": principal_state.get('current_tool_call_id'),
            "is_error": overall_dispatch_op_status.startswith("TOTAL_FAILURE"),
            "content": final_tool_output_content
        }

        principal_state.setdefault('inbox', []).append({
            "item_id": f"inbox_{uuid.uuid4().hex[:8]}",
            "source": "TOOL_RESULT",
            "payload": tool_result_payload,
            "consumption_policy": "consume_on_read",
            "metadata": {"created_at": datetime.now(timezone.utc).isoformat()}
        })

        logger.info("dispatcher_post_async_completed", extra={"overall_status": overall_dispatch_op_status})

        principal_state["current_action"] = None

        try:
            from ...events.event_triggers import trigger_view_model_update

            await trigger_view_model_update(shared, "flow_view")
            await trigger_view_model_update(shared, "timeline_view")
            events_for_post_sync = shared['refs']['run']['runtime'].get("event_manager")
            if events_for_post_sync:
                await events_for_post_sync.emit_turns_sync(shared)
            await trigger_view_model_update(shared, "kanban_view")
        except Exception as e:
            logger.error("dispatcher_view_model_update_failed", extra={"error": str(e)}, exc_info=True)

        return "default"

    async def run_batch_async(self, shared: Dict, prep_res_list: List[Dict]) -> List[Dict]:
        tasks = []
        for prep_item in prep_res_list:
            tasks.append(self.exec_async(prep_item))

        exec_res_list = await asyncio.gather(*tasks, return_exceptions=True)

        processed_exec_res_list = []
        for i, res_or_exc in enumerate(exec_res_list):
            original_prep_item = prep_res_list[i]
            module_id_for_error = original_prep_item.get("module_to_execute", {}).get("module_id", "unknown_module")

            if isinstance(res_or_exc, Exception):
                logger.error("dispatcher_batch_exec_exception", extra={"module_id": module_id_for_error, "error": str(res_or_exc)}, exc_info=res_or_exc)
                processed_exec_res_list.append({
                    "executing_associate_id": f"ErrorAssoc_{module_id_for_error}",
                    "module_id": module_id_for_error,
                    "agent_profile_logical_name_used": original_prep_item.get("resolved_profile_logical_name", "unknown_profile"),
                    "status_of_associate_execution": "error",
                    "deliverables_from_associate": {"error": f"Dispatcher critical error during parallel execution: {str(res_or_exc)}"},
                    "error_detail_from_associate": str(res_or_exc)
                })
            elif "error" in res_or_exc:
                logger.error("dispatcher_batch_exec_error", extra={"module_id": module_id_for_error, "error": res_or_exc['error']})
                processed_exec_res_list.append({
                    "executing_associate_id": f"ErrorAssoc_{module_id_for_error}",
                    "module_id": module_id_for_error,
                    "agent_profile_logical_name_used": original_prep_item.get("resolved_profile_logical_name", "unknown_profile"),
                    "status_of_associate_execution": "error",
                    "deliverables_from_associate": {"error": res_or_exc['error']},
                    "error_detail_from_associate": res_or_exc['error']
                })
            else:
                processed_exec_res_list.append(res_or_exc)
        return processed_exec_res_list


def detect_dispatch_anomalies(shared_state, stale_threshold_minutes: int = 60) -> List[Dict[str, Any]]:
    """
    Detects anomalies in dispatch history that may indicate silent failures.
    
    This function helps identify dispatches that:
    1. Are stuck in RUNNING state for too long (stale dispatches)
    2. Have RUNNING status but the corresponding work module has no sub_context
    
    Args:
        shared_state: The shared state object containing team_state
        stale_threshold_minutes: Number of minutes after which a RUNNING dispatch is considered stale
        
    Returns:
        List of anomaly dicts with details about each detected anomaly
    """
    anomalies = []
    
    # Handle both dict-style and object-style access
    if hasattr(shared_state, 'team_state'):
        team_state = shared_state.team_state
    elif isinstance(shared_state, dict):
        team_state = shared_state.get('team_state', {})
    else:
        team_state = {}
        
    if not team_state:
        return anomalies
    
    dispatch_history = team_state.get("dispatch_history", [])
    work_modules = team_state.get("work_modules", {})
    now = datetime.now(timezone.utc)
    stale_threshold_seconds = stale_threshold_minutes * 60
    
    for dispatch in dispatch_history:
        dispatch_id = dispatch.get("dispatch_id", "unknown")
        module_id = dispatch.get("module_id", "unknown")
        status = dispatch.get("status", "").upper()
        start_timestamp_str = dispatch.get("start_timestamp")
        end_timestamp = dispatch.get("end_timestamp")
        
        # Check for stale RUNNING dispatches
        if status == "RUNNING" and not end_timestamp:
            if start_timestamp_str:
                try:
                    start_time = datetime.fromisoformat(start_timestamp_str.replace("Z", "+00:00"))
                    elapsed_seconds = (now - start_time).total_seconds()
                    
                    if elapsed_seconds > stale_threshold_seconds:
                        # Check if work module has a sub_context
                        work_module = work_modules.get(module_id, {})
                        has_sub_context = work_module.get("sub_context_id") is not None
                        
                        anomaly = {
                            "dispatch_id": dispatch_id,
                            "module_id": module_id,
                            "anomaly_type": "stale_running",
                            "status": status,
                            "elapsed_minutes": round(elapsed_seconds / 60, 1),
                            "start_timestamp": start_timestamp_str,
                            "has_sub_context": has_sub_context,
                            "details": f"Dispatch has been RUNNING for {round(elapsed_seconds / 60, 1)} minutes without completion. "
                                      f"Work module {'has' if has_sub_context else 'has no'} sub_context."
                        }
                        
                        if not has_sub_context:
                            anomaly["details"] += " No sub_context was created - dispatch may have failed silently."
                        
                        anomalies.append(anomaly)
                        logger.warning("dispatch_anomaly_detected", extra={
                            "dispatch_id": dispatch_id,
                            "module_id": module_id,
                            "anomaly_type": "stale_running",
                            "elapsed_minutes": round(elapsed_seconds / 60, 1),
                            "has_sub_context": has_sub_context
                        })
                except (ValueError, TypeError) as e:
                    logger.debug("dispatch_anomaly_time_parse_error", extra={
                        "dispatch_id": dispatch_id,
                        "error": str(e)
                    })
    
    return anomalies
