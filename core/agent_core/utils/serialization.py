from typing import Optional, List, Dict, Any


def get_serializable_run_snapshot(run_context: dict) -> dict:
    """Creates a serializable snapshot of the entire run context."""
    if not run_context: return {}
    
    snapshot = {
        "meta": run_context.get("meta"),
        "config": { # Adding config as per typical snapshot needs
            "run_type": run_context.get("meta", {}).get("run_type"),
            # Potentially add selected non-sensitive config items if needed by UI
        },
        "team_state": run_context.get("team_state"),
        "sub_contexts_state": {}
    }
    
    # Serialize KnowledgeBase
    kb_instance = run_context.get("runtime", {}).get("knowledge_base")
    if kb_instance and hasattr(kb_instance, 'to_dict'):
        snapshot["knowledge_base"] = kb_instance.to_dict()
    else:
        snapshot["knowledge_base"] = None
    
    sub_refs = run_context.get("sub_context_refs", {})
    for key, ref_obj in sub_refs.items():
        if ref_obj and isinstance(ref_obj, dict) and 'state' in ref_obj:
            # --- Start of modification ---
            # No longer create simple_key, use the original key directly
            # The original key is "_partner_context_ref" or "_principal_context_ref"
            snapshot["sub_contexts_state"][key] = ref_obj['state']
            # --- End of modification ---
    
    return snapshot


def get_paginated_run_snapshot(
    run_context: dict,
    mode: str = "summary",
    section: Optional[str] = None,
    context_name: Optional[str] = None,
    work_module_id: Optional[str] = None,
    message_offset: int = 0,
    message_limit: int = 50,
    archive_index: Optional[int] = None
) -> dict:
    """
    Creates a paginated/filtered snapshot of the run context.
    
    Modes:
        - "summary": Returns lightweight overview (no full messages)
        - "full": Returns everything (same as get_serializable_run_snapshot)
        - "section": Returns only the requested section with pagination
    
    Sections (for mode="section"):
        - "meta": Just metadata
        - "team_state": Work modules and dispatch history (with optional work_module_id pagination)
        - "sub_contexts": Agent contexts with message pagination
        - "knowledge_base": Knowledge base entries
    
    For sub_contexts section:
        - context_name: Specific context to retrieve (e.g., "_principal_context_ref")
        - message_offset: Start index for messages (0-based)
        - message_limit: Max messages to return (default 50)
    
    For team_state section (new pagination options):
        - work_module_id: Specific work module to retrieve with full context_archive
        - archive_index: Specific archive within the work module (0-based)
        - message_offset/limit: Pagination within the archive's messages
        If work_module_id is not specified, returns team_state with work modules
        stripped of context_archive (lightweight mode).
    
    Returns:
        dict with requested data and pagination metadata
    """
    if not run_context:
        return {"error": "No run context available"}
    
    if mode == "full":
        return get_serializable_run_snapshot(run_context)
    
    if mode == "summary":
        return _get_summary_snapshot(run_context)
    
    if mode == "section":
        if not section:
            return {"error": "Section name required for mode='section'"}
        return _get_section_snapshot(
            run_context, section, context_name, work_module_id,
            message_offset, message_limit, archive_index
        )
    
    return {"error": f"Unknown mode: {mode}"}


def _get_summary_snapshot(run_context: dict) -> dict:
    """
    Returns a lightweight summary without full message content.
    Includes: meta, team_state (lightweight - no turns, no context_archive), sub_context summaries, knowledge_base keys.
    """
    # Create lightweight team_state (without context_archive and turns to avoid huge responses)
    team_state = run_context.get("team_state", {})
    work_modules = team_state.get("work_modules", {})
    turns = team_state.get("turns", [])
    
    lightweight_modules = {}
    work_module_summaries = {}
    
    for wm_id, wm in work_modules.items():
        if not isinstance(wm, dict):
            continue
        
        # Create lightweight copy without context_archive
        lightweight_wm = {k: v for k, v in wm.items() if k != "context_archive"}
        lightweight_modules[wm_id] = lightweight_wm
        
        # Create summary with archive info
        context_archive = wm.get("context_archive", [])
        archive_summaries = []
        for i, archive in enumerate(context_archive):
            if isinstance(archive, dict):
                messages = archive.get("messages", [])
                archive_summaries.append({
                    "archive_index": i,
                    "message_count": len(messages),
                    "has_deliverables": bool(archive.get("deliverables"))
                })
        
        work_module_summaries[wm_id] = {
            "archive_count": len(context_archive),
            "archives": archive_summaries
        }
    
    lightweight_team_state = {
        **{k: v for k, v in team_state.items() if k not in ("work_modules", "turns")},
        "work_modules": lightweight_modules
    }
    
    snapshot = {
        "mode": "summary",
        "meta": run_context.get("meta"),
        "team_state": lightweight_team_state,
        "work_module_summaries": work_module_summaries,
        "turn_count": len(turns),  # Just the count, not the full turns
        "sub_contexts_summary": {},
        "knowledge_base_summary": {}
    }
    
    # Summarize sub_contexts (message counts, not full messages)
    sub_refs = run_context.get("sub_context_refs", {})
    for key, ref_obj in sub_refs.items():
        if ref_obj and isinstance(ref_obj, dict) and 'state' in ref_obj:
            state = ref_obj['state']
            messages = state.get("messages", [])
            inbox = state.get("inbox", [])
            deliverables = state.get("deliverables", {})
            
            # Get last message preview
            last_message_preview = None
            if messages:
                last_msg = messages[-1]
                content = str(last_msg.get("content", ""))
                last_message_preview = {
                    "role": last_msg.get("role"),
                    "content_preview": content[:200] + ("..." if len(content) > 200 else ""),
                    "has_tool_calls": bool(last_msg.get("tool_calls"))
                }
            
            snapshot["sub_contexts_summary"][key] = {
                "message_count": len(messages),
                "inbox_count": len(inbox),
                "has_deliverables": bool(deliverables),
                "deliverable_keys": list(deliverables.keys()) if deliverables else [],
                "last_message": last_message_preview
            }
    
    # Summarize knowledge_base (just keys and sizes)
    kb_instance = run_context.get("runtime", {}).get("knowledge_base")
    if kb_instance and hasattr(kb_instance, 'to_dict'):
        kb_dict = kb_instance.to_dict()
        for key, value in kb_dict.items():
            if isinstance(value, str):
                snapshot["knowledge_base_summary"][key] = {
                    "type": "string",
                    "size": len(value)
                }
            elif isinstance(value, (list, dict)):
                snapshot["knowledge_base_summary"][key] = {
                    "type": type(value).__name__,
                    "size": len(value)
                }
            else:
                snapshot["knowledge_base_summary"][key] = {
                    "type": type(value).__name__
                }
    
    return snapshot


def _get_section_snapshot(
    run_context: dict,
    section: str,
    context_name: Optional[str],
    work_module_id: Optional[str],
    message_offset: int,
    message_limit: int,
    archive_index: Optional[int]
) -> dict:
    """Returns a specific section with pagination support."""
    
    if section == "meta":
        return {
            "mode": "section",
            "section": "meta",
            "data": run_context.get("meta")
        }
    
    if section == "team_state":
        return _get_team_state_section(
            run_context, work_module_id, archive_index, message_offset, message_limit
        )
    
    if section == "sub_contexts":
        return _get_sub_contexts_section(
            run_context, context_name, message_offset, message_limit
        )
    
    if section == "knowledge_base":
        kb_instance = run_context.get("runtime", {}).get("knowledge_base")
        if kb_instance and hasattr(kb_instance, 'to_dict'):
            return {
                "mode": "section",
                "section": "knowledge_base",
                "data": kb_instance.to_dict()
            }
        return {
            "mode": "section",
            "section": "knowledge_base",
            "data": None
        }
    
    return {"error": f"Unknown section: {section}"}


def _get_team_state_section(
    run_context: dict,
    work_module_id: Optional[str],
    archive_index: Optional[int],
    message_offset: int,
    message_limit: int
) -> dict:
    """
    Returns team_state with optional work module pagination.
    
    If work_module_id is None:
        Returns team_state with work modules stripped of context_archive (lightweight).
        Includes work_module_summaries with archive counts.
    
    If work_module_id is specified:
        Returns that work module with context_archive.
        If archive_index is specified, paginates messages within that archive.
    """
    team_state = run_context.get("team_state", {})
    work_modules = team_state.get("work_modules", {})
    
    if not work_module_id:
        # Return lightweight team_state without context_archive
        lightweight_modules = {}
        work_module_summaries = {}
        
        for wm_id, wm in work_modules.items():
            if not isinstance(wm, dict):
                continue
            
            # Create lightweight copy without context_archive
            lightweight_wm = {k: v for k, v in wm.items() if k != "context_archive"}
            lightweight_modules[wm_id] = lightweight_wm
            
            # Create summary with archive info
            context_archive = wm.get("context_archive", [])
            archive_summaries = []
            for i, archive in enumerate(context_archive):
                if isinstance(archive, dict):
                    messages = archive.get("messages", [])
                    archive_summaries.append({
                        "archive_index": i,
                        "message_count": len(messages),
                        "has_deliverables": bool(archive.get("deliverables"))
                    })
            
            work_module_summaries[wm_id] = {
                "archive_count": len(context_archive),
                "archives": archive_summaries
            }
        
        return {
            "mode": "section",
            "section": "team_state",
            "data": {
                **{k: v for k, v in team_state.items() if k != "work_modules"},
                "work_modules": lightweight_modules
            },
            "work_module_summaries": work_module_summaries,
            "hint": "Use 'work_module_id' to retrieve full context_archive for a specific module"
        }
    
    # Return specific work module with context_archive (optionally paginated)
    if work_module_id not in work_modules:
        return {
            "error": f"Work module '{work_module_id}' not found",
            "available_modules": list(work_modules.keys())
        }
    
    wm = work_modules[work_module_id]
    context_archive = wm.get("context_archive", [])
    
    if archive_index is not None:
        # Return specific archive with message pagination
        if archive_index < 0 or archive_index >= len(context_archive):
            return {
                "error": f"Archive index {archive_index} out of range",
                "archive_count": len(context_archive)
            }
        
        archive = context_archive[archive_index]
        if not isinstance(archive, dict):
            return {"error": f"Invalid archive at index {archive_index}"}
        
        messages = archive.get("messages", [])
        total_messages = len(messages)
        paginated_messages = messages[message_offset:message_offset + message_limit]
        
        return {
            "mode": "section",
            "section": "team_state",
            "work_module_id": work_module_id,
            "archive_index": archive_index,
            "data": {
                "messages": paginated_messages,
                "deliverables": archive.get("deliverables", {}),
                "model": archive.get("model"),
                # Include other archive fields
                **{k: v for k, v in archive.items() 
                   if k not in ("messages", "deliverables", "model")}
            },
            "pagination": {
                "total_messages": total_messages,
                "offset": message_offset,
                "limit": message_limit,
                "returned": len(paginated_messages),
                "has_more": (message_offset + message_limit) < total_messages
            }
        }
    
    # Return full work module with all archives (no message pagination)
    # WARNING: This can still be large for modules dispatched many times
    return {
        "mode": "section",
        "section": "team_state",
        "work_module_id": work_module_id,
        "data": wm,
        "archive_count": len(context_archive),
        "hint": "Use 'archive_index' to paginate messages within a specific archive"
    }


def _get_sub_contexts_section(
    run_context: dict,
    context_name: Optional[str],
    message_offset: int,
    message_limit: int
) -> dict:
    """Returns sub_contexts with message pagination."""
    sub_refs = run_context.get("sub_context_refs", {})
    
    # List available contexts
    available_contexts = list(sub_refs.keys())
    
    if not context_name:
        # Return list of available contexts with summaries
        summaries = {}
        for key, ref_obj in sub_refs.items():
            if ref_obj and isinstance(ref_obj, dict) and 'state' in ref_obj:
                state = ref_obj['state']
                summaries[key] = {
                    "message_count": len(state.get("messages", [])),
                    "inbox_count": len(state.get("inbox", [])),
                    "has_deliverables": bool(state.get("deliverables"))
                }
        return {
            "mode": "section",
            "section": "sub_contexts",
            "available_contexts": available_contexts,
            "context_summaries": summaries,
            "hint": "Specify 'context_name' to retrieve messages for a specific context"
        }
    
    # Get specific context with pagination
    if context_name not in sub_refs:
        return {
            "error": f"Context '{context_name}' not found",
            "available_contexts": available_contexts
        }
    
    ref_obj = sub_refs[context_name]
    if not ref_obj or not isinstance(ref_obj, dict) or 'state' not in ref_obj:
        return {"error": f"Invalid context state for '{context_name}'"}
    
    state = ref_obj['state']
    messages = state.get("messages", [])
    total_messages = len(messages)
    
    # Apply pagination to messages
    paginated_messages = messages[message_offset:message_offset + message_limit]
    
    return {
        "mode": "section",
        "section": "sub_contexts",
        "context_name": context_name,
        "data": {
            "messages": paginated_messages,
            "inbox": state.get("inbox", []),
            "deliverables": state.get("deliverables", {})
        },
        "pagination": {
            "total_messages": total_messages,
            "offset": message_offset,
            "limit": message_limit,
            "returned": len(paginated_messages),
            "has_more": (message_offset + message_limit) < total_messages
        }
    }
