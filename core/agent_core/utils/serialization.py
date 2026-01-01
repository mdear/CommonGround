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
    message_offset: int = 0,
    message_limit: int = 50
) -> dict:
    """
    Creates a paginated/filtered snapshot of the run context.
    
    Modes:
        - "summary": Returns lightweight overview (no full messages)
        - "full": Returns everything (same as get_serializable_run_snapshot)
        - "section": Returns only the requested section with pagination
    
    Sections (for mode="section"):
        - "meta": Just metadata
        - "team_state": Work modules and dispatch history
        - "sub_contexts": Agent contexts with message pagination
        - "knowledge_base": Knowledge base entries
    
    For sub_contexts section:
        - context_name: Specific context to retrieve (e.g., "_principal_context_ref")
        - message_offset: Start index for messages (0-based)
        - message_limit: Max messages to return (default 50)
    
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
            run_context, section, context_name, message_offset, message_limit
        )
    
    return {"error": f"Unknown mode: {mode}"}


def _get_summary_snapshot(run_context: dict) -> dict:
    """
    Returns a lightweight summary without full message content.
    Includes: meta, team_state, sub_context summaries, knowledge_base keys.
    """
    snapshot = {
        "mode": "summary",
        "meta": run_context.get("meta"),
        "team_state": run_context.get("team_state"),
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
    message_offset: int,
    message_limit: int
) -> dict:
    """Returns a specific section with pagination support."""
    
    if section == "meta":
        return {
            "mode": "section",
            "section": "meta",
            "data": run_context.get("meta")
        }
    
    if section == "team_state":
        return {
            "mode": "section",
            "section": "team_state",
            "data": run_context.get("team_state")
        }
    
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
