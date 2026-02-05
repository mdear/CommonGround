import logging
import copy
from typing import Dict, List
import os

from .tool_registry import get_tools_for_profile, format_tools_for_llm_api, get_tool_by_name
from agent_profiles.loader import get_active_llm_config_by_name

logger = logging.getLogger(__name__)


def filter_tools_for_critical_budget(tools: List[Dict], agent_id: str) -> List[Dict]:
    """
    Filters tools to only those allowed at CRITICAL/EXCEEDED context budget thresholds.
    
    At critical budget levels, Partner agents should only have access to read-only tools
    that don't significantly expand context. This ensures headroom is preserved for:
    - Wrapping up current research
    - Preparing handoff plans for future sessions
    - Responding to user queries about status
    
    Tools with `allowed_at_critical=True` in their registry definition are kept.
    Tools without this flag (or with False) are filtered out.
    
    Args:
        tools: List of tool info dictionaries from the registry
        agent_id: Agent ID for logging
        
    Returns:
        Filtered list of tools allowed at critical budget levels
    """
    allowed_tools = []
    filtered_out = []
    
    for tool in tools:
        tool_name = tool.get("name", "unknown")
        if tool.get("allowed_at_critical", False):
            allowed_tools.append(tool)
        else:
            filtered_out.append(tool_name)
    
    if filtered_out:
        logger.info("tools_filtered_at_critical_budget", extra={
            "agent_id": agent_id,
            "filtered_count": len(filtered_out),
            "filtered_tools": filtered_out,
            "allowed_count": len(allowed_tools),
            "allowed_tools": [t.get("name") for t in allowed_tools]
        })
    
    return allowed_tools


def get_formatted_api_tools(agent_node_instance, context: Dict) -> List[Dict]:
    """
    Gets tools based on the profile's tool_access_policy and shared state,
    then formats them for the LLM API.
    
    At CRITICAL or EXCEEDED context budget thresholds, tools are filtered to only
    those marked as `allowed_at_critical=True` (typically read-only tools).
    """
    profile_config = agent_node_instance.loaded_profile
    agent_id = agent_node_instance.agent_id

    applicable_tools = get_tools_for_profile(profile_config, context, agent_id)
    
    # Check budget status and filter tools if at critical levels
    budget_info = context.get("state", {}).get("_context_budget", {})
    budget_status = budget_info.get("status", "HEALTHY")
    
    if budget_status in ("CRITICAL", "EXCEEDED"):
        original_count = len(applicable_tools)
        applicable_tools = filter_tools_for_critical_budget(applicable_tools, agent_id)
        logger.info("tools_restricted_at_critical_budget", extra={
            "agent_id": agent_id,
            "budget_status": budget_status,
            "original_tool_count": original_count,
            "restricted_tool_count": len(applicable_tools)
        })
    
    api_tools_list = format_tools_for_llm_api(applicable_tools)
    
    logger.debug("agent_tools_prepared", extra={"agent_id": agent_id, "tool_count": len(api_tools_list)})
    return api_tools_list
