import logging
import json
from typing import Dict, List, Any
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

async def generate_view_model(run_context: Dict, view_name: str) -> Dict:
    """
    Top-level function to generate a specific view model based on the run_context.
    """
    if view_name == "flow_view":
        return await _generate_flow_view_model(run_context)
    elif view_name == "kanban_view":
        return _generate_kanban_view_model(run_context)
    elif view_name == "timeline_view":
        return _generate_timeline_view_model(run_context)
    else:
        raise ValueError(f"Unknown view_name: {view_name}")



def _format_tool_result_as_markdown(data: Any, indent_level: int = 0) -> str:
    """
    Recursively formats a Python object (dictionary, list, value) into a Markdown list.
    """
    indent = "  " * indent_level
    lines = []

    if isinstance(data, dict):
        for key, value in data.items():
            # For dictionary items, create a new list item
            formatted_value = _format_tool_result_as_markdown(value, indent_level + 1)
            # If the sub-item is multi-line (i.e., a list or dict), display it on a new line
            if '\n' in formatted_value.strip():
                lines.append(f"{indent}* **{key}:**\n{formatted_value}")
            else:
                lines.append(f"{indent}* **{key}:** {formatted_value.strip()}")
    elif isinstance(data, list):
        for item in data:
            # For list items, also create a new list item
            formatted_item = _format_tool_result_as_markdown(item, indent_level + 1)
            lines.append(f"{indent}* {formatted_item.strip()}")
    else:
        # For primitive types, return the value directly
        return f"{indent}{str(data)}"

    return "\n".join(lines)


async def _generate_flow_view_model(run_context: Dict) -> Dict:
    """
    Generates the Flow view model based on the team_state.turns list.
    This function processes the turns, hydrates tool results, and formats them for a graph layout
    """
    team_state = run_context.get("team_state", {})
    turns = team_state.get("turns", [])
    kb = run_context.get("runtime", {}).get("knowledge_base")

    # Sort turns by start_time before processing to ensure stable layout for dagre.
    # Use a default empty string for missing 'start_time' to prevent sorting errors with None.
    sorted_turns = sorted(turns, key=lambda t: t.get('start_time', ''))

    # [Refactor] Call the Knowledge Base (KB) method here once to hydrate all Turns.
    hydrated_sorted_turns = []
    if kb:
        hydrated_sorted_turns = await kb.hydrate_turn_list_tool_results(sorted_turns)
    else:
        hydrated_sorted_turns = sorted_turns # Fallback if no KB

    nodes_by_id: Dict[str, Dict] = {}
    edges: List[Dict] = []
    epoch_separators: List[Dict] = []  # Track epoch separator nodes

    # ==================== EPOCH-BASED DEPTH CALCULATION ====================
    #
    # Strategy: Time flows top-to-bottom. Each disconnected subgraph (epoch) is
    # rendered sequentially, sorted by the earliest timestamp in that epoch.
    # Within each epoch, hierarchical BFS depth is preserved.
    # Epochs are visually separated by epoch divider nodes.
    #
    # =======================================================================

    # 0. Pre-filter turns to only include those that will be rendered
    # (exclude Partner and user_turn which are filtered out later in node creation)
    renderable_turns = [
        turn for turn in hydrated_sorted_turns
        if turn.get("agent_info", {}).get("agent_id") != "Partner"
        and turn.get("turn_type") != "user_turn"
    ]

    # 1. Build parent-child relationships for the graph (only for renderable turns)
    child_to_parents: Dict[str, List[str]] = {}
    parent_to_children: Dict[str, List[str]] = {}
    turn_by_id: Dict[str, Dict] = {turn['turn_id']: turn for turn in renderable_turns}
    all_turn_ids = set(turn_by_id.keys())

    for turn in renderable_turns:
        turn_id = turn['turn_id']
        if turn_id not in parent_to_children:
            parent_to_children[turn_id] = []

        source_ids = turn.get("source_turn_ids", [])
        if source_ids:
            # Filter to only include sources that exist in our turn set
            valid_sources = [sid for sid in source_ids if sid in all_turn_ids]
            if valid_sources:
                child_to_parents[turn_id] = valid_sources
                for source_id in valid_sources:
                    if source_id not in parent_to_children:
                        parent_to_children[source_id] = []
                    parent_to_children[source_id].append(turn_id)

    # 2. Identify epochs (disconnected subgraphs) via flood-fill from roots
    def find_connected_component(start_id: str, visited: set) -> set:
        """Find all nodes connected to start_id (bidirectional traversal)."""
        component = set()
        stack = [start_id]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            component.add(node_id)
            # Traverse both directions
            for child_id in parent_to_children.get(node_id, []):
                if child_id not in visited:
                    stack.append(child_id)
            for parent_id in child_to_parents.get(node_id, []):
                if parent_id not in visited:
                    stack.append(parent_id)
        return component

    visited_global = set()
    epochs: List[set] = []  # Each epoch is a set of turn_ids

    for turn_id in all_turn_ids:
        if turn_id not in visited_global:
            component = find_connected_component(turn_id, visited_global)
            if component:
                epochs.append(component)

    # 3. Sort epochs by the earliest timestamp in each epoch (time flows down)
    def get_epoch_start_time(epoch: set) -> str:
        """Get the earliest timestamp in an epoch for sorting."""
        times = [turn_by_id[tid].get('start_time', '') for tid in epoch if tid in turn_by_id]
        return min(times) if times else ''

    epochs.sort(key=get_epoch_start_time)

    # 4. Calculate depth within each epoch using BFS, then apply epoch offset
    turn_depths: Dict[str, int] = {}
    epoch_info: Dict[str, int] = {}  # turn_id -> epoch_index
    current_depth_offset = 0
    EPOCH_SEPARATOR_DEPTH_GAP = 1  # Gap for the epoch separator node
    initial_epoch_header = None  # Track if we need an "Epoch 1" header at the top

    # If multiple epochs, add space for initial "Epoch 1" header
    if len(epochs) > 1:
        first_epoch_start_time = get_epoch_start_time(epochs[0])
        first_epoch_roots = [tid for tid in epochs[0] if not child_to_parents.get(tid)]
        if not first_epoch_roots:
            first_epoch_roots = [min(epochs[0], key=lambda tid: turn_by_id.get(tid, {}).get('start_time', ''))]

        initial_epoch_header = {
            "id": "epoch-header-0",
            "depth": 0,  # At the very top
            "epoch_index": 0,
            "start_time": first_epoch_start_time,
            "target_turn_ids": first_epoch_roots,  # Connect TO first epoch's root nodes
        }
        current_depth_offset = 1  # Start epoch 1 content at depth 1

    for epoch_idx, epoch_turn_ids in enumerate(epochs):
        # Find roots within this epoch (nodes with no parents in the epoch)
        epoch_roots = [tid for tid in epoch_turn_ids if not child_to_parents.get(tid)]

        # If no explicit roots, use earliest timestamped node
        if not epoch_roots:
            epoch_roots = [min(epoch_turn_ids, key=lambda tid: turn_by_id.get(tid, {}).get('start_time', ''))]

        # BFS within this epoch
        epoch_depths: Dict[str, int] = {}
        queue = deque()

        for root_id in epoch_roots:
            epoch_depths[root_id] = 1
            queue.append(root_id)

        visited_in_epoch = set()
        while queue:
            current_id = queue.popleft()
            if current_id in visited_in_epoch:
                continue
            visited_in_epoch.add(current_id)

            current_depth = epoch_depths.get(current_id, 1)

            for child_id in parent_to_children.get(current_id, []):
                if child_id in epoch_turn_ids:  # Only process children in this epoch
                    new_depth = max(epoch_depths.get(child_id, 0), current_depth + 1)
                    epoch_depths[child_id] = new_depth
                    if child_id not in visited_in_epoch:
                        queue.append(child_id)

        # Apply epoch offset to all depths in this epoch
        max_depth_in_epoch = max(epoch_depths.values()) if epoch_depths else 0

        for tid, local_depth in epoch_depths.items():
            turn_depths[tid] = local_depth + current_depth_offset
            epoch_info[tid] = epoch_idx

        # Create epoch separator if this is not the last epoch
        if epoch_idx < len(epochs) - 1:
            separator_depth = current_depth_offset + max_depth_in_epoch + 1
            epoch_end_time = max(
                (turn_by_id[tid].get('start_time', '') for tid in epoch_turn_ids if tid in turn_by_id),
                default=''
            )
            next_epoch_start_time = get_epoch_start_time(epochs[epoch_idx + 1])

            # Find the last turn(s) in this epoch (leaves) to connect separator FROM
            epoch_leaves = [tid for tid in epoch_turn_ids
                          if not any(child in epoch_turn_ids for child in parent_to_children.get(tid, []))]

            # Find the root turn(s) of the NEXT epoch to connect separator TO
            next_epoch_turn_ids = epochs[epoch_idx + 1]
            next_epoch_roots = [tid for tid in next_epoch_turn_ids if not child_to_parents.get(tid)]
            if not next_epoch_roots:
                # Fallback: use earliest timestamped node in next epoch
                next_epoch_roots = [min(next_epoch_turn_ids, key=lambda tid: turn_by_id.get(tid, {}).get('start_time', ''))]

            separator_id = f"epoch-sep-{epoch_idx}"
            epoch_separators.append({
                "id": separator_id,
                "depth": separator_depth,
                "epoch_index": epoch_idx,
                "end_time": epoch_end_time,
                "next_start_time": next_epoch_start_time,
                "source_turn_ids": epoch_leaves,  # Connect FROM epoch's leaf nodes
                "target_turn_ids": next_epoch_roots,  # Connect TO next epoch's root nodes
            })

            # Update offset for next epoch (current max + separator + gap)
            current_depth_offset = separator_depth + EPOCH_SEPARATOR_DEPTH_GAP
        else:
            current_depth_offset += max_depth_in_epoch

    # ==================== EPOCH-BASED DEPTH CALCULATION END ================

    for turn in hydrated_sorted_turns:
        # Filter out Partner's turns and user_turn
        if turn.get("agent_info", {}).get("agent_id") == "Partner" or turn.get("turn_type") == "user_turn":
            continue

        turn_id = turn["turn_id"]
        node_id = f"delimiter-{turn_id}" if turn.get("turn_type") == "restart_delimiter_turn" else f"turn-{turn_id}"

        # Prepare node data
        agent_info = turn.get("agent_info", {})
        display_name = (
            agent_info.get("assigned_role_name") or
            agent_info.get("profile_logical_name") or
            agent_info.get("agent_id")
        )
        node_data = {
            "label": display_name,
            "nodeType": "turn",
            "status": turn.get("status", "idle"),
            "timestamp": turn.get("start_time"),
            "originalId": turn_id,
            "turn_id": turn_id,
            "agent_id": agent_info.get("agent_id"),
            "tool_interactions": turn.get("tool_interactions", []),
            "content_stream_id": None,
            "final_content": None,
            "depth": turn_depths.get(turn_id, 1) # Add depth field here
        }

        # Special handling for different turn types
        if turn.get("turn_type") == "restart_delimiter_turn":
            node_data["nodeType"] = "gather"
            node_data["label"] = "Flow Restarted"

        if turn.get("turn_type") == "aggregation_turn":
            node_data["nodeType"] = "gather"
            node_data["label"] = "Gather"

        final_content_parts = []

        # 1. Get LLM response content
        llm_interaction = turn.get("llm_interaction")
        llm_content = ""
        if llm_interaction:
            if llm_interaction.get("attempts") and llm_interaction["attempts"]:
                node_data["content_stream_id"] = llm_interaction["attempts"][-1].get("stream_id")

            final_response = llm_interaction.get("final_response")
            if final_response:
                llm_content = final_response.get("content", "") or ""

        # 2. Get and format hydrated tool results
        tool_interactions = turn.get("tool_interactions", [])
        tool_content_parts = []
        if tool_interactions:
            tool_content_parts.append("**Tool Execution:**\n")
            for interaction in tool_interactions:
                tool_name = interaction.get("tool_name")
                input_params = interaction.get("input_params")
                tool_content_parts.append(f"\n* **Tool:** `{tool_name}`")
                if input_params:
                    tool_content_parts.append(f"\n  * **Parameters:**\n")
                    formatted_params = _format_tool_result_as_markdown(input_params, indent_level=2)
                    tool_content_parts.append(formatted_params)
                else:
                    tool_content_parts.append("    *No content returned.*")

        # 3. Intelligently assemble the final content
        if llm_content:
            final_content_parts.append(llm_content)

        if tool_content_parts:
            if final_content_parts:
                final_content_parts.append("\n\n---\n")
            final_content_parts.append("".join(tool_content_parts))

        if final_content_parts:
            node_data["final_content"] = "".join(final_content_parts)

        nodes_by_id[node_id] = {
            "id": node_id,
            "type": "custom",
            "data": node_data
        }

    # --- Step 2: Create all edges ---
    turn_manager = run_context.get("runtime", {}).get("turn_manager")

    for turn in hydrated_sorted_turns:
        if turn.get("agent_info", {}).get("agent_id") == "Partner":
            continue

        if turn.get("turn_type") == "user_turn":
            continue

        # Determine target node ID based on turn type
        if turn.get("turn_type") == "restart_delimiter_turn":
            target_node_id = f"delimiter-{turn['turn_id']}"
        else:
            target_node_id = f"turn-{turn['turn_id']}"

        if target_node_id not in nodes_by_id:
            continue

        source_ids = turn.get("source_turn_ids", [])

        for source_turn_id in source_ids:
            source_node_id = f"turn-{source_turn_id}" # Default to turn prefix

            # Check if the source is a special turn type to adjust its node ID
            source_turn = None
            if turn_manager:
                source_turn = turn_manager._get_turn_by_id(team_state, source_turn_id)
            else: # Fallback for safety
                source_turn = next((t for t in hydrated_sorted_turns if t.get("turn_id") == source_turn_id), None)

            if source_turn and source_turn.get("turn_type") == "restart_delimiter_turn":
                source_node_id = f"delimiter-{source_turn_id}"

            if source_node_id in nodes_by_id:
                is_return_edge = nodes_by_id.get(target_node_id, {}).get('data', {}).get('nodeType') == 'gather'
                edges.append({
                    "id": f"{source_node_id}->{target_node_id}",
                    "source": source_node_id,
                    "target": target_node_id,
                    "animated": turn.get("status") == "running",
                    "edgeType": "return" if is_return_edge else None
                })

    # --- Step 3: Add epoch separator nodes and their edges ---
    for sep in epoch_separators:
        sep_node_id = sep["id"]

        # Create the epoch separator node
        # Label indicates the epoch STARTING below (epoch_index is 0-based, separator is after that epoch)
        nodes_by_id[sep_node_id] = {
            "id": sep_node_id,
            "type": "custom",
            "data": {
                "label": f"Epoch {sep['epoch_index'] + 2}",
                "nodeType": "epoch_separator",
                "status": "completed",
                "timestamp": sep["end_time"],
                "depth": sep["depth"],
                "epoch_index": sep["epoch_index"],
                "originalId": sep_node_id,
            }
        }

        # Create edges FROM epoch leaf nodes TO the separator
        for source_turn_id in sep["source_turn_ids"]:
            source_node_id = f"turn-{source_turn_id}"
            # Check if source might be a delimiter
            source_turn = turn_by_id.get(source_turn_id)
            if source_turn and source_turn.get("turn_type") == "restart_delimiter_turn":
                source_node_id = f"delimiter-{source_turn_id}"

            if source_node_id in nodes_by_id:
                edges.append({
                    "id": f"{source_node_id}->{sep_node_id}",
                    "source": source_node_id,
                    "target": sep_node_id,
                    "animated": False,
                    "edgeType": "epoch_boundary"
                })

        # Create edges FROM the separator TO next epoch's root nodes
        for target_turn_id in sep.get("target_turn_ids", []):
            target_node_id = f"turn-{target_turn_id}"
            # Check if target might be a delimiter
            target_turn = turn_by_id.get(target_turn_id)
            if target_turn and target_turn.get("turn_type") == "restart_delimiter_turn":
                target_node_id = f"delimiter-{target_turn_id}"

            if target_node_id in nodes_by_id:
                edges.append({
                    "id": f"{sep_node_id}->{target_node_id}",
                    "source": sep_node_id,
                    "target": target_node_id,
                    "animated": False,
                    "edgeType": "epoch_boundary"
                })

    # --- Step 4: Add initial "Epoch 1" header if multiple epochs exist ---
    if initial_epoch_header:
        header_node_id = initial_epoch_header["id"]
        nodes_by_id[header_node_id] = {
            "id": header_node_id,
            "type": "custom",
            "data": {
                "label": "Epoch 1",
                "nodeType": "epoch_separator",
                "status": "completed",
                "timestamp": initial_epoch_header["start_time"],
                "depth": initial_epoch_header["depth"],
                "epoch_index": 0,
                "originalId": header_node_id,
            }
        }

        # Create edges FROM the header TO first epoch's root nodes
        for target_turn_id in initial_epoch_header.get("target_turn_ids", []):
            target_node_id = f"turn-{target_turn_id}"
            target_turn = turn_by_id.get(target_turn_id)
            if target_turn and target_turn.get("turn_type") == "restart_delimiter_turn":
                target_node_id = f"delimiter-{target_turn_id}"

            if target_node_id in nodes_by_id:
                edges.append({
                    "id": f"{header_node_id}->{target_node_id}",
                    "source": header_node_id,
                    "target": target_node_id,
                    "animated": False,
                    "edgeType": "epoch_boundary"
                })

    return {"nodes": list(nodes_by_id.values()), "edges": edges}


def _generate_timeline_view_model(run_context: Dict) -> Dict:
    """
    (V3 - Refactored) Generates the Timeline view model based on the team_state.turns list.
    """
    team_state = run_context.get("team_state", {})
    turns = team_state.get("turns", [])
    is_principal_running = team_state.get("is_principal_flow_running", False)

    if not turns:
        return {"lanes": [], "overallStartTime": None, "overallEndTime": None, "timeBreaks": [], "isLive": False}

    lanes_data: Dict[str, List] = {}
    all_events_for_timing = []

    # Sort turns chronologically to correctly process events and time breaks
    sorted_turns = sorted(turns, key=lambda t: t.get('start_time', ''))

    for turn in sorted_turns:
        agent_id = turn.get("agent_info", {}).get("agent_id")
        if not agent_id or agent_id == "Partner": # Filter out Partner from the timeline
            continue

        if agent_id not in lanes_data:
            lanes_data[agent_id] = []

        # Create a block for the agent's turn itself (thinking time)
        turn_block = {
            "moduleId": turn["turn_id"],
            "moduleName": f"Turn ({turn.get('status', '...')})",
            "blockType": "turn", # Add a type for frontend styling
            "startTime": turn["start_time"],
            "endTime": turn["end_time"],
            "status": turn["status"].upper(),
            "agent_id": agent_id
        }
        lanes_data[agent_id].append(turn_block)
        all_events_for_timing.append(turn_block)

        # Create blocks for each tool interaction within the turn
        for tool_interaction in turn.get("tool_interactions", []):
            if not tool_interaction.get("start_time"): continue

            tool_block = {
                "moduleId": tool_interaction["tool_call_id"],
                "moduleName": f"Tool: {tool_interaction['tool_name']}",
                "blockType": "tool", # Add a type for frontend styling
                "startTime": tool_interaction["start_time"],
                "endTime": tool_interaction["end_time"],
                "status": tool_interaction["status"].upper(),
                "agent_id": agent_id
            }
            lanes_data[agent_id].append(tool_block)
            all_events_for_timing.append(tool_block)

    if not all_events_for_timing:
        return {"lanes": [], "overallStartTime": None, "overallEndTime": None, "timeBreaks": [], "isLive": False}

    # Calculate overallStartTime and overallEndTime from all collected events
    overall_start_time = min(datetime.fromisoformat(e["startTime"]) for e in all_events_for_timing)

    now_utc = datetime.now(timezone.utc)
    if is_principal_running:
        overall_end_time = now_utc
    else:
        end_times = [datetime.fromisoformat(e["endTime"]) for e in all_events_for_timing if e.get("endTime")]
        overall_end_time = max(end_times) if end_times else overall_start_time

    # Calculate timeBreaks (this logic can be refined, but shows the principle)
    time_breaks = []
    # Sort all events by start time to find gaps
    sorted_events = sorted(all_events_for_timing, key=lambda e: e['startTime'])
    for i in range(1, len(sorted_events)):
        prev_event = sorted_events[i-1]
        curr_event = sorted_events[i]

        if prev_event.get("endTime") and curr_event.get("startTime"):
            try:
                gap_start_dt = datetime.fromisoformat(prev_event["endTime"])
                gap_end_dt = datetime.fromisoformat(curr_event["startTime"])
                duration = (gap_end_dt - gap_start_dt).total_seconds()

                if duration > 1: # Only record significant breaks
                    time_breaks.append({
                        "breakStart": gap_start_dt.isoformat(),
                        "breakEnd": gap_end_dt.isoformat(),
                        "duration": duration
                    })
            except (ValueError, TypeError) as e:
                logger.warning("timestamp_parse_error_for_time_break", extra={"error": str(e)})

    # Assemble final lanes
    final_lanes = []
    for agent_id, blocks in lanes_data.items():
        final_lanes.append({"agentId": agent_id, "blocks": sorted(blocks, key=lambda x: x["startTime"])})

    # Sort lanes to have Principal first, then others alphabetically
    final_lanes.sort(key=lambda x: (not x["agentId"].startswith("Principal"), x["agentId"]))

    return {
        "lanes": final_lanes,
        "overallStartTime": overall_start_time.isoformat(),
        "overallEndTime": overall_end_time.isoformat(),
        "timeBreaks": time_breaks,
        "isLive": is_principal_running
    }


def _generate_kanban_view_model(run_context: Dict) -> Dict:
    """
    Generates the view model for the Kanban board.
    """
    team_state = run_context.get("team_state", {})
    work_modules = team_state.get("work_modules", {})
    # ongoing_tasks = run_context.get("_ongoing_associate_tasks", {}) # Old path
    ongoing_tasks = run_context.get("sub_context_refs", {}).get("_ongoing_associate_tasks", {}) # New path

    view_by_status: Dict[str, List] = {
        "pending": [], "ongoing": [], "pending_review": [],
        "completed": [], "deprecated": []
    }
    view_by_agent: Dict[str, List] = {}

    for mod_id, module in work_modules.items():
        status = module.get("status", "pending")

        enriched_module = {
            "module_id": mod_id,
            "name": module.get("name"),
            "description": module.get("description"),
            "status": status,
            "updated_at": module.get("updated_at"),
            "assignee_history": module.get("assignee_history", []),
            "review_info": module.get("review_info"),
            "is_rework": len(module.get("context_archive", [])) > 0,
            "agent_id": None, # Add agent_id
            "current_assignee_id": None,
            "live_status_summary": None,
            "latest_deliverables_summary": None,
        }

        context_archive = module.get("context_archive", [])
        if context_archive:
            last_archive_entry = context_archive[-1]
            deliverables = last_archive_entry.get("deliverables")
            if deliverables:
                enriched_module["latest_deliverables_summary"] = f"{len(deliverables)} deliverable(s) found."

        assignee_id = "unassigned"
        if status == "ongoing":
            running_assignee = next((h for h in reversed(enriched_module["assignee_history"]) if h.get("outcome") == "running"), None)
            if running_assignee:
                assignee_id = running_assignee.get("dispatch_id")
                enriched_module["current_assignee_id"] = assignee_id
                if assignee_id in ongoing_tasks:
                    enriched_module["live_status_summary"] = "Task is live and running."

        enriched_module["agent_id"] = assignee_id

        if status in view_by_status:
            view_by_status[status].append(enriched_module)

        if assignee_id not in view_by_agent:
            view_by_agent[assignee_id] = []
        view_by_agent[assignee_id].append(enriched_module)

    return {
        "view_by_status": view_by_status,
        "view_by_agent": view_by_agent,
        "last_updated": datetime.now().isoformat(),
    }
