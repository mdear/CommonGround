import logging
import re
from ..base_tool_node import BaseToolNode
from pocketflow import AsyncNode
from ...framework.tool_registry import tool_registry
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _extract_deliverables_from_messages(
    messages: List[Dict],
    module_description: str = "",
    summarization_budget_chars: int = 0
) -> Dict:
    """
    Extract deliverables from an Associate's message history.

    This is a FALLBACK mechanism when an Associate calls finish_flow without
    first calling generate_message_summary. The Associate SHOULD use the
    summarization tool for quality, but if they don't, we extract their work
    rather than losing it.

    DESIGN PRINCIPLES:
    1. NO TRUNCATION of individual findings - mission-critical content must be preserved
    2. PRIORITIZE recency - later messages usually contain conclusions
    3. ADAPTIVE to budget - if budget provided, be selective about WHICH findings to include
    4. PRESERVE FULL CONTENT of selected findings - never truncate mid-thought

    Args:
        messages: The agent's message history
        module_description: Description of the work module for context
        summarization_budget_chars: Optional budget from Principal's briefing.
            If provided, we SELECT fewer findings but keep them COMPLETE.
            If 0 or not provided, include all substantive findings.

    Returns:
        Dict with 'primary_summary' containing extracted findings
    """
    if not messages:
        return {}

    # Collect substantive content from assistant messages
    findings = []
    tools_used = set()

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Track tools used
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            if tool_name:
                tools_used.add(tool_name)

        # Extract meaningful content (skip very short or empty)
        if content and len(content.strip()) > 50:
            # Clean up the content - remove internal system markers only
            cleaned = re.sub(r'<internal>.*?</internal>', '', content, flags=re.DOTALL)
            cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<internal_system_directive>.*?</internal_system_directive>', '', cleaned, flags=re.DOTALL)
            cleaned = cleaned.strip()

            if cleaned and len(cleaned) > 30:
                findings.append(cleaned)

    if not findings:
        return {}

    # Selection strategy: NO TRUNCATION of content, but may SELECT fewer findings
    #
    # If budget is provided:
    #   - Prioritize the LAST N messages (conclusions are usually at the end)
    #   - Include complete findings that fit within budget
    #   - If a single finding exceeds budget, include it anyway (no truncation)
    #
    # If no budget:
    #   - Include all substantive findings (with reasonable max count)

    selected_findings = []
    total_chars = 0

    if summarization_budget_chars and summarization_budget_chars > 0:
        # Budget mode: be selective but preserve complete findings
        # Reserve ~20% for formatting overhead
        effective_budget = int(summarization_budget_chars * 0.80)

        # Work backwards (most recent = most likely to be conclusions)
        for finding in reversed(findings):
            finding_len = len(finding)

            # Always include at least ONE finding, even if it exceeds budget
            if not selected_findings:
                selected_findings.insert(0, finding)
                total_chars += finding_len
                continue

            # For subsequent findings, check budget
            if total_chars + finding_len <= effective_budget:
                selected_findings.insert(0, finding)
                total_chars += finding_len
            # else: skip this finding (but don't truncate it)
    else:
        # No budget: include all findings (reasonable max for sanity)
        max_findings_unbounded = 15  # Prevent extreme cases

        # Still prioritize recent findings
        for finding in reversed(findings[:max_findings_unbounded]):
            selected_findings.insert(0, finding)
            total_chars += len(finding)

    if not selected_findings:
        return {}

    # Format the summary - findings are COMPLETE, not truncated
    summary_parts = []

    if module_description:
        summary_parts.append(f"## Work Module: {module_description}\n")

    summary_parts.append("## Key Findings\n")

    for i, finding in enumerate(selected_findings, 1):
        # Present each finding as a complete block
        summary_parts.append(f"### Finding {i}\n{finding}\n")

    if tools_used:
        summary_parts.append(f"\n## Tools Used\n- {', '.join(sorted(tools_used))}")

    # Add metadata about extraction
    omitted_count = len(findings) - len(selected_findings)
    if omitted_count > 0:
        summary_parts.append(f"\n\n*Note: Auto-extracted {len(selected_findings)} of {len(findings)} work messages. "
                           f"{omitted_count} earlier messages omitted due to budget constraints. "
                           f"Full work log available in context_archive.*")
    else:
        summary_parts.append(f"\n\n*Note: Auto-extracted from complete work log ({len(findings)} messages)*")

    return {
        "primary_summary": "\n".join(summary_parts)
    }

@tool_registry(
    name="generate_message_summary",
    description="Called by an Associate Agent when its work on a module is complete. This tool prepares a structured prompt for the agent to generate its final, comprehensive deliverable summary. The agent should respect any summarization budget provided in the briefing.",
    parameters={
        "type": "object",
        "properties": {
            "current_associate_findings": {
                "type": "string",
                "description": "A detailed summary of the key findings, conclusions, and deliverables from the associate's work on the current module."
            }
        },
        "required": ["current_associate_findings"]
    },
    toolset_name="flow_control_summary"
)
class GenerateMessageSummaryTool(BaseToolNode):
    """
    A tool that generates an instructional prompt for an Associate Agent
    to create its final deliverable summary. Supports adaptive summarization
    based on budget provided by Principal.
    """
    async def exec_async(self, prep_res: Dict) -> Dict:
        tool_params = prep_res.get("tool_params", {})
        findings = tool_params.get("current_associate_findings")

        # Get summarization budget from the agent's briefing context
        sub_context = prep_res.get("sub_context", {})
        initial_briefing = sub_context.get("state", {}).get("initial_briefing", {})
        summarization_budget = initial_briefing.get("summarization_budget_chars", 0)
        module_description = sub_context.get("meta", {}).get("module_description", "the assigned task")

        # Build adaptive summarization instructions
        if summarization_budget and summarization_budget > 0:
            budget_instruction = f"""
## SUMMARIZATION BUDGET: {summarization_budget:,} characters

**CRITICAL**: The Principal has allocated approximately {summarization_budget:,} characters of context budget for your deliverable.
You MUST intelligently fit your summary within this budget by:
1. **Prioritize**: Keep details MOST pertinent to '{module_description}' in full detail.
2. **Summarize**: Condense less critical supporting information.
3. **Omit**: Drop tangential details that don't directly support the core findings.
4. **Reference**: For any omitted detail, note "Additional details available in work log" if important.

Your final JSON 'primary_summary' value should be approximately {summarization_budget:,} characters or fewer.
"""
        else:
            budget_instruction = """
## SUMMARIZATION: Full Detail Mode

No character budget was specified. Provide your complete, detailed findings without summarization.
Include all relevant information, sources, and supporting details.
"""

        instructional_prompt = f"""
<system_directive>
# FINALIZATION PROTOCOL INITIATED

Your tactical work on this module is complete. Your final action is to synthesize all your work into a structured 'deliverables' package for the Principal Agent.

{budget_instruction}

## Your Preliminary Findings (Provided by you):
<preliminary_findings>
{findings}
</preliminary_findings>

## Your Task:
Based on your preliminary findings and your entire message history for this task, formulate a final, comprehensive summary. This summary MUST be structured as a JSON object with a single key, 'primary_summary'. The value should be a detailed Markdown string.

**Example Output Format:**
```json
{{
  "primary_summary": "## 1. Sub-Task Summary\\n- **Objective**: ...\\n- **Outcome**: ...\\n\\n## 2. Key Findings\\n- Finding A: ...\\n- Finding B: ..."
}}
```

## Action:
1. In your NEXT response, provide the final JSON object in the 'content' field. Do NOT call any tools in that response.
2. In the FOLLOWING response, call `finish_flow` to signal completion.

This two-step process ensures your deliverable is properly captured. Example sequence:

**Response 1 (JSON):** `{{"primary_summary": "## My Findings\\n..."}}`
**Response 2 (Finish):** Call `finish_flow(reason="Deliverable submitted")`

⚠️ CRITICAL: You MUST call `finish_flow` after outputting your JSON - otherwise your work will not be captured!
</system_directive>
        """

        return {
            "status": "success",
            "payload": {
                "instructional_prompt": instructional_prompt.strip()
            }
        }

# The original FinishNode remains, as it's a separate tool.
@tool_registry(
    name="finish_flow",
    description="Signals that the current operational flow should conclude. Use this when all tasks are completed or a definitive end state is reached.",
    parameters={"type": "object", "properties": {
        "reason": {
            "type": "string",
            "description": "Optional reason for finishing the flow."
        }
    }},
    ends_flow=True,
    toolset_name="flow_control_end"
)
class FinishNode(AsyncNode):
    async def prep_async(self, shared: Dict) -> Dict:
        current_action = shared.get("state", {}).get("current_action", {})
        reason = "Flow concluded due to agent policy."
        if current_action:
            reason = current_action.get("reason", "No specific reason provided.")
        parent_agent_id_from_meta = shared['meta'].get("parent_agent_id")

        # Extract deliverables from message history for Associates
        # Associates have a parent_agent_id (the Principal who dispatched them)
        extracted_deliverables = {}
        if parent_agent_id_from_meta:
            # Check if Associate already produced structured deliverables
            # (e.g., via generate_message_summary if they have flow_control_summary toolset)
            existing_deliverables = shared.get("state", {}).get("deliverables", {})

            if existing_deliverables and existing_deliverables.get("primary_summary"):
                # Associate properly produced structured summary - use it
                extracted_deliverables = existing_deliverables
                logger.info("finish_flow_using_existing_deliverables", extra={
                    "agent_id": shared.get("meta", {}).get("agent_id"),
                    "summary_length": len(existing_deliverables.get("primary_summary", ""))
                })
            else:
                # Associate skipped summarization - auto-extract as fallback
                messages = shared.get("state", {}).get("messages", [])
                module_description = shared.get("meta", {}).get("module_description", "")

                # Get summarization budget from initial briefing if available
                initial_briefing = shared.get("state", {}).get("initial_briefing", {})
                summarization_budget = initial_briefing.get("summarization_budget_chars", 0)

                extracted_deliverables = _extract_deliverables_from_messages(
                    messages,
                    module_description,
                    summarization_budget_chars=summarization_budget
                )

                if extracted_deliverables:
                    logger.info("finish_flow_auto_extracted_deliverables", extra={
                        "agent_id": shared.get("meta", {}).get("agent_id"),
                        "summary_length": len(extracted_deliverables.get("primary_summary", "")),
                        "budget_used": summarization_budget
                    })

        return {
            "reason": reason,
            "shared_sub_context": shared,
            "parent_agent_id_for_event": parent_agent_id_from_meta,
            "extracted_deliverables": extracted_deliverables
        }

    async def exec_async(self, prep_res: Dict) -> Dict:
        reason = prep_res.get("reason", "No specific reason provided.")
        extracted_deliverables = prep_res.get("extracted_deliverables", {})

        logger.info("flow_ending", extra={"reason": reason})

        # Use extracted deliverables if available, otherwise fall back to state.final_report
        deliverables = extracted_deliverables
        if not deliverables:
            final_report = prep_res.get("shared_sub_context", {}).get('state', {}).get("final_report")
            if final_report:
                deliverables = {"final_report": final_report}

        return {
            "status": "flow_ending_initiated",
            "reason": reason,
            "result_package": {
                "status": "COMPLETED_SUCCESSFULLY",
                "final_summary": f"Flow completed as instructed. Reason: {reason}",
                "terminating_tool": "finish_flow",
                "error_details": None,
                "deliverables": deliverables
            }
        }

    async def post_async(self, shared: Dict, prep_res: Any, exec_res: Dict) -> Optional[str]:
        current_sub_context = prep_res.get("shared_sub_context")
        if current_sub_context and exec_res and "result_package" in exec_res:
            current_sub_context["state"]["final_result_package"] = exec_res["result_package"]
            # Also store deliverables in state for consistency
            deliverables = exec_res.get("result_package", {}).get("deliverables", {})
            if deliverables:
                current_sub_context["state"]["deliverables"] = deliverables
        return None
