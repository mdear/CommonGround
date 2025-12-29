"""
Context Budget Guardian - Circuit breaker for agent context window management.

This module provides proactive context window monitoring to prevent agents from
hitting ContextWindowExceededError by forcing graceful completion when approaching
the limit.

Design Principles:
- Monitor, don't truncate: We track consumption and trigger early completion
- Respect agent autonomy: At warning threshold, inject guidance; at critical, force action
- Fail gracefully: Even at critical threshold, we guide to completion rather than crash
"""

import logging
from typing import Dict, Optional, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ContextBudgetStatus(Enum):
    """Status levels for context budget consumption."""
    HEALTHY = auto()      # < 40% - Normal operation
    WARNING = auto()      # 40-55% - Inject guidance to wrap up
    CRITICAL = auto()     # 55-70% - Force immediate completion
    EXCEEDED = auto()     # > 70% - Circuit breaker, 30% remains for wrap-up


# Default context limits by model family (tokens)
# These are conservative estimates to leave room for response
DEFAULT_CONTEXT_LIMITS = {
    "anthropic/claude-sonnet-4": 200000,
    "anthropic/claude-sonnet-4-5": 200000,
    "anthropic/claude-3-5-haiku": 200000,
    "anthropic/claude-3-5-sonnet": 200000,
    "anthropic/claude-3-opus": 200000,
    "openai/gpt-4o": 128000,
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4": 8192,
    "openai/gpt-3.5-turbo": 16385,
    "gemini/gemini-2.5-pro": 1000000,
    "gemini/gemini-2.5-flash": 1000000,
}

# Models that support extended 1M context with the anthropic-beta header
MODELS_SUPPORTING_1M_CONTEXT = {
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-3-5-sonnet",
}

# Threshold percentages
# These are set to leave 30% headroom for final summarization/wrap-up
WARNING_THRESHOLD = 0.40   # 40% - Start suggesting wrap-up
CRITICAL_THRESHOLD = 0.55  # 55% - Force completion
EXCEEDED_THRESHOLD = 0.70  # 70% - Circuit breaker triggers, 30% remains for wrap-up

# Budget allocation constants
SUMMARIZATION_RESERVE_PERCENT = 0.30  # Reserve 30% for summarization and wrap-up


def calculate_worker_budget(
    model_context_limit: int,
    num_workers: int,
    base_context_overhead: int = 30000,
    summarization_reserve: float = SUMMARIZATION_RESERVE_PERCENT
) -> Dict:
    """
    Calculate the per-worker token budget based on model capacity and number of workers.

    Algorithm:
    1. Start with model's max context limit
    2. Subtract base context overhead (system prompt, tools, etc.)
    3. Reserve 30% for summarization and wrap-up
    4. Divide remaining 70% evenly among workers

    Args:
        model_context_limit: The model's maximum context window (e.g., 200000 or 1000000)
        num_workers: Number of parallel worker agents
        base_context_overhead: Fixed overhead for system prompt, tools, etc. (default 30K)
        summarization_reserve: Fraction to reserve for summarization (default 0.30)

    Returns:
        Dict with budget allocation details:
        - total_available: Total context after overhead
        - summarization_budget: Reserved for wrap-up
        - worker_budget_total: Total budget for all workers
        - per_worker_budget: Budget per individual worker
        - num_workers: Number of workers
    """
    # Available context after base overhead
    total_available = model_context_limit - base_context_overhead

    # Reserve portion for summarization
    summarization_budget = int(total_available * summarization_reserve)

    # Remaining budget for workers
    worker_budget_total = total_available - summarization_budget

    # Per-worker allocation
    per_worker_budget = worker_budget_total // num_workers if num_workers > 0 else worker_budget_total

    result = {
        "model_context_limit": model_context_limit,
        "base_context_overhead": base_context_overhead,
        "total_available": total_available,
        "summarization_budget": summarization_budget,
        "summarization_reserve_percent": summarization_reserve * 100,
        "worker_budget_total": worker_budget_total,
        "per_worker_budget": per_worker_budget,
        "num_workers": num_workers
    }

    logger.info("worker_budget_calculated", extra=result)
    return result


def get_model_context_limit(model_name: str, llm_config: Optional[Dict] = None) -> int:
    """
    Gets the context limit for a given model.

    Priority:
    1. Explicit max_context_tokens in llm_config
    2. 1M context if model supports it AND extra_headers contains anthropic-beta context header
    3. Model family match in DEFAULT_CONTEXT_LIMITS
    4. Conservative default (100000)

    Args:
        model_name: The model identifier (e.g., "anthropic/claude-sonnet-4-20250514")
        llm_config: Optional LLM configuration dict that may contain max_context_tokens

    Returns:
        The context token limit for the model
    """
    # Check explicit config first
    if llm_config and llm_config.get("max_context_tokens"):
        return llm_config["max_context_tokens"]

    # Check if 1M context is enabled via extra_headers
    if llm_config:
        extra_headers = llm_config.get("extra_headers")
        if extra_headers and isinstance(extra_headers, dict):
            beta_header = extra_headers.get("anthropic-beta", "")
            if "context-1m" in beta_header or "1m" in beta_header.lower():
                # Check if model supports 1M context
                for model_prefix in MODELS_SUPPORTING_1M_CONTEXT:
                    if model_name.startswith(model_prefix):
                        logger.info("context_limit_1m_enabled", extra={
                            "model_name": model_name,
                            "limit": 1000000
                        })
                        return 1000000

    # Try exact match
    if model_name in DEFAULT_CONTEXT_LIMITS:
        return DEFAULT_CONTEXT_LIMITS[model_name]

    # Try prefix match (handles versioned model names like claude-sonnet-4-20250514)
    for model_prefix, limit in DEFAULT_CONTEXT_LIMITS.items():
        if model_name.startswith(model_prefix):
            return limit

    # Try matching without provider prefix (e.g., "claude-sonnet-4-20250514" should match "anthropic/claude-sonnet-4")
    for model_prefix, limit in DEFAULT_CONTEXT_LIMITS.items():
        # Extract the model family part after the provider prefix (e.g., "claude-sonnet-4" from "anthropic/claude-sonnet-4")
        if "/" in model_prefix:
            model_family = model_prefix.split("/", 1)[1]
            if model_name.startswith(model_family):
                return limit

    # Conservative default
    logger.warning("context_limit_unknown_model", extra={
        "model_name": model_name,
        "default_limit": 100000,
        "hint": "Add max_context_tokens to LLM config for accurate limit"
    })
    return 100000


def assess_context_budget(
    predicted_tokens: int,
    model_name: str,
    llm_config: Optional[Dict] = None,
    agent_id: Optional[str] = None
) -> Tuple[ContextBudgetStatus, Dict]:
    """
    Assesses the current context budget consumption and returns status with metadata.

    Args:
        predicted_tokens: The estimated token count for the current turn
        model_name: The model identifier
        llm_config: Optional LLM configuration
        agent_id: Optional agent ID for logging

    Returns:
        Tuple of (status, metadata_dict) where metadata includes:
        - context_limit: The total context limit
        - predicted_tokens: The input token count
        - utilization_percent: Percentage of context used
        - remaining_tokens: Estimated remaining capacity
        - recommendation: Human-readable recommendation
    """
    context_limit = get_model_context_limit(model_name, llm_config)
    utilization = predicted_tokens / context_limit if context_limit > 0 else 1.0
    remaining = context_limit - predicted_tokens

    metadata = {
        "context_limit": context_limit,
        "predicted_tokens": predicted_tokens,
        "utilization_percent": round(utilization * 100, 1),
        "remaining_tokens": max(0, remaining),
        "model_name": model_name,
        "agent_id": agent_id
    }

    if utilization >= EXCEEDED_THRESHOLD:
        status = ContextBudgetStatus.EXCEEDED
        metadata["recommendation"] = "EMERGENCY: Context limit exceeded. Request will likely fail."
        logger.error("context_budget_exceeded", extra=metadata)

    elif utilization >= CRITICAL_THRESHOLD:
        status = ContextBudgetStatus.CRITICAL
        metadata["recommendation"] = "CRITICAL: Must complete immediately. Call generate_message_summary now."
        logger.warning("context_budget_critical", extra=metadata)

    elif utilization >= WARNING_THRESHOLD:
        status = ContextBudgetStatus.WARNING
        metadata["recommendation"] = "WARNING: Context budget running low. Begin wrapping up your work."
        logger.info("context_budget_warning", extra=metadata)

    else:
        status = ContextBudgetStatus.HEALTHY
        metadata["recommendation"] = "Healthy context budget. Continue normal operation."
        logger.debug("context_budget_healthy", extra=metadata)

    return status, metadata


def generate_context_budget_directive(
    status: ContextBudgetStatus,
    metadata: Dict,
    agent_type: Optional[str] = None
) -> Optional[str]:
    """
    Generates a system directive to inject based on context budget status.

    For HEALTHY status, returns None (no injection needed).
    For WARNING/CRITICAL/EXCEEDED, returns a directive to guide the agent.

    Args:
        status: The current ContextBudgetStatus
        metadata: Metadata from assess_context_budget
        agent_type: The agent type ("principal", "partner", "associate", etc.)

    Returns:
        A directive string to inject, or None if no injection needed
    """
    if status == ContextBudgetStatus.HEALTHY:
        return None

    utilization = metadata.get("utilization_percent", 0)
    remaining = metadata.get("remaining_tokens", 0)

    # Select appropriate tool name based on agent type
    if agent_type in ("principal", "partner"):
        wrap_up_tool = "finish_flow"
        wrap_up_action = "conclude your analysis and finalize results"
    else:
        wrap_up_tool = "generate_message_summary"
        wrap_up_action = "summarize your findings and submit your deliverable"

    if status == ContextBudgetStatus.WARNING:
        return f"""
⚠️ **CONTEXT BUDGET WARNING** ⚠️

Your context utilization is at {utilization}% ({remaining:,} tokens remaining).

**Action Required:**
- Begin consolidating your findings
- Avoid making additional large queries
- Plan to call `{wrap_up_tool}` within the next 1-2 turns
- If you have sufficient information, call `{wrap_up_tool}` NOW

Continue with your current task but prioritize completion.
"""

    elif status == ContextBudgetStatus.CRITICAL:
        return f"""
🚨 **CRITICAL: CONTEXT BUDGET EXHAUSTED** 🚨

Your context utilization is at {utilization}% ({remaining:,} tokens remaining).

**MANDATORY ACTION:**
You MUST call `{wrap_up_tool}` tool IMMEDIATELY to {wrap_up_action}.

Do NOT make any more search or query tool calls. Any additional queries will cause a system failure.

{wrap_up_action.capitalize()} NOW.
"""

    elif status == ContextBudgetStatus.EXCEEDED:
        return f"""
🛑 **EMERGENCY: CONTEXT LIMIT EXCEEDED** 🛑

Your context utilization is at {utilization}% - the system is at risk of failure.

**EMERGENCY ACTION:**
Call `{wrap_up_tool}` IMMEDIATELY to {wrap_up_action}.

Your response must be MINIMAL. Include only:
1. The most important finding
2. A note that full analysis was interrupted due to context limits

This is your FINAL opportunity to submit work before system failure.
"""

    return None


def should_force_tool_call(status: ContextBudgetStatus, agent_type: Optional[str] = None) -> Optional[str]:
    """
    Determines if a forced tool call should be injected.

    At CRITICAL or EXCEEDED status, we may want to programmatically
    force the agent to call the summary tool rather than relying on
    the directive alone.

    Args:
        status: The current ContextBudgetStatus
        agent_type: The agent type ("principal", "partner", "associate", etc.)

    Returns:
        Tool name to force, or None if no forced call needed
    """
    if status in (ContextBudgetStatus.CRITICAL, ContextBudgetStatus.EXCEEDED):
        # Principal and Partner agents use finish_flow to wrap up
        # Associates use generate_message_summary to submit deliverables
        if agent_type in ("principal", "partner"):
            return "finish_flow"
        return "generate_message_summary"
    return None


def synthesize_partial_results(
    team_state: Dict,
    triggered_agent_id: Optional[str] = None,
    budget_metadata: Optional[Dict] = None
) -> Dict:
    """
    Synthesize partial results when circuit breaker fires.

    Instead of returning nothing when the context budget is exceeded,
    this function compiles available work from completed modules into
    an actionable summary that can be returned to the user.

    Args:
        team_state: The team state containing work_modules
        triggered_agent_id: ID of the agent that triggered the circuit breaker
        budget_metadata: Metadata from the budget assessment

    Returns:
        A structured report containing:
        - What was requested (original question)
        - What was completed (modules with deliverables)
        - What remains incomplete (in-progress or pending modules)
        - Key findings extracted from deliverables
    """
    work_modules = team_state.get("work_modules", {})
    original_question = team_state.get("question", "Unknown question")

    completed_modules = []
    incomplete_modules = []
    key_findings = []
    tools_used_overall = set()

    for module_id, module in work_modules.items():
        status = module.get("status", "unknown")
        objective = module.get("objective", "Unknown objective")
        assigned_agent = module.get("assigned_agent_id", "Unassigned")

        if status in ("completed", "done", "COMPLETED"):
            deliverables = module.get("deliverables", {})

            completed_modules.append({
                "module_id": module_id,
                "objective": objective,
                "agent": assigned_agent,
                "has_deliverables": bool(deliverables),
                "deliverable_summary": _extract_deliverable_summary(deliverables)
            })

            # Extract key findings from deliverables
            findings = _extract_key_findings(deliverables)
            if findings:
                key_findings.extend(findings)

            # Track tools used
            tools = module.get("tools_used", [])
            if isinstance(tools, list):
                tools_used_overall.update(tools)
        else:
            incomplete_modules.append({
                "module_id": module_id,
                "objective": objective,
                "status": status,
                "agent": assigned_agent
            })

    # Build the synthesis report
    utilization = budget_metadata.get("utilization_percent", 0) if budget_metadata else 0

    synthesis = {
        "circuit_breaker_triggered": True,
        "triggered_by": triggered_agent_id,
        "original_question": original_question[:500] + "..." if len(original_question) > 500 else original_question,
        "context_utilization_at_trigger": f"{utilization:.1f}%",
        "summary": {
            "total_modules_planned": len(work_modules),
            "modules_completed": len(completed_modules),
            "modules_incomplete": len(incomplete_modules),
            "tools_employed": list(tools_used_overall)[:10]  # Limit to 10 tools
        },
        "completed_work": completed_modules,
        "incomplete_work": incomplete_modules,
        "key_findings": key_findings[:10],  # Limit to top 10 findings
        "user_message": _generate_user_message(
            completed_modules,
            incomplete_modules,
            key_findings,
            utilization
        )
    }

    logger.info("circuit_breaker_synthesis_generated", extra={
        "triggered_by": triggered_agent_id,
        "completed_modules": len(completed_modules),
        "incomplete_modules": len(incomplete_modules),
        "key_findings_count": len(key_findings)
    })

    return synthesis


def _extract_deliverable_summary(deliverables: Dict) -> str:
    """Extract a brief summary from deliverables dict."""
    if not deliverables:
        return "No deliverables captured"

    if isinstance(deliverables, str):
        return deliverables[:200] + "..." if len(deliverables) > 200 else deliverables

    if isinstance(deliverables, dict):
        # Look for common summary keys
        for key in ("summary", "main_finding", "conclusion", "result", "answer"):
            if key in deliverables:
                val = str(deliverables[key])
                return val[:200] + "..." if len(val) > 200 else val

        # Fall back to listing available keys
        keys = list(deliverables.keys())[:5]
        return f"Contains: {', '.join(keys)}"

    return str(deliverables)[:200]


def _extract_key_findings(deliverables: Dict) -> list:
    """Extract key findings from deliverables for synthesis."""
    findings = []

    if not deliverables or not isinstance(deliverables, dict):
        return findings

    # Look for findings/conclusions/recommendations
    finding_keys = ("findings", "key_findings", "conclusions", "recommendations",
                   "insights", "summary", "main_points")

    for key in finding_keys:
        if key in deliverables:
            value = deliverables[key]
            if isinstance(value, list):
                findings.extend([str(f)[:200] for f in value[:3]])  # First 3 items
            elif isinstance(value, str):
                findings.append(value[:200])
            elif isinstance(value, dict):
                findings.append(str(value)[:200])

    return findings


def _generate_user_message(
    completed: list,
    incomplete: list,
    findings: list,
    utilization: float
) -> str:
    """Generate a user-friendly message about the partial results."""
    parts = []

    parts.append(f"**Note: Work was interrupted at {utilization:.0f}% context utilization.**\n")

    if completed:
        parts.append(f"✅ **Completed**: {len(completed)} work module(s)")
        for mod in completed[:3]:  # Show first 3
            parts.append(f"  - {mod['objective'][:80]}")

    if incomplete:
        parts.append(f"\n⏳ **Incomplete**: {len(incomplete)} work module(s) remain")
        for mod in incomplete[:3]:
            parts.append(f"  - {mod['objective'][:80]} (Status: {mod['status']})")

    if findings:
        parts.append("\n📌 **Key Findings So Far:**")
        for finding in findings[:5]:
            parts.append(f"  - {finding[:150]}")

    if incomplete:
        parts.append("\n*The incomplete work can be resumed in a follow-up session.*")

    return "\n".join(parts)


class ContextBudgetGuardian:
    """
    Stateful guardian that tracks context consumption across an agent's lifetime.

    This class can be attached to an agent to provide turn-over-turn tracking
    and trend analysis.
    """

    def __init__(
        self,
        model_name: str,
        llm_config: Optional[Dict] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None
    ):
        self.model_name = model_name
        self.llm_config = llm_config
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.context_limit = get_model_context_limit(model_name, llm_config)
        self.turn_history: list = []
        self.warnings_issued = 0
        self.critical_issued = False

    def record_turn(self, predicted_tokens: int) -> Tuple[ContextBudgetStatus, Dict]:
        """
        Records a turn's token consumption and returns the assessment.

        Args:
            predicted_tokens: Token count for this turn

        Returns:
            Tuple of (status, metadata)
        """
        status, metadata = assess_context_budget(
            predicted_tokens=predicted_tokens,
            model_name=self.model_name,
            llm_config=self.llm_config,
            agent_id=self.agent_id
        )

        self.turn_history.append({
            "predicted_tokens": predicted_tokens,
            "status": status.name,
            "utilization_percent": metadata["utilization_percent"]
        })

        # Track escalation
        if status == ContextBudgetStatus.WARNING:
            self.warnings_issued += 1
        elif status in (ContextBudgetStatus.CRITICAL, ContextBudgetStatus.EXCEEDED):
            self.critical_issued = True

        # Add trend info to metadata
        metadata["turn_count"] = len(self.turn_history)
        metadata["warnings_issued"] = self.warnings_issued
        metadata["critical_issued"] = self.critical_issued

        if len(self.turn_history) >= 2:
            prev_util = self.turn_history[-2]["utilization_percent"]
            curr_util = metadata["utilization_percent"]
            metadata["utilization_trend"] = curr_util - prev_util
            metadata["trend_direction"] = "increasing" if curr_util > prev_util else "stable_or_decreasing"

        return status, metadata

    def get_directive(self, status: ContextBudgetStatus, metadata: Dict) -> Optional[str]:
        """
        Gets the appropriate directive based on status and history.

        May adjust directive based on whether warnings have already been issued.
        """
        directive = generate_context_budget_directive(status, metadata, agent_type=self.agent_type)

        # If we've already issued warnings but agent hasn't wrapped up, escalate language
        if directive and self.warnings_issued > 2 and status == ContextBudgetStatus.WARNING:
            directive = directive.replace(
                "Continue with your current task but prioritize completion.",
                "You have received multiple warnings. PRIORITIZE COMPLETION NOW."
            )

        return directive
