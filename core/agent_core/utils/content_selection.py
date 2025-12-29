"""
Content Selection Utilities for Budget-Aware Context Inheritance.

This module provides utilities for selecting content from completed work modules
within a token/character budget. It implements a two-tier selection strategy:

1. Tier 1 (Preferred): Use deliverables.primary_summary (LLM-generated summary)
2. Tier 2 (Fallback): Select raw messages newest-first with hydration before measurement

Design Principles:
- Never truncate individual messages - this loses coherence
- Always hydrate KB tokens BEFORE measuring size - prevents post-selection expansion
- Prefer LLM summaries when available - cleaner, no KB tokens to expand

See docs/architecture/context-budget-management.md for detailed documentation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

# ==============================================================================
# MODULE-LEVEL CONSTANTS
# ==============================================================================

# Approximate characters per token for budget calculations
# This is a conservative estimate (actual varies by content type)
CHARS_PER_TOKEN = 4

# Fraction of target agent's context window reserved for inherited content
# 40% ensures sufficient headroom for the agent's own work
INHERITANCE_BUDGET_FRACTION = 0.40

# Strategy identifiers for logging and metadata
STRATEGY_LLM_SUMMARY = "llm_summary"
STRATEGY_NEWEST_FIRST = "newest_first"
STRATEGY_EMPTY = "empty"


# ==============================================================================
# BUDGET COMPUTATION
# ==============================================================================

def compute_inheritance_budget_chars(
    target_context_limit_tokens: int,
    num_sources: int,
    inheritance_fraction: float = INHERITANCE_BUDGET_FRACTION
) -> int:
    """
    Compute per-source character budget for content inheritance.

    Formula:
        pool_tokens = target_context_limit_tokens * inheritance_fraction
        pool_chars = pool_tokens * CHARS_PER_TOKEN
        per_source_budget = pool_chars // num_sources

    Args:
        target_context_limit_tokens: The spawning agent's context window in tokens
                                     (e.g., 200000 for claude-sonnet-4)
        num_sources: Number of source modules in inherit_messages_from
        inheritance_fraction: Fraction of context reserved for inheritance
                             (default: 0.40 = 40%)

    Returns:
        Per-source character budget (integer)

    Example:
        >>> compute_inheritance_budget_chars(200000, 2)
        160000  # (200K * 0.40 / 2) * 4 chars/token
    """
    if num_sources <= 0:
        logger.warning("inheritance_budget_invalid_sources", extra={
            "num_sources": num_sources,
            "fallback": 0
        })
        return 0

    pool_tokens = int(target_context_limit_tokens * inheritance_fraction)
    pool_chars = pool_tokens * CHARS_PER_TOKEN
    per_source_budget = pool_chars // num_sources

    logger.debug("inheritance_budget_computed", extra={
        "target_context_tokens": target_context_limit_tokens,
        "inheritance_fraction": inheritance_fraction,
        "pool_tokens": pool_tokens,
        "pool_chars": pool_chars,
        "num_sources": num_sources,
        "per_source_budget_chars": per_source_budget
    })

    return per_source_budget


# ==============================================================================
# CONTENT SELECTION
# ==============================================================================

def _estimate_message_chars(message: Dict) -> int:
    """
    Estimate the character count of a message.

    Counts the 'content' field if present. For tool_calls, estimates based on
    function name and arguments.
    """
    total = 0

    # Count content
    content = message.get("content")
    if content:
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            # Multi-modal content (list of content blocks)
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "")))
                else:
                    total += len(str(block))
        else:
            total += len(str(content))

    # Count tool_calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        fn = tc.get("function", {})
        total += len(fn.get("name", ""))
        total += len(str(fn.get("arguments", "")))

    # Add overhead for role, etc.
    total += 50  # Conservative overhead for message structure

    return total


def _select_messages_newest_first(
    messages: List[Dict],
    budget_chars: int,
    source_id: str = "unknown"
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Select messages from newest to oldest until budget is exhausted.

    IMPORTANT: Messages should already be hydrated (KB tokens expanded)
    before calling this function to ensure accurate size measurement.

    Args:
        messages: List of message dicts (should be hydrated)
        budget_chars: Character budget for selection
        source_id: Identifier for logging

    Returns:
        Tuple of (selected_messages, metadata)
        - selected_messages: List of selected messages in original order
        - metadata: Dict with selection statistics
    """
    if not messages:
        return [], {
            "strategy": STRATEGY_EMPTY,
            "chars_used": 0,
            "items_selected": 0,
            "items_available": 0,
            "source_id": source_id
        }

    # Work from newest to oldest
    reversed_messages = list(reversed(messages))
    selected_indices = []  # Track original indices of selected messages
    chars_used = 0

    for idx, msg in enumerate(reversed_messages):
        msg_chars = _estimate_message_chars(msg)

        # Always include at least one message if nothing selected yet
        # (prevents returning empty when budget is very small)
        if chars_used + msg_chars <= budget_chars or not selected_indices:
            selected_indices.append(len(messages) - 1 - idx)  # Convert back to original index
            chars_used += msg_chars

            # If this single message already exceeds budget, stop here
            if chars_used > budget_chars:
                logger.debug("content_selection_single_message_exceeds_budget", extra={
                    "source_id": source_id,
                    "message_chars": msg_chars,
                    "budget_chars": budget_chars
                })
                break
        else:
            # Budget exhausted
            break

    # Restore original order
    selected_indices.sort()
    selected_messages = [messages[i] for i in selected_indices]

    metadata = {
        "strategy": STRATEGY_NEWEST_FIRST,
        "chars_used": chars_used,
        "items_selected": len(selected_messages),
        "items_available": len(messages),
        "source_id": source_id,
        "budget_chars": budget_chars,
        "oldest_selected_index": selected_indices[0] if selected_indices else None,
        "newest_selected_index": selected_indices[-1] if selected_indices else None
    }

    logger.debug("content_selection_newest_first_complete", extra=metadata)

    return selected_messages, metadata


def select_content_within_budget(
    deliverables: Optional[Dict],
    messages: List[Dict],
    budget_chars: int,
    source_id: str = "unknown"
) -> Tuple[Union[str, List[Dict]], Dict[str, Any]]:
    """
    Two-tier content selection within a character budget.

    Strategy:
    1. Tier 1 (Preferred): If deliverables.primary_summary exists AND fits budget,
       use it. LLM summaries are cleaner and don't contain KB tokens.

    2. Tier 2 (Fallback): Select messages newest-first until budget is exhausted.
       Messages MUST be hydrated before calling this function.

    Args:
        deliverables: Dict containing 'primary_summary' and/or other fields
        messages: List of message dicts (should be PRE-HYDRATED for accurate sizing)
        budget_chars: Character budget for selection
        source_id: Identifier for logging

    Returns:
        Tuple of (selected_content, metadata)
        - selected_content: Either summary string (Tier 1) OR list of messages (Tier 2)
        - metadata: Dict with "strategy", "chars_used", "items_selected", etc.

    Note:
        The caller is responsible for hydrating messages BEFORE calling this function.
        If messages contain KB tokens that will be expanded later, the budget
        calculation will be incorrect.
    """
    logger.info("inheritance_content_selection_started", extra={
        "source_id": source_id,
        "budget_chars": budget_chars,
        "has_deliverables": deliverables is not None,
        "message_count": len(messages) if messages else 0
    })

    # Tier 1: Try primary_summary from deliverables
    if deliverables:
        summary = deliverables.get("primary_summary")
        if summary and isinstance(summary, str):
            summary_chars = len(summary)

            if summary_chars <= budget_chars:
                metadata = {
                    "strategy": STRATEGY_LLM_SUMMARY,
                    "chars_used": summary_chars,
                    "items_selected": 1,
                    "items_available": len(messages) if messages else 0,
                    "source_id": source_id,
                    "budget_chars": budget_chars,
                    "summary_chars": summary_chars
                }
                logger.debug("content_selection_using_summary", extra=metadata)
                return summary, metadata
            else:
                logger.debug("content_selection_summary_exceeds_budget", extra={
                    "source_id": source_id,
                    "summary_chars": summary_chars,
                    "budget_chars": budget_chars,
                    "fallback": "newest_first"
                })

    # Tier 2: Fall back to message selection
    if messages:
        return _select_messages_newest_first(messages, budget_chars, source_id)

    # Nothing available
    return [], {
        "strategy": STRATEGY_EMPTY,
        "chars_used": 0,
        "items_selected": 0,
        "items_available": 0,
        "source_id": source_id,
        "budget_chars": budget_chars
    }


# ==============================================================================
# FORMATTING FOR BRIEFING
# ==============================================================================

def format_inherited_content_for_briefing(
    content: Union[str, List[Dict]],
    metadata: Dict[str, Any],
    source_id: str
) -> List[Dict]:
    """
    Format selected content as messages suitable for injection into agent briefing.

    Args:
        content: Either a summary string (from Tier 1) or list of messages (from Tier 2)
        metadata: Selection metadata from select_content_within_budget
        source_id: Identifier of the source work module

    Returns:
        List of message dicts formatted for briefing injection.
        Each message includes _internal metadata for observability.
    """
    strategy = metadata.get("strategy", STRATEGY_EMPTY)

    if strategy == STRATEGY_EMPTY:
        return []

    if strategy == STRATEGY_LLM_SUMMARY:
        # Wrap summary in a structured message
        return [{
            "role": "user",
            "content": f"[Context inherited from {source_id}]\n\n{content}",
            "_internal": {
                "_no_handover": True,  # Don't pass this on to subsequent agents
                "_inherited_from": source_id,
                "_selection_strategy": strategy,
                "_chars": metadata.get("chars_used", 0)
            }
        }]

    if strategy == STRATEGY_NEWEST_FIRST:
        # Return messages with inheritance metadata
        formatted = []
        for i, msg in enumerate(content):
            formatted_msg = msg.copy()
            # Add internal metadata
            formatted_msg["_internal"] = formatted_msg.get("_internal", {}).copy()
            formatted_msg["_internal"]["_no_handover"] = True
            formatted_msg["_internal"]["_inherited_from"] = source_id
            formatted_msg["_internal"]["_selection_strategy"] = strategy
            formatted_msg["_internal"]["_selection_index"] = i
            formatted.append(formatted_msg)

        # Add header message
        header = {
            "role": "user",
            "content": (
                f"[Context inherited from {source_id}: "
                f"{len(content)} messages selected, "
                f"{metadata.get('chars_used', 0):,} chars]"
            ),
            "_internal": {
                "_no_handover": True,
                "_inherited_from": source_id,
                "_is_header": True
            }
        }

        return [header] + formatted

    # Unknown strategy - return empty
    logger.warning("format_inherited_content_unknown_strategy", extra={
        "strategy": strategy,
        "source_id": source_id
    })
    return []


# ==============================================================================
# ASYNC HELPERS (for hydration integration)
# ==============================================================================

async def hydrate_messages_for_selection(
    messages: List[Dict],
    knowledge_base: Any,
    source_id: str = "unknown"
) -> List[Dict]:
    """
    Hydrate messages by expanding KB tokens before content selection.

    This MUST be called before select_content_within_budget when using
    Tier 2 (message selection) to ensure accurate size measurement.

    Args:
        messages: List of message dicts (may contain KB tokens like <#CGKB-00042>)
        knowledge_base: KnowledgeBase instance with hydrate_content method
        source_id: Identifier for logging

    Returns:
        List of hydrated messages with KB tokens expanded
    """
    if not messages:
        return []

    if not knowledge_base:
        logger.warning("hydrate_messages_no_kb", extra={
            "source_id": source_id,
            "message_count": len(messages),
            "warning": "Returning unhydrated messages"
        })
        return messages

    hydrated = []
    for msg in messages:
        hydrated_msg = msg.copy()
        try:
            content = msg.get("content")
            if content:
                hydrated_msg["content"] = await knowledge_base.hydrate_content(content)
        except Exception as e:
            logger.error("hydrate_messages_failed", extra={
                "source_id": source_id,
                "error": str(e)
            }, exc_info=True)
            # Keep original content on error
        hydrated.append(hydrated_msg)

    logger.debug("hydrate_messages_complete", extra={
        "source_id": source_id,
        "message_count": len(hydrated)
    })

    return hydrated


async def select_inherited_content_with_hydration(
    context_archive_entry: Dict,
    budget_chars: int,
    knowledge_base: Any,
    source_id: str = "unknown"
) -> Tuple[Union[str, List[Dict]], Dict[str, Any]]:
    """
    High-level convenience function that handles hydration and selection.

    This function:
    1. Extracts deliverables and messages from a context_archive entry
    2. Hydrates messages (expands KB tokens)
    3. Performs two-tier content selection

    Args:
        context_archive_entry: Dict with 'messages' and 'deliverables' keys
                               (typically context_archive[-1] from a work module)
        budget_chars: Character budget for selection
        knowledge_base: KnowledgeBase instance
        source_id: Identifier for logging

    Returns:
        Tuple of (selected_content, metadata)
    """
    deliverables = context_archive_entry.get("deliverables", {})
    messages = context_archive_entry.get("messages", [])

    # Hydrate messages first
    hydrated_messages = await hydrate_messages_for_selection(
        messages, knowledge_base, source_id
    )

    # Perform selection
    return select_content_within_budget(
        deliverables=deliverables,
        messages=hydrated_messages,
        budget_chars=budget_chars,
        source_id=source_id
    )
