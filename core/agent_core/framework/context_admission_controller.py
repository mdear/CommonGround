"""
Context Admission Controller - Pre-admission budget enforcement for tool results.

Ensures tool results don't spike context past WARNING threshold, giving agents
the opportunity to respond to budget warnings before hitting CRITICAL/EXCEEDED.

This prevents scenarios where a single large tool result (e.g., web_search 
returning 108 items) can jump context from 18% to 94%, bypassing all warning
thresholds and leaving the agent no chance to wrap up gracefully.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .context_budget_guardian import (
    WARNING_THRESHOLD,
    get_model_context_limit
)


# Module-level cache for default model to avoid repeated lookups
_default_model_cache: Optional[str] = None


def _get_default_model() -> str:
    """Get a default model for token counting when none specified."""
    global _default_model_cache
    if _default_model_cache is None:
        # Use Claude as default since it's the primary model in CommonGround
        _default_model_cache = "claude-sonnet-4-20250514"
    return _default_model_cache


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens in text using the provider-appropriate token counter.
    
    Uses the token_counter module which provides accurate counts via:
    - Anthropic's official count_tokens API for Claude models
    - tiktoken for OpenAI models
    - litellm fallback for other providers
    
    Args:
        text: Input text to count tokens for
        model: Optional model name for accurate provider-specific counting.
               If not provided, uses a default model.
    
    Returns:
        Token count
    """
    if not text:
        return 0
    
    try:
        from ..llm.token_counter import count_tokens
        model_to_use = model or _get_default_model()
        return count_tokens(model=model_to_use, text=text)
    except Exception as e:
        # Fallback to heuristic if token_counter fails
        logger.warning("token_counter_fallback", extra={
            "error": str(e),
            "text_length": len(text)
        })
        # ~4 chars per token is a reasonable fallback for English text
        return max(1, len(text) // 4)

logger = logging.getLogger(__name__)


@dataclass
class AdmissionDecision:
    """Result of pre-admission check."""
    admit_full: bool                          # True if result fits without truncation
    admitted_content: Dict                    # Content to add to context
    deferred_content: Optional[List] = None   # Content stored in KB for later
    deferred_kb_tokens: List[str] = field(default_factory=list)  # KB tokens for deferred
    truncation_notice: Optional[str] = None   # Notice for agent about truncation
    
    # Metrics
    original_tokens: int = 0
    admitted_tokens: int = 0
    deferred_tokens: int = 0
    post_admission_utilization: float = 0.0


# Target utilization after admitting tool results
# Set slightly below WARNING to give agent breathing room
ADMISSION_TARGET_UTILIZATION = 0.38  # 38%, just under WARNING at 40%

# Minimum tokens to always admit (don't block small results)
MIN_ADMISSION_TOKENS = 5000

# Maximum single-item size before considering chunking
MAX_SINGLE_ITEM_TOKENS = 20000


def calculate_admission_budget(
    current_tokens: int,
    context_limit: int,
    target_utilization: float = ADMISSION_TARGET_UTILIZATION
) -> int:
    """
    Calculate how many tokens can be admitted while staying under target utilization.
    
    Args:
        current_tokens: Current context token count
        context_limit: Model's context window limit
        target_utilization: Target utilization after admission (default 38%)
    
    Returns:
        Maximum tokens that can be admitted
    """
    target_tokens = int(context_limit * target_utilization)
    available = target_tokens - current_tokens
    
    # Ensure at least some minimum admission (don't block all results)
    return max(available, MIN_ADMISSION_TOKENS)


def check_pre_admission(
    tool_result: Dict,
    current_context_tokens: int,
    model_name: str,
    llm_config: Optional[Dict] = None,
    agent_id: Optional[str] = None
) -> AdmissionDecision:
    """
    Check if tool result can be admitted without exceeding WARNING threshold.
    
    If result would push context past WARNING:
    1. Truncate/prioritize content to fit within budget
    2. Store excess in KB with tokens for later expansion
    3. Add truncation notice for agent
    
    Args:
        tool_result: The tool's exec_async result
        current_context_tokens: Current context window usage
        model_name: Model identifier for context limit lookup
        llm_config: Optional LLM config
        agent_id: Agent ID for logging
    
    Returns:
        AdmissionDecision with admitted/deferred content
    """
    context_limit = get_model_context_limit(model_name, llm_config)
    admission_budget = calculate_admission_budget(current_context_tokens, context_limit)
    
    # Get content to evaluate
    payload = tool_result.get("payload", {})
    kb_items = tool_result.get("_knowledge_items_to_add", [])
    
    # Calculate token counts using provider-specific counter
    payload_str = _serialize_payload(payload)
    payload_tokens = estimate_tokens(payload_str, model=model_name)
    
    kb_content_tokens = 0
    for item in kb_items:
        content = item.get("content", "")
        if isinstance(content, str):
            kb_content_tokens += estimate_tokens(content, model=model_name)
    
    total_result_tokens = payload_tokens + kb_content_tokens
    
    # Check if full admission is possible
    post_admission_tokens = current_context_tokens + total_result_tokens
    post_admission_utilization = post_admission_tokens / context_limit if context_limit > 0 else 1.0
    
    if post_admission_utilization <= WARNING_THRESHOLD:
        # Full admission - no truncation needed
        logger.debug("pre_admission_full_admit", extra={
            "agent_id": agent_id,
            "result_tokens": total_result_tokens,
            "post_utilization": f"{post_admission_utilization:.1%}"
        })
        return AdmissionDecision(
            admit_full=True,
            admitted_content=tool_result,
            deferred_content=None,
            deferred_kb_tokens=[],
            truncation_notice=None,
            original_tokens=total_result_tokens,
            admitted_tokens=total_result_tokens,
            deferred_tokens=0,
            post_admission_utilization=post_admission_utilization
        )
    
    # Truncation needed
    logger.warning("pre_admission_truncation_required", extra={
        "agent_id": agent_id,
        "original_tokens": total_result_tokens,
        "budget_tokens": admission_budget,
        "current_utilization": f"{current_context_tokens/context_limit:.1%}" if context_limit > 0 else "N/A",
        "would_be_utilization": f"{post_admission_utilization:.1%}"
    })
    
    return _truncate_result(
        tool_result=tool_result,
        kb_items=kb_items,
        payload=payload,
        payload_tokens=payload_tokens,
        admission_budget=admission_budget,
        current_context_tokens=current_context_tokens,
        context_limit=context_limit,
        agent_id=agent_id,
        model_name=model_name
    )


def _serialize_payload(payload: Any) -> str:
    """Convert payload to string for token estimation."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        # Convert dict to JSON-like string representation
        import json
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            return str(payload)
    return str(payload)


def _truncate_result(
    tool_result: Dict,
    kb_items: List[Dict],
    payload: Any,
    payload_tokens: int,
    admission_budget: int,
    current_context_tokens: int,
    context_limit: int,
    agent_id: Optional[str],
    model_name: Optional[str] = None
) -> AdmissionDecision:
    """
    Truncate tool result to fit within admission budget.
    
    Strategy:
    1. Always admit the payload (it's usually small and essential)
    2. For KB items: prioritize by relevance signals, admit top N
    3. Defer remaining KB items (still stored, just not in context)
    """
    # Reserve space for payload (minimum 2K tokens or actual size)
    payload_budget = min(payload_tokens, max(2000, admission_budget // 4))
    kb_budget = max(0, admission_budget - payload_budget)
    
    # If no KB items, just return with payload
    if not kb_items:
        post_utilization = (current_context_tokens + payload_tokens) / context_limit if context_limit > 0 else 1.0
        return AdmissionDecision(
            admit_full=True,  # No KB items to defer
            admitted_content=tool_result,
            deferred_content=None,
            deferred_kb_tokens=[],
            truncation_notice=None,
            original_tokens=payload_tokens,
            admitted_tokens=payload_tokens,
            deferred_tokens=0,
            post_admission_utilization=post_utilization
        )
    
    # Sort KB items by quality signals
    scored_items = _score_and_sort_kb_items(kb_items, model=model_name)
    
    # Admit items until budget exhausted
    admitted_items = []
    deferred_items = []
    admitted_kb_tokens = 0
    
    for item, score in scored_items:
        content = item.get("content", "")
        item_tokens = estimate_tokens(content, model=model_name) if isinstance(content, str) else 0
        
        if admitted_kb_tokens + item_tokens <= kb_budget:
            admitted_items.append(item)
            admitted_kb_tokens += item_tokens
        else:
            # Defer this item - store in KB but don't include in context
            deferred_items.append(item)
    
    # Build deferred KB tokens list
    deferred_kb_tokens = []
    for i, item in enumerate(deferred_items):
        token = item.get("token") or item.get("item_id") or f"<#CGKB-DEFERRED-{i}>"
        deferred_kb_tokens.append(token)
    
    # Build truncation notice
    truncation_notice = None
    if deferred_items:
        deferred_token_count = sum(
            estimate_tokens(item.get("content", ""), model=model_name) 
            for item in deferred_items 
            if isinstance(item.get("content"), str)
        )
        display_tokens = deferred_kb_tokens[:5]
        truncation_notice = (
            f"\n\n---\n"
            f"⚠️ **Context Budget Notice**: {len(deferred_items)} additional items "
            f"(~{deferred_token_count:,} tokens) were stored in the knowledge base "
            f"but not included in context to prevent overflow.\n\n"
            f"**Available KB tokens for expansion**: `{', '.join(display_tokens)}`"
            f"{'...' if len(deferred_kb_tokens) > 5 else ''}\n\n"
            f"Use these tokens with the knowledge base tools if you need the additional content.\n"
            f"---"
        )
    
    # Build modified result
    modified_result = tool_result.copy()
    modified_result["_knowledge_items_to_add"] = admitted_items
    modified_result["_deferred_items"] = deferred_items
    modified_result["_deferred_kb_tokens"] = deferred_kb_tokens
    
    # Add notice to payload if possible
    modified_payload = _add_truncation_notice_to_payload(payload, truncation_notice)
    if modified_payload is not None:
        modified_result["payload"] = modified_payload
    
    # Calculate final metrics
    admitted_tokens = payload_tokens + admitted_kb_tokens
    deferred_tokens = sum(
        estimate_tokens(item.get("content", "")) 
        for item in deferred_items 
        if isinstance(item.get("content"), str)
    )
    post_utilization = (current_context_tokens + admitted_tokens) / context_limit if context_limit > 0 else 1.0
    
    logger.info("pre_admission_truncation_complete", extra={
        "agent_id": agent_id,
        "items_admitted": len(admitted_items),
        "items_deferred": len(deferred_items),
        "admitted_tokens": admitted_tokens,
        "deferred_tokens": deferred_tokens,
        "post_utilization": f"{post_utilization:.1%}"
    })
    
    return AdmissionDecision(
        admit_full=False,
        admitted_content=modified_result,
        deferred_content=deferred_items,
        deferred_kb_tokens=deferred_kb_tokens,
        truncation_notice=truncation_notice,
        original_tokens=admitted_tokens + deferred_tokens,
        admitted_tokens=admitted_tokens,
        deferred_tokens=deferred_tokens,
        post_admission_utilization=post_utilization
    )


def _add_truncation_notice_to_payload(payload: Any, notice: Optional[str]) -> Optional[Any]:
    """Add truncation notice to payload if it's a dict with known fields."""
    if notice is None:
        return None
    
    if not isinstance(payload, dict):
        return None
    
    modified = payload.copy()
    
    # Try common payload fields
    if "instructional_prompt" in modified:
        modified["instructional_prompt"] = str(modified["instructional_prompt"]) + notice
        return modified
    
    if "result" in modified:
        modified["result"] = str(modified["result"]) + notice
        return modified
    
    if "content" in modified:
        modified["content"] = str(modified["content"]) + notice
        return modified
    
    if "message" in modified:
        modified["message"] = str(modified["message"]) + notice
        return modified
    
    # Add as a new field if no known field found
    modified["_truncation_notice"] = notice
    return modified


def _score_and_sort_kb_items(items: List[Dict], model: Optional[str] = None) -> List[Tuple[Dict, float]]:
    """
    Score KB items by quality/relevance signals for prioritization.
    
    Scoring factors:
    - Source authority (academic > news > general)
    - Content length (prefer substantial content, 500-5000 tokens)
    - Relevance signals in metadata
    - Position (earlier items may be more relevant)
    
    Args:
        items: List of KB items to score
        model: Optional model name for accurate token counting
    
    Returns:
        List of (item, score) tuples sorted by score descending
    """
    scored = []
    
    for i, item in enumerate(items):
        score = 0.0
        content = item.get("content", "")
        metadata = item.get("metadata", {})
        source_uri = item.get("source_uri", "") or ""
        
        # Content substance score (prefer 500-5000 token items)
        content_tokens = estimate_tokens(content, model=model) if isinstance(content, str) else 0
        if 500 <= content_tokens <= 5000:
            score += 2.0
        elif content_tokens > 5000:
            score += 1.0  # Long content is still valuable
        elif content_tokens > 100:
            score += 0.5
        elif content_tokens < 50:
            score -= 1.0  # Very short content is less valuable
        
        # Source authority score
        source_lower = source_uri.lower()
        if any(domain in source_lower for domain in ['.gov', '.edu', 'scholar.google', 'pubmed', 'doi.org']):
            score += 3.0
        elif any(domain in source_lower for domain in ['nature.com', 'sciencedirect', 'springer', 'wiley', 'jstor']):
            score += 2.5
        elif any(domain in source_lower for domain in ['wikipedia', 'britannica', 'who.int']):
            score += 1.5
        elif any(domain in source_lower for domain in ['medium.com', 'blog', 'reddit']):
            score -= 0.5  # Less authoritative sources
        
        # Relevance metadata boost
        if isinstance(metadata, dict):
            if metadata.get("relevance_score"):
                try:
                    score += float(metadata["relevance_score"])
                except (ValueError, TypeError):
                    pass
            
            # Boost if explicitly marked as important
            if metadata.get("priority") == "high":
                score += 1.0
        
        # Position penalty (slight preference for earlier items)
        # This assumes search results are somewhat ordered by relevance
        score -= (i / 100)  # -0.01 per position
        
        scored.append((item, score))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def estimate_result_tokens(tool_result: Dict, model: Optional[str] = None) -> int:
    """
    Estimate total tokens in a tool result.
    
    Args:
        tool_result: Tool's exec_async result
        model: Optional model name for accurate token counting
    
    Returns:
        Token count
    """
    payload = tool_result.get("payload", {})
    kb_items = tool_result.get("_knowledge_items_to_add", [])
    
    payload_str = _serialize_payload(payload)
    payload_tokens = estimate_tokens(payload_str, model=model)
    
    kb_tokens = 0
    for item in kb_items:
        content = item.get("content", "")
        if isinstance(content, str):
            kb_tokens += estimate_tokens(content, model=model)
    
    return payload_tokens + kb_tokens
