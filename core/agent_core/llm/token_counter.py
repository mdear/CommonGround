"""
Provider-aware token counting module.

This module provides accurate token counting by using provider-specific APIs
when available, with fallback to litellm's estimation for other providers.

Architecture:
- Each LLM provider can have a dedicated token counter implementation
- Provider detection uses model name patterns (from llm_config.model)
- Graceful fallback to litellm when provider API is unavailable

Key insight: litellm uses tiktoken (OpenAI's tokenizer) as fallback for Claude 3+
models, which can underestimate by 25-35%. Anthropic's official count_tokens API
provides accurate counts for Claude models.

Extensibility:
- Add new provider by implementing _count_tokens_<provider>() function
- Register provider in PROVIDER_TOKEN_COUNTERS dict
- Add model detection pattern in _detect_provider()
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum

import litellm

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers with dedicated token counting."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    UNKNOWN = "unknown"


# ============================================================================
# Provider Detection
# ============================================================================

def _detect_provider(model: str) -> LLMProvider:
    """
    Detect the LLM provider from the model name.

    Handles various naming conventions from llm_configs:
    - anthropic/claude-* → Anthropic
    - claude-* → Anthropic
    - bedrock/anthropic.claude-* → Anthropic
    - gpt-*, openai/* → OpenAI
    - gemini-*, google/* → Google
    """
    if not model:
        return LLMProvider.UNKNOWN

    model_lower = model.lower()

    # Anthropic detection
    if (model_lower.startswith("claude") or
        "anthropic/" in model_lower or
        "/claude" in model_lower or
        "anthropic.claude" in model_lower):
        return LLMProvider.ANTHROPIC

    # OpenAI detection
    if (model_lower.startswith("gpt") or
        model_lower.startswith("o1") or
        model_lower.startswith("o3") or
        "openai/" in model_lower or
        model_lower.startswith("text-embedding")):
        return LLMProvider.OPENAI

    # Google detection
    if (model_lower.startswith("gemini") or
        "google/" in model_lower or
        "vertex_ai/" in model_lower):
        return LLMProvider.GOOGLE

    return LLMProvider.UNKNOWN


def _normalize_model_name(model: str, provider: LLMProvider) -> str:
    """
    Normalize model name by stripping provider prefixes.

    Converts litellm-style prefixes to bare model names for provider APIs.
    """
    if not model:
        return model

    # Common prefixes to strip based on provider
    prefix_map = {
        LLMProvider.ANTHROPIC: ["anthropic/", "bedrock/anthropic.", "vertex_ai/"],
        LLMProvider.OPENAI: ["openai/", "azure/"],
        LLMProvider.GOOGLE: ["google/", "vertex_ai/", "gemini/"],
    }

    prefixes = prefix_map.get(provider, [])
    normalized = model

    for prefix in prefixes:
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
            break

    return normalized


# ============================================================================
# Anthropic Token Counter
# ============================================================================

_anthropic_client = None


def _get_anthropic_client():
    """Lazily initialize and cache the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic()
        except ImportError:
            logger.warning("anthropic_sdk_not_installed", extra={
                "message": "anthropic package not installed, falling back to litellm estimation"
            })
            return None
        except Exception as e:
            logger.warning("anthropic_client_init_failed", extra={
                "error": str(e),
                "message": "Failed to initialize Anthropic client, falling back to litellm"
            })
            return None
    return _anthropic_client


def _convert_messages_for_anthropic(
    messages: List[Dict],
    system_prompt: Optional[str] = None
) -> Tuple[Optional[str], List[Dict]]:
    """
    Convert messages to Anthropic's expected format.

    Anthropic expects:
    - system as a separate parameter (not in messages)
    - messages with role: 'user' or 'assistant' only
    - tool_use and tool_result blocks handled specially
    """
    anthropic_messages = []
    extracted_system = system_prompt

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if extracted_system:
                extracted_system = f"{extracted_system}\n\n{content}"
            else:
                extracted_system = content
            continue

        if role in ("user", "assistant"):
            anthropic_msg = {"role": role, "content": content}

            # Handle tool_calls (assistant response with tool use)
            if "tool_calls" in msg and msg["tool_calls"]:
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})

                for tc in msg["tool_calls"]:
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": tc.get("function", {}).get("arguments", {})
                    }
                    if isinstance(tool_use_block["input"], str):
                        try:
                            import json
                            tool_use_block["input"] = json.loads(tool_use_block["input"])
                        except:
                            tool_use_block["input"] = {}
                    content_blocks.append(tool_use_block)

                anthropic_msg["content"] = content_blocks

            anthropic_messages.append(anthropic_msg)

        elif role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else str(content)
            }
            anthropic_messages.append({
                "role": "user",
                "content": [tool_result_block]
            })

    return extracted_system, anthropic_messages


def _count_tokens_anthropic(
    model: str,
    messages: List[Dict],
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict]] = None
) -> Optional[int]:
    """
    Count tokens using Anthropic's official API.

    Returns None if counting fails (caller should fall back to litellm).
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    try:
        anthropic_model = _normalize_model_name(model, LLMProvider.ANTHROPIC)
        system, anthropic_messages = _convert_messages_for_anthropic(messages, system_prompt)

        if not anthropic_messages:
            return 0

        kwargs = {
            "model": anthropic_model,
            "messages": anthropic_messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            anthropic_tools = []
            for tool in tools:
                if "function" in tool:
                    anthropic_tools.append({
                        "name": tool["function"].get("name", ""),
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"].get("parameters", {})
                    })
                else:
                    anthropic_tools.append(tool)
            kwargs["tools"] = anthropic_tools

        response = client.messages.count_tokens(**kwargs)

        logger.debug("anthropic_token_count_success", extra={
            "model": anthropic_model,
            "input_tokens": response.input_tokens,
            "message_count": len(anthropic_messages)
        })

        return response.input_tokens

    except Exception as e:
        logger.warning("anthropic_token_count_failed", extra={
            "model": model,
            "error": str(e),
            "fallback": "litellm"
        })
        return None


# ============================================================================
# OpenAI Token Counter (uses litellm - tiktoken is accurate for OpenAI)
# ============================================================================

def _count_tokens_openai(
    model: str,
    messages: List[Dict],
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict]] = None
) -> Optional[int]:
    """
    Count tokens for OpenAI models.

    litellm uses tiktoken which IS accurate for OpenAI models,
    so we just delegate directly.
    """
    # litellm's tiktoken is accurate for OpenAI - no need for special handling
    return None  # Signal to use litellm fallback


# ============================================================================
# Google Token Counter (placeholder for future implementation)
# ============================================================================

def _count_tokens_google(
    model: str,
    messages: List[Dict],
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict]] = None
) -> Optional[int]:
    """
    Count tokens for Google/Gemini models.

    Google provides countTokens API but requires different message format.
    For now, falls back to litellm. Can be implemented when needed.

    Reference: https://ai.google.dev/gemini-api/docs/tokens
    """
    # TODO: Implement Google's countTokens API when needed
    # from google import genai
    # client.models.count_tokens(model=model, contents=messages)
    return None  # Signal to use litellm fallback


# ============================================================================
# Provider Registry
# ============================================================================

# Map providers to their token counting functions
# Each function returns Optional[int] - None means "use litellm fallback"
PROVIDER_TOKEN_COUNTERS: Dict[LLMProvider, Callable] = {
    LLMProvider.ANTHROPIC: _count_tokens_anthropic,
    LLMProvider.OPENAI: _count_tokens_openai,
    LLMProvider.GOOGLE: _count_tokens_google,
}


# ============================================================================
# Litellm Fallback
# ============================================================================

def _count_tokens_litellm(model: str, messages: List[Dict]) -> int:
    """
    Count tokens using litellm's token_counter.

    This is the universal fallback for all providers.
    """
    try:
        return litellm.token_counter(model=model, messages=messages)
    except Exception as e:
        logger.warning("litellm_token_count_failed", extra={
            "model": model,
            "error": str(e),
            "return_value": 0
        })
        return 0


# ============================================================================
# Main API
# ============================================================================

def count_tokens(
    model: str,
    text: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    llm_config: Optional[Dict[str, Any]] = None
) -> int:
    """
    Count tokens using the most accurate method available for the given model.

    Provider routing:
    - Anthropic (Claude): Uses official count_tokens API (accurate)
    - OpenAI (GPT): Uses litellm/tiktoken (accurate for OpenAI)
    - Google (Gemini): Falls back to litellm (future: use countTokens API)
    - Unknown: Falls back to litellm

    Args:
        model: The model name (e.g., 'claude-sonnet-4-20250514', 'gpt-4')
               Typically comes from llm_config["model"]
        text: Optional text string to count (converted to user message)
        messages: Optional list of message dicts
        system_prompt: Optional system prompt
        tools: Optional list of tool definitions
        llm_config: Optional LLM config dict from LLMConfigResolver
                   May contain model override via litellm_token_counter_model

    Returns:
        Token count (0 on failure)
    """
    # Determine the model to use for counting
    model_for_counting = model
    if llm_config:
        # Check for explicit tokenizer model override
        if llm_config.get("litellm_token_counter_model"):
            model_for_counting = llm_config["litellm_token_counter_model"]
            logger.debug("token_counting_model_override", extra={
                "original_model": model,
                "override_model": model_for_counting
            })
        # Or use the model from config if not provided
        elif not model and llm_config.get("model"):
            model_for_counting = llm_config["model"]

    if not model_for_counting:
        logger.warning("token_count_no_model", extra={"return_value": 0})
        return 0

    # Validate input
    if text is not None and messages is not None:
        raise ValueError("Provide either 'text' or 'messages', not both.")

    # Build messages list
    messages_for_calc: List[Dict] = []

    if system_prompt:
        messages_for_calc.append({"role": "system", "content": system_prompt})

    if text is not None:
        messages_for_calc.append({"role": "user", "content": text})
    elif messages is not None:
        messages_for_calc.extend(messages)

    if not messages_for_calc:
        return 0

    # Detect provider and route to appropriate counter
    provider = _detect_provider(model_for_counting)

    if provider in PROVIDER_TOKEN_COUNTERS:
        counter_fn = PROVIDER_TOKEN_COUNTERS[provider]
        count = counter_fn(
            model=model_for_counting,
            messages=messages_for_calc,
            system_prompt=system_prompt,
            tools=tools
        )

        if count is not None:
            return count

        # Provider counter returned None - fall back to litellm
        logger.debug("provider_token_count_fallback", extra={
            "provider": provider.value,
            "model": model_for_counting,
            "fallback": "litellm"
        })

    # Use litellm for unknown providers or as fallback
    return _count_tokens_litellm(model_for_counting, messages_for_calc)


# ============================================================================
# Convenience exports for backward compatibility
# ============================================================================

def _is_anthropic_model(model: str) -> bool:
    """Check if model is an Anthropic Claude model. Exported for testing."""
    return _detect_provider(model) == LLMProvider.ANTHROPIC


def _normalize_model_for_anthropic(model: str) -> str:
    """Normalize model name for Anthropic API. Exported for testing."""
    return _normalize_model_name(model, LLMProvider.ANTHROPIC)
