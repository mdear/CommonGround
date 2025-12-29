"""
Unit tests for agent_core/llm/token_counter.py

Tests provider-aware token counting with Anthropic API and litellm fallback.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_core.llm.token_counter import (
    count_tokens,
    _is_anthropic_model,
    _normalize_model_for_anthropic,
    _convert_messages_for_anthropic,
    _count_tokens_litellm,
    _detect_provider,
    _normalize_model_name,
    LLMProvider,
    PROVIDER_TOKEN_COUNTERS,
)


class TestDetectProvider:
    """Tests for _detect_provider function."""

    def test_anthropic_direct_models(self):
        """Test detection of direct Claude model names."""
        assert _detect_provider("claude-3-sonnet-20240229") == LLMProvider.ANTHROPIC
        assert _detect_provider("claude-sonnet-4-20250514") == LLMProvider.ANTHROPIC
        assert _detect_provider("claude-3-opus-20240229") == LLMProvider.ANTHROPIC

    def test_anthropic_prefixed_models(self):
        """Test detection of provider-prefixed Claude models."""
        assert _detect_provider("anthropic/claude-3-sonnet") == LLMProvider.ANTHROPIC
        assert _detect_provider("bedrock/anthropic.claude-3-sonnet") == LLMProvider.ANTHROPIC

    def test_openai_models(self):
        """Test detection of OpenAI models."""
        assert _detect_provider("gpt-4") == LLMProvider.OPENAI
        assert _detect_provider("gpt-3.5-turbo") == LLMProvider.OPENAI
        assert _detect_provider("o1-preview") == LLMProvider.OPENAI
        assert _detect_provider("openai/gpt-4") == LLMProvider.OPENAI

    def test_google_models(self):
        """Test detection of Google/Gemini models."""
        assert _detect_provider("gemini-pro") == LLMProvider.GOOGLE
        assert _detect_provider("gemini-1.5-pro") == LLMProvider.GOOGLE
        assert _detect_provider("google/gemini-pro") == LLMProvider.GOOGLE
        assert _detect_provider("vertex_ai/gemini-pro") == LLMProvider.GOOGLE

    def test_unknown_models(self):
        """Test unknown models return UNKNOWN."""
        assert _detect_provider("llama-3-70b") == LLMProvider.UNKNOWN
        assert _detect_provider("mistral-7b") == LLMProvider.UNKNOWN
        assert _detect_provider("") == LLMProvider.UNKNOWN
        assert _detect_provider(None) == LLMProvider.UNKNOWN


class TestIsAnthropicModel:
    """Tests for _is_anthropic_model helper (backward compat)."""

    def test_direct_claude_models(self):
        """Test detection of direct Claude model names."""
        assert _is_anthropic_model("claude-3-sonnet-20240229") is True
        assert _is_anthropic_model("claude-sonnet-4-20250514") is True
        assert _is_anthropic_model("claude-3-opus-20240229") is True
        assert _is_anthropic_model("claude-3-haiku-20240307") is True

    def test_prefixed_claude_models(self):
        """Test detection of provider-prefixed Claude models."""
        assert _is_anthropic_model("anthropic/claude-3-sonnet") is True
        assert _is_anthropic_model("bedrock/anthropic.claude-3-sonnet") is True

    def test_non_claude_models(self):
        """Test non-Claude models return False."""
        assert _is_anthropic_model("gpt-4") is False
        assert _is_anthropic_model("gpt-3.5-turbo") is False
        assert _is_anthropic_model("gemini-pro") is False
        assert _is_anthropic_model("") is False
        assert _is_anthropic_model(None) is False


class TestNormalizeModelName:
    """Tests for _normalize_model_name helper."""

    def test_strips_anthropic_prefix(self):
        """Test stripping anthropic/ prefix."""
        assert _normalize_model_name("anthropic/claude-3-sonnet", LLMProvider.ANTHROPIC) == "claude-3-sonnet"

    def test_strips_bedrock_prefix(self):
        """Test stripping bedrock/anthropic. prefix."""
        assert _normalize_model_name("bedrock/anthropic.claude-3-sonnet", LLMProvider.ANTHROPIC) == "claude-3-sonnet"

    def test_strips_openai_prefix(self):
        """Test stripping openai/ prefix."""
        assert _normalize_model_name("openai/gpt-4", LLMProvider.OPENAI) == "gpt-4"

    def test_strips_google_prefix(self):
        """Test stripping google/ prefix."""
        assert _normalize_model_name("google/gemini-pro", LLMProvider.GOOGLE) == "gemini-pro"
        assert _normalize_model_name("vertex_ai/gemini-pro", LLMProvider.GOOGLE) == "gemini-pro"

    def test_leaves_bare_model_unchanged(self):
        """Test bare model names pass through."""
        assert _normalize_model_name("claude-3-sonnet", LLMProvider.ANTHROPIC) == "claude-3-sonnet"
        assert _normalize_model_name("gpt-4", LLMProvider.OPENAI) == "gpt-4"


class TestNormalizeModelForAnthropic:
    """Tests for _normalize_model_for_anthropic helper (backward compat)."""

    def test_strips_anthropic_prefix(self):
        """Test stripping anthropic/ prefix."""
        assert _normalize_model_for_anthropic("anthropic/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_bedrock_prefix(self):
        """Test stripping bedrock/anthropic. prefix."""
        assert _normalize_model_for_anthropic("bedrock/anthropic.claude-3-sonnet") == "claude-3-sonnet"

    def test_leaves_bare_model_unchanged(self):
        """Test bare model names pass through."""
        assert _normalize_model_for_anthropic("claude-3-sonnet") == "claude-3-sonnet"
        assert _normalize_model_for_anthropic("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


class TestConvertMessagesForAnthropic:
    """Tests for _convert_messages_for_anthropic helper."""

    def test_extracts_system_message(self):
        """Test system message extraction."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

        system, anthropic_msgs = _convert_messages_for_anthropic(messages)

        assert system == "You are helpful"
        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"

    def test_combines_system_prompts(self):
        """Test combining explicit system_prompt with system message."""
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"}
        ]

        system, _ = _convert_messages_for_anthropic(messages, system_prompt="You are helpful")

        assert "You are helpful" in system
        assert "Be concise" in system

    def test_converts_user_assistant_messages(self):
        """Test user/assistant messages pass through."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        _, anthropic_msgs = _convert_messages_for_anthropic(messages)

        assert len(anthropic_msgs) == 2
        assert anthropic_msgs[0] == {"role": "user", "content": "Hello"}
        assert anthropic_msgs[1] == {"role": "assistant", "content": "Hi there!"}

    def test_converts_tool_messages(self):
        """Test tool result messages are converted to user messages."""
        messages = [
            {"role": "tool", "tool_call_id": "call_123", "content": "Result data"}
        ]

        _, anthropic_msgs = _convert_messages_for_anthropic(messages)

        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"
        assert anthropic_msgs[0]["content"][0]["type"] == "tool_result"
        assert anthropic_msgs[0]["content"][0]["tool_use_id"] == "call_123"


class TestCountTokensLitellm:
    """Tests for _count_tokens_litellm fallback."""

    @patch("agent_core.llm.token_counter.litellm.token_counter")
    def test_calls_litellm(self, mock_counter):
        """Test litellm is called correctly."""
        mock_counter.return_value = 50

        result = _count_tokens_litellm("gpt-4", [{"role": "user", "content": "Hello"}])

        assert result == 50
        mock_counter.assert_called_once()

    @patch("agent_core.llm.token_counter.litellm.token_counter")
    def test_handles_exception(self, mock_counter):
        """Test graceful handling of litellm exceptions."""
        mock_counter.side_effect = Exception("Failed")

        result = _count_tokens_litellm("gpt-4", [{"role": "user", "content": "Hello"}])

        assert result == 0


class TestCountTokens:
    """Tests for the main count_tokens function."""

    @patch("agent_core.llm.token_counter._count_tokens_litellm")
    def test_uses_litellm_for_non_anthropic(self, mock_litellm):
        """Test non-Anthropic models use litellm."""
        mock_litellm.return_value = 25

        result = count_tokens(model="gpt-4", text="Hello world")

        assert result == 25
        mock_litellm.assert_called_once()

    def test_uses_anthropic_for_claude(self):
        """Test Claude models use Anthropic API when available."""
        mock_anthropic_fn = MagicMock(return_value=30)

        # Patch the dict entry directly since PROVIDER_TOKEN_COUNTERS holds function refs
        with patch.dict(
            "agent_core.llm.token_counter.PROVIDER_TOKEN_COUNTERS",
            {LLMProvider.ANTHROPIC: mock_anthropic_fn}
        ):
            result = count_tokens(model="claude-sonnet-4-20250514", text="Hello world")

        assert result == 30
        mock_anthropic_fn.assert_called_once()

    def test_falls_back_to_litellm_on_anthropic_failure(self):
        """Test fallback to litellm when Anthropic API fails."""
        mock_anthropic_fn = MagicMock(return_value=None)  # Indicates failure
        mock_litellm_fn = MagicMock(return_value=20)

        with patch.dict(
            "agent_core.llm.token_counter.PROVIDER_TOKEN_COUNTERS",
            {LLMProvider.ANTHROPIC: mock_anthropic_fn}
        ):
            with patch("agent_core.llm.token_counter._count_tokens_litellm", mock_litellm_fn):
                result = count_tokens(model="claude-3-sonnet", text="Hello")

        assert result == 20
        mock_anthropic_fn.assert_called_once()
        mock_litellm_fn.assert_called_once()

    def test_returns_zero_for_no_model(self):
        """Test missing model returns 0."""
        result = count_tokens(model="", text="Hello")
        assert result == 0

    def test_returns_zero_for_empty_input(self):
        """Test empty input returns 0."""
        result = count_tokens(model="gpt-4")
        assert result == 0

    def test_raises_for_both_text_and_messages(self):
        """Test providing both text and messages raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            count_tokens(
                model="gpt-4",
                text="Hello",
                messages=[{"role": "user", "content": "World"}]
            )

    @patch("agent_core.llm.token_counter._count_tokens_litellm")
    def test_uses_override_model_from_config(self, mock_litellm):
        """Test litellm_token_counter_model override is respected."""
        mock_litellm.return_value = 15
        config = {"litellm_token_counter_model": "gpt-3.5-turbo"}

        count_tokens(model="custom-model", text="Hello", llm_config=config)

        # Should use the override model
        call_args = mock_litellm.call_args
        assert call_args[0][0] == "gpt-3.5-turbo"

    def test_includes_system_prompt_in_messages(self):
        """Test system_prompt is included in message count."""
        mock_anthropic_fn = MagicMock(return_value=40)

        with patch.dict(
            "agent_core.llm.token_counter.PROVIDER_TOKEN_COUNTERS",
            {LLMProvider.ANTHROPIC: mock_anthropic_fn}
        ):
            count_tokens(
                model="claude-3-sonnet",
                text="Hello",
                system_prompt="You are helpful"
            )

        call_args = mock_anthropic_fn.call_args
        messages = call_args[1]["messages"]
        # System prompt should be in the messages
        assert any(m.get("role") == "system" for m in messages)


class TestCountTokensIntegration:
    """Integration tests that test the full flow (can be skipped in CI without API keys)."""

    @pytest.mark.skipif(
        True,  # Set to False to run integration tests locally
        reason="Integration tests require API keys"
    )
    def test_anthropic_api_call(self):
        """Test actual Anthropic API call (requires ANTHROPIC_API_KEY)."""
        result = count_tokens(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello, how are you?"}]
        )

        assert result > 0
        assert isinstance(result, int)
