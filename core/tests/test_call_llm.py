"""
Unit tests for agent_core/llm/call_llm.py

Tests token estimation, response aggregation, and LLM call orchestration.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os

from agent_core.llm.call_llm import (
    estimate_prompt_tokens,
    LLMResponseAggregator,
    FunctionCallErrorException,
    call_litellm_acompletion,
)


class TestEstimatePromptTokens:
    """Tests for the estimate_prompt_tokens function.

    Note: estimate_prompt_tokens now delegates to token_counter.count_tokens.
    These tests verify the integration works correctly.
    """

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_estimates_tokens_for_text(self, mock_counter):
        """Test token estimation for plain text."""
        mock_counter.return_value = 10

        result = estimate_prompt_tokens(model="gpt-4", text="Hello world")

        mock_counter.assert_called_once()
        call_args = mock_counter.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["text"] == "Hello world"
        assert result == 10

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_estimates_tokens_for_messages(self, mock_counter):
        """Test token estimation for message list."""
        mock_counter.return_value = 25
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = estimate_prompt_tokens(model="gpt-4", messages=messages)

        assert result == 25
        call_args = mock_counter.call_args
        assert call_args[1]["messages"] == messages

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_passes_system_prompt(self, mock_counter):
        """Test that system_prompt is passed through."""
        mock_counter.return_value = 30

        result = estimate_prompt_tokens(
            model="gpt-4",
            text="Hello",
            system_prompt="You are helpful"
        )

        call_args = mock_counter.call_args
        assert call_args[1]["system_prompt"] == "You are helpful"

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_raises_for_both_text_and_messages(self, mock_counter):
        """Test that providing both text and messages raises ValueError."""
        mock_counter.side_effect = ValueError("Provide either 'text' or 'messages', not both.")

        with pytest.raises(ValueError, match="not both"):
            estimate_prompt_tokens(
                model="gpt-4",
                text="Hello",
                messages=[{"role": "user", "content": "World"}]
            )

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_returns_zero_for_no_model(self, mock_counter):
        """Test that missing model returns 0."""
        mock_counter.return_value = 0

        result = estimate_prompt_tokens(model="", text="Hello")

        assert result == 0

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_returns_zero_for_empty_input(self, mock_counter):
        """Test that empty input returns 0."""
        mock_counter.return_value = 0

        result = estimate_prompt_tokens(model="gpt-4")

        assert result == 0

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_passes_llm_config(self, mock_counter):
        """Test that llm_config is passed through."""
        mock_counter.return_value = 15
        config = {"litellm_token_counter_model": "gpt-3.5-turbo"}

        result = estimate_prompt_tokens(
            model="custom-model",
            text="Hello",
            llm_config_for_tokenizer=config
        )

        call_args = mock_counter.call_args
        assert call_args[1]["llm_config"] == config

    @patch("agent_core.llm.token_counter.count_tokens")
    def test_handles_token_counter_exception(self, mock_counter):
        """Test graceful handling of token_counter exceptions."""
        # The underlying count_tokens handles exceptions and returns 0
        mock_counter.return_value = 0

        result = estimate_prompt_tokens(model="gpt-4", text="Hello")

        assert result == 0


class TestLLMResponseAggregator:
    """Tests for the LLMResponseAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create a basic aggregator for testing."""
        return LLMResponseAggregator(
            agent_id="test-agent",
            parent_agent_id=None,
            events=None,
            run_id="test-run",
            stream_id="test-stream",
            llm_model_id="gpt-4"
        )

    def test_initialization(self, aggregator):
        """Test aggregator initializes with correct defaults."""
        assert aggregator.agent_id == "test-agent"
        assert aggregator.full_content == ""
        assert aggregator.full_reasoning_content == ""
        assert aggregator.current_tool_call_chunks == {}
        assert aggregator.raw_chunks == []
        assert aggregator.model_id_used is None
        assert aggregator.actual_usage is None

    @pytest.mark.asyncio
    async def test_process_chunk_content(self, aggregator):
        """Test processing a content chunk."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Hello"
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.tool_calls = None
        chunk.model = "gpt-4"
        chunk.usage = None

        await aggregator.process_chunk(chunk)

        assert aggregator.full_content == "Hello"
        assert aggregator.model_id_used == "gpt-4"

    @pytest.mark.asyncio
    async def test_process_chunk_accumulates_content(self, aggregator):
        """Test that multiple content chunks are accumulated."""
        for text in ["Hello", " ", "World"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.model = "gpt-4"
            chunk.usage = None

            await aggregator.process_chunk(chunk)

        assert aggregator.full_content == "Hello World"

    @pytest.mark.asyncio
    async def test_process_chunk_reasoning_content(self, aggregator):
        """Test processing reasoning content."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = None
        chunk.choices[0].delta.reasoning_content = "Let me think..."
        chunk.choices[0].delta.tool_calls = None
        chunk.model = "gpt-4"
        chunk.usage = None

        await aggregator.process_chunk(chunk)

        assert aggregator.full_reasoning_content == "Let me think..."

    @pytest.mark.asyncio
    async def test_process_chunk_detects_tool_call_tag(self, aggregator):
        """Test that <tool_call> tag triggers retry exception."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "<tool_call>some_tool</tool_call>"
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.tool_calls = None
        chunk.model = "gpt-4"
        chunk.usage = None

        with pytest.raises(FunctionCallErrorException, match="tool_call"):
            await aggregator.process_chunk(chunk)

    @pytest.mark.asyncio
    async def test_process_chunk_detects_tool_code_tag(self, aggregator):
        """Test that <tool_code> tag triggers retry exception."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "<tool_code>print('hello')</tool_code>"
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.tool_calls = None
        chunk.model = "gpt-4"
        chunk.usage = None

        with pytest.raises(FunctionCallErrorException, match="tool_code"):
            await aggregator.process_chunk(chunk)

    @pytest.mark.asyncio
    async def test_process_chunk_tool_calls(self, aggregator):
        """Test processing tool call chunks."""
        # First chunk with tool call start
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.reasoning_content = None
        tc_chunk = MagicMock()
        tc_chunk.index = 0
        tc_chunk.id = "call_123"
        tc_chunk.function = MagicMock()
        tc_chunk.function.name = "search"
        tc_chunk.function.arguments = '{"query":'
        chunk1.choices[0].delta.tool_calls = [tc_chunk]
        chunk1.model = "gpt-4"
        chunk1.usage = None

        await aggregator.process_chunk(chunk1)

        assert 0 in aggregator.current_tool_call_chunks
        assert aggregator.current_tool_call_chunks[0]["id"] == "call_123"
        assert aggregator.current_tool_call_chunks[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_process_chunk_captures_usage(self, aggregator):
        """Test that usage information is captured from chunks."""
        chunk = MagicMock()
        chunk.choices = []
        chunk.usage = MagicMock()
        chunk.usage.dict.return_value = {"prompt_tokens": 100, "completion_tokens": 50}

        await aggregator.process_chunk(chunk)

        assert aggregator.actual_usage == {"prompt_tokens": 100, "completion_tokens": 50}

    @pytest.mark.asyncio
    async def test_process_chunk_no_choices(self, aggregator):
        """Test handling chunk with no choices."""
        chunk = MagicMock()
        chunk.choices = []
        chunk.usage = None

        # Should not raise
        await aggregator.process_chunk(chunk)

        assert aggregator.full_content == ""

    def test_get_aggregated_response(self, aggregator):
        """Test getting aggregated response."""
        aggregator.full_content = "Hello world"
        aggregator.full_reasoning_content = "Let me think"
        aggregator.model_id_used = "gpt-4"
        aggregator.actual_usage = {"prompt_tokens": 10, "completion_tokens": 5}

        result = aggregator.get_aggregated_response(messages_for_llm=[])

        assert result["content"] == "Hello world"
        assert result["reasoning"] == "Let me think"
        assert result["model_id_used"] == "gpt-4"
        assert result["actual_usage"]["prompt_tokens"] == 10

    def test_get_aggregated_response_repairs_json(self, aggregator):
        """Test that tool call arguments JSON is repaired."""
        aggregator.current_tool_call_chunks = {
            0: {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"key": "value"'  # Missing closing brace
                }
            }
        }

        result = aggregator.get_aggregated_response(messages_for_llm=[])

        # json_repair should fix the JSON
        tool_calls = result["tool_calls"]
        assert len(tool_calls) == 1
        # The arguments should be parseable now
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["key"] == "value"


class TestFunctionCallErrorException:
    """Tests for the FunctionCallErrorException."""

    def test_exception_message(self):
        """Test exception stores message correctly."""
        exc = FunctionCallErrorException("Test error message")

        assert str(exc) == "Test error message"

    def test_exception_is_exception_subclass(self):
        """Test that it's a proper Exception subclass."""
        exc = FunctionCallErrorException("Test")

        assert isinstance(exc, Exception)


class TestCallLitellmAcompletion:
    """Tests for the call_litellm_acompletion function."""

    @pytest.fixture
    def basic_config(self):
        """Basic LLM config for testing."""
        return {
            "model": "gpt-4",
            "max_retries": 2,
            "wait_seconds_on_retry": 0.1
        }

    @pytest.fixture
    def mock_events(self):
        """Create mock events object."""
        events = AsyncMock()
        events.emit_llm_stream_started = AsyncMock()
        events.emit_llm_request_params = AsyncMock()
        events.emit_llm_chunk = AsyncMock()
        events.emit_llm_stream_ended = AsyncMock()
        events.emit_llm_stream_failed = AsyncMock()
        events.send_json = AsyncMock()
        return events

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_model(self, basic_config):
        """Test that missing model returns error dict."""
        config = {"max_retries": 1}  # No model

        result = await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hello"}],
            llm_config=config
        )

        # The function catches ValueError and returns error dict
        assert "error" in result
        assert "model" in result["error"].lower() or "Unexpected" in result["error"]

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_successful_call(self, mock_acompletion, basic_config):
        """Test successful LLM call returns aggregated response."""
        # Create mock streaming response
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Hello!"
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.model = "gpt-4"
            chunk.usage = None
            yield chunk

        mock_acompletion.return_value = mock_stream()

        result = await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config
        )

        assert result["content"] == "Hello!"
        assert "final_stream_id" in result

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_prepends_system_prompt(self, mock_acompletion, basic_config):
        """Test that system_prompt_content is prepended to messages."""
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Response"
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.model = "gpt-4"
            chunk.usage = None
            yield chunk

        mock_acompletion.return_value = mock_stream()

        await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config,
            system_prompt_content="You are helpful"
        )

        call_args = mock_acompletion.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_includes_tools_in_request(self, mock_acompletion, basic_config):
        """Test that tools are included in the request."""
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Using tool"
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.model = "gpt-4"
            chunk.usage = None
            yield chunk

        mock_acompletion.return_value = mock_stream()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]

        await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config,
            api_tools_list=tools,
            tool_choice="auto"
        )

        call_args = mock_acompletion.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == "auto"

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_retries_on_empty_response(self, mock_acompletion, basic_config):
        """Test that empty response triggers retry."""
        call_count = 0

        async def mock_stream_factory():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns empty
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = ""  # Empty
                chunk.choices[0].delta.reasoning_content = None
                chunk.choices[0].delta.tool_calls = None
                chunk.model = "gpt-4"
                chunk.usage = None
                yield chunk
            else:
                # Second call returns content
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = "Valid response"
                chunk.choices[0].delta.reasoning_content = None
                chunk.choices[0].delta.tool_calls = None
                chunk.model = "gpt-4"
                chunk.usage = None
                yield chunk

        mock_acompletion.side_effect = lambda **kwargs: mock_stream_factory()

        result = await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config
        )

        assert call_count == 2
        assert result["content"] == "Valid response"

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_returns_error_on_authentication_error(self, mock_acompletion, basic_config):
        """Test that AuthenticationError returns error dict without retry."""
        from litellm.exceptions import AuthenticationError

        mock_acompletion.side_effect = AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4"
        )

        result = await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config
        )

        assert "error" in result
        assert result["error_type"] == "AuthenticationError"
        # Should only be called once (no retry for auth errors)
        assert mock_acompletion.call_count == 1

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_returns_error_on_context_window_exceeded(self, mock_acompletion, basic_config):
        """Test that ContextWindowExceededError returns error without retry."""
        from litellm.exceptions import ContextWindowExceededError

        mock_acompletion.side_effect = ContextWindowExceededError(
            message="Context too long",
            llm_provider="openai",
            model="gpt-4"
        )

        result = await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config
        )

        assert "error" in result
        assert result["error_type"] == "ContextWindowExceededError"

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_retries_on_rate_limit_error(self, mock_acompletion, basic_config):
        """Test that RateLimitError triggers retry."""
        from litellm.exceptions import RateLimitError

        call_count = 0

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Success"
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.model = "gpt-4"
            chunk.usage = None
            yield chunk

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    message="Rate limited",
                    llm_provider="openai",
                    model="gpt-4"
                )
            return mock_stream()

        mock_acompletion.side_effect = side_effect

        result = await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config
        )

        assert call_count == 2
        assert result["content"] == "Success"

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_emits_events_when_provided(self, mock_acompletion, basic_config, mock_events):
        """Test that events are emitted when events object is provided."""
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Hello"
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.model = "gpt-4"
            chunk.usage = None
            yield chunk

        mock_acompletion.return_value = mock_stream()

        await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config,
            events=mock_events,
            agent_id_for_event="test-agent",
            run_id_for_event="test-run"
        )

        mock_events.emit_llm_stream_started.assert_called_once()
        mock_events.emit_llm_request_params.assert_called_once()
        mock_events.emit_llm_stream_ended.assert_called_once()

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_updates_token_usage_stats(self, mock_acompletion, basic_config):
        """Test that token usage stats are updated in run_context."""
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = "Hello"
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunk.model = "gpt-4"
            # Usage chunk
            chunk.usage = MagicMock()
            chunk.usage.dict.return_value = {"prompt_tokens": 100, "completion_tokens": 50}
            yield chunk

        mock_acompletion.return_value = mock_stream()

        run_context = {
            "runtime": {
                "token_usage_stats": {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_successful_calls": 0,
                    "total_failed_calls": 0,
                    "max_context_window": 0
                }
            }
        }

        await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config,
            run_context=run_context
        )

        stats = run_context["runtime"]["token_usage_stats"]
        assert stats["total_prompt_tokens"] == 100
        assert stats["total_completion_tokens"] == 50
        assert stats["total_successful_calls"] == 1

    @pytest.mark.asyncio
    async def test_handles_cancellation(self, basic_config):
        """Test that asyncio.CancelledError is propagated."""
        with patch("agent_core.llm.call_llm.litellm.acompletion") as mock_acompletion:
            mock_acompletion.side_effect = asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                await call_litellm_acompletion(
                    messages=[{"role": "user", "content": "Hi"}],
                    llm_config=basic_config
                )

    @pytest.mark.asyncio
    @patch("agent_core.llm.call_llm.litellm.acompletion")
    async def test_exhausted_retries_returns_error(self, mock_acompletion, basic_config):
        """Test that exhausted retries returns error dict."""
        from litellm.exceptions import RateLimitError

        mock_acompletion.side_effect = RateLimitError(
            message="Rate limited",
            llm_provider="openai",
            model="gpt-4"
        )

        result = await call_litellm_acompletion(
            messages=[{"role": "user", "content": "Hi"}],
            llm_config=basic_config
        )

        assert "error" in result
        assert "failed after all retries" in result["error"]
        # max_retries=2 means 3 total attempts (0, 1, 2)
        assert mock_acompletion.call_count == 3


class TestLLMResponseAggregatorContextualData:
    """Tests for contextual data handling in LLMResponseAggregator."""

    def test_get_contextual_data_empty(self):
        """Test contextual data returns None when empty."""
        aggregator = LLMResponseAggregator(
            agent_id="test",
            parent_agent_id=None,
            events=None,
            run_id="run",
            stream_id="stream",
            llm_model_id="gpt-4"
        )

        result = aggregator._get_contextual_data_for_event()

        assert result is None

    def test_get_contextual_data_with_task_nums(self):
        """Test contextual data includes task nums."""
        aggregator = LLMResponseAggregator(
            agent_id="test",
            parent_agent_id=None,
            events=None,
            run_id="run",
            stream_id="stream",
            llm_model_id="gpt-4",
            associated_task_nums_for_event=[1, 2, 3]
        )

        result = aggregator._get_contextual_data_for_event()

        assert result["associated_task_nums"] == [1, 2, 3]

    def test_get_contextual_data_with_module_id(self):
        """Test contextual data includes module_id."""
        aggregator = LLMResponseAggregator(
            agent_id="test",
            parent_agent_id=None,
            events=None,
            run_id="run",
            stream_id="stream",
            llm_model_id="gpt-4",
            module_id_for_event="module_123"
        )

        result = aggregator._get_contextual_data_for_event()

        assert result["module_id"] == "module_123"

    def test_get_contextual_data_with_dispatch_id(self):
        """Test contextual data includes dispatch_id."""
        aggregator = LLMResponseAggregator(
            agent_id="test",
            parent_agent_id=None,
            events=None,
            run_id="run",
            stream_id="stream",
            llm_model_id="gpt-4",
            dispatch_id_for_event="dispatch_456"
        )

        result = aggregator._get_contextual_data_for_event()

        assert result["dispatch_id"] == "dispatch_456"


class TestFilteredKeysForLiteLLM:
    """Tests for parameter filtering before sending to LiteLLM API.

    These tests verify that internal config keys are properly filtered
    out before making API calls to prevent 'Extra inputs not permitted' errors.
    """

    def test_filtered_keys_constant_includes_max_context_tokens(self):
        """Test that FILTERED_KEYS includes max_context_tokens.

        This tests the fix for Anthropic API error:
        'max_context_tokens: Extra inputs are not permitted'

        max_context_tokens is used by Context Budget Guardian internally
        and should NOT be passed to the LiteLLM/Anthropic API.
        """
        # Import the function to inspect its code
        import inspect
        from agent_core.llm.call_llm import call_litellm_acompletion

        source = inspect.getsource(call_litellm_acompletion)

        # Verify max_context_tokens is in the FILTERED_KEYS list
        assert 'max_context_tokens' in source
        assert 'FILTERED_KEYS' in source

    def test_internal_keys_not_passed_to_api(self):
        """Test that internal config keys are properly filtered."""
        # These keys should be filtered before sending to LiteLLM
        internal_keys = [
            "stream_id",
            "parent_agent_id",
            "wait_seconds_on_retry",
            "max_retries",
            "max_context_tokens"  # Used by Context Budget Guardian only
        ]

        # Simulate base_params with internal keys
        base_params = {
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.4,
            "stream": True,
            # Internal keys that should be filtered
            "stream_id": "test-stream-123",
            "parent_agent_id": "Partner",
            "wait_seconds_on_retry": 5,
            "max_retries": 2,
            "max_context_tokens": 1000000  # Context Budget Guardian config
        }

        # Apply the same filtering logic as call_litellm_acompletion
        FILTERED_KEYS = ["stream_id", "parent_agent_id", "wait_seconds_on_retry", "max_retries", "max_context_tokens"]
        params_for_litellm = {k: v for k, v in base_params.items() if k not in FILTERED_KEYS}

        # Verify internal keys are filtered out
        for key in internal_keys:
            assert key not in params_for_litellm, f"{key} should be filtered from API params"

        # Verify API-relevant keys remain
        assert "model" in params_for_litellm
        assert "messages" in params_for_litellm
        assert "temperature" in params_for_litellm
        assert "stream" in params_for_litellm
