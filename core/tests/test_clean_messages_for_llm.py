"""
Unit tests for BaseAgentNode._clean_messages_for_llm method.

This module tests the message cleaning/sanitization that occurs before
sending messages to the LLM API. Critical functionality includes:
- Removing internal fields (those starting with _)
- Ensuring content is always a string
- Sanitizing tool_calls to ensure arguments are valid JSON dictionaries
  (required by Anthropic's API: tool_use.input must be a dictionary)

The tool_call sanitization was added to fix a bug where malformed tool
arguments from the LLM (e.g., "" instead of {}) would cause the next
API call to fail with "tool_use.input: Input should be a valid dictionary".
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from agent_core.nodes.base_agent_node import AgentNode


class MockAgentNode(AgentNode):
    """A mock implementation of AgentNode for testing _clean_messages_for_llm."""

    def __init__(self):
        # Skip the full __init__ - we only need the method under test
        pass


@pytest.fixture
def agent_node():
    """Create a mock agent node for testing."""
    return MockAgentNode()


class TestCleanMessagesBasic:
    """Tests for basic message cleaning functionality."""

    def test_preserves_standard_fields(self, agent_node):
        """Test that standard LLM fields are preserved."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = agent_node._clean_messages_for_llm(messages)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there!"}

    def test_removes_internal_fields(self, agent_node):
        """Test that internal fields (starting with _) are removed."""
        messages = [
            {
                "role": "user",
                "content": "Hello",
                "_internal_id": "abc123",
                "_no_handover": True,
                "_timestamp": "2025-01-01",
            }
        ]
        result = agent_node._clean_messages_for_llm(messages)

        assert len(result) == 1
        assert "_internal_id" not in result[0]
        assert "_no_handover" not in result[0]
        assert "_timestamp" not in result[0]
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_converts_none_content_to_empty_string(self, agent_node):
        """Test that None content is converted to empty string."""
        messages = [{"role": "assistant", "content": None}]
        result = agent_node._clean_messages_for_llm(messages)

        assert result[0]["content"] == ""

    def test_converts_dict_content_to_json_string(self, agent_node):
        """Test that dict content is converted to JSON string."""
        messages = [{"role": "user", "content": {"key": "value", "num": 42}}]
        result = agent_node._clean_messages_for_llm(messages)

        assert isinstance(result[0]["content"], str)
        parsed = json.loads(result[0]["content"])
        assert parsed == {"key": "value", "num": 42}

    def test_preserves_tool_call_id_and_name(self, agent_node):
        """Test that tool_call_id and name fields are preserved."""
        messages = [
            {"role": "tool", "tool_call_id": "call-123", "name": "search", "content": "results"}
        ]
        result = agent_node._clean_messages_for_llm(messages)

        assert result[0]["tool_call_id"] == "call-123"
        assert result[0]["name"] == "search"
        assert result[0]["content"] == "results"


class TestToolCallsSanitization:
    """Tests for tool_calls sanitization - the critical fix for Anthropic API compatibility."""

    def test_valid_tool_calls_pass_through(self, agent_node):
        """Test that valid tool_calls with proper JSON arguments pass through."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "python docs"}'
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call-1"
        assert tc["function"]["name"] == "search"
        # Arguments should be valid JSON that parses to a dict
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {"query": "python docs"}

    def test_empty_string_arguments_sanitized_to_empty_dict(self, agent_node):
        """Test that empty string arguments are sanitized to empty dict.

        This is the primary bug fix case - LLM returns "" instead of "{}"
        which causes Anthropic to reject the next API call.
        """
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "dispatch_submodules",
                        "arguments": '""'  # Malformed - should be "{}"
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}  # Should be sanitized to empty dict

    def test_non_dict_arguments_sanitized_to_empty_dict(self, agent_node):
        """Test that non-dict JSON arguments are sanitized."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "some_tool",
                        "arguments": '"just a string"'  # Valid JSON, but not a dict
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}

    def test_invalid_json_arguments_sanitized_to_empty_dict(self, agent_node):
        """Test that invalid JSON arguments are sanitized."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "broken_tool",
                        "arguments": '{invalid json}'  # Not valid JSON
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}

    def test_null_arguments_sanitized_to_empty_dict(self, agent_node):
        """Test that null JSON arguments are sanitized."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "null_tool",
                        "arguments": 'null'  # Valid JSON null
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}

    def test_array_arguments_sanitized_to_empty_dict(self, agent_node):
        """Test that array JSON arguments are sanitized."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "array_tool",
                        "arguments": '["item1", "item2"]'  # Valid JSON array, not dict
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}

    def test_number_arguments_sanitized_to_empty_dict(self, agent_node):
        """Test that number JSON arguments are sanitized."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "number_tool",
                        "arguments": '42'  # Valid JSON number, not dict
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}

    def test_multiple_tool_calls_all_sanitized(self, agent_node):
        """Test that all tool_calls in a message are sanitized."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "valid_tool",
                        "arguments": '{"key": "value"}'  # Valid
                    }
                },
                {
                    "id": "call-2",
                    "function": {
                        "name": "invalid_tool",
                        "arguments": '""'  # Invalid - empty string
                    }
                },
                {
                    "id": "call-3",
                    "function": {
                        "name": "broken_tool",
                        "arguments": 'not json at all'  # Invalid JSON
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tool_calls = result[0]["tool_calls"]
        assert len(tool_calls) == 3

        # First should be preserved
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"key": "value"}

        # Second and third should be sanitized
        assert json.loads(tool_calls[1]["function"]["arguments"]) == {}
        assert json.loads(tool_calls[2]["function"]["arguments"]) == {}

    def test_tool_call_other_fields_preserved(self, agent_node):
        """Test that other tool_call fields are preserved during sanitization."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-unique-123",
                    "type": "function",
                    "function": {
                        "name": "my_special_tool",
                        "arguments": '""'  # Will be sanitized
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call-unique-123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "my_special_tool"

    def test_missing_arguments_defaults_to_empty_dict(self, agent_node):
        """Test that missing arguments field defaults to empty dict."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "no_args_tool"
                        # No "arguments" key at all
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}

    def test_empty_dict_string_arguments_preserved(self, agent_node):
        """Test that '{}' arguments are preserved correctly."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "empty_args_tool",
                        "arguments": '{}'  # Valid empty dict
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {}


class TestToolCallsLogging:
    """Tests for logging behavior during tool_calls sanitization."""

    def test_logs_warning_for_non_dict_arguments(self, agent_node):
        """Test that a warning is logged when arguments are not a dict."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "test_tool",
                        "arguments": '"string_value"'
                    }
                }
            ]
        }]

        with patch('agent_core.nodes.base_agent_node.logger') as mock_logger:
            agent_node._clean_messages_for_llm(messages)

            # Should have logged a warning about sanitization
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "tool_call_arguments_sanitized"
            assert call_args[1]["extra"]["tool_name"] == "test_tool"
            assert call_args[1]["extra"]["reason"] == "not_a_dict"

    def test_logs_warning_for_json_decode_error(self, agent_node):
        """Test that a warning is logged for JSON decode errors."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "broken_tool",
                        "arguments": 'not valid json'
                    }
                }
            ]
        }]

        with patch('agent_core.nodes.base_agent_node.logger') as mock_logger:
            agent_node._clean_messages_for_llm(messages)

            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert call_args[0][0] == "tool_call_arguments_sanitized"
            assert call_args[1]["extra"]["tool_name"] == "broken_tool"
            assert call_args[1]["extra"]["reason"] == "json_decode_error"

    def test_no_warning_for_valid_arguments(self, agent_node):
        """Test that no warning is logged for valid dict arguments."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "valid_tool",
                        "arguments": '{"valid": "args"}'
                    }
                }
            ]
        }]

        with patch('agent_core.nodes.base_agent_node.logger') as mock_logger:
            agent_node._clean_messages_for_llm(messages)

            # Should NOT have called warning
            mock_logger.warning.assert_not_called()


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_messages_list(self, agent_node):
        """Test handling of empty messages list."""
        result = agent_node._clean_messages_for_llm([])
        assert result == []

    def test_message_without_tool_calls(self, agent_node):
        """Test that messages without tool_calls are handled correctly."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = agent_node._clean_messages_for_llm(messages)

        assert "tool_calls" not in result[0]
        assert "tool_calls" not in result[1]

    def test_tool_call_without_function_key(self, agent_node):
        """Test handling of malformed tool_call without function key."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1"
                    # No "function" key
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        # Should not crash, tool_call should pass through
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call-1"

    def test_does_not_mutate_original_messages(self, agent_node):
        """Test that original messages are not mutated."""
        original_args = '""'
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "test_tool",
                        "arguments": original_args
                    }
                }
            ]
        }]

        # Store original reference
        original_func = messages[0]["tool_calls"][0]["function"]

        result = agent_node._clean_messages_for_llm(messages)

        # Original should be unchanged
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == original_args
        # Result should be sanitized
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {}

    def test_unicode_in_arguments(self, agent_node):
        """Test that unicode characters in arguments are preserved."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "unicode_tool",
                        "arguments": '{"text": "こんにちは世界", "emoji": "🎉"}'
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed["text"] == "こんにちは世界"
        assert parsed["emoji"] == "🎉"

    def test_nested_dict_in_arguments(self, agent_node):
        """Test that nested dicts in arguments are preserved."""
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "nested_tool",
                        "arguments": '{"outer": {"inner": {"deep": "value"}}}'
                    }
                }
            ]
        }]
        result = agent_node._clean_messages_for_llm(messages)

        tc = result[0]["tool_calls"][0]
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed["outer"]["inner"]["deep"] == "value"
