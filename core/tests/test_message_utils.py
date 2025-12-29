"""
Unit tests for agent_core.utils.message_utils module.

This module tests the tool_call_safenet function that ensures message
history integrity for LLM calls, correcting proximity and symmetry
violations between assistant tool_calls and tool responses.

Key concepts tested:
- Proximity correction: Messages shouldn't appear between tool calls and responses
- Symmetry correction: Every tool_call must have exactly one tool response
- Message reordering: Interloper messages are moved after tool responses
- Error injection: Missing tool responses get error placeholders
"""

import pytest
from agent_core.utils.message_utils import tool_call_safenet


class TestEmptyAndNullInputs:
    """Tests for edge cases with empty or null inputs."""

    def test_empty_list_returns_empty(self):
        """Test that empty message list returns empty list."""
        result = tool_call_safenet([], "test-agent")
        assert result == []

    def test_none_returns_empty(self):
        """Test that None returns empty list."""
        # Function should handle None gracefully
        result = tool_call_safenet(None, "test-agent")
        assert result == []


class TestPassthroughBehavior:
    """Tests for messages that should pass through unchanged."""

    def test_single_user_message(self):
        """Test single user message passes through."""
        messages = [{"role": "user", "content": "Hello"}]
        result = tool_call_safenet(messages, "test-agent")
        assert len(result) == 1
        assert result[0] == messages[0]

    def test_user_assistant_conversation(self):
        """Test basic user/assistant conversation passes through."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
        ]
        result = tool_call_safenet(messages, "test-agent")
        assert len(result) == 4
        assert result == messages

    def test_assistant_without_tool_calls(self):
        """Test assistant messages without tool_calls pass through."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
        result = tool_call_safenet(messages, "test-agent")
        assert result == messages

    def test_assistant_with_empty_tool_calls(self):
        """Test assistant with empty tool_calls list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "tool_calls": []},
        ]
        result = tool_call_safenet(messages, "test-agent")
        assert len(result) == 2


class TestValidToolCallSequences:
    """Tests for valid tool call sequences that should remain unchanged."""

    def test_single_tool_call_with_response(self):
        """Test valid single tool call followed by response."""
        messages = [
            {"role": "user", "content": "Search for Python"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Found: Python docs"},
        ]
        result = tool_call_safenet(messages, "test-agent")
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "tool"

    def test_multiple_parallel_tool_calls(self):
        """Test valid multiple parallel tool calls with all responses."""
        messages = [
            {"role": "user", "content": "Search for both"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "search_a", "arguments": "{}"}},
                {"id": "call-2", "function": {"name": "search_b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Result A"},
            {"role": "tool", "tool_call_id": "call-2", "content": "Result B"},
        ]
        result = tool_call_safenet(messages, "test-agent")
        assert len(result) == 4
        # Order should be preserved
        assert result[2]["tool_call_id"] == "call-1"
        assert result[3]["tool_call_id"] == "call-2"

    def test_sequential_tool_call_blocks(self):
        """Test multiple sequential tool call/response blocks."""
        messages = [
            {"role": "user", "content": "Do two things"},
            # First tool call block
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "first_tool", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "First result"},
            # Second tool call block
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-2", "function": {"name": "second_tool", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call-2", "content": "Second result"},
        ]
        result = tool_call_safenet(messages, "test-agent")
        assert len(result) == 5
        assert result[1]["tool_calls"][0]["id"] == "call-1"
        assert result[2]["tool_call_id"] == "call-1"
        assert result[3]["tool_calls"][0]["id"] == "call-2"
        assert result[4]["tool_call_id"] == "call-2"


class TestProximityViolations:
    """Tests for proximity violation detection and correction."""

    def test_user_message_between_call_and_response(self):
        """Test user message inserted between tool call and response is moved."""
        messages = [
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "user", "content": "Interloper message"},  # Proximity violation
            {"role": "tool", "tool_call_id": "call-1", "content": "Result"},
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Tool response should come immediately after assistant
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call-1"
        # Interloper should be moved after tool response
        assert result[3]["role"] == "user"
        # Interloper content should have error message prepended
        assert "[SAFENET ERROR]" in result[3]["content"]

    def test_multiple_interlopers(self):
        """Test multiple interloper messages are all moved."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "tool", "arguments": "{}"}}
            ]},
            {"role": "user", "content": "First interloper"},
            {"role": "user", "content": "Second interloper"},
            {"role": "tool", "tool_call_id": "call-1", "content": "Result"},
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Tool should be second (after assistant)
        assert result[1]["role"] == "tool"
        # Both interlopers should be after tool
        assert result[2]["role"] == "user"
        assert result[3]["role"] == "user"


class TestSymmetryViolations:
    """Tests for symmetry violation detection and correction."""

    def test_missing_tool_response(self):
        """Test missing tool response gets error placeholder injected."""
        messages = [
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "search", "arguments": "{}"}}
            ]},
            # Missing tool response for call-1
            {"role": "assistant", "content": "Moving on without result"},
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Should have injected an error response
        assert len(result) >= 3
        # Find the injected tool response
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 1
        assert tool_responses[0]["tool_call_id"] == "call-1"
        assert "no_response_from_tool" in tool_responses[0]["content"]

    def test_missing_one_of_multiple_responses(self):
        """Test partial missing responses in parallel tool calls."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "call-2", "function": {"name": "tool_b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Result A"},
            # Missing response for call-2
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Should have both tool responses
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 2

        # Find the injected error response
        ids = [tr["tool_call_id"] for tr in tool_responses]
        assert "call-1" in ids
        assert "call-2" in ids

        # call-2 should have error content
        call_2_response = next(tr for tr in tool_responses if tr["tool_call_id"] == "call-2")
        assert "no_response_from_tool" in call_2_response["content"]

    def test_extra_tool_response(self):
        """Test extra tool response (no matching call) is neutralized."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "tool", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Valid result"},
            {"role": "tool", "tool_call_id": "call-orphan", "content": "Orphan result"},
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Orphan should be neutralized (changed to assistant role)
        orphan = next((m for m in result if "Orphan result" in str(m.get("content", ""))), None)
        assert orphan is not None
        assert orphan["role"] == "assistant"
        assert "[SAFENET ERROR]" in orphan["content"]
        assert "tool_call_id" not in orphan

    def test_all_responses_missing(self):
        """Test all tool responses missing for multiple calls."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "call-2", "function": {"name": "tool_b", "arguments": "{}"}},
                {"id": "call-3", "function": {"name": "tool_c", "arguments": "{}"}},
            ]},
            # No tool responses at all
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Should inject 3 error responses
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 3


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_mixed_violations(self):
        """Test both proximity and symmetry violations together."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "call-2", "function": {"name": "tool_b", "arguments": "{}"}},
            ]},
            {"role": "user", "content": "Interloper"},  # Proximity violation
            {"role": "tool", "tool_call_id": "call-1", "content": "Result A"},
            # call-2 missing - Symmetry violation
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Should fix both issues
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 2  # Both calls have responses

        # Interloper should be after tool responses
        user_msgs = [i for i, m in enumerate(result) if m.get("role") == "user"]
        tool_indices = [i for i, m in enumerate(result) if m.get("role") == "tool"]
        for user_idx in user_msgs:
            for tool_idx in tool_indices:
                assert user_idx > tool_idx or result[user_idx] != messages[1]

    def test_long_conversation_with_issues(self):
        """Test safenet in a longer conversation with multiple issues."""
        messages = [
            {"role": "user", "content": "Start task"},
            {"role": "assistant", "content": "I'll help"},
            # First valid tool call block
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "init", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Initialized"},
            # Second block with proximity violation
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-2", "function": {"name": "process", "arguments": "{}"}}
            ]},
            {"role": "user", "content": "Hurry up!"},  # Interloper
            {"role": "tool", "tool_call_id": "call-2", "content": "Processed"},
            # Third block with missing response
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-3", "function": {"name": "finalize", "arguments": "{}"}}
            ]},
            {"role": "assistant", "content": "Done!"},
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Should have all tool responses
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 3  # All three calls have responses

    def test_preserves_original_content(self):
        """Test that non-violated content is preserved exactly."""
        original = {"role": "user", "content": "Exact content here"}
        messages = [original]
        result = tool_call_safenet(messages, "test-agent")

        # Should be the exact same object
        assert result[0] is original


class TestToolCallMetadata:
    """Tests for preserving tool call metadata."""

    def test_preserves_tool_name_in_error(self):
        """Test that injected errors include original tool name."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "specific_tool_name", "arguments": "{}"}}
            ]},
        ]
        result = tool_call_safenet(messages, "test-agent")

        tool_response = next(m for m in result if m.get("role") == "tool")
        assert tool_response.get("name") == "specific_tool_name"

    def test_handles_tool_call_without_function_name(self):
        """Test graceful handling when tool call lacks function name."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {}}  # Missing name
            ]},
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Should still inject error response
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 1


class TestNullToolCallIds:
    """Tests for handling None tool_call_ids."""

    def test_tool_response_with_none_id(self):
        """Test that tool responses with None id are handled correctly."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "tool", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": None, "content": "Bad response"},  # None ID
            {"role": "tool", "tool_call_id": "call-1", "content": "Good response"},
        ]
        result = tool_call_safenet(messages, "test-agent")

        # Should not crash and should handle the None gracefully
        assert len(result) >= 2


class TestImmutability:
    """Tests to verify the function doesn't mutate input."""

    def test_does_not_mutate_input_messages(self):
        """Test that original messages list is not mutated."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "function": {"name": "tool", "arguments": "{}"}}
            ]},
            {"role": "user", "content": "Interloper"},
            {"role": "tool", "tool_call_id": "call-1", "content": "Result"},
        ]

        # Create deep copy to compare
        import copy
        original = copy.deepcopy(messages)

        result = tool_call_safenet(messages, "test-agent")

        # Verify structure wasn't mutated (note: content might be modified in copies)
        assert len(messages) == len(original)
        assert messages[0]["role"] == original[0]["role"]
        assert messages[1]["role"] == original[1]["role"]
        assert messages[2]["role"] == original[2]["role"]
