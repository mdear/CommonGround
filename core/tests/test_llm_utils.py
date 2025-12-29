"""
Unit tests for agent_core.llm.utils module.

This module tests utility functions for parsing LLM responses,
particularly the fallback tool call extraction from text content.

Key functions tested:
- _parse_arguments_string: Safely parses Python-style arguments to dict
- extract_tool_calls_from_content: Extracts tool calls from <tool_code> blocks
"""

import pytest
import json
from agent_core.llm.utils import (
    _parse_arguments_string,
    extract_tool_calls_from_content,
)


class TestParseArgumentsString:
    """Tests for the _parse_arguments_string helper function."""

    def test_empty_string(self):
        """Test parsing empty string returns empty dict."""
        result = _parse_arguments_string("", "test-agent")
        assert result == {}

    def test_whitespace_only(self):
        """Test parsing whitespace-only string returns empty dict."""
        result = _parse_arguments_string("   \n\t  ", "test-agent")
        assert result == {}

    def test_single_keyword_string_arg(self):
        """Test parsing single keyword argument with string value."""
        result = _parse_arguments_string('query="hello world"', "test-agent")
        assert result == {"query": "hello world"}

    def test_single_keyword_int_arg(self):
        """Test parsing single keyword argument with integer value."""
        result = _parse_arguments_string("limit=10", "test-agent")
        assert result == {"limit": 10}

    def test_single_keyword_float_arg(self):
        """Test parsing single keyword argument with float value."""
        result = _parse_arguments_string("temperature=0.7", "test-agent")
        assert result == {"temperature": 0.7}

    def test_single_keyword_bool_arg(self):
        """Test parsing single keyword argument with boolean value."""
        result = _parse_arguments_string("verbose=True", "test-agent")
        assert result == {"verbose": True}

        result = _parse_arguments_string("debug=False", "test-agent")
        assert result == {"debug": False}

    def test_multiple_keyword_args(self):
        """Test parsing multiple keyword arguments."""
        result = _parse_arguments_string('query="test", limit=5, debug=True', "test-agent")
        assert result == {"query": "test", "limit": 5, "debug": True}

    def test_list_argument(self):
        """Test parsing list as argument value."""
        result = _parse_arguments_string('items=["a", "b", "c"]', "test-agent")
        assert result == {"items": ["a", "b", "c"]}

    def test_dict_argument(self):
        """Test parsing dict as argument value."""
        result = _parse_arguments_string('config={"key": "value", "count": 1}', "test-agent")
        assert result == {"config": {"key": "value", "count": 1}}

    def test_nested_structures(self):
        """Test parsing nested list/dict structures."""
        result = _parse_arguments_string(
            'data={"nested": [1, 2, {"deep": True}]}',
            "test-agent"
        )
        assert result == {"data": {"nested": [1, 2, {"deep": True}]}}

    def test_none_value(self):
        """Test parsing None as argument value."""
        result = _parse_arguments_string("value=None", "test-agent")
        assert result == {"value": None}

    def test_string_with_special_chars(self):
        """Test parsing strings with special characters."""
        result = _parse_arguments_string('text="Hello, world! How\'s it going?"', "test-agent")
        assert result == {"text": "Hello, world! How's it going?"}

    def test_multiline_string(self):
        """Test parsing multiline strings."""
        # Triple-quoted strings work in Python literals
        result = _parse_arguments_string('text="""line1\nline2\nline3"""', "test-agent")
        assert result == {"text": "line1\nline2\nline3"}

    def test_positional_args(self):
        """Test parsing positional arguments (uncommon but supported)."""
        result = _parse_arguments_string('"positional1", 42', "test-agent")
        assert result == {"arg0": "positional1", "arg1": 42}

    def test_mixed_positional_and_keyword(self):
        """Test parsing mixed positional and keyword arguments."""
        result = _parse_arguments_string('"first", key="second"', "test-agent")
        assert result == {"arg0": "first", "key": "second"}

    def test_syntax_error_returns_empty(self):
        """Test that syntax errors return empty dict."""
        result = _parse_arguments_string("invalid syntax here !!!", "test-agent")
        assert result == {}

    def test_unbalanced_quotes_returns_empty(self):
        """Test that unbalanced quotes return empty dict."""
        result = _parse_arguments_string('query="unclosed', "test-agent")
        assert result == {}

    def test_complex_json_like_string(self):
        """Test parsing complex JSON-like string argument."""
        json_str = '{"users": [{"name": "Alice"}, {"name": "Bob"}], "count": 2}'
        result = _parse_arguments_string(f'data={json_str}', "test-agent")
        assert result == {"data": {"users": [{"name": "Alice"}, {"name": "Bob"}], "count": 2}}


class TestExtractToolCallsFromContent:
    """Tests for the extract_tool_calls_from_content function."""

    def test_no_tool_calls(self):
        """Test content with no tool calls returns empty list."""
        content = "This is just regular text without any tool calls."
        result = extract_tool_calls_from_content(content, "test-agent")
        assert result == []

    def test_single_simple_tool_call(self):
        """Test extracting a single simple tool call."""
        content = '<tool_code>print(MyTool(query="hello"))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "MyTool"

        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "hello"}

    def test_tool_call_with_multiple_args(self):
        """Test extracting tool call with multiple arguments."""
        content = '<tool_code>print(SearchTool(query="test", limit=10, verbose=True))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "test", "limit": 10, "verbose": True}

    def test_multiple_tool_calls(self):
        """Test extracting multiple tool calls from content."""
        content = '''
        Let me search for that.
        <tool_code>print(Search(query="AI"))</tool_code>
        And also calculate.
        <tool_code>print(Calculate(expression="2+2"))</tool_code>
        '''
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 2
        assert result[0]["function"]["name"] == "Search"
        assert result[1]["function"]["name"] == "Calculate"

    def test_tool_call_with_no_args(self):
        """Test extracting tool call with no arguments."""
        content = '<tool_code>print(GetStatus())</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        assert result[0]["function"]["name"] == "GetStatus"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {}

    def test_tool_call_with_complex_args(self):
        """Test extracting tool call with complex nested arguments."""
        content = '<tool_code>print(ProcessData(config={"nested": [1, 2, 3]}))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"config": {"nested": [1, 2, 3]}}

    def test_tool_call_id_format(self):
        """Test that tool_call_id follows expected format."""
        content = '<tool_code>print(TestTool())</tool_code>'
        result = extract_tool_calls_from_content(content, "my-agent")

        assert len(result) == 1
        # ID should start with "fallback_" followed by agent_id
        assert result[0]["id"].startswith("fallback_my-agent_")

    def test_whitespace_in_tool_code_tags(self):
        """Test that whitespace inside tool_code tags is handled."""
        content = '<tool_code>  print(MyTool(arg="value"))  </tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        assert result[0]["function"]["name"] == "MyTool"

    def test_newlines_in_tool_code_tags(self):
        """Test that newlines inside tool_code tags are handled."""
        content = '''<tool_code>
        print(MyTool(
            arg1="value1",
            arg2="value2"
        ))
        </tool_code>'''
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        assert result[0]["function"]["name"] == "MyTool"
        args = json.loads(result[0]["function"]["arguments"])
        assert args["arg1"] == "value1"
        assert args["arg2"] == "value2"

    def test_invalid_tool_call_format_skipped(self):
        """Test that invalid tool call formats are skipped."""
        content = '<tool_code>print()</tool_code>'  # Empty print
        result = extract_tool_calls_from_content(content, "test-agent")
        # Should not extract anything or handle gracefully
        assert isinstance(result, list)

    def test_malformed_tool_code_tag(self):
        """Test that malformed tags don't cause errors."""
        content = '<tool_code>print(Incomplete'  # Unclosed tag
        result = extract_tool_calls_from_content(content, "test-agent")
        assert result == []

    def test_tool_call_with_underscore_name(self):
        """Test tool names with underscores."""
        content = '<tool_code>print(my_tool_name(arg="test"))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        assert result[0]["function"]["name"] == "my_tool_name"

    def test_tool_call_mixed_with_text(self):
        """Test tool calls embedded in regular text."""
        content = '''
        I'll help you with that. First, let me search:
        <tool_code>print(Search(query="example"))</tool_code>

        That should give us the information we need.
        '''
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        assert result[0]["function"]["name"] == "Search"

    def test_empty_content(self):
        """Test with empty content."""
        result = extract_tool_calls_from_content("", "test-agent")
        assert result == []

    def test_tool_with_string_containing_parentheses(self):
        """Test tool calls with strings containing parentheses."""
        content = '<tool_code>print(Format(text="Hello (world)"))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"text": "Hello (world)"}


class TestToolCallStructure:
    """Tests verifying the structure of extracted tool calls matches OpenAI format."""

    def test_structure_matches_openai_format(self):
        """Test that extracted tool calls match OpenAI's tool_calls format."""
        content = '<tool_code>print(TestFunc(param="value"))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        tool_call = result[0]

        # Must have 'id', 'type', 'function' keys
        assert "id" in tool_call
        assert "type" in tool_call
        assert "function" in tool_call

        # Type must be "function"
        assert tool_call["type"] == "function"

        # Function must have 'name' and 'arguments'
        assert "name" in tool_call["function"]
        assert "arguments" in tool_call["function"]

        # Arguments must be JSON string
        assert isinstance(tool_call["function"]["arguments"], str)
        parsed = json.loads(tool_call["function"]["arguments"])
        assert isinstance(parsed, dict)

    def test_arguments_is_valid_json_string(self):
        """Test that arguments field is always valid JSON."""
        content = '<tool_code>print(MyTool(a=1, b="two", c=[3, 4]))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        args_str = result[0]["function"]["arguments"]

        # Should not raise
        parsed = json.loads(args_str)
        assert parsed["a"] == 1
        assert parsed["b"] == "two"
        assert parsed["c"] == [3, 4]


class TestEdgeCasesAndRobustness:
    """Tests for edge cases and error handling."""

    def test_unicode_in_arguments(self):
        """Test handling of unicode characters in arguments."""
        content = '<tool_code>print(Translate(text="こんにちは世界"))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["text"] == "こんにちは世界"

    def test_emoji_in_arguments(self):
        """Test handling of emoji in arguments."""
        content = '<tool_code>print(Post(message="Hello 👋 World 🌍"))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["message"] == "Hello 👋 World 🌍"

    def test_very_long_argument(self):
        """Test handling of very long argument values."""
        long_text = "x" * 10000
        content = f'<tool_code>print(Process(data="{long_text}"))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert len(args["data"]) == 10000

    def test_special_agent_id_chars(self):
        """Test with special characters in agent_id."""
        content = '<tool_code>print(Test())</tool_code>'
        result = extract_tool_calls_from_content(content, "agent-with-dashes_and_underscores")

        assert len(result) == 1
        assert "agent-with-dashes_and_underscores" in result[0]["id"]

    def test_consecutive_tool_calls(self):
        """Test tool calls appearing consecutively without text between."""
        content = '<tool_code>print(First())</tool_code><tool_code>print(Second())</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 2
        assert result[0]["function"]["name"] == "First"
        assert result[1]["function"]["name"] == "Second"

    def test_deeply_nested_arguments(self):
        """Test handling of deeply nested argument structures."""
        content = '<tool_code>print(Deep(data={"l1": {"l2": {"l3": {"l4": "deep"}}}}))</tool_code>'
        result = extract_tool_calls_from_content(content, "test-agent")

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["data"]["l1"]["l2"]["l3"]["l4"] == "deep"
