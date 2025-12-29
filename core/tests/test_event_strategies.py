"""
Tier 3 Unit Tests for agent_core/events/event_strategies.py and agent_core/events/ingestors.py

Tests the event handling system:
- EventHandlingStrategy class
- EVENT_STRATEGY_REGISTRY configuration
- Ingestor functions for various event types

Test Categories:
1. EventHandlingStrategy: Structure and configuration
2. Ingestor Registry: Registration and retrieval
3. Core Ingestors: templated_content, generic_message, tool_result
4. Specialized Ingestors: markdown_formatter, work_modules, json_history
5. Helper Functions: _apply_simple_template_interpolation, _recursive_markdown_formatter
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from agent_core.events.event_strategies import (
    EventHandlingStrategy,
    EVENT_STRATEGY_REGISTRY
)
from agent_core.events.ingestors import (
    INGESTOR_REGISTRY,
    register_ingestor,
    _apply_simple_template_interpolation,
    templated_content_ingestor,
    generic_message_ingestor,
    tool_result_ingestor,
    markdown_formatter_ingestor,
    work_modules_ingestor,
    principal_history_summary_ingestor,
    json_history_ingestor,
    tagged_content_ingestor,
    observer_failure_ingestor,
    user_prompt_ingestor,
    protocol_aware_ingestor,
    _recursive_markdown_formatter
)


class TestEventHandlingStrategy:
    """Tests for the EventHandlingStrategy class."""

    def test_init_stores_all_attributes(self):
        """Test that all attributes are stored correctly."""
        def mock_ingestor(p, params, ctx):
            return "result"

        strategy = EventHandlingStrategy(
            ingestor_func=mock_ingestor,
            default_injection_mode="append_as_new_message",
            default_params={"role": "user", "persistent": True}
        )

        assert strategy.ingestor is mock_ingestor
        assert strategy.default_injection_mode == "append_as_new_message"
        assert strategy.default_params["role"] == "user"
        assert strategy.default_params["persistent"] is True

    def test_strategy_callable(self):
        """Test that the ingestor function is callable through strategy."""
        def echo_ingestor(payload, params, ctx):
            return f"Received: {payload}"

        strategy = EventHandlingStrategy(
            ingestor_func=echo_ingestor,
            default_injection_mode="prepend",
            default_params={}
        )

        result = strategy.ingestor("test payload", {}, {})
        assert result == "Received: test payload"


class TestEventStrategyRegistry:
    """Tests for the EVENT_STRATEGY_REGISTRY configuration."""

    def test_registry_has_expected_event_types(self):
        """Test that all expected event types are registered."""
        expected_types = [
            "TOOL_RESULT",
            "AGENT_STARTUP_BRIEFING",
            "SELF_REFLECTION_PROMPT",
            "INTERNAL_DIRECTIVE",
            "PARTNER_DIRECTIVE",
            "PRINCIPAL_COMPLETED",
            "WORK_MODULES_STATUS_UPDATE",
            "PRINCIPAL_ACTIVITY_UPDATE",
            "FIM_INSTRUCTION",
            "JSON_HISTORY_FOR_LLM",
            "TOOL_INPUTS_BRIEFING",
            "ORIGINAL_QUESTION",
            "OBSERVER_FAILURE",
            "USER_PROMPT"
        ]

        for event_type in expected_types:
            assert event_type in EVENT_STRATEGY_REGISTRY, f"Missing: {event_type}"

    def test_tool_result_strategy_config(self):
        """Test TOOL_RESULT strategy configuration."""
        strategy = EVENT_STRATEGY_REGISTRY["TOOL_RESULT"]

        assert strategy.default_injection_mode == "append_as_new_message"
        assert strategy.default_params["role"] == "tool"
        assert strategy.default_params["is_persistent_in_memory"] is True
        assert strategy.ingestor is tool_result_ingestor

    def test_agent_startup_briefing_strategy_config(self):
        """Test AGENT_STARTUP_BRIEFING strategy configuration."""
        strategy = EVENT_STRATEGY_REGISTRY["AGENT_STARTUP_BRIEFING"]

        assert strategy.default_injection_mode == "append_as_new_message"
        assert strategy.default_params["role"] == "user"
        assert strategy.ingestor is protocol_aware_ingestor

    def test_partner_directive_has_formatting_params(self):
        """Test PARTNER_DIRECTIVE has special formatting parameters."""
        strategy = EVENT_STRATEGY_REGISTRY["PARTNER_DIRECTIVE"]

        assert "title" in strategy.default_params
        assert "key_renames" in strategy.default_params
        assert strategy.default_params["key_renames"]["content"] == "Instruction"

    def test_observer_failure_is_transient(self):
        """Test OBSERVER_FAILURE is configured as non-persistent."""
        strategy = EVENT_STRATEGY_REGISTRY["OBSERVER_FAILURE"]

        assert strategy.default_params["is_persistent_in_memory"] is False
        assert strategy.default_params["role"] == "system"


class TestRegisterIngestor:
    """Tests for the @register_ingestor decorator."""

    def test_decorator_registers_function(self):
        """Test that the decorator adds function to registry."""
        @register_ingestor("test_custom_ingestor")
        def custom_ingestor(payload, params, context):
            return "custom"

        assert "test_custom_ingestor" in INGESTOR_REGISTRY
        assert INGESTOR_REGISTRY["test_custom_ingestor"] is custom_ingestor

    def test_decorator_allows_override_with_warning(self):
        """Test that overriding an ingestor logs a warning."""
        @register_ingestor("override_test")
        def first_version(p, params, c):
            return "first"

        with patch('agent_core.events.ingestors.logger.warning') as mock_warn:
            @register_ingestor("override_test")
            def second_version(p, params, c):
                return "second"

            mock_warn.assert_called()

        assert INGESTOR_REGISTRY["override_test"]({}, {}, {}) == "second"


class TestApplySimpleTemplateInterpolation:
    """Tests for the _apply_simple_template_interpolation helper."""

    def test_no_template_markers(self):
        """Test text without template markers is unchanged."""
        result = _apply_simple_template_interpolation("plain text", {})
        assert result == "plain text"

    def test_single_variable_replacement(self):
        """Test single variable replacement."""
        context = {"state": {"user_name": "Alice"}}
        result = _apply_simple_template_interpolation(
            "Hello, {{ state.user_name }}!",
            context
        )
        assert result == "Hello, Alice!"

    def test_multiple_variable_replacement(self):
        """Test multiple variables in same string."""
        context = {
            "state": {"name": "Bob"},
            "meta": {"count": 5}
        }
        result = _apply_simple_template_interpolation(
            "{{ state.name }} has {{ meta.count }} items",
            context
        )
        assert result == "Bob has 5 items"

    def test_missing_variable_keeps_template(self):
        """Test that missing variables keep original template marker."""
        context = {"state": {}}
        result = _apply_simple_template_interpolation(
            "Value: {{ state.missing }}",
            context
        )
        assert result == "Value: {{ state.missing }}"

    def test_non_string_input(self):
        """Test that non-string input is returned unchanged."""
        result = _apply_simple_template_interpolation(123, {})
        assert result == 123

    def test_whitespace_in_template(self):
        """Test that whitespace in templates is handled."""
        # The function uses get_nested_value_from_context which expects dotted paths
        # Keys must be accessible via the context path resolution
        context = {"state": {"key": "value"}}
        result = _apply_simple_template_interpolation(
            "{{   state.key   }}",
            context
        )
        assert result == "value"


class TestTemplatedContentIngestor:
    """Tests for the templated_content_ingestor function."""

    def test_invalid_payload_returns_error(self):
        """Test that invalid payload returns error message."""
        result = templated_content_ingestor("not a dict", {}, {})
        assert "[Error:" in result

    def test_missing_content_key_returns_error(self):
        """Test that missing content_key returns error."""
        result = templated_content_ingestor({"other": "value"}, {}, {})
        assert "[Error:" in result

    def test_template_not_found_returns_error(self):
        """Test that missing template returns error."""
        context = {
            "loaded_profile": {
                "name": "TestProfile",
                "text_definitions": {}
            }
        }
        result = templated_content_ingestor(
            {"content_key": "nonexistent"},
            {},
            context
        )
        assert "[Error:" in result
        assert "nonexistent" in result

    def test_template_found_and_interpolated(self):
        """Test that template is found and interpolated."""
        context = {
            "loaded_profile": {
                "name": "TestProfile",
                "text_definitions": {
                    "greeting": "Hello, {{ state.user }}!"
                }
            },
            "state": {"user": "World"}
        }
        result = templated_content_ingestor(
            {"content_key": "greeting"},
            {},
            context
        )
        assert result == "Hello, World!"

    def test_wrapper_tags_applied(self):
        """Test that wrapper tags are applied."""
        context = {
            "loaded_profile": {
                "text_definitions": {"msg": "content"}
            }
        }
        result = templated_content_ingestor(
            {"content_key": "msg"},
            {"wrapper_tags": ["<start>", "</end>"]},
            context
        )
        assert result == "<start>content</end>"


class TestGenericMessageIngestor:
    """Tests for the generic_message_ingestor function."""

    def test_default_template(self):
        """Test with default template."""
        result = generic_message_ingestor("test payload", {}, {})
        assert result == "test payload"

    def test_custom_template_with_payload(self):
        """Test custom template with payload placeholder."""
        result = generic_message_ingestor(
            "my value",
            {"content_template": "Result: {{ payload }}"},
            {}
        )
        assert result == "Result: my value"

    def test_dict_payload_with_key_replacement(self):
        """Test dict payload with specific key replacements."""
        result = generic_message_ingestor(
            {"name": "Alice", "count": 5},
            {"content_template": "{{ payload.name }} has {{ payload.count }} items"},
            {}
        )
        assert result == "Alice has 5 items"


class TestToolResultIngestor:
    """Tests for the tool_result_ingestor function."""

    def test_non_dict_returns_string(self):
        """Test that non-dict payload is stringified."""
        result = tool_result_ingestor("simple result", {}, {})
        assert result == "simple result"

    def test_dehydrated_token_returned_directly(self):
        """Test that dehydrated tokens are returned unchanged."""
        payload = {"content": "<#CGKB-abc123-def456>", "tool_name": "test"}
        result = tool_result_ingestor(payload, {}, {})
        assert result == "<#CGKB-abc123-def456>"

    def test_error_wrapped_with_tags(self):
        """Test that errors are wrapped with error tags."""
        payload = {
            "tool_name": "failing_tool",
            "is_error": True,
            "content": {"reason": "Connection failed"}
        }
        result = tool_result_ingestor(payload, {}, {})

        assert "<tool_error_report>" in result
        assert "</tool_error_report>" in result
        assert "failing_tool" in result

    def test_main_content_for_llm_prioritized(self):
        """Test that main_content_for_llm is used when present."""
        payload = {
            "tool_name": "smart_tool",
            "content": {
                "main_content_for_llm": {"summary": "Important info"},
                "extra_data": "ignored"
            }
        }
        result = tool_result_ingestor(payload, {}, {})

        assert "Important info" in result
        assert "extra_data" not in result

    def test_raw_json_escape_hatch(self):
        """Test that _raw_json returns JSON directly."""
        raw_data = {"key": "value", "nested": {"inner": 123}}
        payload = {
            "tool_name": "json_tool",
            "content": {"_raw_json": raw_data}
        }
        result = tool_result_ingestor(payload, {}, {})

        assert json.loads(result) == raw_data


class TestMarkdownFormatterIngestor:
    """Tests for the markdown_formatter_ingestor function."""

    def test_non_dict_stringified(self):
        """Test that non-dict payloads are stringified."""
        result = markdown_formatter_ingestor(["a", "list"], {}, {})
        assert result == "['a', 'list']"

    def test_default_title(self):
        """Test default title is applied."""
        result = markdown_formatter_ingestor({"key": "value"}, {}, {})
        assert "### Contextual Information" in result

    def test_custom_title(self):
        """Test custom title is applied."""
        result = markdown_formatter_ingestor(
            {"key": "value"},
            {"title": "### Custom Title"},
            {}
        )
        assert "### Custom Title" in result

    def test_key_renames(self):
        """Test that key renames are applied."""
        result = markdown_formatter_ingestor(
            {"old_key": "value"},
            {"key_renames": {"old_key": "New Key Name"}},
            {}
        )
        assert "**New Key Name**" in result

    def test_exclude_keys(self):
        """Test that excluded keys are omitted."""
        result = markdown_formatter_ingestor(
            {"visible": "yes", "hidden": "no"},
            {"exclude_keys": ["hidden"]},
            {}
        )
        # The key is title-cased in output, so check for "Visible" not "visible"
        assert "Visible" in result
        assert "hidden" not in result.lower()  # Check that hidden key is excluded


class TestWorkModulesIngestor:
    """Tests for the work_modules_ingestor function."""

    def test_non_dict_returns_error_message(self):
        """Test that non-dict returns error message."""
        result = work_modules_ingestor("not a dict", {}, {})
        assert "not in the expected format" in result

    def test_empty_dict_shows_no_modules(self):
        """Test empty dict shows no modules message."""
        result = work_modules_ingestor({}, {}, {})
        assert "No work modules" in result

    def test_excludes_large_fields(self):
        """Test that large fields like context_archive are excluded."""
        payload = {
            "mod_1": {
                "status": "completed",
                "context_archive": ["msg1", "msg2"] * 1000,  # Large
                "deliverables": {"summary": "Result"}
            }
        }
        result = work_modules_ingestor(payload, {}, {})

        assert "status" in result.lower()
        assert "context_archive" not in result

    def test_summarizes_deliverables(self):
        """Test that deliverables are summarized."""
        payload = {
            "mod_1": {
                "status": "done",
                "deliverables": {"a": 1, "b": 2, "c": 3}
            }
        }
        result = work_modules_ingestor(payload, {}, {})

        assert "(3 items)" in result


class TestPrincipalHistorySummaryIngestor:
    """Tests for the principal_history_summary_ingestor function."""

    def test_non_list_returns_default(self):
        """Test that non-list returns default message."""
        result = principal_history_summary_ingestor("not a list", {}, {})
        assert "no recorded activity" in result

    def test_empty_list_returns_default(self):
        """Test that empty list returns default message."""
        result = principal_history_summary_ingestor([], {}, {})
        assert "no recorded activity" in result

    def test_formats_messages(self):
        """Test that messages are formatted."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        result = principal_history_summary_ingestor(messages, {}, {})

        assert "[USER]" in result
        assert "[ASSISTANT]" in result
        assert "Hello" in result

    def test_truncates_long_content(self):
        """Test that long content is truncated."""
        messages = [
            {"role": "user", "content": "x" * 500}
        ]
        result = principal_history_summary_ingestor(messages, {}, {})

        assert "..." in result
        assert len(result) < 600  # Definitely truncated

    def test_includes_tool_calls(self):
        """Test that tool calls are included."""
        messages = [
            {
                "role": "assistant",
                "content": "Using tool",
                "tool_calls": [
                    {"function": {"name": "search_web", "arguments": '{"q": "test"}'}}
                ]
            }
        ]
        result = principal_history_summary_ingestor(messages, {}, {})

        assert "search_web" in result

    def test_respects_max_messages(self):
        """Test that max_messages limit is respected."""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        result = principal_history_summary_ingestor(messages, {"max_messages": 5}, {})

        assert "omitting" in result


class TestJsonHistoryIngestor:
    """Tests for the json_history_ingestor function."""

    def test_non_list_returns_error(self):
        """Test that non-list returns error message."""
        result = json_history_ingestor({"not": "list"}, {}, {})
        assert "[Error:" in result

    def test_valid_list_serialized(self):
        """Test that valid list is serialized to JSON."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        result = json_history_ingestor(history, {}, {})

        assert "<message_history_json>" in result
        assert "</message_history_json>" in result
        parsed = json.loads(result.split("<message_history_json>")[1].split("</message_history_json>")[0])
        assert parsed == history


class TestTaggedContentIngestor:
    """Tests for the tagged_content_ingestor function."""

    def test_with_wrapper_tags(self):
        """Test content wrapped with tags."""
        result = tagged_content_ingestor(
            "my content",
            {"wrapper_tags": ["<custom>", "</custom>"]},
            {}
        )
        assert result == "<custom>my content</custom>"

    def test_without_wrapper_tags(self):
        """Test content without tags returns stringified."""
        result = tagged_content_ingestor("content", {}, {})
        assert result == "content"

    def test_invalid_wrapper_tags_format(self):
        """Test invalid wrapper tags format returns plain content."""
        result = tagged_content_ingestor(
            "content",
            {"wrapper_tags": "not a list"},
            {}
        )
        assert result == "content"


class TestObserverFailureIngestor:
    """Tests for the observer_failure_ingestor function."""

    def test_non_dict_returns_error(self):
        """Test that non-dict returns error message."""
        result = observer_failure_ingestor("not dict", {}, {})
        assert "[Error:" in result

    def test_formats_failure_details(self):
        """Test that failure details are formatted."""
        payload = {
            "failed_observer_id": "context_observer_1",
            "error_message": "Database connection failed"
        }
        result = observer_failure_ingestor(payload, {}, {})

        assert "<system_error" in result
        assert "context_observer_1" in result
        assert "Database connection failed" in result
        assert "MUST inform the user" in result


class TestUserPromptIngestor:
    """Tests for the user_prompt_ingestor function."""

    def test_extracts_prompt_from_dict(self):
        """Test extracting prompt from dict payload."""
        result = user_prompt_ingestor({"prompt": "What is AI?"}, {}, {})
        assert result == "What is AI?"

    def test_stringifies_non_dict(self):
        """Test that non-dict is stringified."""
        result = user_prompt_ingestor("direct question", {}, {})
        assert result == "direct question"

    def test_missing_prompt_key(self):
        """Test dict without prompt key returns empty."""
        result = user_prompt_ingestor({"other": "value"}, {}, {})
        assert result == ""


class TestProtocolAwareIngestor:
    """Tests for the protocol_aware_ingestor function."""

    def test_invalid_payload_returns_error(self):
        """Test that invalid payload returns error."""
        result = protocol_aware_ingestor("not a dict", {}, {})
        assert "[Error:" in result

    def test_missing_data_returns_error(self):
        """Test that missing data key returns error."""
        result = protocol_aware_ingestor(
            {"schema_for_rendering": {}},
            {},
            {}
        )
        assert "[Error:" in result

    def test_valid_payload_rendered(self):
        """Test that valid payload is rendered."""
        payload = {
            "data": {
                "task": "Complete the project",
                "context": ["item1", "item2"]
            },
            "schema_for_rendering": {
                "type": "object",
                "x-handover-title": "Mission Briefing",
                "properties": {
                    "task": {"x-handover-title": "Your Task"},
                    "context": {"x-handover-title": "Background"}
                }
            }
        }
        result = protocol_aware_ingestor(payload, {}, {})

        assert "## Mission Briefing" in result
        assert "Your Task" in result
        assert "Complete the project" in result


class TestRecursiveMarkdownFormatter:
    """Tests for the _recursive_markdown_formatter helper."""

    def test_empty_dict(self):
        """Test empty dict formatting."""
        result = _recursive_markdown_formatter({}, {}, 0)
        assert "(empty)" in result[0]

    def test_empty_list(self):
        """Test empty list formatting."""
        result = _recursive_markdown_formatter([], {}, 0)
        assert "(empty)" in result[0]

    def test_simple_dict(self):
        """Test simple dict formatting."""
        result = _recursive_markdown_formatter({"key": "value"}, {}, 0)
        lines = "\n".join(result)

        assert "**Key:**" in lines
        assert "value" in lines

    def test_nested_dict(self):
        """Test nested dict formatting."""
        data = {"outer": {"inner": "deep value"}}
        result = _recursive_markdown_formatter(data, {}, 0)
        lines = "\n".join(result)

        assert "**Outer:**" in lines
        assert "**Inner:**" in lines
        assert "deep value" in lines

    def test_list_formatting(self):
        """Test list formatting."""
        result = _recursive_markdown_formatter(["a", "b", "c"], {}, 0)
        lines = "\n".join(result)

        assert "a" in lines
        assert "b" in lines
        assert "c" in lines

    def test_multiline_string(self):
        """Test multi-line string handling."""
        result = _recursive_markdown_formatter("line1\nline2\nline3", {}, 0)

        assert len(result) == 3
        assert "line1" in result[0]
        assert "line2" in result[1]

    def test_schema_based_rendering(self):
        """Test rendering with schema provides titles."""
        data = {"task_description": "Do something"}
        schema = {
            "type": "object",
            "properties": {
                "task_description": {"x-handover-title": "Your Mission"}
            }
        }
        result = _recursive_markdown_formatter(data, schema, 0)
        lines = "\n".join(result)

        assert "**Your Mission:**" in lines

    def test_primitive_types(self):
        """Test primitive type formatting."""
        assert "42" in _recursive_markdown_formatter(42, {}, 0)[0]
        assert "True" in _recursive_markdown_formatter(True, {}, 0)[0]
        assert "3.14" in _recursive_markdown_formatter(3.14, {}, 0)[0]
