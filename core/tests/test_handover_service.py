"""
Tier 3 Unit Tests for agent_core/framework/handover_service.py

Tests the HandoverService class which manages agent handover protocols:
- Protocol loading and caching
- Schema extraction from tool parameters
- Path template resolution with placeholders
- Inheritance rule processing (direct, iterative, path-based)
- Message filtering with _no_handover flag
- Final payload and schema assembly

Test Categories:
1. Protocol Loading: load_protocols(), get_protocol_schema()
2. Parameter Extraction: _extract_from_tool_params()
3. Path Resolution: _resolve_path()
4. Execution Flow: execute() - full handover execution
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import yaml

# Import the class under test - need to handle the module-level load
with patch('agent_core.framework.handover_service.HandoverService.load_protocols'):
    from agent_core.framework.handover_service import HandoverService


class TestHandoverServiceProtocolLoading:
    """Tests for protocol loading and schema retrieval."""

    def setup_method(self):
        """Reset the class-level protocols cache before each test."""
        HandoverService._protocols = {}

    def test_load_protocols_empty_directory(self, tmp_path):
        """Test loading from an empty directory."""
        with patch('agent_core.framework.handover_service.PROTOCOLS_DIR', tmp_path):
            HandoverService._protocols = {}
            HandoverService.load_protocols()
            assert HandoverService._protocols == {}

    def test_load_protocols_valid_yaml(self, tmp_path):
        """Test loading valid protocol YAML files."""
        protocol_data = {
            "protocol_name": "test_protocol",
            "context_parameters": {"type": "object", "properties": {"task": {"type": "string"}}},
            "inheritance": [],
            "target_inbox_item": {"source": "TEST_SOURCE"}
        }
        yaml_file = tmp_path / "test_protocol.yaml"
        yaml_file.write_text(yaml.dump(protocol_data))

        with patch('agent_core.framework.handover_service.PROTOCOLS_DIR', tmp_path):
            HandoverService._protocols = {}
            HandoverService.load_protocols()

            assert "test_protocol" in HandoverService._protocols
            assert HandoverService._protocols["test_protocol"]["protocol_name"] == "test_protocol"

    def test_load_protocols_skips_if_already_loaded(self, tmp_path):
        """Test that load_protocols skips if protocols already exist."""
        HandoverService._protocols = {"existing": {"protocol_name": "existing"}}

        with patch('agent_core.framework.handover_service.PROTOCOLS_DIR', tmp_path):
            # Create a new file that shouldn't be loaded
            yaml_file = tmp_path / "new.yaml"
            yaml_file.write_text(yaml.dump({"protocol_name": "new"}))

            HandoverService.load_protocols()

            # Should still only have the existing protocol
            assert "new" not in HandoverService._protocols
            assert "existing" in HandoverService._protocols

    def test_load_protocols_handles_invalid_yaml(self, tmp_path):
        """Test handling of invalid YAML files."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: {[")

        with patch('agent_core.framework.handover_service.PROTOCOLS_DIR', tmp_path):
            with patch('agent_core.framework.handover_service.logger'):
                HandoverService._protocols = {}
                # Should not raise, just log error
                HandoverService.load_protocols()
                assert HandoverService._protocols == {}

    def test_load_protocols_skips_missing_protocol_name(self, tmp_path):
        """Test that files without protocol_name are skipped."""
        protocol_data = {"some_key": "some_value"}  # No protocol_name
        yaml_file = tmp_path / "no_name.yaml"
        yaml_file.write_text(yaml.dump(protocol_data))

        with patch('agent_core.framework.handover_service.PROTOCOLS_DIR', tmp_path):
            HandoverService._protocols = {}
            HandoverService.load_protocols()
            assert HandoverService._protocols == {}

    def test_get_protocol_schema_existing(self):
        """Test retrieving schema from existing protocol."""
        schema = {"type": "object", "properties": {"task": {"type": "string"}}}
        HandoverService._protocols = {
            "my_protocol": {
                "protocol_name": "my_protocol",
                "context_parameters": schema
            }
        }

        result = HandoverService.get_protocol_schema("my_protocol")
        assert result == schema

    def test_get_protocol_schema_nonexistent(self):
        """Test retrieving schema from non-existent protocol returns None."""
        HandoverService._protocols = {}
        result = HandoverService.get_protocol_schema("nonexistent")
        assert result is None

    def test_get_protocol_schema_no_context_params(self):
        """Test protocol without context_parameters returns None."""
        HandoverService._protocols = {
            "bare_protocol": {"protocol_name": "bare_protocol"}
        }
        result = HandoverService.get_protocol_schema("bare_protocol")
        assert result is None


class TestExtractFromToolParams:
    """Tests for _extract_from_tool_params method."""

    def test_extract_matching_properties(self):
        """Test extracting properties that exist in tool_params."""
        schema = {
            "type": "object",
            "properties": {
                "task_description": {"type": "string"},
                "priority": {"type": "integer"}
            }
        }
        tool_params = {
            "task_description": "Do something",
            "priority": 5,
            "extra_param": "ignored"
        }

        result = HandoverService._extract_from_tool_params(schema, tool_params)

        assert result == {"task_description": "Do something", "priority": 5}
        assert "extra_param" not in result

    def test_extract_partial_match(self):
        """Test extraction when only some properties are present."""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"}
            }
        }
        tool_params = {"required_field": "value"}

        result = HandoverService._extract_from_tool_params(schema, tool_params)

        assert result == {"required_field": "value"}
        assert "optional_field" not in result

    def test_extract_non_object_schema(self):
        """Test with non-object type schema."""
        schema = {"type": "array", "items": {"type": "string"}}
        tool_params = {"some": "params"}

        result = HandoverService._extract_from_tool_params(schema, tool_params)
        assert result == {}

    def test_extract_no_properties(self):
        """Test with object schema but no properties."""
        schema = {"type": "object"}
        tool_params = {"some": "params"}

        result = HandoverService._extract_from_tool_params(schema, tool_params)
        assert result == {}

    def test_extract_empty_tool_params(self):
        """Test extraction from empty tool_params."""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}}
        }

        result = HandoverService._extract_from_tool_params(schema, {})
        assert result == {}


class TestResolvePath:
    """Tests for _resolve_path method."""

    def test_resolve_single_placeholder(self):
        """Test resolving a path with a single placeholder."""
        path_template = "state.modules.{{ module_id }}.data"
        replacements = {"module_id": "meta.current_module"}
        source_context = {"meta": {"current_module": "mod_123"}}

        result = HandoverService._resolve_path(path_template, replacements, source_context)

        assert result == "state.modules.mod_123.data"

    def test_resolve_multiple_placeholders(self):
        """Test resolving a path with multiple placeholders."""
        path_template = "state.{{ agent }}.{{ action }}.result"
        replacements = {
            "agent": "meta.agent_id",
            "action": "state.current_action_id"
        }
        source_context = {
            "meta": {"agent_id": "principal"},
            "state": {"current_action_id": "search"}
        }

        result = HandoverService._resolve_path(path_template, replacements, source_context)

        assert result == "state.principal.search.result"

    def test_resolve_unresolvable_placeholder_returns_none(self):
        """Test that unresolvable placeholders return None."""
        path_template = "state.{{ missing }}.data"
        replacements = {"missing": "nonexistent.path"}
        source_context = {"state": {}}

        result = HandoverService._resolve_path(path_template, replacements, source_context)

        assert result is None

    def test_resolve_partial_resolution_returns_none(self):
        """Test that partial resolution (leftover placeholders) returns None."""
        path_template = "state.{{ first }}.{{ second }}.data"
        replacements = {"first": "meta.value"}
        source_context = {"meta": {"value": "resolved"}}

        # second placeholder has no replacement
        result = HandoverService._resolve_path(path_template, replacements, source_context)

        # Should still have {{ second }} unresolved, so returns None
        assert result is None

    def test_resolve_no_placeholders(self):
        """Test path with no placeholders."""
        path_template = "state.fixed.path"
        replacements = {}
        source_context = {}

        result = HandoverService._resolve_path(path_template, replacements, source_context)

        assert result == "state.fixed.path"


class TestHandoverServiceExecute:
    """Tests for the execute() async method."""

    def setup_method(self):
        """Reset protocols cache before each test."""
        HandoverService._protocols = {}

    @pytest.mark.asyncio
    async def test_execute_protocol_not_found(self):
        """Test execute raises ValueError for unknown protocol."""
        HandoverService._protocols = {}

        with pytest.raises(ValueError, match="not found"):
            await HandoverService.execute("unknown_protocol", {})

    @pytest.mark.asyncio
    async def test_execute_basic_protocol(self):
        """Test execute with a simple protocol (no inheritance)."""
        HandoverService._protocols = {
            "simple_protocol": {
                "protocol_name": "simple_protocol",
                "context_parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}}
                },
                "inheritance": [],
                "target_inbox_item": {"source": "AGENT_STARTUP_BRIEFING"}
            }
        }

        source_context = {
            "state": {
                "current_action": {"task": "Test task", "type": "handoff"}
            }
        }

        result = await HandoverService.execute("simple_protocol", source_context)

        assert result["source"] == "AGENT_STARTUP_BRIEFING"
        assert result["payload"]["data"]["task"] == "Test task"
        assert "schema_for_rendering" in result["payload"]

    @pytest.mark.asyncio
    async def test_execute_with_inheritance_condition_true(self):
        """Test execute with inheritance rules that pass condition."""
        HandoverService._protocols = {
            "conditional_protocol": {
                "protocol_name": "conditional_protocol",
                "context_parameters": {"type": "object", "properties": {}},
                "inheritance": [
                    {
                        "condition": "len(v['state.messages']) > 0",
                        "from_source": {"path": "state.messages", "replace": {}},
                        "as_payload_key": "inherited_messages",
                        "x-handover-title": "History"
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {
            "state": {
                "current_action": {},
                "messages": [{"role": "user", "content": "Hello"}]
            }
        }

        result = await HandoverService.execute("conditional_protocol", source_context)

        assert "inherited_messages" in result["payload"]["data"]
        assert len(result["payload"]["data"]["inherited_messages"]) == 1

    @pytest.mark.asyncio
    async def test_execute_with_inheritance_condition_false(self):
        """Test execute with inheritance rules that fail condition."""
        HandoverService._protocols = {
            "conditional_protocol": {
                "protocol_name": "conditional_protocol",
                "context_parameters": {"type": "object", "properties": {}},
                "inheritance": [
                    {
                        "condition": "len(v['state.messages']) > 0",
                        "from_source": {"path": "state.messages", "replace": {}},
                        "as_payload_key": "inherited_messages"
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {
            "state": {
                "current_action": {},
                "messages": []  # Empty - condition will fail
            }
        }

        result = await HandoverService.execute("conditional_protocol", source_context)

        assert "inherited_messages" not in result["payload"]["data"]

    @pytest.mark.asyncio
    async def test_execute_filters_no_handover_messages(self):
        """Test that messages with _no_handover flag are filtered."""
        HandoverService._protocols = {
            "filter_protocol": {
                "protocol_name": "filter_protocol",
                "context_parameters": {"type": "object", "properties": {}},
                "inheritance": [
                    {
                        "condition": "True",
                        "from_source": {"path": "state.messages", "replace": {}},
                        "as_payload_key": "inherited_messages"
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {
            "state": {
                "current_action": {},
                "messages": [
                    {"role": "user", "content": "Keep me"},
                    {"role": "assistant", "content": "Filter me", "_internal": {"_no_handover": True}},
                    {"role": "user", "content": "Keep me too"}
                ]
            }
        }

        result = await HandoverService.execute("filter_protocol", source_context)

        inherited = result["payload"]["data"]["inherited_messages"]
        assert len(inherited) == 2
        assert all(not msg.get("_internal", {}).get("_no_handover") for msg in inherited)

    @pytest.mark.asyncio
    async def test_execute_with_path_replacement(self):
        """Test execute with path resolution using replace."""
        HandoverService._protocols = {
            "path_protocol": {
                "protocol_name": "path_protocol",
                "context_parameters": {"type": "object", "properties": {}},
                "inheritance": [
                    {
                        "condition": "True",
                        "from_source": {
                            "path": "state.modules.{{ mod_id }}.data",
                            "replace": {"mod_id": "meta.current_module"}
                        },
                        "as_payload_key": "module_data"
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {
            "state": {
                "current_action": {},
                "modules": {
                    "active_module": {"data": {"key": "value"}}
                }
            },
            "meta": {"current_module": "active_module"}
        }

        result = await HandoverService.execute("path_protocol", source_context)

        assert result["payload"]["data"]["module_data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_execute_with_iterative_inheritance(self):
        """Test execute with iterative inheritance (path_to_iterate)."""
        HandoverService._protocols = {
            "iterative_protocol": {
                "protocol_name": "iterative_protocol",
                "context_parameters": {"type": "object", "properties": {}},
                "inheritance": [
                    {
                        "condition": "True",
                        "from_source": {
                            "path_to_iterate": "state.items.{{ item_id }}.results",
                            "iterate_on": {"item_id": "meta.item_ids"}
                        },
                        "as_payload_key": "all_results"
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {
            "state": {
                "current_action": {},
                "items": {
                    "item_1": {"results": [{"result": "A"}]},
                    "item_2": {"results": [{"result": "B"}, {"result": "C"}]}
                }
            },
            "meta": {"item_ids": ["item_1", "item_2"]}
        }

        result = await HandoverService.execute("iterative_protocol", source_context)

        all_results = result["payload"]["data"]["all_results"]
        assert len(all_results) == 3
        assert {"result": "A"} in all_results
        assert {"result": "B"} in all_results
        assert {"result": "C"} in all_results

    @pytest.mark.asyncio
    async def test_execute_schema_for_rendering_populated(self):
        """Test that schema_for_rendering is properly populated."""
        HandoverService._protocols = {
            "schema_protocol": {
                "protocol_name": "schema_protocol",
                "context_parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "The task"}
                    }
                },
                "inheritance": [
                    {
                        "condition": "True",
                        "from_source": {"path": "state.history", "replace": {}},
                        "as_payload_key": "context_history",
                        "x-handover-title": "Previous Context",
                        "schema": {"type": "array"}
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {
            "state": {
                "current_action": {"task": "Do something"},
                "history": ["event1", "event2"]
            }
        }

        result = await HandoverService.execute("schema_protocol", source_context)

        schema = result["payload"]["schema_for_rendering"]
        assert schema["type"] == "object"
        assert "task" in schema["properties"]
        assert "context_history" in schema["properties"]
        assert schema["properties"]["context_history"]["x-handover-title"] == "Previous Context"

    @pytest.mark.asyncio
    async def test_execute_invalid_condition_continues(self):
        """Test that invalid eval conditions are handled gracefully."""
        HandoverService._protocols = {
            "bad_condition": {
                "protocol_name": "bad_condition",
                "context_parameters": {"type": "object", "properties": {}},
                "inheritance": [
                    {
                        "condition": "this is not valid python",
                        "from_source": {"path": "state.data", "replace": {}},
                        "as_payload_key": "data"
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {
            "state": {"current_action": {}, "data": "some data"}
        }

        # Should not raise, just skip the rule
        result = await HandoverService.execute("bad_condition", source_context)

        assert "data" not in result["payload"]["data"]

    @pytest.mark.asyncio
    async def test_execute_missing_from_source_or_payload_key(self):
        """Test that rules without from_source or as_payload_key are skipped."""
        HandoverService._protocols = {
            "incomplete_rule": {
                "protocol_name": "incomplete_rule",
                "context_parameters": {"type": "object", "properties": {}},
                "inheritance": [
                    {
                        "condition": "True",
                        "from_source": {"path": "state.data", "replace": {}}
                        # Missing as_payload_key
                    },
                    {
                        "condition": "True",
                        "as_payload_key": "orphan_key"
                        # Missing from_source
                    }
                ],
                "target_inbox_item": {"source": "TEST_SOURCE"}
            }
        }

        source_context = {"state": {"current_action": {}, "data": "test"}}

        result = await HandoverService.execute("incomplete_rule", source_context)

        # Neither should be in the payload
        assert "data" not in result["payload"]["data"]
        assert "orphan_key" not in result["payload"]["data"]
