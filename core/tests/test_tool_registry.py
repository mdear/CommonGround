"""
Unit tests for agent_core/framework/tool_registry.py

Tests tool registration, retrieval, and formatting functions.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agent_core.framework.tool_registry import (
    _TOOL_REGISTRY,
    _sanitize_schema_for_api,
    tool_registry,
    get_registered_tools,
    get_tool_by_name,
    get_tool_node_class,
    get_tools_by_toolset_names,
    get_all_toolsets_with_tools,
    format_tools_for_llm_api,
    format_tools_for_prompt,
    format_tools_for_prompt_by_toolset,
    format_simplified_tools_for_prompt_by_toolset,
    register_native_mcp_tool,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the tool registry before and after each test."""
    _TOOL_REGISTRY.clear()
    yield
    _TOOL_REGISTRY.clear()


class TestSanitizeSchemaForApi:
    """Tests for the _sanitize_schema_for_api function."""

    def test_removes_x_prefixed_keys(self):
        """Test that keys starting with x- are removed."""
        schema = {
            "type": "object",
            "x-custom": "should be removed",
            "properties": {
                "name": {"type": "string", "x-internal": True}
            }
        }

        result = _sanitize_schema_for_api(schema)

        assert "x-custom" not in result
        assert "x-internal" not in result["properties"]["name"]
        assert result["type"] == "object"
        assert result["properties"]["name"]["type"] == "string"

    def test_handles_nested_dicts(self):
        """Test that nested dictionaries are processed recursively."""
        schema = {
            "properties": {
                "outer": {
                    "x-metadata": "remove",
                    "properties": {
                        "inner": {"x-deep": True, "type": "number"}
                    }
                }
            }
        }

        result = _sanitize_schema_for_api(schema)

        assert "x-metadata" not in result["properties"]["outer"]
        assert "x-deep" not in result["properties"]["outer"]["properties"]["inner"]

    def test_handles_lists(self):
        """Test that lists are processed recursively."""
        schema = {
            "oneOf": [
                {"type": "string", "x-option": 1},
                {"type": "number", "x-option": 2}
            ]
        }

        result = _sanitize_schema_for_api(schema)

        assert len(result["oneOf"]) == 2
        assert "x-option" not in result["oneOf"][0]
        assert "x-option" not in result["oneOf"][1]

    def test_preserves_primitives(self):
        """Test that primitive values are preserved."""
        schema = {"type": "string", "minLength": 1, "maxLength": 100}

        result = _sanitize_schema_for_api(schema)

        assert result == schema

    def test_handles_empty_dict(self):
        """Test handling of empty dictionary."""
        result = _sanitize_schema_for_api({})

        assert result == {}

    def test_handles_none(self):
        """Test handling of None value."""
        result = _sanitize_schema_for_api(None)

        assert result is None


class TestToolRegistryDecorator:
    """Tests for the @tool_registry decorator."""

    def test_registers_tool(self):
        """Test that decorator registers a tool in the registry."""
        from pocketflow import BaseNode

        @tool_registry(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}}
        )
        class TestTool(BaseNode):
            pass

        assert "test_tool" in _TOOL_REGISTRY
        assert _TOOL_REGISTRY["test_tool"]["name"] == "test_tool"
        assert _TOOL_REGISTRY["test_tool"]["description"] == "A test tool"

    def test_stores_node_class(self):
        """Test that decorator stores the node class reference."""
        from pocketflow import BaseNode

        @tool_registry(
            name="test_tool_class",
            description="Test",
            parameters={}
        )
        class TestToolClass(BaseNode):
            pass

        assert _TOOL_REGISTRY["test_tool_class"]["node_class"] is TestToolClass

    def test_sets_tool_info_attribute(self):
        """Test that decorator sets _tool_info attribute on class."""
        from pocketflow import BaseNode

        @tool_registry(
            name="tool_with_attr",
            description="Test",
            parameters={}
        )
        class ToolWithAttr(BaseNode):
            pass

        assert hasattr(ToolWithAttr, "_tool_info")
        assert ToolWithAttr._tool_info["name"] == "tool_with_attr"

    def test_raises_for_non_basenode(self):
        """Test that decorator raises TypeError for non-BaseNode classes."""
        with pytest.raises(TypeError, match="BaseNode"):
            @tool_registry(
                name="invalid_tool",
                description="Test",
                parameters={}
            )
            class NotANode:
                pass

    def test_ends_flow_default_false(self):
        """Test that ends_flow defaults to False."""
        from pocketflow import BaseNode

        @tool_registry(
            name="normal_tool",
            description="Test",
            parameters={}
        )
        class NormalTool(BaseNode):
            pass

        assert _TOOL_REGISTRY["normal_tool"]["ends_flow"] is False

    def test_ends_flow_set_true(self):
        """Test that ends_flow can be set to True."""
        from pocketflow import BaseNode

        @tool_registry(
            name="final_tool",
            description="Test",
            parameters={},
            ends_flow=True
        )
        class FinalTool(BaseNode):
            pass

        assert _TOOL_REGISTRY["final_tool"]["ends_flow"] is True

    def test_toolset_name_defaults_to_tool_name(self):
        """Test that toolset_name defaults to the tool name."""
        from pocketflow import BaseNode

        @tool_registry(
            name="standalone_tool",
            description="Test",
            parameters={}
        )
        class StandaloneTool(BaseNode):
            pass

        assert _TOOL_REGISTRY["standalone_tool"]["toolset_name"] == "standalone_tool"

    def test_custom_toolset_name(self):
        """Test setting a custom toolset name."""
        from pocketflow import BaseNode

        @tool_registry(
            name="grouped_tool",
            description="Test",
            parameters={},
            toolset_name="my_toolset"
        )
        class GroupedTool(BaseNode):
            pass

        assert _TOOL_REGISTRY["grouped_tool"]["toolset_name"] == "my_toolset"

    def test_implementation_type_is_internal(self):
        """Test that implementation_type is set to 'internal'."""
        from pocketflow import BaseNode

        @tool_registry(
            name="internal_tool",
            description="Test",
            parameters={}
        )
        class InternalTool(BaseNode):
            pass

        assert _TOOL_REGISTRY["internal_tool"]["implementation_type"] == "internal"


class TestGetRegisteredTools:
    """Tests for get_registered_tools function."""

    def test_returns_empty_list_when_no_tools(self):
        """Test returns empty list when registry is empty."""
        result = get_registered_tools()

        assert result == []

    def test_returns_all_registered_tools(self):
        """Test returns all tools in registry."""
        _TOOL_REGISTRY["tool1"] = {"name": "tool1"}
        _TOOL_REGISTRY["tool2"] = {"name": "tool2"}

        result = get_registered_tools()

        assert len(result) == 2
        names = [t["name"] for t in result]
        assert "tool1" in names
        assert "tool2" in names


class TestGetToolByName:
    """Tests for get_tool_by_name function."""

    def test_returns_tool_when_found(self):
        """Test returns tool info when name exists."""
        _TOOL_REGISTRY["my_tool"] = {"name": "my_tool", "description": "Test"}

        result = get_tool_by_name("my_tool")

        assert result["name"] == "my_tool"
        assert result["description"] == "Test"

    def test_returns_none_when_not_found(self):
        """Test returns None when tool doesn't exist."""
        result = get_tool_by_name("nonexistent")

        assert result is None


class TestGetToolNodeClass:
    """Tests for get_tool_node_class function."""

    def test_returns_node_class(self):
        """Test returns the node class for a tool."""
        class MockNode:
            pass

        _TOOL_REGISTRY["class_tool"] = {"name": "class_tool", "node_class": MockNode}

        result = get_tool_node_class("class_tool")

        assert result is MockNode

    def test_returns_none_for_missing_tool(self):
        """Test returns None when tool doesn't exist."""
        result = get_tool_node_class("nonexistent")

        assert result is None


class TestGetToolsByToolsetNames:
    """Tests for get_tools_by_toolset_names function."""

    def test_returns_empty_for_empty_list(self):
        """Test returns empty list for empty toolset names."""
        result = get_tools_by_toolset_names([])

        assert result == []

    def test_returns_tools_matching_toolset(self):
        """Test returns tools matching the specified toolsets."""
        _TOOL_REGISTRY["tool_a"] = {"name": "tool_a", "toolset_name": "set1"}
        _TOOL_REGISTRY["tool_b"] = {"name": "tool_b", "toolset_name": "set2"}
        _TOOL_REGISTRY["tool_c"] = {"name": "tool_c", "toolset_name": "set1"}

        result = get_tools_by_toolset_names(["set1"])

        assert len(result) == 2
        names = [t["name"] for t in result]
        assert "tool_a" in names
        assert "tool_c" in names
        assert "tool_b" not in names

    def test_returns_tools_from_multiple_toolsets(self):
        """Test returns tools from multiple specified toolsets."""
        _TOOL_REGISTRY["tool_a"] = {"name": "tool_a", "toolset_name": "set1"}
        _TOOL_REGISTRY["tool_b"] = {"name": "tool_b", "toolset_name": "set2"}
        _TOOL_REGISTRY["tool_c"] = {"name": "tool_c", "toolset_name": "set3"}

        result = get_tools_by_toolset_names(["set1", "set2"])

        assert len(result) == 2


class TestGetAllToolsetsWithTools:
    """Tests for get_all_toolsets_with_tools function."""

    def test_returns_empty_dict_when_no_tools(self):
        """Test returns empty dict when registry is empty."""
        result = get_all_toolsets_with_tools()

        assert result == {}

    def test_groups_tools_by_toolset(self):
        """Test groups tools by their toolset name."""
        _TOOL_REGISTRY["tool_a"] = {"name": "tool_a", "toolset_name": "set1", "description": "A"}
        _TOOL_REGISTRY["tool_b"] = {"name": "tool_b", "toolset_name": "set1", "description": "B"}
        _TOOL_REGISTRY["tool_c"] = {"name": "tool_c", "toolset_name": "set2", "description": "C"}

        result = get_all_toolsets_with_tools()

        assert "set1" in result
        assert "set2" in result
        assert len(result["set1"]) == 2
        assert len(result["set2"]) == 1


class TestFormatToolsForLlmApi:
    """Tests for format_tools_for_llm_api function."""

    def test_formats_tool_for_api(self):
        """Test formats tool in OpenAI function format."""
        tools = [{
            "name": "search",
            "description": "Search for information",
            "toolset_name": "search_tools",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }]

        result = format_tools_for_llm_api(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert "search_tools" in result[0]["function"]["description"]

    def test_sanitizes_parameters(self):
        """Test that x- prefixed keys are removed from parameters."""
        tools = [{
            "name": "tool",
            "description": "Test",
            "toolset_name": "test",
            "parameters": {
                "type": "object",
                "x-internal": "remove",
                "properties": {"arg": {"type": "string", "x-meta": True}}
            }
        }]

        result = format_tools_for_llm_api(tools)

        params = result[0]["function"]["parameters"]
        assert "x-internal" not in params
        assert "x-meta" not in params["properties"]["arg"]

    def test_appends_toolset_to_description(self):
        """Test that toolset name is appended to description."""
        tools = [{
            "name": "my_tool",
            "description": "Does something",
            "toolset_name": "my_toolset",
            "parameters": {}
        }]

        result = format_tools_for_llm_api(tools)

        assert "(Belongs to toolset: 'my_toolset')" in result[0]["function"]["description"]

    def test_handles_non_dict_parameters(self):
        """Test handling of invalid non-dict parameters."""
        tools = [{
            "name": "broken_tool",
            "description": "Test",
            "toolset_name": "test",
            "parameters": "invalid"  # Should be dict
        }]

        result = format_tools_for_llm_api(tools)

        # Should use empty object as fallback
        assert result[0]["function"]["parameters"]["type"] == "object"


class TestFormatToolsForPrompt:
    """Tests for format_tools_for_prompt function."""

    def test_formats_tools_as_markdown(self):
        """Test formats tools as Markdown text."""
        tools = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"}
        ]

        result = format_tools_for_prompt(tools)

        assert "### Registered Tools" in result
        assert "**tool1**" in result
        assert "First tool" in result
        assert "**tool2**" in result

    def test_handles_empty_list(self):
        """Test handles empty tools list."""
        result = format_tools_for_prompt([])

        assert "### Registered Tools" in result


class TestFormatToolsForPromptByToolset:
    """Tests for format_tools_for_prompt_by_toolset function."""

    def test_groups_tools_by_toolset(self):
        """Test groups tools by toolset in output."""
        tools_by_toolset = {
            "search": [{"name": "web_search", "description": "Search web"}],
            "files": [{"name": "read_file", "description": "Read a file"}]
        }

        result = format_tools_for_prompt_by_toolset(tools_by_toolset)

        assert "#### Toolset: search" in result
        assert "#### Toolset: files" in result
        assert "**web_search**" in result
        assert "**read_file**" in result

    def test_skips_empty_toolsets(self):
        """Test skips toolsets with no tools."""
        tools_by_toolset = {
            "full": [{"name": "tool", "description": "Test"}],
            "empty": []
        }

        result = format_tools_for_prompt_by_toolset(tools_by_toolset)

        assert "Toolset: full" in result
        assert "Toolset: empty" not in result


class TestFormatSimplifiedToolsForPromptByToolset:
    """Tests for format_simplified_tools_for_prompt_by_toolset function."""

    def test_includes_reference_header(self):
        """Test includes header about associate agent tools."""
        tools_by_toolset = {"test": [{"name": "tool", "description": "Desc"}]}

        result = format_simplified_tools_for_prompt_by_toolset(tools_by_toolset)

        assert "Associate Agent Available Tools Reference" in result
        assert "You cannot call these directly" in result


class TestRegisterNativeMcpTool:
    """Tests for register_native_mcp_tool function."""

    def test_registers_mcp_tool(self):
        """Test registers a native MCP tool."""
        register_native_mcp_tool(
            name="mcp_search",
            description="Search via MCP",
            parameters={"type": "object", "properties": {}},
            server_name="search_server"
        )

        # Name should be prefixed with server name
        assert "search_server_mcp_search" in _TOOL_REGISTRY
        tool = _TOOL_REGISTRY["search_server_mcp_search"]
        assert tool["original_name"] == "mcp_search"
        assert tool["implementation_type"] == "native_mcp"
        assert tool["mcp_server_name"] == "search_server"

    def test_uses_server_name_as_toolset(self):
        """Test that server name is used as toolset name."""
        register_native_mcp_tool(
            name="tool",
            description="Test",
            parameters={},
            server_name="my_server"
        )

        assert _TOOL_REGISTRY["my_server_tool"]["toolset_name"] == "my_server"

    def test_skips_invalid_parameters(self):
        """Test skips registration for non-dict parameters."""
        register_native_mcp_tool(
            name="bad_tool",
            description="Test",
            parameters="invalid",  # Not a dict
            server_name="server"
        )

        assert "server_bad_tool" not in _TOOL_REGISTRY

    def test_stores_knowledge_item_type(self):
        """Test stores default_knowledge_item_type."""
        register_native_mcp_tool(
            name="kb_tool",
            description="Test",
            parameters={},
            server_name="server",
            default_knowledge_item_type="DOCUMENT"
        )

        assert _TOOL_REGISTRY["server_kb_tool"]["default_knowledge_item_type"] == "DOCUMENT"

    def test_stores_output_field_mappings(self):
        """Test stores source_uri and title field mappings."""
        register_native_mcp_tool(
            name="output_tool",
            description="Test",
            parameters={},
            server_name="server",
            source_uri_field_in_output="url",
            title_field_in_output="name"
        )

        tool = _TOOL_REGISTRY["server_output_tool"]
        assert tool["source_uri_field_in_output"] == "url"
        assert tool["title_field_in_output"] == "name"

    def test_overwrites_existing_tool_with_warning(self):
        """Test overwrites existing tool with same name."""
        register_native_mcp_tool(
            name="dupe",
            description="First",
            parameters={},
            server_name="server"
        )
        register_native_mcp_tool(
            name="dupe",
            description="Second",
            parameters={},
            server_name="server"
        )

        assert _TOOL_REGISTRY["server_dupe"]["description"] == "Second"


class TestToolRegistryIntegration:
    """Integration tests for tool registry functionality."""

    def test_full_workflow(self):
        """Test complete workflow of registering and retrieving tools."""
        from pocketflow import BaseNode

        # Register internal tool
        @tool_registry(
            name="internal_search",
            description="Internal search",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            toolset_name="search"
        )
        class InternalSearch(BaseNode):
            pass

        # Register MCP tool
        register_native_mcp_tool(
            name="external_search",
            description="External search",
            parameters={"type": "object", "properties": {}},
            server_name="search"  # Same toolset
        )

        # Get by toolset
        tools = get_tools_by_toolset_names(["search"])
        assert len(tools) == 2

        # Format for LLM
        api_tools = format_tools_for_llm_api(tools)
        assert len(api_tools) == 2

        # Get node class
        node_class = get_tool_node_class("internal_search")
        assert node_class is InternalSearch


class TestAllowedAtCritical:
    """Tests for the allowed_at_critical tool registry parameter."""

    def test_allowed_at_critical_defaults_to_false(self):
        """Test that allowed_at_critical defaults to False."""
        from pocketflow import BaseNode

        @tool_registry(
            name="default_tool",
            description="A tool with default settings",
            parameters={"type": "object", "properties": {}}
        )
        class DefaultTool(BaseNode):
            pass

        tool = _TOOL_REGISTRY["default_tool"]
        assert tool.get("allowed_at_critical") is False

    def test_allowed_at_critical_can_be_true(self):
        """Test that allowed_at_critical can be set to True."""
        from pocketflow import BaseNode

        @tool_registry(
            name="read_only_tool",
            description="A read-only status tool",
            parameters={"type": "object", "properties": {}},
            allowed_at_critical=True
        )
        class ReadOnlyTool(BaseNode):
            pass

        tool = _TOOL_REGISTRY["read_only_tool"]
        assert tool.get("allowed_at_critical") is True

    def test_allowed_at_critical_can_be_false(self):
        """Test that allowed_at_critical can be explicitly set to False."""
        from pocketflow import BaseNode

        @tool_registry(
            name="write_tool",
            description="A write tool",
            parameters={"type": "object", "properties": {}},
            allowed_at_critical=False
        )
        class WriteTool(BaseNode):
            pass

        tool = _TOOL_REGISTRY["write_tool"]
        assert tool.get("allowed_at_critical") is False


class TestFlowTerminatingToolsCriticalAccess:
    """
    Regression tests to ensure flow-terminating tools remain available at critical budget.
    
    These tools are essential for graceful shutdown and must have allowed_at_critical=True.
    If these tests fail, agents will lose the ability to wrap up cleanly at budget limits.
    """

    def test_finish_flow_is_critical_safe(self):
        """finish_flow must remain available at CRITICAL budget for Principal graceful shutdown."""
        # Import to trigger registration (registry was cleared by fixture, re-register)
        from agent_core.nodes.custom_nodes.finish_node import FinishNode
        
        # Re-check the class's _tool_info attribute (set by decorator)
        assert hasattr(FinishNode, "_tool_info"), "FinishNode should have _tool_info from @tool_registry"
        tool_info = FinishNode._tool_info
        assert tool_info.get("allowed_at_critical") is True, (
            "finish_flow must have allowed_at_critical=True for Principal graceful shutdown. "
            "Without this, Principal cannot call finish_flow at CRITICAL/EXCEEDED budget."
        )

    def test_generate_message_summary_is_critical_safe(self):
        """generate_message_summary must remain available at CRITICAL budget for Associate graceful shutdown."""
        # Import to trigger registration
        from agent_core.nodes.custom_nodes.finish_node import GenerateMessageSummaryTool
        
        # Check the class's _tool_info attribute
        assert hasattr(GenerateMessageSummaryTool, "_tool_info"), "GenerateMessageSummaryTool should have _tool_info"
        tool_info = GenerateMessageSummaryTool._tool_info
        assert tool_info.get("allowed_at_critical") is True, (
            "generate_message_summary must have allowed_at_critical=True for Associate graceful shutdown. "
            "Without this, Associate cannot submit deliverables at CRITICAL/EXCEEDED budget."
        )

    def test_get_principal_status_is_critical_safe(self):
        """GetPrincipalStatusSummaryTool must remain available for Partner monitoring at critical budget."""
        # Import to trigger registration
        from agent_core.nodes.custom_nodes.get_principal_status_tool import GetPrincipalStatusSummaryTool
        
        # Check the class's _tool_info attribute
        assert hasattr(GetPrincipalStatusSummaryTool, "_tool_info"), "GetPrincipalStatusSummaryTool should have _tool_info"
        tool_info = GetPrincipalStatusSummaryTool._tool_info
        assert tool_info.get("allowed_at_critical") is True, (
            "GetPrincipalStatusSummaryTool must have allowed_at_critical=True for Partner status monitoring. "
            "Without this, Partner cannot check Principal status at CRITICAL/EXCEEDED budget."
        )
