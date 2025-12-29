"""
Tier 3 Unit Tests for agent_core/nodes/base_tool_node.py

Tests the BaseToolNode class and helper functions for tool execution:
- BaseToolNode: prep_async(), post_async()
- add_tool_result_to_inbox(): Helper for adding tool results
- dehydrate_payload_recursively(): Intelligent payload dehydration

Test Categories:
1. BaseToolNode Lifecycle: __init__, prep_async, post_async
2. Tool Result Inbox: add_tool_result_to_inbox()
3. Payload Dehydration: dehydrate_payload_recursively()
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import uuid
from datetime import datetime, timezone

from agent_core.nodes.base_tool_node import (
    BaseToolNode,
    add_tool_result_to_inbox,
    dehydrate_payload_recursively
)


class MockToolNode(BaseToolNode):
    """A mock implementation of BaseToolNode for testing."""

    def __init__(self, **kwargs):
        # Set _tool_info before calling super().__init__
        self._tool_info = {
            "name": "mock_tool",
            "description": "A mock tool for testing",
            "toolset": "test"
        }
        super().__init__(**kwargs)

    async def exec_async(self, prep_res):
        """Mock implementation."""
        return {
            "status": "success",
            "payload": {"result": "test result"}
        }


class TestBaseToolNodeInit:
    """Tests for BaseToolNode initialization."""

    def test_init_with_tool_info(self):
        """Test that decorated nodes initialize properly."""
        node = MockToolNode()
        assert node._tool_info["name"] == "mock_tool"

    def test_init_without_tool_info_raises(self):
        """Test that undecorated nodes raise TypeError."""
        class UnDecoratedNode(BaseToolNode):
            async def exec_async(self, prep_res):
                return {}

        with pytest.raises(TypeError, match="was not decorated with @tool_registry"):
            UnDecoratedNode()


class TestBaseToolNodePrepAsync:
    """Tests for BaseToolNode.prep_async()."""

    @pytest.mark.asyncio
    async def test_prep_extracts_tool_params(self):
        """Test that prep_async extracts parameters from current_action."""
        node = MockToolNode()
        shared = {
            "state": {
                "current_action": {
                    "type": "tool_call",
                    "tool_name": "mock_tool",
                    "tool_call_id": "tc_123",
                    "implementation_type": "custom",
                    "query": "search query",
                    "limit": 10
                }
            }
        }

        result = await node.prep_async(shared)

        assert result["tool_params"]["query"] == "search query"
        assert result["tool_params"]["limit"] == 10
        # Reserved keywords should be excluded
        assert "type" not in result["tool_params"]
        assert "tool_name" not in result["tool_params"]
        assert "tool_call_id" not in result["tool_params"]
        assert "implementation_type" not in result["tool_params"]

    @pytest.mark.asyncio
    async def test_prep_returns_shared_context(self):
        """Test that prep_async includes shared_context in result."""
        node = MockToolNode()
        shared = {"state": {"current_action": {}}, "refs": {}}

        result = await node.prep_async(shared)

        assert result["shared_context"] is shared

    @pytest.mark.asyncio
    async def test_prep_handles_empty_state(self):
        """Test prep_async with minimal shared dict."""
        node = MockToolNode()
        shared = {}

        result = await node.prep_async(shared)

        assert result["tool_params"] == {}
        assert result["shared_context"] is shared


class TestBaseToolNodePostAsync:
    """Tests for BaseToolNode.post_async()."""

    @pytest.mark.asyncio
    async def test_post_adds_success_result_to_inbox(self):
        """Test that successful results are added to inbox."""
        node = MockToolNode()
        shared = {
            "state": {
                "current_action": {},
                "current_tool_call_id": "tc_456",
                "inbox": []
            },
            "refs": {"run": {"runtime": {}}}
        }
        prep_res = {"tool_params": {}, "shared_context": shared}
        exec_res = {
            "status": "success",
            "payload": {"answer": "42"}
        }

        result = await node.post_async(shared, prep_res, exec_res)

        assert result == "default"
        assert len(shared["state"]["inbox"]) == 1
        inbox_item = shared["state"]["inbox"][0]
        assert inbox_item["source"] == "TOOL_RESULT"
        assert inbox_item["payload"]["tool_name"] == "mock_tool"
        assert inbox_item["payload"]["is_error"] is False

    @pytest.mark.asyncio
    async def test_post_adds_error_result_to_inbox(self):
        """Test that error results are added to inbox with error content."""
        node = MockToolNode()
        shared = {
            "state": {
                "current_action": {},
                "current_tool_call_id": "tc_789",
                "inbox": []
            },
            "refs": {"run": {"runtime": {}}}
        }
        prep_res = {"tool_params": {}, "shared_context": shared}
        exec_res = {
            "status": "error",
            "payload": None,
            "error_message": "Something went wrong"
        }

        await node.post_async(shared, prep_res, exec_res)

        inbox_item = shared["state"]["inbox"][0]
        assert inbox_item["payload"]["is_error"] is True
        assert inbox_item["payload"]["content"]["error"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_post_clears_current_action(self):
        """Test that post_async clears current_action."""
        node = MockToolNode()
        shared = {
            "state": {
                "current_action": {"tool_name": "mock_tool"},
                "inbox": []
            },
            "refs": {"run": {"runtime": {}}}
        }
        prep_res = {"tool_params": {}, "shared_context": shared}
        exec_res = {"status": "success", "payload": {}}

        await node.post_async(shared, prep_res, exec_res)

        assert shared["state"]["current_action"] is None

    @pytest.mark.asyncio
    async def test_post_handles_knowledge_base_items(self):
        """Test that _knowledge_items_to_add are processed."""
        node = MockToolNode()
        mock_kb = AsyncMock()

        shared = {
            "state": {
                "current_action": {},
                "current_tool_call_id": "tc_kb",
                "inbox": []
            },
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "meta": {"agent_id": "test_agent"}
        }
        prep_res = {"tool_params": {}, "shared_context": shared}
        exec_res = {
            "status": "success",
            "payload": {},
            "_knowledge_items_to_add": [
                {"content": "fact 1", "metadata": {}},
                {"content": "fact 2", "metadata": {"extra": "data"}}
            ]
        }

        await node.post_async(shared, prep_res, exec_res)

        assert mock_kb.add_item.call_count == 2
        # Check metadata was added
        first_call = mock_kb.add_item.call_args_list[0][0][0]
        assert first_call["metadata"]["source_tool_name"] == "mock_tool"

    @pytest.mark.asyncio
    async def test_post_skips_kb_on_error(self):
        """Test that knowledge base items are not added on error."""
        node = MockToolNode()
        mock_kb = AsyncMock()

        shared = {
            "state": {"current_action": {}, "inbox": []},
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}}
        }
        prep_res = {"tool_params": {}, "shared_context": shared}
        exec_res = {
            "status": "error",
            "payload": None,
            "_knowledge_items_to_add": [{"content": "should not add"}]
        }

        await node.post_async(shared, prep_res, exec_res)

        mock_kb.add_item.assert_not_called()


class TestAddToolResultToInbox:
    """Tests for the add_tool_result_to_inbox helper."""

    @pytest.mark.asyncio
    async def test_adds_result_with_all_fields(self):
        """Test adding a complete tool result."""
        state = {"inbox": []}

        await add_tool_result_to_inbox(
            state=state,
            tool_name="test_tool",
            tool_call_id="tc_001",
            is_error=False,
            content={"result": "success"}
        )

        assert len(state["inbox"]) == 1
        item = state["inbox"][0]
        assert item["source"] == "TOOL_RESULT"
        assert item["payload"]["tool_name"] == "test_tool"
        assert item["payload"]["tool_call_id"] == "tc_001"
        assert item["payload"]["is_error"] is False
        assert item["payload"]["content"] == {"result": "success"}
        assert item["consumption_policy"] == "consume_on_read"
        assert "created_at" in item["metadata"]

    @pytest.mark.asyncio
    async def test_creates_inbox_if_missing(self):
        """Test that inbox is created if not present."""
        state = {}

        await add_tool_result_to_inbox(
            state=state,
            tool_name="tool",
            tool_call_id=None,
            is_error=False,
            content="test"
        )

        assert "inbox" in state
        assert len(state["inbox"]) == 1

    @pytest.mark.asyncio
    async def test_appends_to_existing_inbox(self):
        """Test that results are appended to existing inbox."""
        state = {"inbox": [{"existing": "item"}]}

        await add_tool_result_to_inbox(
            state=state,
            tool_name="tool",
            tool_call_id="tc",
            is_error=False,
            content="new"
        )

        assert len(state["inbox"]) == 2
        assert state["inbox"][0]["existing"] == "item"

    @pytest.mark.asyncio
    async def test_handles_error_result(self):
        """Test adding an error result."""
        state = {"inbox": []}

        await add_tool_result_to_inbox(
            state=state,
            tool_name="failing_tool",
            tool_call_id="tc_fail",
            is_error=True,
            content={"error": "Operation failed"}
        )

        item = state["inbox"][0]
        assert item["payload"]["is_error"] is True

    @pytest.mark.asyncio
    async def test_item_id_is_unique(self):
        """Test that each item gets a unique ID."""
        state = {"inbox": []}

        await add_tool_result_to_inbox(state, "t1", "tc1", False, "c1")
        await add_tool_result_to_inbox(state, "t2", "tc2", False, "c2")

        ids = [item["item_id"] for item in state["inbox"]]
        assert ids[0] != ids[1]
        assert all(id.startswith("inbox_") for id in ids)


class TestDehydratePayloadRecursively:
    """Tests for the dehydrate_payload_recursively function."""

    @pytest.mark.asyncio
    async def test_no_kb_returns_data_unchanged(self):
        """Test that without KB, data is returned unchanged."""
        context = {"refs": {"run": {"runtime": {}}}}
        tool_info = {"name": "test_tool"}
        data = {"key": "value", "nested": {"inner": "data"}}

        result = await dehydrate_payload_recursively(data, context, tool_info)

        assert result == data

    @pytest.mark.asyncio
    async def test_small_data_not_dehydrated(self):
        """Test that small data is not dehydrated."""
        mock_kb = AsyncMock()
        mock_kb.store_with_token = AsyncMock(return_value="<#CGKB-token>")

        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {}
        }
        tool_info = {"name": "test_tool"}
        data = {"small": "data"}  # Well under 1KB

        result = await dehydrate_payload_recursively(data, context, tool_info)

        assert result == data
        mock_kb.store_with_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_string_dehydrated(self):
        """Test that large strings are dehydrated."""
        mock_kb = AsyncMock()
        mock_kb.store_with_token = AsyncMock(return_value="<#CGKB-bigstring>")

        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {"current_tool_call_id": "tc_str"}
        }
        tool_info = {"name": "test_tool"}
        large_string = "x" * 2000  # Over 1KB

        result = await dehydrate_payload_recursively(large_string, context, tool_info)

        assert result == "<#CGKB-bigstring>"
        mock_kb.store_with_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_dict_value_dehydrated(self):
        """Test that large dict values are dehydrated."""
        mock_kb = AsyncMock()
        mock_kb.store_with_token = AsyncMock(return_value="<#CGKB-largevalue>")

        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {}
        }
        tool_info = {"name": "test_tool"}
        data = {
            "small_key": "small_value",
            "large_key": "y" * 2000
        }

        result = await dehydrate_payload_recursively(data, context, tool_info)

        assert result["small_key"] == "small_value"
        assert result["large_key"] == "<#CGKB-largevalue>"

    @pytest.mark.asyncio
    async def test_large_list_item_dehydrated(self):
        """Test that large list items are dehydrated."""
        mock_kb = AsyncMock()
        mock_kb.store_with_token = AsyncMock(return_value="<#CGKB-listitem>")

        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {}
        }
        tool_info = {"name": "test_tool"}
        data = ["small", "z" * 2000, "also small"]

        result = await dehydrate_payload_recursively(data, context, tool_info)

        assert result[0] == "small"
        assert result[1] == "<#CGKB-listitem>"
        assert result[2] == "also small"

    @pytest.mark.asyncio
    async def test_nested_dehydration(self):
        """Test that nested structures are processed recursively."""
        mock_kb = AsyncMock()
        mock_kb.store_with_token = AsyncMock(return_value="<#CGKB-nested>")

        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {}
        }
        tool_info = {"name": "test_tool"}
        # The dehydration checks individual key-value pairs; make inner content large enough
        data = {
            "level1": {
                "level2": {
                    "large_content": "a" * 2000
                }
            }
        }

        result = await dehydrate_payload_recursively(data, context, tool_info)

        # The dehydration may occur at different levels depending on size calculation
        # Check that store_with_token was called at least once
        assert mock_kb.store_with_token.called

    @pytest.mark.asyncio
    async def test_metadata_includes_tool_info(self):
        """Test that dehydration metadata includes tool info."""
        mock_kb = AsyncMock()
        stored_metadata = None

        async def capture_metadata(content, metadata):
            nonlocal stored_metadata
            stored_metadata = metadata
            return "<#CGKB-captured>"

        mock_kb.store_with_token = capture_metadata

        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {"current_tool_call_id": "tc_meta"}
        }
        tool_info = {"name": "capture_tool"}
        data = "b" * 2000

        await dehydrate_payload_recursively(data, context, tool_info)

        assert stored_metadata["item_type"] == "DEHYDRATED_TOOL_PAYLOAD_PART"
        assert stored_metadata["source_tool_name"] == "capture_tool"
        assert stored_metadata["tool_call_id"] == "tc_meta"
        assert stored_metadata["original_path"] == "payload"

    @pytest.mark.asyncio
    async def test_primitives_unchanged(self):
        """Test that primitive types are returned unchanged."""
        mock_kb = AsyncMock()
        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {}
        }
        tool_info = {"name": "test_tool"}

        assert await dehydrate_payload_recursively(42, context, tool_info) == 42
        assert await dehydrate_payload_recursively(3.14, context, tool_info) == 3.14
        assert await dehydrate_payload_recursively(True, context, tool_info) is True
        assert await dehydrate_payload_recursively(None, context, tool_info) is None
        assert await dehydrate_payload_recursively("short", context, tool_info) == "short"

    @pytest.mark.asyncio
    async def test_empty_structures(self):
        """Test handling of empty structures."""
        mock_kb = AsyncMock()
        context = {
            "refs": {"run": {"runtime": {"knowledge_base": mock_kb}}},
            "state": {}
        }
        tool_info = {"name": "test_tool"}

        assert await dehydrate_payload_recursively({}, context, tool_info) == {}
        assert await dehydrate_payload_recursively([], context, tool_info) == []


class TestBaseToolNodeExecAsync:
    """Tests for the abstract exec_async method."""

    @pytest.mark.asyncio
    async def test_exec_must_be_overridden(self):
        """Test that exec_async raises NotImplementedError in base class."""
        class PartialNode(BaseToolNode):
            def __init__(self):
                self._tool_info = {"name": "partial"}
                super().__init__()

            # Deliberately NOT implementing exec_async

        # Create a class that inherits but doesn't implement
        class NoExecNode(PartialNode):
            pass

        node = NoExecNode()
        with pytest.raises(NotImplementedError):
            await node.exec_async({})

    @pytest.mark.asyncio
    async def test_exec_implementation_called(self):
        """Test that implemented exec_async is called correctly."""
        exec_called = False

        class ImplementedNode(BaseToolNode):
            def __init__(self):
                self._tool_info = {"name": "implemented"}
                super().__init__()

            async def exec_async(self, prep_res):
                nonlocal exec_called
                exec_called = True
                return {"status": "success", "payload": prep_res.get("tool_params")}

        node = ImplementedNode()
        result = await node.exec_async({"tool_params": {"key": "value"}})

        assert exec_called
        assert result["status"] == "success"
        assert result["payload"] == {"key": "value"}
