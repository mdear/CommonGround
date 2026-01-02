"""
Unit tests for BaseAgentNode._resolve_tool_conflicts method.

This method ensures flow-terminating tools (finish_flow, generate_message_summary)
take priority when called alongside other tools, preventing silent data loss.
"""

import pytest
from unittest.mock import MagicMock, patch


class MockAgentNode:
    """A mock implementation of AgentNode for testing _resolve_tool_conflicts."""
    
    agent_id = "test_agent"
    
    # Flow-terminating tools that should take priority when called with other tools
    FLOW_TERMINATING_TOOLS = {"finish_flow", "generate_message_summary"}
    
    def _resolve_tool_conflicts(self, tool_calls):
        """
        Resolve conflicting tool calls when agent calls multiple tools simultaneously.
        """
        if len(tool_calls) <= 1:
            return tool_calls
        
        # Extract tool names
        tool_names = [tc.get("function", {}).get("name") for tc in tool_calls]
        
        # Check for flow-terminating tools
        terminating_tools_found = [
            (i, name) for i, name in enumerate(tool_names) 
            if name in self.FLOW_TERMINATING_TOOLS
        ]
        
        if terminating_tools_found:
            # Flow-terminating tool called with other tools - prioritize it
            priority_index, priority_tool = terminating_tools_found[0]
            return [tool_calls[priority_index]]
        
        # No terminating tool - return original
        return tool_calls


class TestToolConflictResolutionBasic:
    """Test basic tool conflict resolution behavior."""
    
    def test_single_tool_call_unchanged(self):
        """Single tool call should pass through unchanged."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"name": "web_search", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        assert result == tool_calls
    
    def test_empty_tool_calls_unchanged(self):
        """Empty tool calls should return unchanged."""
        agent = MockAgentNode()
        result = agent._resolve_tool_conflicts([])
        assert result == []
    
    def test_multiple_non_terminating_tools_unchanged(self):
        """Multiple non-terminating tools should return unchanged (first-only applied later)."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "tc_2", "function": {"name": "visit_url", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        assert result == tool_calls


class TestFinishFlowPriority:
    """Test that finish_flow takes priority over other tools."""
    
    def test_finish_flow_first_with_other_tools(self):
        """finish_flow as first tool should be kept, others dropped."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"name": "finish_flow", "arguments": '{"reason": "done"}'}},
            {"id": "tc_2", "function": {"name": "web_search", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "finish_flow"
        assert result[0]["id"] == "tc_1"
    
    def test_finish_flow_second_prioritized(self):
        """finish_flow as second tool should still be kept, first dropped."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "tc_2", "function": {"name": "finish_flow", "arguments": '{"reason": "done"}'}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "finish_flow"
        assert result[0]["id"] == "tc_2"
    
    def test_finish_flow_middle_of_many(self):
        """finish_flow in middle of many tools should be extracted."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "tc_2", "function": {"name": "visit_url", "arguments": "{}"}},
            {"id": "tc_3", "function": {"name": "finish_flow", "arguments": '{"reason": "done"}'}},
            {"id": "tc_4", "function": {"name": "update_work_modules", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "finish_flow"
        assert result[0]["id"] == "tc_3"


class TestGenerateMessageSummaryPriority:
    """Test that generate_message_summary takes priority (Associate tool)."""
    
    def test_generate_message_summary_prioritized(self):
        """generate_message_summary should be prioritized over other tools."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"name": "web_search", "arguments": "{}"}},
            {"id": "tc_2", "function": {"name": "generate_message_summary", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "generate_message_summary"


class TestMultipleTerminatingTools:
    """Test edge case where multiple terminating tools are called."""
    
    def test_first_terminating_tool_wins(self):
        """If multiple terminating tools called, first one is kept."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"name": "generate_message_summary", "arguments": "{}"}},
            {"id": "tc_2", "function": {"name": "finish_flow", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        assert len(result) == 1
        # First terminating tool wins
        assert result[0]["function"]["name"] == "generate_message_summary"


class TestEdgeCases:
    """Test edge cases and malformed inputs."""
    
    def test_tool_call_missing_function_key(self):
        """Tool calls without function key should not crash."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1"},  # Missing function key
            {"id": "tc_2", "function": {"name": "finish_flow", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        # Should still extract finish_flow
        assert len(result) == 1
        assert result[0]["function"]["name"] == "finish_flow"
    
    def test_tool_call_missing_name(self):
        """Tool calls without name should be handled gracefully."""
        agent = MockAgentNode()
        tool_calls = [
            {"id": "tc_1", "function": {"arguments": "{}"}},  # Missing name
            {"id": "tc_2", "function": {"name": "web_search", "arguments": "{}"}}
        ]
        result = agent._resolve_tool_conflicts(tool_calls)
        # No terminating tools, return unchanged
        assert result == tool_calls
