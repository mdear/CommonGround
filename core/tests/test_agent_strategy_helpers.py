"""
Unit tests for agent_core/framework/agent_strategy_helpers.py

Tests tool filtering at critical budget thresholds.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent_core.framework.agent_strategy_helpers import (
    filter_tools_for_critical_budget,
    get_formatted_api_tools
)


class TestFilterToolsForCriticalBudget:
    """Tests for the filter_tools_for_critical_budget function."""

    def test_filters_out_tools_not_allowed_at_critical(self):
        """Test that tools without allowed_at_critical=True are filtered out."""
        tools = [
            {"name": "ReadOnlyTool", "allowed_at_critical": True, "description": "Read status"},
            {"name": "WriteTool", "allowed_at_critical": False, "description": "Launch something"},
            {"name": "DefaultTool", "description": "No flag set"},  # Missing flag defaults to filter out
        ]
        
        result = filter_tools_for_critical_budget(tools, "test_agent")
        
        assert len(result) == 1
        assert result[0]["name"] == "ReadOnlyTool"

    def test_keeps_all_tools_with_allowed_at_critical_true(self):
        """Test that all tools with allowed_at_critical=True are kept."""
        tools = [
            {"name": "StatusTool", "allowed_at_critical": True},
            {"name": "MonitorTool", "allowed_at_critical": True},
            {"name": "QueryTool", "allowed_at_critical": True},
        ]
        
        result = filter_tools_for_critical_budget(tools, "test_agent")
        
        assert len(result) == 3

    def test_returns_empty_list_when_no_tools_allowed(self):
        """Test returns empty list when no tools have allowed_at_critical=True."""
        tools = [
            {"name": "WriteTool", "allowed_at_critical": False},
            {"name": "LaunchTool", "allowed_at_critical": False},
        ]
        
        result = filter_tools_for_critical_budget(tools, "test_agent")
        
        assert len(result) == 0

    def test_handles_empty_tool_list(self):
        """Test handling of empty tool list."""
        result = filter_tools_for_critical_budget([], "test_agent")
        
        assert result == []

    def test_missing_flag_treated_as_false(self):
        """Test that missing allowed_at_critical flag is treated as False."""
        tools = [
            {"name": "NoFlagTool", "description": "Has no flag"},
        ]

        result = filter_tools_for_critical_budget(tools, "test_agent")

        assert len(result) == 0

    def test_preserves_tool_dict_structure(self):
        """Test that filtered tools retain all their original properties."""
        tools = [
            {
                "name": "StatusTool",
                "allowed_at_critical": True,
                "description": "Check status",
                "parameters": {"type": "object"},
                "toolset_name": "monitoring",
            },
        ]

        result = filter_tools_for_critical_budget(tools, "test_agent")

        assert len(result) == 1
        assert result[0]["description"] == "Check status"
        assert result[0]["parameters"] == {"type": "object"}
        assert result[0]["toolset_name"] == "monitoring"


class TestGetFormattedApiToolsWithBudgetRestriction:
    """Tests for get_formatted_api_tools budget-aware filtering."""

    @pytest.fixture
    def mock_agent_node(self):
        """Create a mock agent node instance."""
        node = MagicMock()
        node.agent_id = "Partner_Test"
        node.loaded_profile = {
            "profile_id": "test_profile",
            "tool_access_policy": {
                "allowed_individual_tools": ["TestTool"]
            }
        }
        return node

    @pytest.fixture
    def mock_tools(self):
        """Mock tool definitions."""
        return [
            {"name": "ReadTool", "allowed_at_critical": True, "description": "Read"},
            {"name": "WriteTool", "allowed_at_critical": False, "description": "Write"},
        ]

    def test_no_filtering_at_healthy_budget(self, mock_agent_node, mock_tools):
        """Test no filtering when budget is HEALTHY."""
        context = {
            "state": {
                "_context_budget": {"status": "HEALTHY"}
            }
        }
        
        with patch('agent_core.framework.agent_strategy_helpers.get_tools_for_profile', return_value=mock_tools):
            with patch('agent_core.framework.agent_strategy_helpers.format_tools_for_llm_api', side_effect=lambda x: x):
                result = get_formatted_api_tools(mock_agent_node, context)
        
        assert len(result) == 2

    def test_no_filtering_at_warning_budget(self, mock_agent_node, mock_tools):
        """Test no filtering when budget is WARNING."""
        context = {
            "state": {
                "_context_budget": {"status": "WARNING"}
            }
        }
        
        with patch('agent_core.framework.agent_strategy_helpers.get_tools_for_profile', return_value=mock_tools):
            with patch('agent_core.framework.agent_strategy_helpers.format_tools_for_llm_api', side_effect=lambda x: x):
                result = get_formatted_api_tools(mock_agent_node, context)
        
        assert len(result) == 2

    def test_filtering_at_critical_budget(self, mock_agent_node, mock_tools):
        """Test filtering when budget is CRITICAL."""
        context = {
            "state": {
                "_context_budget": {"status": "CRITICAL"}
            }
        }
        
        with patch('agent_core.framework.agent_strategy_helpers.get_tools_for_profile', return_value=mock_tools):
            with patch('agent_core.framework.agent_strategy_helpers.format_tools_for_llm_api', side_effect=lambda x: x):
                result = get_formatted_api_tools(mock_agent_node, context)
        
        assert len(result) == 1
        assert result[0]["name"] == "ReadTool"

    def test_filtering_at_exceeded_budget(self, mock_agent_node, mock_tools):
        """Test filtering when budget is EXCEEDED."""
        context = {
            "state": {
                "_context_budget": {"status": "EXCEEDED"}
            }
        }
        
        with patch('agent_core.framework.agent_strategy_helpers.get_tools_for_profile', return_value=mock_tools):
            with patch('agent_core.framework.agent_strategy_helpers.format_tools_for_llm_api', side_effect=lambda x: x):
                result = get_formatted_api_tools(mock_agent_node, context)
        
        assert len(result) == 1
        assert result[0]["name"] == "ReadTool"

    def test_handles_missing_budget_info(self, mock_agent_node, mock_tools):
        """Test no filtering when budget info is missing (defaults to HEALTHY)."""
        context = {
            "state": {}
        }

        with patch('agent_core.framework.agent_strategy_helpers.get_tools_for_profile', return_value=mock_tools):
            with patch('agent_core.framework.agent_strategy_helpers.format_tools_for_llm_api', side_effect=lambda x: x):
                result = get_formatted_api_tools(mock_agent_node, context)

        assert len(result) == 2

    def test_handles_missing_state_key(self, mock_agent_node, mock_tools):
        """Test no filtering when state key is missing entirely."""
        context = {}

        with patch('agent_core.framework.agent_strategy_helpers.get_tools_for_profile', return_value=mock_tools):
            with patch('agent_core.framework.agent_strategy_helpers.format_tools_for_llm_api', side_effect=lambda x: x):
                result = get_formatted_api_tools(mock_agent_node, context)

        assert len(result) == 2


class TestPartnerToolRestrictionScenarios:
    """Integration-style tests for Partner tool restriction at critical budget."""

    def test_partner_scenario_at_critical_threshold(self):
        """
        Test realistic Partner scenario at CRITICAL threshold.
        
        Partner should only have access to read-only tools like
        GetPrincipalStatusSummaryTool, not write tools like
        LaunchPrincipalExecutionTool or SendDirectiveToPrincipalTool.
        """
        partner_tools = [
            {"name": "GetPrincipalStatusSummaryTool", "allowed_at_critical": True, 
             "description": "Read status", "toolset_name": "monitoring_tools"},
            {"name": "LaunchPrincipalExecutionTool", "allowed_at_critical": False,
             "description": "Launch principal", "toolset_name": "LaunchPrincipalExecutionTool"},
            {"name": "SendDirectiveToPrincipalTool", "allowed_at_critical": False,
             "description": "Send directive", "toolset_name": "SendDirectiveToPrincipalTool"},
        ]
        
        result = filter_tools_for_critical_budget(partner_tools, "Partner_Test")

        # Only status tool should remain
        assert len(result) == 1
        assert result[0]["name"] == "GetPrincipalStatusSummaryTool"
