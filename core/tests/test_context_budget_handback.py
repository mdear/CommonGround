"""
Unit tests for context_budget_handback module.

Tests the ContextBudgetHandback dataclass and helper functions for
packaging partial work when subagents exceed context budget.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from agent_core.framework.context_budget_handback import (
    ContextBudgetHandback,
    build_handback_from_context,
    notify_principal_of_handback
)


class TestContextBudgetHandback:
    """Tests for ContextBudgetHandback dataclass."""
    
    def test_handback_creation_with_defaults(self):
        """Test creating a handback with minimal required fields."""
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="Associate_WebSearcher",
            utilization_percent=95.5,
            predicted_tokens=190000,
            context_limit=200000
        )
        
        assert handback.agent_id == "test_agent"
        assert handback.module_id == "WM_1"
        assert handback.utilization_percent == 95.5
        assert handback.kb_tokens == []
        assert handback.kb_token_count == 0
        assert handback.tool_calls_completed == []
    
    def test_handback_creation_with_all_fields(self):
        """Test creating a handback with all fields populated."""
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="Associate_WebSearcher",
            utilization_percent=95.5,
            predicted_tokens=190000,
            context_limit=200000,
            kb_tokens=["<#CGKB-00001>", "<#CGKB-00002>"],
            kb_token_count=2,
            estimated_kb_content_tokens=5000,
            tool_calls_completed=[{"tool": "web_search", "arguments_preview": "test query"}],
            tool_calls_in_progress={"tool": "visit_url"},
            last_assistant_content_preview="Here are my findings...",
            turns_completed=5,
            start_timestamp="2025-12-30T10:00:00Z",
            overflow_timestamp="2025-12-30T11:00:00Z"
        )
        
        assert handback.kb_token_count == 2
        assert len(handback.tool_calls_completed) == 1
        assert handback.turns_completed == 5
    
    def test_to_dict(self):
        """Test converting handback to dictionary."""
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="Associate_WebSearcher",
            utilization_percent=95.5,
            predicted_tokens=190000,
            context_limit=200000
        )
        
        result = handback.to_dict()
        
        assert isinstance(result, dict)
        assert result["agent_id"] == "test_agent"
        assert result["utilization_percent"] == 95.5
    
    def test_from_dict(self):
        """Test creating handback from dictionary."""
        data = {
            "agent_id": "test_agent",
            "module_id": "WM_1",
            "profile_name": "Associate_WebSearcher",
            "utilization_percent": 95.5,
            "predicted_tokens": 190000,
            "context_limit": 200000,
            "kb_tokens": ["<#CGKB-00001>"],
            "kb_token_count": 1,
            "estimated_kb_content_tokens": 2500,
            "tool_calls_completed": [],
            "tool_calls_in_progress": None,
            "last_assistant_content_preview": "",
            "turns_completed": 3,
            "start_timestamp": "",
            "overflow_timestamp": ""
        }
        
        handback = ContextBudgetHandback.from_dict(data)
        
        assert handback.agent_id == "test_agent"
        assert handback.kb_token_count == 1
        assert handback.turns_completed == 3
    
    def test_get_principal_summary_prompt(self):
        """Test generating summary prompt for Principal."""
        handback = ContextBudgetHandback(
            agent_id="Assoc_WebSearche_4",
            module_id="WM_4",
            profile_name="Associate_WebSearcher_Academic",
            utilization_percent=94.5,
            predicted_tokens=189017,
            context_limit=200000,
            kb_tokens=["<#CGKB-00029>", "<#CGKB-00030>", "<#CGKB-00031>"],
            kb_token_count=3,
            estimated_kb_content_tokens=15000,
            tool_calls_completed=[
                {"tool": "web_search", "arguments_preview": "palliative care GP"},
                {"tool": "visit_url", "arguments_preview": "https://example.com"}
            ],
            turns_completed=5
        )
        
        prompt = handback.get_principal_summary_prompt()
        
        assert "Assoc_WebSearche_4" in prompt
        assert "WM_4" in prompt
        assert "94.5%" in prompt
        assert "<#CGKB-00029>" in prompt
        assert "web_search" in prompt
        assert "Summarization Required" in prompt
    
    def test_get_principal_summary_prompt_truncates_long_kb_list(self):
        """Test that KB token list is truncated if too long."""
        kb_tokens = [f"<#CGKB-{i:05d}>" for i in range(50)]
        
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="test_profile",
            utilization_percent=95.0,
            predicted_tokens=190000,
            context_limit=200000,
            kb_tokens=kb_tokens,
            kb_token_count=50
        )
        
        prompt = handback.get_principal_summary_prompt()
        
        # Should show first 20 and indicate more
        assert "<#CGKB-00000>" in prompt
        assert "<#CGKB-00019>" in prompt
        assert "+30 more" in prompt
    
    def test_get_deliverables_summary(self):
        """Test generating deliverables summary string."""
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="test_profile",
            utilization_percent=95.0,
            predicted_tokens=190000,
            context_limit=200000,
            kb_tokens=["<#CGKB-00001>", "<#CGKB-00002>"],
            kb_token_count=2
        )
        
        summary = handback.get_deliverables_summary()
        
        assert "CONTEXT BUDGET EXCEEDED" in summary
        assert "test_agent" in summary
        assert "2 KB items" in summary
        assert "95.0%" in summary


class TestBuildHandbackFromContext:
    """Tests for build_handback_from_context function."""
    
    def test_build_handback_minimal_context(self):
        """Test building handback from minimal context."""
        context = {
            "state": {
                "messages": [
                    {"role": "user", "content": "Search for X"},
                    {"role": "assistant", "content": "I will search for X"}
                ],
                "_context_budget": {
                    "utilization_percent": 95.0,
                    "context_limit": 200000
                }
            },
            "meta": {
                "agent_id": "test_agent",
                "module_id": "WM_1"
            },
            "refs": {
                "run": {
                    "runtime": {}
                }
            }
        }
        
        prep_res = {
            "predicted_total_tokens": 190000
        }
        
        handback = build_handback_from_context(
            agent_id="test_agent",
            context=context,
            prep_res=prep_res,
            profile_name="test_profile"
        )
        
        assert handback.agent_id == "test_agent"
        assert handback.utilization_percent == 95.0
        assert handback.turns_completed == 1  # One assistant message
    
    def test_build_handback_with_tool_calls(self):
        """Test building handback captures tool call history."""
        context = {
            "state": {
                "messages": [
                    {"role": "user", "content": "Search for X"},
                    {
                        "role": "assistant", 
                        "content": "I will search",
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {"name": "web_search", "arguments": '{"query": "test"}'}
                            }
                        ]
                    },
                    {"role": "tool", "content": "Results..."},
                    {"role": "assistant", "content": "Based on results..."}
                ],
                "_context_budget": {
                    "utilization_percent": 95.0,
                    "context_limit": 200000
                }
            },
            "meta": {"agent_id": "test_agent"},
            "refs": {"run": {"runtime": {}}}
        }
        
        prep_res = {"predicted_total_tokens": 190000}
        
        handback = build_handback_from_context(
            agent_id="test_agent",
            context=context,
            prep_res=prep_res,
            profile_name="test_profile"
        )
        
        assert len(handback.tool_calls_completed) == 1
        assert handback.tool_calls_completed[0]["tool"] == "web_search"
        assert handback.turns_completed == 2  # Two assistant messages
    
    def test_build_handback_extracts_last_content(self):
        """Test that last assistant content is captured."""
        context = {
            "state": {
                "messages": [
                    {"role": "user", "content": "Search for X"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "user", "content": "Continue"},
                    {"role": "assistant", "content": "This is my latest analysis of the findings..."}
                ],
                "_context_budget": {"utilization_percent": 95.0}
            },
            "meta": {"agent_id": "test_agent"},
            "refs": {"run": {"runtime": {}}}
        }
        
        prep_res = {"predicted_total_tokens": 190000}
        
        handback = build_handback_from_context(
            agent_id="test_agent",
            context=context,
            prep_res=prep_res,
            profile_name="test_profile"
        )
        
        assert "latest analysis" in handback.last_assistant_content_preview


class TestNotifyPrincipalOfHandback:
    """Tests for notify_principal_of_handback function."""
    
    def test_notify_adds_to_inbox(self):
        """Test that notification is added to Principal's inbox."""
        principal_context = {
            "state": {
                "inbox": []
            }
        }
        
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="test_profile",
            utilization_percent=95.0,
            predicted_tokens=190000,
            context_limit=200000,
            kb_tokens=["<#CGKB-00001>"],
            kb_token_count=1
        )
        
        notify_principal_of_handback(principal_context, handback)
        
        assert len(principal_context["state"]["inbox"]) == 1
        
        notification = principal_context["state"]["inbox"][0]
        assert notification["source"] == "CONTEXT_BUDGET_HANDBACK"
        assert "test_agent" in notification["payload"]["message"]
        assert notification["metadata"]["priority"] == "high"
    
    def test_notify_creates_inbox_if_missing(self):
        """Test that inbox is created if not present."""
        principal_context = {
            "state": {}
        }
        
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="test_profile",
            utilization_percent=95.0,
            predicted_tokens=190000,
            context_limit=200000
        )
        
        notify_principal_of_handback(principal_context, handback)
        
        assert "inbox" in principal_context["state"]
        assert len(principal_context["state"]["inbox"]) == 1
    
    def test_notify_includes_handback_data(self):
        """Test that full handback data is included in notification."""
        principal_context = {"state": {"inbox": []}}
        
        handback = ContextBudgetHandback(
            agent_id="test_agent",
            module_id="WM_1",
            profile_name="test_profile",
            utilization_percent=95.0,
            predicted_tokens=190000,
            context_limit=200000,
            kb_tokens=["<#CGKB-00001>", "<#CGKB-00002>"],
            kb_token_count=2
        )
        
        notify_principal_of_handback(principal_context, handback)
        
        notification = principal_context["state"]["inbox"][0]
        handback_in_payload = notification["payload"]["handback"]
        
        assert handback_in_payload["agent_id"] == "test_agent"
        assert handback_in_payload["kb_token_count"] == 2
