"""
Unit tests for context_admission_controller module.

Tests the pre-admission budget enforcement that prevents context spikes
by truncating oversized tool results before they enter the context.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent_core.framework.context_admission_controller import (
    AdmissionDecision,
    check_pre_admission,
    calculate_admission_budget,
    estimate_tokens,
    estimate_result_tokens,
    _score_and_sort_kb_items,
    _serialize_payload,
    ADMISSION_TARGET_UTILIZATION,
    MIN_ADMISSION_TOKENS,
    WARNING_THRESHOLD
)


class TestEstimateTokens:
    """Tests for estimate_tokens helper function."""
    
    def test_empty_string_returns_zero(self):
        """Test empty string returns zero tokens."""
        assert estimate_tokens("") == 0
    
    def test_counts_tokens_approximately(self):
        """Test token counting gives reasonable estimates."""
        text = "This is a test sentence with several words."
        count = estimate_tokens(text)
        # ~44 chars / 4 = ~11 tokens
        assert count > 5
        assert count < 20
    
    def test_counts_long_text(self):
        """Test counting works for longer text."""
        text = "word " * 1000  # 5000 chars
        count = estimate_tokens(text)
        # ~5000 chars / 4 = ~1250 tokens
        assert count > 1000
        assert count < 1500
    
    def test_minimum_one_token(self):
        """Test that even single chars get at least 1 token."""
        assert estimate_tokens("a") >= 1


class TestAdmissionDecision:
    """Tests for AdmissionDecision dataclass."""
    
    def test_decision_creation_full_admission(self):
        """Test creating a decision for full admission."""
        decision = AdmissionDecision(
            admit_full=True,
            admitted_content={"payload": "test"},
            original_tokens=5000,
            admitted_tokens=5000,
            post_admission_utilization=0.225
        )
        
        assert decision.admit_full is True
        assert decision.deferred_content is None
        assert decision.original_tokens == 5000
    
    def test_decision_creation_truncated(self):
        """Test creating a decision with truncation."""
        decision = AdmissionDecision(
            admit_full=False,
            admitted_content={"payload": "truncated"},
            deferred_content=[{"content": "deferred item"}],
            deferred_kb_tokens=["<#CGKB-DEFERRED-0>"],
            truncation_notice="⚠️ Items were deferred",
            original_tokens=100000,
            admitted_tokens=20000,
            deferred_tokens=80000,
            post_admission_utilization=0.40
        )
        
        assert decision.admit_full is False
        assert decision.admitted_tokens == 20000
        assert len(decision.deferred_kb_tokens) == 1


class TestCalculateAdmissionBudget:
    """Tests for calculate_admission_budget function."""
    
    def test_budget_from_low_utilization(self):
        """Test budget calculation when utilization is low."""
        # At 20K tokens (10% of 200K), should have (38% - 10%) = 28% available
        budget = calculate_admission_budget(
            current_tokens=20000,
            context_limit=200000
        )
        # 38% of 200K = 76K target, minus 20K current = 56K available
        assert budget == 56000
    
    def test_budget_from_moderate_utilization(self):
        """Test budget calculation at moderate utilization."""
        # At 60K tokens (30% of 200K), should have (38% - 30%) = 8% available
        budget = calculate_admission_budget(
            current_tokens=60000,
            context_limit=200000
        )
        # 38% of 200K = 76K target, minus 60K current = 16K available
        assert budget == 16000
    
    def test_budget_near_target_returns_minimum(self):
        """Test budget at target threshold returns minimum."""
        # At 74K tokens (37% of 200K), only 1% headroom = 2K
        # Should return MIN_ADMISSION_TOKENS instead
        budget = calculate_admission_budget(
            current_tokens=74000,
            context_limit=200000
        )
        # 38% of 200K = 76K target, minus 74K = 2K, but min is 5K
        assert budget == MIN_ADMISSION_TOKENS
    
    def test_budget_over_target_returns_minimum(self):
        """Test budget when already over target."""
        # At 100K tokens (50% of 200K), already over 38% target
        budget = calculate_admission_budget(
            current_tokens=100000,
            context_limit=200000
        )
        # Should still get minimum
        assert budget == MIN_ADMISSION_TOKENS
    
    def test_budget_with_custom_target(self):
        """Test budget calculation with custom target utilization."""
        budget = calculate_admission_budget(
            current_tokens=40000,  # 20% of 200K
            context_limit=200000,
            target_utilization=0.50  # Higher target (50%)
        )
        # 50% of 200K = 100K target, minus 40K current = 60K
        assert budget == 60000


class TestCheckPreAdmission:
    """Tests for check_pre_admission function."""
    
    def test_full_admission_small_result(self):
        """Test that small results are fully admitted."""
        tool_result = {
            "payload": {"result": "Small result"},
            "_knowledge_items_to_add": []
        }
        
        # 20K tokens = 10% utilization, plenty of room
        decision = check_pre_admission(
            tool_result=tool_result,
            current_context_tokens=20000,
            model_name="anthropic/claude-sonnet-4-20250514"
        )
        
        assert decision.admit_full is True
        assert decision.deferred_content is None
    
    def test_full_admission_with_kb_items_under_threshold(self):
        """Test full admission when KB items fit within budget."""
        kb_items = [
            {"content": "Short content " * 50, "source_uri": "https://example.com"}
            for _ in range(5)
        ]
        
        tool_result = {
            "payload": {"result": "Search completed"},
            "_knowledge_items_to_add": kb_items
        }
        
        # Low utilization, small KB items should fit
        decision = check_pre_admission(
            tool_result=tool_result,
            current_context_tokens=10000,
            model_name="anthropic/claude-sonnet-4-20250514"
        )
        
        assert decision.admit_full is True
        assert len(decision.deferred_kb_tokens) == 0
    
    def test_truncation_large_result(self):
        """Test that large results are truncated."""
        # Create many KB items that would exceed budget
        kb_items = [
            {"content": "Content block " * 500, "source_uri": f"https://example{i}.com"}
            for i in range(50)
        ]
        
        tool_result = {
            "payload": {"result": "Search completed"},
            "_knowledge_items_to_add": kb_items
        }
        
        # At 35% utilization, only 3% budget remaining (6K tokens)
        decision = check_pre_admission(
            tool_result=tool_result,
            current_context_tokens=70000,  # 35% of 200K
            model_name="anthropic/claude-sonnet-4-20250514"
        )
        
        assert decision.admit_full is False
        assert decision.deferred_content is not None
        assert len(decision.deferred_kb_tokens) > 0
        assert decision.admitted_tokens < decision.original_tokens
    
    @patch("agent_core.framework.context_admission_controller.get_model_context_limit")
    def test_uses_model_context_limit(self, mock_get_limit):
        """Test that model context limit is properly used."""
        mock_get_limit.return_value = 100000  # Smaller context
        
        tool_result = {
            "payload": {"result": "Test"},
            "_knowledge_items_to_add": []
        }
        
        decision = check_pre_admission(
            tool_result=tool_result,
            current_context_tokens=10000,
            model_name="test-model"
        )
        
        mock_get_limit.assert_called_once()
        # Post utilization should be based on 100K limit
        assert decision.post_admission_utilization > 0


class TestScoreAndSortKbItems:
    """Tests for _score_and_sort_kb_items function."""
    
    def test_sorts_by_source_authority(self):
        """Test that items are sorted by source authority."""
        items = [
            {"content": "Regular content " * 100, "source_uri": "https://blog.example.com"},
            {"content": "Academic content " * 100, "source_uri": "https://example.edu/paper"},
            {"content": "Gov content " * 100, "source_uri": "https://example.gov/report"},
        ]
        
        sorted_items = _score_and_sort_kb_items(items)
        
        # Gov and edu sources should rank higher than blog
        # Both .gov and .edu get +3.0, so either could be first based on position
        top_sources = [sorted_items[0][0]["source_uri"], sorted_items[1][0]["source_uri"]]
        assert any("gov" in s or "edu" in s for s in top_sources)
        # Blog should be last (lowest authority)
        assert "blog" in sorted_items[2][0]["source_uri"]
    
    def test_penalizes_very_short_content(self):
        """Test that very short content is penalized."""
        items = [
            {"content": "Short", "source_uri": "https://example.com"},
            {"content": "Medium length content here " * 50, "source_uri": "https://example.com"},
        ]
        
        sorted_items = _score_and_sort_kb_items(items)
        
        # Medium content should rank higher than very short
        assert len(sorted_items[0][0]["content"]) > len(sorted_items[1][0]["content"])
    
    def test_handles_empty_items(self):
        """Test handling empty item list."""
        sorted_items = _score_and_sort_kb_items([])
        assert sorted_items == []
    
    def test_uses_relevance_metadata(self):
        """Test that relevance metadata boosts score."""
        items = [
            {"content": "Content A " * 100, "source_uri": "https://example.com", "metadata": {}},
            {"content": "Content B " * 100, "source_uri": "https://example.com", "metadata": {"relevance_score": 0.95}},
        ]
        
        sorted_items = _score_and_sort_kb_items(items)
        
        # Item with relevance score should rank higher
        assert sorted_items[0][0]["metadata"].get("relevance_score") == 0.95


class TestSerializePayload:
    """Tests for _serialize_payload function."""
    
    def test_string_passthrough(self):
        """Test that string payloads pass through unchanged."""
        result = _serialize_payload("test string")
        assert result == "test string"
    
    def test_dict_serialization(self):
        """Test that dicts are JSON serialized."""
        payload = {"key": "value", "number": 42}
        result = _serialize_payload(payload)
        assert "key" in result
        assert "value" in result
    
    def test_none_returns_empty(self):
        """Test that None returns empty string."""
        result = _serialize_payload(None)
        assert result == ""


class TestEstimateResultTokens:
    """Tests for estimate_result_tokens function."""
    
    def test_counts_payload_tokens(self):
        """Test counting payload tokens."""
        tool_result = {
            "payload": {"result": "Test content " * 100},
            "_knowledge_items_to_add": []
        }
        
        tokens = estimate_result_tokens(tool_result)
        assert tokens > 0
    
    def test_counts_kb_items_tokens(self):
        """Test counting KB items tokens."""
        tool_result = {
            "payload": {},
            "_knowledge_items_to_add": [
                {"content": "KB content " * 100},
                {"content": "More KB content " * 100}
            ]
        }
        
        tokens = estimate_result_tokens(tool_result)
        # Should count tokens from both KB items
        assert tokens > estimate_tokens("KB content " * 100)
    
    def test_empty_result(self):
        """Test empty result returns minimal tokens."""
        tool_result = {
            "payload": {},
            "_knowledge_items_to_add": []
        }
        
        tokens = estimate_result_tokens(tool_result)
        # Empty dict serializes to "{}", real tokenizer may count differently than heuristic
        # Should still be small (under 20 tokens for empty payload)
        assert tokens <= 20


class TestIntegration:
    """Integration tests for admission controller."""
    
    def test_full_workflow_small_result(self):
        """Test complete workflow for small result."""
        tool_result = {
            "payload": {"instructional_prompt": "Here are the search results"},
            "_knowledge_items_to_add": [
                {"content": "Result content " * 50, "source_uri": "https://example.com"}
            ]
        }
        
        decision = check_pre_admission(
            tool_result=tool_result,
            current_context_tokens=30000,  # 15% utilization
            model_name="anthropic/claude-sonnet-4-20250514"
        )
        
        assert decision.admit_full is True
        assert decision.truncation_notice is None
        assert decision.post_admission_utilization < WARNING_THRESHOLD
    
    def test_full_workflow_large_result_truncated(self):
        """Test complete workflow for large result requiring truncation."""
        # Create large result that will exceed budget
        kb_items = [
            {
                "content": f"{'Content block ' * 300}\n" * 5,
                "source_uri": f"https://example{i}.com",
                "token": f"<#CGKB-{i:05d}>"
            }
            for i in range(30)
        ]
        
        tool_result = {
            "payload": {"instructional_prompt": "Search results"},
            "_knowledge_items_to_add": kb_items
        }
        
        # At 35% utilization, only 3% budget remaining
        decision = check_pre_admission(
            tool_result=tool_result,
            current_context_tokens=70000,  # 35% of 200K
            model_name="anthropic/claude-sonnet-4-20250514"
        )
        
        assert decision.admit_full is False
        assert decision.deferred_content is not None
        assert len(decision.deferred_kb_tokens) > 0
        assert decision.truncation_notice is not None
        assert "Context Budget Notice" in decision.truncation_notice
    
    def test_truncation_preserves_high_value_items(self):
        """Test that truncation keeps high-value sources."""
        # Mix of sources with different authority
        kb_items = [
            {"content": "Blog content " * 200, "source_uri": "https://blog.example.com"},
            {"content": "Academic content " * 200, "source_uri": "https://university.edu/paper"},
            {"content": "Gov content " * 200, "source_uri": "https://agency.gov/report"},
            {"content": "Medium content " * 200, "source_uri": "https://medium.com/article"},
        ]
        
        tool_result = {
            "payload": {"result": "Done"},
            "_knowledge_items_to_add": kb_items
        }
        
        # Force truncation with high utilization
        decision = check_pre_admission(
            tool_result=tool_result,
            current_context_tokens=72000,  # 36% - only 2% budget
            model_name="anthropic/claude-sonnet-4-20250514"
        )
        
        # If truncation occurred, check that admitted content prioritizes authority
        if not decision.admit_full:
            admitted_items = decision.admitted_content.get("_knowledge_items_to_add", [])
            if admitted_items:
                # First admitted should be high-authority source
                first_source = admitted_items[0].get("source_uri", "")
                assert ".gov" in first_source or ".edu" in first_source
