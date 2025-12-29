"""
Tests for context_budget_guardian.py - budget calculations and threshold logic.

These tests cover the new Context Budget Guardian that provides proactive
context window monitoring to prevent agents from hitting ContextWindowExceededError.
"""
import pytest
import sys
from pathlib import Path

CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))

from agent_core.framework.context_budget_guardian import (
    ContextBudgetStatus,
    DEFAULT_CONTEXT_LIMITS,
    WARNING_THRESHOLD,
    CRITICAL_THRESHOLD,
    EXCEEDED_THRESHOLD,
    get_model_context_limit,
    calculate_worker_budget,
    assess_context_budget,
    generate_context_budget_directive,
)


class TestContextBudgetThresholds:
    """Tests for threshold constants."""

    def test_thresholds_are_ascending(self):
        """Thresholds should be in ascending order."""
        assert WARNING_THRESHOLD < CRITICAL_THRESHOLD < EXCEEDED_THRESHOLD

    def test_thresholds_leave_headroom(self):
        """EXCEEDED threshold should leave at least 30% headroom."""
        assert EXCEEDED_THRESHOLD <= 0.70

    def test_warning_threshold_value(self):
        """WARNING should trigger at 40%."""
        assert WARNING_THRESHOLD == 0.40

    def test_critical_threshold_value(self):
        """CRITICAL should trigger at 55%."""
        assert CRITICAL_THRESHOLD == 0.55

    def test_exceeded_threshold_value(self):
        """EXCEEDED should trigger at 70%."""
        assert EXCEEDED_THRESHOLD == 0.70


class TestGetModelContextLimit:
    """Tests for get_model_context_limit function."""

    def test_explicit_config_takes_priority(self):
        """Explicit max_context_tokens in config should override defaults."""
        config = {"max_context_tokens": 500000}
        result = get_model_context_limit("any-model", config)
        assert result == 500000

    def test_exact_model_match(self):
        """Exact model name should match."""
        result = get_model_context_limit("openai/gpt-4o")
        assert result == 128000

    def test_model_prefix_match(self):
        """Versioned model names should match by prefix."""
        # claude-sonnet-4-20250514 should match anthropic/claude-sonnet-4
        result = get_model_context_limit("claude-sonnet-4-20250514")
        assert result == 200000

    def test_anthropic_claude_models(self):
        """Various Claude model formats should be detected."""
        models = [
            "anthropic/claude-sonnet-4",
            "claude-sonnet-4-20250514",
            "anthropic/claude-3-5-sonnet",
            "claude-3-5-sonnet-20241022",
        ]
        for model in models:
            result = get_model_context_limit(model)
            assert result == 200000, f"Failed for model: {model}"

    def test_openai_gpt4o_models(self):
        """GPT-4o models should return 128K."""
        models = ["openai/gpt-4o", "gpt-4o", "gpt-4o-2024-05-13"]
        for model in models:
            result = get_model_context_limit(model)
            assert result == 128000, f"Failed for model: {model}"

    def test_gemini_models(self):
        """Gemini models should return 1M."""
        result = get_model_context_limit("gemini/gemini-2.5-pro")
        assert result == 1000000

    def test_unknown_model_returns_conservative_default(self):
        """Unknown models should return conservative 100K default."""
        result = get_model_context_limit("unknown/mystery-model-v99")
        assert result == 100000

    def test_1m_context_with_header(self):
        """Models with anthropic-beta header should get 1M context."""
        config = {
            "extra_headers": {
                "anthropic-beta": "context-1m-2025-01-01"
            }
        }
        result = get_model_context_limit("anthropic/claude-sonnet-4-5", config)
        assert result == 1000000

    def test_1m_context_requires_supporting_model(self):
        """1M context header should only work for supporting models."""
        config = {
            "extra_headers": {
                "anthropic-beta": "context-1m-2025-01-01"
            }
        }
        # GPT-4 doesn't support 1M even with the header
        result = get_model_context_limit("openai/gpt-4o", config)
        assert result == 128000


class TestCalculateWorkerBudget:
    """Tests for calculate_worker_budget function."""

    def test_basic_calculation(self):
        """Basic budget calculation with default settings."""
        result = calculate_worker_budget(
            model_context_limit=200000,
            num_workers=3
        )

        # Total available = 200000 - 30000 overhead = 170000
        assert result["total_available"] == 170000

        # Summarization = 30% of 170000 = 51000
        assert result["summarization_budget"] == 51000

        # Worker total = 170000 - 51000 = 119000
        assert result["worker_budget_total"] == 119000

        # Per worker = 119000 / 3 = 39666
        assert result["per_worker_budget"] == 39666

    def test_single_worker(self):
        """Single worker should get full worker budget."""
        result = calculate_worker_budget(
            model_context_limit=200000,
            num_workers=1
        )

        assert result["per_worker_budget"] == result["worker_budget_total"]

    def test_zero_workers_handled(self):
        """Zero workers should not cause division by zero."""
        result = calculate_worker_budget(
            model_context_limit=200000,
            num_workers=0
        )

        # Should get full worker budget (no division)
        assert result["per_worker_budget"] == result["worker_budget_total"]

    def test_custom_overhead(self):
        """Custom base_context_overhead should be respected."""
        result = calculate_worker_budget(
            model_context_limit=200000,
            num_workers=2,
            base_context_overhead=50000
        )

        assert result["total_available"] == 150000

    def test_custom_summarization_reserve(self):
        """Custom summarization_reserve should be respected."""
        result = calculate_worker_budget(
            model_context_limit=200000,
            num_workers=2,
            summarization_reserve=0.50  # 50% reserve
        )

        # With 50% reserve, summarization gets half
        assert result["summarization_budget"] == int(170000 * 0.50)

    def test_large_context_model(self):
        """1M context model should scale appropriately."""
        result = calculate_worker_budget(
            model_context_limit=1000000,
            num_workers=5
        )

        # With 1M context, workers get much more budget
        assert result["per_worker_budget"] > 100000


class TestAssessContextBudget:
    """Tests for assess_context_budget function."""

    def test_healthy_status(self):
        """Under 40% should be HEALTHY."""
        status, metadata = assess_context_budget(
            predicted_tokens=50000,  # 25% of 200K
            model_name="anthropic/claude-sonnet-4"
        )

        assert status == ContextBudgetStatus.HEALTHY
        assert metadata["utilization_percent"] == 25.0
        assert "Healthy" in metadata["recommendation"]

    def test_warning_status(self):
        """40-55% should be WARNING."""
        status, metadata = assess_context_budget(
            predicted_tokens=90000,  # 45% of 200K
            model_name="anthropic/claude-sonnet-4"
        )

        assert status == ContextBudgetStatus.WARNING
        assert metadata["utilization_percent"] == 45.0
        assert "WARNING" in metadata["recommendation"]

    def test_critical_status(self):
        """55-70% should be CRITICAL."""
        status, metadata = assess_context_budget(
            predicted_tokens=120000,  # 60% of 200K
            model_name="anthropic/claude-sonnet-4"
        )

        assert status == ContextBudgetStatus.CRITICAL
        assert metadata["utilization_percent"] == 60.0
        assert "CRITICAL" in metadata["recommendation"]

    def test_exceeded_status(self):
        """Over 70% should be EXCEEDED."""
        status, metadata = assess_context_budget(
            predicted_tokens=150000,  # 75% of 200K
            model_name="anthropic/claude-sonnet-4"
        )

        assert status == ContextBudgetStatus.EXCEEDED
        assert metadata["utilization_percent"] == 75.0
        assert "EMERGENCY" in metadata["recommendation"]

    def test_metadata_includes_all_fields(self):
        """Metadata should include all expected fields."""
        status, metadata = assess_context_budget(
            predicted_tokens=50000,
            model_name="anthropic/claude-sonnet-4",
            agent_id="test_agent"
        )

        assert "context_limit" in metadata
        assert "predicted_tokens" in metadata
        assert "utilization_percent" in metadata
        assert "remaining_tokens" in metadata
        assert "model_name" in metadata
        assert "agent_id" in metadata
        assert "recommendation" in metadata

    def test_remaining_tokens_calculation(self):
        """remaining_tokens should be calculated correctly."""
        status, metadata = assess_context_budget(
            predicted_tokens=50000,
            model_name="anthropic/claude-sonnet-4"  # 200K limit
        )

        assert metadata["remaining_tokens"] == 150000

    def test_boundary_at_warning_threshold(self):
        """Exactly at 40% should be WARNING."""
        status, _ = assess_context_budget(
            predicted_tokens=80000,  # 40% of 200K
            model_name="anthropic/claude-sonnet-4"
        )

        assert status == ContextBudgetStatus.WARNING

    def test_just_under_warning_threshold(self):
        """Just under 40% should be HEALTHY."""
        status, _ = assess_context_budget(
            predicted_tokens=79000,  # 39.5% of 200K
            model_name="anthropic/claude-sonnet-4"
        )

        assert status == ContextBudgetStatus.HEALTHY


class TestGenerateContextBudgetDirective:
    """Tests for generate_context_budget_directive function."""

    def test_healthy_returns_none(self):
        """HEALTHY status should return None (no directive needed)."""
        result = generate_context_budget_directive(
            ContextBudgetStatus.HEALTHY,
            {"utilization_percent": 30, "remaining_tokens": 140000}
        )

        assert result is None

    def test_warning_returns_directive(self):
        """WARNING status should return a directive."""
        result = generate_context_budget_directive(
            ContextBudgetStatus.WARNING,
            {"utilization_percent": 45, "remaining_tokens": 110000}
        )

        assert result is not None
        assert "WARNING" in result
        assert "45%" in result

    def test_critical_returns_directive(self):
        """CRITICAL status should return a directive."""
        result = generate_context_budget_directive(
            ContextBudgetStatus.CRITICAL,
            {"utilization_percent": 60, "remaining_tokens": 80000}
        )

        assert result is not None
        assert "CRITICAL" in result

    def test_exceeded_returns_directive(self):
        """EXCEEDED status should return a directive."""
        result = generate_context_budget_directive(
            ContextBudgetStatus.EXCEEDED,
            {"utilization_percent": 75, "remaining_tokens": 50000}
        )

        assert result is not None
        assert "EMERGENCY" in result or "EXCEEDED" in result

    def test_principal_agent_uses_finish_flow(self):
        """Principal agent directive should mention finish_flow."""
        result = generate_context_budget_directive(
            ContextBudgetStatus.WARNING,
            {"utilization_percent": 45, "remaining_tokens": 110000},
            agent_type="principal"
        )

        assert "finish_flow" in result

    def test_associate_agent_uses_generate_message_summary(self):
        """Associate agent directive should mention generate_message_summary."""
        result = generate_context_budget_directive(
            ContextBudgetStatus.WARNING,
            {"utilization_percent": 45, "remaining_tokens": 110000},
            agent_type="associate"
        )

        assert "generate_message_summary" in result


class TestContextBudgetStatusEnum:
    """Tests for the ContextBudgetStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses should be defined."""
        assert hasattr(ContextBudgetStatus, 'HEALTHY')
        assert hasattr(ContextBudgetStatus, 'WARNING')
        assert hasattr(ContextBudgetStatus, 'CRITICAL')
        assert hasattr(ContextBudgetStatus, 'EXCEEDED')

    def test_statuses_are_unique(self):
        """Each status should have a unique value."""
        values = [
            ContextBudgetStatus.HEALTHY.value,
            ContextBudgetStatus.WARNING.value,
            ContextBudgetStatus.CRITICAL.value,
            ContextBudgetStatus.EXCEEDED.value,
        ]
        assert len(values) == len(set(values))
