"""
Tests for content_selection.py - Budget-aware content inheritance utilities.

These tests cover the two-tier content selection strategy used to prevent
Associates from being "born over-budget" when inheriting context from
completed work modules.

Test Coverage:
1. Budget computation (compute_inheritance_budget_chars)
2. Message size estimation (_estimate_message_chars)
3. Newest-first message selection (_select_messages_newest_first)
4. Two-tier selection (select_content_within_budget)
5. Briefing formatting (format_inherited_content_for_briefing)
6. Async hydration helpers
7. Edge cases and error handling
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))

from agent_core.utils.content_selection import (
    # Constants
    CHARS_PER_TOKEN,
    INHERITANCE_BUDGET_FRACTION,
    STRATEGY_LLM_SUMMARY,
    STRATEGY_NEWEST_FIRST,
    STRATEGY_EMPTY,
    # Functions
    compute_inheritance_budget_chars,
    _estimate_message_chars,
    _select_messages_newest_first,
    select_content_within_budget,
    format_inherited_content_for_briefing,
    hydrate_messages_for_selection,
    select_inherited_content_with_hydration,
)


class TestConstants:
    """Tests for module-level constants."""

    def test_chars_per_token_is_reasonable(self):
        """CHARS_PER_TOKEN should be a reasonable approximation."""
        assert CHARS_PER_TOKEN == 4
        assert isinstance(CHARS_PER_TOKEN, int)

    def test_inheritance_budget_fraction_is_reasonable(self):
        """INHERITANCE_BUDGET_FRACTION should leave room for agent's own work."""
        assert INHERITANCE_BUDGET_FRACTION == 0.40
        assert 0 < INHERITANCE_BUDGET_FRACTION < 0.5  # Less than half

    def test_strategy_constants_are_strings(self):
        """Strategy constants should be strings."""
        assert isinstance(STRATEGY_LLM_SUMMARY, str)
        assert isinstance(STRATEGY_NEWEST_FIRST, str)
        assert isinstance(STRATEGY_EMPTY, str)

    def test_strategy_constants_are_unique(self):
        """Strategy constants should be distinct values."""
        strategies = {STRATEGY_LLM_SUMMARY, STRATEGY_NEWEST_FIRST, STRATEGY_EMPTY}
        assert len(strategies) == 3


class TestComputeInheritanceBudgetChars:
    """Tests for compute_inheritance_budget_chars function."""

    def test_basic_computation(self):
        """Test basic budget computation with typical values."""
        # 200K tokens, 2 sources
        result = compute_inheritance_budget_chars(200000, 2)
        # (200000 * 0.40 / 2) * 4 = 160000
        assert result == 160000

    def test_single_source(self):
        """Single source should get full inheritance pool."""
        result = compute_inheritance_budget_chars(200000, 1)
        # (200000 * 0.40 / 1) * 4 = 320000
        assert result == 320000

    def test_many_sources(self):
        """Many sources should split budget evenly."""
        result = compute_inheritance_budget_chars(200000, 5)
        # (200000 * 0.40 / 5) * 4 = 64000
        assert result == 64000

    def test_1m_context(self):
        """Test with 1M context limit."""
        result = compute_inheritance_budget_chars(1000000, 4)
        # (1000000 * 0.40 / 4) * 4 = 400000
        assert result == 400000

    def test_custom_fraction(self):
        """Test with custom inheritance fraction."""
        result = compute_inheritance_budget_chars(200000, 2, inheritance_fraction=0.20)
        # (200000 * 0.20 / 2) * 4 = 80000
        assert result == 80000

    def test_zero_sources_returns_zero(self):
        """Zero sources should return zero budget."""
        result = compute_inheritance_budget_chars(200000, 0)
        assert result == 0

    def test_negative_sources_returns_zero(self):
        """Negative sources should return zero budget."""
        result = compute_inheritance_budget_chars(200000, -1)
        assert result == 0

    def test_result_is_integer(self):
        """Result should be an integer (floor division)."""
        result = compute_inheritance_budget_chars(100000, 3)
        assert isinstance(result, int)


class TestEstimateMessageChars:
    """Tests for _estimate_message_chars function."""

    def test_simple_content(self):
        """Test with simple string content."""
        msg = {"role": "assistant", "content": "Hello world"}
        result = _estimate_message_chars(msg)
        # len("Hello world") = 11, plus ~50 overhead
        assert result >= 11
        assert result < 100

    def test_empty_content(self):
        """Test with empty content."""
        msg = {"role": "assistant", "content": ""}
        result = _estimate_message_chars(msg)
        # Should still have overhead
        assert result >= 50

    def test_no_content_field(self):
        """Test message without content field."""
        msg = {"role": "assistant"}
        result = _estimate_message_chars(msg)
        # Should return overhead only
        assert result >= 50

    def test_tool_calls(self):
        """Test message with tool calls."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "search_web",
                        "arguments": '{"query": "python testing"}'
                    }
                }
            ]
        }
        result = _estimate_message_chars(msg)
        # Should include tool call overhead
        assert result > 50

    def test_multimodal_content(self):
        """Test with list content (multimodal)."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "..."}}
            ]
        }
        result = _estimate_message_chars(msg)
        assert result > 50

    def test_large_content(self):
        """Test with large content."""
        large_text = "x" * 10000
        msg = {"role": "assistant", "content": large_text}
        result = _estimate_message_chars(msg)
        assert result >= 10000


class TestSelectMessagesNewestFirst:
    """Tests for _select_messages_newest_first function."""

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages of varying sizes."""
        return [
            {"role": "user", "content": "Message 1 - 100 chars" + "x" * 80},      # ~100 chars + overhead
            {"role": "assistant", "content": "Message 2 - 200 chars" + "x" * 180}, # ~200 chars + overhead
            {"role": "user", "content": "Message 3 - 150 chars" + "x" * 130},      # ~150 chars + overhead
            {"role": "assistant", "content": "Message 4 - 300 chars" + "x" * 280}, # ~300 chars + overhead
            {"role": "user", "content": "Message 5 - 50 chars" + "x" * 30},        # ~50 chars + overhead
        ]

    def test_selects_newest_first(self, sample_messages):
        """Should select messages from newest to oldest."""
        selected, metadata = _select_messages_newest_first(
            sample_messages,
            budget_chars=500,
            source_id="test"
        )
        # Should include message 5 (newest) first
        assert len(selected) > 0
        # First message in selected should be one of the original ones
        assert selected[-1] == sample_messages[-1] or selected[-1] == sample_messages[-2]

    def test_respects_budget(self, sample_messages):
        """Should stop when budget is exhausted."""
        selected, metadata = _select_messages_newest_first(
            sample_messages,
            budget_chars=200,  # Very tight budget
            source_id="test"
        )
        # Should not include all messages
        assert len(selected) < len(sample_messages)

    def test_always_includes_at_least_one(self, sample_messages):
        """Should always include at least one message even if over budget."""
        selected, metadata = _select_messages_newest_first(
            sample_messages,
            budget_chars=10,  # Impossibly small budget
            source_id="test"
        )
        assert len(selected) >= 1

    def test_preserves_original_order(self, sample_messages):
        """Selected messages should be in original order."""
        selected, metadata = _select_messages_newest_first(
            sample_messages,
            budget_chars=10000,  # Large budget
            source_id="test"
        )
        # Should be in original order
        original_indices = [sample_messages.index(msg) for msg in selected]
        assert original_indices == sorted(original_indices)

    def test_empty_messages(self):
        """Should handle empty message list."""
        selected, metadata = _select_messages_newest_first([], budget_chars=1000, source_id="test")
        assert selected == []
        assert metadata["strategy"] == STRATEGY_EMPTY
        assert metadata["items_selected"] == 0

    def test_metadata_correctness(self, sample_messages):
        """Should return accurate metadata."""
        selected, metadata = _select_messages_newest_first(
            sample_messages,
            budget_chars=500,
            source_id="test_source"
        )
        assert metadata["strategy"] == STRATEGY_NEWEST_FIRST
        assert metadata["source_id"] == "test_source"
        assert metadata["items_available"] == len(sample_messages)
        assert metadata["items_selected"] == len(selected)
        assert "chars_used" in metadata
        assert "budget_chars" in metadata


class TestSelectContentWithinBudget:
    """Tests for select_content_within_budget function."""

    @pytest.fixture
    def sample_deliverables(self):
        """Sample deliverables with primary_summary."""
        return {
            "primary_summary": "This is a 50 char summary for testing purposes.",
            "other_data": {"key": "value"}
        }

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for fallback selection."""
        return [
            {"role": "user", "content": "First message with some content."},
            {"role": "assistant", "content": "Response with detailed information about the topic."},
            {"role": "user", "content": "Follow-up question about specific details."},
        ]

    def test_tier1_uses_summary_when_fits(self, sample_deliverables, sample_messages):
        """Tier 1: Should use summary when it fits budget."""
        selected, metadata = select_content_within_budget(
            deliverables=sample_deliverables,
            messages=sample_messages,
            budget_chars=1000,  # Large enough for summary
            source_id="test"
        )
        assert metadata["strategy"] == STRATEGY_LLM_SUMMARY
        assert isinstance(selected, str)
        assert selected == sample_deliverables["primary_summary"]

    def test_tier2_fallback_when_summary_too_large(self, sample_deliverables, sample_messages):
        """Tier 2: Should fall back to messages when summary exceeds budget."""
        selected, metadata = select_content_within_budget(
            deliverables=sample_deliverables,
            messages=sample_messages,
            budget_chars=20,  # Too small for summary
            source_id="test"
        )
        assert metadata["strategy"] == STRATEGY_NEWEST_FIRST
        assert isinstance(selected, list)

    def test_tier2_when_no_summary(self, sample_messages):
        """Tier 2: Should use messages when no summary exists."""
        selected, metadata = select_content_within_budget(
            deliverables={},  # No primary_summary
            messages=sample_messages,
            budget_chars=1000,
            source_id="test"
        )
        assert metadata["strategy"] == STRATEGY_NEWEST_FIRST

    def test_tier2_when_deliverables_none(self, sample_messages):
        """Tier 2: Should use messages when deliverables is None."""
        selected, metadata = select_content_within_budget(
            deliverables=None,
            messages=sample_messages,
            budget_chars=1000,
            source_id="test"
        )
        assert metadata["strategy"] == STRATEGY_NEWEST_FIRST

    def test_empty_when_no_content(self):
        """Should return empty strategy when nothing available."""
        selected, metadata = select_content_within_budget(
            deliverables={},
            messages=[],
            budget_chars=1000,
            source_id="test"
        )
        assert metadata["strategy"] == STRATEGY_EMPTY
        assert selected == []

    def test_metadata_includes_budget_info(self, sample_deliverables, sample_messages):
        """Metadata should include budget information."""
        _, metadata = select_content_within_budget(
            deliverables=sample_deliverables,
            messages=sample_messages,
            budget_chars=1000,
            source_id="test_source"
        )
        assert "chars_used" in metadata
        assert "source_id" in metadata
        assert metadata["source_id"] == "test_source"


class TestFormatInheritedContentForBriefing:
    """Tests for format_inherited_content_for_briefing function."""

    def test_format_llm_summary(self):
        """Should format LLM summary into single message."""
        content = "This is the summary content."
        metadata = {"strategy": STRATEGY_LLM_SUMMARY, "chars_used": 100}

        result = format_inherited_content_for_briefing(content, metadata, "WM_1")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "WM_1" in result[0]["content"]
        assert "summary content" in result[0]["content"]
        assert result[0]["_internal"]["_no_handover"] is True
        assert result[0]["_internal"]["_inherited_from"] == "WM_1"

    def test_format_newest_first_messages(self):
        """Should format message list with header."""
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Message 2"},
        ]
        metadata = {
            "strategy": STRATEGY_NEWEST_FIRST,
            "chars_used": 200,
            "items_selected": 2
        }

        result = format_inherited_content_for_briefing(messages, metadata, "WM_2")

        # Header + 2 messages = 3 items
        assert len(result) == 3
        # First should be header
        assert "[Context inherited from WM_2" in result[0]["content"]
        assert result[0]["_internal"]["_is_header"] is True
        # Rest should be messages with inheritance metadata
        assert result[1]["_internal"]["_inherited_from"] == "WM_2"
        assert result[2]["_internal"]["_inherited_from"] == "WM_2"

    def test_format_empty_returns_empty(self):
        """Should return empty list for empty strategy."""
        result = format_inherited_content_for_briefing(
            [],
            {"strategy": STRATEGY_EMPTY},
            "WM_1"
        )
        assert result == []

    def test_all_messages_marked_no_handover(self):
        """All formatted messages should have _no_handover flag."""
        messages = [{"role": "user", "content": "Test"}]
        metadata = {"strategy": STRATEGY_NEWEST_FIRST, "chars_used": 50}

        result = format_inherited_content_for_briefing(messages, metadata, "WM_1")

        for msg in result:
            assert msg.get("_internal", {}).get("_no_handover") is True


class TestHydrateMessagesForSelection:
    """Tests for hydrate_messages_for_selection async function."""

    @pytest.mark.asyncio
    async def test_hydrates_messages(self):
        """Should hydrate messages using knowledge base."""
        messages = [
            {"role": "user", "content": "Check <#CGKB-00001> for details"},
            {"role": "assistant", "content": "Found information in <#CGKB-00002>"},
        ]

        mock_kb = MagicMock()
        mock_kb.hydrate_content = AsyncMock(side_effect=lambda c: c.replace(
            "<#CGKB-00001>", "HYDRATED_CONTENT_1"
        ).replace(
            "<#CGKB-00002>", "HYDRATED_CONTENT_2"
        ))

        result = await hydrate_messages_for_selection(messages, mock_kb, "test")

        assert "HYDRATED_CONTENT_1" in result[0]["content"]
        assert "HYDRATED_CONTENT_2" in result[1]["content"]

    @pytest.mark.asyncio
    async def test_handles_missing_kb(self):
        """Should return original messages when KB is None."""
        messages = [{"role": "user", "content": "Test <#CGKB-00001>"}]

        result = await hydrate_messages_for_selection(messages, None, "test")

        assert result == messages
        assert "<#CGKB-00001>" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self):
        """Should handle empty message list."""
        mock_kb = MagicMock()

        result = await hydrate_messages_for_selection([], mock_kb, "test")

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_hydration_error(self):
        """Should keep original content on hydration error."""
        messages = [{"role": "user", "content": "Test <#CGKB-00001>"}]

        mock_kb = MagicMock()
        mock_kb.hydrate_content = AsyncMock(side_effect=Exception("KB error"))

        result = await hydrate_messages_for_selection(messages, mock_kb, "test")

        # Should have original content
        assert result[0]["content"] == messages[0]["content"]


class TestSelectInheritedContentWithHydration:
    """Tests for select_inherited_content_with_hydration async function."""

    @pytest.fixture
    def sample_archive_entry(self):
        """Sample context archive entry."""
        return {
            "messages": [
                {"role": "user", "content": "Query about <#CGKB-00001>"},
                {"role": "assistant", "content": "Response with details"},
            ],
            "deliverables": {
                "primary_summary": "Summary of the work done."
            }
        }

    @pytest.mark.asyncio
    async def test_uses_tier1_when_summary_fits(self, sample_archive_entry):
        """Should use Tier 1 (summary) when it fits budget."""
        mock_kb = MagicMock()
        mock_kb.hydrate_content = AsyncMock(side_effect=lambda c: c)

        selected, metadata = await select_inherited_content_with_hydration(
            context_archive_entry=sample_archive_entry,
            budget_chars=1000,
            knowledge_base=mock_kb,
            source_id="WM_1"
        )

        assert metadata["strategy"] == STRATEGY_LLM_SUMMARY
        assert selected == sample_archive_entry["deliverables"]["primary_summary"]

    @pytest.mark.asyncio
    async def test_uses_tier2_when_no_summary(self):
        """Should use Tier 2 (messages) when no summary available."""
        archive_entry = {
            "messages": [
                {"role": "user", "content": "Test message"},
            ],
            "deliverables": {}  # No summary
        }

        mock_kb = MagicMock()
        mock_kb.hydrate_content = AsyncMock(side_effect=lambda c: c)

        selected, metadata = await select_inherited_content_with_hydration(
            context_archive_entry=archive_entry,
            budget_chars=1000,
            knowledge_base=mock_kb,
            source_id="WM_1"
        )

        assert metadata["strategy"] == STRATEGY_NEWEST_FIRST

    @pytest.mark.asyncio
    async def test_hydrates_before_selection(self):
        """Should hydrate messages before selection for accurate sizing."""
        # Message with KB token that expands significantly
        archive_entry = {
            "messages": [
                {"role": "user", "content": "<#CGKB-00001>"},  # Short when dehydrated
            ],
            "deliverables": {}
        }

        # Simulate KB token expanding to large content
        large_content = "x" * 50000  # 50K chars
        mock_kb = MagicMock()
        mock_kb.hydrate_content = AsyncMock(return_value=large_content)

        selected, metadata = await select_inherited_content_with_hydration(
            context_archive_entry=archive_entry,
            budget_chars=1000,  # Small budget
            knowledge_base=mock_kb,
            source_id="WM_1"
        )

        # Should have hydrated - KB method called
        mock_kb.hydrate_content.assert_called()

        # Should report chars_used based on hydrated size
        if metadata["strategy"] == STRATEGY_NEWEST_FIRST:
            assert metadata["chars_used"] >= 1000 or len(selected) == 1  # At least one message


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_non_string_summary_ignored(self):
        """Non-string summary should fall back to Tier 2."""
        selected, metadata = select_content_within_budget(
            deliverables={"primary_summary": 12345},  # Not a string
            messages=[{"role": "user", "content": "Test"}],
            budget_chars=1000,
            source_id="test"
        )
        assert metadata["strategy"] == STRATEGY_NEWEST_FIRST

    def test_unicode_content_handling(self):
        """Should handle unicode content correctly."""
        messages = [{"role": "user", "content": "こんにちは世界 🌍"}]
        selected, metadata = select_content_within_budget(
            deliverables={},
            messages=messages,
            budget_chars=1000,
            source_id="test"
        )
        assert len(selected) > 0

    def test_nested_content_in_messages(self):
        """Should handle nested content structures."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ]
            }
        ]
        result = _estimate_message_chars(messages[0])
        assert result > 0

    def test_very_large_budget(self):
        """Should handle very large budgets without issues."""
        messages = [{"role": "user", "content": "Small message"}]
        selected, metadata = select_content_within_budget(
            deliverables={},
            messages=messages,
            budget_chars=10000000,  # 10M chars
            source_id="test"
        )
        assert len(selected) == len(messages)

    def test_zero_budget(self):
        """Should still return at least one message with zero budget."""
        messages = [{"role": "user", "content": "Test"}]
        selected, metadata = _select_messages_newest_first(
            messages,
            budget_chars=0,
            source_id="test"
        )
        assert len(selected) >= 1  # At least one
