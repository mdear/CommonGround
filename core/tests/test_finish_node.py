"""
Tests for finish_node.py - specifically the _extract_deliverables_from_messages function.

These tests cover the no-truncation extraction logic added to handle Associates
that call finish_flow without first calling generate_message_summary.
"""
import pytest
import sys
from pathlib import Path

# Import the function directly (it's module-level, not in a class)
CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))

from agent_core.nodes.custom_nodes.finish_node import _extract_deliverables_from_messages


class TestExtractDeliverablesFromMessages:
    """Tests for the _extract_deliverables_from_messages function."""

    # =========================================================================
    # Basic Functionality
    # =========================================================================

    def test_empty_messages_returns_empty_dict(self):
        """Empty message list should return empty dict."""
        result = _extract_deliverables_from_messages([])
        assert result == {}

    def test_none_messages_returns_empty_dict(self):
        """None should be handled gracefully (though type hint says List)."""
        # This tests defensive coding
        result = _extract_deliverables_from_messages(None)
        assert result == {}

    def test_only_user_messages_returns_empty(self):
        """Messages with only user role should return empty (no assistant content)."""
        messages = [
            {"role": "user", "content": "Hello, can you help me?"},
            {"role": "user", "content": "I need analysis of the codebase."},
        ]
        result = _extract_deliverables_from_messages(messages)
        assert result == {}

    def test_short_content_filtered_out(self):
        """Assistant messages with very short content (<50 chars) should be skipped."""
        messages = [
            {"role": "assistant", "content": "OK"},
            {"role": "assistant", "content": "Sure, I'll help."},
            {"role": "assistant", "content": "Let me check."},
        ]
        result = _extract_deliverables_from_messages(messages)
        assert result == {}

    def test_extracts_substantive_assistant_content(self, sample_assistant_messages):
        """Should extract meaningful content from assistant messages."""
        result = _extract_deliverables_from_messages(sample_assistant_messages)

        assert "primary_summary" in result
        summary = result["primary_summary"]

        # Should contain the substantive findings
        assert "Final Analysis" in summary or "codebase" in summary.lower()
        assert "Finding" in summary  # Our formatting adds "Finding N"

    # =========================================================================
    # NO TRUNCATION Guarantee
    # =========================================================================

    def test_no_truncation_long_finding(self):
        """Individual findings should NEVER be truncated, even if very long."""
        # Create a very long finding (5000 chars)
        long_content = "This is a comprehensive analysis. " * 150  # ~5100 chars

        messages = [
            {"role": "assistant", "content": long_content}
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        # The full content should be present (not truncated)
        # We need to check that "comprehensive analysis" appears many times
        assert summary.count("comprehensive analysis") >= 100

    def test_no_truncation_with_budget(self):
        """Even with a small budget, included findings should be complete."""
        # Create a finding that's larger than the budget
        long_content = "Important finding with critical details. " * 100  # ~4000 chars

        messages = [
            {"role": "assistant", "content": long_content}
        ]

        # Set a budget smaller than the content
        result = _extract_deliverables_from_messages(
            messages,
            summarization_budget_chars=1000
        )
        summary = result.get("primary_summary", "")

        # First finding should ALWAYS be included completely
        assert summary.count("Important finding") >= 50

    def test_budget_selects_fewer_findings_not_truncate(self):
        """Budget should select fewer findings, not truncate them."""
        # Create 5 distinct findings
        messages = [
            {"role": "assistant", "content": f"Finding {i}: " + "x" * 200}
            for i in range(1, 6)
        ]

        # With no budget, should include all 5
        result_no_budget = _extract_deliverables_from_messages(messages)
        summary_no_budget = result_no_budget.get("primary_summary", "")

        # With tight budget, should include fewer
        result_with_budget = _extract_deliverables_from_messages(
            messages,
            summarization_budget_chars=500
        )
        summary_with_budget = result_with_budget.get("primary_summary", "")

        # Count how many "Finding X:" appear in each
        no_budget_count = sum(1 for i in range(1, 6) if f"Finding {i}:" in summary_no_budget)
        with_budget_count = sum(1 for i in range(1, 6) if f"Finding {i}:" in summary_with_budget)

        # Budget version should have fewer findings
        assert with_budget_count < no_budget_count
        # But at least one finding should be present
        assert with_budget_count >= 1

    # =========================================================================
    # Content Cleaning
    # =========================================================================

    def test_removes_thinking_tags(self):
        """<thinking> tags should be stripped from content."""
        messages = [
            {
                "role": "assistant",
                "content": "<thinking>Internal reasoning here</thinking>\n\nThe actual analysis shows important results that should be preserved in the output."
            }
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        assert "Internal reasoning" not in summary
        assert "actual analysis" in summary

    def test_removes_internal_tags(self):
        """<internal> tags should be stripped from content."""
        messages = [
            {
                "role": "assistant",
                "content": "<internal>System note</internal>\n\nThe visible content that users should see in the final output summary."
            }
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        assert "System note" not in summary
        assert "visible content" in summary

    def test_removes_internal_system_directive_tags(self):
        """<internal_system_directive> tags should be stripped."""
        messages = [
            {
                "role": "assistant",
                "content": "<internal_system_directive>Do X</internal_system_directive>\n\nHere is the analysis result that should appear in deliverables."
            }
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        assert "Do X" not in summary
        assert "analysis result" in summary

    # =========================================================================
    # Tool Tracking
    # =========================================================================

    def test_tracks_tools_used(self):
        """Should track which tools were called."""
        messages = [
            {
                "role": "assistant",
                "content": "I will search for the information you requested using available tools.",
                "tool_calls": [
                    {"id": "1", "function": {"name": "web_search"}, "type": "function"},
                    {"id": "2", "function": {"name": "read_file"}, "type": "function"},
                ]
            },
            {"role": "tool", "content": "results", "tool_call_id": "1"},
            {"role": "tool", "content": "file content", "tool_call_id": "2"},
            {
                "role": "assistant",
                "content": "Based on my search and file reading, here are the comprehensive results of my analysis."
            }
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        # Tools section should be present
        assert "Tools Used" in summary
        assert "web_search" in summary
        assert "read_file" in summary

    # =========================================================================
    # Recency Priority
    # =========================================================================

    def test_prioritizes_recent_messages(self):
        """Later messages (conclusions) should be prioritized over earlier ones."""
        messages = [
            {"role": "assistant", "content": "Early analysis: Starting to look at the problem and gathering initial data."},
            {"role": "assistant", "content": "Middle work: Processing the data and running various analyses on it."},
            {"role": "assistant", "content": "FINAL CONCLUSION: This is the definitive answer after all analysis is complete."},
        ]

        # With very tight budget, should get the last one (conclusion)
        result = _extract_deliverables_from_messages(
            messages,
            summarization_budget_chars=200
        )
        summary = result.get("primary_summary", "")

        # Should contain the conclusion
        assert "FINAL CONCLUSION" in summary

    # =========================================================================
    # Module Description
    # =========================================================================

    def test_includes_module_description(self):
        """Module description should be included in output if provided."""
        messages = [
            {"role": "assistant", "content": "Here is my detailed analysis of the requested topic with findings."}
        ]

        result = _extract_deliverables_from_messages(
            messages,
            module_description="Analyze user authentication flow"
        )
        summary = result.get("primary_summary", "")

        assert "Analyze user authentication flow" in summary
        assert "Work Module" in summary

    # =========================================================================
    # Metadata
    # =========================================================================

    def test_includes_extraction_metadata(self):
        """Should include note about auto-extraction."""
        messages = [
            {"role": "assistant", "content": "Here is the analysis with substantive content for testing purposes."}
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        assert "Auto-extracted" in summary or "auto-extracted" in summary

    def test_notes_omitted_findings_count(self):
        """When findings are omitted due to budget, should note how many."""
        # Create many findings
        messages = [
            {"role": "assistant", "content": f"Analysis point {i}: " + "detailed content " * 20}
            for i in range(10)
        ]

        result = _extract_deliverables_from_messages(
            messages,
            summarization_budget_chars=500
        )
        summary = result.get("primary_summary", "")

        # Should mention omitted messages if any were skipped
        if "omitted" in summary.lower():
            assert "earlier messages omitted" in summary.lower() or "messages omitted" in summary.lower()


class TestExtractDeliverablesEdgeCases:
    """Edge case tests for extraction function."""

    def test_message_without_content_key(self):
        """Messages missing 'content' key should be handled."""
        messages = [
            {"role": "assistant"},  # No content key
            {"role": "assistant", "content": "Valid content that should be extracted from messages."},
        ]

        result = _extract_deliverables_from_messages(messages)
        # Should not crash, should extract valid content
        assert "primary_summary" in result

    def test_tool_calls_without_function_key(self):
        """Malformed tool_calls should not crash extraction."""
        messages = [
            {
                "role": "assistant",
                "content": "Calling a tool to get the necessary information for analysis.",
                "tool_calls": [
                    {"id": "1"},  # Missing function key
                    {"id": "2", "function": {}},  # Empty function
                    {"id": "3", "function": {"name": "valid_tool"}},  # Valid
                ]
            }
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        # Should extract content and track the valid tool
        assert "valid_tool" in summary

    def test_handles_none_content(self):
        """Content that is None should be handled."""
        messages = [
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "Valid content that should be extracted properly from this longer message."},
        ]

        result = _extract_deliverables_from_messages(messages)
        # Only the valid message has content, and it's >50 chars, so should extract
        assert "primary_summary" in result

    def test_max_findings_limit(self):
        """Should have a reasonable max limit on findings to prevent extreme cases."""
        # Create 50 findings
        messages = [
            {"role": "assistant", "content": f"Finding {i}: Detailed analysis content here." + "x" * 100}
            for i in range(50)
        ]

        result = _extract_deliverables_from_messages(messages)
        summary = result.get("primary_summary", "")

        # Should not include all 50 - there's a max_findings_unbounded = 15
        finding_count = summary.count("### Finding")
        assert finding_count <= 15
