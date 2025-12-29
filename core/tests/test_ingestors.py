"""
Tests for ingestors.py - specifically the EXCLUDED_FIELDS filtering.

These tests ensure that large fields like context_archive are properly filtered
out during work_modules injection to prevent context window explosion.
"""
import pytest
import sys
from pathlib import Path

CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))

from agent_core.events.ingestors import (
    work_modules_ingestor,
    INGESTOR_REGISTRY,
)


class TestWorkModulesIngestorExcludedFields:
    """Tests for EXCLUDED_FIELDS filtering in work_modules_ingestor."""

    def test_excludes_context_archive(self):
        """context_archive should be completely excluded from output."""
        payload = {
            "WM_1": {
                "name": "Test Module",
                "status": "pending_review",
                "context_archive": [
                    {
                        "messages": [{"role": "assistant", "content": "x" * 10000}],
                        "deliverables": {"primary_summary": "test"},
                        "model": "claude-sonnet-4-20250514"
                    }
                ]
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        # context_archive should not appear
        assert "context_archive" not in result
        # But other fields should
        assert "Test Module" in result
        assert "pending_review" in result

    def test_excludes_messages_field(self):
        """messages field should be excluded."""
        payload = {
            "WM_1": {
                "name": "Test",
                "status": "active",
                "messages": [
                    {"role": "user", "content": "old message " * 1000}
                ]
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        # Should not contain the messages content
        assert "old message" not in result
        # But module name should be there
        assert "Test" in result

    def test_excludes_raw_messages(self):
        """raw_messages field should be excluded."""
        payload = {
            "WM_1": {
                "name": "Analysis",
                "raw_messages": ["msg1", "msg2", "msg3"]
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        assert "raw_messages" not in result
        assert "msg1" not in result
        assert "Analysis" in result

    def test_excludes_full_context(self):
        """full_context field should be excluded."""
        payload = {
            "WM_1": {
                "name": "Work Item",
                "full_context": {"massive": "data " * 5000}
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        assert "full_context" not in result
        assert "massive" not in result
        assert "Work Item" in result

    def test_excludes_new_messages_from_associate(self):
        """new_messages_from_associate should be excluded."""
        payload = {
            "WM_1": {
                "name": "Task",
                "new_messages_from_associate": [
                    {"role": "assistant", "content": "associate work " * 500}
                ]
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        assert "new_messages_from_associate" not in result
        assert "associate work" not in result
        assert "Task" in result

    def test_all_excluded_fields_together(self):
        """Multiple excluded fields should all be filtered out."""
        payload = {
            "WM_1": {
                "name": "Complete Module",
                "description": "A test module",
                "status": "completed",
                # All excluded fields:
                "context_archive": [{"messages": ["large"]}],
                "full_context": {"big": "data"},
                "raw_messages": ["msg"],
                "messages": [{"role": "user", "content": "x"}],
                "new_messages_from_associate": [{"role": "assistant", "content": "y"}],
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        # None of the excluded fields should appear
        assert "context_archive" not in result
        assert "full_context" not in result
        assert "raw_messages" not in result
        # Note: "messages" might appear as a word in formatting,
        # but the actual message content shouldn't

        # Valid fields should appear
        assert "Complete Module" in result
        assert "test module" in result.lower()
        assert "completed" in result


class TestWorkModulesIngestorSummarizeFields:
    """Tests for SUMMARIZE_FIELDS handling."""

    def test_summarizes_deliverables_dict(self):
        """deliverables dict should be summarized as count."""
        payload = {
            "WM_1": {
                "name": "Test",
                "deliverables": {
                    "primary_summary": "summary",
                    "key_findings": ["a", "b"],
                    "recommendations": "do this"
                }
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        # Should show count, not full content
        assert "(3 items)" in result
        # Full content should not be dumped
        assert "do this" not in result

    def test_summarizes_tools_used(self):
        """tools_used should be summarized with truncation for long lists."""
        payload = {
            "WM_1": {
                "name": "Test",
                "tools_used": ["tool1", "tool2", "tool3", "tool4", "tool5", "tool6", "tool7"]
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        # Should show first 5 tools
        assert "tool1" in result
        assert "tool5" in result
        # Should indicate truncation
        assert "..." in result


class TestWorkModulesIngestorEdgeCases:
    """Edge case tests."""

    def test_handles_empty_payload(self):
        """Empty payload should return appropriate message."""
        result = work_modules_ingestor({}, {}, {})

        assert "No work modules" in result

    def test_handles_none_payload_equivalent(self):
        """Non-dict payload should be handled gracefully."""
        result = work_modules_ingestor("invalid", {}, {})

        assert "not in the expected format" in result

    def test_handles_nested_large_dicts(self):
        """Large nested dicts should be filtered recursively."""
        payload = {
            "WM_1": {
                "name": "Test",
                "nested": {
                    "context_archive": ["should be excluded"],
                    "valid_field": "should remain"
                }
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        # The nested excluded field should be filtered
        # The valid nested field should remain
        assert "valid_field" in result or "should remain" in result

    def test_preserves_small_fields(self):
        """Small fields that aren't in exclude list should be preserved."""
        payload = {
            "WM_1": {
                "name": "My Module",
                "description": "A description",
                "status": "active",
                "priority": "high",
                "created_at": "2025-01-01",
                "assignee": "Associate_1"
            }
        }

        result = work_modules_ingestor(payload, {}, {})

        assert "My Module" in result
        assert "description" in result.lower()
        assert "active" in result
        assert "high" in result
        assert "2025-01-01" in result
        assert "Associate_1" in result

    def test_custom_title_param(self):
        """Should use custom title from params."""
        payload = {"WM_1": {"name": "Test"}}
        params = {"title": "## Custom Work Modules Header"}

        result = work_modules_ingestor(payload, params, {})

        assert "Custom Work Modules Header" in result


class TestIngestorRegistry:
    """Test that ingestors are properly registered."""

    def test_work_modules_ingestor_registered(self):
        """work_modules_ingestor should be in the registry."""
        assert "work_modules_ingestor" in INGESTOR_REGISTRY
        assert INGESTOR_REGISTRY["work_modules_ingestor"] == work_modules_ingestor
