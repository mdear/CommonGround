"""
Tests for stale/orphaned session detection in GetPrincipalStatusSummaryTool.

These tests verify that the Partner agent correctly identifies sessions that:
1. Have modules stuck in "ongoing" status without recent updates
2. Appear to be orphaned (all active modules are stale)
3. Were likely interrupted by WebSocket disconnects
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_core.nodes.custom_nodes.get_principal_status_tool import (
    GetPrincipalStatusSummaryTool,
    STALE_ONGOING_THRESHOLD_MINUTES,
    ORPHANED_SESSION_THRESHOLD_MINUTES
)


class TestStaleModuleDetection:
    """Tests for the _detect_stale_modules helper method."""

    @pytest.fixture
    def tool(self):
        """Create a GetPrincipalStatusSummaryTool instance."""
        return GetPrincipalStatusSummaryTool()

    @pytest.fixture
    def now(self):
        """Current timestamp for testing."""
        return datetime.now(timezone.utc)

    def test_no_stale_modules_when_recently_updated(self, tool, now):
        """Modules updated within threshold should not be flagged as stale."""
        recent_update = (now - timedelta(minutes=5)).isoformat()
        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": recent_update},
            "WM_2": {"status": "ongoing", "updated_at": recent_update},
        }

        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        assert len(stale_modules) == 0
        assert is_orphaned is False
        assert warning == ""

    def test_detects_single_stale_module(self, tool, now):
        """A single module past the threshold should be detected."""
        old_update = (now - timedelta(minutes=STALE_ONGOING_THRESHOLD_MINUTES + 5)).isoformat()
        recent_update = (now - timedelta(minutes=2)).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": old_update},
            "WM_2": {"status": "ongoing", "updated_at": recent_update},
        }

        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        assert len(stale_modules) == 1
        assert stale_modules[0]["module_id"] == "WM_1"
        assert is_orphaned is False  # Not all active modules are stale
        assert "WARNING" in warning
        assert "WM_1" in warning

    def test_detects_orphaned_session_all_modules_stale(self, tool, now):
        """When ALL active modules are stale, session should be flagged as orphaned."""
        old_update = (now - timedelta(minutes=STALE_ONGOING_THRESHOLD_MINUTES + 30)).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": old_update},
            "WM_2": {"status": "ongoing", "updated_at": old_update},
            "WM_3": {"status": "ongoing", "updated_at": old_update},
            "WM_4": {"status": "pending", "updated_at": old_update},  # Not active, ignored
            "WM_5": {"status": "completed", "updated_at": old_update},  # Not active, ignored
        }

        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        assert len(stale_modules) == 3  # Only ongoing modules
        assert is_orphaned is True
        assert "CRITICAL" in warning
        assert "ORPHANED" in warning

    def test_ignores_completed_modules(self, tool, now):
        """Completed modules should not be checked for staleness."""
        old_update = (now - timedelta(hours=24)).isoformat()

        work_modules = {
            "WM_1": {"status": "completed", "updated_at": old_update},
            "WM_2": {"status": "pending_review", "updated_at": old_update},
            "WM_3": {"status": "pending", "updated_at": old_update},
        }

        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        assert len(stale_modules) == 0
        assert is_orphaned is False
        assert warning == ""

    def test_handles_missing_updated_at(self, tool, now):
        """Modules without updated_at should not crash."""
        work_modules = {
            "WM_1": {"status": "ongoing"},  # No updated_at
            "WM_2": {"status": "ongoing", "updated_at": None},
        }

        # Should not raise
        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        # Can't determine staleness without timestamp, so not flagged
        assert len(stale_modules) == 0

    def test_handles_invalid_timestamp_format(self, tool, now):
        """Invalid timestamp formats should be handled gracefully."""
        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": "not-a-timestamp"},
            "WM_2": {"status": "ongoing", "updated_at": "2025-13-45T99:99:99"},  # Invalid
        }

        # Should not raise
        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        # Invalid timestamps can't be evaluated, so not flagged
        assert len(stale_modules) == 0

    def test_handles_timezone_naive_timestamps(self, tool, now):
        """Timestamps without timezone info should be handled."""
        # Timezone-naive timestamp (will be treated as UTC)
        old_update_naive = (now - timedelta(minutes=STALE_ONGOING_THRESHOLD_MINUTES + 10)).replace(tzinfo=None).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": old_update_naive},
        }

        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        assert len(stale_modules) == 1

    def test_very_old_session_hours_ago(self, tool, now):
        """Sessions from hours ago should be clearly identified."""
        # Simulate the tunneling-festive-mammoth scenario: ~27 hours old
        very_old_update = (now - timedelta(hours=27)).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": very_old_update},
            "WM_2": {"status": "ongoing", "updated_at": very_old_update},
            "WM_3": {"status": "ongoing", "updated_at": very_old_update},
            "WM_4": {"status": "ongoing", "updated_at": very_old_update},
            "WM_5": {"status": "ongoing", "updated_at": very_old_update},
            "WM_6": {"status": "pending", "updated_at": very_old_update},
        }

        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        assert len(stale_modules) == 5  # Only the 5 ongoing modules
        assert is_orphaned is True
        assert "CRITICAL" in warning
        assert "ORPHANED" in warning

        # Check that minutes are calculated correctly (should be ~1620 minutes)
        max_minutes = max(m["minutes_since_update"] for m in stale_modules)
        assert max_minutes > 1600  # ~27 hours = 1620 minutes

    def test_mixed_status_modules(self, tool, now):
        """Test with a mix of ongoing, completed, and pending modules."""
        old_update = (now - timedelta(minutes=STALE_ONGOING_THRESHOLD_MINUTES + 20)).isoformat()
        recent_update = (now - timedelta(minutes=2)).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": old_update},  # Stale
            "WM_2": {"status": "completed", "updated_at": old_update},  # Ignored
            "WM_3": {"status": "pending_review", "updated_at": old_update},  # Ignored
            "WM_4": {"status": "in_progress", "updated_at": recent_update},  # Active but recent
            "WM_5": {"status": "pending", "updated_at": old_update},  # Ignored
        }

        stale_modules, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        # Only WM_1 should be stale (WM_4 is recent)
        assert len(stale_modules) == 1
        assert stale_modules[0]["module_id"] == "WM_1"
        assert is_orphaned is False  # WM_4 is still active and not stale


class TestWarningMessageFormat:
    """Tests for warning message formatting."""

    @pytest.fixture
    def tool(self):
        return GetPrincipalStatusSummaryTool()

    @pytest.fixture
    def now(self):
        return datetime.now(timezone.utc)

    def test_warning_includes_module_ids(self, tool, now):
        """Warning message should list affected module IDs."""
        old_update = (now - timedelta(minutes=60)).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": old_update},
            "WM_2": {"status": "ongoing", "updated_at": old_update},
        }

        _, _, warning = tool._detect_stale_modules(work_modules, now)

        assert "WM_1" in warning
        assert "WM_2" in warning

    def test_warning_includes_time_duration(self, tool, now):
        """Warning should include how long modules have been stale."""
        old_update = (now - timedelta(minutes=45)).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": old_update},
        }

        _, _, warning = tool._detect_stale_modules(work_modules, now)

        assert "45" in warning  # Should mention ~45 minutes

    def test_orphaned_warning_mentions_websocket(self, tool, now):
        """Orphaned session warning should mention WebSocket disconnect."""
        old_update = (now - timedelta(hours=2)).isoformat()

        work_modules = {
            "WM_1": {"status": "ongoing", "updated_at": old_update},
        }

        _, is_orphaned, warning = tool._detect_stale_modules(work_modules, now)

        assert is_orphaned is True
        assert "WebSocket" in warning or "disconnect" in warning.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
