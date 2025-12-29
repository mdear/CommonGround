"""
Unit tests for agent_core.framework.profile_utils module.

This module tests profile retrieval utilities that find agent profiles
from the profiles store based on name, instance ID, revision, and active status.

Key functionality tested:
- get_active_profile_by_name: Find latest active profile by logical name
- get_profile_by_instance_id: Find profile by UUID instance ID
- get_active_profile_instance_id_by_name: Get instance ID by name
- Revision (rev) comparison for multiple active versions
- Timestamp comparison for same-revision profiles
- is_deleted filtering
"""

import pytest
from datetime import datetime, timezone
from agent_core.framework.profile_utils import (
    get_active_profile_by_name,
    get_profile_by_instance_id,
    get_active_profile_instance_id_by_name,
)


class TestGetActiveProfileByName:
    """Tests for get_active_profile_by_name function."""

    def test_finds_single_active_profile(self):
        """Test finding a single active profile by name."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
            }
        }

        result = get_active_profile_by_name(profiles_store, "TestProfile")

        assert result is not None
        assert result["name"] == "TestProfile"
        assert result["profile_id"] == "uuid-1"

    def test_returns_none_for_missing_name(self):
        """Test returns None when profile name not found."""
        profiles_store = {
            "uuid-1": {"name": "OtherProfile", "is_active": True, "is_deleted": False}
        }

        result = get_active_profile_by_name(profiles_store, "NonExistent")

        assert result is None

    def test_ignores_inactive_profiles(self):
        """Test that inactive profiles are not returned."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "TestProfile",
                "is_active": False,  # Inactive
                "is_deleted": False,
            }
        }

        result = get_active_profile_by_name(profiles_store, "TestProfile")

        assert result is None

    def test_ignores_deleted_profiles(self):
        """Test that deleted profiles are not returned."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": True,  # Deleted
            }
        }

        result = get_active_profile_by_name(profiles_store, "TestProfile")

        assert result is None

    def test_selects_highest_revision(self):
        """Test that highest revision is selected when multiple active versions exist."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
            },
            "uuid-2": {
                "profile_id": "uuid-2",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 3,  # Higher revision
            },
            "uuid-3": {
                "profile_id": "uuid-3",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 2,
            },
        }

        result = get_active_profile_by_name(profiles_store, "TestProfile")

        assert result["profile_id"] == "uuid-2"
        assert result["rev"] == 3

    def test_selects_newer_timestamp_for_same_revision(self):
        """Test that newer timestamp is selected when revisions are equal."""
        profiles_store = {
            "uuid-older": {
                "profile_id": "uuid-older",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
                "timestamp": "2024-01-01T10:00:00Z",
            },
            "uuid-newer": {
                "profile_id": "uuid-newer",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,  # Same revision
                "timestamp": "2024-01-02T10:00:00Z",  # Newer
            },
        }

        result = get_active_profile_by_name(profiles_store, "TestProfile")

        assert result["profile_id"] == "uuid-newer"

    def test_handles_missing_timestamp(self):
        """Test handling profiles with missing timestamps."""
        profiles_store = {
            "uuid-with-ts": {
                "profile_id": "uuid-with-ts",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
                "timestamp": "2024-01-01T10:00:00Z",
            },
            "uuid-no-ts": {
                "profile_id": "uuid-no-ts",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,  # Same revision, no timestamp
            },
        }

        # Should not crash, should return one of them
        result = get_active_profile_by_name(profiles_store, "TestProfile")
        assert result is not None
        assert result["name"] == "TestProfile"

    def test_returns_deep_copy(self):
        """Test that returned profile is a deep copy."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "nested": {"key": "value"},
            }
        }

        result = get_active_profile_by_name(profiles_store, "TestProfile")

        # Modify the result
        result["nested"]["key"] = "modified"

        # Original should be unchanged
        assert profiles_store["uuid-1"]["nested"]["key"] == "value"

    def test_empty_profiles_store(self):
        """Test with empty profiles store."""
        result = get_active_profile_by_name({}, "TestProfile")
        assert result is None

    def test_none_profiles_store(self):
        """Test with None profiles store."""
        result = get_active_profile_by_name(None, "TestProfile")
        assert result is None

    def test_none_name(self):
        """Test with None name."""
        profiles_store = {"uuid-1": {"name": "Test", "is_active": True, "is_deleted": False}}
        result = get_active_profile_by_name(profiles_store, None)
        assert result is None

    def test_empty_name(self):
        """Test with empty string name."""
        profiles_store = {"uuid-1": {"name": "Test", "is_active": True, "is_deleted": False}}
        result = get_active_profile_by_name(profiles_store, "")
        assert result is None

    def test_missing_rev_defaults_to_zero(self):
        """Test profiles without rev field default to 0."""
        profiles_store = {
            "uuid-no-rev": {
                "profile_id": "uuid-no-rev",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                # No 'rev' field
            },
            "uuid-rev-1": {
                "profile_id": "uuid-rev-1",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
            },
        }

        result = get_active_profile_by_name(profiles_store, "TestProfile")

        # rev=1 should win over no rev (defaults to 0)
        assert result["profile_id"] == "uuid-rev-1"


class TestGetProfileByInstanceId:
    """Tests for get_profile_by_instance_id function."""

    def test_finds_profile_by_id(self):
        """Test finding profile by instance ID."""
        profiles_store = {
            "uuid-abc": {
                "profile_id": "uuid-abc",
                "name": "TestProfile",
                "is_deleted": False,
            }
        }

        result = get_profile_by_instance_id(profiles_store, "uuid-abc")

        assert result is not None
        assert result["profile_id"] == "uuid-abc"

    def test_returns_none_for_missing_id(self):
        """Test returns None for non-existent instance ID."""
        profiles_store = {"uuid-1": {"name": "Test"}}

        result = get_profile_by_instance_id(profiles_store, "nonexistent")

        assert result is None

    def test_returns_none_for_deleted_profile(self):
        """Test returns None for deleted profile."""
        profiles_store = {
            "uuid-deleted": {
                "profile_id": "uuid-deleted",
                "name": "DeletedProfile",
                "is_deleted": True,
            }
        }

        result = get_profile_by_instance_id(profiles_store, "uuid-deleted")

        assert result is None

    def test_returns_deep_copy(self):
        """Test that returned profile is a deep copy."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "Test",
                "is_deleted": False,
                "data": {"nested": "value"},
            }
        }

        result = get_profile_by_instance_id(profiles_store, "uuid-1")
        result["data"]["nested"] = "modified"

        # Original unchanged
        assert profiles_store["uuid-1"]["data"]["nested"] == "value"

    def test_empty_profiles_store(self):
        """Test with empty profiles store."""
        result = get_profile_by_instance_id({}, "uuid-1")
        assert result is None

    def test_none_profiles_store(self):
        """Test with None profiles store."""
        result = get_profile_by_instance_id(None, "uuid-1")
        assert result is None

    def test_none_instance_id(self):
        """Test with None instance ID."""
        profiles_store = {"uuid-1": {"name": "Test"}}
        result = get_profile_by_instance_id(profiles_store, None)
        assert result is None

    def test_empty_instance_id(self):
        """Test with empty string instance ID."""
        profiles_store = {"uuid-1": {"name": "Test"}}
        result = get_profile_by_instance_id(profiles_store, "")
        assert result is None


class TestGetActiveProfileInstanceIdByName:
    """Tests for get_active_profile_instance_id_by_name function."""

    def test_returns_instance_id(self):
        """Test returns instance ID for active profile."""
        profiles_store = {
            "uuid-target": {
                "profile_id": "uuid-target",
                "name": "TargetProfile",
                "is_active": True,
                "is_deleted": False,
            }
        }

        result = get_active_profile_instance_id_by_name(profiles_store, "TargetProfile")

        assert result == "uuid-target"

    def test_returns_none_when_not_found(self):
        """Test returns None when profile not found."""
        profiles_store = {}

        result = get_active_profile_instance_id_by_name(profiles_store, "NonExistent")

        assert result is None

    def test_returns_highest_rev_instance_id(self):
        """Test returns instance ID of highest revision."""
        profiles_store = {
            "uuid-rev1": {
                "profile_id": "uuid-rev1",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
            },
            "uuid-rev2": {
                "profile_id": "uuid-rev2",
                "name": "TestProfile",
                "is_active": True,
                "is_deleted": False,
                "rev": 2,
            },
        }

        result = get_active_profile_instance_id_by_name(profiles_store, "TestProfile")

        assert result == "uuid-rev2"


class TestTimestampParsing:
    """Tests for timestamp parsing edge cases."""

    def test_handles_iso_format_with_z(self):
        """Test timestamp parsing with Z timezone."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "Test",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
                "timestamp": "2024-06-15T14:30:00Z",
            },
            "uuid-2": {
                "profile_id": "uuid-2",
                "name": "Test",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
                "timestamp": "2024-06-15T14:30:01Z",  # 1 second newer
            },
        }

        result = get_active_profile_by_name(profiles_store, "Test")
        assert result["profile_id"] == "uuid-2"

    def test_handles_iso_format_with_offset(self):
        """Test timestamp parsing with timezone offset."""
        profiles_store = {
            "uuid-1": {
                "profile_id": "uuid-1",
                "name": "Test",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
                "timestamp": "2024-06-15T14:30:00+00:00",
            },
        }

        result = get_active_profile_by_name(profiles_store, "Test")
        assert result is not None

    def test_handles_invalid_timestamp_gracefully(self):
        """Test that invalid timestamps don't crash."""
        profiles_store = {
            "uuid-bad-ts": {
                "profile_id": "uuid-bad-ts",
                "name": "Test",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
                "timestamp": "not-a-valid-timestamp",
            },
            "uuid-good-ts": {
                "profile_id": "uuid-good-ts",
                "name": "Test",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }

        # Should not crash and should return one of them
        result = get_active_profile_by_name(profiles_store, "Test")
        assert result is not None


class TestMultipleProfiles:
    """Tests with multiple profiles in store."""

    def test_finds_correct_profile_among_many(self):
        """Test finding correct profile in a store with many profiles."""
        profiles_store = {
            f"uuid-{i}": {
                "profile_id": f"uuid-{i}",
                "name": f"Profile_{i}",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
            }
            for i in range(100)
        }

        result = get_active_profile_by_name(profiles_store, "Profile_42")

        assert result["profile_id"] == "uuid-42"

    def test_handles_mix_of_active_inactive_deleted(self):
        """Test with mix of active, inactive, and deleted profiles."""
        profiles_store = {
            "uuid-active": {
                "profile_id": "uuid-active",
                "name": "Target",
                "is_active": True,
                "is_deleted": False,
                "rev": 1,
            },
            "uuid-inactive": {
                "profile_id": "uuid-inactive",
                "name": "Target",
                "is_active": False,
                "is_deleted": False,
                "rev": 2,  # Higher rev but inactive
            },
            "uuid-deleted": {
                "profile_id": "uuid-deleted",
                "name": "Target",
                "is_active": True,
                "is_deleted": True,
                "rev": 3,  # Highest rev but deleted
            },
        }

        result = get_active_profile_by_name(profiles_store, "Target")

        # Should only find the active, non-deleted one
        assert result["profile_id"] == "uuid-active"
