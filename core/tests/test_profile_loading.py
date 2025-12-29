"""
Tests for profile loading and toolset access verification.

These tests verify that the profile changes we made (adding flow_control_summary
to Base_Associate) are properly inherited and that profile loading works correctly.
"""
import pytest
import sys
from pathlib import Path

CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))

from agent_profiles.loader import (
    AGENT_PROFILES,
    get_global_active_profile_by_logical_name_copy,
    _deep_merge,
)


class TestProfileInheritance:
    """Tests for profile inheritance mechanism."""

    def test_deep_merge_simple_dict(self):
        """Simple dict merge should work."""
        parent = {"a": 1, "b": 2}
        child = {"b": 3, "c": 4}
        result = _deep_merge(parent, child)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested_dict(self):
        """Nested dict merge should be recursive."""
        parent = {"outer": {"inner1": 1, "inner2": 2}}
        child = {"outer": {"inner2": 3, "inner3": 4}}
        result = _deep_merge(parent, child)

        assert result["outer"]["inner1"] == 1
        assert result["outer"]["inner2"] == 3
        assert result["outer"]["inner3"] == 4

    def test_deep_merge_list_with_ids(self):
        """Lists of dicts with 'id' keys should merge by id."""
        parent = {"items": [{"id": "a", "val": 1}, {"id": "b", "val": 2}]}
        child = {"items": [{"id": "b", "val": 3}, {"id": "c", "val": 4}]}
        result = _deep_merge(parent, child)

        # Should have 3 items: a from parent, b from child (overwritten), c from child
        assert len(result["items"]) == 3
        ids = {item["id"] for item in result["items"]}
        assert ids == {"a", "b", "c"}

        # b should have child's value
        b_item = next(item for item in result["items"] if item["id"] == "b")
        assert b_item["val"] == 3

    def test_deep_merge_simple_list(self):
        """Simple lists should concatenate and dedupe."""
        parent = {"tags": ["a", "b"]}
        child = {"tags": ["b", "c"]}
        result = _deep_merge(parent, child)

        assert set(result["tags"]) == {"a", "b", "c"}


class TestBaseAssociateToolsets:
    """Tests for Base_Associate toolset configuration."""

    def test_base_associate_exists(self):
        """Base_Associate profile should exist."""
        profile = get_global_active_profile_by_logical_name_copy("Base_Associate")
        assert profile is not None
        assert profile["name"] == "Base_Associate"

    def test_base_associate_has_flow_control_end(self):
        """Base_Associate should have flow_control_end toolset."""
        profile = get_global_active_profile_by_logical_name_copy("Base_Associate")
        assert profile is not None

        allowed = profile.get("tool_access_policy", {}).get("allowed_toolsets", [])
        assert "flow_control_end" in allowed

    def test_base_associate_has_flow_control_summary(self):
        """Base_Associate should have flow_control_summary toolset (our fix)."""
        profile = get_global_active_profile_by_logical_name_copy("Base_Associate")
        assert profile is not None

        allowed = profile.get("tool_access_policy", {}).get("allowed_toolsets", [])
        assert "flow_control_summary" in allowed, \
            "Base_Associate should have flow_control_summary - this is the toolset fix we added"


class TestAssociateProfilesInheritToolsets:
    """Tests that Associate profiles inherit toolsets from Base_Associate."""

    @pytest.fixture
    def associate_profiles(self):
        """Get all profiles that inherit from Base_Associate."""
        associates = []
        for profile in AGENT_PROFILES.values():
            if profile.get("base_profile") == "Base_Associate" or \
               profile.get("type") == "associate":
                associates.append(profile)
        return associates

    def test_associates_have_base_toolsets(self, associate_profiles):
        """All Associate profiles should have the base toolsets."""
        for profile in associate_profiles:
            allowed = profile.get("tool_access_policy", {}).get("allowed_toolsets", [])

            # Should inherit flow_control_end
            assert "flow_control_end" in allowed, \
                f"{profile['name']} missing flow_control_end"

            # Should inherit flow_control_summary (our fix)
            assert "flow_control_summary" in allowed, \
                f"{profile['name']} missing flow_control_summary - inheritance may be broken"

    def test_localrag_has_summarization_access(self):
        """Associate_LocalRAG should have access to generate_message_summary."""
        profile = get_global_active_profile_by_logical_name_copy("Associate_LocalRAG_EN")

        if profile is None:
            pytest.skip("Associate_LocalRAG_EN profile not found")

        allowed = profile.get("tool_access_policy", {}).get("allowed_toolsets", [])
        assert "flow_control_summary" in allowed, \
            "LocalRAG needs flow_control_summary to call generate_message_summary"

    def test_smartrag_has_summarization_access(self):
        """Associate_SmartRAG should have access to generate_message_summary."""
        profile = get_global_active_profile_by_logical_name_copy("Associate_SmartRAG_EN")

        if profile is None:
            pytest.skip("Associate_SmartRAG_EN profile not found")

        allowed = profile.get("tool_access_policy", {}).get("allowed_toolsets", [])
        assert "flow_control_summary" in allowed, \
            "SmartRAG needs flow_control_summary to call generate_message_summary"


class TestProfileToolReferences:
    """Tests that profiles don't reference non-existent tools."""

    def test_localrag_references_correct_tool(self):
        """LocalRAG should reference generate_message_summary, not handover_to_summary."""
        profile = get_global_active_profile_by_logical_name_copy("Associate_LocalRAG_EN")

        if profile is None:
            pytest.skip("Associate_LocalRAG_EN profile not found")

        # Convert profile to string to search for tool references
        profile_str = str(profile)

        # Should NOT reference the non-existent tool
        assert "handover_to_summary" not in profile_str, \
            "LocalRAG still references non-existent handover_to_summary tool"

        # Should reference the correct tool
        assert "generate_message_summary" in profile_str, \
            "LocalRAG should reference generate_message_summary"

    def test_smartrag_references_correct_tool(self):
        """SmartRAG should reference generate_message_summary, not handover_to_summary."""
        profile = get_global_active_profile_by_logical_name_copy("Associate_SmartRAG_EN")

        if profile is None:
            pytest.skip("Associate_SmartRAG_EN profile not found")

        profile_str = str(profile)

        assert "handover_to_summary" not in profile_str, \
            "SmartRAG still references non-existent handover_to_summary tool"

        assert "generate_message_summary" in profile_str, \
            "SmartRAG should reference generate_message_summary"


class TestProfileLoading:
    """Tests for general profile loading functionality."""

    def test_profiles_loaded(self):
        """At least some profiles should be loaded."""
        assert len(AGENT_PROFILES) > 0

    def test_base_agent_exists(self):
        """Base_Agent profile should exist."""
        profile = get_global_active_profile_by_logical_name_copy("Base_Agent")
        assert profile is not None

    def test_profiles_have_required_fields(self):
        """All profiles should have required metadata fields."""
        for profile_id, profile in AGENT_PROFILES.items():
            assert "name" in profile, f"Profile {profile_id} missing name"
            assert "profile_id" in profile, f"Profile {profile_id} missing profile_id"
            assert "is_active" in profile, f"Profile {profile_id} missing is_active"

    def test_get_profile_returns_copy(self):
        """get_global_active_profile_by_logical_name_copy should return a copy."""
        profile1 = get_global_active_profile_by_logical_name_copy("Base_Agent")
        profile2 = get_global_active_profile_by_logical_name_copy("Base_Agent")

        if profile1 is None:
            pytest.skip("Base_Agent not found")

        # Should be equal but not the same object
        assert profile1 == profile2
        assert profile1 is not profile2

        # Modifying one should not affect the other
        profile1["test_mutation"] = True
        assert "test_mutation" not in profile2
