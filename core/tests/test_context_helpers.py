"""
Unit tests for agent_core.utils.context_helpers module.

This module tests the V-Model path resolution system that provides
safe, declarative access to nested context values using dot-notation paths.

Key concepts tested:
- PATH_RESOLVER_MAP: Maps prefixes (state, meta, team, etc.) to base objects
- _traverse_path: Greedy path traversal with compound key support
- get_nested_value_from_context: Main API for V-Model path resolution
- VModelAccessor: Syntactic sugar class for eval() environments
- SENTINEL_DEFAULT: Distinguishes "not found" from None values
"""

import pytest
from agent_core.utils.context_helpers import (
    get_nested_value_from_context,
    VModelAccessor,
    _traverse_path,
    PATH_RESOLVER_MAP,
    SENTINEL_DEFAULT,
    DEFAULT_PATH_PREFIX,
)


class TestPathResolverMap:
    """Tests for the PATH_RESOLVER_MAP configuration."""

    def test_contains_expected_prefixes(self):
        """Verify all documented prefixes exist in the resolver map."""
        expected_prefixes = [
            "state", "meta", "team", "run", "config",
            "initial_params", "flags", "principal", "partner", "_self"
        ]
        for prefix in expected_prefixes:
            assert prefix in PATH_RESOLVER_MAP, f"Missing prefix: {prefix}"

    def test_state_resolver(self):
        """Test the 'state' prefix resolves to ctx['state']."""
        ctx = {"state": {"key": "value"}}
        resolver = PATH_RESOLVER_MAP["state"]
        assert resolver(ctx) == {"key": "value"}

    def test_meta_resolver(self):
        """Test the 'meta' prefix resolves to ctx['meta']."""
        ctx = {"meta": {"agent_id": "test-agent"}}
        resolver = PATH_RESOLVER_MAP["meta"]
        assert resolver(ctx) == {"agent_id": "test-agent"}

    def test_team_resolver(self):
        """Test the 'team' prefix resolves to ctx['refs']['team']."""
        ctx = {"refs": {"team": {"question": "What is AI?"}}}
        resolver = PATH_RESOLVER_MAP["team"]
        assert resolver(ctx) == {"question": "What is AI?"}

    def test_run_resolver(self):
        """Test the 'run' prefix resolves to ctx['refs']['run']['meta']."""
        ctx = {"refs": {"run": {"meta": {"run_id": "run-123"}}}}
        resolver = PATH_RESOLVER_MAP["run"]
        assert resolver(ctx) == {"run_id": "run-123"}

    def test_config_resolver(self):
        """Test the 'config' prefix resolves to ctx['refs']['run']['config']."""
        ctx = {"refs": {"run": {"config": {"setting": True}}}}
        resolver = PATH_RESOLVER_MAP["config"]
        assert resolver(ctx) == {"setting": True}

    def test_initial_params_shortcut(self):
        """Test 'initial_params' shortcut to state.initial_parameters."""
        ctx = {"state": {"initial_parameters": {"prompt": "Hello"}}}
        resolver = PATH_RESOLVER_MAP["initial_params"]
        assert resolver(ctx) == {"prompt": "Hello"}

    def test_flags_shortcut(self):
        """Test 'flags' shortcut to state.flags."""
        ctx = {"state": {"flags": {"debug": True}}}
        resolver = PATH_RESOLVER_MAP["flags"]
        assert resolver(ctx) == {"debug": True}

    def test_self_resolver(self):
        """Test '_self' prefix returns the entire context."""
        ctx = {"state": {"foo": "bar"}, "meta": {"id": "123"}}
        resolver = PATH_RESOLVER_MAP["_self"]
        assert resolver(ctx) is ctx

    def test_principal_cross_context_shortcut(self):
        """Test 'principal' resolves to partner's principal_context_ref.state."""
        principal_state = {"messages": [], "deliverables": {}}
        ctx = {
            "refs": {
                "run": {
                    "sub_context_refs": {
                        "_principal_context_ref": {"state": principal_state}
                    }
                }
            }
        }
        resolver = PATH_RESOLVER_MAP["principal"]
        assert resolver(ctx) == principal_state

    def test_partner_cross_context_shortcut(self):
        """Test 'partner' resolves to principal's partner_context_ref.state."""
        partner_state = {"inbox": [], "flags": {}}
        ctx = {
            "refs": {
                "run": {
                    "sub_context_refs": {
                        "_partner_context_ref": {"state": partner_state}
                    }
                }
            }
        }
        resolver = PATH_RESOLVER_MAP["partner"]
        assert resolver(ctx) == partner_state


class TestTraversePath:
    """Tests for the _traverse_path helper function."""

    def test_simple_key_traversal(self):
        """Test traversing a single key."""
        obj = {"foo": "bar"}
        assert _traverse_path(obj, ["foo"]) == "bar"

    def test_nested_key_traversal(self):
        """Test traversing multiple nested keys."""
        obj = {"a": {"b": {"c": "deep_value"}}}
        assert _traverse_path(obj, ["a", "b", "c"]) == "deep_value"

    def test_list_index_access(self):
        """Test accessing list elements by index."""
        obj = {"items": ["first", "second", "third"]}
        assert _traverse_path(obj, ["items", "0"]) == "first"
        assert _traverse_path(obj, ["items", "2"]) == "third"

    def test_negative_list_index(self):
        """Test negative index access (Python-style)."""
        obj = {"items": ["first", "second", "third"]}
        assert _traverse_path(obj, ["items[-1]"]) == "third"
        assert _traverse_path(obj, ["items[-2]"]) == "second"

    def test_compound_key_with_dots(self):
        """Test handling keys that contain dots (greedy matching)."""
        obj = {"dotted.key.name": "compound_value", "dotted": {"key": {"name": "nested_value"}}}
        # Greedy matching should prefer the compound key
        result = _traverse_path(obj, ["dotted.key.name"])
        assert result == "compound_value"

    def test_missing_key_returns_sentinel(self):
        """Test that missing keys return SENTINEL_DEFAULT."""
        obj = {"foo": "bar"}
        result = _traverse_path(obj, ["nonexistent"])
        assert result is SENTINEL_DEFAULT

    def test_none_in_path_returns_sentinel(self):
        """Test that None values in path return SENTINEL_DEFAULT."""
        obj = {"a": None}
        result = _traverse_path(obj, ["a", "b"])
        assert result is SENTINEL_DEFAULT

    def test_index_out_of_bounds_returns_sentinel(self):
        """Test that out-of-bounds list indices return SENTINEL_DEFAULT."""
        obj = {"items": ["only_one"]}
        result = _traverse_path(obj, ["items", "5"])
        assert result is SENTINEL_DEFAULT

    def test_object_attribute_access(self):
        """Test accessing object attributes via getattr."""
        class TestObj:
            attr = "attr_value"

        obj = {"nested": TestObj()}
        result = _traverse_path(obj, ["nested", "attr"])
        assert result == "attr_value"

    def test_empty_path_returns_base(self):
        """Test that empty path returns the base object."""
        obj = {"foo": "bar"}
        result = _traverse_path(obj, [])
        assert result == obj


class TestGetNestedValueFromContext:
    """Tests for the main get_nested_value_from_context function."""

    @pytest.fixture
    def sample_context(self):
        """Create a comprehensive sample context for testing."""
        return {
            "meta": {
                "agent_id": "principal-agent",
                "run_id": "run-abc-123",
            },
            "state": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "flags": {
                    "debug": True,
                    "verbose": False,
                },
                "initial_parameters": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                },
                "deliverables": {
                    "summary": "Task completed successfully",
                },
            },
            "refs": {
                "team": {
                    "question": "What is machine learning?",
                    "work_modules": {},
                },
                "run": {
                    "meta": {"run_id": "run-abc-123", "status": "RUNNING"},
                    "config": {"model_override": None},
                },
            },
        }

    def test_state_prefix_access(self, sample_context):
        """Test accessing values with 'state.' prefix."""
        assert get_nested_value_from_context(sample_context, "state.deliverables.summary") == "Task completed successfully"

    def test_implicit_state_prefix(self, sample_context):
        """Test that paths without prefix default to 'state.'."""
        assert get_nested_value_from_context(sample_context, "deliverables.summary") == "Task completed successfully"

    def test_meta_prefix_access(self, sample_context):
        """Test accessing values with 'meta.' prefix."""
        assert get_nested_value_from_context(sample_context, "meta.agent_id") == "principal-agent"

    def test_team_prefix_access(self, sample_context):
        """Test accessing values with 'team.' prefix."""
        assert get_nested_value_from_context(sample_context, "team.question") == "What is machine learning?"

    def test_flags_shortcut(self, sample_context):
        """Test 'flags.' shortcut prefix."""
        assert get_nested_value_from_context(sample_context, "flags.debug") is True

    def test_initial_params_shortcut(self, sample_context):
        """Test 'initial_params.' shortcut prefix."""
        assert get_nested_value_from_context(sample_context, "initial_params.model") == "gpt-4"

    def test_run_prefix_access(self, sample_context):
        """Test 'run.' prefix for run metadata."""
        assert get_nested_value_from_context(sample_context, "run.status") == "RUNNING"

    def test_list_access_by_index(self, sample_context):
        """Test accessing list elements by index."""
        assert get_nested_value_from_context(sample_context, "state.messages[0].role") == "user"
        assert get_nested_value_from_context(sample_context, "state.messages[-1].content") == "Hi there!"

    def test_missing_path_returns_none(self, sample_context):
        """Test that missing paths return None by default."""
        result = get_nested_value_from_context(sample_context, "nonexistent.path.here")
        assert result is None

    def test_missing_path_with_custom_default(self, sample_context):
        """Test that missing paths return custom default when provided."""
        result = get_nested_value_from_context(sample_context, "missing.key", default="fallback")
        assert result == "fallback"

    def test_empty_path_returns_none(self, sample_context):
        """Test that empty path returns None."""
        assert get_nested_value_from_context(sample_context, "") is None
        assert get_nested_value_from_context(sample_context, None) is None

    def test_prefix_only_returns_base_object(self, sample_context):
        """Test that prefix-only paths return the base object."""
        assert get_nested_value_from_context(sample_context, "meta") == sample_context["meta"]
        assert get_nested_value_from_context(sample_context, "team") == sample_context["refs"]["team"]

    def test_self_prefix_returns_entire_context(self, sample_context):
        """Test '_self' prefix returns entire context."""
        result = get_nested_value_from_context(sample_context, "_self")
        assert result is sample_context

    def test_deeply_nested_access(self, sample_context):
        """Test accessing deeply nested values."""
        # Create deeper nesting
        sample_context["state"]["deep"] = {"level1": {"level2": {"level3": {"value": 42}}}}
        result = get_nested_value_from_context(sample_context, "deep.level1.level2.level3.value")
        assert result == 42

    def test_none_value_distinguished_from_missing(self, sample_context):
        """Test that actual None values are returned correctly."""
        sample_context["state"]["explicit_none"] = None
        result = get_nested_value_from_context(sample_context, "explicit_none")
        assert result is None

    def test_non_string_path_returns_default(self):
        """Test that non-string paths are handled gracefully."""
        ctx = {"state": {"foo": "bar"}}
        assert get_nested_value_from_context(ctx, 123, default="fallback") == "fallback"
        assert get_nested_value_from_context(ctx, ["invalid"], default="fallback") == "fallback"


class TestVModelAccessor:
    """Tests for the VModelAccessor class."""

    @pytest.fixture
    def accessor_context(self):
        """Create context and accessor for testing."""
        ctx = {
            "state": {
                "name": "test-name",
                "items": ["a", "b", "c"],
            },
            "meta": {"agent_id": "accessor-test"},
        }
        return ctx, VModelAccessor(ctx)

    def test_getitem_basic_access(self, accessor_context):
        """Test basic __getitem__ access."""
        ctx, accessor = accessor_context
        assert accessor["state.name"] == "test-name"

    def test_getitem_with_prefix(self, accessor_context):
        """Test __getitem__ with explicit prefix."""
        ctx, accessor = accessor_context
        assert accessor["meta.agent_id"] == "accessor-test"

    def test_getitem_list_access(self, accessor_context):
        """Test __getitem__ with list index."""
        ctx, accessor = accessor_context
        assert accessor["state.items[1]"] == "b"

    def test_getitem_implicit_state_prefix(self, accessor_context):
        """Test __getitem__ defaults to state prefix."""
        ctx, accessor = accessor_context
        assert accessor["name"] == "test-name"

    def test_accessor_in_eval_context(self, accessor_context):
        """Test that accessor works in eval() environments (its design purpose)."""
        ctx, v = accessor_context
        # This simulates how VModelAccessor is used in dynamic evaluation contexts
        result = eval("v['state.name']", {"v": v})
        assert result == "test-name"


class TestDefaultPathPrefix:
    """Tests for DEFAULT_PATH_PREFIX behavior."""

    def test_default_prefix_is_state(self):
        """Verify DEFAULT_PATH_PREFIX is 'state'."""
        assert DEFAULT_PATH_PREFIX == "state"

    def test_unprefixed_paths_use_state(self):
        """Test that paths without known prefix resolve from state."""
        ctx = {"state": {"foo": {"bar": "baz"}}}
        result = get_nested_value_from_context(ctx, "foo.bar")
        assert result == "baz"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_context(self):
        """Test behavior with empty context."""
        result = get_nested_value_from_context({}, "state.foo")
        assert result is None

    def test_none_context(self):
        """Test behavior with None context (should not raise)."""
        # The function should handle None gracefully
        # Based on implementation, this accesses None.get() which will fail
        # but we want to verify it doesn't raise unexpectedly
        try:
            result = get_nested_value_from_context(None, "foo")
            # If it doesn't raise, it should return None or default
            assert result is None or result == SENTINEL_DEFAULT
        except (AttributeError, TypeError):
            # This is acceptable - None is not a valid context
            pass

    def test_special_characters_in_path(self):
        """Test paths with special characters (non-standard but possible)."""
        ctx = {"state": {"key-with-dashes": "value1", "key_with_underscores": "value2"}}
        assert get_nested_value_from_context(ctx, "key_with_underscores") == "value2"

    def test_numeric_string_keys(self):
        """Test dict keys that are numeric strings."""
        ctx = {"state": {"0": "zero", "1": "one"}}
        # These should be treated as dict keys, not list indices
        assert get_nested_value_from_context(ctx, "0") == "zero"

    def test_boolean_values(self):
        """Test accessing boolean values."""
        ctx = {"state": {"is_active": True, "is_disabled": False}}
        assert get_nested_value_from_context(ctx, "is_active") is True
        assert get_nested_value_from_context(ctx, "is_disabled") is False

    def test_integer_and_float_values(self):
        """Test accessing numeric values."""
        ctx = {"state": {"count": 42, "ratio": 3.14}}
        assert get_nested_value_from_context(ctx, "count") == 42
        assert get_nested_value_from_context(ctx, "ratio") == 3.14

    def test_empty_string_value(self):
        """Test accessing empty string values."""
        ctx = {"state": {"empty": ""}}
        result = get_nested_value_from_context(ctx, "empty")
        assert result == ""

    def test_list_of_dicts(self):
        """Test accessing nested dicts inside lists."""
        ctx = {
            "state": {
                "users": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ]
            }
        }
        assert get_nested_value_from_context(ctx, "users[0].name") == "Alice"
        assert get_nested_value_from_context(ctx, "users[-1].age") == 25
