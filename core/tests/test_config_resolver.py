"""
Unit tests for agent_core.llm.config_resolver module.

This module tests the LLMConfigResolver class that converts YAML-based
LLM configurations into final LiteLLM parameters, with support for
environment variables and file-based secrets.

Key functionality tested:
- _recursive_resolve: Directive parsing (_type: from_env, json_from_file)
- Environment variable resolution with defaults
- JSON parsing from env vars
- Boolean/null conversion from string env vars
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from agent_core.llm.config_resolver import LLMConfigResolver


class TestRecursiveResolveFromEnv:
    """Tests for _recursive_resolve with from_env directive."""

    @pytest.fixture
    def resolver(self):
        """Create a resolver with empty shared configs."""
        return LLMConfigResolver({})

    def test_simple_env_var_resolution(self, resolver):
        """Test resolving a simple environment variable."""
        config = {"_type": "from_env", "var": "TEST_VAR"}

        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = resolver._recursive_resolve(config)

        assert result == "test_value"

    def test_env_var_with_default_when_set(self, resolver):
        """Test env var with default when var is set."""
        config = {"_type": "from_env", "var": "SET_VAR", "default": "fallback"}

        with patch.dict(os.environ, {"SET_VAR": "actual_value"}):
            result = resolver._recursive_resolve(config)

        assert result == "actual_value"

    def test_env_var_with_default_when_unset(self, resolver):
        """Test env var with default when var is not set."""
        config = {"_type": "from_env", "var": "UNSET_VAR_XYZ", "default": "fallback"}

        # Ensure the var is not set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UNSET_VAR_XYZ", None)
            result = resolver._recursive_resolve(config)

        assert result == "fallback"

    def test_required_env_var_missing_raises(self, resolver):
        """Test that missing required env var raises ValueError."""
        config = {"_type": "from_env", "var": "REQUIRED_MISSING", "required": True}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REQUIRED_MISSING", None)
            with pytest.raises(ValueError) as exc_info:
                resolver._recursive_resolve(config)

        assert "REQUIRED_MISSING" in str(exc_info.value)

    def test_env_var_converts_true_string_to_bool(self, resolver):
        """Test that 'true' string is converted to boolean True."""
        config = {"_type": "from_env", "var": "BOOL_VAR"}

        with patch.dict(os.environ, {"BOOL_VAR": "true"}):
            result = resolver._recursive_resolve(config)

        assert result is True

    def test_env_var_converts_false_string_to_bool(self, resolver):
        """Test that 'false' string is converted to boolean False."""
        config = {"_type": "from_env", "var": "BOOL_VAR"}

        with patch.dict(os.environ, {"BOOL_VAR": "FALSE"}):  # Case insensitive
            result = resolver._recursive_resolve(config)

        assert result is False

    def test_env_var_converts_null_string_to_none(self, resolver):
        """Test that 'null' string is converted to None."""
        config = {"_type": "from_env", "var": "NULL_VAR"}

        with patch.dict(os.environ, {"NULL_VAR": "null"}):
            result = resolver._recursive_resolve(config)

        assert result is None

    def test_env_var_parses_json_object(self, resolver):
        """Test that JSON object string is parsed."""
        config = {"_type": "from_env", "var": "JSON_VAR"}
        json_value = '{"key": "value", "count": 42}'

        with patch.dict(os.environ, {"JSON_VAR": json_value}):
            result = resolver._recursive_resolve(config)

        assert result == {"key": "value", "count": 42}

    def test_env_var_parses_json_array(self, resolver):
        """Test that JSON array string is parsed."""
        config = {"_type": "from_env", "var": "ARRAY_VAR"}
        json_value = '["a", "b", "c"]'

        with patch.dict(os.environ, {"ARRAY_VAR": json_value}):
            result = resolver._recursive_resolve(config)

        assert result == ["a", "b", "c"]

    def test_env_var_invalid_json_returns_string(self, resolver):
        """Test that invalid JSON is returned as string."""
        config = {"_type": "from_env", "var": "BAD_JSON"}

        with patch.dict(os.environ, {"BAD_JSON": "{not valid json"}):
            result = resolver._recursive_resolve(config)

        # Should return as string since JSON parsing failed
        assert result == "{not valid json"

    def test_missing_var_key_raises(self, resolver):
        """Test that missing 'var' key raises ValueError."""
        config = {"_type": "from_env"}  # Missing 'var'

        with pytest.raises(ValueError) as exc_info:
            resolver._recursive_resolve(config)

        assert "var" in str(exc_info.value)

    def test_unset_optional_env_var_returns_none(self, resolver):
        """Test that unset optional env var returns None."""
        config = {"_type": "from_env", "var": "OPTIONAL_UNSET"}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPTIONAL_UNSET", None)
            result = resolver._recursive_resolve(config)

        assert result is None


class TestRecursiveResolveJsonFromFile:
    """Tests for _recursive_resolve with json_from_file directive."""

    @pytest.fixture
    def resolver(self):
        return LLMConfigResolver({})

    def test_loads_json_from_file(self, resolver):
        """Test loading JSON content from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"loaded": "from_file", "number": 123}, f)
            temp_path = f.name

        try:
            config = {"_type": "json_from_file", "path": temp_path}
            result = resolver._recursive_resolve(config)

            assert result == {"loaded": "from_file", "number": 123}
        finally:
            os.unlink(temp_path)

    def test_missing_path_key_raises(self, resolver):
        """Test that missing 'path' key raises ValueError."""
        config = {"_type": "json_from_file"}  # Missing 'path'

        with pytest.raises(ValueError) as exc_info:
            resolver._recursive_resolve(config)

        assert "path" in str(exc_info.value)

    def test_nonexistent_file_raises(self, resolver):
        """Test that nonexistent file raises FileNotFoundError."""
        config = {"_type": "json_from_file", "path": "/nonexistent/path/file.json"}

        with pytest.raises(FileNotFoundError):
            resolver._recursive_resolve(config)


class TestRecursiveResolvePassthrough:
    """Tests for _recursive_resolve with non-directive values."""

    @pytest.fixture
    def resolver(self):
        return LLMConfigResolver({})

    def test_passthrough_string(self, resolver):
        """Test that plain string passes through unchanged."""
        result = resolver._recursive_resolve("plain_string")
        assert result == "plain_string"

    def test_passthrough_number(self, resolver):
        """Test that numbers pass through unchanged."""
        assert resolver._recursive_resolve(42) == 42
        assert resolver._recursive_resolve(3.14) == 3.14

    def test_passthrough_none(self, resolver):
        """Test that None passes through unchanged."""
        assert resolver._recursive_resolve(None) is None

    def test_passthrough_list(self, resolver):
        """Test that list without _type passes through."""
        data = [1, 2, 3]
        assert resolver._recursive_resolve(data) == [1, 2, 3]

    def test_passthrough_dict_without_type(self, resolver):
        """Test that dict without _type passes through."""
        data = {"key": "value", "nested": {"inner": True}}
        assert resolver._recursive_resolve(data) == data

    def test_unknown_type_directive_returns_as_is(self, resolver):
        """Test that unknown _type directive returns config as-is."""
        config = {"_type": "unknown_directive", "data": "value"}
        result = resolver._recursive_resolve(config)

        assert result == config


class TestResolverResolveMethod:
    """Tests for the main resolve method."""

    def test_resolve_requires_llm_config_ref(self):
        """Test that missing llm_config_ref raises ValueError."""
        resolver = LLMConfigResolver({})
        profile = {"name": "TestProfile"}  # Missing llm_config_ref

        with pytest.raises(ValueError) as exc_info:
            resolver.resolve(profile)

        assert "llm_config_ref" in str(exc_info.value)

    def test_resolve_raises_for_missing_config(self):
        """Test that missing referenced config raises ValueError."""
        resolver = LLMConfigResolver({})
        profile = {"name": "TestProfile", "llm_config_ref": "NonExistent"}

        # Mock at source since it's a delayed import
        with patch("agent_profiles.loader.get_active_llm_config_by_name") as mock:
            mock.return_value = None

            with pytest.raises(ValueError) as exc_info:
                resolver.resolve(profile)

        assert "NonExistent" in str(exc_info.value)

    def test_resolve_processes_config_values(self):
        """Test that resolve processes all config values."""
        shared_configs = {}
        resolver = LLMConfigResolver(shared_configs)
        profile = {"name": "TestProfile", "llm_config_ref": "TestConfig"}

        base_config = {
            "name": "TestConfig",
            "is_active": True,
            "config": {
                "model": "gpt-4",
                "temperature": 0.7,
                "api_key": {"_type": "from_env", "var": "API_KEY"},
            }
        }

        with patch("agent_profiles.loader.get_active_llm_config_by_name") as mock:
            mock.return_value = base_config

            with patch.dict(os.environ, {"API_KEY": "secret-key"}):
                result = resolver.resolve(profile)

        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert result["api_key"] == "secret-key"

    def test_resolve_filters_none_values(self):
        """Test that None values are filtered from final params."""
        resolver = LLMConfigResolver({})
        profile = {"name": "TestProfile", "llm_config_ref": "TestConfig"}

        base_config = {
            "name": "TestConfig",
            "is_active": True,
            "config": {
                "model": "gpt-4",
                "optional_param": {"_type": "from_env", "var": "UNSET_OPTIONAL"},
            }
        }

        with patch("agent_profiles.loader.get_active_llm_config_by_name") as mock:
            mock.return_value = base_config

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("UNSET_OPTIONAL", None)
                result = resolver.resolve(profile)

        assert "model" in result
        assert "optional_param" not in result  # None value filtered out


class TestEnvVarEdgeCases:
    """Edge case tests for environment variable handling."""

    @pytest.fixture
    def resolver(self):
        return LLMConfigResolver({})

    def test_whitespace_in_json_env_var(self, resolver):
        """Test JSON parsing with leading/trailing whitespace."""
        config = {"_type": "from_env", "var": "WHITESPACE_JSON"}

        with patch.dict(os.environ, {"WHITESPACE_JSON": '  {"key": "value"}  '}):
            result = resolver._recursive_resolve(config)

        assert result == {"key": "value"}

    def test_empty_string_env_var(self, resolver):
        """Test handling of empty string env var."""
        config = {"_type": "from_env", "var": "EMPTY_VAR"}

        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            result = resolver._recursive_resolve(config)

        assert result == ""

    def test_integer_string_converted_to_int(self, resolver):
        """Test that integer strings are converted to int for API compatibility."""
        config = {"_type": "from_env", "var": "NUMBER_STR"}

        with patch.dict(os.environ, {"NUMBER_STR": "12345"}):
            result = resolver._recursive_resolve(config)

        # Should be converted to int for API compatibility (e.g., max_tokens)
        assert result == 12345
        assert isinstance(result, int)

    def test_float_string_converted_to_float(self, resolver):
        """Test that float strings are converted to float for API compatibility."""
        config = {"_type": "from_env", "var": "FLOAT_STR"}

        with patch.dict(os.environ, {"FLOAT_STR": "0.4"}):
            result = resolver._recursive_resolve(config)

        # Should be converted to float for API compatibility (e.g., temperature)
        assert result == 0.4
        assert isinstance(result, float)

    def test_temperature_string_converted_to_float(self, resolver):
        """Test that temperature env var string is properly converted to float.

        This tests the fix for Anthropic API error:
        'temperature: Input should be a valid number'
        """
        config = {"_type": "from_env", "var": "PRINCIPAL_TEMPERATURE", "default": 0.4}

        # Simulate env var set as string (common when exported in shell)
        with patch.dict(os.environ, {"PRINCIPAL_TEMPERATURE": "0.7"}):
            result = resolver._recursive_resolve(config)

        assert result == 0.7
        assert isinstance(result, float)

    def test_scientific_notation_converted_to_float(self, resolver):
        """Test that scientific notation strings are converted to float."""
        config = {"_type": "from_env", "var": "SCI_NUM"}

        with patch.dict(os.environ, {"SCI_NUM": "1e-5"}):
            result = resolver._recursive_resolve(config)

        assert result == 1e-5
        assert isinstance(result, float)

    def test_non_numeric_string_stays_string(self, resolver):
        """Test that non-numeric strings remain as strings."""
        config = {"_type": "from_env", "var": "TEXT_STR"}

        with patch.dict(os.environ, {"TEXT_STR": "hello-world"}):
            result = resolver._recursive_resolve(config)

        assert result == "hello-world"
        assert isinstance(result, str)

    def test_mixed_case_bool_conversion(self, resolver):
        """Test boolean conversion is case-insensitive."""
        config = {"_type": "from_env", "var": "MIXED_BOOL"}

        for bool_str in ["True", "TRUE", "true", "TrUe"]:
            with patch.dict(os.environ, {"MIXED_BOOL": bool_str}):
                result = resolver._recursive_resolve(config)
                assert result is True

    def test_default_can_be_any_type(self, resolver):
        """Test that default value can be any type."""
        # Default as dict
        config = {"_type": "from_env", "var": "UNSET", "default": {"nested": "default"}}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UNSET", None)
            result = resolver._recursive_resolve(config)

        assert result == {"nested": "default"}
