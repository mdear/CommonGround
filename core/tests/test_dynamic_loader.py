"""
Unit tests for agent_core.framework.dynamic_loader module.

This module tests the dynamic callable loading functionality
used to import functions/classes from string paths at runtime.

Key functionality tested:
- get_callable_from_path: Dynamic import from 'module.function' strings
- Error handling for invalid paths, missing modules, missing attributes
"""

import pytest
from agent_core.framework.dynamic_loader import get_callable_from_path


class TestGetCallableFromPath:
    """Tests for get_callable_from_path function."""

    def test_loads_builtin_function(self):
        """Test loading a builtin function."""
        # json.dumps is a standard library function
        func = get_callable_from_path("json.dumps")

        assert callable(func)
        # Verify it's actually json.dumps by using it
        result = func({"key": "value"})
        assert result == '{"key": "value"}'

    def test_loads_function_from_standard_library(self):
        """Test loading from standard library modules."""
        func = get_callable_from_path("os.path.join")

        assert callable(func)
        assert func("a", "b") == "a/b" or func("a", "b") == "a\\b"  # OS-dependent

    def test_loads_class_from_module(self):
        """Test loading a class."""
        cls = get_callable_from_path("datetime.datetime")

        assert cls is not None
        # Verify it's the datetime class
        instance = cls(2024, 1, 1)
        assert instance.year == 2024

    def test_loads_deeply_nested_path(self):
        """Test loading from deeply nested module path."""
        func = get_callable_from_path("urllib.parse.urlparse")

        assert callable(func)
        result = func("http://example.com/path")
        assert result.netloc == "example.com"

    def test_raises_import_error_for_missing_module(self):
        """Test raises error for non-existent module."""
        with pytest.raises(ImportError):
            get_callable_from_path("nonexistent_module_xyz.some_function")

    def test_raises_attribute_error_for_missing_function(self):
        """Test raises error for non-existent function in valid module."""
        with pytest.raises(AttributeError):
            get_callable_from_path("json.nonexistent_function_xyz")

    def test_raises_value_error_for_no_dot(self):
        """Test raises error for path without dot separator."""
        with pytest.raises(ValueError):
            get_callable_from_path("nodotpath")

    def test_raises_for_empty_string(self):
        """Test raises error for empty string path."""
        with pytest.raises(ValueError):
            get_callable_from_path("")

    def test_loads_from_agent_core_module(self):
        """Test loading from the agent_core package itself."""
        # Load a known function from agent_core
        func = get_callable_from_path(
            "agent_core.utils.context_helpers.get_nested_value_from_context"
        )

        assert callable(func)

    def test_loads_constant_or_module_attribute(self):
        """Test loading a module-level constant/attribute."""
        # sys.version is a string attribute, not callable
        version = get_callable_from_path("sys.version")

        assert isinstance(version, str)
        assert "." in version  # Version string contains dots

    def test_caches_import_appropriately(self):
        """Test that repeated calls work (importlib handles caching)."""
        func1 = get_callable_from_path("json.loads")
        func2 = get_callable_from_path("json.loads")

        # Should be the same function
        assert func1 is func2


class TestEdgeCases:
    """Edge case tests for dynamic loader."""

    def test_path_with_multiple_dots(self):
        """Test path with many nested modules."""
        func = get_callable_from_path("email.mime.text.MIMEText")

        assert func is not None

    def test_trailing_dot_raises(self):
        """Test that trailing dot raises error."""
        with pytest.raises((ValueError, ImportError, AttributeError)):
            get_callable_from_path("json.")

    def test_leading_dot_raises(self):
        """Test that leading dot raises error (TypeError from importlib for relative import)."""
        with pytest.raises((ValueError, ImportError, TypeError)):
            get_callable_from_path(".json.dumps")

    def test_double_dot_raises(self):
        """Test that double dot raises error."""
        with pytest.raises((ValueError, ImportError)):
            get_callable_from_path("json..dumps")


class TestRealWorldUseCases:
    """Tests simulating real-world usage patterns."""

    def test_load_custom_node_class(self):
        """Test loading a custom node class pattern."""
        # This simulates how the agent system loads custom nodes
        # Using a standard library class as proxy
        cls = get_callable_from_path("collections.OrderedDict")

        instance = cls()
        instance["key"] = "value"
        assert list(instance.keys()) == ["key"]

    def test_load_strategy_function(self):
        """Test loading a strategy/handler function pattern."""
        # Simulates loading event handlers dynamically
        func = get_callable_from_path("operator.add")

        assert func(2, 3) == 5

    def test_load_validation_callable(self):
        """Test loading a validation function."""
        func = get_callable_from_path("re.compile")

        pattern = func(r"\d+")
        assert pattern.match("123") is not None
