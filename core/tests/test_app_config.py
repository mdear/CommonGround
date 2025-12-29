"""
Unit tests for agent_core.config.app_config module.

This module tests MCP (Model Context Protocol) server configuration loading:
- Thread-safe configuration loading
- Environment variable and file-based config sources
- Category-based server organization
- Immutable proxy patterns for config access

Key functions tested:
- _load_mcp_config_internal: Internal config parser
- get_mcp_server_categories: Thread-safe category accessor
- get_native_mcp_servers: Thread-safe server config accessor
- reload_mcp_config: Hot-reload functionality
"""

import pytest
import os
import json
import tempfile
from types import MappingProxyType
from unittest.mock import patch, MagicMock

# Import the module to test
from agent_core.config import app_config


class TestMCPConfigLoading:
    """Tests for MCP configuration loading."""

    @pytest.fixture(autouse=True)
    def reset_config_state(self):
        """Reset the module's internal state before each test."""
        # Save original state
        original_loaded = app_config._mcp_config_loaded
        original_servers = app_config._native_mcp_servers_internal.copy()
        original_categories = app_config._mcp_server_categories_internal.copy()

        # Reset for test
        app_config._mcp_config_loaded = False
        app_config._native_mcp_servers_internal.clear()
        app_config._mcp_server_categories_internal.clear()

        yield

        # Restore original state
        app_config._mcp_config_loaded = original_loaded
        app_config._native_mcp_servers_internal.clear()
        app_config._native_mcp_servers_internal.update(original_servers)
        app_config._mcp_server_categories_internal.clear()
        app_config._mcp_server_categories_internal.update(original_categories)

    def test_load_from_json_file(self, tmp_path):
        """Test loading MCP config from a JSON file."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "test_server": {
                    "enabled": True,
                    "category": "user_specified",
                    "transport": {"type": "stdio", "command": "test"}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()

        assert "test_server" in servers
        assert servers["test_server"]["enabled"] is True
        assert categories["test_server"]["category"] == "user_specified"

    def test_load_from_env_var(self):
        """Test loading MCP config from environment variable."""
        config_data = {
            "mcpServers": {
                "env_server": {
                    "enabled": True,
                    "category": "google_related",
                    "transport": {"type": "sse", "url": "http://example.com"}
                }
            }
        }

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": "/nonexistent/path.json",
            "NATIVE_MCP_SERVERS_CONFIG": json.dumps(config_data),
        }):
            servers, categories = app_config._load_mcp_config_internal()

        assert "env_server" in servers
        assert categories["env_server"]["category"] == "google_related"

    def test_disabled_server_not_in_servers(self, tmp_path):
        """Test that disabled servers are excluded from the servers dict."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "disabled_server": {
                    "enabled": False,
                    "category": "user_specified",
                    "transport": {"type": "stdio", "command": "test"}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()

        # Server should be in categories but not in enabled servers
        assert "disabled_server" not in servers
        assert "disabled_server" in categories
        assert categories["disabled_server"]["enabled"] is False

    def test_server_without_transport_excluded(self, tmp_path):
        """Test servers without transport config are excluded."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "no_transport": {
                    "enabled": True,
                    "category": "user_specified"
                    # Missing 'transport' field
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()

        assert "no_transport" not in servers


class TestCategoryHandling:
    """Tests for MCP server category handling."""

    @pytest.fixture(autouse=True)
    def reset_config_state(self):
        """Reset config state for each test."""
        app_config._mcp_config_loaded = False
        app_config._native_mcp_servers_internal.clear()
        app_config._mcp_server_categories_internal.clear()
        yield
        app_config._mcp_config_loaded = False

    def test_default_category_is_uncategorized(self, tmp_path):
        """Test servers without explicit category default to 'uncategorized'."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "no_category_server": {
                    "enabled": True,
                    "transport": {"type": "stdio", "command": "test"}
                    # No 'category' field
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()

        assert categories["no_category_server"]["category"] == "uncategorized"
        assert categories["no_category_server"]["has_explicit_category"] is False

    def test_explicit_category_tracked(self, tmp_path):
        """Test explicit category is tracked correctly."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "categorized_server": {
                    "enabled": True,
                    "category": "user_specified",
                    "transport": {"type": "stdio", "command": "test"}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()

        assert categories["categorized_server"]["has_explicit_category"] is True

    def test_multiple_categories(self, tmp_path):
        """Test loading servers with different categories."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "google_server": {
                    "enabled": True,
                    "category": "google_related",
                    "transport": {"type": "stdio", "command": "google"}
                },
                "user_server": {
                    "enabled": True,
                    "category": "user_specified",
                    "transport": {"type": "stdio", "command": "user"}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()

        assert categories["google_server"]["category"] == "google_related"
        assert categories["user_server"]["category"] == "user_specified"


class TestThreadSafety:
    """Tests for thread-safe configuration access."""

    @pytest.fixture(autouse=True)
    def reset_config_state(self):
        """Reset config state for each test."""
        app_config._mcp_config_loaded = False
        app_config._native_mcp_servers_internal.clear()
        app_config._mcp_server_categories_internal.clear()
        yield
        app_config._mcp_config_loaded = False

    def test_get_mcp_server_categories_returns_proxy(self, tmp_path):
        """Test that get_mcp_server_categories returns immutable MappingProxyType."""
        config_file = tmp_path / "mcp.json"
        config_data = {"mcpServers": {}}
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            categories = app_config.get_mcp_server_categories()

        assert isinstance(categories, MappingProxyType)

    def test_mapping_proxy_is_immutable(self, tmp_path):
        """Test that MappingProxyType prevents mutation."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "test": {
                    "enabled": True,
                    "category": "test",
                    "transport": {"type": "stdio", "command": "test"}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            categories = app_config.get_mcp_server_categories()

        with pytest.raises(TypeError):
            categories["new_key"] = "value"

    def test_get_native_mcp_servers_returns_copy(self, tmp_path):
        """Test that get_native_mcp_servers returns a deep copy."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "test": {
                    "enabled": True,
                    "category": "test",
                    "transport": {"type": "stdio", "command": "test", "args": ["a", "b"]}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers1 = app_config.get_native_mcp_servers()
            servers2 = app_config.get_native_mcp_servers()

        # Should be equal but different objects
        assert servers1 == servers2
        assert servers1 is not servers2

        # Modifying one should not affect the other
        servers1["test"]["transport"]["args"].append("c")
        assert len(servers2["test"]["transport"]["args"]) == 2

    def test_double_checked_locking(self, tmp_path):
        """Test that config is loaded only once."""
        config_file = tmp_path / "mcp.json"
        config_data = {"mcpServers": {}}
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            # First call loads
            app_config._ensure_mcp_config_loaded()
            assert app_config._mcp_config_loaded is True

            # Second call should not reload
            with patch.object(app_config, '_load_mcp_config_internal') as mock_load:
                app_config._ensure_mcp_config_loaded()
                mock_load.assert_not_called()


class TestReloadConfig:
    """Tests for configuration hot-reload."""

    @pytest.fixture(autouse=True)
    def reset_config_state(self):
        """Reset config state for each test."""
        app_config._mcp_config_loaded = False
        app_config._native_mcp_servers_internal.clear()
        app_config._mcp_server_categories_internal.clear()
        yield
        app_config._mcp_config_loaded = False

    def test_reload_updates_config(self, tmp_path):
        """Test that reload_mcp_config updates the configuration."""
        config_file = tmp_path / "mcp.json"

        # Initial config
        config_data = {
            "mcpServers": {
                "initial_server": {
                    "enabled": True,
                    "category": "test",
                    "transport": {"type": "stdio", "command": "initial"}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            # Load initial config
            app_config._ensure_mcp_config_loaded()
            servers = app_config.get_native_mcp_servers()
            assert "initial_server" in servers

            # Update config file
            config_data = {
                "mcpServers": {
                    "updated_server": {
                        "enabled": True,
                        "category": "test",
                        "transport": {"type": "stdio", "command": "updated"}
                    }
                }
            }
            config_file.write_text(json.dumps(config_data))

            # Reload
            app_config.reload_mcp_config()
            servers = app_config.get_native_mcp_servers()

            assert "updated_server" in servers
            assert "initial_server" not in servers


class TestErrorHandling:
    """Tests for error handling in config loading."""

    @pytest.fixture(autouse=True)
    def reset_config_state(self):
        """Reset config state for each test."""
        app_config._mcp_config_loaded = False
        app_config._native_mcp_servers_internal.clear()
        app_config._mcp_server_categories_internal.clear()
        yield
        app_config._mcp_config_loaded = False

    def test_invalid_json_in_file(self, tmp_path):
        """Test handling of invalid JSON in config file."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text("{ invalid json }")

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            # Should not raise, returns empty
            servers, categories = app_config._load_mcp_config_internal()
            assert servers == {}
            assert categories == {}

    def test_invalid_json_in_env_var(self):
        """Test handling of invalid JSON in environment variable."""
        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": "/nonexistent/path.json",
            "NATIVE_MCP_SERVERS_CONFIG": "not valid json",
        }):
            # Should not raise, returns empty
            servers, categories = app_config._load_mcp_config_internal()
            assert servers == {}
            assert categories == {}

    def test_missing_mcp_servers_key(self, tmp_path):
        """Test handling config without mcpServers key."""
        config_file = tmp_path / "mcp.json"
        config_file.write_text(json.dumps({"other_key": "value"}))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()
            assert servers == {}
            assert categories == {}

    def test_non_dict_server_config(self, tmp_path):
        """Test handling non-dict server configurations."""
        config_file = tmp_path / "mcp.json"
        config_data = {
            "mcpServers": {
                "invalid_server": "not a dict",
                "valid_server": {
                    "enabled": True,
                    "category": "test",
                    "transport": {"type": "stdio", "command": "test"}
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {
            "NATIVE_MCP_SERVERS_CONFIG_PATH": str(config_file),
            "NATIVE_MCP_SERVERS_CONFIG": "",
        }):
            servers, categories = app_config._load_mcp_config_internal()
            # Invalid should be skipped, valid should be loaded
            assert "invalid_server" not in servers
            assert "valid_server" in servers


class TestModuleLevelExports:
    """Tests for module-level exported constants."""

    def test_mcp_server_categories_is_mapping_proxy(self):
        """Test MCP_SERVER_CATEGORIES is exported as MappingProxyType."""
        # This tests the module-level export at import time
        assert isinstance(app_config.MCP_SERVER_CATEGORIES, MappingProxyType)

    def test_native_mcp_servers_is_dict(self):
        """Test NATIVE_MCP_SERVERS is exported as dict."""
        assert isinstance(app_config.NATIVE_MCP_SERVERS, dict)
