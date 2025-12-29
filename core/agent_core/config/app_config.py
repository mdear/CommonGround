import os
import json
import logging
import threading
import copy
from types import MappingProxyType
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Thread-safe MCP configuration loader
# Uses a lock to ensure atomic loading and immutable proxy for read access
_mcp_config_lock = threading.Lock()
_mcp_server_categories_internal: Dict[str, Any] = {}
_native_mcp_servers_internal: Dict[str, Any] = {}
_mcp_config_loaded = False


def _load_mcp_config_internal() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Internal function to load MCP configuration.
    Returns tuple of (enabled_servers, categories).

    IMPORTANT: Category-based toolset matching is STRICT:
    - Servers WITHOUT an explicit 'category' field default to 'uncategorized'
    - 'uncategorized' servers are NOT matched by 'all_user_specified_mcp_servers' or 'all_google_related_mcp_servers'
    - To include a server in a category toolset, you MUST explicitly set its 'category' field

    Standard categories:
      - "google_related": Google/Gemini ecosystem servers
      - "user_specified": User-added domain-specific servers
      - "uncategorized": Default for servers without explicit category (NOT matched by category toolsets)
    """
    config_str = os.getenv("NATIVE_MCP_SERVERS_CONFIG")
    config_path = os.getenv("NATIVE_MCP_SERVERS_CONFIG_PATH", "mcp.json")
    config = {}

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.debug("native_mcp_config_loaded_from_file", extra={"config_path": config_path})
        except Exception as e:
            logger.error("native_mcp_config_file_load_failed", extra={"config_path": config_path, "error": str(e)})
    elif config_str:
        try:
            config = json.loads(config_str)
            logger.debug("native_mcp_config_loaded_from_env")
        except json.JSONDecodeError as e:
            logger.error("native_mcp_config_env_parse_failed", extra={"error": str(e)})

    # Build category mapping and return only enabled server configurations
    enabled_servers = {}
    categories = {}

    if isinstance(config, dict) and "mcpServers" in config:
        for name, server_conf in config["mcpServers"].items():
            if not isinstance(server_conf, dict):
                continue

            # STRICT category handling: default to 'uncategorized' if not specified
            # 'uncategorized' servers will NOT be matched by category-based toolsets
            explicit_category = server_conf.get("category")
            category = explicit_category if explicit_category else "uncategorized"

            if not explicit_category:
                logger.warning("mcp_server_missing_category", extra={
                    "server_name": name,
                    "hint": "Server has no 'category' field. It will NOT be included in category-based toolsets like 'all_user_specified_mcp_servers'. Add 'category': 'user_specified' to include it."
                })

            categories[name] = {
                "category": category,
                "enabled": server_conf.get("enabled", False),
                "has_explicit_category": explicit_category is not None
            }

            if server_conf.get("enabled", False):
                if "transport" in server_conf:
                    enabled_servers[name] = server_conf
                else:
                    logger.warning("native_mcp_server_config_incomplete", extra={"server_name": name, "missing_field": "transport"})
            else:
                logger.debug("native_mcp_server_disabled_or_invalid", extra={"server_name": name})

    logger.info("native_mcp_servers_loaded", extra={"enabled_count": len(enabled_servers), "categories": categories})
    return enabled_servers, categories


def _ensure_mcp_config_loaded() -> None:
    """
    Thread-safe initialization of MCP configuration.
    Uses double-checked locking pattern for efficiency.
    """
    global _mcp_config_loaded, _mcp_server_categories_internal, _native_mcp_servers_internal

    if _mcp_config_loaded:
        return

    with _mcp_config_lock:
        # Double-check after acquiring lock
        if _mcp_config_loaded:
            return

        servers, categories = _load_mcp_config_internal()
        _native_mcp_servers_internal = servers
        _mcp_server_categories_internal = categories
        _mcp_config_loaded = True


def get_mcp_server_categories() -> MappingProxyType:
    """
    Returns an immutable view of MCP server categories.
    Thread-safe and prevents accidental modification.
    """
    _ensure_mcp_config_loaded()
    return MappingProxyType(_mcp_server_categories_internal)


def get_native_mcp_servers() -> Dict[str, Any]:
    """
    Returns the enabled MCP server configurations.
    Thread-safe initialization on first access.

    Returns a deep copy to prevent callers from mutating the internal state,
    since MCP server configs contain nested dicts (transport params, etc.).
    """
    _ensure_mcp_config_loaded()
    return copy.deepcopy(_native_mcp_servers_internal)


def reload_mcp_config() -> None:
    """
    Force reload of MCP configuration.
    Thread-safe - acquires lock before reloading.
    Use sparingly, typically only for hot-reload scenarios.
    """
    global _mcp_config_loaded, _mcp_server_categories_internal, _native_mcp_servers_internal

    with _mcp_config_lock:
        servers, categories = _load_mcp_config_internal()
        _native_mcp_servers_internal = servers
        _mcp_server_categories_internal = categories
        _mcp_config_loaded = True
        logger.info("mcp_config_reloaded")


# Backward compatibility: Module-level access via property-like pattern
# These trigger lazy initialization on first access
class _MCPConfigProxy:
    """Proxy class to provide backward-compatible module-level attribute access."""

    @property
    def MCP_SERVER_CATEGORIES(self) -> MappingProxyType:
        return get_mcp_server_categories()

    @property
    def NATIVE_MCP_SERVERS(self) -> Dict[str, Any]:
        return get_native_mcp_servers()


# For backward compatibility, expose as module-level variables
# Note: These are now lazy-loaded and thread-safe
def _get_mcp_server_categories():
    """Backward-compatible accessor for MCP_SERVER_CATEGORIES."""
    return get_mcp_server_categories()


def _get_native_mcp_servers():
    """Backward-compatible accessor for NATIVE_MCP_SERVERS."""
    return get_native_mcp_servers()


# Trigger initial load for backward compatibility
# This ensures the config is loaded at module import time as before
_ensure_mcp_config_loaded()

# Expose read-only views for backward compatibility
# WARNING: These are snapshots at import time. Use get_*() functions for guaranteed fresh access.
# MCP_SERVER_CATEGORIES uses MappingProxyType to prevent mutation
# NATIVE_MCP_SERVERS returns a deepcopy to prevent mutation of nested dicts
MCP_SERVER_CATEGORIES = MappingProxyType(_mcp_server_categories_internal)
NATIVE_MCP_SERVERS = copy.deepcopy(_native_mcp_servers_internal)
