"""
Unit tests for MCP reconnection functionality.

Tests the automatic reconnection logic in MCPProxyNode when ClosedResourceError occurs.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import anyio


class TestMCPReconnection:
    """Tests for MCP server reconnection functionality."""

    @pytest.fixture
    def mock_session_group(self):
        """Create a mock session group."""
        session_group = MagicMock()
        session_group.sessions = []
        session_group.call_tool = AsyncMock()
        session_group.connect_to_server = AsyncMock()
        session_group.disconnect_from_server = AsyncMock()
        return session_group

    @pytest.fixture
    def mock_tool_result(self):
        """Create a mock successful tool result."""
        content_item = MagicMock()
        content_item.text = "Tool execution successful"

        result = MagicMock()
        result.content = [content_item]
        return result

    @pytest.fixture
    def mcp_proxy_node(self):
        """Create an MCPProxyNode instance for testing."""
        from agent_core.nodes.mcp_proxy_node import MCPProxyNode

        tool_info = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }

        return MCPProxyNode(
            unique_tool_name="TestServer_test_tool",
            original_tool_name="test_tool",
            server_name="TestServer",
            tool_info=tool_info
        )

    @pytest.mark.asyncio
    async def test_successful_call_no_reconnect_needed(self, mcp_proxy_node, mock_session_group, mock_tool_result):
        """Test that successful calls don't trigger reconnection."""
        mock_session_group.call_tool.return_value = mock_tool_result

        prep_res = {
            "tool_params": {"arg1": "value1"},
            "shared_context": {
                "runtime_objects": {"mcp_session_group": mock_session_group}
            }
        }

        result = await mcp_proxy_node.exec_async(prep_res)

        assert result["status"] == "success"
        assert "Tool execution successful" in result["payload"]["response_preview"]
        mock_session_group.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_on_closed_resource_error(self, mcp_proxy_node, mock_session_group, mock_tool_result):
        """Test that ClosedResourceError triggers reconnection."""
        # First call fails, second succeeds after reconnection
        mock_session_group.call_tool.side_effect = [
            anyio.ClosedResourceError(),
            mock_tool_result
        ]

        with patch('agent_core.nodes.mcp_proxy_node.reconnect_mcp_server', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = True

            prep_res = {
                "tool_params": {"arg1": "value1"},
                "shared_context": {
                    "runtime_objects": {"mcp_session_group": mock_session_group}
                }
            }

            result = await mcp_proxy_node.exec_async(prep_res)

            # Reconnection should have been attempted
            mock_reconnect.assert_called_once_with(mock_session_group, "TestServer")

            # After reconnection, tool call should succeed
            assert result["status"] == "success"
            assert mock_session_group.call_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_reconnect_failure_returns_error(self, mcp_proxy_node, mock_session_group):
        """Test that failed reconnection returns appropriate error."""
        mock_session_group.call_tool.side_effect = anyio.ClosedResourceError()

        with patch('agent_core.nodes.mcp_proxy_node.reconnect_mcp_server', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = False  # Reconnection fails

            prep_res = {
                "tool_params": {"arg1": "value1"},
                "shared_context": {
                    "runtime_objects": {"mcp_session_group": mock_session_group}
                }
            }

            result = await mcp_proxy_node.exec_async(prep_res)

            assert result["status"] == "error"
            assert "CRITICAL_CONNECTION_FAILURE" in result["payload"]["type"]
            assert "Reconnection attempt" in result["error_message"]

    @pytest.mark.asyncio
    async def test_max_reconnect_attempts_respected(self, mcp_proxy_node, mock_session_group):
        """Test that reconnection stops after MAX_RECONNECT_ATTEMPTS."""
        # Always fail with ClosedResourceError
        mock_session_group.call_tool.side_effect = anyio.ClosedResourceError()

        with patch('agent_core.nodes.mcp_proxy_node.reconnect_mcp_server', new_callable=AsyncMock) as mock_reconnect:
            # Reconnection always "succeeds" but tool calls keep failing
            mock_reconnect.return_value = True

            prep_res = {
                "tool_params": {"arg1": "value1"},
                "shared_context": {
                    "runtime_objects": {"mcp_session_group": mock_session_group}
                }
            }

            result = await mcp_proxy_node.exec_async(prep_res)

            # Should have tried MAX_RECONNECT_ATTEMPTS - 1 reconnections
            # (first attempt doesn't need reconnection)
            from agent_core.nodes.mcp_proxy_node import MAX_RECONNECT_ATTEMPTS
            assert mock_reconnect.call_count == MAX_RECONNECT_ATTEMPTS - 1
            assert result["status"] == "error"


class TestReconnectMCPServer:
    """Tests for the reconnect_mcp_server function."""

    @pytest.fixture
    def mock_session_group(self):
        """Create a mock session group with a dead session."""
        session_group = MagicMock()

        # Create a dead session
        dead_session = MagicMock()
        dead_session.server_name_from_config = "TestServer"
        session_group.sessions = [dead_session]

        session_group.connect_to_server = AsyncMock()
        session_group.disconnect_from_server = AsyncMock()

        return session_group

    @pytest.mark.asyncio
    async def test_reconnect_success(self, mock_session_group):
        """Test successful reconnection to an MCP server."""
        from agent_core.services.server_manager import reconnect_mcp_server

        # Create a mock new session
        new_session = MagicMock()
        mock_session_group.connect_to_server.return_value = new_session

        with patch('agent_core.services.server_manager.get_native_mcp_servers') as mock_get_servers:
            mock_get_servers.return_value = {
                "TestServer": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "test_server"]
                }
            }

            result = await reconnect_mcp_server(mock_session_group, "TestServer")

            assert result is True
            mock_session_group.disconnect_from_server.assert_called_once()
            mock_session_group.connect_to_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_server_not_found(self, mock_session_group):
        """Test reconnection fails when server config not found."""
        from agent_core.services.server_manager import reconnect_mcp_server

        with patch('agent_core.services.server_manager.get_native_mcp_servers') as mock_get_servers:
            mock_get_servers.return_value = {}  # No servers configured

            result = await reconnect_mcp_server(mock_session_group, "NonExistentServer")

            assert result is False

    @pytest.mark.asyncio
    async def test_reconnect_unsupported_transport(self, mock_session_group):
        """Test reconnection fails for unsupported transport types."""
        from agent_core.services.server_manager import reconnect_mcp_server

        with patch('agent_core.services.server_manager.get_native_mcp_servers') as mock_get_servers:
            mock_get_servers.return_value = {
                "TestServer": {
                    "transport": "unsupported_type"
                }
            }

            result = await reconnect_mcp_server(mock_session_group, "TestServer")

            assert result is False

    @pytest.mark.asyncio
    async def test_reconnect_http_transport(self, mock_session_group):
        """Test reconnection works for HTTP transport."""
        from agent_core.services.server_manager import reconnect_mcp_server

        new_session = MagicMock()
        mock_session_group.connect_to_server.return_value = new_session

        with patch('agent_core.services.server_manager.get_native_mcp_servers') as mock_get_servers:
            mock_get_servers.return_value = {
                "TestServer": {
                    "transport": "http",
                    "url": "http://localhost:8080"
                }
            }

            result = await reconnect_mcp_server(mock_session_group, "TestServer")

            assert result is True
            mock_session_group.connect_to_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_handles_connection_error(self, mock_session_group):
        """Test reconnection handles connection errors gracefully."""
        from agent_core.services.server_manager import reconnect_mcp_server

        mock_session_group.connect_to_server.side_effect = ConnectionRefusedError("Server not available")

        with patch('agent_core.services.server_manager.get_native_mcp_servers') as mock_get_servers:
            mock_get_servers.return_value = {
                "TestServer": {
                    "transport": "http",
                    "url": "http://localhost:8080"
                }
            }

            result = await reconnect_mcp_server(mock_session_group, "TestServer")

            assert result is False


class TestMCPProxyNodeMissingSessionGroup:
    """Test MCPProxyNode behavior when session group is missing."""

    @pytest.mark.asyncio
    async def test_missing_session_group_returns_error(self):
        """Test that missing session group returns appropriate error."""
        from agent_core.nodes.mcp_proxy_node import MCPProxyNode

        tool_info = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }

        node = MCPProxyNode(
            unique_tool_name="TestServer_test_tool",
            original_tool_name="test_tool",
            server_name="TestServer",
            tool_info=tool_info
        )

        prep_res = {
            "tool_params": {"arg1": "value1"},
            "shared_context": {
                "runtime_objects": {}  # No session group
            }
        }

        result = await node.exec_async(prep_res)

        assert result["status"] == "error"
        assert "MCP Session Group not found" in result["error_message"]
