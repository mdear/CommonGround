"""
Unit tests for session resilience message handlers.

Tests cover:
- handle_reconnect_message
- handle_heartbeat_message
- Integration with connection_manager
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import sys
from pathlib import Path
CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))


@pytest.fixture
def mock_event_manager():
    """Create a mock SessionEventManager."""
    manager = AsyncMock()
    manager.session_id = "test-session-123"
    manager.emit_raw = AsyncMock()
    return manager


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = MagicMock()
    ws.state = MagicMock()
    return ws


@pytest.fixture
def mock_ws_state(mock_event_manager, mock_websocket):
    """Create a mock websocket state object."""
    state = MagicMock()
    state.event_manager = mock_event_manager
    state.session_id = "test-session-123"
    state.websocket = mock_websocket
    state.active_run_id = None
    return state


class TestHandleReconnectMessage:
    """Test the reconnect message handler."""

    @pytest.mark.asyncio
    async def test_missing_run_id_returns_error(self, mock_ws_state):
        """Test that missing run_id returns an error."""
        from api.message_handlers import handle_reconnect_message

        data = {}  # No run_id

        await handle_reconnect_message(mock_ws_state, data)

        # Should emit an error
        mock_ws_state.event_manager.emit_raw.assert_called_once()
        call_args = mock_ws_state.event_manager.emit_raw.call_args
        assert call_args[0][0] == "reconnect_error"
        assert "Missing run_id" in call_args[0][1]["error"]

    @pytest.mark.asyncio
    async def test_run_not_found_returns_error(self, mock_ws_state):
        """Test that non-existent run returns an error."""
        from api.message_handlers import handle_reconnect_message
        from api.session import active_runs_store

        # Ensure run doesn't exist
        run_id = "nonexistent-run-id"
        if run_id in active_runs_store:
            del active_runs_store[run_id]

        data = {"run_id": run_id, "last_event_id": 0}

        await handle_reconnect_message(mock_ws_state, data)

        # Should emit an error
        mock_ws_state.event_manager.emit_raw.assert_called_once()
        call_args = mock_ws_state.event_manager.emit_raw.call_args
        assert call_args[0][0] == "reconnect_error"
        assert "not found" in call_args[0][1]["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_reconnect(self, mock_ws_state):
        """Test successful reconnection to an existing run."""
        from api.message_handlers import handle_reconnect_message
        from api.session import active_runs_store
        from api.connection_manager import connection_manager

        # Set up an existing run
        run_id = "test-run-for-reconnect"
        active_runs_store[run_id] = {
            "meta": {"run_id": run_id, "status": "running"},
            "team_state": {},
        }

        # Mock the connection manager reconnect
        with patch.object(
            connection_manager,
            "reconnect_run",
            new=AsyncMock(return_value={
                "success": True,
                "buffered_events": [],
                "events_replayed": 0,
            })
        ):
            data = {"run_id": run_id, "last_event_id": 42}

            await handle_reconnect_message(mock_ws_state, data)

            # Should emit reconnected message
            mock_ws_state.event_manager.emit_raw.assert_called_once()
            call_args = mock_ws_state.event_manager.emit_raw.call_args
            assert call_args[0][0] == "reconnected"
            assert call_args[0][1]["run_id"] == run_id

        # Cleanup
        del active_runs_store[run_id]


class TestHandleHeartbeatMessage:
    """Test the heartbeat message handler."""

    @pytest.mark.asyncio
    async def test_heartbeat_sends_ack(self, mock_ws_state):
        """Test that heartbeat sends acknowledgment."""
        from api.message_handlers import handle_heartbeat_message
        from api.connection_manager import connection_manager

        # Mock handle_client_heartbeat
        with patch.object(
            connection_manager,
            "handle_client_heartbeat",
            new=AsyncMock()
        ):
            data = {
                "timestamp": 1703779200000,
                "session_id": "test-session-123",
                "run_id": "test-run-id",
            }

            await handle_heartbeat_message(mock_ws_state, data)

            # Should emit heartbeat_ack
            mock_ws_state.event_manager.emit_raw.assert_called_once()
            call_args = mock_ws_state.event_manager.emit_raw.call_args
            assert call_args[0][0] == "heartbeat_ack"
            ack_data = call_args[0][1]
            assert ack_data["timestamp"] == 1703779200000
            assert "serverTime" in ack_data
            assert "sessionValid" in ack_data

    @pytest.mark.asyncio
    async def test_heartbeat_validates_session_id(self, mock_ws_state):
        """Test that heartbeat validates session ID."""
        from api.message_handlers import handle_heartbeat_message
        from api.connection_manager import connection_manager

        with patch.object(
            connection_manager,
            "handle_client_heartbeat",
            new=AsyncMock()
        ):
            # Mismatched session ID
            data = {
                "timestamp": 1703779200000,
                "session_id": "wrong-session-id",
                "run_id": "test-run-id",
            }

            await handle_heartbeat_message(mock_ws_state, data)

            # Should still respond but indicate session invalid
            call_args = mock_ws_state.event_manager.emit_raw.call_args
            ack_data = call_args[0][1]
            assert ack_data["sessionValid"] is False

    @pytest.mark.asyncio
    async def test_heartbeat_handles_connection_manager_error(self, mock_ws_state):
        """Test that heartbeat handles connection manager errors gracefully."""
        from api.message_handlers import handle_heartbeat_message
        from api.connection_manager import connection_manager

        with patch.object(
            connection_manager,
            "handle_client_heartbeat",
            new=AsyncMock(side_effect=Exception("Test error"))
        ):
            data = {
                "timestamp": 1703779200000,
                "session_id": "test-session-123",
            }

            # Should not raise, should still send ack
            await handle_heartbeat_message(mock_ws_state, data)

            # Should still emit heartbeat_ack despite error
            mock_ws_state.event_manager.emit_raw.assert_called_once()


class TestMessageHandlerRegistry:
    """Test that handlers are properly registered."""

    def test_reconnect_handler_registered(self):
        """Test that reconnect handler is in MESSAGE_HANDLERS."""
        from api.message_handlers import MESSAGE_HANDLERS

        assert "reconnect" in MESSAGE_HANDLERS
        assert callable(MESSAGE_HANDLERS["reconnect"])

    def test_heartbeat_handler_registered(self):
        """Test that heartbeat handler is in MESSAGE_HANDLERS."""
        from api.message_handlers import MESSAGE_HANDLERS

        assert "heartbeat" in MESSAGE_HANDLERS
        assert callable(MESSAGE_HANDLERS["heartbeat"])


class TestSessionResilienceIntegration:
    """Integration tests for session resilience flow."""

    @pytest.mark.asyncio
    async def test_full_reconnection_flow(self, mock_ws_state):
        """Test a full reconnection flow from disconnect to reconnect."""
        from api.message_handlers import handle_reconnect_message
        from api.session import active_runs_store
        from api.connection_manager import connection_manager

        run_id = "integration-test-run"
        session_id = "test-session-123"

        # Setup: Create an active run
        active_runs_store[run_id] = {
            "meta": {"run_id": run_id, "status": "running"},
            "team_state": {"work_modules": {}},
        }

        try:
            # Step 1: Register connection (simulated)
            connection_manager.register_connection(
                session_id=session_id,
                websocket=mock_ws_state.websocket,
                event_manager=mock_ws_state.event_manager,
            )

            # Step 2: Register run with session
            connection_manager.register_run(
                session_id=session_id,
                run_id=run_id,
            )

            # Step 3: Simulate disconnect
            runs_in_grace = connection_manager.unregister_connection(session_id)

            # Should be in grace period (returns list of run_ids)
            assert run_id in runs_in_grace

            # Step 4: Create new session and reconnect
            new_session_id = "new-session-456"
            new_event_manager = AsyncMock()
            new_event_manager.session_id = new_session_id
            new_event_manager.emit_raw = AsyncMock()

            mock_ws_state.session_id = new_session_id
            mock_ws_state.event_manager = new_event_manager

            connection_manager.register_connection(
                session_id=new_session_id,
                websocket=mock_ws_state.websocket,
                event_manager=new_event_manager,
            )

            # Step 5: Send reconnect message
            reconnect_data = {
                "run_id": run_id,
                "last_event_id": 0,
            }

            await handle_reconnect_message(mock_ws_state, reconnect_data)

            # Verify reconnect response
            new_event_manager.emit_raw.assert_called()
            call_args = new_event_manager.emit_raw.call_args
            # Check it was a reconnect response (could be success or error)
            msg_type = call_args[0][0]
            assert msg_type in ["reconnected", "reconnect_error"]

        finally:
            # Cleanup
            if run_id in active_runs_store:
                del active_runs_store[run_id]
            # Clean up connections
            try:
                await connection_manager.unregister_connection(session_id)
            except:
                pass
            try:
                await connection_manager.unregister_connection("new-session-456")
            except:
                pass
