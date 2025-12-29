"""
Unit tests for the Connection Manager module.

Tests cover:
- Connection lifecycle (register, unregister)
- Heartbeat state management
- Grace period handling
- Event buffering
- Run state management
- Reconnection logic
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.connection_manager import (
    ConnectionManager,
    ConnectionState,
    ConnectionConfig,
    RunConnectionState,
    HeartbeatState,
    BufferedEvent
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fresh_connection_manager():
    """Create a fresh ConnectionManager instance for each test."""
    # Reset singleton for testing
    ConnectionManager._instance = None
    manager = ConnectionManager()
    yield manager
    # Clean up
    ConnectionManager._instance = None


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.fixture
def mock_event_manager():
    """Create a mock SessionEventManager."""
    em = MagicMock()
    em.session_id = "test-session-123"
    em.emit_system_event = AsyncMock()
    em.emit_raw = AsyncMock()
    return em


# =============================================================================
# HeartbeatState Tests
# =============================================================================

class TestHeartbeatState:
    """Tests for HeartbeatState dataclass."""

    def test_initial_state(self):
        """HeartbeatState starts with no pings/pongs and zero missed."""
        state = HeartbeatState()
        assert state.last_ping_sent is None
        assert state.last_pong_received is None
        assert state.missed_heartbeats == 0
        assert state.heartbeat_task is None

    def test_record_ping(self):
        """Recording a ping updates last_ping_sent."""
        state = HeartbeatState()
        before = datetime.now()
        state.record_ping()
        after = datetime.now()

        assert state.last_ping_sent is not None
        assert before <= state.last_ping_sent <= after

    def test_record_pong_resets_missed(self):
        """Recording a pong resets missed heartbeats to zero."""
        state = HeartbeatState()
        state.missed_heartbeats = 5

        state.record_pong()

        assert state.missed_heartbeats == 0
        assert state.last_pong_received is not None

    def test_record_missed_increments(self):
        """Recording missed increments the counter."""
        state = HeartbeatState()

        assert state.record_missed() == 1
        assert state.record_missed() == 2
        assert state.record_missed() == 3
        assert state.missed_heartbeats == 3


# =============================================================================
# BufferedEvent Tests
# =============================================================================

class TestBufferedEvent:
    """Tests for BufferedEvent dataclass."""

    def test_event_creation(self):
        """BufferedEvent stores event data correctly."""
        event = BufferedEvent(
            timestamp=datetime.now(),
            event_type="test_event",
            event_data={"key": "value"},
            run_id="test-run"
        )

        assert event.event_type == "test_event"
        assert event.event_data == {"key": "value"}
        assert event.run_id == "test-run"

    def test_event_not_expired(self):
        """Recent events are not expired."""
        event = BufferedEvent(
            timestamp=datetime.now(),
            event_type="test",
            event_data={}
        )

        assert not event.is_expired(ttl_seconds=60)

    def test_event_expired(self):
        """Old events are expired."""
        old_time = datetime.now() - timedelta(seconds=120)
        event = BufferedEvent(
            timestamp=old_time,
            event_type="test",
            event_data={}
        )

        assert event.is_expired(ttl_seconds=60)


# =============================================================================
# RunConnectionState Tests
# =============================================================================

class TestRunConnectionState:
    """Tests for RunConnectionState dataclass."""

    def test_initial_state(self):
        """RunConnectionState starts in CONNECTED state."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")

        assert state.state == ConnectionState.CONNECTED
        assert state.run_id == "test-run"
        assert state.session_id == "test-session"
        assert state.disconnected_at is None
        assert state.grace_period_expires is None

    def test_start_grace_period(self):
        """Starting grace period updates state correctly."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")

        before = datetime.now()
        state.start_grace_period()
        after = datetime.now()

        assert state.state == ConnectionState.GRACE_PERIOD
        assert state.disconnected_at is not None
        assert before <= state.disconnected_at <= after
        assert state.grace_period_expires is not None

        # Grace period should be in the future
        expected_expiry = state.disconnected_at + timedelta(
            seconds=ConnectionConfig.RECONNECTION_GRACE_PERIOD_SECONDS
        )
        assert abs((state.grace_period_expires - expected_expiry).total_seconds()) < 1

    def test_grace_period_not_expired(self):
        """Grace period is not expired when within window."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")
        state.start_grace_period()

        assert not state.is_grace_period_expired()

    def test_grace_period_expired(self):
        """Grace period is expired when past window."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")
        state.state = ConnectionState.GRACE_PERIOD
        state.disconnected_at = datetime.now() - timedelta(seconds=200)
        state.grace_period_expires = datetime.now() - timedelta(seconds=80)

        assert state.is_grace_period_expired()

    def test_reconnect(self):
        """Reconnecting resets state to CONNECTED."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")
        state.start_grace_period()

        state.reconnect()

        assert state.state == ConnectionState.CONNECTED
        assert state.disconnected_at is None
        assert state.grace_period_expires is None

    def test_buffer_event(self):
        """Events can be buffered."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")

        state.buffer_event("test_type", {"data": "value"})

        assert len(state.event_buffer) == 1
        event = state.event_buffer[0]
        assert event.event_type == "test_type"
        assert event.event_data == {"data": "value"}

    def test_get_buffered_events_clears(self):
        """Getting buffered events clears the buffer by default."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")
        state.buffer_event("event1", {})
        state.buffer_event("event2", {})

        events = state.get_buffered_events(clear=True)

        assert len(events) == 2
        assert len(state.event_buffer) == 0

    def test_get_buffered_events_preserves(self):
        """Getting buffered events can preserve the buffer."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")
        state.buffer_event("event1", {})

        events = state.get_buffered_events(clear=False)

        assert len(events) == 1
        assert len(state.event_buffer) == 1

    def test_save_checkpoint(self):
        """Checkpoints can be saved."""
        state = RunConnectionState(run_id="test-run", session_id="test-session")
        checkpoint_data = {"status": "running", "turn_count": 5}

        state.save_checkpoint(checkpoint_data)

        assert state.checkpoint_data == checkpoint_data
        assert state.last_checkpoint is not None


# =============================================================================
# ConnectionManager Tests
# =============================================================================

class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_singleton_pattern(self, fresh_connection_manager):
        """ConnectionManager is a singleton."""
        manager1 = ConnectionManager()
        manager2 = ConnectionManager()

        assert manager1 is manager2

    def test_register_connection(self, fresh_connection_manager, mock_websocket, mock_event_manager):
        """Registering a connection stores references."""
        manager = fresh_connection_manager

        manager.register_connection("session-1", mock_websocket, mock_event_manager)

        assert "session-1" in manager._websockets
        assert "session-1" in manager._event_managers
        assert "session-1" in manager._heartbeat_states

    def test_register_run(self, fresh_connection_manager):
        """Registering a run creates RunConnectionState."""
        manager = fresh_connection_manager

        run_state = manager.register_run("run-1", "session-1")

        assert run_state is not None
        assert run_state.run_id == "run-1"
        assert run_state.session_id == "session-1"
        assert run_state.state == ConnectionState.CONNECTED
        assert manager.get_run_state("run-1") is run_state

    @pytest.mark.asyncio
    async def test_unregister_connection_starts_grace_period(
        self, fresh_connection_manager, mock_websocket, mock_event_manager
    ):
        """Unregistering a connection starts grace period for active runs."""
        manager = fresh_connection_manager

        manager.register_connection("session-1", mock_websocket, mock_event_manager)
        manager.register_run("run-1", "session-1")

        runs_in_grace = manager.unregister_connection("session-1")

        assert "run-1" in runs_in_grace
        run_state = manager.get_run_state("run-1")
        assert run_state.state == ConnectionState.GRACE_PERIOD

        # Clean up the monitor task
        if manager._monitor_task:
            manager._monitor_task.cancel()
            try:
                await manager._monitor_task
            except asyncio.CancelledError:
                pass

    def test_is_run_active_connected(self, fresh_connection_manager):
        """Connected runs are active."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        assert manager.is_run_active("run-1")

    def test_is_run_active_grace_period(self, fresh_connection_manager):
        """Runs in grace period are active."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        run_state = manager.get_run_state("run-1")
        run_state.start_grace_period()

        assert manager.is_run_active("run-1")

    def test_is_run_active_terminated(self, fresh_connection_manager):
        """Terminated runs are not active."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")
        manager.terminate_run("run-1")

        assert not manager.is_run_active("run-1")

    def test_can_reconnect_in_grace(self, fresh_connection_manager):
        """Can reconnect during grace period."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        run_state = manager.get_run_state("run-1")
        run_state.start_grace_period()

        assert manager.can_reconnect("run-1")

    def test_cannot_reconnect_if_connected(self, fresh_connection_manager):
        """Cannot reconnect if already connected."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        assert not manager.can_reconnect("run-1")

    def test_cannot_reconnect_if_expired(self, fresh_connection_manager):
        """Cannot reconnect if grace period expired."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        run_state = manager.get_run_state("run-1")
        run_state.state = ConnectionState.GRACE_PERIOD
        run_state.grace_period_expires = datetime.now() - timedelta(seconds=10)

        assert not manager.can_reconnect("run-1")

    def test_terminate_run_returns_tasks(self, fresh_connection_manager):
        """Terminating a run returns its tasks."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        # Add a mock task
        mock_task = MagicMock()
        run_state = manager.get_run_state("run-1")
        run_state.tasks["task-1"] = mock_task

        tasks = manager.terminate_run("run-1")

        assert mock_task in tasks
        assert manager.get_run_state("run-1") is None

    def test_should_buffer_event_in_grace(self, fresh_connection_manager):
        """Events should be buffered when in grace period."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        run_state = manager.get_run_state("run-1")
        run_state.start_grace_period()

        assert manager.should_buffer_event("run-1")

    def test_should_not_buffer_when_connected(self, fresh_connection_manager):
        """Events should not be buffered when connected."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        assert not manager.should_buffer_event("run-1")

    def test_buffer_event(self, fresh_connection_manager):
        """Events can be buffered via manager."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        manager.buffer_event("run-1", "test_event", {"data": "value"})

        run_state = manager.get_run_state("run-1")
        assert len(run_state.event_buffer) == 1

    def test_save_checkpoint(self, fresh_connection_manager):
        """Checkpoints can be saved via manager."""
        manager = fresh_connection_manager
        manager.register_run("run-1", "session-1")

        checkpoint = {"status": "running"}
        manager.save_checkpoint("run-1", checkpoint)

        assert manager.get_checkpoint("run-1") == checkpoint

    def test_handle_pong(self, fresh_connection_manager, mock_websocket, mock_event_manager):
        """Handling pong resets missed heartbeats."""
        manager = fresh_connection_manager
        manager.register_connection("session-1", mock_websocket, mock_event_manager)

        heartbeat_state = manager._heartbeat_states["session-1"]
        heartbeat_state.missed_heartbeats = 2

        manager.handle_pong("session-1")

        assert heartbeat_state.missed_heartbeats == 0

    def test_get_stats(self, fresh_connection_manager, mock_websocket, mock_event_manager):
        """Stats reflect current state."""
        manager = fresh_connection_manager

        manager.register_connection("session-1", mock_websocket, mock_event_manager)
        manager.register_run("run-1", "session-1")
        manager.register_run("run-2", "session-1")

        # Put one run in grace period
        run_state = manager.get_run_state("run-2")
        run_state.start_grace_period()

        stats = manager.get_stats()

        assert stats["total_runs"] == 2
        assert stats["connected_runs"] == 1
        assert stats["grace_period_runs"] == 1
        assert stats["active_websockets"] == 1

    def test_get_runs_in_grace_period(self, fresh_connection_manager):
        """Can get list of runs in grace period."""
        manager = fresh_connection_manager

        manager.register_run("run-1", "session-1")
        manager.register_run("run-2", "session-1")

        run_state = manager.get_run_state("run-2")
        run_state.start_grace_period()

        grace_runs = manager.get_runs_in_grace_period()

        assert "run-2" in grace_runs
        assert "run-1" not in grace_runs

    def test_get_active_run_ids(self, fresh_connection_manager):
        """Can get all active run IDs."""
        manager = fresh_connection_manager

        manager.register_run("run-1", "session-1")
        manager.register_run("run-2", "session-1")
        manager.register_run("run-3", "session-1")

        # Put one in grace period
        run_state = manager.get_run_state("run-2")
        run_state.start_grace_period()

        # Terminate one
        manager.terminate_run("run-3")

        active_runs = manager.get_active_run_ids()

        assert "run-1" in active_runs
        assert "run-2" in active_runs
        assert "run-3" not in active_runs


# =============================================================================
# Reconnection Tests
# =============================================================================

class TestReconnection:
    """Tests for reconnection functionality."""

    @pytest.mark.asyncio
    async def test_reconnect_run_success(
        self, fresh_connection_manager, mock_websocket, mock_event_manager
    ):
        """Successful reconnection during grace period."""
        manager = fresh_connection_manager

        # Setup initial connection and run
        manager.register_connection("session-1", mock_websocket, mock_event_manager)
        manager.register_run("run-1", "session-1")

        # Buffer some events
        run_state = manager.get_run_state("run-1")
        run_state.start_grace_period()
        run_state.buffer_event("event1", {"data": "test"})

        # Create new connection
        new_ws = MagicMock()
        new_em = MagicMock()
        new_em.session_id = "session-2"
        new_em.emit_system_event = AsyncMock()
        new_em.emit_raw = AsyncMock()

        # Reconnect
        result = await manager.reconnect_run("run-1", "session-2", new_ws, new_em)

        assert result is not None
        assert result["success"] is True
        assert "events_replayed" in result

        # Verify run state was updated
        run_state = manager.get_run_state("run-1")
        assert run_state.state == ConnectionState.CONNECTED
        assert run_state.session_id == "session-2"

        # Events should have been replayed
        assert new_em.emit_system_event.called
        assert new_em.emit_raw.called

    @pytest.mark.asyncio
    async def test_reconnect_run_not_found(self, fresh_connection_manager, mock_websocket, mock_event_manager):
        """Reconnection fails if run doesn't exist."""
        manager = fresh_connection_manager

        result = await manager.reconnect_run("nonexistent", "session-1", mock_websocket, mock_event_manager)

        assert result is not None
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_reconnect_run_not_in_grace(
        self, fresh_connection_manager, mock_websocket, mock_event_manager
    ):
        """Reconnection fails if not in grace period."""
        manager = fresh_connection_manager

        manager.register_connection("session-1", mock_websocket, mock_event_manager)
        manager.register_run("run-1", "session-1")

        # Try to reconnect while still connected
        new_ws = MagicMock()
        new_em = MagicMock()
        new_em.session_id = "session-2"

        result = await manager.reconnect_run("run-1", "session-2", new_ws, new_em)

        assert result is not None
        assert result["success"] is False
        assert "not in reconnectable state" in result["error"]


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConnectionConfig:
    """Tests for configuration values."""

    def test_default_heartbeat_interval(self):
        """Default heartbeat interval is 30 seconds."""
        assert ConnectionConfig.HEARTBEAT_INTERVAL_SECONDS == 30.0

    def test_default_grace_period(self):
        """Default grace period is 120 seconds."""
        assert ConnectionConfig.RECONNECTION_GRACE_PERIOD_SECONDS == 120.0

    def test_default_max_missed_heartbeats(self):
        """Default max missed heartbeats is 3."""
        assert ConnectionConfig.MAX_MISSED_HEARTBEATS == 3

    def test_default_max_buffered_events(self):
        """Default max buffered events is 1000."""
        assert ConnectionConfig.MAX_BUFFERED_EVENTS == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
