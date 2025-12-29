"""
Connection Manager for WebSocket Resilience

This module provides resilient WebSocket connection management with:
- Heartbeat (ping/pong) mechanism to detect stale connections
- Reconnection grace period to survive temporary disconnects
- Event buffering during disconnection for replay on reconnect
- Run context persistence for resumability

Architecture:

    ┌─────────────┐    heartbeat    ┌──────────────────┐
    │   Client    │◄──────────────► │ ConnectionManager│
    └─────────────┘                 └────────┬─────────┘
                                             │
                   ┌─────────────────────────┼─────────────────────────┐
                   │                         │                         │
           ┌───────▼───────┐         ┌───────▼───────┐         ┌───────▼───────┐
           │ RunContext    │         │ EventBuffer   │         │ HeartbeatMgr  │
           │ (persistent)  │         │ (per run)     │         │ (per socket)  │
           └───────────────┘         └───────────────┘         └───────────────┘
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from fastapi import WebSocket
    from api.events import SessionEventManager

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

class ConnectionConfig:
    """Configuration for connection resilience."""

    # Heartbeat settings
    HEARTBEAT_INTERVAL_SECONDS: float = 30.0  # Send ping every 30 seconds
    HEARTBEAT_TIMEOUT_SECONDS: float = 10.0   # Expect pong within 10 seconds
    MAX_MISSED_HEARTBEATS: int = 3            # Disconnect after 3 missed pongs

    # Reconnection settings
    RECONNECTION_GRACE_PERIOD_SECONDS: float = 120.0  # 2 minutes to reconnect

    # Event buffering settings
    MAX_BUFFERED_EVENTS: int = 1000           # Max events to buffer per run
    EVENT_BUFFER_TTL_SECONDS: float = 300.0   # Events expire after 5 minutes

    # Checkpoint settings
    CHECKPOINT_ON_TURN_COMPLETE: bool = True  # Save state after each turn


# =============================================================================
# Connection State
# =============================================================================

class ConnectionState(Enum):
    """WebSocket connection state."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    GRACE_PERIOD = "grace_period"
    TERMINATED = "terminated"


@dataclass
class HeartbeatState:
    """Tracks heartbeat state for a connection."""
    last_ping_sent: Optional[datetime] = None
    last_pong_received: Optional[datetime] = None
    missed_heartbeats: int = 0
    heartbeat_task: Optional[asyncio.Task] = None

    def record_ping(self):
        """Record that a ping was sent."""
        self.last_ping_sent = datetime.now()

    def record_pong(self):
        """Record that a pong was received."""
        self.last_pong_received = datetime.now()
        self.missed_heartbeats = 0

    def record_missed(self) -> int:
        """Record a missed heartbeat and return total missed count."""
        self.missed_heartbeats += 1
        return self.missed_heartbeats


@dataclass
class BufferedEvent:
    """An event buffered during disconnection."""
    timestamp: datetime
    event_type: str
    event_data: Dict[str, Any]
    run_id: Optional[str] = None

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if this event has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > ttl_seconds


@dataclass
class RunConnectionState:
    """Connection state for a specific run."""
    run_id: str
    session_id: str
    state: ConnectionState = ConnectionState.CONNECTED

    # Timing
    connected_at: datetime = field(default_factory=datetime.now)
    disconnected_at: Optional[datetime] = None
    grace_period_expires: Optional[datetime] = None

    # Event buffer for replay on reconnect
    event_buffer: deque = field(default_factory=lambda: deque(maxlen=ConnectionConfig.MAX_BUFFERED_EVENTS))

    # Associated tasks (not cancelled during grace period)
    tasks: Dict[str, asyncio.Task] = field(default_factory=dict)

    # Checkpoint data
    last_checkpoint: Optional[datetime] = None
    checkpoint_data: Optional[Dict[str, Any]] = None

    def start_grace_period(self):
        """Start the reconnection grace period."""
        self.state = ConnectionState.GRACE_PERIOD
        self.disconnected_at = datetime.now()
        self.grace_period_expires = datetime.now() + timedelta(
            seconds=ConnectionConfig.RECONNECTION_GRACE_PERIOD_SECONDS
        )
        logger.info("grace_period_started", extra={
            "run_id": self.run_id,
            "session_id": self.session_id,
            "expires_at": self.grace_period_expires.isoformat(),
            "grace_seconds": ConnectionConfig.RECONNECTION_GRACE_PERIOD_SECONDS
        })

    def is_grace_period_expired(self) -> bool:
        """Check if grace period has expired."""
        if self.state != ConnectionState.GRACE_PERIOD:
            return False
        if self.grace_period_expires is None:
            return True
        return datetime.now() > self.grace_period_expires

    def reconnect(self):
        """Mark as reconnected."""
        self.state = ConnectionState.CONNECTED
        self.disconnected_at = None
        self.grace_period_expires = None
        logger.info("run_reconnected", extra={
            "run_id": self.run_id,
            "session_id": self.session_id,
            "buffered_events": len(self.event_buffer)
        })

    def buffer_event(self, event_type: str, event_data: Dict[str, Any]):
        """Buffer an event for later replay."""
        event = BufferedEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            event_data=event_data,
            run_id=self.run_id
        )
        self.event_buffer.append(event)

    def get_buffered_events(self, clear: bool = True) -> List[BufferedEvent]:
        """Get buffered events, optionally clearing the buffer."""
        # Filter out expired events
        ttl = ConnectionConfig.EVENT_BUFFER_TTL_SECONDS
        valid_events = [e for e in self.event_buffer if not e.is_expired(ttl)]

        if clear:
            self.event_buffer.clear()

        return valid_events

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Save a checkpoint."""
        self.last_checkpoint = datetime.now()
        self.checkpoint_data = checkpoint_data
        logger.debug("checkpoint_saved", extra={
            "run_id": self.run_id,
            "checkpoint_time": self.last_checkpoint.isoformat()
        })


# =============================================================================
# Connection Manager
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections with resilience features.

    This is a singleton that tracks all active connections and run states,
    providing heartbeat monitoring, reconnection support, and event buffering.
    """

    _instance: Optional['ConnectionManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Run states keyed by run_id
        self._run_states: Dict[str, RunConnectionState] = {}

        # Heartbeat states keyed by session_id
        self._heartbeat_states: Dict[str, HeartbeatState] = {}

        # WebSocket references keyed by session_id
        self._websockets: Dict[str, 'WebSocket'] = {}

        # Event managers keyed by session_id
        self._event_managers: Dict[str, 'SessionEventManager'] = {}

        # Mapping from run_id to session_id for lookups
        self._run_to_session: Dict[str, str] = {}

        # Grace period monitor task
        self._monitor_task: Optional[asyncio.Task] = None

        self._initialized = True
        logger.info("connection_manager_initialized")

    # -------------------------------------------------------------------------
    # Connection Lifecycle
    # -------------------------------------------------------------------------

    def register_connection(
        self,
        session_id: str,
        websocket: 'WebSocket',
        event_manager: 'SessionEventManager'
    ):
        """Register a new WebSocket connection."""
        self._websockets[session_id] = websocket
        self._event_managers[session_id] = event_manager
        self._heartbeat_states[session_id] = HeartbeatState()

        logger.info("connection_registered", extra={
            "session_id": session_id
        })

    def unregister_connection(self, session_id: str):
        """Unregister a WebSocket connection (but don't terminate runs yet)."""
        # Start grace period for all runs on this session
        runs_in_grace = []
        for run_id, run_state in self._run_states.items():
            if run_state.session_id == session_id:
                if run_state.state == ConnectionState.CONNECTED:
                    run_state.start_grace_period()
                    runs_in_grace.append(run_id)

        # Remove websocket reference but keep run states
        self._websockets.pop(session_id, None)

        # Stop heartbeat
        heartbeat_state = self._heartbeat_states.pop(session_id, None)
        if heartbeat_state and heartbeat_state.heartbeat_task:
            heartbeat_state.heartbeat_task.cancel()

        logger.info("connection_unregistered", extra={
            "session_id": session_id,
            "runs_in_grace_period": runs_in_grace
        })

        # Ensure monitor is running
        self._ensure_monitor_running()

        return runs_in_grace

    def register_run(self, run_id: str, session_id: str) -> RunConnectionState:
        """Register a new run on a session."""
        run_state = RunConnectionState(
            run_id=run_id,
            session_id=session_id,
            state=ConnectionState.CONNECTED
        )
        self._run_states[run_id] = run_state
        self._run_to_session[run_id] = session_id

        logger.info("run_registered", extra={
            "run_id": run_id,
            "session_id": session_id
        })

        return run_state

    def get_run_state(self, run_id: str) -> Optional[RunConnectionState]:
        """Get the connection state for a run."""
        return self._run_states.get(run_id)

    def is_run_active(self, run_id: str) -> bool:
        """Check if a run is still active (connected or in grace period)."""
        run_state = self._run_states.get(run_id)
        if not run_state:
            return False
        return run_state.state in (ConnectionState.CONNECTED, ConnectionState.GRACE_PERIOD)

    def can_reconnect(self, run_id: str) -> bool:
        """Check if a run can be reconnected to."""
        run_state = self._run_states.get(run_id)
        if not run_state:
            return False
        if run_state.state == ConnectionState.GRACE_PERIOD:
            return not run_state.is_grace_period_expired()
        return False

    async def reconnect_run(
        self,
        run_id: str,
        new_session_id: str,
        websocket: 'WebSocket',
        event_manager: 'SessionEventManager'
    ) -> Dict[str, Any]:
        """
        Reconnect to an existing run during grace period.

        Returns:
            Dict with 'success' bool and additional info:
            - success: True if reconnection succeeded
            - error: Error message if failed
            - buffered_events: List of replayed events if success
            - events_replayed: Count of replayed events
        """
        run_state = self._run_states.get(run_id)

        if not run_state:
            logger.warning("reconnect_failed_no_run", extra={"run_id": run_id})
            return {
                "success": False,
                "error": f"Run {run_id} not found in connection manager"
            }

        if not self.can_reconnect(run_id):
            logger.warning("reconnect_failed_not_in_grace", extra={
                "run_id": run_id,
                "state": run_state.state.value
            })
            return {
                "success": False,
                "error": f"Run {run_id} is not in reconnectable state (state: {run_state.state.value})"
            }

        # Update connection references
        old_session_id = run_state.session_id
        run_state.session_id = new_session_id
        run_state.reconnect()

        # Update mappings
        self._run_to_session[run_id] = new_session_id
        self._websockets[new_session_id] = websocket
        self._event_managers[new_session_id] = event_manager
        self._heartbeat_states[new_session_id] = HeartbeatState()

        # Get buffered events before replay
        buffered_events = run_state.get_buffered_events(clear=False)
        events_count = len(buffered_events)

        logger.info("run_reconnected_success", extra={
            "run_id": run_id,
            "old_session_id": old_session_id,
            "new_session_id": new_session_id,
            "buffered_events": events_count
        })

        # Replay buffered events
        await self._replay_buffered_events(run_id, event_manager)

        return {
            "success": True,
            "buffered_events": [
                {"type": e.event_type, "data": e.event_data, "timestamp": e.timestamp.isoformat()}
                for e in buffered_events
            ],
            "events_replayed": events_count,
        }

    async def _replay_buffered_events(
        self,
        run_id: str,
        event_manager: 'SessionEventManager'
    ):
        """Replay buffered events to a reconnected client."""
        run_state = self._run_states.get(run_id)
        if not run_state:
            return

        events = run_state.get_buffered_events(clear=True)

        if not events:
            return

        logger.info("replaying_buffered_events", extra={
            "run_id": run_id,
            "event_count": len(events)
        })

        # Send a "replay_start" marker
        await event_manager.emit_system_event(
            "replay_start",
            {"run_id": run_id, "event_count": len(events)}
        )

        # Replay each event
        for event in events:
            try:
                await event_manager.emit_raw(event.event_type, event.event_data)
            except Exception as e:
                logger.error("event_replay_failed", extra={
                    "run_id": run_id,
                    "event_type": event.event_type,
                    "error": str(e)
                })

        # Send a "replay_end" marker
        await event_manager.emit_system_event(
            "replay_end",
            {"run_id": run_id, "events_replayed": len(events)}
        )

    def terminate_run(self, run_id: str) -> List[asyncio.Task]:
        """
        Terminate a run and return its tasks for cancellation.

        This should be called when:
        - Grace period expires
        - User explicitly stops the run
        - Run completes normally
        """
        run_state = self._run_states.pop(run_id, None)
        self._run_to_session.pop(run_id, None)

        if not run_state:
            return []

        run_state.state = ConnectionState.TERMINATED
        tasks = list(run_state.tasks.values())

        logger.info("run_terminated", extra={
            "run_id": run_id,
            "session_id": run_state.session_id,
            "task_count": len(tasks)
        })

        return tasks

    # -------------------------------------------------------------------------
    # Heartbeat Management
    # -------------------------------------------------------------------------

    async def start_heartbeat(self, session_id: str):
        """Start heartbeat monitoring for a session."""
        heartbeat_state = self._heartbeat_states.get(session_id)
        if not heartbeat_state:
            return

        # Cancel existing heartbeat task if any
        if heartbeat_state.heartbeat_task and not heartbeat_state.heartbeat_task.done():
            heartbeat_state.heartbeat_task.cancel()

        heartbeat_state.heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(session_id)
        )

        logger.debug("heartbeat_started", extra={"session_id": session_id})

    async def _heartbeat_loop(self, session_id: str):
        """Heartbeat loop that sends pings and monitors pongs."""
        try:
            while True:
                await asyncio.sleep(ConnectionConfig.HEARTBEAT_INTERVAL_SECONDS)

                websocket = self._websockets.get(session_id)
                heartbeat_state = self._heartbeat_states.get(session_id)

                if not websocket or not heartbeat_state:
                    break

                try:
                    # Send ping
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    })
                    heartbeat_state.record_ping()

                    logger.debug("heartbeat_ping_sent", extra={"session_id": session_id})

                except Exception as e:
                    # Connection likely dead
                    missed = heartbeat_state.record_missed()
                    logger.warning("heartbeat_ping_failed", extra={
                        "session_id": session_id,
                        "missed_count": missed,
                        "error": str(e)
                    })

                    if missed >= ConnectionConfig.MAX_MISSED_HEARTBEATS:
                        logger.warning("heartbeat_max_missed", extra={
                            "session_id": session_id,
                            "missed_count": missed
                        })
                        # Trigger disconnection handling
                        self.unregister_connection(session_id)
                        break

        except asyncio.CancelledError:
            logger.debug("heartbeat_cancelled", extra={"session_id": session_id})
        except Exception as e:
            logger.error("heartbeat_loop_error", extra={
                "session_id": session_id,
                "error": str(e)
            })

    def handle_pong(self, session_id: str):
        """Handle a pong response from client."""
        heartbeat_state = self._heartbeat_states.get(session_id)
        if heartbeat_state:
            heartbeat_state.record_pong()
            logger.debug("heartbeat_pong_received", extra={"session_id": session_id})

    async def handle_client_heartbeat(
        self,
        session_id: str,
        client_timestamp: Optional[float] = None,
        run_id: Optional[str] = None,
    ):
        """
        Handle client-initiated heartbeat.

        This complements the server-initiated ping/pong by allowing the client
        to verify server responsiveness. Useful for detecting scenarios where:
        - Server process is overloaded
        - Server is alive but not processing requests

        Args:
            session_id: The session sending the heartbeat
            client_timestamp: Timestamp from client (for latency calculation)
            run_id: Optional run ID for run-specific tracking
        """
        heartbeat_state = self._heartbeat_states.get(session_id)

        if heartbeat_state:
            # Track client heartbeat timing (use last_pong_received for simplicity)
            heartbeat_state.record_pong()

        # Calculate latency if timestamp provided
        latency_ms = None
        if client_timestamp:
            try:
                now_ms = time.time() * 1000
                latency_ms = now_ms - client_timestamp
            except (TypeError, ValueError):
                pass

        logger.debug(
            "client_heartbeat_received",
            extra={
                "session_id": session_id,
                "run_id": run_id,
                "latency_ms": latency_ms,
            }
        )

        # If run_id provided, touch the run state to show activity
        if run_id:
            run_state = self._run_states.get(run_id)
            if run_state and run_state.session_id == session_id:
                # Update activity timestamp (could extend grace period if in grace)
                # For now, just log that run is still being monitored
                logger.debug(
                    "client_heartbeat_run_activity",
                    extra={"run_id": run_id, "session_id": session_id}
                )

    # -------------------------------------------------------------------------
    # Event Buffering
    # -------------------------------------------------------------------------

    def should_buffer_event(self, run_id: str) -> bool:
        """Check if events should be buffered (connection in grace period)."""
        run_state = self._run_states.get(run_id)
        if not run_state:
            return False
        return run_state.state == ConnectionState.GRACE_PERIOD

    def buffer_event(self, run_id: str, event_type: str, event_data: Dict[str, Any]):
        """Buffer an event for later replay."""
        run_state = self._run_states.get(run_id)
        if run_state:
            run_state.buffer_event(event_type, event_data)
            logger.debug("event_buffered", extra={
                "run_id": run_id,
                "event_type": event_type,
                "buffer_size": len(run_state.event_buffer)
            })

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_checkpoint(self, run_id: str, checkpoint_data: Dict[str, Any]):
        """Save a checkpoint for a run."""
        run_state = self._run_states.get(run_id)
        if run_state:
            run_state.save_checkpoint(checkpoint_data)

    def get_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the last checkpoint for a run."""
        run_state = self._run_states.get(run_id)
        if run_state:
            return run_state.checkpoint_data
        return None

    # -------------------------------------------------------------------------
    # Grace Period Monitor
    # -------------------------------------------------------------------------

    def _ensure_monitor_running(self):
        """Ensure the grace period monitor is running."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._grace_period_monitor())

    async def _grace_period_monitor(self):
        """Monitor runs in grace period and terminate expired ones."""
        try:
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds

                expired_runs = []
                for run_id, run_state in list(self._run_states.items()):
                    if run_state.state == ConnectionState.GRACE_PERIOD:
                        if run_state.is_grace_period_expired():
                            expired_runs.append(run_id)

                for run_id in expired_runs:
                    logger.warning("grace_period_expired", extra={"run_id": run_id})
                    tasks = self.terminate_run(run_id)

                    # Cancel all tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error("task_cancellation_error", extra={
                                    "run_id": run_id,
                                    "error": str(e)
                                })

                # Stop monitor if no runs in grace period
                if not any(
                    rs.state == ConnectionState.GRACE_PERIOD
                    for rs in self._run_states.values()
                ):
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("grace_period_monitor_error", extra={"error": str(e)})

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    def register_task(self, run_id: str, task_name: str, task: asyncio.Task):
        """Register a task for a run."""
        run_state = self._run_states.get(run_id)
        if run_state:
            run_state.tasks[task_name] = task

    def unregister_task(self, run_id: str, task_name: str):
        """Unregister a task from a run."""
        run_state = self._run_states.get(run_id)
        if run_state:
            run_state.tasks.pop(task_name, None)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_session_for_run(self, run_id: str) -> Optional[str]:
        """Get the session_id for a run."""
        return self._run_to_session.get(run_id)

    def get_event_manager_for_run(self, run_id: str) -> Optional['SessionEventManager']:
        """Get the event manager for a run."""
        session_id = self._run_to_session.get(run_id)
        if session_id:
            return self._event_managers.get(session_id)
        return None

    def get_websocket_for_run(self, run_id: str) -> Optional['WebSocket']:
        """Get the websocket for a run."""
        session_id = self._run_to_session.get(run_id)
        if session_id:
            return self._websockets.get(session_id)
        return None

    def get_runs_in_grace_period(self) -> List[str]:
        """Get all run IDs currently in grace period."""
        return [
            run_id for run_id, state in self._run_states.items()
            if state.state == ConnectionState.GRACE_PERIOD
        ]

    def get_active_run_ids(self) -> List[str]:
        """Get all active run IDs (connected or in grace period)."""
        return [
            run_id for run_id, state in self._run_states.items()
            if state.state in (ConnectionState.CONNECTED, ConnectionState.GRACE_PERIOD)
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        return {
            "total_runs": len(self._run_states),
            "connected_runs": sum(
                1 for rs in self._run_states.values()
                if rs.state == ConnectionState.CONNECTED
            ),
            "grace_period_runs": sum(
                1 for rs in self._run_states.values()
                if rs.state == ConnectionState.GRACE_PERIOD
            ),
            "active_websockets": len(self._websockets),
            "active_heartbeats": len(self._heartbeat_states)
        }


# Global instance
connection_manager = ConnectionManager()
