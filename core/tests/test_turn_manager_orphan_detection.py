"""Tests for turn_manager orphan tool interaction detection."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from agent_core.framework.turn_manager import TurnManager, ORPHAN_TOOL_INTERACTION_TIMEOUT_SECONDS


class TestOrphanToolInteractionDetection:
    """Tests for detecting and handling orphaned tool interactions.
    
    The TurnManager.detect_orphaned_tool_interactions method detects tool interactions
    that have been in "running" state for longer than the timeout threshold. This helps
    identify silent failures where tools crashed before sending results back.
    """

    @pytest.fixture
    def turn_manager(self):
        """Create a TurnManager instance for testing."""
        return TurnManager()

    @pytest.fixture
    def team_state(self):
        """Create a team_state dict for testing."""
        return {"turns": []}

    def _create_turn_with_tool_interaction(
        self,
        turn_id: str,
        tool_name: str,
        tool_status: str,
        turn_status: str = "completed",
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> dict:
        """Helper to create a turn dict with a tool interaction."""
        now = datetime.now(timezone.utc)
        if start_time is None:
            # Default to 1 hour ago (well past the timeout)
            start_time = now - timedelta(hours=1)
        if end_time is None and turn_status == "completed":
            end_time = now - timedelta(minutes=30)
        
        return {
            "turn_id": turn_id,
            "status": turn_status,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat() if end_time else None,
            "tool_interactions": [
                {
                    "tool_call_id": f"tool_{turn_id}",
                    "tool_name": tool_name,
                    "status": tool_status,
                    "start_time": start_time.isoformat(),
                    "end_time": None if tool_status == "running" else (start_time + timedelta(seconds=10)).isoformat(),
                }
            ],
            "outputs": {"next_action": tool_name},
        }

    def test_detect_orphans_finds_timed_out_running_tools(
        self, turn_manager, team_state
    ):
        """Test that orphan detection finds tools stuck in 'running' state past timeout."""
        # Create a turn with a tool stuck in "running" for over an hour
        orphan_turn = self._create_turn_with_tool_interaction(
            turn_id="turn_Assoc_WebSearche_8_9954a710",
            tool_name="visit_url",
            tool_status="running",
            turn_status="completed",
        )
        team_state["turns"] = [orphan_turn]

        orphans = turn_manager.detect_orphaned_tool_interactions(team_state)

        assert len(orphans) == 1
        turn_id, tool_call_id, tool_interaction = orphans[0]
        assert turn_id == "turn_Assoc_WebSearche_8_9954a710"
        assert tool_interaction["tool_name"] == "visit_url"
        assert tool_interaction["status"] == "running"

    def test_detect_orphans_ignores_properly_completed_tools(
        self, turn_manager, team_state
    ):
        """Test that properly completed tools are not flagged as orphans."""
        completed_turn = self._create_turn_with_tool_interaction(
            turn_id="turn_Partner_12345",
            tool_name="web_search",
            tool_status="completed",
            turn_status="completed",
        )
        team_state["turns"] = [completed_turn]

        orphans = turn_manager.detect_orphaned_tool_interactions(team_state)

        assert len(orphans) == 0

    def test_detect_orphans_ignores_recently_started_tools(
        self, turn_manager, team_state
    ):
        """Test that recently started running tools are not flagged (still within timeout)."""
        # Tool started just now - should not be flagged
        recent_start = datetime.now(timezone.utc) - timedelta(seconds=30)
        running_turn = self._create_turn_with_tool_interaction(
            turn_id="turn_Associate_active",
            tool_name="visit_url",
            tool_status="running",
            turn_status="running",
            start_time=recent_start,
            end_time=None,
        )
        running_turn["end_time"] = None
        team_state["turns"] = [running_turn]

        orphans = turn_manager.detect_orphaned_tool_interactions(team_state)

        assert len(orphans) == 0

    def test_detect_orphans_finds_multiple_orphans(
        self, turn_manager, team_state
    ):
        """Test detection of multiple orphaned tool interactions."""
        orphan1 = self._create_turn_with_tool_interaction(
            turn_id="turn_Assoc_WebSearche_8_abc123",
            tool_name="visit_url",
            tool_status="running",
            turn_status="completed",
        )
        orphan2 = self._create_turn_with_tool_interaction(
            turn_id="turn_Assoc_WebSearche_9_def456",
            tool_name="web_search",
            tool_status="running",
            turn_status="completed",
        )
        completed_turn = self._create_turn_with_tool_interaction(
            turn_id="turn_Partner_normal",
            tool_name="manage_work_modules",
            tool_status="completed",
            turn_status="completed",
        )
        team_state["turns"] = [orphan1, orphan2, completed_turn]

        orphans = turn_manager.detect_orphaned_tool_interactions(team_state)

        assert len(orphans) == 2
        orphan_turn_ids = {o[0] for o in orphans}
        assert "turn_Assoc_WebSearche_8_abc123" in orphan_turn_ids
        assert "turn_Assoc_WebSearche_9_def456" in orphan_turn_ids

    def test_detect_orphans_handles_empty_turns(
        self, turn_manager, team_state
    ):
        """Test that empty turns list doesn't cause errors."""
        team_state["turns"] = []

        orphans = turn_manager.detect_orphaned_tool_interactions(team_state)

        assert len(orphans) == 0

    def test_detect_orphans_handles_turns_without_tool_interactions(
        self, turn_manager, team_state
    ):
        """Test turns without tool_interactions field are handled gracefully."""
        turn_no_tools = {
            "turn_id": "turn_user_message",
            "status": "completed",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            # No tool_interactions field
        }
        team_state["turns"] = [turn_no_tools]

        orphans = turn_manager.detect_orphaned_tool_interactions(team_state)

        assert len(orphans) == 0

    def test_detect_orphans_handles_missing_turns_key(self, turn_manager):
        """Test that missing turns key is handled gracefully."""
        team_state = {}  # No turns key

        orphans = turn_manager.detect_orphaned_tool_interactions(team_state)

        assert len(orphans) == 0

    def test_detect_orphans_respects_custom_timeout(
        self, turn_manager, team_state
    ):
        """Test that custom timeout parameter is respected."""
        # Tool started 10 minutes ago
        start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        running_turn = self._create_turn_with_tool_interaction(
            turn_id="turn_test",
            tool_name="visit_url",
            tool_status="running",
            turn_status="completed",
            start_time=start_time,
        )
        team_state["turns"] = [running_turn]

        # With 5 minute timeout (default), should be detected
        orphans = turn_manager.detect_orphaned_tool_interactions(team_state, timeout_seconds=300)
        assert len(orphans) == 1

        # With 15 minute timeout, should NOT be detected
        orphans = turn_manager.detect_orphaned_tool_interactions(team_state, timeout_seconds=900)
        assert len(orphans) == 0


class TestOrphanToolInteractionRecovery:
    """Tests for recovering from orphaned tool interactions via finalize_orphaned_tool_interactions."""

    @pytest.fixture
    def turn_manager(self):
        """Create a TurnManager instance for testing."""
        return TurnManager()

    @pytest.fixture
    def team_state(self):
        """Create a team_state dict for testing."""
        return {"turns": []}

    def test_finalize_orphaned_tools_marks_them_as_error(self, turn_manager, team_state):
        """Test that orphaned tools are marked as error for recovery."""
        now = datetime.now(timezone.utc)
        orphan_turn = {
            "turn_id": "turn_Assoc_WebSearche_8_orphan",
            "status": "completed",
            "start_time": (now - timedelta(hours=1)).isoformat(),
            "end_time": (now - timedelta(minutes=30)).isoformat(),
            "tool_interactions": [
                {
                    "tool_call_id": "tool_orphan_1",
                    "tool_name": "visit_url",
                    "status": "running",
                    "start_time": (now - timedelta(hours=1)).isoformat(),
                    "end_time": None,
                }
            ],
            "outputs": {"next_action": "visit_url"},
        }
        team_state["turns"] = [orphan_turn]

        # Finalize orphans (marks them as error)
        finalized_count = turn_manager.finalize_orphaned_tool_interactions(team_state)

        assert finalized_count == 1
        # Verify the tool interaction was updated
        updated_turn = team_state["turns"][0]
        assert updated_turn["tool_interactions"][0]["status"] == "error"
        assert "error_details" in updated_turn["tool_interactions"][0]
        assert "timed out" in updated_turn["tool_interactions"][0]["error_details"].lower()

    def test_finalize_preserves_completed_tools(
        self, turn_manager, team_state
    ):
        """Test that finalizing orphans doesn't affect properly completed tools."""
        now = datetime.now(timezone.utc)
        completed_turn = {
            "turn_id": "turn_normal",
            "status": "completed",
            "start_time": (now - timedelta(hours=1)).isoformat(),
            "end_time": (now - timedelta(minutes=30)).isoformat(),
            "tool_interactions": [
                {
                    "tool_call_id": "tool_ok",
                    "tool_name": "web_search",
                    "status": "completed",
                    "start_time": (now - timedelta(hours=1)).isoformat(),
                    "end_time": (now - timedelta(minutes=45)).isoformat(),
                }
            ],
            "outputs": {"next_action": "web_search"},
        }
        team_state["turns"] = [completed_turn]

        finalized_count = turn_manager.finalize_orphaned_tool_interactions(team_state)

        assert finalized_count == 0
        assert team_state["turns"][0]["tool_interactions"][0]["status"] == "completed"

    def test_finalize_multiple_orphans(self, turn_manager, team_state):
        """Test finalizing multiple orphaned tool interactions."""
        now = datetime.now(timezone.utc)
        orphan1 = {
            "turn_id": "turn_orphan_1",
            "status": "completed",
            "start_time": (now - timedelta(hours=2)).isoformat(),
            "end_time": (now - timedelta(hours=1)).isoformat(),
            "tool_interactions": [
                {
                    "tool_call_id": "tool_1",
                    "tool_name": "visit_url",
                    "status": "running",
                    "start_time": (now - timedelta(hours=2)).isoformat(),
                    "end_time": None,
                }
            ],
        }
        orphan2 = {
            "turn_id": "turn_orphan_2",
            "status": "completed",
            "start_time": (now - timedelta(hours=2)).isoformat(),
            "end_time": (now - timedelta(hours=1)).isoformat(),
            "tool_interactions": [
                {
                    "tool_call_id": "tool_2",
                    "tool_name": "web_search",
                    "status": "running",
                    "start_time": (now - timedelta(hours=2)).isoformat(),
                    "end_time": None,
                }
            ],
        }
        team_state["turns"] = [orphan1, orphan2]

        finalized_count = turn_manager.finalize_orphaned_tool_interactions(team_state)

        assert finalized_count == 2
        assert team_state["turns"][0]["tool_interactions"][0]["status"] == "error"
        assert team_state["turns"][1]["tool_interactions"][0]["status"] == "error"


class TestDispatchHistoryAnomalyDetection:
    """Tests for detecting dispatch history anomalies."""

    @pytest.fixture
    def mock_shared_state(self):
        """Create a mock shared state with dispatch_history."""
        state = MagicMock()
        state.team_state = {
            "dispatch_history": [],
            "work_modules": {},
        }
        return state

    def test_detect_stale_running_dispatches(self, mock_shared_state):
        """Test detection of dispatches stuck in RUNNING state."""
        from agent_core.nodes.custom_nodes.dispatcher_node import detect_dispatch_anomalies
        
        now = datetime.now(timezone.utc)
        stale_dispatch = {
            "dispatch_id": "Assoc_WebSearche_8",
            "module_id": "WM_8",
            "status": "RUNNING",
            "start_timestamp": (now - timedelta(hours=3)).isoformat(),
            "end_timestamp": None,
            "error_details": None,
        }
        mock_shared_state.team_state["dispatch_history"] = [stale_dispatch]
        mock_shared_state.team_state["work_modules"] = {
            "WM_8": {"status": "ongoing", "sub_context_id": None}
        }

        anomalies = detect_dispatch_anomalies(
            mock_shared_state, 
            stale_threshold_minutes=60
        )

        assert len(anomalies) == 1
        assert anomalies[0]["dispatch_id"] == "Assoc_WebSearche_8"
        assert anomalies[0]["anomaly_type"] == "stale_running"

    def test_detect_dispatches_without_subcontext(self, mock_shared_state):
        """Test detection of dispatches that completed but have no sub_context."""
        from agent_core.nodes.custom_nodes.dispatcher_node import detect_dispatch_anomalies
        
        now = datetime.now(timezone.utc)
        dispatch = {
            "dispatch_id": "Assoc_WebSearche_9",
            "module_id": "WM_9",
            "status": "RUNNING",
            "start_timestamp": (now - timedelta(minutes=30)).isoformat(),
            "end_timestamp": None,
            "error_details": None,
        }
        mock_shared_state.team_state["dispatch_history"] = [dispatch]
        mock_shared_state.team_state["work_modules"] = {
            "WM_9": {"status": "ongoing", "sub_context_id": None}
        }

        anomalies = detect_dispatch_anomalies(
            mock_shared_state,
            stale_threshold_minutes=15  # Lower threshold
        )

        assert len(anomalies) == 1
        assert anomalies[0]["module_id"] == "WM_9"
        assert "no sub_context" in anomalies[0].get("details", "").lower() or anomalies[0]["anomaly_type"] == "stale_running"

    def test_no_anomalies_for_completed_dispatches(self, mock_shared_state):
        """Test that properly completed dispatches don't trigger anomalies."""
        from agent_core.nodes.custom_nodes.dispatcher_node import detect_dispatch_anomalies
        
        now = datetime.now(timezone.utc)
        completed_dispatch = {
            "dispatch_id": "Assoc_Analyst_1",
            "module_id": "WM_1",
            "status": "COMPLETED",
            "start_timestamp": (now - timedelta(hours=2)).isoformat(),
            "end_timestamp": (now - timedelta(hours=1)).isoformat(),
            "final_summary": "Task completed successfully",
            "error_details": None,
        }
        mock_shared_state.team_state["dispatch_history"] = [completed_dispatch]
        mock_shared_state.team_state["work_modules"] = {
            "WM_1": {"status": "completed", "sub_context_id": "ctx_wm1"}
        }

        anomalies = detect_dispatch_anomalies(mock_shared_state)

        assert len(anomalies) == 0


class TestIntegrationOrphanDetection:
    """Integration tests for orphan detection in realistic scenarios."""

    @pytest.fixture
    def realistic_team_state(self):
        """Create a realistic team_state dict mimicking the stuck session."""
        now = datetime.now(timezone.utc)
        return {
            "turns": [
                # Normal completed Partner turn
                {
                    "turn_id": "turn_Partner_normal",
                    "status": "completed",
                    "start_time": (now - timedelta(hours=5)).isoformat(),
                    "end_time": (now - timedelta(hours=4, minutes=55)).isoformat(),
                    "tool_interactions": [
                        {
                            "tool_call_id": "tool_1",
                            "tool_name": "manage_work_modules",
                            "status": "completed",
                            "start_time": (now - timedelta(hours=5)).isoformat(),
                            "end_time": (now - timedelta(hours=4, minutes=58)).isoformat(),
                        }
                    ],
                    "outputs": {"next_action": "manage_work_modules"},
                },
                # Principal dispatch turn with incomplete tool
                {
                    "turn_id": "turn_Principal_1aeaca64",
                    "status": "completed",
                    "start_time": (now - timedelta(hours=4)).isoformat(),
                    "end_time": (now - timedelta(hours=3, minutes=59)).isoformat(),
                    "tool_interactions": [
                        {
                            "tool_call_id": "toolu_0167AXjWtbDweEr5pXYovW3o",
                            "tool_name": "dispatch_submodules",
                            "status": "running",  # ORPHANED!
                            "start_time": (now - timedelta(hours=4)).isoformat(),
                            "end_time": None,
                        }
                    ],
                    "outputs": {"next_action": "dispatch_submodules"},
                },
                # WM_8 Associate turn with orphaned tool
                {
                    "turn_id": "turn_Assoc_WebSearche_8_9954a710",
                    "status": "completed",
                    "start_time": (now - timedelta(hours=3, minutes=30)).isoformat(),
                    "end_time": (now - timedelta(hours=3, minutes=20)).isoformat(),
                    "tool_interactions": [
                        {
                            "tool_call_id": "tool_wm8",
                            "tool_name": "visit_url",
                            "status": "running",  # ORPHANED!
                            "start_time": (now - timedelta(hours=3, minutes=30)).isoformat(),
                            "end_time": None,
                        }
                    ],
                    "outputs": {"next_action": "visit_url"},
                },
                # WM_9 Associate turn with orphaned tool
                {
                    "turn_id": "turn_Assoc_WebSearche_9_dd68373e",
                    "status": "completed",
                    "start_time": (now - timedelta(hours=3, minutes=25)).isoformat(),
                    "end_time": (now - timedelta(hours=3, minutes=15)).isoformat(),
                    "tool_interactions": [
                        {
                            "tool_call_id": "tool_wm9",
                            "tool_name": "visit_url",
                            "status": "running",  # ORPHANED!
                            "start_time": (now - timedelta(hours=3, minutes=25)).isoformat(),
                            "end_time": None,
                        }
                    ],
                    "outputs": {"next_action": "visit_url"},
                },
            ],
            "dispatch_history": [
                {
                    "dispatch_id": "Assoc_WebSearche_8",
                    "module_id": "WM_8",
                    "status": "RUNNING",
                    "start_timestamp": (now - timedelta(hours=4)).isoformat(),
                    "end_timestamp": None,
                },
                {
                    "dispatch_id": "Assoc_WebSearche_9",
                    "module_id": "WM_9",
                    "status": "RUNNING",
                    "start_timestamp": (now - timedelta(hours=4)).isoformat(),
                    "end_timestamp": None,
                },
            ],
            "work_modules": {
                "WM_8": {"status": "ongoing", "sub_context_id": None},
                "WM_9": {"status": "ongoing", "sub_context_id": None},
            },
            "is_principal_flow_running": False,
        }

    def test_full_orphan_detection_on_stuck_session(self, realistic_team_state):
        """Test that all orphans are detected in a realistic stuck session."""
        turn_manager = TurnManager()
        
        orphans = turn_manager.detect_orphaned_tool_interactions(realistic_team_state)

        # Should find 3 orphans: dispatch_submodules, visit_url (WM_8), visit_url (WM_9)
        assert len(orphans) == 3
        
        # Orphans are tuples of (turn_id, tool_call_id, tool_interaction)
        orphan_tools = {o[2]["tool_name"] for o in orphans}
        assert "dispatch_submodules" in orphan_tools
        assert "visit_url" in orphan_tools

        orphan_turn_ids = {o[0] for o in orphans}
        assert "turn_Principal_1aeaca64" in orphan_turn_ids
        assert "turn_Assoc_WebSearche_8_9954a710" in orphan_turn_ids
        assert "turn_Assoc_WebSearche_9_dd68373e" in orphan_turn_ids

    def test_dispatch_anomaly_detection_on_stuck_session(self, realistic_team_state):
        """Test dispatch anomaly detection on realistic stuck session."""
        from agent_core.nodes.custom_nodes.dispatcher_node import detect_dispatch_anomalies
        
        # detect_dispatch_anomalies expects shared_state with team_state attribute
        # or a dict with 'team_state' key
        mock_shared = MagicMock()
        mock_shared.team_state = realistic_team_state
        
        anomalies = detect_dispatch_anomalies(
            mock_shared,
            stale_threshold_minutes=60
        )

        # Should find 2 stale dispatches (WM_8 and WM_9)
        assert len(anomalies) == 2
        
        anomaly_modules = {a["module_id"] for a in anomalies}
        assert "WM_8" in anomaly_modules
        assert "WM_9" in anomaly_modules

    def test_recovery_marks_all_orphans_as_error(self, realistic_team_state):
        """Test that recovery process marks all orphaned tools as error."""
        turn_manager = TurnManager()
        
        finalized_count = turn_manager.finalize_orphaned_tool_interactions(realistic_team_state)

        assert finalized_count == 3

        # Verify all orphans are now marked as error
        orphans_after = turn_manager.detect_orphaned_tool_interactions(realistic_team_state)
        assert len(orphans_after) == 0
