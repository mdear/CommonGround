"""
Unit tests for agent_core.framework.turn_manager module.

This module tests the TurnManager class which manages the lifecycle
of Agent Turns - the atomic units of agent execution tracking.

Key functionality tested:
- Turn creation and initialization
- LLM interaction tracking
- Tool interaction recording
- Turn finalization and error handling
- Flow ID management for execution tracing
- Special turns (delimiter, aggregation)
"""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from agent_core.framework.turn_manager import TurnManager


class TestTurnManagerHelpers:
    """Tests for TurnManager helper methods."""

    @pytest.fixture
    def turn_manager(self):
        """Create a TurnManager instance."""
        return TurnManager()

    @pytest.fixture
    def team_state_with_turns(self):
        """Create team_state with sample turns."""
        return {
            "turns": [
                {"turn_id": "turn_1", "status": "completed"},
                {"turn_id": "turn_2", "status": "completed"},
                {"turn_id": "turn_3", "status": "running"},
            ]
        }

    def test_get_turn_by_id_finds_turn(self, turn_manager, team_state_with_turns):
        """Test _get_turn_by_id finds existing turn."""
        turn = turn_manager._get_turn_by_id(team_state_with_turns, "turn_2")
        assert turn is not None
        assert turn["turn_id"] == "turn_2"

    def test_get_turn_by_id_returns_none_for_missing(self, turn_manager, team_state_with_turns):
        """Test _get_turn_by_id returns None for missing turn."""
        turn = turn_manager._get_turn_by_id(team_state_with_turns, "nonexistent")
        assert turn is None

    def test_get_turn_by_id_handles_empty_turns(self, turn_manager):
        """Test _get_turn_by_id handles empty turns list."""
        team_state = {"turns": []}
        turn = turn_manager._get_turn_by_id(team_state, "any_id")
        assert turn is None

    def test_get_turn_by_id_handles_missing_turns_key(self, turn_manager):
        """Test _get_turn_by_id handles missing turns key."""
        team_state = {}
        turn = turn_manager._get_turn_by_id(team_state, "any_id")
        assert turn is None

    def test_get_turn_by_id_handles_none_turn_id(self, turn_manager, team_state_with_turns):
        """Test _get_turn_by_id handles None turn_id."""
        turn = turn_manager._get_turn_by_id(team_state_with_turns, None)
        assert turn is None

    def test_get_turn_by_id_searches_reverse(self, turn_manager):
        """Test _get_turn_by_id searches from most recent (reverse order)."""
        # Same turn_id appears twice (shouldn't happen, but tests reverse search)
        team_state = {
            "turns": [
                {"turn_id": "dup", "value": "first"},
                {"turn_id": "dup", "value": "second"},
            ]
        }
        turn = turn_manager._get_turn_by_id(team_state, "dup")
        assert turn["value"] == "second"  # Returns the later one


class TestAddTurn:
    """Tests for add_turn method."""

    @pytest.fixture
    def turn_manager(self):
        return TurnManager()

    def test_add_turn_to_empty_team_state(self, turn_manager):
        """Test adding turn to team_state without existing turns."""
        team_state = {}
        turn_object = {"turn_id": "new_turn", "turn_type": "agent_turn"}

        turn_manager.add_turn(team_state, turn_object)

        assert "turns" in team_state
        assert len(team_state["turns"]) == 1
        assert team_state["turns"][0]["turn_id"] == "new_turn"

    def test_add_turn_appends_to_existing(self, turn_manager):
        """Test adding turn appends to existing turns list."""
        team_state = {"turns": [{"turn_id": "existing"}]}
        turn_object = {"turn_id": "new_turn", "turn_type": "agent_turn"}

        turn_manager.add_turn(team_state, turn_object)

        assert len(team_state["turns"]) == 2
        assert team_state["turns"][-1]["turn_id"] == "new_turn"


class TestStartNewTurn:
    """Tests for start_new_turn method."""

    @pytest.fixture
    def turn_manager(self):
        return TurnManager()

    @pytest.fixture
    def sample_context(self):
        """Create a sample context for testing."""
        return {
            "meta": {
                "agent_id": "test-agent",
                "run_id": "run-123",
            },
            "state": {
                "last_turn_id": None,
            },
            "loaded_profile": {
                "name": "TestProfile",
                "profile_id": "profile-uuid",
            },
            "refs": {
                "team": {"turns": []},
            },
        }

    def test_start_new_turn_creates_turn(self, turn_manager, sample_context):
        """Test start_new_turn creates a new turn."""
        stream_id = "stream-abc"

        turn_id = turn_manager.start_new_turn(sample_context, stream_id)

        assert turn_id is not None
        assert turn_id.startswith("turn_test-agent_")
        assert len(sample_context["refs"]["team"]["turns"]) == 1

    def test_start_new_turn_sets_current_turn_id(self, turn_manager, sample_context):
        """Test start_new_turn sets current_turn_id in state."""
        turn_id = turn_manager.start_new_turn(sample_context, "stream-1")

        assert sample_context["state"]["current_turn_id"] == turn_id

    def test_start_new_turn_populates_agent_info(self, turn_manager, sample_context):
        """Test start_new_turn populates agent_info correctly."""
        turn_manager.start_new_turn(sample_context, "stream-1")

        turn = sample_context["refs"]["team"]["turns"][0]
        assert turn["agent_info"]["agent_id"] == "test-agent"
        assert turn["agent_info"]["profile_logical_name"] == "TestProfile"
        assert turn["agent_info"]["profile_instance_id"] == "profile-uuid"

    def test_start_new_turn_initializes_llm_interaction(self, turn_manager, sample_context):
        """Test start_new_turn initializes LLM interaction with stream_id."""
        stream_id = "stream-xyz"
        turn_manager.start_new_turn(sample_context, stream_id)

        turn = sample_context["refs"]["team"]["turns"][0]
        assert turn["llm_interaction"]["status"] == "running"
        assert turn["llm_interaction"]["attempts"][0]["stream_id"] == stream_id
        assert turn["llm_interaction"]["attempts"][0]["status"] == "pending"

    def test_start_new_turn_links_to_previous(self, turn_manager, sample_context):
        """Test start_new_turn links to previous turn via source_turn_ids."""
        # Add a previous turn
        sample_context["state"]["last_turn_id"] = "prev-turn-id"
        sample_context["refs"]["team"]["turns"] = [
            {"turn_id": "prev-turn-id", "flow_id": "flow-existing"}
        ]

        turn_manager.start_new_turn(sample_context, "stream-1")

        new_turn = sample_context["refs"]["team"]["turns"][-1]
        assert new_turn["source_turn_ids"] == ["prev-turn-id"]
        assert new_turn["flow_id"] == "flow-existing"  # Inherits flow_id

    def test_start_new_turn_creates_new_flow_if_no_previous(self, turn_manager, sample_context):
        """Test start_new_turn creates new flow_id when no previous turn."""
        turn_manager.start_new_turn(sample_context, "stream-1")

        turn = sample_context["refs"]["team"]["turns"][0]
        assert turn["flow_id"].startswith("flow_root_")
        assert turn["source_turn_ids"] == []

    def test_start_new_turn_status_is_running(self, turn_manager, sample_context):
        """Test start_new_turn sets status to running."""
        turn_manager.start_new_turn(sample_context, "stream-1")

        turn = sample_context["refs"]["team"]["turns"][0]
        assert turn["status"] == "running"
        assert turn["end_time"] is None


class TestToolInteractions:
    """Tests for tool interaction tracking."""

    @pytest.fixture
    def turn_manager(self):
        return TurnManager()

    @pytest.fixture
    def context_with_turn(self):
        """Context with an active turn."""
        turn = {
            "turn_id": "turn-active",
            "tool_interactions": [],
        }
        return {
            "state": {"current_turn_id": "turn-active"},
            "refs": {"team": {"turns": [turn]}},
            "meta": {"agent_id": "test-agent"},
        }

    def test_add_tool_interaction(self, turn_manager, context_with_turn):
        """Test adding a tool interaction."""
        tool_call = {
            "id": "call-123",
            "function": {
                "name": "search_tool",
                "arguments": '{"query": "test"}',
            },
        }

        turn_manager.add_tool_interaction(context_with_turn, tool_call)

        turn = context_with_turn["refs"]["team"]["turns"][0]
        assert len(turn["tool_interactions"]) == 1
        ti = turn["tool_interactions"][0]
        assert ti["tool_call_id"] == "call-123"
        assert ti["tool_name"] == "search_tool"
        assert ti["status"] == "running"
        assert ti["input_params"] == {"query": "test"}

    def test_update_tool_interaction_result_success(self, turn_manager, context_with_turn):
        """Test updating tool interaction with success result."""
        # Add a running tool interaction first
        context_with_turn["refs"]["team"]["turns"][0]["tool_interactions"] = [
            {"tool_call_id": "call-1", "status": "running"}
        ]

        turn_manager.update_tool_interaction_result(
            context_with_turn, "call-1", {"result": "success"}, is_error=False
        )

        ti = context_with_turn["refs"]["team"]["turns"][0]["tool_interactions"][0]
        assert ti["status"] == "completed"
        assert ti["result_payload"] == {"result": "success"}
        assert ti["end_time"] is not None

    def test_update_tool_interaction_result_error(self, turn_manager, context_with_turn):
        """Test updating tool interaction with error result."""
        context_with_turn["refs"]["team"]["turns"][0]["tool_interactions"] = [
            {"tool_call_id": "call-err", "status": "running"}
        ]

        turn_manager.update_tool_interaction_result(
            context_with_turn, "call-err", "Connection failed", is_error=True
        )

        ti = context_with_turn["refs"]["team"]["turns"][0]["tool_interactions"][0]
        assert ti["status"] == "error"
        assert ti["error_details"] == "Connection failed"

    def test_record_failed_tool_interaction(self, turn_manager, context_with_turn):
        """Test recording an immediately failed tool interaction."""
        tool_call = {
            "id": "call-invalid",
            "function": {"name": "unknown_tool"},
        }

        turn_manager.record_failed_tool_interaction(
            context_with_turn, tool_call, "Tool not found"
        )

        turn = context_with_turn["refs"]["team"]["turns"][0]
        ti = turn["tool_interactions"][0]
        assert ti["status"] == "error"
        assert ti["error_details"] == "Tool not found"
        assert ti["start_time"] == ti["end_time"]  # Immediate failure


class TestLLMInteraction:
    """Tests for LLM interaction tracking."""

    @pytest.fixture
    def turn_manager(self):
        return TurnManager()

    @pytest.fixture
    def context_with_llm_turn(self):
        """Context with turn having LLM interaction."""
        turn = {
            "turn_id": "turn-llm",
            "llm_interaction": {
                "status": "running",
                "attempts": [{"stream_id": "s1", "status": "pending", "error": None}],
                "final_response": None,
                "actual_usage": None,
            },
        }
        return {
            "state": {"current_turn_id": "turn-llm"},
            "refs": {"team": {"turns": [turn]}},
        }

    def test_update_llm_interaction_end_success(self, turn_manager, context_with_llm_turn):
        """Test updating LLM interaction on successful completion."""
        llm_response = {
            "content": "Here is the answer",
            "tool_calls": None,
            "reasoning": "I analyzed the question",
            "model_id_used": "gpt-4",
            "actual_usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        turn_manager.update_llm_interaction_end(context_with_llm_turn, llm_response)

        llm_int = context_with_llm_turn["refs"]["team"]["turns"][0]["llm_interaction"]
        assert llm_int["status"] == "completed"
        assert llm_int["final_response"]["content"] == "Here is the answer"
        assert llm_int["actual_usage"]["prompt_tokens"] == 100
        assert llm_int["attempts"][0]["status"] == "success"

    def test_update_llm_interaction_end_with_error(self, turn_manager, context_with_llm_turn):
        """Test updating LLM interaction with error."""
        llm_response = {
            "content": None,
            "error": "Rate limit exceeded",
        }

        turn_manager.update_llm_interaction_end(context_with_llm_turn, llm_response)

        llm_int = context_with_llm_turn["refs"]["team"]["turns"][0]["llm_interaction"]
        assert llm_int["attempts"][0]["status"] == "failed"
        assert llm_int["attempts"][0]["error"] == "Rate limit exceeded"


class TestTurnFinalization:
    """Tests for turn finalization methods."""

    @pytest.fixture
    def turn_manager(self):
        return TurnManager()

    @pytest.fixture
    def context_with_running_turn(self):
        """Context with a running turn."""
        turn = {
            "turn_id": "turn-running",
            "status": "running",
            "end_time": None,
            "llm_interaction": {"status": "running", "attempts": []},
        }
        return {
            "state": {"current_turn_id": "turn-running"},
            "refs": {"team": {"turns": [turn]}},
        }

    def test_finalize_current_turn(self, turn_manager, context_with_running_turn):
        """Test finalize_current_turn sets status to completed."""
        turn_manager.finalize_current_turn(context_with_running_turn)

        turn = context_with_running_turn["refs"]["team"]["turns"][0]
        assert turn["status"] == "completed"
        assert turn["end_time"] is not None

    def test_finalize_current_turn_passes_baton(self, turn_manager, context_with_running_turn):
        """Test finalize_current_turn updates last_turn_id."""
        turn_manager.finalize_current_turn(context_with_running_turn)

        assert context_with_running_turn["state"]["last_turn_id"] == "turn-running"

    def test_finalize_current_turn_with_next_action(self, turn_manager, context_with_running_turn):
        """Test finalize_current_turn records next_action in outputs."""
        turn_manager.finalize_current_turn(context_with_running_turn, next_action="continue")

        turn = context_with_running_turn["refs"]["team"]["turns"][0]
        assert turn["outputs"]["next_action"] == "continue"

    def test_fail_current_turn(self, turn_manager, context_with_running_turn):
        """Test fail_current_turn marks turn as error."""
        turn_manager.fail_current_turn(context_with_running_turn, "Unexpected error")

        turn = context_with_running_turn["refs"]["team"]["turns"][0]
        assert turn["status"] == "error"
        assert turn["error_details"] == "Unexpected error"
        assert turn["end_time"] is not None

    def test_fail_current_turn_updates_llm_interaction(self, turn_manager, context_with_running_turn):
        """Test fail_current_turn also fails LLM interaction."""
        context_with_running_turn["refs"]["team"]["turns"][0]["llm_interaction"]["attempts"] = [
            {"stream_id": "s1", "status": "pending", "error": None}
        ]

        turn_manager.fail_current_turn(context_with_running_turn, "LLM failed")

        llm_int = context_with_running_turn["refs"]["team"]["turns"][0]["llm_interaction"]
        assert llm_int["status"] == "error"
        assert llm_int["attempts"][0]["status"] == "failed"

    def test_cancel_current_turn(self, turn_manager, context_with_running_turn):
        """Test cancel_current_turn marks turn as cancelled."""
        turn_manager.cancel_current_turn(context_with_running_turn)

        turn = context_with_running_turn["refs"]["team"]["turns"][0]
        assert turn["status"] == "cancelled"


class TestSpecialTurns:
    """Tests for special turn creation (delimiter, aggregation)."""

    @pytest.fixture
    def turn_manager(self):
        return TurnManager()

    def test_create_restart_delimiter_turn(self, turn_manager):
        """Test creating a restart delimiter turn."""
        team_state = {"turns": []}

        delimiter_id = turn_manager.create_restart_delimiter_turn(
            team_state, "run-1", "flow-old", "last-turn-id"
        )

        assert delimiter_id.startswith("delimiter_")
        assert len(team_state["turns"]) == 1

        turn = team_state["turns"][0]
        assert turn["turn_type"] == "restart_delimiter_turn"
        assert turn["flow_id"] == "flow-old"
        assert turn["source_turn_ids"] == ["last-turn-id"]
        assert turn["status"] == "completed"

    def test_create_aggregation_turn(self, turn_manager):
        """Test creating an aggregation turn."""
        team_state = {"turns": []}
        dispatch_turn = {
            "flow_id": "flow-main",
            "agent_info": {"agent_id": "dispatcher"},
        }

        agg_id = turn_manager.create_aggregation_turn(
            team_state,
            run_id="run-1",
            dispatch_turn=dispatch_turn,
            last_turn_ids_of_subflows=["sub-1", "sub-2"],
            dispatch_tool_call_id="dispatch-call-1",
            aggregation_summary="Both tasks completed",
        )

        assert agg_id == "agg_dispatch-call-1"

        turn = team_state["turns"][0]
        assert turn["turn_type"] == "aggregation_turn"
        assert turn["source_turn_ids"] == ["sub-1", "sub-2"]
        assert turn["outputs"]["aggregated_results_summary"] == "Both tasks completed"


class TestEnrichTurnInputs:
    """Tests for enrich_turn_inputs method."""

    @pytest.fixture
    def turn_manager(self):
        return TurnManager()

    @pytest.fixture
    def context_for_enrichment(self):
        """Context with turn ready for enrichment."""
        turn = {
            "turn_id": "turn-enrich",
            "source_tool_call_id": None,
            "inputs": {"processed_inbox_items": []},
            "llm_interaction": {"predicted_usage": None},
        }
        return {
            "state": {"current_turn_id": "turn-enrich"},
            "refs": {"team": {"turns": [turn]}},
            "meta": {"agent_id": "test-agent"},
        }

    def test_enrich_turn_inputs_populates_inbox_items(self, turn_manager, context_for_enrichment):
        """Test that inbox processing log is populated."""
        processing_result = {
            "processing_log": [
                {"item_id": "inbox-1", "source": "USER_INPUT"},
                {"item_id": "inbox-2", "source": "DIRECTIVE"},
            ]
        }
        llm_call_package = {"predicted_total_tokens": 500}
        system_prompt_details = {"construction_log": [], "final_prompt": "System prompt"}

        turn_manager.enrich_turn_inputs(
            context_for_enrichment, "turn-enrich",
            processing_result, llm_call_package, system_prompt_details
        )

        turn = context_for_enrichment["refs"]["team"]["turns"][0]
        assert len(turn["inputs"]["processed_inbox_items"]) == 2

    def test_enrich_turn_inputs_sets_source_tool_call_id(self, turn_manager, context_for_enrichment):
        """Test that source_tool_call_id is set from TOOL_RESULT inbox item."""
        processing_result = {
            "processing_log": [
                {"source": "USER_INPUT"},
                {"source": "TOOL_RESULT", "payload": {"tool_call_id": "tool-call-xyz"}},
            ]
        }

        turn_manager.enrich_turn_inputs(
            context_for_enrichment, "turn-enrich",
            processing_result, {}, {"construction_log": [], "final_prompt": ""}
        )

        turn = context_for_enrichment["refs"]["team"]["turns"][0]
        assert turn["source_tool_call_id"] == "tool-call-xyz"

    def test_enrich_turn_inputs_sets_predicted_usage(self, turn_manager, context_for_enrichment):
        """Test that predicted token usage is set."""
        llm_call_package = {"predicted_total_tokens": 1500}

        turn_manager.enrich_turn_inputs(
            context_for_enrichment, "turn-enrich",
            {"processing_log": []}, llm_call_package,
            {"construction_log": [], "final_prompt": ""}
        )

        turn = context_for_enrichment["refs"]["team"]["turns"][0]
        assert turn["llm_interaction"]["predicted_usage"]["prompt_tokens"] == 1500
