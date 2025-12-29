"""
Unit tests for agent_core.models.context and agent_core.models.turn modules.

These modules define TypedDict structures for the context system:
- RunContext: The root context for a business run
- SubContext: Context for individual sub-processes (Partner, Principal, Associate)
- TeamState: Shared state across team members
- Turn: Event log structure for turn tracking

Tests focus on:
- Type structure validation
- Required vs optional fields
- Typical usage patterns
"""

import pytest
from typing import get_type_hints, get_origin, get_args
from agent_core.models.context import (
    RunContext,
    RunContextMeta,
    RunContextConfig,
    RunContextRuntime,
    RunContextSubContexts,
    SubContext,
    SubContextMeta,
    SubContextState,
    SubContextRuntimeObjects,
    SubContextRefs,
    TeamState,
)
from agent_core.models.turn import (
    Turn,
    TurnInputs,
    AgentInfo,
    LLMInteraction,
    LLMAttempt,
    ToolInteraction,
    ProcessedInboxItemLog,
)


class TestTeamState:
    """Tests for TeamState TypedDict."""

    def test_can_create_empty_team_state(self):
        """Test TeamState can be created with no fields (total=False)."""
        state: TeamState = {}
        assert isinstance(state, dict)

    def test_typical_team_state(self):
        """Test creating a typical TeamState instance."""
        state: TeamState = {
            "question": "What is machine learning?",
            "work_modules": {"module-1": {"status": "complete"}},
            "_work_module_next_id": 2,
            "profiles_list_instance_ids": ["partner-1", "principal-1"],
            "is_principal_flow_running": True,
            "dispatch_history": [],
            "turns": [],
            "partner_directives_queue": [],
        }
        assert state["question"] == "What is machine learning?"
        assert state["is_principal_flow_running"] is True

    def test_work_modules_field(self):
        """Test work_modules contains arbitrary nested structure."""
        state: TeamState = {
            "work_modules": {
                "WM001": {
                    "id": "WM001",
                    "title": "Analysis Module",
                    "status": "in_progress",
                    "deliverables": {},
                }
            },
            "_work_module_next_id": 2,
        }
        assert "WM001" in state["work_modules"]
        assert state["work_modules"]["WM001"]["status"] == "in_progress"


class TestSubContextMeta:
    """Tests for SubContextMeta TypedDict."""

    def test_required_fields(self):
        """Test SubContextMeta has required fields."""
        meta: SubContextMeta = {
            "run_id": "run-123",
            "agent_id": "agent-abc",
            "parent_agent_id": None,
            "assigned_role_name": "Analyst",
        }
        assert meta["run_id"] == "run-123"
        assert meta["agent_id"] == "agent-abc"
        assert meta["parent_agent_id"] is None
        assert meta["assigned_role_name"] == "Analyst"

    def test_all_fields_present(self):
        """Test all SubContextMeta fields can be set."""
        hints = get_type_hints(SubContextMeta)
        expected_fields = {"run_id", "agent_id", "parent_agent_id", "assigned_role_name"}
        assert set(hints.keys()) == expected_fields


class TestSubContextState:
    """Tests for SubContextState TypedDict."""

    def test_can_create_minimal_state(self):
        """Test SubContextState can be created with minimal fields."""
        state: SubContextState = {
            "messages": [],
        }
        assert state["messages"] == []

    def test_full_state_structure(self):
        """Test creating a comprehensive SubContextState."""
        state: SubContextState = {
            "messages": [{"role": "user", "content": "Hello"}],
            "current_action": None,
            "inbox": [],
            "flags": {"handover_requested": False},
            "initial_parameters": {"task": "analyze"},
            "deliverables": {"summary": "Complete"},
            "last_activity_timestamp": "2024-01-01T00:00:00Z",
            "current_iteration_count": 5,
            "archived_messages_history": [],
            "status_summary_for_partner": {},
            "execution_milestones": [],
            "tool_inputs": {},
            "current_tool_call_id": None,
            "current_actor_id": "actor-1",
            "last_turn_id": "turn-999",
            "consecutive_empty_llm_responses": 0,
            "_current_llm_stream_id": None,
            "agent_start_utc_timestamp": "2024-01-01T00:00:00Z",
            "profiles_list_instance_ids": [],
            "principal_launch_config_history": [],
        }
        assert len(state["messages"]) == 1
        assert state["current_iteration_count"] == 5


class TestSubContext:
    """Tests for SubContext TypedDict."""

    def test_subcontext_structure(self):
        """Test SubContext combines meta, state, runtime_objects, and refs."""
        # Create minimal valid SubContext
        sub_ctx: SubContext = {
            "meta": {
                "run_id": "run-1",
                "agent_id": "agent-1",
                "parent_agent_id": None,
                "assigned_role_name": None,
            },
            "state": {"messages": []},
            "runtime_objects": {},
            "refs": {
                "run": {},  # Simplified
                "team": {},
            },
        }
        assert sub_ctx["meta"]["agent_id"] == "agent-1"
        assert isinstance(sub_ctx["state"], dict)


class TestRunContextMeta:
    """Tests for RunContextMeta TypedDict."""

    def test_status_literals(self):
        """Test RunContextMeta status field accepts valid literals."""
        valid_statuses = ["CREATED", "RUNNING", "AWAITING_INPUT", "COMPLETED", "FAILED", "CANCELLED"]

        for status in valid_statuses:
            meta: RunContextMeta = {
                "run_id": "run-test",
                "run_type": "research",
                "creation_timestamp": "2024-01-01T00:00:00Z",
                "status": status,
            }
            assert meta["status"] == status


class TestRunContext:
    """Tests for RunContext TypedDict (the root context)."""

    def test_run_context_structure(self):
        """Test RunContext has all required components."""
        hints = get_type_hints(RunContext)
        expected_fields = {"meta", "config", "team_state", "runtime", "sub_context_refs", "project_id"}
        assert set(hints.keys()) == expected_fields

    def test_minimal_run_context(self):
        """Test creating a minimal RunContext."""
        run_ctx: RunContext = {
            "meta": {
                "run_id": "run-main",
                "run_type": "chat",
                "creation_timestamp": "2024-01-01T00:00:00Z",
                "status": "RUNNING",
            },
            "config": {
                "agent_profiles_store": {},
                "shared_llm_configs_ref": {},
            },
            "team_state": {},
            "runtime": {},
            "sub_context_refs": {
                "_partner_context_ref": None,
                "_principal_context_ref": None,
                "_ongoing_associate_tasks": {},
            },
            "project_id": "project-abc",
        }
        assert run_ctx["meta"]["status"] == "RUNNING"
        assert run_ctx["project_id"] == "project-abc"


class TestAgentInfo:
    """Tests for AgentInfo TypedDict."""

    def test_agent_info_fields(self):
        """Test AgentInfo has expected fields."""
        info: AgentInfo = {
            "agent_id": "principal-agent-1",
            "profile_logical_name": "Base_Principal",
            "profile_instance_id": "instance-123",
            "assigned_role_name": "Lead Analyst",
        }
        assert info["agent_id"] == "principal-agent-1"
        assert info["assigned_role_name"] == "Lead Analyst"


class TestLLMAttempt:
    """Tests for LLMAttempt TypedDict."""

    def test_successful_attempt(self):
        """Test LLMAttempt for successful call."""
        attempt: LLMAttempt = {
            "stream_id": "stream-abc",
            "status": "success",
            "error": None,
        }
        assert attempt["status"] == "success"

    def test_failed_attempt(self):
        """Test LLMAttempt for failed call."""
        attempt: LLMAttempt = {
            "stream_id": "stream-xyz",
            "status": "failed",
            "error": "Connection timeout",
        }
        assert attempt["status"] == "failed"
        assert attempt["error"] == "Connection timeout"


class TestLLMInteraction:
    """Tests for LLMInteraction TypedDict."""

    def test_llm_interaction_with_usage(self):
        """Test LLMInteraction with usage tracking fields."""
        interaction: LLMInteraction = {
            "status": "completed",
            "attempts": [
                {"stream_id": "s1", "status": "success", "error": None}
            ],
            "final_request": {"messages": []},
            "final_response": {"content": "Response text"},
            "predicted_usage": {"input_tokens": 100, "output_tokens": 50},
            "actual_usage": {"input_tokens": 105, "output_tokens": 48},
        }
        assert interaction["predicted_usage"]["input_tokens"] == 100
        assert interaction["actual_usage"]["output_tokens"] == 48


class TestToolInteraction:
    """Tests for ToolInteraction TypedDict."""

    def test_completed_tool_interaction(self):
        """Test ToolInteraction for completed tool call."""
        interaction: ToolInteraction = {
            "tool_call_id": "call-123",
            "tool_name": "search_documents",
            "start_time": "2024-01-01T10:00:00Z",
            "end_time": "2024-01-01T10:00:05Z",
            "status": "completed",
            "input_params": {"query": "test"},
            "result_payload": {"results": ["doc1", "doc2"]},
            "error_details": None,
        }
        assert interaction["status"] == "completed"
        assert interaction["tool_name"] == "search_documents"

    def test_error_tool_interaction(self):
        """Test ToolInteraction for failed tool call."""
        interaction: ToolInteraction = {
            "tool_call_id": "call-456",
            "tool_name": "external_api",
            "start_time": "2024-01-01T11:00:00Z",
            "end_time": "2024-01-01T11:00:10Z",
            "status": "error",
            "input_params": {"endpoint": "/data"},
            "result_payload": None,
            "error_details": "API returned 500 Internal Server Error",
        }
        assert interaction["status"] == "error"
        assert "500" in interaction["error_details"]


class TestProcessedInboxItemLog:
    """Tests for ProcessedInboxItemLog TypedDict."""

    def test_full_inbox_log(self):
        """Test ProcessedInboxItemLog with all fields."""
        log: ProcessedInboxItemLog = {
            "item_id": "inbox-item-1",
            "source": "partner_directive",
            "triggering_observer_id": "observer-abc",
            "handling_strategy_source": "profile",
            "ingestor_used": "directive_ingestor",
            "injection_mode": "append",
            "injected_content": "New directive: analyze data",
            "predicted_token_count": 150,
        }
        assert log["handling_strategy_source"] == "profile"
        assert log["predicted_token_count"] == 150


class TestTurnInputs:
    """Tests for TurnInputs TypedDict."""

    def test_turn_inputs_with_processed_items(self):
        """Test TurnInputs containing processed inbox items."""
        inputs: TurnInputs = {
            "processed_inbox_items": [
                {
                    "item_id": "item-1",
                    "source": "user_input",
                    "ingestor_used": "user_message_ingestor",
                    "injection_mode": "prepend",
                    "injected_content": "User asked about AI",
                }
            ]
        }
        assert len(inputs["processed_inbox_items"]) == 1


class TestTurn:
    """Tests for Turn TypedDict (the main turn tracking structure)."""

    def test_turn_structure(self):
        """Test Turn has all expected fields."""
        hints = get_type_hints(Turn)
        expected_fields = {
            "turn_id", "run_id", "flow_id", "agent_info", "turn_type",
            "status", "start_time", "end_time", "source_turn_ids",
            "source_tool_call_id", "inputs", "outputs", "llm_interaction",
            "tool_interactions", "metadata", "error_details"
        }
        assert set(hints.keys()) == expected_fields

    def test_minimal_turn(self):
        """Test creating a minimal Turn."""
        turn: Turn = {
            "turn_id": "turn-001",
            "run_id": "run-main",
            "flow_id": "flow-principal",
            "agent_info": {
                "agent_id": "agent-1",
                "profile_logical_name": "Base_Principal",
                "profile_instance_id": "inst-1",
                "assigned_role_name": None,
            },
            "turn_type": "agent_turn",
            "status": "completed",
            "start_time": "2024-01-01T10:00:00Z",
            "end_time": "2024-01-01T10:00:30Z",
            "source_turn_ids": [],
            "source_tool_call_id": None,
            "inputs": {"processed_inbox_items": []},
            "outputs": {},
            "llm_interaction": None,
            "tool_interactions": [],
            "metadata": None,
            "error_details": None,
        }
        assert turn["turn_id"] == "turn-001"
        assert turn["turn_type"] == "agent_turn"
        assert turn["status"] == "completed"

    def test_turn_type_literals(self):
        """Test Turn turn_type accepts valid literals."""
        valid_types = ["agent_turn", "dispatch_turn", "aggregation_turn", "user_turn"]

        for turn_type in valid_types:
            turn: Turn = {
                "turn_id": "t1",
                "run_id": "r1",
                "flow_id": "f1",
                "agent_info": {
                    "agent_id": "a1",
                    "profile_logical_name": "Test",
                    "profile_instance_id": "i1",
                    "assigned_role_name": None,
                },
                "turn_type": turn_type,
                "status": "running",
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": None,
                "source_turn_ids": [],
                "source_tool_call_id": None,
                "inputs": {},
                "outputs": {},
                "llm_interaction": None,
                "tool_interactions": [],
                "metadata": None,
                "error_details": None,
            }
            assert turn["turn_type"] == turn_type

    def test_turn_with_llm_and_tools(self):
        """Test Turn with full LLM interaction and tool calls."""
        turn: Turn = {
            "turn_id": "turn-full",
            "run_id": "run-1",
            "flow_id": "flow-1",
            "agent_info": {
                "agent_id": "agent-full",
                "profile_logical_name": "TestAgent",
                "profile_instance_id": "inst-full",
                "assigned_role_name": "Analyst",
            },
            "turn_type": "agent_turn",
            "status": "completed",
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:01:00Z",
            "source_turn_ids": ["turn-prev"],
            "source_tool_call_id": "call-trigger",
            "inputs": {
                "processed_inbox_items": [
                    {
                        "item_id": "inbox-1",
                        "source": "directive",
                        "ingestor_used": "default",
                        "injection_mode": "append",
                        "injected_content": "Process this",
                    }
                ]
            },
            "outputs": {"state_keys_modified": ["deliverables"]},
            "llm_interaction": {
                "status": "completed",
                "attempts": [{"stream_id": "s1", "status": "success", "error": None}],
                "final_request": "[omitted]",
                "final_response": {"content": "Done"},
                "predicted_usage": {"input_tokens": 500, "output_tokens": 100},
                "actual_usage": {"input_tokens": 510, "output_tokens": 95},
            },
            "tool_interactions": [
                {
                    "tool_call_id": "tc-1",
                    "tool_name": "search",
                    "start_time": "2024-01-01T12:00:10Z",
                    "end_time": "2024-01-01T12:00:15Z",
                    "status": "completed",
                    "input_params": {"query": "test"},
                    "result_payload": {"results": []},
                    "error_details": None,
                }
            ],
            "metadata": {"iteration": 3},
            "error_details": None,
        }

        assert len(turn["tool_interactions"]) == 1
        assert turn["llm_interaction"]["status"] == "completed"
        assert turn["outputs"]["state_keys_modified"] == ["deliverables"]


class TestTypeAnnotations:
    """Tests verifying TypedDict type annotations are correct."""

    def test_run_context_meta_types(self):
        """Test RunContextMeta field types."""
        hints = get_type_hints(RunContextMeta)
        assert hints["run_id"] == str
        assert hints["run_type"] == str
        assert hints["creation_timestamp"] == str

    def test_team_state_has_optional_fields(self):
        """Test TeamState uses total=False (all fields optional)."""
        # We can verify this by checking __required_keys__ and __optional_keys__
        if hasattr(TeamState, '__required_keys__'):
            # Python 3.9+ TypedDict
            assert len(TeamState.__required_keys__) == 0
            assert len(TeamState.__optional_keys__) > 0
