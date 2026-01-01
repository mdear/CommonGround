"""
Unit tests for paginated run snapshot functionality.

Tests the get_paginated_run_snapshot function which provides
section-based and paginated access to live session state.
"""

import pytest
from agent_core.utils.serialization import (
    get_serializable_run_snapshot,
    get_paginated_run_snapshot,
    _get_summary_snapshot,
    _get_section_snapshot,
    _get_sub_contexts_section
)


class MockKnowledgeBase:
    """Mock knowledge base for testing."""
    def __init__(self, data: dict = None):
        self._data = data or {"key1": "value1", "key2": [1, 2, 3]}
    
    def to_dict(self):
        return self._data


@pytest.fixture
def sample_run_context():
    """Create a sample run context for testing."""
    return {
        "meta": {
            "run_id": "test-run-id",
            "status": "running",
            "run_type": "partner_led",
            "created_at": "2025-01-01T00:00:00"
        },
        "team_state": {
            "work_modules": {
                "WM_1": {"id": "WM_1", "title": "Module 1", "status": "completed"},
                "WM_2": {"id": "WM_2", "title": "Module 2", "status": "in_progress"}
            },
            "dispatch_history": [
                {"module_id": "WM_1", "status": "SUCCESS", "profile_logical_name": "researcher"},
                {"module_id": "WM_2", "status": "RUNNING", "profile_logical_name": "analyst"}
            ],
            "is_principal_flow_running": True
        },
        "sub_context_refs": {
            "_principal_context_ref": {
                "state": {
                    "messages": [
                        {"role": "user", "content": f"User message {i}"} 
                        for i in range(100)
                    ] + [
                        {"role": "assistant", "content": f"Assistant response {i}", "tool_calls": [{"id": "tc1"}]}
                        for i in range(50)
                    ],
                    "inbox": [{"type": "report", "content": "Report 1"}],
                    "deliverables": {"final_report": "This is the final report"}
                }
            },
            "_partner_context_ref": {
                "state": {
                    "messages": [
                        {"role": "system", "content": "System prompt"},
                        {"role": "user", "content": "Partner message"}
                    ],
                    "inbox": [],
                    "deliverables": {}
                }
            }
        },
        "runtime": {
            "knowledge_base": MockKnowledgeBase()
        }
    }


@pytest.fixture
def empty_run_context():
    """Create an empty run context."""
    return {}


class TestGetPaginatedRunSnapshot:
    """Tests for get_paginated_run_snapshot function."""
    
    def test_empty_context_returns_error(self, empty_run_context):
        """Empty context should return error."""
        result = get_paginated_run_snapshot(empty_run_context)
        assert "error" in result
    
    def test_none_context_returns_error(self):
        """None context should return error."""
        result = get_paginated_run_snapshot(None)
        assert "error" in result
    
    def test_unknown_mode_returns_error(self, sample_run_context):
        """Unknown mode should return error."""
        result = get_paginated_run_snapshot(sample_run_context, mode="invalid")
        assert "error" in result
        assert "Unknown mode" in result["error"]
    
    def test_full_mode_returns_complete_snapshot(self, sample_run_context):
        """Full mode should return complete serializable snapshot."""
        result = get_paginated_run_snapshot(sample_run_context, mode="full")
        
        assert "meta" in result
        assert "team_state" in result
        assert "sub_contexts_state" in result
        assert result["meta"]["run_id"] == "test-run-id"
    
    def test_summary_mode_returns_lightweight_response(self, sample_run_context):
        """Summary mode should return lightweight overview."""
        result = get_paginated_run_snapshot(sample_run_context, mode="summary")
        
        assert result["mode"] == "summary"
        assert "meta" in result
        assert "team_state" in result
        assert "sub_contexts_summary" in result
        assert "knowledge_base_summary" in result
        
        # Should have summaries, not full data
        assert "_principal_context_ref" in result["sub_contexts_summary"]
        principal_summary = result["sub_contexts_summary"]["_principal_context_ref"]
        assert "message_count" in principal_summary
        assert principal_summary["message_count"] == 150  # 100 + 50 messages
        assert "last_message" in principal_summary
    
    def test_section_mode_requires_section(self, sample_run_context):
        """Section mode without section name should return error."""
        result = get_paginated_run_snapshot(sample_run_context, mode="section")
        assert "error" in result
    
    def test_section_meta_returns_metadata(self, sample_run_context):
        """Section=meta should return only metadata."""
        result = get_paginated_run_snapshot(
            sample_run_context, 
            mode="section", 
            section="meta"
        )
        
        assert result["mode"] == "section"
        assert result["section"] == "meta"
        assert result["data"]["run_id"] == "test-run-id"
        assert result["data"]["status"] == "running"
    
    def test_section_team_state_returns_work_modules(self, sample_run_context):
        """Section=team_state should return work modules and dispatch history."""
        result = get_paginated_run_snapshot(
            sample_run_context, 
            mode="section", 
            section="team_state"
        )
        
        assert result["mode"] == "section"
        assert result["section"] == "team_state"
        assert "work_modules" in result["data"]
        assert "dispatch_history" in result["data"]
        assert len(result["data"]["work_modules"]) == 2
    
    def test_section_sub_contexts_lists_available(self, sample_run_context):
        """Section=sub_contexts without context_name should list available contexts."""
        result = get_paginated_run_snapshot(
            sample_run_context, 
            mode="section", 
            section="sub_contexts"
        )
        
        assert result["mode"] == "section"
        assert result["section"] == "sub_contexts"
        assert "available_contexts" in result
        assert "_principal_context_ref" in result["available_contexts"]
        assert "_partner_context_ref" in result["available_contexts"]
        assert "context_summaries" in result
    
    def test_section_sub_contexts_with_pagination(self, sample_run_context):
        """Section=sub_contexts with context_name should return paginated messages."""
        result = get_paginated_run_snapshot(
            sample_run_context,
            mode="section",
            section="sub_contexts",
            context_name="_principal_context_ref",
            message_offset=0,
            message_limit=20
        )
        
        assert result["mode"] == "section"
        assert result["context_name"] == "_principal_context_ref"
        assert "data" in result
        assert "pagination" in result
        
        # Check pagination metadata
        pagination = result["pagination"]
        assert pagination["total_messages"] == 150
        assert pagination["offset"] == 0
        assert pagination["limit"] == 20
        assert pagination["returned"] == 20
        assert pagination["has_more"] == True
        
        # Check data
        assert len(result["data"]["messages"]) == 20
    
    def test_pagination_middle_page(self, sample_run_context):
        """Test fetching middle page of messages."""
        result = get_paginated_run_snapshot(
            sample_run_context,
            mode="section",
            section="sub_contexts",
            context_name="_principal_context_ref",
            message_offset=50,
            message_limit=30
        )
        
        pagination = result["pagination"]
        assert pagination["offset"] == 50
        assert pagination["returned"] == 30
        assert pagination["has_more"] == True
    
    def test_pagination_last_page(self, sample_run_context):
        """Test fetching last page of messages."""
        result = get_paginated_run_snapshot(
            sample_run_context,
            mode="section",
            section="sub_contexts",
            context_name="_principal_context_ref",
            message_offset=140,
            message_limit=50
        )
        
        pagination = result["pagination"]
        assert pagination["offset"] == 140
        assert pagination["returned"] == 10  # Only 10 remaining
        assert pagination["has_more"] == False
    
    def test_invalid_context_name_returns_error(self, sample_run_context):
        """Invalid context name should return error with available options."""
        result = get_paginated_run_snapshot(
            sample_run_context,
            mode="section",
            section="sub_contexts",
            context_name="_nonexistent_context"
        )
        
        assert "error" in result
        assert "not found" in result["error"]
        assert "available_contexts" in result
    
    def test_section_knowledge_base(self, sample_run_context):
        """Section=knowledge_base should return KB content."""
        result = get_paginated_run_snapshot(
            sample_run_context,
            mode="section",
            section="knowledge_base"
        )
        
        assert result["mode"] == "section"
        assert result["section"] == "knowledge_base"
        assert result["data"]["key1"] == "value1"
        assert result["data"]["key2"] == [1, 2, 3]
    
    def test_unknown_section_returns_error(self, sample_run_context):
        """Unknown section should return error."""
        result = get_paginated_run_snapshot(
            sample_run_context,
            mode="section",
            section="invalid_section"
        )
        
        assert "error" in result
        assert "Unknown section" in result["error"]


class TestSummarySnapshot:
    """Tests for _get_summary_snapshot helper function."""
    
    def test_summary_includes_message_counts(self, sample_run_context):
        """Summary should include message counts for each context."""
        result = _get_summary_snapshot(sample_run_context)
        
        principal = result["sub_contexts_summary"]["_principal_context_ref"]
        assert principal["message_count"] == 150
        assert principal["inbox_count"] == 1
        assert principal["has_deliverables"] == True
        assert "final_report" in principal["deliverable_keys"]
    
    def test_summary_includes_last_message_preview(self, sample_run_context):
        """Summary should include preview of last message."""
        result = _get_summary_snapshot(sample_run_context)
        
        principal = result["sub_contexts_summary"]["_principal_context_ref"]
        last_msg = principal["last_message"]
        
        assert last_msg["role"] == "assistant"
        assert "content_preview" in last_msg
        assert last_msg["has_tool_calls"] == True
    
    def test_summary_truncates_long_content(self, sample_run_context):
        """Content preview should be truncated for long messages."""
        # Add a very long message
        sample_run_context["sub_context_refs"]["_principal_context_ref"]["state"]["messages"].append({
            "role": "assistant",
            "content": "x" * 500  # Long content
        })
        
        result = _get_summary_snapshot(sample_run_context)
        principal = result["sub_contexts_summary"]["_principal_context_ref"]
        
        # Preview should be truncated with ellipsis
        assert len(principal["last_message"]["content_preview"]) <= 203  # 200 + "..."
    
    def test_knowledge_base_summary(self, sample_run_context):
        """Knowledge base summary should include type and size info."""
        result = _get_summary_snapshot(sample_run_context)
        
        kb_summary = result["knowledge_base_summary"]
        assert kb_summary["key1"]["type"] == "string"
        assert kb_summary["key1"]["size"] == 6  # len("value1")
        assert kb_summary["key2"]["type"] == "list"
        assert kb_summary["key2"]["size"] == 3


class TestBackwardsCompatibility:
    """Tests to ensure backwards compatibility with existing code."""
    
    def test_original_function_still_works(self, sample_run_context):
        """get_serializable_run_snapshot should still work."""
        result = get_serializable_run_snapshot(sample_run_context)
        
        assert "meta" in result
        assert "team_state" in result
        assert "sub_contexts_state" in result
        assert "_principal_context_ref" in result["sub_contexts_state"]
    
    def test_full_mode_matches_original(self, sample_run_context):
        """Full mode should produce same result as original function."""
        original = get_serializable_run_snapshot(sample_run_context)
        paginated = get_paginated_run_snapshot(sample_run_context, mode="full")
        
        # Both should have same structure
        assert original.keys() == paginated.keys()
        assert original["meta"] == paginated["meta"]
        assert original["team_state"] == paginated["team_state"]


class TestReconstructedDataCompatibility:
    """
    Tests verifying that reconstructed live data is compatible with 
    the persisted JSON format used by analyze_session.py.
    """
    
    def test_reconstructed_structure_matches_persisted(self, sample_run_context):
        """Verify reconstructed data has same top-level structure as persisted JSON."""
        # Get full snapshot (what would be saved by reconstruct)
        result = get_paginated_run_snapshot(sample_run_context, mode="full")
        
        # These are the keys expected by analyze_session.py
        expected_keys = {"meta", "team_state", "sub_contexts_state", "knowledge_base", "config"}
        
        # Result should have at least these keys
        assert expected_keys.issubset(set(result.keys()))
    
    def test_work_modules_structure(self, sample_run_context):
        """Work modules should have the structure expected by analyze_session."""
        result = get_paginated_run_snapshot(sample_run_context, mode="full")
        
        work_modules = result["team_state"]["work_modules"]
        for wm_id, wm in work_modules.items():
            # These fields are used by analyze_session.py
            assert "id" in wm or wm_id  # ID should be present
            assert "status" in wm
    
    def test_sub_contexts_state_structure(self, sample_run_context):
        """Sub contexts should have messages, inbox, deliverables."""
        result = get_paginated_run_snapshot(sample_run_context, mode="full")
        
        for ctx_name, ctx_state in result["sub_contexts_state"].items():
            assert "messages" in ctx_state
            assert isinstance(ctx_state["messages"], list)
            # inbox and deliverables may be optional in some states
