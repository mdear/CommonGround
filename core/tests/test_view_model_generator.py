"""
Tests for the view_model_generator module, specifically the epoch-based depth calculation
for the FlowView.

The FlowView uses epoch-based depth calculation to ensure time flows top-to-bottom.
Disconnected subgraphs (e.g., from separate Principal dispatches) are identified as 
separate "epochs" and rendered sequentially sorted by timestamp.
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from agent_core.utils.view_model_generator import _generate_flow_view_model


def create_turn(turn_id: str, agent_id: str, start_time: datetime, 
                source_turn_ids: list = None, turn_type: str = "agent_turn",
                status: str = "completed") -> dict:
    """Helper to create a turn dict for testing."""
    return {
        "turn_id": turn_id,
        "agent_info": {
            "agent_id": agent_id,
            "profile_logical_name": f"{agent_id}_profile",
            "assigned_role_name": agent_id,
        },
        "turn_type": turn_type,
        "status": status,
        "start_time": start_time.isoformat(),
        "source_turn_ids": source_turn_ids or [],
        "tool_interactions": [],
        "llm_interaction": None,
    }


class TestEpochDetection:
    """Tests for epoch detection based on disconnected subgraphs."""

    @pytest.mark.asyncio
    async def test_single_epoch_no_separator(self):
        """Single connected graph should have no epoch separators."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        turns = [
            create_turn("t1", "Principal", base_time),
            create_turn("t2", "WM_1", base_time + timedelta(minutes=1), source_turn_ids=["t1"]),
            create_turn("t3", "Principal", base_time + timedelta(minutes=5), source_turn_ids=["t2"]),
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        # Should have 3 nodes, no epoch separators
        node_types = [n["data"]["nodeType"] for n in result["nodes"]]
        assert "epoch_separator" not in node_types
        assert len(result["nodes"]) == 3

    @pytest.mark.asyncio
    async def test_two_epochs_has_separator(self):
        """Two disconnected subgraphs should have epoch separator."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Epoch 1: t1 -> t2
        # Epoch 2: t3 -> t4 (disconnected from epoch 1)
        turns = [
            create_turn("t1", "Principal", base_time),
            create_turn("t2", "WM_1", base_time + timedelta(minutes=1), source_turn_ids=["t1"]),
            create_turn("t3", "Principal", base_time + timedelta(hours=1)),  # No source - new epoch
            create_turn("t4", "WM_2", base_time + timedelta(hours=1, minutes=1), source_turn_ids=["t3"]),
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        # Should have 4 turn nodes + 2 epoch separators (header + divider)
        node_types = [n["data"]["nodeType"] for n in result["nodes"]]
        epoch_separators = [n for n in result["nodes"] if n["data"]["nodeType"] == "epoch_separator"]
        
        assert len(epoch_separators) == 2  # Epoch 1 header + Epoch 2 divider
        
        # Check labels
        labels = {n["data"]["label"] for n in epoch_separators}
        assert "Epoch 1" in labels
        assert "Epoch 2" in labels

    @pytest.mark.asyncio
    async def test_epochs_sorted_by_timestamp(self):
        """Epochs should be sorted by earliest timestamp, not insertion order."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Insert epoch 2 first (later time), then epoch 1 (earlier time)
        turns = [
            create_turn("t3", "Principal", base_time + timedelta(hours=2)),  # Epoch 2 - later
            create_turn("t4", "WM_2", base_time + timedelta(hours=2, minutes=1), source_turn_ids=["t3"]),
            create_turn("t1", "Principal", base_time),  # Epoch 1 - earlier
            create_turn("t2", "WM_1", base_time + timedelta(minutes=1), source_turn_ids=["t1"]),
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        # Get depths - epoch 1 nodes should have lower depth than epoch 2 nodes
        node_by_id = {n["id"]: n for n in result["nodes"]}
        
        # t1 (epoch 1) should be above t3 (epoch 2)
        t1_depth = node_by_id.get("turn-t1", {}).get("data", {}).get("depth", 999)
        t3_depth = node_by_id.get("turn-t3", {}).get("data", {}).get("depth", 0)
        
        assert t1_depth < t3_depth, "Epoch 1 should render above Epoch 2"


class TestEpochSeparatorEdges:
    """Tests for edges connecting to/from epoch separators."""

    @pytest.mark.asyncio
    async def test_separator_has_incoming_and_outgoing_edges(self):
        """Epoch separator should have edges from previous epoch and to next epoch."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        turns = [
            create_turn("t1", "Principal", base_time),
            create_turn("t2", "Principal", base_time + timedelta(hours=1)),  # New epoch
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        # Find epoch separator between epochs
        epoch_2_sep = next(
            (n for n in result["nodes"] 
             if n["data"]["nodeType"] == "epoch_separator" and n["data"]["label"] == "Epoch 2"),
            None
        )
        assert epoch_2_sep is not None
        
        sep_id = epoch_2_sep["id"]
        
        # Should have edge from t1 to separator
        incoming_edges = [e for e in result["edges"] if e["target"] == sep_id]
        outgoing_edges = [e for e in result["edges"] if e["source"] == sep_id]
        
        assert len(incoming_edges) >= 1, "Separator should have incoming edge from previous epoch"
        assert len(outgoing_edges) >= 1, "Separator should have outgoing edge to next epoch"

    @pytest.mark.asyncio
    async def test_epoch_1_header_has_outgoing_edge_only(self):
        """Epoch 1 header should only have outgoing edges (no incoming)."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        turns = [
            create_turn("t1", "Principal", base_time),
            create_turn("t2", "Principal", base_time + timedelta(hours=1)),  # New epoch
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        # Find Epoch 1 header
        epoch_1_header = next(
            (n for n in result["nodes"] 
             if n["data"]["nodeType"] == "epoch_separator" and n["data"]["label"] == "Epoch 1"),
            None
        )
        assert epoch_1_header is not None
        
        header_id = epoch_1_header["id"]
        
        incoming_edges = [e for e in result["edges"] if e["target"] == header_id]
        outgoing_edges = [e for e in result["edges"] if e["source"] == header_id]
        
        assert len(incoming_edges) == 0, "Epoch 1 header should have no incoming edges"
        assert len(outgoing_edges) >= 1, "Epoch 1 header should have outgoing edge to first node"


class TestFilteredTurns:
    """Tests that Partner and user_turn are filtered from epoch detection."""

    @pytest.mark.asyncio
    async def test_partner_turns_excluded(self):
        """Partner turns should not appear in flow view or affect epochs."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        turns = [
            create_turn("partner1", "Partner", base_time),  # Should be excluded
            create_turn("t1", "Principal", base_time + timedelta(minutes=1)),
            create_turn("t2", "WM_1", base_time + timedelta(minutes=2), source_turn_ids=["t1"]),
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        # Partner turn should not be in nodes
        node_ids = [n["id"] for n in result["nodes"]]
        assert "turn-partner1" not in node_ids
        
        # Should still be single epoch (no separator)
        node_types = [n["data"]["nodeType"] for n in result["nodes"]]
        assert "epoch_separator" not in node_types

    @pytest.mark.asyncio
    async def test_user_turns_excluded(self):
        """User turns should not appear in flow view or affect epochs."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        turns = [
            create_turn("user1", "User", base_time, turn_type="user_turn"),  # Should be excluded
            create_turn("t1", "Principal", base_time + timedelta(minutes=1)),
            create_turn("t2", "WM_1", base_time + timedelta(minutes=2), source_turn_ids=["t1"]),
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        # User turn should not be in nodes
        node_ids = [n["id"] for n in result["nodes"]]
        assert "turn-user1" not in node_ids


class TestDepthCalculation:
    """Tests for hierarchical depth calculation within epochs."""

    @pytest.mark.asyncio
    async def test_depth_increases_with_hierarchy(self):
        """Child nodes should have greater depth than parent nodes."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        turns = [
            create_turn("t1", "Principal", base_time),
            create_turn("t2", "WM_1", base_time + timedelta(minutes=1), source_turn_ids=["t1"]),
            create_turn("t3", "WM_2", base_time + timedelta(minutes=1), source_turn_ids=["t1"]),
            create_turn("t4", "Principal", base_time + timedelta(minutes=5), source_turn_ids=["t2", "t3"]),
        ]
        
        run_context = {
            "team_state": {"turns": turns},
            "runtime": {"knowledge_base": None, "turn_manager": None},
        }
        
        result = await _generate_flow_view_model(run_context)
        
        node_by_id = {n["id"]: n for n in result["nodes"]}
        
        t1_depth = node_by_id["turn-t1"]["data"]["depth"]
        t2_depth = node_by_id["turn-t2"]["data"]["depth"]
        t3_depth = node_by_id["turn-t3"]["data"]["depth"]
        t4_depth = node_by_id["turn-t4"]["data"]["depth"]
        
        # t1 is parent of t2 and t3
        assert t2_depth > t1_depth
        assert t3_depth > t1_depth
        
        # t2 and t3 are siblings at same level
        assert t2_depth == t3_depth
        
        # t4 is child of t2 and t3
        assert t4_depth > t2_depth
        assert t4_depth > t3_depth
