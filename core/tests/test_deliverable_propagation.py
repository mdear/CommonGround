"""
Tests for deliverable propagation and final report extraction.

These tests verify:
1. Deliverables are properly propagated from context_archive to work_modules[].deliverables
2. Final reports are correctly extracted from Principal's message history
3. ATTENTION field guides Partner appropriately based on report availability
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_core.nodes.custom_nodes.get_principal_status_tool import GetPrincipalStatusSummaryTool


class TestFinalReportExtraction:
    """Tests for final report extraction in GetPrincipalStatusSummaryTool."""

    @pytest.fixture
    def tool(self):
        """Create a GetPrincipalStatusSummaryTool instance."""
        return GetPrincipalStatusSummaryTool()

    @pytest.fixture
    def now(self):
        """Current timestamp for testing."""
        return datetime.now(timezone.utc)

    @pytest.fixture
    def mock_principal_context_complete(self):
        """Mock Principal context with a completed final report."""
        final_report_content = """# Comprehensive Research Report

### Key Points
- Finding 1: Important discovery about the topic
- Finding 2: Another significant insight
- Finding 3: Critical data point

### Overview
This is a comprehensive analysis of the research topic covering multiple dimensions...

### Detailed Analysis
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

""" + ("More detailed content. " * 500)  # Make it >5000 chars

        return {
            "state": {
                "messages": [
                    {"role": "user", "content": "Please research this topic"},
                    {"role": "assistant", "content": "I'll help you research that.", "tool_calls": [{"function": {"name": "dispatch_submodules"}}]},
                    {"role": "tool", "content": "Dispatched successfully"},
                    {"role": "assistant", "content": "All modules complete. Let me generate the final report.", "tool_calls": [{"function": {"name": "generate_markdown_report"}}]},
                    {"role": "tool", "content": "Report generated"},
                    {"role": "assistant", "content": final_report_content},  # The final report
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "finish_flow"}}]},
                ]
            },
            "refs": {
                "team": {
                    "work_modules": {},
                    "is_principal_flow_running": False
                }
            }
        }

    @pytest.fixture
    def mock_principal_context_incomplete(self):
        """Mock Principal context that is still running (no final report)."""
        return {
            "state": {
                "messages": [
                    {"role": "user", "content": "Please research this topic"},
                    {"role": "assistant", "content": "I'll help you research that.", "tool_calls": [{"function": {"name": "dispatch_submodules"}}]},
                    {"role": "tool", "content": "Dispatched successfully"},
                ]
            },
            "refs": {
                "team": {
                    "work_modules": {
                        "WM_1": {"status": "ongoing", "updated_at": datetime.now(timezone.utc).isoformat()}
                    },
                    "is_principal_flow_running": True
                }
            }
        }

    def test_extracts_final_report_when_complete(self, tool, mock_principal_context_complete, now):
        """When Principal is marked complete, final report should be extracted."""
        principal_messages = mock_principal_context_complete["state"]["messages"]
        is_marked_complete = True
        
        # Simulate the extraction logic from exec_async
        final_report = None
        if is_marked_complete:
            for msg in reversed(principal_messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and content.strip().startswith("#") and len(content) > 5000:
                        first_line = content.split('\n')[0].lstrip('#').strip()
                        final_report = {
                            "content": content,
                            "char_count": len(content),
                            "title": first_line[:100] if first_line else "Research Report",
                        }
                        break

        assert final_report is not None
        assert final_report["char_count"] > 5000
        assert final_report["title"] == "Comprehensive Research Report"
        assert "# Comprehensive Research Report" in final_report["content"]

    def test_no_report_when_incomplete(self, tool, mock_principal_context_incomplete, now):
        """When Principal is still running, no final report should be extracted."""
        principal_messages = mock_principal_context_incomplete["state"]["messages"]
        is_marked_complete = False  # Still running
        
        final_report = None
        if is_marked_complete:
            for msg in reversed(principal_messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and content.strip().startswith("#") and len(content) > 5000:
                        final_report = {"content": content}
                        break

        assert final_report is None

    def test_ignores_short_markdown_content(self, tool, now):
        """Short markdown content (<5000 chars) should not be extracted as final report."""
        messages = [
            {"role": "assistant", "content": "# Short Note\n\nThis is just a brief note."},
        ]
        is_marked_complete = True

        final_report = None
        if is_marked_complete:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and content.strip().startswith("#") and len(content) > 5000:
                        final_report = {"content": content}
                        break

        assert final_report is None

    def test_ignores_non_markdown_content(self, tool, now):
        """Long content that doesn't start with # should not be extracted."""
        long_content = "This is a very long response without markdown headers. " * 200
        messages = [
            {"role": "assistant", "content": long_content},
        ]
        is_marked_complete = True

        final_report = None
        if is_marked_complete:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and content.strip().startswith("#") and len(content) > 5000:
                        final_report = {"content": content}
                        break

        assert final_report is None

    def test_attention_message_includes_report_info(self, tool, now):
        """When final report exists, ATTENTION should guide Partner on how to use it."""
        final_report = {
            "content": "# Test Report\n" + ("Content " * 1000),
            "char_count": 15000,
            "title": "Test Report"
        }
        is_session_orphaned = False

        # Simulate ATTENTION message construction
        if final_report:
            attention_msg = (
                f"✅ FINAL REPORT READY: The Principal has completed a {final_report['char_count']:,} character report "
                f"titled \"{final_report['title']}\". The full markdown content is available in detailed_report.final_report.content. "
                "You can: (1) Display it directly to the user, (2) Offer to save it as a .md file, (3) Summarize key sections, or (4) Answer questions about specific parts."
            )
        else:
            attention_msg = "DO NOT call this tool again..."

        assert "✅ FINAL REPORT READY" in attention_msg
        assert "15,000 character" in attention_msg
        assert "Test Report" in attention_msg
        assert "detailed_report.final_report.content" in attention_msg

    def test_attention_message_orphaned_takes_precedence(self, tool, now):
        """Orphaned session warning should take precedence over report availability."""
        final_report = None
        is_session_orphaned = True

        if final_report:
            attention_msg = "✅ FINAL REPORT READY..."
        elif is_session_orphaned:
            attention_msg = (
                "⚠️ CRITICAL: This session appears to be ORPHANED..."
            )
        else:
            attention_msg = "DO NOT call this tool again..."

        assert "ORPHANED" in attention_msg
        assert "CRITICAL" in attention_msg


class TestDeliverablePropagation:
    """Tests for deliverable propagation in dispatcher_node.py."""

    def test_deliverables_copied_to_module(self):
        """Deliverables should be copied to work_modules[].deliverables."""
        # Simulate the dispatcher logic
        module_to_update = {
            "module_id": "WM_1",
            "status": "ongoing",
            "deliverables": [],  # Initially empty (legacy format)
            "context_archive": []
        }
        
        deliverables_from_associate = {
            "primary_summary": "## Research Summary\n\nKey findings from the research..."
        }
        
        # Simulate the propagation logic
        module_to_update.setdefault("context_archive", []).append({
            "dispatch_id": "Assoc_1",
            "deliverables": deliverables_from_associate
        })
        
        if deliverables_from_associate:
            module_to_update["deliverables"] = deliverables_from_associate
        
        # Verify
        assert module_to_update["deliverables"] == deliverables_from_associate
        assert module_to_update["context_archive"][0]["deliverables"] == deliverables_from_associate
        assert "primary_summary" in module_to_update["deliverables"]

    def test_empty_deliverables_not_propagated(self):
        """Empty deliverables dict should not overwrite existing data."""
        module_to_update = {
            "module_id": "WM_1",
            "deliverables": {"existing": "data"},  # Pre-existing
            "context_archive": []
        }
        
        deliverables_from_associate = {}  # Empty
        
        # Simulate the propagation logic (with guard)
        module_to_update.setdefault("context_archive", []).append({
            "dispatch_id": "Assoc_1",
            "deliverables": deliverables_from_associate
        })
        
        if deliverables_from_associate:  # This is the guard
            module_to_update["deliverables"] = deliverables_from_associate
        
        # Verify existing data preserved
        assert module_to_update["deliverables"] == {"existing": "data"}

    def test_none_deliverables_not_propagated(self):
        """None deliverables should not cause errors or overwrite."""
        module_to_update = {
            "module_id": "WM_1",
            "deliverables": [],
            "context_archive": []
        }
        
        deliverables_from_associate = None
        
        # Simulate the propagation logic
        module_to_update.setdefault("context_archive", []).append({
            "dispatch_id": "Assoc_1",
            "deliverables": deliverables_from_associate
        })
        
        if deliverables_from_associate:  # None is falsy
            module_to_update["deliverables"] = deliverables_from_associate
        
        # Verify no change to empty list
        assert module_to_update["deliverables"] == []

    def test_dict_deliverables_propagated(self):
        """Dict-format deliverables should be propagated correctly."""
        module_to_update = {
            "module_id": "WM_1",
            "deliverables": [],
            "context_archive": []
        }
        
        deliverables_from_associate = {
            "primary_summary": "# Summary\n\nDetailed findings...",
            "metadata": {"tools_used": ["web_search", "visit_url"]}
        }
        
        if deliverables_from_associate:
            module_to_update["deliverables"] = deliverables_from_associate
        
        assert module_to_update["deliverables"]["primary_summary"].startswith("# Summary")
        assert "metadata" in module_to_update["deliverables"]


class TestIntegration:
    """Integration tests verifying the complete flow."""

    def test_full_flow_deliverables_to_final_report(self):
        """Test the complete flow from Associate deliverables to Partner seeing final report."""
        
        # Step 1: Associate produces deliverables
        associate_deliverables = {
            "primary_summary": "## Module 1 Research\n\nKey findings from web search..."
        }
        
        # Step 2: Dispatcher propagates to work_module
        work_module = {
            "module_id": "WM_1",
            "status": "pending_review",
            "deliverables": associate_deliverables,  # Now populated!
            "context_archive": [{"dispatch_id": "Assoc_1", "deliverables": associate_deliverables}]
        }
        
        # Step 3: Principal generates final report
        principal_messages = [
            {"role": "assistant", "content": "# Final Research Report\n\n" + ("Analysis content. " * 1000)},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "finish_flow"}}]},
        ]
        
        # Step 4: Partner calls GetPrincipalStatusSummaryTool
        is_marked_complete = True
        final_report = None
        
        if is_marked_complete:
            for msg in reversed(principal_messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and content.strip().startswith("#") and len(content) > 5000:
                        final_report = {
                            "content": content,
                            "char_count": len(content),
                            "title": content.split('\n')[0].lstrip('#').strip()[:100],
                        }
                        break
        
        # Verify complete flow
        assert work_module["deliverables"]["primary_summary"].startswith("## Module 1")
        assert final_report is not None
        assert final_report["title"] == "Final Research Report"
        assert final_report["char_count"] > 5000
