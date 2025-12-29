"""
Pytest configuration and shared fixtures for CommonGround tests.
"""
import sys
from pathlib import Path

import pytest

# Ensure agent_core is importable
CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))


@pytest.fixture
def sample_assistant_messages():
    """Sample assistant messages for testing extraction logic."""
    return [
        {"role": "user", "content": "Analyze the codebase structure"},
        {
            "role": "assistant",
            "content": "I'll analyze the codebase structure by examining the key directories.",
            "tool_calls": [
                {
                    "id": "call_001",
                    "function": {"name": "list_directory", "arguments": '{"path": "/src"}'},
                    "type": "function"
                }
            ]
        },
        {
            "role": "tool",
            "content": "src/\n  main.py\n  utils.py\n  config.py",
            "tool_call_id": "call_001",
            "name": "list_directory"
        },
        {
            "role": "assistant",
            "content": "<thinking>Let me analyze what I found.</thinking>\n\nThe codebase has a clean structure with three main files:\n1. main.py - Entry point\n2. utils.py - Utility functions\n3. config.py - Configuration management\n\nThis follows standard Python project conventions.",
        },
        {
            "role": "assistant",
            "content": "## Final Analysis\n\nAfter examining the codebase, I found:\n- **Architecture**: Clean separation of concerns\n- **Entry Point**: main.py serves as the primary entry\n- **Utilities**: Shared functions in utils.py\n- **Configuration**: Centralized in config.py\n\nRecommendation: The structure is well-organized and maintainable.",
        }
    ]


@pytest.fixture
def sample_work_module():
    """Sample work module data for testing ingestors."""
    return {
        "module_id": "WM_1",
        "name": "Code Analysis",
        "description": "Analyze the codebase structure",
        "status": "pending_review",
        "context_archive": [
            {
                "messages": [{"role": "assistant", "content": "Analysis complete"}],
                "deliverables": {"primary_summary": "Found 3 files"},
                "model": "claude-sonnet-4-20250514"
            }
        ],
        "messages": [{"role": "user", "content": "old message"}],
        "assignee_history": [{"agent": "Associate_001"}],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T01:00:00Z"
    }


@pytest.fixture
def mock_context():
    """Minimal mock context for testing."""
    class MockSubContext:
        def __init__(self):
            self.state = {
                "messages": [],
                "flags": {},
                "initial_parameters": {}
            }
            self.meta = {"agent_id": "test_agent"}

        def get(self, key, default=None):
            return getattr(self, key, default)

    return MockSubContext()
