#!/usr/bin/env python3
"""
CommonGround Session Analyzer

A comprehensive tool for deep analysis of CommonGround project sessions.
Provides multi-level observability into:
- Session flow and timeline
- Agent handoffs and delegation
- Token utilization and budget compliance
- Tool usage patterns
- Error detection and diagnosis

Usage:
    python analyze_session.py <session_path_or_url> [--level LEVEL] [--focus FOCUS]

Input:
    Can be either:
    - A file path: projects/MyProject/session-id.json
    - A session URL: http://localhost:3800/webview/r?id=tentacled-pearl-oriole
    - Just a session ID: tentacled-pearl-oriole

Levels:
    summary  - High-level overview (default)
    detailed - Per-agent breakdown with key metrics
    deep     - Full message-level analysis
    timeline - Chronological event trace

Focus:
    all       - Full session analysis (default)
    principal - Focus on Principal agent
    partner   - Focus on Partner agent
    WM_N      - Focus on specific work module (e.g., WM_1, WM_2)
    errors    - Focus on errors and issues
    tokens    - Focus on token utilization

Examples:
    python analyze_session.py http://localhost:3800/webview/r?id=tentacled-pearl-oriole
    python analyze_session.py tentacled-pearl-oriole
    python analyze_session.py projects/MyProject/session-id.json
    python analyze_session.py projects/MyProject/session-id.json --level detailed
    python analyze_session.py projects/MyProject/session-id.json --level deep --focus WM_1
    python analyze_session.py projects/MyProject/session-id.json --focus tokens
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs
import re
import glob

# Add core to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
CORE_DIR = SCRIPT_DIR.parent / "core"
PROJECTS_DIR = CORE_DIR / "projects"
sys.path.insert(0, str(CORE_DIR))


# =============================================================================
# INPUT RESOLUTION
# =============================================================================

def resolve_session_input(input_str: str) -> Path:
    """
    Resolve various input formats to a session JSON file path.

    Accepts:
    - Full file path: /path/to/session.json or projects/MyProject/session.json
    - URL: http://localhost:3800/webview/r?id=session-id
    - Session ID: tentacled-pearl-oriole

    Returns:
        Path to the session JSON file

    Raises:
        FileNotFoundError if session cannot be found
    """
    input_str = input_str.strip()
    session_id = None

    # Check if it's a URL
    if input_str.startswith("http://") or input_str.startswith("https://"):
        parsed = urlparse(input_str)
        query_params = parse_qs(parsed.query)

        # Try to get session ID from 'id' parameter
        if "id" in query_params:
            session_id = query_params["id"][0]
        else:
            # Try to extract from path (e.g., /r/session-id)
            path_parts = parsed.path.strip("/").split("/")
            if path_parts:
                session_id = path_parts[-1]

    # Check if it's already a file path
    elif input_str.endswith(".json"):
        path = Path(input_str)
        if path.exists():
            return path
        # Try relative to core dir
        path = CORE_DIR / input_str
        if path.exists():
            return path
        raise FileNotFoundError(f"Session file not found: {input_str}")

    # Otherwise treat as session ID
    else:
        session_id = input_str

    # Search for session ID in projects
    if session_id:
        # Search all project directories for the session
        pattern = str(PROJECTS_DIR / "**" / f"{session_id}.json")
        matches = glob.glob(pattern, recursive=True)

        if matches:
            # Return the first match (most recent if multiple)
            return Path(matches[0])

        # Also try without the .json extension in case it was included
        if session_id.endswith(".json"):
            session_id = session_id[:-5]
            pattern = str(PROJECTS_DIR / "**" / f"{session_id}.json")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return Path(matches[0])

        raise FileNotFoundError(
            f"Session '{session_id}' not found in projects directory.\n"
            f"Searched: {PROJECTS_DIR}/**/{session_id}.json"
        )

    raise ValueError(f"Could not parse session input: {input_str}")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TokenMetrics:
    """Token usage metrics for an agent context."""
    message_count: int = 0
    estimated_tokens: int = 0
    context_limit: int = 200000  # Default
    utilization_percent: float = 0.0
    status: str = "UNKNOWN"

    def calculate(self):
        """Calculate utilization and status."""
        if self.context_limit > 0:
            self.utilization_percent = (self.estimated_tokens / self.context_limit) * 100

        if self.utilization_percent < 40:
            self.status = "HEALTHY"
        elif self.utilization_percent < 55:
            self.status = "WARNING"
        elif self.utilization_percent < 70:
            self.status = "CRITICAL"
        else:
            self.status = "EXCEEDED"


@dataclass
class AgentSummary:
    """Summary of an agent's activity."""
    agent_id: str
    agent_type: str  # principal, partner, associate
    model: str = "unknown"
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    tool_calls: Counter = field(default_factory=Counter)
    errors: List[Dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    status: str = "unknown"


@dataclass
class WorkModuleSummary:
    """Summary of a work module."""
    module_id: str
    name: str
    description: str
    status: str
    assigned_agent: str
    agent_profile: str = "unknown"  # The profile logical name (e.g., Associate_SmartRAG_EN)
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    tool_calls: Counter = field(default_factory=Counter)
    deliverables_count: int = 0
    message_count: int = 0
    dispatch_status: str = "unknown"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class SessionAnalysis:
    """Complete session analysis result."""
    session_id: str
    run_type: str
    status: str
    created_at: Optional[str]

    # Agent summaries
    partner: Optional[AgentSummary] = None
    principal: Optional[AgentSummary] = None
    work_modules: Dict[str, WorkModuleSummary] = field(default_factory=dict)

    # Aggregates
    total_tokens: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    dispatch_count: int = 0
    successful_dispatches: int = 0

    # Issues detected
    issues: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def estimate_tokens(content: Any) -> int:
    """Estimate token count from content (rough: 1 token ≈ 4 chars)."""
    if content is None:
        return 0
    return len(str(content)) // 4


def get_context_limit(model_name: str) -> int:
    """Get context limit for a model. Tries to use guardian if available."""
    try:
        from agent_core.framework.context_budget_guardian import get_model_context_limit
        return get_model_context_limit(model_name)
    except ImportError:
        pass
    except Exception:
        # Guardian import succeeded but function call failed
        pass

    # Fallback defaults - use improved matching
    model_lower = model_name.lower()

    # Claude models - all have 200K context
    if "claude" in model_lower:
        return 200000

    # OpenAI models
    if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
        return 128000
    if "gpt-4" in model_lower:
        return 8192
    if "gpt-3.5" in model_lower:
        return 16385

    # Gemini models
    if "gemini" in model_lower:
        return 1000000

    # Default for unknown models
    return 200000


def analyze_messages(messages: List[Dict], model_name: str = "unknown") -> Tuple[TokenMetrics, Counter, List[Dict]]:
    """Analyze a list of messages for tokens, tool calls, and errors."""
    metrics = TokenMetrics()
    metrics.context_limit = get_context_limit(model_name)
    tool_calls = Counter()
    errors = []

    metrics.message_count = len(messages)

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        # Count tokens
        content = msg.get("content", "")
        metrics.estimated_tokens += estimate_tokens(content)

        # Count tool calls
        for tc in msg.get("tool_calls", []):
            if isinstance(tc, dict):
                name = tc.get("function", {}).get("name", "unknown")
                tool_calls[name] += 1

        # Detect errors
        content_str = str(content).lower()
        if "error" in content_str or "failed" in content_str:
            if msg.get("role") == "tool":
                errors.append({
                    "type": "tool_error",
                    "preview": str(content)[:200]
                })

    metrics.calculate()
    return metrics, tool_calls, errors


# Default model when not stored in session (Claude Sonnet 4 is the typical default)
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def analyze_partner(sub_contexts: Dict) -> Optional[AgentSummary]:
    """Analyze Partner agent context."""
    partner_ctx = sub_contexts.get("_partner_context_ref", {})
    if not partner_ctx:
        return None

    messages = partner_ctx.get("messages", [])
    model = partner_ctx.get("model", DEFAULT_MODEL)

    summary = AgentSummary(
        agent_id="Partner",
        agent_type="partner",
        model=model
    )

    summary.tokens, summary.tool_calls, summary.errors = analyze_messages(messages, model)
    summary.status = summary.tokens.status

    return summary


def analyze_principal(sub_contexts: Dict) -> Optional[AgentSummary]:
    """Analyze Principal agent context."""
    principal_ctx = sub_contexts.get("_principal_context_ref", {})
    if not principal_ctx:
        return None

    messages = principal_ctx.get("messages", [])
    model = principal_ctx.get("model", DEFAULT_MODEL)

    # Check for context_budget status in the context
    budget_status = principal_ctx.get("context_budget", {})

    summary = AgentSummary(
        agent_id="Principal",
        agent_type="principal",
        model=model
    )

    summary.tokens, summary.tool_calls, summary.errors = analyze_messages(messages, model)

    # Use stored budget status if available
    if budget_status:
        summary.tokens.status = budget_status.get("status", summary.tokens.status)
        if budget_status.get("utilization_percent"):
            summary.tokens.utilization_percent = budget_status["utilization_percent"]
    else:
        summary.status = summary.tokens.status

    return summary


def analyze_work_modules(team_state: Dict) -> Dict[str, WorkModuleSummary]:
    """Analyze all work modules."""
    work_modules = team_state.get("work_modules", {})
    summaries = {}

    for wm_id, wm in work_modules.items():
        if not isinstance(wm, dict):
            continue

        summary = WorkModuleSummary(
            module_id=wm_id,
            name=wm.get("name", "unnamed"),
            description=wm.get("description", "")[:100],
            status=wm.get("status", "unknown"),
            assigned_agent="unknown",
            created_at=wm.get("created_at"),
            updated_at=wm.get("updated_at")
        )

        # Get assigned agent from history
        assignee_history = wm.get("assignee_history", [])
        if assignee_history and isinstance(assignee_history[0], dict):
            summary.assigned_agent = assignee_history[0].get("agent", "unknown")

        # Analyze context archive
        context_archive = wm.get("context_archive", [])
        if isinstance(context_archive, list) and context_archive:
            archive = context_archive[0] if isinstance(context_archive[0], dict) else {}
            messages = archive.get("messages", [])
            model = archive.get("model", DEFAULT_MODEL)

            summary.tokens, summary.tool_calls, _ = analyze_messages(messages, model)
            summary.message_count = len(messages)
            summary.deliverables_count = len(archive.get("deliverables", []))

        summaries[wm_id] = summary

    # Analyze dispatch history - get dispatch status and profile name
    dispatch_history = team_state.get("dispatch_history", [])
    for dispatch in dispatch_history:
        if isinstance(dispatch, dict):
            module_id = dispatch.get("module_id")
            if module_id and module_id in summaries:
                summaries[module_id].dispatch_status = dispatch.get("status", "unknown")
                # Get the actual profile name from dispatch history
                profile_name = dispatch.get("profile_logical_name", "")
                if profile_name:
                    summaries[module_id].agent_profile = profile_name

    return summaries


def detect_issues(analysis: SessionAnalysis) -> List[Dict]:
    """Detect potential issues in the session."""
    issues = []

    # Check Principal exceeded budget
    if analysis.principal and analysis.principal.tokens.status == "EXCEEDED":
        issues.append({
            "severity": "HIGH",
            "type": "context_exceeded",
            "agent": "Principal",
            "details": f"Principal exceeded context budget: {analysis.principal.tokens.utilization_percent:.1f}%"
        })

    # Check for infinite loop patterns (many repeated tool calls)
    if analysis.principal:
        for tool, count in analysis.principal.tool_calls.most_common(3):
            if count > 50 and tool in ["generate_message_summary", "finish_flow"]:
                issues.append({
                    "severity": "HIGH",
                    "type": "potential_loop",
                    "agent": "Principal",
                    "details": f"Tool '{tool}' called {count} times - possible infinite loop"
                })

    # Check for failed dispatches
    for wm_id, wm in analysis.work_modules.items():
        if "FAIL" in wm.dispatch_status.upper():
            issues.append({
                "severity": "MEDIUM",
                "type": "dispatch_failed",
                "agent": wm_id,
                "details": f"Work module {wm_id} dispatch failed: {wm.dispatch_status}"
            })

    # Check for empty deliverables on completed modules
    for wm_id, wm in analysis.work_modules.items():
        if wm.status == "pending_review" and wm.deliverables_count == 0:
            issues.append({
                "severity": "LOW",
                "type": "no_deliverables",
                "agent": wm_id,
                "details": f"Work module {wm_id} completed but has no deliverables"
            })

    return issues


def analyze_session(session_path: Path) -> SessionAnalysis:
    """Perform complete session analysis."""
    with open(session_path, 'r') as f:
        data = json.load(f)

    meta = data.get("meta", {})
    team_state = data.get("team_state", {})
    sub_contexts = data.get("sub_contexts_state", {})

    analysis = SessionAnalysis(
        session_id=meta.get("run_id", "unknown"),
        run_type=meta.get("run_type", "unknown"),
        status=meta.get("status", "unknown"),
        created_at=meta.get("creation_timestamp")
    )

    # Analyze agents
    analysis.partner = analyze_partner(sub_contexts)
    analysis.principal = analyze_principal(sub_contexts)
    analysis.work_modules = analyze_work_modules(team_state)

    # Calculate aggregates
    if analysis.partner:
        analysis.total_tokens += analysis.partner.tokens.estimated_tokens
        analysis.total_messages += analysis.partner.tokens.message_count
        analysis.total_tool_calls += sum(analysis.partner.tool_calls.values())

    if analysis.principal:
        analysis.total_tokens += analysis.principal.tokens.estimated_tokens
        analysis.total_messages += analysis.principal.tokens.message_count
        analysis.total_tool_calls += sum(analysis.principal.tool_calls.values())

    for wm in analysis.work_modules.values():
        analysis.total_tokens += wm.tokens.estimated_tokens
        analysis.total_messages += wm.message_count
        analysis.total_tool_calls += sum(wm.tool_calls.values())
        analysis.dispatch_count += 1
        if "SUCCESS" in wm.dispatch_status.upper():
            analysis.successful_dispatches += 1

    # Detect issues
    analysis.issues = detect_issues(analysis)

    return analysis


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def status_color(status: str) -> str:
    """Get color for a status."""
    status_upper = status.upper()
    if status_upper in ["HEALTHY", "SUCCESS", "COMPLETED_SUCCESS"]:
        return Colors.GREEN
    elif status_upper in ["WARNING", "PENDING_REVIEW"]:
        return Colors.YELLOW
    elif status_upper in ["CRITICAL"]:
        return Colors.MAGENTA
    elif status_upper in ["EXCEEDED", "FAILED", "ERROR"]:
        return Colors.RED
    return Colors.RESET


def format_tokens(metrics: TokenMetrics) -> str:
    """Format token metrics with color."""
    color = status_color(metrics.status)
    return f"{color}{metrics.estimated_tokens:,}{Colors.RESET} / {metrics.context_limit:,} ({metrics.utilization_percent:.1f}% {metrics.status})"


def print_header(text: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}{Colors.RESET}")


def print_subheader(text: str, char: str = "-", width: int = 60):
    """Print a formatted subheader."""
    print(f"\n{Colors.CYAN}{char * width}")
    print(f"  {text}")
    print(f"{char * width}{Colors.RESET}")


def print_summary(analysis: SessionAnalysis):
    """Print summary level output."""
    print_header(f"SESSION ANALYSIS: {analysis.session_id}")

    print(f"\n{Colors.BOLD}Session Info:{Colors.RESET}")
    print(f"  Run Type: {analysis.run_type}")
    print(f"  Status: {status_color(analysis.status)}{analysis.status}{Colors.RESET}")
    print(f"  Created: {analysis.created_at or 'unknown'}")

    print(f"\n{Colors.BOLD}Aggregates:{Colors.RESET}")
    print(f"  Total Messages: {analysis.total_messages:,}")
    print(f"  Total Tokens: ~{analysis.total_tokens:,}")
    print(f"  Total Tool Calls: {analysis.total_tool_calls:,}")
    print(f"  Dispatches: {analysis.successful_dispatches}/{analysis.dispatch_count} successful")

    # Agent overview
    print_subheader("AGENT OVERVIEW")

    if analysis.partner:
        p = analysis.partner
        print(f"\n  {Colors.BOLD}Partner{Colors.RESET}")
        print(f"    Tokens: {format_tokens(p.tokens)}")
        print(f"    Messages: {p.tokens.message_count}")

    if analysis.principal:
        p = analysis.principal
        color = status_color(p.tokens.status)
        print(f"\n  {Colors.BOLD}Principal{Colors.RESET}")
        print(f"    Tokens: {format_tokens(p.tokens)}")
        print(f"    Messages: {p.tokens.message_count}")
        print(f"    Top Tools: {', '.join(f'{t}({c})' for t, c in p.tool_calls.most_common(5))}")

    if analysis.work_modules:
        print(f"\n  {Colors.BOLD}Work Modules ({len(analysis.work_modules)}){Colors.RESET}")
        for wm_id, wm in sorted(analysis.work_modules.items()):
            status_c = status_color(wm.dispatch_status)
            profile_str = f" ({wm.agent_profile})" if wm.agent_profile != "unknown" else ""
            print(f"    {wm_id}{profile_str}: {status_c}{wm.dispatch_status}{Colors.RESET} - {wm.tokens.estimated_tokens:,} tokens, {wm.message_count} msgs")

    # Issues
    if analysis.issues:
        print_subheader(f"ISSUES DETECTED ({len(analysis.issues)})")
        for issue in analysis.issues:
            sev_color = Colors.RED if issue["severity"] == "HIGH" else (Colors.YELLOW if issue["severity"] == "MEDIUM" else Colors.GRAY)
            print(f"\n  {sev_color}[{issue['severity']}]{Colors.RESET} {issue['type']}")
            print(f"    Agent: {issue['agent']}")
            print(f"    {issue['details']}")
    else:
        print(f"\n{Colors.GREEN}✓ No issues detected{Colors.RESET}")


def print_detailed(analysis: SessionAnalysis):
    """Print detailed level output."""
    print_summary(analysis)

    # Detailed agent analysis
    if analysis.principal:
        print_subheader("PRINCIPAL AGENT DETAILS")
        p = analysis.principal
        print(f"\n  Model: {p.model}")
        print(f"  Token Budget Status: {status_color(p.tokens.status)}{p.tokens.status}{Colors.RESET}")
        print(f"\n  {Colors.BOLD}Tool Usage:{Colors.RESET}")
        for tool, count in p.tool_calls.most_common():
            bar = "█" * min(count // 5, 40)
            print(f"    {tool:40} {count:5} {Colors.GRAY}{bar}{Colors.RESET}")

        if p.errors:
            print(f"\n  {Colors.RED}Errors ({len(p.errors)}):{Colors.RESET}")
            for err in p.errors[:5]:
                print(f"    - {err['type']}: {err['preview'][:100]}...")

    # Work module details
    if analysis.work_modules:
        print_subheader("WORK MODULE DETAILS")
        for wm_id, wm in sorted(analysis.work_modules.items()):
            print(f"\n  {Colors.BOLD}{wm_id}: {wm.name[:50]}{Colors.RESET}")
            print(f"    Profile: {Colors.CYAN}{wm.agent_profile}{Colors.RESET}")
            print(f"    Status: {status_color(wm.status)}{wm.status}{Colors.RESET}")
            print(f"    Dispatch: {status_color(wm.dispatch_status)}{wm.dispatch_status}{Colors.RESET}")
            print(f"    Tokens: {format_tokens(wm.tokens)}")
            print(f"    Messages: {wm.message_count}")
            print(f"    Deliverables: {wm.deliverables_count}")
            if wm.tool_calls:
                tools = ", ".join(f"{t}({c})" for t, c in wm.tool_calls.most_common(5))
                print(f"    Tools: {tools}")


def print_deep(analysis: SessionAnalysis, focus: str = "all", session_path: Path = None):
    """Print deep level output with message-level details."""
    print_detailed(analysis)

    if not session_path:
        print(f"\n{Colors.YELLOW}Note: Deep analysis requires session path for message details{Colors.RESET}")
        return

    with open(session_path, 'r') as f:
        data = json.load(f)

    sub_contexts = data.get("sub_contexts_state", {})

    # Deep dive based on focus
    if focus in ["all", "principal"] and analysis.principal:
        print_subheader("PRINCIPAL MESSAGE TRACE")
        principal_ctx = sub_contexts.get("_principal_context_ref", {})
        messages = principal_ctx.get("messages", [])

        # Show first 10 and last 10 messages
        print(f"\n  First 10 messages:")
        for i, msg in enumerate(messages[:10]):
            role = msg.get("role", "?")
            tc = [tc.get("function", {}).get("name") for tc in msg.get("tool_calls", [])]
            tc_str = f" -> {tc}" if tc else ""
            content_preview = str(msg.get("content", ""))[:80].replace("\n", " ")
            print(f"    [{i:4}] {role:10}{tc_str}")

        if len(messages) > 20:
            print(f"\n    ... {len(messages) - 20} messages omitted ...")

        print(f"\n  Last 10 messages:")
        for i, msg in enumerate(messages[-10:]):
            idx = len(messages) - 10 + i
            role = msg.get("role", "?")
            tc = [tc.get("function", {}).get("name") for tc in msg.get("tool_calls", [])]
            tc_str = f" -> {tc}" if tc else ""
            print(f"    [{idx:4}] {role:10}{tc_str}")

    # Work module deep dive
    if focus.startswith("WM_"):
        team_state = data.get("team_state", {})
        work_modules = team_state.get("work_modules", {})
        wm = work_modules.get(focus)
        if wm:
            print_subheader(f"DEEP DIVE: {focus}")
            context_archive = wm.get("context_archive", [])
            if context_archive and isinstance(context_archive[0], dict):
                archive = context_archive[0]
                messages = archive.get("messages", [])

                print(f"\n  All {len(messages)} messages:")
                for i, msg in enumerate(messages):
                    role = msg.get("role", "?")
                    tc = [tc.get("function", {}).get("name") for tc in msg.get("tool_calls", [])]
                    tc_str = f" -> {tc}" if tc else ""
                    content = str(msg.get("content", ""))[:100].replace("\n", " ")
                    print(f"    [{i:3}] {role:10}{tc_str}")
                    if content and role == "assistant" and not tc:
                        print(f"          {Colors.GRAY}{content}...{Colors.RESET}")


def print_token_focus(analysis: SessionAnalysis):
    """Print token-focused analysis."""
    print_header("TOKEN UTILIZATION ANALYSIS")

    print(f"\n{Colors.BOLD}Budget Thresholds:{Colors.RESET}")
    print(f"  {Colors.GREEN}HEALTHY{Colors.RESET}:  < 40%")
    print(f"  {Colors.YELLOW}WARNING{Colors.RESET}:  40-55%")
    print(f"  {Colors.MAGENTA}CRITICAL{Colors.RESET}: 55-70%")
    print(f"  {Colors.RED}EXCEEDED{Colors.RESET}: > 70%")

    print_subheader("TOKEN UTILIZATION BY AGENT")

    # Create a visual chart
    agents = []
    if analysis.partner:
        agents.append(("Partner", analysis.partner.tokens))
    if analysis.principal:
        agents.append(("Principal", analysis.principal.tokens))
    for wm_id, wm in sorted(analysis.work_modules.items()):
        agents.append((wm_id, wm.tokens))

    # Find max for scaling
    max_tokens = max(a[1].estimated_tokens for a in agents) if agents else 1

    for name, metrics in agents:
        bar_width = int((metrics.estimated_tokens / max_tokens) * 40) if max_tokens > 0 else 0
        color = status_color(metrics.status)
        bar = "█" * bar_width

        print(f"\n  {name:20}")
        print(f"    {color}{bar}{Colors.RESET}")
        print(f"    {metrics.estimated_tokens:,} / {metrics.context_limit:,} tokens ({metrics.utilization_percent:.1f}%)")
        print(f"    Status: {color}{metrics.status}{Colors.RESET}")

    # Summary
    total_available = sum(a[1].context_limit for a in agents)
    total_used = sum(a[1].estimated_tokens for a in agents)
    overall_util = (total_used / total_available * 100) if total_available > 0 else 0

    print_subheader("SUMMARY")
    print(f"\n  Total tokens used: {total_used:,}")
    print(f"  Total available: {total_available:,}")
    print(f"  Overall utilization: {overall_util:.1f}%")


def print_timeline(analysis: SessionAnalysis, session_path: Path = None):
    """Print chronological event timeline."""
    print_header("SESSION TIMELINE")

    if not session_path:
        print(f"\n{Colors.YELLOW}Note: Timeline requires session path{Colors.RESET}")
        return

    with open(session_path, 'r') as f:
        data = json.load(f)

    team_state = data.get("team_state", {})
    work_modules = team_state.get("work_modules", {})
    dispatch_history = team_state.get("dispatch_history", [])

    # Collect timeline events
    events = []

    # Session start
    meta = data.get("meta", {})
    if meta.get("creation_timestamp"):
        events.append({
            "time": meta["creation_timestamp"],
            "type": "session_start",
            "details": f"Session created: {analysis.run_type}"
        })

    # Work module creation and updates
    for wm_id, wm in work_modules.items():
        if wm.get("created_at"):
            events.append({
                "time": wm["created_at"],
                "type": "wm_created",
                "details": f"{wm_id} created: {wm.get('name', 'unnamed')[:40]}"
            })
        if wm.get("updated_at"):
            events.append({
                "time": wm["updated_at"],
                "type": "wm_updated",
                "details": f"{wm_id} updated: status={wm.get('status', 'unknown')}"
            })

    # Sort by time
    events.sort(key=lambda e: e.get("time", ""))

    print(f"\n{Colors.BOLD}Chronological Events:{Colors.RESET}")
    for i, event in enumerate(events):
        type_colors = {
            "session_start": Colors.BLUE,
            "wm_created": Colors.CYAN,
            "wm_updated": Colors.GREEN,
            "dispatch": Colors.YELLOW,
            "error": Colors.RED
        }
        color = type_colors.get(event["type"], Colors.RESET)
        time_str = event.get("time", "unknown")[:19]  # Trim to readable format

        print(f"\n  {Colors.GRAY}{time_str}{Colors.RESET}")
        print(f"  {color}[{event['type']:15}]{Colors.RESET} {event['details']}")

    # Dispatch summary
    if dispatch_history:
        print_subheader("DISPATCH SEQUENCE")
        for i, dispatch in enumerate(dispatch_history):
            status = dispatch.get("status", "unknown")
            color = status_color(status)
            print(f"  {i+1}. {dispatch.get('module_id', '?'):10} {color}{status}{Colors.RESET}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze CommonGround session for debugging and observability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("session_input",
                       help="Session file path, URL (http://localhost:3800/webview/r?id=SESSION_ID), or session ID")
    parser.add_argument("--level", "-l",
                       choices=["summary", "detailed", "deep", "timeline"],
                       default="summary",
                       help="Analysis detail level (default: summary)")
    parser.add_argument("--focus", "-f",
                       default="all",
                       help="Focus area: all, principal, partner, WM_N, errors, tokens")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON instead of formatted text")

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    # Resolve session input to file path
    try:
        session_path = resolve_session_input(args.session_input)
        print(f"{Colors.GRAY}Resolved session: {session_path}{Colors.RESET}\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)

    # Perform analysis
    try:
        analysis = analyze_session(session_path)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Error: Invalid JSON in session file: {e}{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}Error analyzing session: {e}{Colors.RESET}")
        raise

    # Output based on format
    if args.json:
        # Convert to JSON-serializable dict
        output = {
            "session_id": analysis.session_id,
            "run_type": analysis.run_type,
            "status": analysis.status,
            "created_at": analysis.created_at,
            "total_tokens": analysis.total_tokens,
            "total_messages": analysis.total_messages,
            "total_tool_calls": analysis.total_tool_calls,
            "dispatch_count": analysis.dispatch_count,
            "successful_dispatches": analysis.successful_dispatches,
            "issues": analysis.issues,
            "partner": {
                "tokens": analysis.partner.tokens.estimated_tokens if analysis.partner else 0,
                "status": analysis.partner.tokens.status if analysis.partner else "N/A",
            } if analysis.partner else None,
            "principal": {
                "tokens": analysis.principal.tokens.estimated_tokens if analysis.principal else 0,
                "status": analysis.principal.tokens.status if analysis.principal else "N/A",
                "utilization_percent": analysis.principal.tokens.utilization_percent if analysis.principal else 0,
            } if analysis.principal else None,
            "work_modules": {
                wm_id: {
                    "agent_profile": wm.agent_profile,
                    "status": wm.status,
                    "dispatch_status": wm.dispatch_status,
                    "tokens": wm.tokens.estimated_tokens,
                    "messages": wm.message_count,
                }
                for wm_id, wm in analysis.work_modules.items()
            }
        }
        print(json.dumps(output, indent=2))
        return

    # Formatted output based on level and focus
    if args.focus == "tokens":
        print_token_focus(analysis)
    elif args.level == "timeline":
        print_timeline(analysis, session_path)
    elif args.level == "deep":
        print_deep(analysis, args.focus, session_path)
    elif args.level == "detailed":
        print_detailed(analysis)
    else:
        print_summary(analysis)


if __name__ == "__main__":
    main()
