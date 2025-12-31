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
- Thrashing root cause analysis

Usage:
    python analyze_session.py <session_path_or_url> [--mode MODE] [--agent AGENT]

Input:
    Can be either:
    - A file path: projects/MyProject/session-id.json
    - A session URL: http://localhost:3800/webview/r?id=tentacled-pearl-oriole
    - Just a session ID: tentacled-pearl-oriole

Modes (what to analyze):
    summary   - High-level overview with issue detection (default)
    detailed  - Per-agent breakdown with key metrics and tool usage
    tokens    - Token utilization analysis with visual charts
    handoff   - Deliverable flow and handoff issue analysis
    thrashing - Root cause analysis for duplicate dispatches
    timeline  - Chronological event trace
    errors    - Focus on errors and issues only
    all       - Run all analysis modes sequentially
              ⚠️  WARNING: 'all' produces very large output that may crash
              some environments (e.g., VS Code agent terminal). Call modes
              individually instead, or redirect output to a file.

Agent Filter (optional, narrows scope):
    --agent principal  - Focus on Principal agent
    --agent partner    - Focus on Partner agent  
    --agent WM_1       - Focus on specific work module (e.g., WM_1, WM_2)

Output Options:
    --json      - Output as JSON instead of formatted text
    --no-color  - Disable colored output

Examples:
    python analyze_session.py tentacled-pearl-oriole
    python analyze_session.py tentacled-pearl-oriole --mode detailed
    python analyze_session.py tentacled-pearl-oriole --mode tokens
    python analyze_session.py tentacled-pearl-oriole --mode handoff
    python analyze_session.py tentacled-pearl-oriole --mode thrashing
    python analyze_session.py tentacled-pearl-oriole --mode all
    python analyze_session.py tentacled-pearl-oriole --mode detailed --agent WM_1
    python analyze_session.py tentacled-pearl-oriole --json
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
    dispatch_count: int = 0  # How many times this module was dispatched (>1 = thrashing)
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

        # Detect errors - look for actual error indicators, not just the word "error"
        content_str = str(content).lower()
        if msg.get("role") == "tool":
            # Check for actual error patterns, excluding success messages
            is_error = False
            if "overall status**: `success`" in content_str:
                is_error = False  # Not an error - it's a success report
            elif any(pattern in content_str for pattern in [
                "tool_execution_failed",
                "exception",
                "traceback",
                "status\": \"error",
                "\"error\":",
                "failed to",
                "could not",
                "unable to",
                "error occurred",
                "error:",
            ]):
                is_error = True
            
            if is_error:
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

        # Analyze ALL context archives (important for modules dispatched multiple times)
        context_archive = wm.get("context_archive", [])
        total_tokens = TokenMetrics()
        total_tool_calls = Counter()
        total_messages = 0
        total_deliverables = 0
        
        for archive in context_archive:
            if not isinstance(archive, dict):
                continue
            messages = archive.get("messages", [])
            model = archive.get("model", DEFAULT_MODEL)
            
            archive_tokens, archive_tools, _ = analyze_messages(messages, model)
            total_tokens.estimated_tokens += archive_tokens.estimated_tokens
            total_tokens.message_count += archive_tokens.message_count
            total_tool_calls.update(archive_tools)
            total_messages += len(messages)
            
            # Count deliverables - check both dict format and list format
            deliverables = archive.get("deliverables", {})
            if isinstance(deliverables, dict) and deliverables.get("primary_summary"):
                total_deliverables += 1
            elif isinstance(deliverables, list):
                total_deliverables += len(deliverables)
        
        summary.tokens = total_tokens
        summary.tool_calls = total_tool_calls
        summary.message_count = total_messages
        summary.deliverables_count = total_deliverables
        summary.dispatch_count = len(context_archive)  # Track how many times dispatched

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
            thrash_indicator = f" {Colors.RED}(dispatched {wm.dispatch_count}x!){Colors.RESET}" if wm.dispatch_count > 1 else ""
            print(f"\n  {Colors.BOLD}{wm_id}: {wm.name[:50]}{Colors.RESET}{thrash_indicator}")
            print(f"    Profile: {Colors.CYAN}{wm.agent_profile}{Colors.RESET}")
            print(f"    Status: {status_color(wm.status)}{wm.status}{Colors.RESET}")
            print(f"    Dispatch: {status_color(wm.dispatch_status)}{wm.dispatch_status}{Colors.RESET}")
            print(f"    Tokens: {format_tokens(wm.tokens)}")
            print(f"    Messages: {wm.message_count}")
            print(f"    Deliverables: {wm.deliverables_count}")
            if wm.tool_calls:
                tools = ", ".join(f"{t}({c})" for t, c in wm.tool_calls.most_common(5))
                print(f"    Tools: {tools}")


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


def print_handoff_analysis(analysis: SessionAnalysis, session_path: Path = None):
    """
    Analyze delegation handoffs, message inheritance, and deliverable flow.
    
    This mode answers:
    1. Why did an agent not return deliverables properly?
    2. Why couldn't the principal access a subagent's messages?
    3. Why couldn't newly spawned subagents access earlier agent's messages?
    """
    print_header("HANDOFF & DELIVERABLE FLOW ANALYSIS")

    if not session_path:
        print(f"\n{Colors.YELLOW}Note: Handoff analysis requires session path{Colors.RESET}")
        return

    with open(session_path, 'r') as f:
        data = json.load(f)

    team_state = data.get("team_state", {})
    work_modules = team_state.get("work_modules", {})
    dispatch_history = team_state.get("dispatch_history", [])

    # ==========================================================================
    # SECTION 1: Dispatch History Analysis
    # ==========================================================================
    print_subheader("1. DISPATCH HISTORY (Delegation Chain)")
    
    # Track duplicate dispatches
    dispatch_counts = Counter(d.get("module_id") for d in dispatch_history)
    duplicates = {k: v for k, v in dispatch_counts.items() if v > 1}
    
    if duplicates:
        print(f"\n  {Colors.RED}⚠ DUPLICATE DISPATCHES DETECTED:{Colors.RESET}")
        for module_id, count in duplicates.items():
            print(f"    {module_id} dispatched {count} times (possible thrashing)")
    
    print(f"\n  {Colors.BOLD}Dispatch Sequence:{Colors.RESET}")
    for i, dispatch in enumerate(dispatch_history):
        module_id = dispatch.get("module_id", "?")
        status = dispatch.get("status", "unknown")
        profile = dispatch.get("profile_logical_name", "unknown")
        timestamp = dispatch.get("timestamp", dispatch.get("dispatched_at", ""))[:19]
        color = status_color(status)
        
        # Check for notes_from_principal
        notes = dispatch.get("notes_from_principal", "")
        notes_preview = f" | notes: {notes[:60]}..." if notes else ""
        
        print(f"\n    {Colors.GRAY}[{i+1}] {timestamp}{Colors.RESET}")
        print(f"        Module: {module_id} -> Profile: {Colors.CYAN}{profile}{Colors.RESET}")
        print(f"        Status: {color}{status}{Colors.RESET}{notes_preview}")

    # ==========================================================================
    # SECTION 2: Work Module Deliverables Analysis  
    # ==========================================================================
    print_subheader("2. DELIVERABLE EXTRACTION ANALYSIS")
    
    for wm_id, wm in sorted(work_modules.items()):
        wm_name = wm.get("name", "unnamed")[:50]
        status = wm.get("status", "unknown")
        
        # Check deliverables array (what Principal sees)
        deliverables_arr = wm.get("deliverables", [])
        
        # Check context_archive (what was actually produced)
        context_archive = wm.get("context_archive", [])
        archived_deliverables = []
        archived_messages = []
        
        for archive in context_archive:
            if isinstance(archive, dict):
                arch_del = archive.get("deliverables", {})
                if arch_del:
                    archived_deliverables.append(arch_del)
                arch_msgs = archive.get("messages", [])
                archived_messages.extend(arch_msgs)
        
        print(f"\n  {Colors.BOLD}{wm_id}: {wm_name}{Colors.RESET}")
        print(f"    Status: {status_color(status)}{status}{Colors.RESET}")
        
        # Deliverables array check
        if deliverables_arr:
            print(f"    {Colors.GREEN}✓ deliverables[] has {len(deliverables_arr)} items{Colors.RESET}")
        else:
            print(f"    {Colors.YELLOW}⚠ deliverables[] is EMPTY{Colors.RESET}")
        
        # Context archive check
        if archived_deliverables:
            for j, ad in enumerate(archived_deliverables):
                primary = ad.get("primary_summary", "")
                print(f"    {Colors.GREEN}✓ context_archive[{j}].deliverables.primary_summary: {len(primary)} chars{Colors.RESET}")
                if primary:
                    preview = primary[:150].replace("\n", " ")
                    print(f"      Preview: {Colors.GRAY}{preview}...{Colors.RESET}")
        else:
            print(f"    {Colors.RED}✗ No deliverables in context_archive{Colors.RESET}")
        
        # Check for finish_flow in messages (did agent properly finish?)
        finish_calls = [m for m in archived_messages if any(
            tc.get("function", {}).get("name") == "finish_flow" 
            for tc in m.get("tool_calls", [])
        )]
        if finish_calls:
            print(f"    {Colors.GREEN}✓ finish_flow called {len(finish_calls)} time(s){Colors.RESET}")
        else:
            print(f"    {Colors.RED}✗ finish_flow NOT called - agent may not have completed properly{Colors.RESET}")

    # ==========================================================================
    # SECTION 3: Message Inheritance Analysis
    # ==========================================================================
    print_subheader("3. MESSAGE INHERITANCE CHAIN")
    
    # Check what messages each work module inherited
    for wm_id, wm in sorted(work_modules.items()):
        context_archive = wm.get("context_archive", [])
        if not context_archive:
            continue
            
        for arch_idx, archive in enumerate(context_archive):
            if not isinstance(archive, dict):
                continue
                
            messages = archive.get("messages", [])
            if not messages:
                continue
            
            # First message is typically the briefing/inherited content
            first_msg = messages[0] if messages else {}
            first_content = str(first_msg.get("content", ""))
            
            print(f"\n  {Colors.BOLD}{wm_id} (archive {arch_idx}):{Colors.RESET}")
            print(f"    Total messages: {len(messages)}")
            
            # Check for inherited message markers
            if "inherit" in first_content.lower() or "previous" in first_content.lower():
                print(f"    {Colors.GREEN}✓ Appears to have inherited context{Colors.RESET}")
            
            # Look for references to other work modules
            other_wm_refs = re.findall(r'WM_\d+', first_content)
            if other_wm_refs:
                print(f"    References to other modules: {', '.join(set(other_wm_refs))}")
            
            # Check first message length (briefing size)
            briefing_size = len(first_content)
            print(f"    Initial briefing size: {briefing_size:,} chars (~{briefing_size//4:,} tokens)")
            
            # Check if briefing mentions deliverables from previous agents
            if "deliverable" in first_content.lower():
                print(f"    {Colors.GREEN}✓ Briefing mentions deliverables{Colors.RESET}")
            else:
                print(f"    {Colors.YELLOW}⚠ Briefing does NOT mention deliverables{Colors.RESET}")

    # ==========================================================================
    # SECTION 4: Potential Issues Summary
    # ==========================================================================
    print_subheader("4. HANDOFF ISSUES DETECTED")
    
    issues_found = []
    
    # Check for duplicate dispatches
    if duplicates:
        issues_found.append({
            "severity": "HIGH",
            "type": "duplicate_dispatch",
            "details": f"Modules dispatched multiple times: {list(duplicates.keys())} - indicates thrashing"
        })
    
    # Check for empty deliverables on completed modules
    for wm_id, wm in work_modules.items():
        if wm.get("status") in ["completed", "pending_review"]:
            if not wm.get("deliverables"):
                # Check if archive has deliverables (data model mismatch)
                context_archive = wm.get("context_archive", [])
                has_archived = any(
                    isinstance(a, dict) and a.get("deliverables", {}).get("primary_summary")
                    for a in context_archive
                )
                if has_archived:
                    issues_found.append({
                        "severity": "MEDIUM", 
                        "type": "deliverable_not_propagated",
                        "details": f"{wm_id}: Deliverables exist in context_archive but NOT in work_modules.deliverables[] - Principal may not see them"
                    })
                else:
                    issues_found.append({
                        "severity": "HIGH",
                        "type": "no_deliverables",
                        "details": f"{wm_id}: Completed but NO deliverables anywhere"
                    })
    
    # Check dispatch vs completion status mismatch
    for dispatch in dispatch_history:
        module_id = dispatch.get("module_id")
        dispatch_status = dispatch.get("status", "")
        if module_id in work_modules:
            wm_status = work_modules[module_id].get("status", "")
            if "RUNNING" in dispatch_status and wm_status == "completed":
                issues_found.append({
                    "severity": "LOW",
                    "type": "status_mismatch",
                    "details": f"{module_id}: dispatch_history says RUNNING but work_module says completed"
                })
    
    if issues_found:
        for issue in issues_found:
            sev = issue["severity"]
            sev_color = Colors.RED if sev == "HIGH" else (Colors.YELLOW if sev == "MEDIUM" else Colors.GRAY)
            print(f"\n  {sev_color}[{sev}]{Colors.RESET} {issue['type']}")
            print(f"    {issue['details']}")
    else:
        print(f"\n  {Colors.GREEN}✓ No handoff issues detected{Colors.RESET}")


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


def print_thrashing_analysis(analysis: SessionAnalysis, session_path: Path = None):
    """
    Analyze WHY thrashing occurred - trace Principal's decision-making.
    
    Shows:
    1. Principal's tool calls leading up to each duplicate dispatch
    2. What information Principal had when making decisions
    3. Why Principal thought work wasn't done
    """
    print_header("THRASHING ROOT CAUSE ANALYSIS")
    
    if not session_path:
        print(f"\n{Colors.YELLOW}Note: Thrashing analysis requires session path{Colors.RESET}")
        return
    
    with open(session_path, 'r') as f:
        data = json.load(f)
    
    team_state = data.get("team_state", {})
    sub_contexts = data.get("sub_contexts_state", {})
    dispatch_history = team_state.get("dispatch_history", [])
    work_modules = team_state.get("work_modules", {})
    
    # Find duplicate dispatches
    dispatch_counts = Counter(d.get("module_id") for d in dispatch_history)
    duplicates = {mid: count for mid, count in dispatch_counts.items() if count > 1}
    
    if not duplicates:
        print(f"\n{Colors.GREEN}✓ No duplicate dispatches found (no thrashing){Colors.RESET}")
        return
    
    print_subheader("1. DUPLICATE DISPATCH SUMMARY")
    for mid, count in duplicates.items():
        print(f"\n  {Colors.RED}⚠ {mid}{Colors.RESET} dispatched {count} times")
        # Show each dispatch
        for i, dispatch in enumerate(dispatch_history):
            if dispatch.get("module_id") == mid:
                ts = dispatch.get("start_timestamp", "?")[:19]
                status = dispatch.get("status", "?")
                profile = dispatch.get("profile_logical_name", "?")
                color = status_color(status)
                print(f"    [{i+1}] {ts} -> {profile} -> {color}{status}{Colors.RESET}")
    
    # Analyze Principal's messages around dispatch decisions
    print_subheader("2. PRINCIPAL DECISION TRACE")
    principal_ctx = sub_contexts.get("_principal_context_ref", {})
    principal_messages = principal_ctx.get("messages", [])
    
    # Find dispatch_work_modules tool calls
    dispatch_calls = []
    for i, msg in enumerate(principal_messages):
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                func_name = tc.get("function", {}).get("name", "")
                if func_name == "dispatch_work_modules":
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                        dispatch_calls.append({
                            "msg_index": i,
                            "args": args,
                            "tool_call_id": tc.get("id")
                        })
                    except:
                        pass
    
    print(f"\n  Found {len(dispatch_calls)} dispatch_work_modules calls:")
    
    for dc in dispatch_calls:
        idx = dc["msg_index"]
        args = dc["args"]
        dispatches = args.get("dispatches", [])
        
        print(f"\n  {Colors.CYAN}Message #{idx}{Colors.RESET}")
        
        # Show what modules were being dispatched
        for d in dispatches:
            mid = d.get("module_id_to_assign", "?")
            inherit = d.get("inherit_messages_from", [])
            is_duplicate = mid in duplicates
            dup_marker = f" {Colors.RED}(DUPLICATE){Colors.RESET}" if is_duplicate else ""
            print(f"    -> Dispatching: {mid}{dup_marker}")
            if inherit:
                print(f"       Inheriting from: {inherit}")
        
        # Look at the assistant message content before the dispatch
        if idx > 0:
            prev_msg = principal_messages[idx]
            content = prev_msg.get("content", "")
            if content:
                # Find relevant snippets about the module
                for mid in duplicates:
                    if mid in str(content):
                        # Extract context around the mention
                        lines = str(content).split('\n')
                        relevant = [l for l in lines if mid in l][:5]
                        if relevant:
                            print(f"    {Colors.GRAY}Principal's reasoning about {mid}:{Colors.RESET}")
                            for line in relevant:
                                print(f"      {line[:100]}...")
    
    # Check what the tool results looked like
    print_subheader("3. TOOL RESULTS PRINCIPAL SAW")
    
    for mid in duplicates:
        print(f"\n  {Colors.BOLD}{mid}{Colors.RESET}:")
        
        # Find tool results for this module
        relevant_results = []
        for i, msg in enumerate(principal_messages):
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                if mid in content:
                    tool_id = msg.get("tool_call_id", "?")
                    preview = content[:300].replace('\n', ' ')
                    relevant_results.append({
                        "index": i,
                        "tool_id": tool_id,
                        "preview": preview
                    })
        
        if relevant_results:
            for r in relevant_results[:3]:  # Show first 3
                print(f"    [msg {r['index']}] {r['preview'][:200]}...")
        else:
            print(f"    {Colors.YELLOW}No tool results found mentioning {mid}{Colors.RESET}")
    
    # Check work module status at end
    print_subheader("4. FINAL WORK MODULE STATE")
    for mid in duplicates:
        wm = work_modules.get(mid, {})
        status = wm.get("status", "?")
        archives = len(wm.get("context_archive", []))
        deliverables = wm.get("deliverables", [])
        
        print(f"\n  {mid}:")
        print(f"    Status: {status_color(status)}{status}{Colors.RESET}")
        print(f"    Context archives: {archives}")
        print(f"    work_modules.deliverables[]: {len(deliverables)} items")
        
        # Check what's in context_archive
        for i, arch in enumerate(wm.get("context_archive", [])):
            del_dict = arch.get("deliverables", {})
            summary = del_dict.get("primary_summary", "")
            print(f"    Archive[{i}]: deliverables.primary_summary = {len(summary)} chars")
    
    # Diagnosis
    print_subheader("5. ROOT CAUSE DIAGNOSIS")
    
    # Check if deliverables were in wrong location
    for mid in duplicates:
        wm = work_modules.get(mid, {})
        has_archive_deliverables = any(
            arch.get("deliverables", {}).get("primary_summary")
            for arch in wm.get("context_archive", [])
        )
        has_top_level_deliverables = len(wm.get("deliverables", [])) > 0
        
        if has_archive_deliverables and not has_top_level_deliverables:
            print(f"\n  {Colors.RED}[DATA MODEL ISSUE]{Colors.RESET} {mid}:")
            print(f"    Deliverables ARE in context_archive (correct for inheritance)")
            print(f"    But work_modules[{mid}].deliverables[] is empty (legacy field)")
            print(f"    {Colors.YELLOW}This is expected - the system reads from context_archive{Colors.RESET}")
    
    # Check for flow_decider issues
    flow_decider_calls = sum(1 for msg in principal_messages 
                             if msg.get("role") == "tool" and 
                             "flow_decider" in str(msg.get("name", "")))
    
    if flow_decider_calls > 0:
        print(f"\n  Flow decider invocations: {flow_decider_calls}")
    
    # Check for empty LLM responses
    empty_responses = sum(1 for msg in principal_messages 
                          if msg.get("role") == "assistant" and 
                          not msg.get("content") and 
                          not msg.get("tool_calls"))
    
    if empty_responses > 0:
        print(f"\n  {Colors.YELLOW}[LLM ISSUE]{Colors.RESET} Empty assistant responses: {empty_responses}")
        print(f"    May indicate model confusion or prompt issues")


def print_errors(analysis: SessionAnalysis, session_path: Path = None):
    """Print error-focused analysis."""
    print_header("ERROR ANALYSIS")
    
    # Show detected issues from analysis
    if analysis.issues:
        print_subheader(f"DETECTED ISSUES ({len(analysis.issues)})")
        for issue in analysis.issues:
            sev = issue["severity"]
            sev_color = Colors.RED if sev == "HIGH" else (Colors.YELLOW if sev == "MEDIUM" else Colors.GRAY)
            print(f"\n  {sev_color}[{sev}]{Colors.RESET} {issue['type']}")
            print(f"    Agent: {issue['agent']}")
            print(f"    {issue['details']}")
    else:
        print(f"\n{Colors.GREEN}✓ No issues detected in analysis{Colors.RESET}")
    
    # Show agent errors
    if analysis.principal and analysis.principal.errors:
        print_subheader(f"PRINCIPAL ERRORS ({len(analysis.principal.errors)})")
        for err in analysis.principal.errors[:10]:
            print(f"\n  {Colors.RED}[{err['type']}]{Colors.RESET}")
            print(f"    {err['preview'][:200]}...")
    
    if analysis.partner and analysis.partner.errors:
        print_subheader(f"PARTNER ERRORS ({len(analysis.partner.errors)})")
        for err in analysis.partner.errors[:10]:
            print(f"\n  {Colors.RED}[{err['type']}]{Colors.RESET}")
            print(f"    {err['preview'][:200]}...")
    
    # Scan for errors in work modules
    if session_path:
        with open(session_path, 'r') as f:
            data = json.load(f)
        
        team_state = data.get("team_state", {})
        work_modules = team_state.get("work_modules", {})
        
        for wm_id, wm in work_modules.items():
            context_archive = wm.get("context_archive", [])
            wm_errors = []
            
            for archive in context_archive:
                if not isinstance(archive, dict):
                    continue
                messages = archive.get("messages", [])
                for msg in messages:
                    if msg.get("role") == "tool":
                        content = str(msg.get("content", "")).lower()
                        if "error" in content or "failed" in content or "exception" in content:
                            wm_errors.append(str(msg.get("content", ""))[:200])
            
            if wm_errors:
                print_subheader(f"{wm_id} ERRORS ({len(wm_errors)})")
                for err in wm_errors[:5]:
                    print(f"\n  {Colors.RED}•{Colors.RESET} {err}...")
    
    # Summary
    total_errors = len(analysis.issues)
    if analysis.principal:
        total_errors += len(analysis.principal.errors)
    if analysis.partner:
        total_errors += len(analysis.partner.errors)
    
    print_subheader("SUMMARY")
    if total_errors == 0:
        print(f"\n  {Colors.GREEN}✓ No errors found in session{Colors.RESET}")
    else:
        print(f"\n  {Colors.RED}Total errors/issues: {total_errors}{Colors.RESET}")


def print_agent_detail(analysis: SessionAnalysis, agent_filter: str, session_path: Path = None):
    """Print detailed analysis for a specific agent."""
    
    if agent_filter == "principal":
        if not analysis.principal:
            print(f"{Colors.YELLOW}No Principal agent found in session{Colors.RESET}")
            return
        
        print_header("PRINCIPAL AGENT ANALYSIS")
        p = analysis.principal
        print(f"\n{Colors.BOLD}Agent Info:{Colors.RESET}")
        print(f"  Model: {p.model}")
        print(f"  Messages: {p.tokens.message_count}")
        print(f"  Tokens: {format_tokens(p.tokens)}")
        
        print(f"\n{Colors.BOLD}Tool Usage:{Colors.RESET}")
        for tool, count in p.tool_calls.most_common():
            bar = "█" * min(count // 2, 40)
            print(f"  {tool:40} {count:5} {Colors.GRAY}{bar}{Colors.RESET}")
        
        if p.errors:
            print(f"\n{Colors.RED}Errors ({len(p.errors)}):{Colors.RESET}")
            for err in p.errors[:5]:
                print(f"  - {err['type']}: {err['preview'][:100]}...")
        
        # Show message trace if session available
        if session_path:
            with open(session_path, 'r') as f:
                data = json.load(f)
            sub_contexts = data.get("sub_contexts_state", {})
            principal_ctx = sub_contexts.get("_principal_context_ref", {})
            messages = principal_ctx.get("messages", [])
            
            print_subheader(f"MESSAGE TRACE ({len(messages)} messages)")
            for i, msg in enumerate(messages[:15]):
                role = msg.get("role", "?")
                tc = [tc.get("function", {}).get("name") for tc in msg.get("tool_calls", [])]
                tc_str = f" -> {tc}" if tc else ""
                print(f"  [{i:3}] {role:10}{tc_str}")
            if len(messages) > 15:
                print(f"  ... {len(messages) - 15} more messages")
    
    elif agent_filter == "partner":
        if not analysis.partner:
            print(f"{Colors.YELLOW}No Partner agent found in session{Colors.RESET}")
            return
        
        print_header("PARTNER AGENT ANALYSIS")
        p = analysis.partner
        print(f"\n{Colors.BOLD}Agent Info:{Colors.RESET}")
        print(f"  Model: {p.model}")
        print(f"  Messages: {p.tokens.message_count}")
        print(f"  Tokens: {format_tokens(p.tokens)}")
        
        if p.tool_calls:
            print(f"\n{Colors.BOLD}Tool Usage:{Colors.RESET}")
            for tool, count in p.tool_calls.most_common():
                print(f"  {tool}: {count}")
    
    elif agent_filter.startswith("WM_"):
        wm = analysis.work_modules.get(agent_filter)
        if not wm:
            print(f"{Colors.YELLOW}Work module {agent_filter} not found{Colors.RESET}")
            return
        
        print_header(f"WORK MODULE: {agent_filter}")
        print(f"\n{Colors.BOLD}Module Info:{Colors.RESET}")
        print(f"  Name: {wm.name}")
        print(f"  Description: {wm.description}")
        print(f"  Profile: {Colors.CYAN}{wm.agent_profile}{Colors.RESET}")
        print(f"  Status: {status_color(wm.status)}{wm.status}{Colors.RESET}")
        print(f"  Dispatch: {status_color(wm.dispatch_status)}{wm.dispatch_status}{Colors.RESET}")
        print(f"  Messages: {wm.message_count}")
        print(f"  Tokens: {format_tokens(wm.tokens)}")
        print(f"  Deliverables: {wm.deliverables_count}")
        
        if wm.tool_calls:
            print(f"\n{Colors.BOLD}Tool Usage:{Colors.RESET}")
            for tool, count in wm.tool_calls.most_common():
                print(f"  {tool}: {count}")
        
        # Show message trace from context_archive
        if session_path:
            with open(session_path, 'r') as f:
                data = json.load(f)
            team_state = data.get("team_state", {})
            work_modules = team_state.get("work_modules", {})
            wm_data = work_modules.get(agent_filter, {})
            context_archive = wm_data.get("context_archive", [])
            
            for arch_idx, archive in enumerate(context_archive):
                if not isinstance(archive, dict):
                    continue
                messages = archive.get("messages", [])
                deliverables = archive.get("deliverables", {})
                
                print_subheader(f"ARCHIVE {arch_idx} ({len(messages)} messages)")
                
                for i, msg in enumerate(messages):
                    role = msg.get("role", "?")
                    tc = [tc.get("function", {}).get("name") for tc in msg.get("tool_calls", [])]
                    tc_str = f" -> {tc}" if tc else ""
                    print(f"  [{i:3}] {role:10}{tc_str}")
                
                if deliverables.get("primary_summary"):
                    summary = deliverables["primary_summary"]
                    print(f"\n  {Colors.GREEN}Deliverable ({len(summary)} chars):{Colors.RESET}")
                    print(f"  {Colors.GRAY}{summary[:300]}...{Colors.RESET}")
    else:
        print(f"{Colors.RED}Unknown agent filter: {agent_filter}{Colors.RESET}")
        print(f"Use: principal, partner, or WM_N (e.g., WM_1, WM_2)")


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
                       help="Session file path, URL, or session ID")
    parser.add_argument("--mode", "-m",
                       choices=["summary", "detailed", "tokens", "handoff", "thrashing", "timeline", "errors", "all"],
                       default="summary",
                       help="Analysis mode (default: summary)")
    parser.add_argument("--agent", "-a",
                       default=None,
                       help="Filter to specific agent: principal, partner, or WM_N")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON instead of formatted text")
    
    # Legacy support for old arguments
    parser.add_argument("--level", "-l", dest="legacy_level", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--focus", "-f", dest="legacy_focus", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.legacy_level or args.legacy_focus:
        print(f"{Colors.YELLOW}Note: --level and --focus are deprecated. Use --mode and --agent instead.{Colors.RESET}\n")
        if args.legacy_level in ["detailed", "deep", "timeline"]:
            args.mode = args.legacy_level if args.legacy_level != "deep" else "detailed"
        if args.legacy_focus and args.legacy_focus != "all":
            if args.legacy_focus in ["tokens", "handoff", "thrashing", "errors"]:
                args.mode = args.legacy_focus
            elif args.legacy_focus in ["principal", "partner"] or args.legacy_focus.startswith("WM_"):
                args.agent = args.legacy_focus

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

    # JSON output
    if args.json:
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
                    "dispatch_count": wm.dispatch_count,
                    "deliverables_count": wm.deliverables_count,
                }
                for wm_id, wm in analysis.work_modules.items()
            }
        }
        print(json.dumps(output, indent=2))
        return

    # If agent filter specified, show agent-specific detail
    if args.agent:
        print_agent_detail(analysis, args.agent, session_path)
        return

    # Mode-based output
    if args.mode == "all":
        print_detailed(analysis)  # includes summary
        print_token_focus(analysis)
        print_handoff_analysis(analysis, session_path)
        print_thrashing_analysis(analysis, session_path)
        print_errors(analysis, session_path)
        print_timeline(analysis, session_path)
    elif args.mode == "summary":
        print_summary(analysis)
    elif args.mode == "detailed":
        print_detailed(analysis)
    elif args.mode == "tokens":
        print_token_focus(analysis)
    elif args.mode == "handoff":
        print_handoff_analysis(analysis, session_path)
    elif args.mode == "thrashing":
        print_thrashing_analysis(analysis, session_path)
    elif args.mode == "timeline":
        print_timeline(analysis, session_path)
    elif args.mode == "errors":
        print_errors(analysis, session_path)
    else:
        print_summary(analysis)


if __name__ == "__main__":
    main()
