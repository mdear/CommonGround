# Context Management & Deliverable Handoff - Detailed Fix Design

## Executive Summary

This document addresses three issues discovered during live session analysis:
1. **Epoch 1 `finish_flow` dropped** - Multi-tool call caused critical tool to be ignored
2. **Epoch 2 callback interrupted** - Deliverables existed but weren't propagated to Partner
3. **Partner context exhaustion** - No automatic turn summarization for large-context agents

## Current Architecture: Existing Context Management Features

### What Already Exists

| Feature | Agent Types | Mechanism | Location |
|---------|-------------|-----------|----------|
| **Context Budget Guardian** | All | Monitor utilization, inject directives at thresholds | `context_budget_guardian.py` |
| **Context Admission Controller** | All | Pre-filter large tool results to prevent budget spikes | `context_admission_controller.py` |
| **Context Budget Handback** | Associates | Package KB tokens + partial work for Principal when Associate exceeds budget | `context_budget_handback.py` |
| **Content Selection** | Principal (inheritance) | Budget-aware selection from completed modules using LLM summaries or newest-first | `content_selection.py` |
| **Worker Budget Calculator** | Dispatcher | Divides context pool evenly among parallel workers | `context_budget_guardian.py` |

### Current Thresholds (context_budget_guardian.py)

```python
WARNING_THRESHOLD = 0.60   # 60% - Inject guidance to wrap up
CRITICAL_THRESHOLD = 0.75  # 75% - Force completion (Principal/Associate only)
EXCEEDED_THRESHOLD = 0.85  # 85% - Circuit breaker fires (system messages only)
```

### User Prompt Bypass (Circuit Breaker Exception)

The guardian reserves 15% headroom above the EXCEEDED threshold. This headroom is specifically for **user-initiated messages**, allowing users to continue interacting with agents even after the guardian cap is reached.

**Behavior:**
- **System-generated messages** (observer events, tool results, etc.): Blocked at EXCEEDED threshold
- **User-initiated messages**: Allowed through with informative warning, can use reserved headroom up to actual model context limit

**User-initiated sources:**
- `USER_PROMPT` - Direct user message to agent (e.g., user → Partner)
- `PARTNER_DIRECTIVE` - User request relayed Partner → Principal (e.g., user asks Partner to check Principal status)
- `PRINCIPAL_COMPLETED` - Principal's response returning to Partner after user-initiated research

**Implementation:** `base_agent_node.py` detects user-initiated sources in the inbox processing log and bypasses the circuit breaker:
```python
USER_INITIATED_SOURCES = {"USER_PROMPT", "PARTNER_DIRECTIVE", "PRINCIPAL_COMPLETED"}
is_user_initiated = any(
    log_entry.get("source") in USER_INITIATED_SOURCES
    for log_entry in processing_result.get("processing_log", [])
)
skip_llm_call = budget_status == ContextBudgetStatus.EXCEEDED and not is_user_initiated
```

### Tool Restriction at Critical Budget (Partner Agent)

At CRITICAL and EXCEEDED thresholds, Partner agents have restricted tool access to preserve context headroom for wrap-up operations.

**Design Rationale:**
- External tools (web search, MCP servers) were used during planning phase
- Write-oriented Principal tools would expand context significantly  
- Only read-only tools should remain available to monitor status and prepare handoffs

**Implementation:** Tools are tagged with `allowed_at_critical=True` in their registry definition:

| Tool | `allowed_at_critical` | Reason |
|------|----------------------|--------|
| `GetPrincipalStatusSummaryTool` | ✅ True | Read-only status query |
| `LaunchPrincipalExecutionTool` | ❌ False | Creates new Principal context |
| `SendDirectiveToPrincipalTool` | ❌ False | Sends directives, expands context |
| MCP Server tools | ❌ False | External calls, potentially large results |

**Filtering Logic** (`agent_strategy_helpers.py`):
```python
def filter_tools_for_critical_budget(tools: List[Dict], agent_id: str) -> List[Dict]:
    return [t for t in tools if t.get("allowed_at_critical", False)]

def get_formatted_api_tools(agent_node_instance, context: Dict) -> List[Dict]:
    applicable_tools = get_tools_for_profile(...)
    budget_status = context.get("state", {}).get("_context_budget", {}).get("status", "HEALTHY")
    
    if budget_status in ("CRITICAL", "EXCEEDED"):
        applicable_tools = filter_tools_for_critical_budget(applicable_tools, agent_id)
    
    return format_tools_for_llm_api(applicable_tools)
```

This applies to **all agent types** (Principal, Partner, Associate) - any agent that receives a user-initiated message will allow it through.

### What Does NOT Exist

1. **Turn Summarization** - No automatic compression of old turns for Principal/Partner
2. **Sliding Window** - No rolling window that drops oldest content
3. **Progressive Compression** - No multi-tier summary hierarchy
4. **Multi-Tool Priority** - No priority handling when agents call multiple tools

## Implementation Status

| Fix | Status | Implementation |
|-----|--------|----------------|
| **Fix 1: Tool Priority** | ✅ IMPLEMENTED | `base_agent_node.py::_resolve_tool_conflicts()` |
| **Fix 2: Sync Completion** | ✅ IMPLEMENTED | `flow.py::_handle_principal_completion_sync()` |
| **Fix 3: Turn Summarization** | ⏸️ DEFERRED | User prefers phase-based handoffs (by design) |
| **Fix 4: User Prompt Bypass** | ✅ IMPLEMENTED | `base_agent_node.py`, `context_budget_guardian.py` |

**Tests**: `test_tool_conflict_resolution.py` (10 tests), `test_deliverable_propagation.py` (27 tests), `test_context_budget_guardian.py` (41 tests)


---

## Fix 1: Prevent Multi-Tool `finish_flow` Drops

### Root Cause
When Principal calls `finish_flow` alongside other tools in the same turn, the system processes only the first tool and drops the rest. Since `finish_flow` terminates the flow, it should have priority.

### Current Behavior (base_agent_node.py)
```python
# Line ~1034: Only first tool call is processed
tool_calls = response.tool_calls or []
if tool_calls:
    first_tool = tool_calls[0]  # Others are dropped
```

### Proposed Solution: Tool Priority Injection

#### Option A: Pre-Response Tool Filtering (Recommended)
**Philosophy**: Prevent the issue at the source by detecting conflicting tool combinations.

```python
# In base_agent_node.py, add to post_async after receiving tool calls

FLOW_TERMINATING_TOOLS = {"finish_flow", "generate_message_summary"}
TOOL_CONFLICT_RULES = {
    # If finish_flow is called with other tools, only keep finish_flow
    "finish_flow": {"priority": 100, "behavior": "exclusive"},
    "generate_message_summary": {"priority": 90, "behavior": "exclusive"},
}

def _resolve_tool_conflicts(self, tool_calls: List[Dict]) -> List[Dict]:
    """
    Resolve conflicting tool calls when agent calls multiple tools.
    
    Rules:
    1. If a flow-terminating tool is present, it takes priority
    2. Log a warning so we can improve prompts that cause multi-tool issues
    """
    if len(tool_calls) <= 1:
        return tool_calls
    
    tool_names = [tc.get("function", {}).get("name") for tc in tool_calls]
    terminating_tools = [t for t in tool_names if t in FLOW_TERMINATING_TOOLS]
    
    if terminating_tools:
        # Agent called flow-terminating tool + other tools
        priority_tool = max(terminating_tools, key=lambda t: TOOL_CONFLICT_RULES[t]["priority"])
        logger.warning("tool_conflict_resolved", extra={
            "original_tools": tool_names,
            "kept_tool": priority_tool,
            "dropped_tools": [t for t in tool_names if t != priority_tool]
        })
        # Return only the priority tool
        return [tc for tc in tool_calls if tc.get("function", {}).get("name") == priority_tool]
    
    return tool_calls  # No conflict
```

#### Option B: Deferred Execution Queue
**Philosophy**: Don't drop tools; execute them after flow-terminating tool completes.

```python
# Less recommended - complicates flow control significantly
# The flow is terminating, so other tools are irrelevant anyway
```

### Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| **Option A** | Simple, prevents data loss for critical tools | Drops non-critical tools silently (logged) |
| **Option B** | No tool loss | Complex, tools after termination are meaningless |

**Recommendation**: Option A with logging to identify prompts that cause multi-tool conflicts.

### Implementation (Completed)

**Changes**:
1. Added class constant `FLOW_TERMINATING_TOOLS = {"finish_flow", "generate_message_summary"}`
2. Added method `_resolve_tool_conflicts()` that prioritizes terminating tools
3. Hook inserted BEFORE existing multi-tool warning

**Key Design Decision**: Conflict resolution happens BEFORE the existing "first-only" truncation, so terminating tools are never dropped.

---

## Fix 2: Ensure Callback Runs Before State Save

### Root Cause
The `_principal_flow_done_callback` runs asynchronously when the Principal task completes. If the session is saved before the callback executes, Partner inbox remains empty.

### Current Flow (launch_principal_tool.py)
```python
# Line ~460: Task completion triggers callback
task.add_done_callback(
    lambda t: self._principal_flow_done_callback(t, principal_run_id, run_context_ref)
)
```

### Proposed Solution: Synchronous Callback with State Guard

#### Design A: Inline Completion Handler (Recommended)

Instead of relying on `add_done_callback`, handle completion inline in the flow:

```python
# In flow.py or launch_principal_tool.py

class PrincipalFlowManager:
    """Manages Principal flow lifecycle with guaranteed completion handling."""
    
    async def run_principal_with_completion_guard(
        self,
        run_context: Dict,
        principal_config: Dict
    ) -> Dict:
        """
        Run Principal flow with guaranteed completion handling.
        
        The completion handler runs BEFORE returning, ensuring:
        1. Deliverables are propagated to Partner inbox
        2. Session record is updated with deliverables
        3. State is consistent before any save operation
        """
        try:
            result = await self._execute_principal_flow(principal_config)
            
            # SYNCHRONOUS completion handling - not a callback
            await self._handle_principal_completion(
                result=result,
                run_context=run_context
            )
            
            return result
            
        except asyncio.CancelledError:
            await self._handle_principal_cancellation(run_context)
            raise
        except Exception as e:
            await self._handle_principal_error(e, run_context)
            raise
    
    async def _handle_principal_completion(
        self,
        result: Dict,
        run_context: Dict
    ):
        """
        Handle Principal completion synchronously before returning.
        
        Critical: This runs BEFORE the calling code can save state,
        ensuring deliverables are propagated.
        """
        partner_ctx = run_context['sub_context_refs'].get("_partner_context_ref")
        if not partner_ctx:
            logger.error("principal_completion_no_partner_context")
            return
        
        # 1. Extract deliverables
        deliverables = result.get("deliverables", {})
        final_report = deliverables.get("final_report")
        
        # 2. Save report to disk
        report_url = None
        if final_report:
            report_url = await self._save_report_to_disk(final_report, run_context)
        
        # 3. Update session record (single source of truth)
        sessions = run_context.get("team_state", {}).get("principal_execution_sessions", [])
        if sessions:
            current_session = sessions[-1]
            current_session["deliverables"] = deliverables
            current_session["report_url"] = report_url
        
        # 4. Add to Partner inbox (critical!)
        partner_state = partner_ctx.get("state", {})
        partner_state.setdefault("inbox", []).append({
            "item_id": f"inbox_{uuid.uuid4().hex[:8]}",
            "source": "PRINCIPAL_COMPLETED",
            "payload": {
                "status": result.get("status"),
                "summary": result.get("final_summary"),
                "has_final_report": bool(final_report),
                "report_url": report_url,
                "epoch_number": len(sessions),
            },
            "consumption_policy": "consume_on_read",
            "metadata": {"created_at": datetime.now(timezone.utc).isoformat()}
        })
        
        logger.info("principal_completion_handled_synchronously", extra={
            "has_deliverables": bool(deliverables),
            "report_url": report_url
        })
```

#### Design B: Two-Phase Commit Pattern

```python
# Add state guard that blocks save until callback completes

class StateGuard:
    """Ensures critical callbacks complete before state save."""
    
    def __init__(self):
        self._pending_callbacks: Dict[str, asyncio.Event] = {}
    
    async def register_pending_completion(self, operation_id: str):
        """Register a pending completion that must finish before save."""
        self._pending_callbacks[operation_id] = asyncio.Event()
    
    async def mark_completion_done(self, operation_id: str):
        """Mark a completion as done, allowing save to proceed."""
        if operation_id in self._pending_callbacks:
            self._pending_callbacks[operation_id].set()
            del self._pending_callbacks[operation_id]
    
    async def wait_for_all_completions(self, timeout: float = 30.0):
        """Block save until all pending completions are done."""
        if not self._pending_callbacks:
            return
        
        events = list(self._pending_callbacks.values())
        await asyncio.wait_for(
            asyncio.gather(*[e.wait() for e in events]),
            timeout=timeout
        )
```

### Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| **Design A** | Simple, deterministic, no race conditions | Requires refactoring callback to inline handler |
| **Design B** | Maintains async nature | More complex, still has timeout edge cases |

**Recommendation**: Design A - synchronous completion handling is the right pattern for critical state transitions.

### Implementation (Completed)

**Key Design Decisions**:
1. **Sync handler runs INSIDE try block**: Before `return result_package`, guaranteeing execution before `completion_event.set()`
2. **Callback becomes fallback**: Checks for existing `PRINCIPAL_COMPLETED` in inbox before adding (idempotent)
3. **Report saved to disk**: `/api/reports/{project_id}/{run_id}_epoch{N}.md` with security validation

---

## Fix 3: Context Summarization for Partner/Principal

### The Tradeoff: Fidelity vs. Efficiency

#### Why Automatic Turn Summarization is Risky

| Concern | Impact |
|---------|--------|
| **Loss of nuance** | LLM summaries lose specific details, quotes, error messages |
| **Broken references** | "As I mentioned earlier" points to content no longer in context |
| **Reasoning chain breaks** | Multi-step reasoning depends on intermediate conclusions |
| **Tool result loss** | Summarizing tool output loses structured data |

#### When Summarization Works Well

| Scenario | Why It Works |
|----------|--------------|
| **Cross-module inheritance** | Modules are self-contained; summary is for NEW agent |
| **Post-completion archiving** | Work is done; summary is for future reference |
| **Handback packages** | Agent exceeded budget; Principal will re-expand selectively |

### Proposed Approach: Tiered Context Management

Rather than auto-summarizing turns, implement a **tiered compression strategy** that preserves fidelity where it matters:

```python
# In context_budget_guardian.py

class TieredContextManager:
    """
    Manages context pressure through tiered compression.
    
    Tiers (in order of compression):
    1. Tool Results → Compress verbose tool outputs first
    2. Historical Turns → Summarize old turns (>N turns ago)
    3. KB Content → Collapse expanded KB tokens to references
    4. System Context → Last resort, reduce system prompt detail
    """
    
    TIER_THRESHOLDS = {
        "tool_compression": 0.50,    # Start at 50%
        "turn_summarization": 0.65,  # Start at 65%
        "kb_collapse": 0.75,         # Start at 75%
        "system_reduction": 0.85,    # Emergency at 85%
    }
    
    def assess_compression_needs(
        self,
        utilization: float,
        message_count: int,
        agent_type: str
    ) -> List[str]:
        """
        Determine which compression tiers should be active.
        
        Returns list of active compression strategies.
        """
        active = []
        
        if utilization >= self.TIER_THRESHOLDS["tool_compression"]:
            active.append("compress_tool_results")
        
        if utilization >= self.TIER_THRESHOLDS["turn_summarization"]:
            # Only for agents with many turns
            if message_count > 20:
                active.append("summarize_old_turns")
        
        if utilization >= self.TIER_THRESHOLDS["kb_collapse"]:
            active.append("collapse_kb_tokens")
        
        if utilization >= self.TIER_THRESHOLDS["system_reduction"]:
            active.append("reduce_system_detail")
        
        return active
```

#### Tier 1: Tool Result Compression (Safest)

```python
def compress_tool_results(self, messages: List[Dict], budget_chars: int) -> List[Dict]:
    """
    Compress verbose tool results while preserving structure.
    
    Strategy:
    - Keep tool name and status
    - Truncate large payloads with "...see KB token <#CGKB-XXXXX>"
    - Preserve error messages in full
    """
    compressed = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > 2000:
                # Large tool result - summarize
                compressed_content = self._summarize_tool_result(content, max_chars=800)
                compressed.append({**msg, "content": compressed_content})
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)
    return compressed
```

#### Tier 2: Turn Summarization (Moderate Risk)

```python
def summarize_old_turns(
    self,
    messages: List[Dict],
    preserve_last_n: int = 10
) -> List[Dict]:
    """
    Summarize older turns while keeping recent context intact.
    
    Strategy:
    - Keep last N turns verbatim (preserve immediate reasoning)
    - Summarize turns 0...(len-N) into a synthetic "context_summary" message
    - Preserve all user messages (don't lose their input)
    """
    if len(messages) <= preserve_last_n:
        return messages
    
    old_messages = messages[:-preserve_last_n]
    recent_messages = messages[-preserve_last_n:]
    
    # Generate summary of old turns
    summary = self._generate_turn_summary(old_messages)
    
    # Create synthetic summary message
    summary_msg = {
        "role": "system",
        "content": f"""
## Historical Context Summary

The following summarizes earlier turns in this conversation:

{summary}

---
*Recent conversation continues below*
""",
        "_synthetic": True,
        "_summarized_turn_count": len(old_messages)
    }
    
    return [summary_msg] + recent_messages
```

#### Your Strategy: Phase-Based Research with Deliverables

Your current approach is **superior to automatic summarization** because:

1. **Human-guided boundaries** - You decide where phases end, not arbitrary turn counts
2. **Explicit deliverables** - Each phase produces a structured artifact
3. **Fresh context** - New modules start clean, only inheriting relevant deliverables
4. **No cumulative drift** - Summaries of summaries lose fidelity; deliverables don't

```
┌─────────────────────────────────────────────────────────────────────┐
│  YOUR STRATEGY (Recommended)                                        │
│                                                                     │
│  Phase 1         Phase 2         Phase 3                            │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                        │
│  │ Research│ ──▶ │ Research│ ──▶ │ Research│                        │
│  │ Fresh   │     │ Fresh   │     │ Fresh   │                        │
│  │ Context │     │ Context │     │ Context │                        │
│  └────┬────┘     └────┬────┘     └────┬────┘                        │
│       │               │               │                             │
│       ▼               ▼               ▼                             │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                        │
│  │Deliverable│   │Deliverable│   │Deliverable│                      │
│  │ (Clean)  │───▶│ + Prior  │───▶│ + Prior  │                       │
│  └─────────┘     └─────────┘     └─────────┘                        │
│                                                                     │
│  ✅ Clean boundaries   ✅ Structured output   ✅ Full fidelity       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  AUTO-SUMMARIZATION (Riskier)                                       │
│                                                                     │
│  Turn 1 → Turn 2 → ... → Turn 20 → [SUMMARIZE] → Turn 21 → ...     │
│                                                                     │
│  ⚠️ Arbitrary boundaries   ⚠️ Loss of nuance   ⚠️ Drift over time   │
└─────────────────────────────────────────────────────────────────────┘
```

### Recommended Enhancement: Configurable Summarization Mode

Add an **opt-in** summarization feature with configurable aggressiveness:

```python
# In llm_configs/*.yaml or agent profile

context_management:
  # Automatic summarization mode
  auto_summarize: "off"  # "off" | "conservative" | "aggressive"
  
  # Conservative: Only compress tool results
  # Aggressive: Summarize old turns + compress tools
  
  # How many recent turns to preserve when summarizing
  preserve_recent_turns: 10
  
  # Minimum turns before considering summarization
  min_turns_before_summarize: 25
  
  # Store original turns in KB for potential re-expansion
  archive_summarized_turns: true
```

---

## Summary of Recommendations

| Fix | Complexity | Risk | Priority |
|-----|------------|------|----------|
| **Fix 1: Tool Priority** | Low | Low | 🔴 High - Prevents silent data loss |
| **Fix 2: Sync Completion** | Medium | Low | 🔴 High - Guarantees deliverable propagation |
| **Fix 3: Turn Summarization** | High | Medium | 🟡 Medium - Your phase strategy is better |

### Implementation Order

1. **Immediate**: Fix 1 + Fix 2 (prevent recurrence of observed issues)
2. **Optional**: Fix 3 with "off" default (opt-in for users who want it)

### Your Phase Strategy Validation

Your approach of:
- Breaking work into phases
- Getting deliverables per phase
- Feeding past deliverables to future modules
- Breaking into multiple runs

Is **architecturally sound** and **preferred over automatic summarization** because it:
- Maintains full fidelity within each phase
- Creates clean handoff boundaries
- Produces structured, reviewable artifacts
- Avoids cumulative summarization drift

The only gap your strategy doesn't address is **within-phase context exhaustion**, which is exactly what happened with Partner (202% utilization). The tiered compression in Fix 3 would help here, but only as a safety net—the real fix is the configurable phase boundaries you're already using.
