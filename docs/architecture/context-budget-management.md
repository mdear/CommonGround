# Context Budget Management Architecture

## Problem Statement

The CommonGround multi-agent system experiences context window explosions that trigger circuit breakers, resulting in incomplete or failed responses. Analysis of session `orange-seagull-of-teaching` revealed:

- **Principal Token Count**: 211,901 tokens (106% of 200K limit)
- **Partner Token Count**: 241,967 tokens (121% of 200K limit)
- **Root Cause**: Full context archives (~121K chars) injected via `work_modules_ingestor`
- **Secondary Issue**: Tool definitions duplicated in messages (~40-50K tokens)

## Design Principles

1. **Information Flows Up as Summaries**: Subagent work products flow up to Principal/Partner as compressed deliverables, not full context
2. **Detail Preserved at Source**: Full context archives remain available for drill-down but are never injected by default
3. **Budget Allocation is Hierarchical**: Each level reserves capacity for summarization before delegating
4. **Graceful Degradation**: When limits approach, synthesize partial results rather than fail completely
5. **1M Context as Safety Net**: Extended context (1M tokens) provides headroom, not permission to be wasteful

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CONTEXT BUDGET HIERARCHY                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         PARTNER (1M tokens)                       │   │
│  │  ├─ System Prompt: ~20K                                           │   │
│  │  ├─ Tool Definitions: ~50K                                        │   │
│  │  ├─ Summarization Reserve: 300K (30%)                             │   │
│  │  └─ Working Budget: 630K                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        PRINCIPAL (1M tokens)                      │   │
│  │  ├─ System Prompt: ~20K                                           │   │
│  │  ├─ Tool Definitions: ~50K                                        │   │
│  │  ├─ Work Module Summaries: variable (target <50K)                 │   │
│  │  ├─ Summarization Reserve: 300K (30%)                             │   │
│  │  └─ Working Budget: 580K                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                    ┌───────────────┼───────────────┐                     │
│                    ▼               ▼               ▼                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                         │
│  │  Associate  │ │  Associate  │ │  Associate  │  (200K each default)   │
│  │  Module A   │ │  Module B   │ │  Module C   │                         │
│  │  ├─ Budget  │ │  ├─ Budget  │ │  ├─ Budget  │  Per-worker: 119K      │
│  │  └─ Reserve │ │  └─ Reserve │ │  └─ Reserve │                         │
│  └─────────────┘ └─────────────┘ └─────────────┘                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Fixes

### 1. Work Modules Ingestor (CRITICAL)

**Problem**: Injects full `context_archive` when formatting work modules status.

**Solution**: Filter out `context_archive` and large fields, show only summary metadata.

**File**: `core/agent_core/events/ingestors.py`

```python
@register_ingestor("work_modules_ingestor")
def work_modules_ingestor(payload: Any, params: Dict, context: Dict) -> str:
    """
    Formats work_modules dictionary as Markdown.

    IMPORTANT: Only includes summary fields. Full context_archive is stored
    but never injected into prompts. Use dispatch_result_ingestor for deliverables.
    """
    if not isinstance(payload, dict):
        return "Work modules data is not in the expected format (dictionary)."

    # Fields to EXCLUDE from injection (available for drill-down only)
    EXCLUDED_FIELDS = {
        'context_archive',  # Full message history - too large
        'full_context',     # Any full context dump
        'raw_messages',     # Raw message lists
        'messages',         # Message arrays
    }

    # Fields to SUMMARIZE (show counts/metadata only)
    SUMMARIZE_FIELDS = {
        'deliverables': lambda v: f"[{len(v) if isinstance(v, (list, dict)) else 0} items]",
        'tools_used': lambda v: f"[{len(v) if isinstance(v, list) else 0} tools]",
    }

    def _filter_module(module_data: dict) -> dict:
        """Filter a single work module to exclude large fields."""
        filtered = {}
        for key, value in module_data.items():
            if key in EXCLUDED_FIELDS:
                continue
            if key in SUMMARIZE_FIELDS:
                filtered[key] = SUMMARIZE_FIELDS[key](value)
            else:
                filtered[key] = value
        return filtered

    lines = [params.get("title", "### Current Work Modules Status")]
    if not payload:
        lines.append("No work modules are currently defined.")
    else:
        # Filter each module before formatting
        filtered_modules = {
            mod_id: _filter_module(mod_data) if isinstance(mod_data, dict) else mod_data
            for mod_id, mod_data in payload.items()
        }
        formatted_modules = _recursive_markdown_formatter(filtered_modules, {}, level=0)
        lines.extend(formatted_modules)

    return "\n".join(lines)
```

### 2. Context Budget Guardian Updates

**File**: `core/agent_core/framework/context_budget_guardian.py`

The guardian already supports 1M context detection. Key behaviors:

1. **Priority Resolution**:
   - First: Check explicit `max_context_tokens` in config
   - Second: Check `extra_headers` for `anthropic-beta: context-1m-*`
   - Third: Model family defaults
   - Fourth: Conservative 100K default

2. **Thresholds** (already implemented):
   - WARNING: 40% - Start suggesting wrap-up
   - CRITICAL: 55% - Force completion
   - EXCEEDED: 70% - Circuit breaker, 30% remains for wrap-up

### 2.1 Provider-Aware Token Counting

**File**: `core/agent_core/llm/token_counter.py`

**Problem**: LiteLLM's `token_counter()` uses tiktoken (OpenAI's tokenizer) as fallback for Claude 3+ models, which underestimates token counts by 25-35%. This caused the Context Budget Guardian to believe it had more headroom than actually available, leading to context window overflows.

**Solution**: Provider-aware token counting that uses official provider APIs when available:

- **Anthropic (Claude)**: Uses `client.messages.count_tokens()` - free, accurate, separate rate limits
- **OpenAI (GPT)**: Uses litellm/tiktoken (accurate for OpenAI models)
- **Google (Gemini)**: Falls back to litellm (future: use countTokens API)
- **Unknown providers**: Falls back to litellm estimation

**Architecture**:
```python
# Provider detection via model name patterns
LLMProvider = Enum("LLMProvider", ["ANTHROPIC", "OPENAI", "GOOGLE", "UNKNOWN"])

# Registry of provider-specific counters
PROVIDER_TOKEN_COUNTERS = {
    LLMProvider.ANTHROPIC: _count_tokens_anthropic,  # Uses official API
    LLMProvider.OPENAI: _count_tokens_openai,        # Returns None (litellm accurate)
    LLMProvider.GOOGLE: _count_tokens_google,        # Placeholder for future
}
```

**Integration**: `estimate_prompt_tokens()` in `call_llm.py` delegates to this module, maintaining backward compatibility while providing accurate counts for Claude models.

### 3. Circuit Breaker Graceful Degradation

**Problem**: When circuit breaker fires, it returns nothing useful.

**Solution**: Synthesize partial results from available work.

**File**: `core/agent_core/framework/context_budget_guardian.py` (new function)

```python
def synthesize_partial_results(
    team_state: Dict,
    triggered_agent: str,
    budget_metadata: Dict
) -> Dict:
    """
    When circuit breaker fires, compile available work into actionable summary.

    Returns a structured report of:
    - What was requested
    - What was completed
    - What remains incomplete
    - Key findings so far
    """
    work_modules = team_state.get("work_modules", {})

    completed_modules = []
    incomplete_modules = []
    partial_deliverables = []

    for module_id, module in work_modules.items():
        status = module.get("status", "unknown")
        if status in ("completed", "done"):
            completed_modules.append({
                "id": module_id,
                "objective": module.get("objective", "Unknown"),
                "deliverables": module.get("deliverables", {})
            })
            if module.get("deliverables"):
                partial_deliverables.extend(
                    _extract_key_findings(module["deliverables"])
                )
        else:
            incomplete_modules.append({
                "id": module_id,
                "objective": module.get("objective", "Unknown"),
                "status": status,
                "assigned_to": module.get("assigned_agent_id")
            })

    return {
        "circuit_breaker_synthesis": True,
        "triggered_by": triggered_agent,
        "budget_at_trigger": budget_metadata,
        "summary": {
            "total_modules": len(work_modules),
            "completed": len(completed_modules),
            "incomplete": len(incomplete_modules)
        },
        "completed_work": completed_modules,
        "incomplete_work": incomplete_modules,
        "key_findings_so_far": partial_deliverables,
        "recommendation": (
            f"Context budget exceeded ({budget_metadata.get('utilization_percent', 0):.1f}% used). "
            f"Returning {len(completed_modules)} completed modules. "
            f"{len(incomplete_modules)} modules remain incomplete and can be resumed."
        )
    }
```

### 4. Deliverable Capture Enforcement

**Problem**: Subagents completing with 0 deliverables.

**Solution**: Enforce deliverable registration through FIM prompts.

**File**: `core/agent_profiles/profiles/Associate_*_EN.yaml` (all associate profiles)

Add to `fim_protocol`:

```yaml
fim_protocol:
  trigger_conditions:
    budget_threshold_percent: 70
    max_turns_without_fim: 5

  mandatory_deliverable_check:
    enabled: true
    on_completion: |
      Before signaling completion, verify:
      1. You have registered at least one deliverable using `register_deliverable`
      2. The deliverable contains actionable findings, not just "task attempted"
      3. If you cannot produce a meaningful deliverable, explain why in the deliverable
```

### 5. LLM Config Update (COMPLETED)

**File**: `core/agent_profiles/llm_configs/principal_llm.yaml`

```yaml
config:
  api_key:
    _type: "from_env"
    var: "ANTHROPIC_API_KEY"
    required: true
  model: "anthropic/claude-sonnet-4-5-20250929"
  temperature:
    _type: "from_env"
    var: "PRINCIPAL_TEMPERATURE"
    required: false
    default: 0.4
  extra_headers:
    _type: "from_env"
    var: "ANTHROPIC_EXTRA_HEADERS"
    required: false
    default: null
  max_context_tokens:
    _type: "from_env"
    var: "PRINCIPAL_MAX_CONTEXT_TOKENS"
    required: false
    default: 1000000  # 1M default for models that support it
```

## Environment Configuration

**File**: `core/.env`

```bash
# 1M Context Configuration
ANTHROPIC_EXTRA_HEADERS={"anthropic-beta": "context-1m-2025-08-07"}

# Optional: Override default max context tokens
# PRINCIPAL_MAX_CONTEXT_TOKENS=1000000
```

## Token Budget Guidelines

### For Principal Agent

| Component | Target Budget | Notes |
|-----------|--------------|-------|
| System Prompt | 15-20K | Core instructions, role definition |
| Tool Definitions | 40-50K | Can be reduced by trimming docstrings |
| Work Module Summaries | 30-50K | Filtered view, no context_archive |
| Active Conversation | 200-400K | Working memory |
| Summarization Reserve | 300K | For final synthesis |
| **Total Available** | **1M** | With 1M context enabled |

### For Associate Agents

| Component | Target Budget | Notes |
|-----------|--------------|-------|
| System Prompt | 10-15K | Role + briefing |
| Tool Definitions | 20-30K | Subset of Principal's tools |
| Work Context | 50-100K | Task-specific context |
| Working Memory | 50-100K | Conversation history |
| **Total Available** | **200K** | Standard context |

## Implementation Priority

1. **[CRITICAL] Fix work_modules_ingestor** - Immediate token reduction
2. **[HIGH] Verify 1M context** - Safety net (DONE ✓)
3. **[HIGH] Graceful circuit breaker** - Better failure recovery
4. **[MEDIUM] Deliverable enforcement** - Data quality
5. **[LOW] Tool docstring optimization** - Long-term token savings

## Monitoring & Alerts

The system should log warnings when:

1. Single work module exceeds 10K tokens when serialized
2. Total work_modules payload exceeds 50K tokens
3. Context utilization exceeds WARNING threshold (40%)
4. Any agent completes with 0 deliverables
5. Circuit breaker fires (should be exceptional, not routine)

## Testing Checklist

- [ ] Verify 1M context enabled: `extra_headers` resolved correctly
- [ ] Verify `max_context_tokens: 1000000` in resolved config
- [ ] Verify work_modules_ingestor excludes context_archive
- [ ] Test circuit breaker synthesizes partial results
- [ ] Confirm deliverable capture enforcement works
- [ ] Load test with complex multi-module scenario
- [ ] Verify inheritance budget computation works correctly
- [ ] Test content selection uses LLM summary when available
- [ ] Confirm hydration occurs before selection (not after)

---

## 6. Content Inheritance Budget Management

### 6.1 Problem Statement

When Associates are spawned with `inherit_messages_from` parameter, unbounded raw message history can be injected into the new agent's briefing, causing context explosion at birth.

**Observed Failure (Session: `independent-saffron-kittiwake`)**:
- E_1 (no inheritance): Born at 983 tokens (0.5% budget)
- E_6 (inherits from WM_2, WM_3): Born at 171,947 tokens (86% budget)
- Root Cause: 60KB of raw messages inherited without budget limits
- Result: Infinite loop as circuit breaker triggered immediately

### 6.2 Architecture: Content Inheritance Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTENT INHERITANCE DATA FLOW (FIXED)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Source Module Finishes - e.g., WM_2]                                      │
│       │                                                                     │
│       │ finish_flow() or generate_message_summary() called                  │
│       ▼                                                                     │
│  context_archive[-1] = {                                                    │
│    "messages": [...raw history with KB tokens...],                          │
│    "deliverables": {"primary_summary": "...LLM-generated summary..."}       │
│  }                                                                          │
│                                                                             │
│  [Principal dispatches E_6 with inherit_messages_from: [WM_2, WM_3]]        │
│       │                                                                     │
│       ▼                                                                     │
│  dispatcher_node._preselect_inherited_content()  ◄── NEW                    │
│       │                                                                     │
│       │ 1. Compute target's context limit (e.g., 200K tokens)               │
│       │ 2. Reserve INHERITANCE_BUDGET_FRACTION (40%) for inheritance        │
│       │ 3. Divide pool among sources: 200K * 0.40 / 2 = 40K tokens each     │
│       │ 4. Convert to chars: 40K * 4 = 160K chars per source                │
│       │                                                                     │
│       │ For each source module:                                             │
│       │   ┌─────────────────────────────────────────────────────────┐       │
│       │   │ TIER 1: Try deliverables.primary_summary                │       │
│       │   │   - If exists AND fits budget → USE IT (no KB tokens!)  │       │
│       │   │   - Skip to next source                                 │       │
│       │   ├─────────────────────────────────────────────────────────┤       │
│       │   │ TIER 2: Fall back to messages                           │       │
│       │   │   - HYDRATE messages first (expand KB tokens)           │       │
│       │   │   - Select newest-first until budget exhausted          │       │
│       │   │   - Never truncate individual messages                  │       │
│       │   └─────────────────────────────────────────────────────────┘       │
│       │                                                                     │
│       ▼                                                                     │
│  Pre-selected content stored in assignment parameters                       │
│       │                                                                     │
│       ▼                                                                     │
│  HandoverService.execute() uses pre-selected content (via condition)        │
│       │                                                                     │
│       ▼                                                                     │
│  E_6 InboxProcessor renders budget-limited inherited content                │
│       │                                                                     │
│       ▼                                                                     │
│  E_6 born within budget ✓                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Key Design Decisions

#### 6.3.1 Two-Tier Content Selection Strategy

| Tier | Content Source | When Used | Pros | Cons |
|------|---------------|-----------|------|------|
| **1** | `deliverables.primary_summary` | When exists and fits budget | Clean, no KB tokens, LLM-quality summary | May lose granular detail |
| **2** | Raw messages (newest-first) | When no summary or summary too large | Preserves recent context | Requires hydration first |

#### 6.3.2 Hydration Before Selection (Critical)

Messages in `context_archive` may contain Knowledge Base tokens (e.g., `<#CGKB-00042>`). These tokens are compact placeholders that expand when hydrated.

**Problem**: If we select messages by dehydrated size, then hydration happens later (in `base_agent_node._hydrate_messages()`), the final size could exceed budget.

**Solution**: When using Tier 2 (message selection), we MUST:
1. Hydrate messages from KB **before** measuring size
2. Select from hydrated messages to get accurate sizing
3. Selected content is already hydrated → no double-expansion

#### 6.3.3 Budget Computation Formula

```python
# Constants (elevated to module level in content_selection.py)
CHARS_PER_TOKEN = 4
INHERITANCE_BUDGET_FRACTION = 0.40  # 40% of target's context

# Computation
target_context_tokens = get_model_context_limit(target_model, llm_config)
pool_tokens = target_context_tokens * INHERITANCE_BUDGET_FRACTION
pool_chars = pool_tokens * CHARS_PER_TOKEN
per_source_budget_chars = pool_chars // num_sources
```

**Example**:
- Target model: claude-sonnet-4 (200K tokens)
- Inheritance pool: 200K * 0.40 = 80K tokens
- Sources: 2 modules (WM_2, WM_3)
- Per-source: 80K / 2 = 40K tokens = 160K chars each

### 6.4 Implementation Files

| File | Change Type | Purpose |
|------|-------------|---------|
| `agent_core/utils/content_selection.py` | **CREATE** | Shared utilities for budget-aware content selection |
| `agent_core/nodes/custom_nodes/dispatcher_node.py` | **MODIFY** | Add `_preselect_inherited_content()` method |
| `agent_profiles/handover_protocols/principal_to_associate_briefing.yaml` | **MODIFY** | Add conditional rule for pre-selected content |

### 6.5 Content Selection Module API

**File**: `agent_core/utils/content_selection.py`

```python
# Constants
CHARS_PER_TOKEN = 4
INHERITANCE_BUDGET_FRACTION = 0.40
STRATEGY_LLM_SUMMARY = "llm_summary"
STRATEGY_NEWEST_FIRST = "newest_first"

# Core Functions
def compute_inheritance_budget_chars(
    target_context_limit_tokens: int,
    num_sources: int,
    inheritance_fraction: float = INHERITANCE_BUDGET_FRACTION
) -> int:
    """Compute per-source character budget for content inheritance."""

def select_content_within_budget(
    deliverables: Optional[Dict],
    messages: List[Dict],
    budget_chars: int,
    source_id: str = "unknown"
) -> Tuple[Union[str, List[Dict]], Dict[str, Any]]:
    """
    Two-tier content selection within a character budget.

    Returns:
        Tuple of (selected_content, metadata)
        - selected_content: Either summary string OR list of selected messages
        - metadata: Dict with "strategy", "chars_used", "items_selected", etc.
    """

def format_inherited_content_for_briefing(
    content: Union[str, List[Dict]],
    metadata: Dict[str, Any],
    source_id: str
) -> List[Dict]:
    """Format selected content as messages for injection into briefing."""
```

### 6.6 Backward Compatibility

The design maintains full backward compatibility:

1. **No inheritance**: If `inherit_messages_from` is empty/absent, no changes to existing flow
2. **Protocol conditions**: New YAML rules use conditions to prefer pre-selected content when available
3. **Fallback path**: If pre-selection fails, original message iteration still works (with existing size issues)

### 6.7 Monitoring & Logging

New log events for observability:

| Log Event | Level | When |
|-----------|-------|------|
| `inheritance_budget_computed` | DEBUG | Budget calculation completed |
| `inheritance_content_selection_started` | INFO | Starting content selection |
| `content_selection_using_summary` | DEBUG | Tier 1 path taken |
| `content_selection_summary_exceeds_budget` | DEBUG | Falling back to Tier 2 |
| `content_selection_newest_first_complete` | DEBUG | Tier 2 selection done |
| `inheritance_content_selected` | INFO | Per-source selection result |
