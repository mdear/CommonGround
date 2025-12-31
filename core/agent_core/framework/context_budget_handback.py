"""
Context Budget Handback - Data structures for Principal-delegated summarization.

When a subagent exceeds its context budget, instead of losing all research,
this module packages the collected work for the Principal to summarize.

The Principal typically has a larger context allocation and can:
1. Expand KB tokens to retrieve the research content
2. Summarize the partial findings itself
3. Decide whether to retry, accept partial, or mark incomplete
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextBudgetHandback:
    """
    Package returned to Principal when subagent exceeds context budget.
    Contains all information needed for Principal to summarize partial work.
    """
    # Identification
    agent_id: str
    module_id: str
    profile_name: str
    
    # Budget state at trigger
    utilization_percent: float
    predicted_tokens: int
    context_limit: int
    
    # Collected research artifacts
    kb_tokens: List[str] = field(default_factory=list)  # ["<#CGKB-00029>", ...]
    kb_token_count: int = 0
    estimated_kb_content_tokens: int = 0
    
    # Tool execution history
    tool_calls_completed: List[Dict] = field(default_factory=list)
    tool_calls_in_progress: Optional[Dict] = None
    
    # Partial content
    last_assistant_content_preview: str = ""
    turns_completed: int = 0
    
    # Timestamps
    start_timestamp: str = ""
    overflow_timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ContextBudgetHandback":
        """Create from dictionary."""
        return cls(**data)
    
    def get_principal_summary_prompt(self) -> str:
        """Generate a prompt for Principal to summarize the partial work."""
        kb_tokens_display = ', '.join(self.kb_tokens[:20])
        if len(self.kb_tokens) > 20:
            kb_tokens_display += f"... (+{len(self.kb_tokens) - 20} more)"
        
        tool_history = ""
        if self.tool_calls_completed:
            tool_lines = []
            for i, tc in enumerate(self.tool_calls_completed[:10], 1):
                tool_name = tc.get("tool", "unknown")
                args_preview = tc.get("arguments_preview", "")[:100]
                tool_lines.append(f"  {i}. `{tool_name}`: {args_preview}")
            tool_history = "\n".join(tool_lines)
            if len(self.tool_calls_completed) > 10:
                tool_history += f"\n  ... (+{len(self.tool_calls_completed) - 10} more)"
        else:
            tool_history = "  (No tool calls completed before overflow)"
        
        return f"""
## CONTEXT BUDGET HANDBACK - Summarization Required

Agent `{self.agent_id}` (profile: `{self.profile_name}`) exceeded its context budget 
while working on module `{self.module_id}`.

### Budget Status at Overflow
- **Utilization**: {self.utilization_percent:.1f}%
- **Tokens used**: {self.predicted_tokens:,} / {self.context_limit:,}
- **Turns completed**: {self.turns_completed}

### Work Completed Before Overflow
- **Tool calls executed**: {len(self.tool_calls_completed)}
- **Knowledge base items collected**: {self.kb_token_count}
- **Estimated content tokens**: ~{self.estimated_kb_content_tokens:,}

### Tool Call History
{tool_history}

### Collected KB Tokens
The following KB tokens contain the research data collected before overflow:
```
{kb_tokens_display}
```

### Last Agent Output (Preview)
{self.last_assistant_content_preview if self.last_assistant_content_preview else "(No output captured)"}

### Your Task as Principal
The subagent's context was too full to self-summarize. You should:

1. **Expand KB tokens** to retrieve the collected research content
   - Use batches that fit your context budget
   - Start with the first few tokens to assess content quality

2. **Summarize key findings** from the expanded content
   - Focus on findings relevant to module objective
   - Note any incomplete areas

3. **Decide next steps**:
   - **Accept partial**: If findings are sufficient, mark module complete
   - **Retry narrower**: Dispatch new agent with more focused scope
   - **Mark incomplete**: Proceed with other modules, note gap

**Recommended action**: Start by expanding 5-10 KB tokens to assess the research quality.
"""

    def get_deliverables_summary(self) -> str:
        """Generate a summary for the deliverables field."""
        return (
            f"[CONTEXT BUDGET EXCEEDED] Agent `{self.agent_id}` collected "
            f"{self.kb_token_count} KB items before exceeding budget at "
            f"{self.utilization_percent:.1f}% utilization. "
            f"KB tokens available for Principal summarization: "
            f"{', '.join(self.kb_tokens[:5])}{'...' if len(self.kb_tokens) > 5 else ''}"
        )


def build_handback_from_context(
    agent_id: str,
    context: Dict,
    prep_res: Dict,
    profile_name: str = "unknown"
) -> ContextBudgetHandback:
    """
    Build a handback package from agent context when circuit breaker fires.
    
    Args:
        agent_id: The agent's identifier
        context: The full agent context
        prep_res: The prep_async result containing budget info
        profile_name: The agent's profile name
    
    Returns:
        ContextBudgetHandback with all recoverable work packaged
    """
    state = context.get("state", {})
    meta = context.get("meta", {})
    budget_info = state.get("_context_budget", {})
    
    # Extract KB tokens from this agent's session
    kb = context.get('refs', {}).get('run', {}).get('runtime', {}).get("knowledge_base")
    agent_kb_tokens = []
    estimated_kb_tokens = 0
    
    if kb:
        # Get all KB items added by this agent
        items_dict = getattr(kb, 'items', {})
        if callable(items_dict):
            items_dict = {}  # Fallback if items is a method
        
        for item_id, item in items_dict.items():
            item_metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            if item_metadata.get("source_agent_id") == agent_id:
                token = item.get("token", item_id) if isinstance(item, dict) else item_id
                agent_kb_tokens.append(token)
                # Rough estimate: 4 chars per token
                content = item.get("content", "") if isinstance(item, dict) else ""
                estimated_kb_tokens += len(content) // 4
    
    # Extract tool call history from messages
    tool_calls_completed = []
    messages = state.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                tool_calls_completed.append({
                    "tool": func.get("name"),
                    "arguments_preview": str(func.get("arguments", ""))[:200],
                    "tool_call_id": tc.get("id", "")
                })
    
    # Get last assistant content
    last_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"]
            if isinstance(content, str):
                last_content = content[:500]
            break
    
    # Get module ID from various sources
    module_id = (
        meta.get("module_id") or 
        state.get("initial_parameters", {}).get("module_id") or
        meta.get("agent_id", "unknown")
    )
    
    handback = ContextBudgetHandback(
        agent_id=agent_id,
        module_id=module_id,
        profile_name=profile_name,
        utilization_percent=budget_info.get("utilization_percent", 0),
        predicted_tokens=prep_res.get("predicted_total_tokens", 0),
        context_limit=budget_info.get("context_limit", 200000),
        kb_tokens=agent_kb_tokens,
        kb_token_count=len(agent_kb_tokens),
        estimated_kb_content_tokens=estimated_kb_tokens,
        tool_calls_completed=tool_calls_completed,
        tool_calls_in_progress=state.get("current_action"),
        last_assistant_content_preview=last_content,
        turns_completed=len([m for m in messages if m.get("role") == "assistant"]),
        start_timestamp=meta.get("start_timestamp", ""),
        overflow_timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    logger.info("context_budget_handback_built", extra={
        "agent_id": agent_id,
        "module_id": module_id,
        "kb_tokens_collected": len(agent_kb_tokens),
        "tool_calls_completed": len(tool_calls_completed),
        "utilization_percent": budget_info.get("utilization_percent", 0)
    })
    
    return handback


def notify_principal_of_handback(principal_context: Dict, handback: ContextBudgetHandback) -> None:
    """
    Inject handback notification into Principal's inbox.
    
    Args:
        principal_context: The Principal agent's context
        handback: The handback package from the overflowed subagent
    """
    import uuid
    
    notification = {
        "item_id": f"handback_{handback.agent_id}_{uuid.uuid4().hex[:8]}",
        "source": "CONTEXT_BUDGET_HANDBACK",
        "payload": {
            "content_key": "principal_handback_notification",
            "handback": handback.to_dict(),
            "message": f"""
## 🔄 Context Budget Handback from `{handback.agent_id}`

Module `{handback.module_id}` exceeded context budget at **{handback.utilization_percent:.1f}%** utilization.

**Research Collected (Available for Your Summarization):**
- **{handback.kb_token_count}** knowledge base items
- **{len(handback.tool_calls_completed)}** tool calls completed
- Estimated **~{handback.estimated_kb_content_tokens:,}** tokens of content

**KB Tokens to Expand:**
```
{', '.join(handback.kb_tokens[:10])}{'...' if len(handback.kb_tokens) > 10 else ''}
```

**Recommended Action:**
Expand the KB tokens above to retrieve the collected research, then summarize the key findings yourself. 
This agent's context was too full to self-summarize.
"""
        },
        "consumption_policy": "preserve",
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "priority": "high",
            "requires_action": True,
            "handback_agent_id": handback.agent_id,
            "handback_module_id": handback.module_id
        }
    }
    
    principal_context["state"].setdefault("inbox", []).append(notification)
    
    logger.info("principal_handback_notification_sent", extra={
        "agent_id": handback.agent_id,
        "module_id": handback.module_id,
        "kb_tokens": handback.kb_token_count
    })
