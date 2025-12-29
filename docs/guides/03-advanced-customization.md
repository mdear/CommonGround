# Advanced Customization

This guide covers more advanced customization topics for developers looking to extend the core capabilities of the framework.

## 1. Customizing Data Handover with Handover Protocols

#### 1.1 Purpose
When one agent needs to start another and pass a complex set of initial context (e.g., Partner launching Principal), the **Handover Protocol** provides a declarative, reusable way to define this data transfer. It decouples the data extraction logic from the tool's implementation, making the system cleaner and more maintainable.

#### 1.2 Location and Mechanism
*   **Location**: `agent_profiles/handover_protocols/`
*   **Mechanism**:
    1.  A tool (like `LaunchPrincipalExecutionTool`) declares its use of a protocol in its `@tool_registry` decorator: `handover_protocol="<protocol_name>"`.
    2.  When the tool is called, the `DispatcherNode` (or a similar flow controller) invokes the `HandoverService`.
    3.  The `HandoverService` reads the corresponding protocol YAML file and uses its rules to extract data from the source agent's context.
    4.  It packages this data into a standard `AGENT_STARTUP_BRIEFING` `InboxItem` and places it in the target agent's inbox, completing the handover.

#### 1.3 Core Configuration
*   **`context_parameters`**: Defines parameters that are expected to be provided **directly by the LLM** when it calls the tool. The schema defined here is automatically merged into the tool's `parameters`, so you don't need to define them in both places.
*   **`inheritance`**: The core of the protocol. It's a list of rules defining data to be inherited from the **source agent's full context**.
*   **`target_inbox_item`**: Defines the metadata (primarily the `source`) of the `InboxItem` that will be created.

**Example**: A snippet from `principal_to_associate_briefing.yaml`.
```yaml
inheritance:
  - from_source:
      # Use a template to specify which work module to get from team_state
      path: "team.work_modules.{{ module_id }}"
      replace:
        # Resolve the placeholder 'module_id' from the tool's input parameters
        module_id: "state.current_action.parameters.module_id_to_assign"
    # The extracted data will be stored under this key in the final payload
    as_payload_key: "module_details"
    # This rule only runs if the condition is true
    condition: "v['state.current_action.parameters.module_id_to_assign']"
```

#### 1.4 Managing Context Inheritance
To prevent context from growing indefinitely and consuming excessive tokens, the framework provides a mechanism to control what gets passed between agents. When inheriting message histories (e.g., `as_payload_key: "inherited_messages"`), the system will automatically filter out any messages that have an internal `_no_handover` flag. This flag is automatically added to messages that are part of an agent's initial briefing, ensuring that an agent doesn't pass its own startup instructions on to the next agent in the chain. This is a key feature for maintaining performance in long-running, multi-agent tasks.

#### 1.5 Budget-Aware Content Inheritance (NEW)

When using `inherit_messages_from` to pass context from completed work modules to new Associates, the system enforces **budget-aware content selection** to prevent context explosion.

**Problem Addressed**: Without limits, inheriting raw message history from multiple modules can cause new agents to be "born over-budget" (e.g., 86% of context used before doing any work).

**Two-Tier Selection Strategy**:

1.  **Tier 1 (Preferred)**: Use `deliverables.primary_summary` from the source module's archive
    - This is the LLM-generated summary created when the source module finished
    - Clean, compact, no Knowledge Base tokens to expand
    - Used if it exists AND fits within the computed budget

2.  **Tier 2 (Fallback)**: Select messages newest-to-oldest from raw history
    - Used when no summary exists or summary exceeds budget
    - Messages are **hydrated first** to get accurate size (KB tokens expanded)
    - Selection stops when budget is exhausted
    - Individual messages are never truncated

**Budget Computation**:
```
per_source_budget = (target_context_limit * 0.40) / num_sources
```
- `target_context_limit`: The spawning agent's context window (e.g., 200K tokens)
- `0.40`: 40% reserved for inherited content (constant: `INHERITANCE_BUDGET_FRACTION`)
- `num_sources`: Number of modules in `inherit_messages_from`

**Example**:
```yaml
# In dispatch_submodules call
assignments:
  - module_id_to_assign: "WM_6"
    inherit_messages_from: ["WM_2", "WM_3"]  # Two sources
    # Budget: (200K * 0.40) / 2 = 40K tokens = 160K chars per source
```

**Implementation**: The content selection happens in `dispatcher_node._preselect_inherited_content()` BEFORE the HandoverService is called, ensuring budget compliance at dispatch time.

## 2. Optimizing External Tools with MCP Prompt Overrides

#### 2.1 Purpose
Sometimes, the description for a tool discovered from an external MCP server is too generic or not well-suited for a specific task. This mechanism allows you to override the tool's description without modifying the external service.

#### 2.2 Location and Mechanism
*   **Location**: `mcp_prompt_override.yaml` in the project root.
*   **Mechanism**:
    1.  During startup, after the `ToolRegistry` discovers all internal and external tools, it checks for `mcp_prompt_override.yaml`.
    2.  If the file exists, it reads the key-value pairs.
    3.  For each key (which must be the unique name of an MCP tool, e.g., `G.google_web_search`), it replaces the tool's registered `description` with the new string value.
    4.  Agents will then see and use this improved description when constructing their prompts.

*   **Format**: A simple key-value mapping.
    ```yaml
    # mcp_prompt_override.yaml

    "G.google_web_search": "Performs a precise academic search on Google. Prioritizes academic databases and well-known journals. Query format: 'keyword site:scholar.google.com'"

    "G.web_fetch": "After fetching a web page, extracts the core arguments and data points. Ignores advertisements and navigation links."
    ```

## 3. Developing Custom Observers and Ingestors

### 3.1 Custom Observers (`agent_core/events/observers.py`)
If the declarative observers in YAML are not sufficient for your logic, you can create a custom Python-based observer.

*   **Use Case**: For complex, stateful observation logic.
*   **Registration**: Use the `@register_observer("my_observer_name")` decorator.
*   **Structure**: Inherit from the `Observer` base class and implement the `async def observe(self, context: Dict) -> None:` method. Inside this method, you can perform any logic and add `InboxItem`s to `context['state']['inbox']`.

### 3.2 Custom Ingestors (`agent_core/events/ingestors.py`)
An Ingestor is a pure function that transforms an `InboxItem`'s `payload` into a string for the LLM.

*   **Use Case**: To define how a new type of `InboxItem` should be presented to the LLM.
*   **Registration**: Use the `@register_ingestor("my_ingestor_name")` decorator.
*   **Function Signature**: `def my_ingestor(payload: Any, params: Dict, context: Dict) -> str:`. It must be a pure function with no side effects.

## 4. Custom LLM Configurations
LLM configurations are managed in `agent_profiles/llm_configs/`. The system supports inheritance and environment variable overrides, providing a flexible way to manage different models and parameters for various agents or tasks. You can add new YAML files here to define new LLM configurations.
