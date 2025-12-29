# Customizing Agent Behavior: A Deep Dive into Agent Profiles

This guide provides a comprehensive overview of how to customize agent behavior using the declarative `Agent Profile` YAML files. This is the primary method for shaping how agents think, decide, and act.

## 1. Agent Profile Core Structure

Agent Profiles are located in the `agent_profiles/profiles/` directory. Each `.yaml` file defines a reusable template for an agent's behavior.

*   `name` (string): The Profile's **logical name** (e.g., `Principal`, `Associate_WebSearcher`).
*   `type` (string): The Profile's role type, such as `principal`, `associate`, or `partner`.
*   `llm_config_ref` (string): A reference to the logical name of a shared LLM configuration.
*   `pre_turn_observers` / `post_turn_observers` (list): The **core of reactive behavior**, defining how the agent responds to state changes.
*   `flow_decider` (list): The **core of decision logic**, defining what the agent does when it doesn't call a tool.
*   `system_prompt_construction` (object): Defines how to build the LLM's system prompt from various components.
*   `tool_access_policy` (object): Defines which tools the agent is permitted to use.
*   `text_definitions` (object): A dictionary for storing all static text snippets used in the profile, promoting reusability.

## 2. The Core of Reactive Behavior: Observers

`Observers` are the cornerstone of the framework's event-driven architecture. They allow an agent to react to changes in its own state in a declarative, predictable manner.

*   **`pre_turn_observers`**: Execute **before** the agent's "thinking" phase (the LLM call). They are typically used for setup tasks, such as checking for initial parameters or performing state self-healing.
*   **`post_turn_observers`**: Execute **after** the LLM has responded. They are used for reactive logic, such as prompting the agent for self-reflection if it fails to call a tool, or notifying it when a long-running task is complete.

Each `observer` object contains the following fields:
*   `id` (string): A unique identifier for the observer.
*   `type` (string): Must be `declarative`.
*   `condition` (string): A Python expression that is evaluated against the agent's context. The observer only runs if this expression returns `True`. You can safely use the `v['path.to.value']` syntax to access any part of the agent's state.
*   `action` (object): The action to perform if the condition is met. The most common action is `add_to_inbox`, which creates a new `InboxItem` event.

**Example**: A `post_turn_observer` that triggers when an agent doesn't call a tool.
```yaml
post_turn_observers:
  - id: "observer_on_no_tool_call"
    type: "declarative"
    condition: "not v['state.current_action']" # Checks if current_action is empty
    action:
      type: "add_to_inbox"
      target_agent_id: "self"
      inbox_item:
        source: "SELF_REFLECTION_PROMPT"
        consumption_policy: "consume_on_read"
```

## 3. The Core of Decision Logic: Flow Decider

The `flow_decider` is the agent's primary mechanism for deciding its next action when the LLM does not explicitly call a tool. It is a list of "condition-action" rules that are evaluated in order during the `post_async` phase of the agent's lifecycle.

*   **`condition`**: Same as observers, this uses the `v['...']` syntax to safely inspect the agent's context.
*   **`action`**: Defines the outcome. Common `action.type` values include:
    *   `continue_with_tool`: If `state.current_action` is set (meaning a tool was called), this action continues the flow by executing that tool.
    *   `end_agent_turn`: Terminates the agent's current work cycle, optionally providing a success or error outcome.
    *   `loop_with_inbox_item`: Injects a `SELF_REFLECTION_PROMPT` into the agent's own inbox, causing it to re-evaluate its state in the next turn (a form of self-correction).
    *   `await_user_input`: Pauses the agent's execution until a new message is received from the user.

**Example**: A simple `flow_decider` for a user-facing Partner Agent.
```yaml
flow_decider:
  # If a tool was called, execute it.
  - id: "rule_tool_call_exists"
    condition: "v['state.current_action']"
    action:
      type: "continue_with_tool"

  # Otherwise, wait for the user's next message.
  - id: "rule_no_tool_call_fallback"
    condition: "True" # This is a catch-all rule
    action:
      type: "await_user_input"
```

## 4. Engineering the System Prompt (`system_prompt_construction`)

This section is solely responsible for constructing the **system prompt**. All dynamic, turn-by-turn context is injected via the `Inbox` and its `Ingestors`, not here.

*   `system_prompt_segments`: A list of components that are assembled in a specified `order` to form the final system prompt.
*   **Segment Types**:
    *   `static_text`: The content is sourced directly from the `text_definitions` block or an inline `content` field. It supports template interpolation for state values (e.g., `{{ state.agent_start_utc_timestamp }}`).
    *   `state_value`: Dynamically fetches a value from the agent's context and formats it using a specified `Ingestor`.
    *   `tool_description`: Automatically generates a formatted list of all tools available to the agent.
    *   `tool_contributed_context`: Injects contextual information provided by tools themselves.

## 5. Controlling Capabilities (`tool_access_policy`)

This section defines which tools an agent is allowed to access. It acts as a capability whitelist.

*   `allowed_toolsets`: A list of `toolset_name`s. The agent will have access to all tools belonging to these sets. Supports special category names for MCP servers (see note below).
*   `allowed_individual_tools`: A list of specific tool names for more granular control.

**Example**:
```yaml
tool_access_policy:
  allowed_toolsets:
    - "planning_tools"
    - "monitoring_tools"
  allowed_individual_tools:
    - "LaunchPrincipalExecutionTool"
```

> [!TIP]
> For MCP servers, you can use category toolsets like `all_user_specified_mcp_servers` to include all enabled servers of that category. See [Built-in Tools](./06-built-in-tools.md#4-custom-mcp-servers--categories) for details.
