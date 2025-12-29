# Common Ground Agent Core - System Architecture & Design

**Purpose**: To provide developers and contributors with a deep and accurate understanding of the system's architecture, core concepts, execution flow, and key code modules.

---

## **1. System Overview & Core Philosophy**

### 1.1 System Goal
To build a **modular, extensible, and fully observable** multi-agent AI framework. The system is designed to solve complex research and analysis tasks by simulating an efficient team of human specialists, capable of multi-step reasoning, multi-source information processing, and dynamic planning.

### 1.2 Core Design Principles
*   **Layered Collaboration**: A three-tier agent hierarchy (Partner, Principal, Associate) separates concerns. The Partner handles user interaction and strategic planning, the Principal manages project execution and task decomposition, and Associates are specialized experts that execute discrete tasks.
*   **Declarative Behavior**: Agent behavior, logic, and "personality" are driven by human-readable YAML files (Agent Profiles), not hard-coded in Python. This provides immense flexibility and maintainability, allowing for rapid prototyping and modification of agent capabilities without changing the core engine.
*   **Event-Driven Communication**: All asynchronous communication between agents is handled via a private, standardized **`Inbox`**. This event queue decouples components, making the system more robust and scalable by eliminating brittle, complex direct state-checking.
*   **Unified State & Full Observability**: A central **`RunContext`** object manages all state for a single business run. A structured **`Turn`** model meticulously logs every agent interaction, ensuring every decision and action is transparent, auditable, and analyzable.
*   **Capability as a Service**: Any AI capability, whether internal logic (e.g., planning), an external API (e.g., web search), or a data service (e.g., RAG), is encapsulated as a standard "tool" or "service" that any agent can consume on demand.

### 1.3 Key Terminology
*   **Work Module**: The core unit of task management, defined in the shared `TeamState`. It is a persistent, stateful "work ticket" containing a description, status (`pending`, `ongoing`, `completed`, etc.), and an archive of all historical context related to its execution.
*   **Agent Profile**: A YAML file in `agent_profiles/profiles/` that defines an agent's entire behavioral model. It specifies the agent's LLM, accessible tools, system prompt construction logic, and, most importantly, its decision-making rules (`flow_decider`) and reactive behaviors (`observers`).
*   **Inbox**: An agent's private event queue (`state.inbox`). It is the sole entry point for all asynchronous inputs, including tool results, startup briefings, and directives from other agents.
*   **Turn**: The atomic unit of observability, defined in `models/turn.py`. It's a detailed record of one complete "thought-action" cycle for an agent, capturing the inputs, the LLM interaction (including token counts), tool calls, and final state changes.
*   **RunContext**: The globally unique root object representing a complete business run, defined in `models/context.py`. It holds all configuration, shared state (`team_state`), and non-serializable runtime objects (like the `event_manager` and `mcp_session_pool`).

---

## **2. System Architecture & Components**

### 2.1 Component Layers
The system is organized into four distinct layers:
1.  **Presentation & API Layer**: The entry point for all external interactions.
2.  **Core Logic & Orchestration Layer**: The "brain" of the system, containing the agent execution engine and its supporting framework services.
3.  **Backend Services & Infrastructure**: A collection of specialized services providing foundational capabilities like data storage, retrieval, and external tool access.
4.  **Declarative Configuration**: A set of YAML files that define and control the behavior of the entire system.

### 2.2 Key File & Directory Structure
The project's Python files are organized to reflect this layered architecture.

```
core/
├── agent_core/
│   ├── api/                    # Handles FastAPI server, WebSocket connections, and message routing.
│   ├── config/                 # Application and contextual logging configuration.
│   ├── events/                 # Defines event handling strategies and content formatters ('Ingestors').
│   ├── framework/              # Core framework services: Tool Registry, Handover Service, Turn Manager.
│   ├── iic/                    # IIC file format parser and project/run persistence logic.
│   ├── llm/                    # LLM calling, dual-retry mechanisms, and declarative config resolution.
│   ├── models/                 # Core data structures: RunContext, SubContext, Turn.
│   ├── nodes/                  # The PocketFlow node system.
│   │   ├── base_agent_node.py  # CRITICAL: The unified, central execution engine for ALL agents.
│   │   └── custom_nodes/       # Implementations for all internal Python tools (e.g., planners, dispatchers).
│   ├── rag/                    # The advanced, federated RAG (Retrieval-Augmented Generation) system.
│   ├── services/               # Backend services: MCP Session Pool, File Monitor.
│   └── state/                  # State management, including the RunContext factory.
├── agent_profiles/             # All declarative YAML configurations.
│   ├── llm_configs/            # Reusable LLM configurations.
│   ├── profiles/               # Agent behavior profiles.
│   └── handover_protocols/     # Protocols for declarative context passing between agents.
├── agent_core/flow.py          # High-level "runner" script that initiates agent flows.
└── run_server.py               # Main entry point to start the FastAPI application.
```

---

## **3. Anatomy of a Research Run (Execution Flow)**

This section describes the step-by-step lifecycle of a typical research task, linking concepts to their implementation.

1.  **Startup & Partner Planning**: A user sends a request via WebSocket. `api/message_handlers.py::handle_start_run_message` is invoked. It calls `agent_core/state/management.py::create_run_context` to build the global `RunContext`. The initial prompt is placed in the **Partner Agent's Inbox**. The Partner agent (an instance of `BaseAgentNode`) processes this, converses with the user, and calls the 'manage_work_modules' tool. This tool, implemented in `agent_core/nodes/custom_nodes/stage_planner_node.py`, updates the `work_modules` dictionary within the shared `TeamState`.

2.  **Context Handover**: The Partner calls the 'LaunchPrincipalExecutionTool'. This tool invokes the `agent_core/framework/handover_service.py::HandoverService`. The service reads the `agent_profiles/handover_protocols/partner_to_principal_initial_briefing.yaml` protocol. It uses the rules in this file to safely extract data (like the user's query and the work modules) from the Partner's `RunContext` and packages it into a new `AGENT_STARTUP_BRIEFING` event. This event is then placed in the **Principal Agent's Inbox**.

3.  **Principal & Associate Execution Loop**:
    *   **Dispatch**: The Principal Agent processes its inbox. It calls the 'dispatch_submodules' tool, implemented in `nodes/custom_nodes/dispatcher_node.py`.
    *   **Delegate**: The `DispatcherNode` again uses the `HandoverService`, this time with the `principal_to_associate_briefing.yaml` protocol, to create isolated startup briefings for one or more **Associate Agents**. This briefing contains everything the Associate needs to know, such as its specific module description and any inherited context.
    *   **Execute**: Each Associate Agent (another `BaseAgentNode` instance) processes its briefing and executes its specialized task, for example, by calling the 'rag_query' tool.
    *   **Deliver**: Upon completion, the Associate uses the 'generate_message_summary' tool (`agent_core/nodes/custom_nodes/finish_node.py`) to receive an "instructional prompt". Following this prompt, the agent synthesizes its work into a structured JSON `deliverable`.
    *   **Aggregate & Report**: The `DispatcherNode` collects the `deliverable`, archives the full conversational history of the Associate's work into the `TeamState`'s `work_modules[id].context_archive`, and places a `TOOL_RESULT` event, containing the deliverable, back into the **Principal's Inbox**. This closes the execution loop.

4.  **Finalization**: The Principal iteratively loops through Step 3 until all work modules are 'completed'. It then calls 'generate_markdown_report' to get instructions for synthesizing the final report. After generating the report, it calls 'finish_flow' to terminate the process, which in turn notifies the Partner via its own inbox.

---

## **4. Deep Dive into Key Mechanisms**

### 4.1 Declarative Behavior: The Agent Profile
*   **Location**: `agent_profiles/`
*   **Mechanism**: Agent behavior is defined in YAML. The `BaseAgentNode` reads its assigned profile to configure its entire lifecycle.
*   **Key Fields**:
    *   `llm_config_ref`: Points to a reusable LLM configuration.
    *   `tool_access_policy`: A whitelist of `toolset_name`s the agent can use.
    *   `system_prompt_construction`: A list of ordered, conditional segments that are dynamically assembled to create the system prompt.
    *   `pre_turn_observers` / `post_turn_observers`: Rules that allow the agent to react to its own state. For example, a `post_turn_observer` can check if the agent failed to call a tool and inject a self-reflection prompt into its own inbox.
    *   `flow_decider`: The agent's central decision-making logic. It's a list of rules evaluated after the LLM call to determine the next action (e.g., 'continue_with_tool', 'end_agent_turn', 'await_user_input').

### 4.2 Event-Driven Communication: The Inbox Model
*   **Location**: `framework/inbox_processor.py`, `events/event_strategies.py`
*   **Mechanism**: The `InboxProcessor` is the heart of the agent's perception loop. Before each "thought" cycle, it consumes the agent's inbox. Crucially, it **sorts items by a predefined priority** defined in `EVENT_STRATEGY_REGISTRY` (e.g., `TOOL_RESULT` is handled before a `USER_PROMPT`). This ensures deterministic behavior. It then uses registered `Ingestor` functions from `events/ingestors.py` to format the event payload into LLM-readable text.

### 4.3 Full Observability: The Turn Model & TurnManager
*   **Location**: `models/turn.py`, `framework/turn_manager.py`
*   **Mechanism**: The `TurnManager` service is instantiated once per run. For every agent "thought-action" cycle, it creates a rich `Turn` object. This object is not just a log entry; it's a structured data record containing:
    *   `flow_id` and `source_turn_ids`: For dependency tracking and visualization.
    *   `inputs`: A structured log of which `InboxItem`s were processed in this turn.
    *   `llm_interaction`: Detailed logs of the LLM call, including `predicted_usage` and `actual_usage` of tokens.
    *   `tool_interactions`: A list of all tools called, their parameters, and their results.

### 4.4 Unified Tool System & MCP Session Pooling
*   **Location**: `framework/tool_registry.py`, `services/server_manager.py`
*   **Mechanism**: The `ToolRegistry` is initialized at startup. It discovers internal Python tools from `nodes/custom_nodes/` via the `@tool_registry` decorator. Simultaneously, it connects to external MCP servers defined in `mcp.json` to discover and register their tools. To manage these external connections efficiently, `server_manager.py` implements a global **MCP Session Pool**. Instead of creating a new connection for every tool call, agents acquire a session from the pool and release it afterward, drastically reducing overhead.

### 4.5 Robust LLM Calls: The Dual-Retry Mechanism
*   **Location**: `llm/call_llm.py`
*   **Mechanism**: A two-layer defense system ensures reliable communication with LLM APIs:
    1.  **Network/API Layer**: The `@robust_retry_with_backoff` decorator wraps the API call, catching and retrying transient errors like `RateLimitError` and `Timeout` with exponential backoff.
    2.  **Application Logic Layer**: An internal `for` loop within `call_litellm_acompletion` handles logical failures. If the LLM returns an empty or malformed response, a `FunctionCallErrorException` is raised and caught. The loop then dynamically modifies the prompt (e.g., adding "You must respond...") and retries the call.

### 4.6 Advanced RAG: The Federation Service (Work In Progress)
*   **Location**: `rag/`, `rag_configs/`
*   **Mechanism**: The `RAGFederationService` in `rag/federation.py` acts as a unified entry point to one or more underlying RAG data sources. These sources are defined in YAML files in `rag_configs/`. The service can perform concurrent searches across all active sources. The system uses DuckDB (`rag/duckdb_api.py`) for efficient local vector and metadata storage. Most importantly, `services/file_monitor.py` watches the workspace for file changes and **automatically triggers the RAG indexing process**, ensuring the knowledge base is always up-to-date.

### 4.7 State Persistence: The IIC File System (Work In Progress)
*   **Location**: `iic/`, `projects/`
*   **Mechanism**: The system ensures run state is durable. The logic in `iic/core/iic_handlers.py` manages persistence to the `projects/` directory. For each run, two files are maintained:
    1.  `{run_id}.json`: A complete, serializable snapshot of the entire `RunContext`, saved periodically.
    2.  `{run_name}.iic`: A lightweight file containing core metadata (ID, name, creation date) for quick browsing and identification.
    A `project.iic` file in each project's root directory serves as a fast-lookup index for all runs within it.
### 4.8 Budget-Aware Content Inheritance
*   **Location**: `utils/content_selection.py`, `nodes/custom_nodes/dispatcher_node.py`
*   **Problem**: When Associates inherit context from completed work modules via `inherit_messages_from`, unbounded message injection can cause new agents to exceed their context budget at birth.
*   **Mechanism**: A two-tier content selection strategy enforces budget limits:
    1.  **Budget Computation**: `(target_context_limit × 0.40) ÷ num_sources` allocates fair share per source module.
    2.  **Tier 1 Selection**: Use `deliverables.primary_summary` (LLM-generated summary from finish_flow) if it exists and fits budget.
    3.  **Tier 2 Selection**: Fall back to newest-first message selection with **hydration before measurement** to ensure accurate sizing.
*   **Key Design Decision**: Knowledge Base tokens in archived messages are expanded (hydrated) BEFORE size measurement, not after. This prevents post-selection expansion from exceeding budget limits.
*   **Integration Point**: Content selection happens in `dispatcher_node._preselect_inherited_content()` BEFORE the HandoverService is called, ensuring budget compliance at dispatch time.

See [Context Budget Management Architecture](architecture/context-budget-management.md) for detailed design documentation.