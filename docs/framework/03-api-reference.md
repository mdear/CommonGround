# API Reference Manual

**Version**: 0.1

## Table of Contents

1.  [Overview](#1-overview)
    *   1.1 API Purpose
    *   1.2 Communication Protocols and Core Concepts
2.  [HTTP API](#2-http-api)
    *   2.1 `POST /session` - Create WebSocket Connection Token
3.  [WebSocket API](#3-websocket-api)
    *   3.1 Connection Endpoint
    *   3.2 General Message Format
    *   3.3 Client -> Server Messages
        *   3.3.1 `start_run` (Core)
        *   3.3.2 `send_to_run` (Core)
        *   3.3.3 `stop_run`
        *   3.3.4 `stop_managed_principal`
        *   3.3.5 `request_available_toolsets`
        *   3.3.6 `request_run_profiles`
        *   3.3.7 `request_run_context`
        *   3.3.8 `request_knowledge_base`
        *   3.3.9 `subscribe_to_view`
        *   3.3.10 `manage_work_modules_request`
    *   3.4 Server -> Client Events
        *   3.4.1 Lifecycle and Status Events
        *   3.4.2 LLM Interaction Events
        *   3.4.3 Tool and Task Dispatch Events
        *   3.4.4 System and Data Events
4.  [Core State Object Reference](#4-core-state-object-reference)
    *   4.1 `RunContext` (Global Run Context)
    *   4.2 `TeamState` (Shared Team State)
    *   4.3 `SubContext` (Sub-flow Context)
5.  [Example Interaction Flow](#5-example-interaction-flow)
6.  [Error Handling](#6-error-handling)

---

## 1. Overview

### 1.1 API Purpose

This API is designed to provide client applications with a powerful interface to interact with the agent backend. It allows clients to start and manage various independent, concurrent business processes (called "Runs"), such as complex research tasks, simple chat completions, or interactions with a multi-agent team.

### 1.2 Communication Protocols and Core Concepts

*   **HTTP/S**: Used only for a one-time initial operation: obtaining a temporary WebSocket connection token.
*   **WebSocket (WS/WSS)**: Used for all subsequent real-time, bidirectional communication. All messages use JSON format.
*   **Run**: Represents an independent, stateful business process instance. Each "Run" has a unique, server-generated `run_id`. All client operations (like sending messages or stopping flows) must target a specific `run_id`.
*   **Event-Driven**: The backend proactively pushes events to the client via WebSocket to reflect the state changes, thought processes, and execution results of the AI agents.

---

## 2. HTTP API

### 2.1 `POST /session` - Create WebSocket Connection Token

This endpoint is used to obtain a temporary `session_id`, which is a **one-time token** for establishing a WebSocket connection. It does not create any long-term session state.

*   **Path**: `/session`
*   **Method**: `POST`
*   **Request Body (JSON)**:
    ```json
    {}
    ```
    (The request body is empty)
*   **Success Response (200 OK, JSON)**:
    ```json
    {
      "session_id": "string - A unique temporary ID for the WebSocket connection URL.",
      "status": "success"
    }
    ```

### 2.2 Project & Run Management

These RESTful endpoints are used to manage project and run metadata.

*   **`GET /projects`**: Lists all projects.
*   **`POST /project`**: Creates a new project.
    *   **Request Body**: `{"name": "New Project Name"}`
*   **`GET /project/{project_id}`**: Gets details for a specific project.
*   **`PUT /project/{project_id}`**: Updates project metadata.
    *   **Request Body**: `{"name": "Updated Project Name"}`
*   **`DELETE /project/{project_id}`**: Deletes a project (soft delete).
*   **`POST /project/{project_id}/upload`**: Uploads a file to the project's `assets` directory.
*   **`PUT /run/{run_id}`**: Updates run metadata.
*   **`DELETE /run/{run_id}`**: Deletes a run (soft delete).
*   **`PUT /run/{run_id}/name`**: Renames a run.
    *   **Request Body**: `{"new_name": "New Run Name"}`
*   **`POST /run/move`**: Moves a run between projects.
    *   **Request Body**: `{"run_id": "...", "from_project_id": "...", "to_project_id": "..."}`
*   **`GET /metadata`**: Gets OpenGraph metadata for a specified URL.
    *   **Query Parameter**: `url=<URL_to_fetch>`

---

## 3. WebSocket API

### 3.1 Connection Endpoint

After obtaining a `session_id`, the client should immediately use it to establish a WebSocket connection.

*   **URL**: `ws://<server_host>:<server_port>/ws/{session_id}`
    (e.g., `ws://127.0.0.1:8000/ws/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

### 3.2 General Message Format

All WebSocket messages should be UTF-8 encoded JSON strings and follow the basic structure: `{"type": "...", "data": {...}}`.

### 3.3 Client -> Server Messages

#### 3.3.1 `start_run` (Core)

Requests to start a new business run and obtain a `run_id`. This is the first step in a two-stage startup protocol. This message does not contain any business data.

*   **`type`**: `"start_run"`
*   **`data` (object, required)**:
    *   `request_id` (string, **required**): A client-generated unique ID for this request (e.g., UUIDv4). Used to associate the server's `run_ready` response with this request.
    *   `run_type` (string, **required**): The type of run to start. Supported values:
        *   `"partner_interaction"`: Starts a full interactive session with the Partner Agent.
        *   `"principal_direct"`: Directly starts the Principal Agent to perform a research task.
        *   `"chat_completion"`: Performs a simple chat completion.
        *   `"fim"`: Performs a Fill-in-the-middle text generation.
    *   `project_id` (string, optional): The associated IIC project ID.
    *   `resume_from_run_id` (string, optional): Used to resume a historical run; pass the `run_id` to be resumed.
    *   `initial_filename` (string, optional): A user-provided initial filename (without extension) when creating a new run.

#### 3.3.2 `send_to_run` (Core)

Sends a message or directive to a running `run_id`. This is also the second step of the two-stage startup protocol, used to **activate** a newly created run.

*   **`type`**: `"send_to_run"`
*   **`data` (object, required)**:
    *   `run_id` (string, required): The target run's `server_run_id` (provided by the `run_ready` event).
    *   `message_payload` (object, **required**): The main message content to be sent to the Agent. For a newly created run, the first message must contain this field to activate the flow.
        *   For `"partner_interaction"`: `{"prompt": "User's input text"}`
        *   For `"principal_direct"`: `{"directive_type": "...", "payload": {...}}` (simulates a directive from the Partner)
    *   `extra_payload` (object, optional): A generic container for sending meta-instructions.
        *   `profile_updates` (array, optional): An array of objects to dynamically update an Agent Profile at runtime. Each object has the following structure:
            ```json
            {
              "action": "CREATE" | "UPDATE" | "DISABLE" | "RENAME",
              "profile_logical_name": "string", // The logical name of the target Profile
              "base_on_profile_logical_name": "string", // (for CREATE)
              "updates_to_apply": { /* ... */ }, // (for CREATE/UPDATE)
              "new_logical_name_for_rename": "string" // (for RENAME)
            }
            ```
        *   `llm_config_updates` (array, optional): An array of objects to dynamically update LLM configurations at runtime. The structure will be similar to `profile_updates` (this feature is reserved for future expansion).

#### 3.3.3 `stop_run`

Requests to stop a specified run.

*   **`type`**: `"stop_run"`
*   **`data` (object, required)**:
    *   `run_id` (string, required): The `server_run_id` of the run to be stopped.

#### 3.3.4 `stop_managed_principal`

Requests to stop a Principal Agent that was launched and is managed by a Partner Agent.

*   **`type`**: `"stop_managed_principal"`
*   **`data` (object, required)**:
    *   `managing_partner_run_id` (string, required): The `server_run_id` of the Partner Agent managing the target Principal.

#### 3.3.5 `request_available_toolsets`

Requests the server to return information about all registered toolsets.

*   **`type`**: `"request_available_toolsets"`
*   **`data`**: `{}` (no parameters)

#### 3.3.6 `request_run_profiles`

Requests the full content of the `agent_profiles_store` for a specified run.

*   **`type`**: `"request_run_profiles"`
*   **`data` (object, required)**:
    *   `run_id` (string, required)

#### 3.3.7 `request_run_context`

Requests the serialized `run_context` for a specified run. Supports three modes for efficient data retrieval:

*   **`type`**: `"request_run_context"`
*   **`data` (object, required)**:
    *   `run_id` (string, required): The target run's ID.
    *   `mode` (string, optional): Query mode. Default: `"summary"`.
        *   `"summary"`: Returns lightweight overview without full messages (always fast, small response).
        *   `"full"`: Returns complete snapshot (may exceed WebSocket limits for large sessions).
        *   `"section"`: Returns only the requested section with optional pagination.
    *   `section` (string, required for mode="section"): Section to retrieve.
        *   `"meta"`: Session metadata only.
        *   `"team_state"`: Work modules and dispatch history.
        *   `"sub_contexts"`: Agent contexts with message pagination.
        *   `"knowledge_base"`: Knowledge base entries.
    *   `context_name` (string, optional): For `sub_contexts` section, specifies which context to retrieve (e.g., `"_principal_context_ref"`, `"_partner_context_ref"`). If omitted, returns list of available contexts.
    *   `message_offset` (integer, optional): Pagination offset for messages. Default: `0`.
    *   `message_limit` (integer, optional): Maximum messages to return. Default: `50`.

**Example: Summary Mode (Default)**
```json
{
  "type": "request_run_context",
  "data": {
    "run_id": "my-run-id"
  }
}
```

**Example: Paginated Messages**
```json
{
  "type": "request_run_context",
  "data": {
    "run_id": "my-run-id",
    "mode": "section",
    "section": "sub_contexts",
    "context_name": "_principal_context_ref",
    "message_offset": 0,
    "message_limit": 50
  }
}
```

**Response Structure (Summary Mode)**
```json
{
  "type": "run_context_response",
  "run_id": "my-run-id",
  "data": {
    "context": {
      "mode": "summary",
      "meta": { "run_id": "...", "status": "...", "run_type": "..." },
      "team_state": { "work_modules": {...}, "dispatch_history": [...] },
      "sub_contexts_summary": {
        "_principal_context_ref": {
          "message_count": 150,
          "inbox_count": 2,
          "has_deliverables": true,
          "deliverable_keys": ["final_report"],
          "last_message": { "role": "assistant", "content_preview": "..." }
        }
      },
      "knowledge_base_summary": { "key1": { "type": "string", "size": 100 } }
    }
  }
}
```

**Response Structure (Paginated Sub-Contexts)**
```json
{
  "type": "run_context_response",
  "run_id": "my-run-id",
  "data": {
    "context": {
      "mode": "section",
      "section": "sub_contexts",
      "context_name": "_principal_context_ref",
      "data": {
        "messages": [...],
        "inbox": [...],
        "deliverables": {...}
      },
      "pagination": {
        "total_messages": 150,
        "offset": 0,
        "limit": 50,
        "returned": 50,
        "has_more": true
      }
    }
  }
}
```

#### 3.3.8 `request_knowledge_base`

Requests the content of the `KnowledgeBase` for a specified run.

*   **`type`**: `"request_knowledge_base"`
*   **`data` (object, required)**:
    *   `run_id` (string, required)

#### 3.3.9 `subscribe_to_view`

Subscribes to a view model to receive `view_model_update` events when the backend state is updated.

*   **`type`**: `"subscribe_to_view"`
*   **`data` (object, required)**:
    *   `run_id` (string, required): The target run's `server_run_id`.
    *   `view_name` (string, required): The name of the view to subscribe to. Supported values: `"flow_view"`, `"kanban_view"`, `"timeline_view"`.

#### 3.3.10 `manage_work_modules_request`

Directly requests CRUD operations on work modules, typically initiated by the `PlanningPanel` UI.

*   **`type`**: `"manage_work_modules_request"`
*   **`data` (object, required)**:
    *   `run_id` (string, required): The target run's `server_run_id`.
    *   `actions` (array, required): An array of action objects, with a structure identical to the `actions` parameter of the `manage_work_modules` tool.

### 3.4 Server -> Client Events

#### 3.4.1 Lifecycle and Status Events

*   **`run_ready`**: Confirms that a new run is ready on the backend and returns its official `run_id`. This is the response to a `start_run` request.
    *   `data`: `{"request_id": "string", "run_id": "string", "status": "success"}`
*   **`run_config_updated`**: The runtime configuration (e.g., Profile or LLM config) has been updated.
    *   `run_id`
    *   `data`: `{"config_type": "agent_profile" | "llm_config" | ..., "item_identifier": "string", "details": object}`
*   **`turn_completed`**: An Agent's turn has successfully completed. This event is primarily used to trigger backend operations like persistence.
    *   `run_id`, `agent_id`
    *   `data`: `{"turn_id": "string"}`
*   **`view_model_update`**: The server pushes this event when a subscribed view model is updated.
    *   `run_id`
    *   `data`: `{"view_name": "flow_view" | "kanban_view" | "timeline_view", "model": object}`

#### 3.4.2 LLM Interaction Events

*   **`llm_stream_started`**: An LLM call attempt has started.
    *   `run_id`, `agent_id`, `parent_agent_id`, `stream_id`, `llm_id`
*   **`llm_chunk`**: A streaming fragment of the LLM response.
    *   `run_id`, `agent_id`, `parent_agent_id`, `stream_id`, `llm_id`
    *   `data`: `{"content": "string", "chunk_type": "content" | "reasoning" | "tool_name" | ..., "module_id": "string" (optional), "dispatch_id": "string" (optional)}`
*   **`llm_stream_ended`**: The LLM stream has successfully ended.
    *   `run_id`, `agent_id`, `parent_agent_id`, `stream_id`
*   **`llm_stream_failed`**: An LLM call attempt has failed.
    *   `run_id`, `agent_id`, `parent_agent_id`, `stream_id`
    *   `data`: `{"reason": "string"}`
*   **`llm_response`**: The complete LLM response (usually sent after the stream has ended).
    *   `run_id`, `agent_id`, `parent_agent_id`, `stream_id`
    *   `data`: `{"content": "...", "tool_calls": [...], "reasoning": "...", "module_id": "string" (optional), "dispatch_id": "string" (optional)}`
*   **`llm_request_params`**: The request parameters sent to the LLM (for debugging).
    *   `run_id`, `agent_id`, `stream_id`, `llm_id`
    *   `data`: `{"params": object}`

#### 3.4.3 Tool and Task Dispatch Events

*   **`work_module_updated`**: A work module's status or content has been updated. This is the **primary event** for tracking the module lifecycle.
    *   `run_id`
    *   `data`: `{"module": object}` (The complete work module object)

#### 3.4.4 System and Data Events

*   **`turns_sync`**: A snapshot of the complete `turns` list, serving as the **authoritative source of truth** for rendering conversation and flow history on the frontend.
    *   `run_id`
    *   `data`: `{"turns": [object]}` (A complete list of `Turn` objects)
*   **`available_toolsets_response`**: The response to `request_available_toolsets`.
    *   `data`: `{"toolsets": object}`
*   **`run_profiles_response`**: The response to `request_run_profiles`.
    *   `run_id`, `data`: `{"profiles": object | null, "error": "string | null"}`
*   **`run_context_response`**: The response to `request_run_context`.
    *   `run_id`, `data`: `{"context": object | null, "error": "string | null"}`
*   **`knowledge_base_response`**: The response to `request_knowledge_base`.
    *   `run_id`, `data`: `{"knowledge_base": object | null, "error": "string | null"}`
*   **`error`**: A generic error message.
    *   `run_id`, `agent_id`, `data`: `{"message": "string"}`
*   **`project_structure_updated`**: This system-level event is broadcast to all active user sessions when the project structure changes (e.g., a run is renamed or deleted).
    *   `data`: `{"reason": "rename_run" | "delete_project" | ..., "details": object}`

---

## 4. Core State Object Reference

### 4.1 `RunContext` (Global Run Context)

Represents the top-level context for a business run, serving as the single source of truth for all state and configuration.

*   `meta` (object): The run's identity information, including `run_id`, `run_type`, `creation_timestamp`, and `status` (e.g., `"CREATED"`, `"RUNNING"`, `"AWAITING_INPUT"`).
*   `config` (object): A snapshot of the configuration at the start of the run, including `agent_profiles_store` and `shared_llm_configs_ref`.
*   `team_state` (object): Mutable state shared across team members. See below.
*   `runtime` (object): Global, non-serializable runtime objects like `event_manager`, `server_manager`, and `knowledge_base`.
*   `sub_context_refs` (object): References to all active sub-flow contexts (`SubContext`).

### 4.2 `TeamState` (Shared Team State)

State stored in the `RunContext` and shared by all team members.

*   `work_modules` (Dict[str, object]): A dictionary with `module_id` as the key and the work module object as the value. This is the core of the project state. Each work module contains:
    *   `module_id` (string): Unique identifier for the module.
    *   `status` (string): Current status (`pending`, `in_progress`, `pending_review`, `completed`).
    *   `context_archive` (List[object]): Archived Associate contexts upon completion.
    *   `deliverables` (object | null): Propagated deliverables from the completed Associate. This field is automatically populated by the `DispatcherNode` when an Associate completes, copying the `deliverables` from `context_archive` for easy access.
*   `question` (str): The core research question.
*   `profiles_list_instance_ids` (List[str]): A list of Associate Profile instance IDs available to the Principal.
*   `is_principal_flow_running` (bool): Whether the Principal is currently running.
*   `dispatch_history` (List[Dict]): A history of Associate task dispatches.
*   `turns` (List[Dict]): A list containing all `Turn` objects that have occurred, for full traceability.

### 4.3 `SubContext` (Sub-flow Context)

Represents the context for a specific Agent flow (e.g., Partner, Principal, Associate).

*   `meta` (object): The Agent's identity information, including `run_id`, `agent_id`, `parent_agent_id`.
*   `state` (object): The Agent's **private, serializable** state.
    *   `messages` (List[Dict]): The conversation history.
    *   `current_action` (Dict): The tool action currently pending execution.
    *   `inbox` (List[Dict]): The Agent's event inbox, a core part of the new architecture.
    *   `initial_parameters` (Dict): Startup parameters (especially important for Associates).
    *   `deliverables` (Dict): The final output from an Associate.
*   `runtime_objects` (object): The Agent's unique, non-serializable runtime objects (like `asyncio.Event`).
*   `refs` (object): References to the `RunContext` and `TeamState`.

#### 4.4 `FlowViewModel`

Describes the topology of the `FlowView`. The view uses **epoch-based depth calculation** to ensure time flows top-to-bottom. Disconnected subgraphs (e.g., from separate Principal dispatches) are identified as separate "epochs" and rendered sequentially sorted by timestamp. When multiple epochs exist, visual epoch separator nodes are inserted between them.

*   `nodes` (List[object]): A list of nodes. Each node object contains:
    *   `id` (string): The unique ID of the node.
    *   `type`: Fixed as `"custom"`.
    *   `data` (object): Node data, containing:
        *   `label` (string): The display name of the node.
        *   `nodeType` (string): `"turn" | "gather" | "epoch_separator"`. The `epoch_separator` type is used to visually separate disconnected execution epochs (e.g., multiple Principal dispatches).
        *   `status` (string): `"idle" | "running" | "completed" | "error" | "cancelled"`.
        *   `depth` (number): The depth of the node in the flow diagram, used for hierarchical layout. Depths are calculated per-epoch with offsets to ensure chronological top-to-bottom ordering.
        *   `epoch_index` (number, optional): For `epoch_separator` nodes, indicates which epoch boundary this separator represents.
        *   `content_stream_id` (string | null): The stream ID used to associate `llm_chunk` events.
        *   `timestamp` (string): ISO 8601 timestamp.
        *   `originalId` (string): The ID of the original message or tool call associated with this node.
        *   `tool_interactions` (array | null): (For `turn` nodes only) Contains detailed information about all tool interactions in this turn.
        *   `final_content` (string | null): (For `turn` nodes only) The final text content from the LLM at the end of the turn.
*   `edges` (List[object]): A list of edges. Each edge object contains:
    *   `id` (string): The unique ID of the edge.
    *   `source` (string): The source node ID.
    *   `target` (string): The target node ID.
    *   `animated` (boolean): Whether to display an animation.
    *   `edgeType` (string | null): The semantic type of the edge, such as `"return"` for gather nodes or `"epoch_boundary"` for connections to/from epoch separators.

#### 4.5 `KanbanViewModel`

Provides pre-grouped data for the Kanban view.

*   `view_by_status` (object): Work modules grouped by status.
*   `view_by_agent` (object): Work modules grouped by assignee.
*   `last_updated` (string): ISO 8601 timestamp.

#### 4.6 `TimelineViewModel`

Provides pre-aggregated data for the timeline view. It includes "time break" information used to compress long idle periods on the frontend.

*   `lanes` (List[object]): A list of lanes, each representing an Agent.
    *   `agentId` (string): The Agent's ID.
    *   `blocks` (List[object]): A list of activity blocks for this Agent.
        *   `moduleId` (string): The associated ID (`turn_id` or `module_id`).
        *   `moduleName` (string): The display name of the block.
        *   `startTime` (string): The **actual** start time in ISO 8601 format.
        *   `endTime` (string | null): The **actual** end time in ISO 8601 format, or `null` if still running.
        *   `status` (string): The status of the block.
*   `overallStartTime` (string | null): The earliest **actual** start time of all activities.
*   `overallEndTime` (string | null): The latest **actual** end time of all activities.
*   `isLive` (boolean): A flag indicating if the timeline is in "live" mode (i.e., if the `Principal` flow is running). If `true`, the frontend should extend the timeline to the current time in real-time. If `false`, the timeline should be frozen at `overallEndTime`.
*   `timeBreaks` (List[object]): A list of time breaks representing system idle periods that should be compressed.
    *   `breakStart` (string): ISO timestamp for the start of the idle period.
    *   `breakEnd` (string): ISO timestamp for the end of the idle period.
    *   `duration` (number): The total duration of the idle period in seconds.

#### 4.7 WebSocket Event Contract

To ensure decoupling and parallel development between the frontend and backend, both parties must strictly adhere to the following data structures for WebSocket events.

*   **Client -> Server**:
    *   `subscribe_to_view`: The `data` object must support `view_name: "flow_view" | "kanban_view" | "timeline_view"`.
*   **Server -> Client**:
    *   `view_model_update`: `data.view_name` will contain `"flow_view"`, `"kanban_view"`, or `"timeline_view"`, and `data.model` will contain the corresponding ViewModel object.
    *   `view_model_update_failed`: Contains the failed `view_name` and an error message.
    *   `run_context_response`: The structure of `data.context` now follows the output of `get_serializable_run_snapshot`, containing `meta`, `config`, `team_state`, and `sub_contexts_state`.

---

## 5. Example Interaction Flow

This is a typical interaction flow.

1.  **Client**: `POST /session` -> Receives `session_id: "sid_123"`.
2.  **Client**: Connects to `ws://.../ws/sid_123`.
3.  **Client**: Sends a `start_run` request.
    ```json
    {
      "type": "start_run",
      "data": {
        "run_type": "partner_interaction",
        "request_id": "req_xyz_789"
      }
    }
    ```
4.  **Server**: Creates the run context and responds with a `run_ready` event.
    ```json
    {
      "type": "run_ready",
      "data": {
        "request_id": "req_xyz_789",
        "run_id": "rid_abc",
        "status": "success"
      }
    }
    ```
5.  **Client**: After receiving `run_ready`, sends the first business message to **activate** the run.
    ```json
    {
      "type": "send_to_run",
      "data": {
        "run_id": "rid_abc",
        "message_payload": {
          "prompt": "Research AI ethics"
        }
      }
    }
    ```
6.  **Server**: Receives the first message, starts the `partner_interaction` flow, and begins streaming state updates via `turns_sync`, `view_model_update`, etc.
7.  **Client**: The user continues the interaction by sending subsequent messages.
    ```json
    {
      "type": "send_to_run",
      "data": {
        "run_id": "rid_abc",
        "message_payload": {
          "prompt": "Okay, please start planning"
        }
      }
    }
    ```
8.  **Server**: The Partner Agent continues execution, calls tools, launches the Principal, and continuously sends a series of events like `work_module_updated`.
9.  **Client**: (At a later time) Sends `{"type": "stop_run", "data": {"run_id": "rid_abc"}}` to end the entire interaction.

---

## 6. Error Handling

*   **HTTP**: Uses standard HTTP status codes (`4xx` for client errors, `5xx` for server errors).
*   **WebSocket**:
    *   If the connection fails to establish, a standard WebSocket close frame will be received with a status code (e.g., `1008 Policy Violation`).
    *   After the connection is established, business logic errors are sent via a `{type: "error", ...}` message.
