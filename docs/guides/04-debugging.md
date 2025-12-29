# Debugging Guide

The framework includes several features to aid in debugging and understanding agent behavior.

## 1. Logging
The primary tool for debugging is the structured logging system.

*   **Log Level**: To see detailed operational logs, set the log level to `DEBUG` in your `.env` file or via the command line argument when running `run_server.py`.
    ```
    # in .env
    LOG_LEVEL="DEBUG"
    ```
    ```bash
    # or via command line
    python run_server.py --log-level DEBUG
    ```
*   **Structured Logs**: Logs are in JSON format, containing rich context like `run_id`, `agent_id`, and `turn_id`, making them easy to parse and filter.

## 2. Environment Variables for Debugging
You can enable advanced debugging features by setting environment variables in your `.env` file.

*   **`STATE_DUMP="true"`**: At the end of a flow, this will dump a complete JSON snapshot of the final `RunContext` to a file in the `reports/` directory. This is invaluable for post-mortem analysis of the agent's state.

*   **`CAPTURE_LLM_REQUEST_BODY="true"`**: This will record the exact request payload (including messages, system prompt, tools, and parameters like temperature) sent to the LLM for every call. This data is stored in the `Turn` object at `llm_interaction.final_request` and can be inspected in the web UI's DevTools panel or in the state dump. Use this to verify the final prompt and configuration seen by the model. Note that this can significantly increase the size of the state object.

## 3. Using the Web UI for Debugging
The built-in web UI is a powerful tool for real-time observation.

*   **DevTools Panel**: This panel provides a raw, real-time stream of all WebSocket events. You can inspect `turns_sync` events to see the detailed structure of each `Turn` object as it's created and updated.
*   **Flow, Kanban, and Timeline Views**: These visualizations are built directly from the `Turn` data and provide high-level insights into the agent team's workflow, task status, and execution timing. Use them to identify bottlenecks or incorrect logic flows.

## 4. Command-Line Scripts

The `scripts/` directory contains utility scripts for operating and analyzing CommonGround sessions.

### 4.1 Service Manager (`commonground.sh`)

A bash script for managing backend and frontend services in development.

**Location**: `scripts/commonground.sh`

**Usage**:
```bash
# Start both backend and frontend
./scripts/commonground.sh start

# Start only backend (port 8800)
./scripts/commonground.sh start backend

# Start only frontend (port 3800)
./scripts/commonground.sh start frontend

# Stop all services
./scripts/commonground.sh stop

# Restart services
./scripts/commonground.sh restart

# Check service status
./scripts/commonground.sh status
```

**Features**:
- Manages PID files in `.pids/` directory
- Logs output to `logs/backend.log` and `logs/frontend.log`
- Automatically activates the Python virtual environment
- Color-coded status output

### 4.2 Session Analyzer (`analyze_session.py`)

A Python tool for deep analysis of completed CommonGround sessions. Provides multi-level observability into session flow, agent handoffs, token utilization, tool usage patterns, and error detection.

**Location**: `scripts/analyze_session.py`

**Input Formats**:
The analyzer accepts multiple input formats:
```bash
# By URL (copy from browser)
python scripts/analyze_session.py http://localhost:3800/webview/r?id=tentacled-pearl-oriole

# By session ID
python scripts/analyze_session.py tentacled-pearl-oriole

# By file path
python scripts/analyze_session.py projects/MyProject/session-id.json
```

**Analysis Levels** (`--level`):
| Level | Description |
|-------|-------------|
| `summary` | High-level overview (default) |
| `detailed` | Per-agent breakdown with key metrics |
| `deep` | Full message-level analysis |
| `timeline` | Chronological event trace |

**Focus Areas** (`--focus`):
| Focus | Description |
|-------|-------------|
| `all` | Full session analysis (default) |
| `principal` | Focus on Principal agent |
| `partner` | Focus on Partner agent |
| `WM_N` | Focus on specific work module (e.g., `WM_1`, `WM_2`) |
| `errors` | Focus on errors and issues |
| `tokens` | Focus on token utilization |

**Examples**:
```bash
# Quick summary of a session
python scripts/analyze_session.py tentacled-pearl-oriole

# Detailed breakdown with per-agent metrics
python scripts/analyze_session.py tentacled-pearl-oriole --level detailed

# Deep dive into a specific work module
python scripts/analyze_session.py tentacled-pearl-oriole --level deep --focus WM_1

# Analyze token usage patterns
python scripts/analyze_session.py tentacled-pearl-oriole --focus tokens

# Find errors in a session
python scripts/analyze_session.py tentacled-pearl-oriole --focus errors

# Full timeline trace
python scripts/analyze_session.py tentacled-pearl-oriole --level timeline
```

**Output Includes**:
- Session metadata and duration
- Agent handoff chain visualization
- Token budget compliance per agent
- Tool invocation patterns and success rates
- Error categorization and diagnosis hints
- Work module status summary
