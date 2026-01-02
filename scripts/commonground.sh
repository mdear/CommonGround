#!/bin/bash
# CommonGround Service Manager
# Usage: commonground.sh [start|stop|status|restart] [backend|frontend|all]
#
# Examples:
#   commonground.sh start           # Start both backend and frontend
#   commonground.sh start backend   # Start backend only
#   commonground.sh stop frontend   # Stop frontend only
#   commonground.sh status          # Show status of all services
#   commonground.sh restart         # Restart both services

set -e

# Configuration
PROJECT_DIR="$HOME/workspaces/git/CommonGround"
VENV_DIR="$HOME/workspaces/venvs/CommonGround"
PID_DIR="$PROJECT_DIR/.pids"
LOG_DIR="$PROJECT_DIR/logs"

# =============================================================================
# PORT CONFIGURATION: Read from .env (Single Source of Truth)
# =============================================================================
ENV_FILE="$PROJECT_DIR/core/.env"

# Function to read a variable from .env file
read_env_var() {
    local var_name="$1"
    local default_value="$2"
    if [[ -f "$ENV_FILE" ]]; then
        local value=$(grep -E "^${var_name}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2 | tr -d ' ')
        if [[ -n "$value" ]]; then
            echo "$value"
            return
        fi
    fi
    echo "$default_value"
}

BACKEND_PORT=$(read_env_var "BACKEND_PORT" "8800")
FRONTEND_PORT=$(read_env_var "FRONTEND_PORT" "3800")
API_HOST=$(read_env_var "API_HOST" "0.0.0.0")

# Create directories
mkdir -p "$PID_DIR" "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#######################################
# Utility functions
#######################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

is_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

get_pid() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        cat "$pid_file"
    fi
}

#######################################
# Backend functions
#######################################

start_backend() {
    local pid_file="$PID_DIR/backend.pid"

    if is_running "$pid_file"; then
        log_info "Backend is already running (PID: $(get_pid "$pid_file"))"
        return 0
    fi

    # Check venv exists
    if [ ! -d "$VENV_DIR" ]; then
        log_error "Virtual environment not found at $VENV_DIR"
        log_error "Run: uv venv $VENV_DIR --python 3.12 && source $VENV_DIR/bin/activate && cd $PROJECT_DIR/core && uv pip install -r requirements.txt"
        return 1
    fi

    log_info "Starting backend on port $BACKEND_PORT..."

    # Start backend in background
    (
        source "$VENV_DIR/bin/activate"
        cd "$PROJECT_DIR/core"
        exec python3 run_server.py --host 0.0.0.0 --port $BACKEND_PORT
    ) > "$LOG_DIR/backend.log" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    # Wait a moment and verify it started
    sleep 2
    if is_running "$pid_file"; then
        log_info "Backend started (PID: $pid)"
        log_info "Access: http://localhost:$BACKEND_PORT"
        log_info "Log: $LOG_DIR/backend.log"
    else
        log_error "Backend failed to start. Check $LOG_DIR/backend.log"
        rm -f "$pid_file"
        return 1
    fi
}

stop_backend() {
    local pid_file="$PID_DIR/backend.pid"

    if ! is_running "$pid_file"; then
        log_info "Backend is not running"
        rm -f "$pid_file"
        return 0
    fi

    local pid=$(get_pid "$pid_file")
    log_info "Stopping backend (PID: $pid)..."

    # Send SIGTERM
    kill "$pid" 2>/dev/null

    # Wait for graceful shutdown (up to 10 seconds)
    local count=0
    while [ $count -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
        ((count++)) || true
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warn "Force killing backend..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$pid_file"
    log_info "Backend stopped"
}

status_backend() {
    local pid_file="$PID_DIR/backend.pid"

    if is_running "$pid_file"; then
        echo -e "Backend:  ${GREEN}RUNNING${NC} (PID: $(get_pid "$pid_file"), Port: $BACKEND_PORT)"
    else
        echo -e "Backend:  ${RED}STOPPED${NC}"
        rm -f "$pid_file" 2>/dev/null
    fi
}

#######################################
# Frontend functions
#######################################

start_frontend() {
    local pid_file="$PID_DIR/frontend.pid"

    if is_running "$pid_file"; then
        log_info "Frontend is already running (PID: $(get_pid "$pid_file"))"
        return 0
    fi

    # Check for orphaned process on port (use fuser, more reliable than lsof)
    local orphan_pid=$(fuser $FRONTEND_PORT/tcp 2>/dev/null | tr -d ' ' || true)
    if [ -n "$orphan_pid" ]; then
        log_warn "Found orphaned process on port $FRONTEND_PORT (PID: $orphan_pid), killing..."
        kill -9 $orphan_pid 2>/dev/null || true
        sleep 1
    fi

    # Check npm exists, offer to install if not
    if ! command -v npm &> /dev/null; then
        log_warn "npm not found. Attempting to install Node.js..."

        # Try to install Node.js
        if command -v curl &> /dev/null; then
            log_info "Installing Node.js 20.x via NodeSource..."
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && \
            sudo apt-get install -y nodejs

            if ! command -v npm &> /dev/null; then
                log_error "Node.js installation failed"
                return 1
            fi
            log_info "Node.js installed successfully"
        else
            log_error "curl not found. Cannot auto-install Node.js"
            log_error "Install manually: sudo apt-get install -y nodejs npm"
            return 1
        fi
    fi

    # Check if dependencies installed
    if [ ! -d "$PROJECT_DIR/frontend/node_modules" ]; then
        log_info "Installing frontend dependencies..."
        (cd "$PROJECT_DIR/frontend" && npm install)
    fi

    log_info "Starting frontend on port $FRONTEND_PORT..."

    # Start frontend in background
    (
        cd "$PROJECT_DIR/frontend"
        exec npm run dev -- -H 0.0.0.0 -p $FRONTEND_PORT
    ) > "$LOG_DIR/frontend.log" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    # Wait a moment and verify it started
    sleep 3
    if is_running "$pid_file"; then
        log_info "Frontend started (PID: $pid)"
        log_info "Access: http://localhost:$FRONTEND_PORT"
        log_info "Log: $LOG_DIR/frontend.log"
    else
        log_error "Frontend failed to start. Check $LOG_DIR/frontend.log"
        rm -f "$pid_file"
        return 1
    fi
}

stop_frontend() {
    local pid_file="$PID_DIR/frontend.pid"

    # Helper function to get PIDs on frontend port
    get_port_pids() {
        fuser $FRONTEND_PORT/tcp 2>/dev/null | tr -d ' ' || true
    }

    if ! is_running "$pid_file"; then
        log_info "Frontend is not running"
        rm -f "$pid_file"
        # Also check if port is in use by orphaned process
        local orphan_pid=$(get_port_pids)
        if [ -n "$orphan_pid" ]; then
            log_warn "Found orphaned process on port $FRONTEND_PORT (PID: $orphan_pid), killing..."
            kill -9 $orphan_pid 2>/dev/null || true
        fi
        return 0
    fi

    local pid=$(get_pid "$pid_file")
    log_info "Stopping frontend (PID: $pid)..."

    # Kill all processes using the frontend port (catches npm + next-server children)
    local port_pids=$(get_port_pids)
    if [ -n "$port_pids" ]; then
        kill $port_pids 2>/dev/null || true
    fi

    # Also kill the parent process and its children by PID
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true

    # Wait for graceful shutdown
    local count=0
    while [ $count -lt 5 ]; do
        port_pids=$(get_port_pids)
        if [ -z "$port_pids" ]; then
            break
        fi
        sleep 1
        ((count++)) || true
    done

    # Force kill if port still in use
    port_pids=$(get_port_pids)
    if [ -n "$port_pids" ]; then
        log_warn "Force killing remaining processes..."
        kill -9 $port_pids 2>/dev/null || true
        sleep 1
    fi

    rm -f "$pid_file"
    log_info "Frontend stopped"
}

status_frontend() {
    local pid_file="$PID_DIR/frontend.pid"

    if is_running "$pid_file"; then
        echo -e "Frontend: ${GREEN}RUNNING${NC} (PID: $(get_pid "$pid_file"), Port: $FRONTEND_PORT)"
    else
        echo -e "Frontend: ${RED}STOPPED${NC}"
        rm -f "$pid_file" 2>/dev/null
    fi
}

clean_frontend() {
    log_info "Clearing Next.js cache..."
    # Use --force and suppress errors (files may be held briefly after process stops)
    rm -rf "$PROJECT_DIR/frontend/.next" 2>/dev/null || true
    rm -rf "$PROJECT_DIR/frontend/node_modules/.cache" 2>/dev/null || true
    # Retry once if directory still exists (race condition with process cleanup)
    if [[ -d "$PROJECT_DIR/frontend/.next" ]]; then
        sleep 0.5
        rm -rf "$PROJECT_DIR/frontend/.next" 2>/dev/null || true
    fi
    log_info "Next.js cache cleared"
}

#######################################
# Combined functions
#######################################

start_all() {
    start_backend
    start_frontend
}

stop_all() {
    stop_frontend
    stop_backend
}

status_all() {
    echo "=========================================="
    echo "  CommonGround Service Status"
    echo "=========================================="
    status_backend
    status_frontend
    echo "=========================================="
}

restart_all() {
    stop_all
    clean_frontend
    sleep 1
    start_all
}

#######################################
# Main
#######################################

show_usage() {
    echo "Usage: $(basename "$0") [command] [service]"
    echo ""
    echo "Commands:"
    echo "  start    Start service(s)"
    echo "  stop     Stop service(s)"
    echo "  restart  Restart service(s)"
    echo "  status   Show service status"
    echo "  logs     Tail service logs"
    echo "  clean    Clear caches (Next.js .next folder)"
    echo ""
    echo "Services:"
    echo "  backend   Backend API server (port $BACKEND_PORT)"
    echo "  frontend  Frontend dev server (port $FRONTEND_PORT)"
    echo "  all       Both services (default)"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0") start           # Start both"
    echo "  $(basename "$0") start backend   # Start backend only"
    echo "  $(basename "$0") stop frontend   # Stop frontend only"
    echo "  $(basename "$0") status          # Show status"
    echo "  $(basename "$0") logs backend    # Tail backend logs"
    echo "  $(basename "$0") clean frontend  # Clear Next.js cache"
}

# Parse arguments
COMMAND="${1:-status}"
SERVICE="${2:-all}"

case "$COMMAND" in
    start)
        case "$SERVICE" in
            backend)  start_backend ;;
            frontend) start_frontend ;;
            all)      start_all ;;
            *)        log_error "Unknown service: $SERVICE"; show_usage; exit 1 ;;
        esac
        ;;
    stop)
        case "$SERVICE" in
            backend)  stop_backend ;;
            frontend) stop_frontend ;;
            all)      stop_all ;;
            *)        log_error "Unknown service: $SERVICE"; show_usage; exit 1 ;;
        esac
        ;;
    restart)
        case "$SERVICE" in
            backend)  stop_backend; sleep 1; start_backend ;;
            frontend) stop_frontend; clean_frontend; sleep 1; start_frontend ;;
            all)      restart_all ;;
            *)        log_error "Unknown service: $SERVICE"; show_usage; exit 1 ;;
        esac
        ;;
    status)
        status_all
        ;;
    logs)
        case "$SERVICE" in
            backend)  tail -f "$LOG_DIR/backend.log" ;;
            frontend) tail -f "$LOG_DIR/frontend.log" ;;
            all)      tail -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" ;;
            *)        log_error "Unknown service: $SERVICE"; show_usage; exit 1 ;;
        esac
        ;;
    clean)
        case "$SERVICE" in
            frontend|all) clean_frontend ;;
            backend)  log_info "No cache to clean for backend" ;;
            *)        log_error "Unknown service: $SERVICE"; show_usage; exit 1 ;;
        esac
        ;;
    -h|--help|help)
        show_usage
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
