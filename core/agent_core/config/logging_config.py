# agent_core/config/logging_config.py
import logging
import sys
import os
from contextvars import ContextVar, copy_context
import asyncio

# 1. Define context variables to pass context throughout the application
run_id_var = ContextVar('run_id', default=None)
agent_id_var = ContextVar('agent_id', default=None)
turn_id_var = ContextVar('turn_id', default=None)

# 2. Context copying utility functions
def copy_logging_context():
    """Copies the current logging context to a new async task."""
    return copy_context()

def run_with_context(context, func, *args, **kwargs):
    """Runs a function in the specified context."""
    return context.run(func, *args, **kwargs)

async def create_task_with_context(coro):
    """Creates an async task with the current context."""
    ctx = copy_context()
    return asyncio.create_task(coro, context=ctx)

# 3. Define a Filter to inject context variable values into each LogRecord
class ContextFilter(logging.Filter):
    def filter(self, record):
        record.run_id = run_id_var.get()
        record.agent_id = agent_id_var.get()
        record.turn_id = turn_id_var.get()
        return True

# 4. Override the global logging configuration function
def setup_global_logging(log_level_str: str = "INFO", log_file: str = None):
    """
    Configures the root logger for structured JSON logging.
    
    Args:
        log_level_str: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to stdout.
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Get the root logger for configuration, which will affect all child loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers and filters to prevent duplicate output on hot reload
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Clear existing filters
    root_logger.filters.clear()
    
    # Force clear all existing logger handlers to ensure they use our configuration
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.filters.clear()
        logger.propagate = True  # Ensure logs propagate to the root logger
        
    # Create handler - supports file output
    if log_file:
        # Ensure the log directory exists
        import os
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        log_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # Also create a console handler for important information
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Only display WARNING and above in the console
    else:
        # Only output to stdout
        log_handler = logging.StreamHandler(sys.stdout)
    
    # Add our custom ContextFilter to the handler, not the logger
    context_filter = ContextFilter()
    log_handler.addFilter(context_filter)
    
    # Create a more concise formatter for readable structured logs
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # Add context fields
            record.run_id = run_id_var.get()
            record.agent_id = agent_id_var.get()
            record.turn_id = turn_id_var.get()
            
            # Collect extra fields
            extra_fields = {}
            excluded_fields = ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                              'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                              'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'message', 'asctime', 'run_id', 'agent_id', 'turn_id']
            
            for key, value in record.__dict__.items():
                if key not in excluded_fields:
                    extra_fields[key] = value
            
            # Format basic info
            base_msg = super().format(record)
            
            # Add extra fields (if any)
            if extra_fields:
                import json
                extra_str = json.dumps(extra_fields, ensure_ascii=False, default=str)
                return f"{base_msg} | extra: {extra_str}"
            else:
                return base_msg
    
    # Use the more concise format
    formatter = CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s [run_id:%(run_id)s, agent_id:%(agent_id)s, turn_id:%(turn_id)s]'
    )
    log_handler.setFormatter(formatter)
    root_logger.addHandler(log_handler)
    
    # If using file output, also add the console handler
    if log_file:
        console_handler.addFilter(context_filter)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Disable log propagation for some overly verbose third-party libraries unless debugging them
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)  # Suppress uvicorn's default logs
    
    # Ensure our application logs are not filtered
    logging.getLogger("agent_core").setLevel(log_level)
    logging.getLogger("__main__").setLevel(log_level)
    logging.getLogger("api").setLevel(log_level)

    logging.info("structured_logging_configured", extra={"log_level": log_level_str, "log_file": log_file})
