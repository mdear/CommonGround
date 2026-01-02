import dotenv
import logging
import argparse
import os
import uvicorn
# Import the new logging configuration function
from agent_core.config.logging_config import setup_global_logging

# Load environment variables early to get default port
dotenv.load_dotenv(".env", override=True)
DEFAULT_PORT = int(os.environ.get("BACKEND_PORT", "8800"))
DEFAULT_HOST = os.environ.get("API_HOST", "127.0.0.1")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run the PocketFlow Search Agent API Server')
    
    parser.add_argument(
        '--host',
        type=str,
        default=DEFAULT_HOST,
        help=f'Server host address (default: {DEFAULT_HOST}, from .env API_HOST)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=DEFAULT_PORT,
        help=f'Server port (default: {DEFAULT_PORT}, from .env BACKEND_PORT)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reloading (for development)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: None, logs to stdout only)'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()

    # .env already loaded at module level for defaults
    # Reload to ensure latest values
    dotenv.load_dotenv(".env", override=True, verbose=True)
    
    # Set log file in environment so lifespan manager can access it
    if args.log_file:
        os.environ["LOG_FILE"] = args.log_file

    # Use the new global logging configuration function
    setup_global_logging(args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__) # Use __name__

    logger.info("server_starting", extra={"host": args.host, "port": args.port, "log_level": args.log_level, "reload": args.reload})

    # Start the server
    try:
        uvicorn.run(
            "api.server:app", # Ensure it points to the correct FastAPI app object
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower(), # uvicorn uses lowercase levels
            log_config=None, # Disable uvicorn's default log config to use our structured logging
            access_log=False # Disable access logs, use our structured logging instead
        )
    except KeyboardInterrupt:
        logger.info("server_interrupted")
    except Exception as e:
        logger.error("server_startup_failed", extra={"error_message": str(e)}, exc_info=True)
    finally:
        logger.info("server_shutdown")

if __name__ == "__main__":
    main() 
