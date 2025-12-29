import uuid
import os
import logging
from typing import Dict, TYPE_CHECKING, List, Optional
from datetime import datetime  # Ensure datetime is imported
from fastapi import Response

if TYPE_CHECKING:
    from api.events import SessionEventManager  # For type hinting only

from api.session_security import (
    SessionSecurityConfig,
    SessionSecurityManager,
    SessionTokens,
    RefreshResult,
    ValidationResult,
    init_security_manager,
    get_security_manager,
)

logger = logging.getLogger(__name__)

# New: Used to store pending WebSocket session credentials
# Key: session_id, Value: creation timestamp
pending_websocket_sessions: Dict[str, datetime] = {}

# New: Global run store for storing active business run contexts
# Key: server_run_id, Value: run_context (Dict)
active_runs_store: Dict[str, Dict] = {}

# New: Global list to store all active event managers
active_event_managers: List['SessionEventManager'] = []


def initialize_session_security() -> SessionSecurityManager:
    """
    Initialize the session security manager with config from environment.

    Call this during app startup.
    """
    jwt_secret = os.environ.get("JWT_SECRET")
    if not jwt_secret:
        # Generate a random secret if not configured (development only)
        logger.warning(
            "JWT_SECRET not set - generating random secret. "
            "This is only acceptable for development!"
        )
        import secrets
        jwt_secret = secrets.token_urlsafe(64)

    config = SessionSecurityConfig(
        jwt_secret=jwt_secret,
        jwt_expiry_minutes=int(os.environ.get("JWT_EXPIRY_MINUTES", "15")),
        refresh_token_expiry_hours=int(os.environ.get("REFRESH_TOKEN_EXPIRY_HOURS", "24")),
        client_heartbeat_interval_seconds=int(
            os.environ.get("CLIENT_HEARTBEAT_INTERVAL_SECONDS", "20")
        ),
        client_heartbeat_timeout_seconds=int(
            os.environ.get("CLIENT_HEARTBEAT_TIMEOUT_SECONDS", "10")
        ),
        client_max_missed_heartbeats=int(
            os.environ.get("CLIENT_MAX_MISSED_HEARTBEATS", "2")
        ),
        # In development (HTTP), don't require Secure cookie
        fingerprint_cookie_secure=os.environ.get("COOKIE_SECURE", "true").lower() == "true",
    )

    return init_security_manager(config)


async def create_session(
    response: Response,
    project_id: str = "default",
    existing_fingerprint: Optional[str] = None,
) -> dict:
    """Creates a secure WebSocket connection credential with JWT.

    This function generates:
    - A unique session_id
    - A signed JWT token with fingerprint binding
    - A refresh token for silent token renewal
    - Sets the fingerprint HttpOnly cookie

    Args:
        response: FastAPI Response object to set cookies on
        project_id: Project identifier for the session
        existing_fingerprint: Optional existing fingerprint for reconnection

    Returns:
        dict: Session credentials including session_id, jwt_token, refresh_token
    """
    try:
        session_id = str(uuid.uuid4())

        # Create JWT and tokens via security manager
        security_manager = get_security_manager()
        tokens: SessionTokens = security_manager.create_session_tokens(
            session_id=session_id,
            project_id=project_id,
            existing_fingerprint=existing_fingerprint,
        )

        # Set fingerprint cookie (HttpOnly, Secure, SameSite=Strict)
        cookie_settings = security_manager.get_cookie_settings()
        response.set_cookie(
            value=tokens.fingerprint,
            **cookie_settings,
        )

        # Record this session_id and its creation timestamp
        pending_websocket_sessions[session_id] = datetime.now()
        logger.info(
            "secure_session_created",
            extra={
                "session_id": session_id,
                "project_id": project_id,
                "expires_in": tokens.expires_in,
            }
        )

        return {
            "session_id": tokens.session_id,
            "jwt_token": tokens.jwt_token,
            "refresh_token": tokens.refresh_token,
            "expires_in": tokens.expires_in,
            "status": "success",
        }

    except Exception as e:
        logger.error(
            "secure_session_creation_failed",
            extra={"error": str(e)},
            exc_info=True
        )
        raise


async def refresh_session_tokens(
    refresh_token: str,
    fingerprint_cookie: Optional[str],
) -> Optional[RefreshResult]:
    """
    Refresh session tokens using a refresh token.

    Args:
        refresh_token: The refresh token from client
        fingerprint_cookie: The fingerprint from HttpOnly cookie

    Returns:
        RefreshResult with new tokens, or None if invalid
    """
    try:
        security_manager = get_security_manager()
        result = security_manager.refresh_tokens(
            refresh_token=refresh_token,
            fingerprint_cookie=fingerprint_cookie,
        )

        if result:
            logger.info("session_tokens_refreshed")
        else:
            logger.warning("session_token_refresh_failed")

        return result

    except Exception as e:
        logger.error(
            "session_token_refresh_error",
            extra={"error": str(e)},
            exc_info=True
        )
        return None


def validate_session_jwt(
    jwt_token: str,
    fingerprint_cookie: Optional[str],
) -> ValidationResult:
    """
    Validate a session JWT token with fingerprint verification.

    Args:
        jwt_token: The JWT token to validate
        fingerprint_cookie: The fingerprint from HttpOnly cookie

    Returns:
        ValidationResult with validity status and session info
    """
    security_manager = get_security_manager()
    return security_manager.validate_jwt_with_fingerprint(
        jwt_token=jwt_token,
        fingerprint_cookie=fingerprint_cookie,
    )


def revoke_session(session_id: str) -> None:
    """
    Revoke all tokens for a session.

    Args:
        session_id: The session to revoke
    """
    security_manager = get_security_manager()
    security_manager.revoke_session(session_id)

    # Also remove from pending sessions
    if session_id in pending_websocket_sessions:
        del pending_websocket_sessions[session_id]

    logger.info("session_revoked", extra={"session_id": session_id})

# Old functions related to sessions and session_metadata (get_session, remove_session, cleanup_sessions)
# will be removed or heavily refactored, as business state is now managed by run_id and active_runs_store.
# The lifecycle of SessionEventManager is now bound to the WebSocket connection, not the old session object.

# def get_session(session_id: str) -> Optional[dict]:
#     """(Old logic) Get the top_level_shared object of the session"""
#     # ... old code ...
#     pass # Will be refactored or removed in subsequent steps based on the new run_id mechanism

# def remove_session(session_id: str, immediate: bool = True):
#     """(Old logic) Remove the session"""
#     # ... old code ...
#     pass # Will be refactored or removed in subsequent steps based on the new run_id mechanism

# def cleanup_sessions():
#     """(Old logic) Periodically clean up expired sessions marked for deletion"""
#     # ... old code ...
#     pass

# cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
# cleanup_thread.start()
