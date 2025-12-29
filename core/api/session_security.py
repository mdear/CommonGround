"""
JWT-based session security with HttpOnly fingerprint cookie binding.

Implements:
- RFC 8725 JWT Best Current Practices
- OWASP Token Sidejacking Prevention
- Auth0 Refresh Token Rotation Pattern

See docs/architecture/session-resilience.md for full design documentation.
"""

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging

import jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class SessionSecurityConfig:
    """Configuration for session security."""

    # JWT Settings
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 15
    jwt_issuer: str = "commonground"
    jwt_audience: str = "commonground-ws-reconnect"

    # Refresh Token Settings
    refresh_token_expiry_hours: int = 24
    refresh_token_rotation: bool = True

    # Auto-Refresh Settings
    auto_refresh_threshold: float = 0.9  # 90% of lifespan

    # Fingerprint Cookie Settings
    fingerprint_cookie_name: str = "__Secure-Fgp"
    fingerprint_bytes: int = 32  # 256 bits of entropy
    fingerprint_cookie_secure: bool = True
    fingerprint_cookie_httponly: bool = True
    fingerprint_cookie_samesite: str = "Strict"
    fingerprint_cookie_max_age: int = 86400  # 24 hours

    # Client Heartbeat Settings
    client_heartbeat_interval_seconds: int = 20
    client_heartbeat_timeout_seconds: int = 10
    client_max_missed_heartbeats: int = 2


class TokenPayload(BaseModel):
    """Structured JWT token payload."""

    # Standard claims
    iss: str  # Issuer
    sub: str  # Subject (session_id)
    aud: str  # Audience
    exp: int  # Expiry timestamp
    iat: int  # Issued at timestamp
    nbf: int  # Not before timestamp
    jti: str  # JWT ID (unique, for revocation)

    # Custom claims
    pid: str  # Project ID
    ver: int = 1  # Token version (for forced revocation)
    fph: str  # Fingerprint hash (SHA256 of cookie value)


class SessionTokens(BaseModel):
    """Response model for session token creation."""

    session_id: str
    jwt_token: str
    refresh_token: str
    expires_in: int  # Seconds until JWT expires
    fingerprint: str  # Raw fingerprint (to set in cookie)


class RefreshResult(BaseModel):
    """Response model for token refresh."""

    jwt_token: str
    refresh_token: str
    expires_in: int


class ValidationResult(BaseModel):
    """Result of JWT validation."""

    valid: bool
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    error: Optional[str] = None
    token_payload: Optional[TokenPayload] = None


@dataclass
class RefreshTokenState:
    """State for a refresh token."""

    token_hash: str  # SHA256 of the refresh token
    session_id: str
    project_id: str
    fingerprint_hash: str
    expires_at: datetime
    used: bool = False
    token_version: int = 1


class SessionSecurityManager:
    """
    Manages JWT session tokens with fingerprint cookie binding.

    Security properties:
    - JWT signed with HS256 (configurable)
    - HttpOnly fingerprint cookie prevents XSS token theft
    - Refresh token rotation detects token reuse
    - Explicit audience/issuer validation (RFC 8725)
    """

    def __init__(self, config: SessionSecurityConfig):
        self.config = config
        # In-memory store for refresh tokens (use Redis in production)
        self._refresh_tokens: dict[str, RefreshTokenState] = {}
        # Token version per session (for forced revocation)
        self._session_versions: dict[str, int] = {}
        # Revoked JTIs (for explicit token revocation)
        self._revoked_jtis: set[str] = set()

    def _generate_fingerprint(self) -> str:
        """Generate a cryptographically secure random fingerprint."""
        return secrets.token_urlsafe(self.config.fingerprint_bytes)

    def _hash_fingerprint(self, fingerprint: str) -> str:
        """Hash fingerprint using SHA256."""
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def _hash_refresh_token(self, token: str) -> str:
        """Hash refresh token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _generate_jti(self) -> str:
        """Generate unique JWT ID."""
        return secrets.token_urlsafe(16)

    def _generate_refresh_token(self) -> str:
        """Generate secure refresh token."""
        return secrets.token_urlsafe(32)

    def _get_session_version(self, session_id: str) -> int:
        """Get current token version for session."""
        return self._session_versions.get(session_id, 1)

    def create_session_tokens(
        self,
        session_id: str,
        project_id: str,
        existing_fingerprint: Optional[str] = None,
    ) -> SessionTokens:
        """
        Create new session tokens (JWT + refresh token + fingerprint).

        Args:
            session_id: The session identifier
            project_id: The project identifier
            existing_fingerprint: Optional existing fingerprint for reconnection

        Returns:
            SessionTokens with JWT, refresh token, and fingerprint
        """
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        exp_ts = int((now + timedelta(minutes=self.config.jwt_expiry_minutes)).timestamp())

        # Use existing fingerprint for reconnection, or generate new
        fingerprint = existing_fingerprint or self._generate_fingerprint()
        fingerprint_hash = self._hash_fingerprint(fingerprint)

        # Get or initialize session version
        version = self._get_session_version(session_id)
        if session_id not in self._session_versions:
            self._session_versions[session_id] = version

        # Create JWT payload
        payload = TokenPayload(
            iss=self.config.jwt_issuer,
            sub=session_id,
            aud=self.config.jwt_audience,
            exp=exp_ts,
            iat=now_ts,
            nbf=now_ts,
            jti=self._generate_jti(),
            pid=project_id,
            ver=version,
            fph=fingerprint_hash,
        )

        # Encode JWT with explicit header type (RFC 8725 Section 3.11)
        jwt_token = jwt.encode(
            payload.model_dump(),
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
            headers={"typ": "session+jwt"},
        )

        # Generate refresh token
        refresh_token = self._generate_refresh_token()
        refresh_expires = now + timedelta(hours=self.config.refresh_token_expiry_hours)

        # Store refresh token state
        self._refresh_tokens[self._hash_refresh_token(refresh_token)] = RefreshTokenState(
            token_hash=self._hash_refresh_token(refresh_token),
            session_id=session_id,
            project_id=project_id,
            fingerprint_hash=fingerprint_hash,
            expires_at=refresh_expires,
            token_version=version,
        )

        logger.info(
            f"Created session tokens for session={session_id}, "
            f"project={project_id}, expires_in={self.config.jwt_expiry_minutes}m"
        )

        return SessionTokens(
            session_id=session_id,
            jwt_token=jwt_token,
            refresh_token=refresh_token,
            expires_in=self.config.jwt_expiry_minutes * 60,
            fingerprint=fingerprint,
        )

    def validate_jwt_with_fingerprint(
        self,
        jwt_token: str,
        fingerprint_cookie: Optional[str],
    ) -> ValidationResult:
        """
        Validate JWT and verify fingerprint binding.

        Args:
            jwt_token: The JWT to validate
            fingerprint_cookie: The fingerprint from HttpOnly cookie

        Returns:
            ValidationResult with validity status and extracted claims
        """
        try:
            # Decode with all validations (RFC 8725)
            payload_dict = jwt.decode(
                jwt_token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                audience=self.config.jwt_audience,
                issuer=self.config.jwt_issuer,
                options={
                    "require": ["exp", "iat", "nbf", "iss", "aud", "sub", "jti"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )

            payload = TokenPayload(**payload_dict)

            # Check if JTI has been revoked
            if payload.jti in self._revoked_jtis:
                logger.warning(f"Rejected revoked JWT: jti={payload.jti}")
                return ValidationResult(valid=False, error="Token has been revoked")

            # Check token version (forced revocation)
            current_version = self._get_session_version(payload.sub)
            if payload.ver < current_version:
                logger.warning(
                    f"Rejected outdated token version: "
                    f"token_ver={payload.ver}, current_ver={current_version}"
                )
                return ValidationResult(valid=False, error="Token version outdated")

            # Verify fingerprint binding (OWASP Token Sidejacking Prevention)
            if not fingerprint_cookie:
                logger.warning("JWT validation failed: missing fingerprint cookie")
                return ValidationResult(valid=False, error="Missing fingerprint cookie")

            actual_fingerprint_hash = self._hash_fingerprint(fingerprint_cookie)
            if actual_fingerprint_hash != payload.fph:
                logger.warning(
                    f"JWT validation failed: fingerprint mismatch for session={payload.sub}"
                )
                return ValidationResult(valid=False, error="Fingerprint mismatch")

            logger.debug(f"JWT validated successfully for session={payload.sub}")
            return ValidationResult(
                valid=True,
                session_id=payload.sub,
                project_id=payload.pid,
                token_payload=payload,
            )

        except jwt.ExpiredSignatureError:
            logger.debug("JWT validation failed: token expired")
            return ValidationResult(valid=False, error="Token expired")
        except jwt.InvalidAudienceError:
            logger.warning("JWT validation failed: invalid audience")
            return ValidationResult(valid=False, error="Invalid audience")
        except jwt.InvalidIssuerError:
            logger.warning("JWT validation failed: invalid issuer")
            return ValidationResult(valid=False, error="Invalid issuer")
        except jwt.DecodeError as e:
            logger.warning(f"JWT validation failed: decode error - {e}")
            return ValidationResult(valid=False, error="Invalid token format")
        except Exception as e:
            logger.error(f"JWT validation failed: unexpected error - {e}")
            return ValidationResult(valid=False, error="Validation failed")

    def refresh_tokens(
        self,
        refresh_token: str,
        fingerprint_cookie: Optional[str],
    ) -> Optional[RefreshResult]:
        """
        Refresh session tokens using refresh token.

        Implements Auth0 refresh token rotation:
        - Each refresh token can only be used once
        - Reuse detection indicates potential token theft

        Args:
            refresh_token: The refresh token
            fingerprint_cookie: The fingerprint from HttpOnly cookie

        Returns:
            RefreshResult with new JWT and refresh token, or None if invalid
        """
        token_hash = self._hash_refresh_token(refresh_token)
        state = self._refresh_tokens.get(token_hash)

        if not state:
            logger.warning("Refresh token not found")
            return None

        # Check if already used (reuse detection)
        if state.used:
            logger.warning(
                f"Refresh token reuse detected for session={state.session_id}! "
                f"Potential token theft - revoking all tokens for session"
            )
            self.revoke_session(state.session_id)
            return None

        # Check expiry
        if datetime.now(timezone.utc) > state.expires_at:
            logger.debug(f"Refresh token expired for session={state.session_id}")
            del self._refresh_tokens[token_hash]
            return None

        # Verify fingerprint matches
        if not fingerprint_cookie:
            logger.warning("Refresh failed: missing fingerprint cookie")
            return None

        actual_fingerprint_hash = self._hash_fingerprint(fingerprint_cookie)
        if actual_fingerprint_hash != state.fingerprint_hash:
            logger.warning(
                f"Refresh failed: fingerprint mismatch for session={state.session_id}"
            )
            return None

        # Mark current refresh token as used
        state.used = True

        # Generate new tokens (rotation)
        new_tokens = self.create_session_tokens(
            session_id=state.session_id,
            project_id=state.project_id,
            existing_fingerprint=fingerprint_cookie,  # Reuse same fingerprint
        )

        logger.info(f"Tokens refreshed for session={state.session_id}")

        return RefreshResult(
            jwt_token=new_tokens.jwt_token,
            refresh_token=new_tokens.refresh_token,
            expires_in=new_tokens.expires_in,
        )

    def revoke_session(self, session_id: str) -> None:
        """
        Revoke all tokens for a session.

        Increments the token version, invalidating all existing JWTs.
        Also removes all refresh tokens for the session.
        """
        # Increment version to invalidate existing JWTs
        current_version = self._session_versions.get(session_id, 1)
        self._session_versions[session_id] = current_version + 1

        # Remove all refresh tokens for this session
        to_remove = [
            token_hash
            for token_hash, state in self._refresh_tokens.items()
            if state.session_id == session_id
        ]
        for token_hash in to_remove:
            del self._refresh_tokens[token_hash]

        logger.info(
            f"Revoked session={session_id}, "
            f"new_version={current_version + 1}, "
            f"removed_refresh_tokens={len(to_remove)}"
        )

    def revoke_token(self, jti: str) -> None:
        """Revoke a specific JWT by its JTI."""
        self._revoked_jtis.add(jti)
        logger.info(f"Revoked JWT: jti={jti}")

    def cleanup_expired(self) -> int:
        """
        Clean up expired refresh tokens.

        Returns:
            Number of tokens cleaned up
        """
        now = datetime.now(timezone.utc)
        expired = [
            token_hash
            for token_hash, state in self._refresh_tokens.items()
            if state.expires_at < now
        ]
        for token_hash in expired:
            del self._refresh_tokens[token_hash]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired refresh tokens")

        return len(expired)

    def get_cookie_settings(self) -> dict:
        """Get cookie settings for fingerprint cookie."""
        return {
            "key": self.config.fingerprint_cookie_name,
            "httponly": self.config.fingerprint_cookie_httponly,
            "secure": self.config.fingerprint_cookie_secure,
            "samesite": self.config.fingerprint_cookie_samesite,
            "max_age": self.config.fingerprint_cookie_max_age,
            "path": "/",
        }


# Global instance (initialized by app startup)
_security_manager: Optional[SessionSecurityManager] = None


def get_security_manager() -> SessionSecurityManager:
    """Get the global security manager instance."""
    if _security_manager is None:
        raise RuntimeError("SessionSecurityManager not initialized")
    return _security_manager


def init_security_manager(config: SessionSecurityConfig) -> SessionSecurityManager:
    """Initialize the global security manager."""
    global _security_manager
    _security_manager = SessionSecurityManager(config)
    logger.info("SessionSecurityManager initialized")
    return _security_manager
