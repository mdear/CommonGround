"""
Unit tests for session security (JWT + fingerprint cookie binding).

Tests cover:
- Token creation and validation
- Fingerprint binding (OWASP Token Sidejacking Prevention)
- Refresh token rotation (Auth0 pattern)
- Token revocation
- Edge cases and error handling
"""
import pytest
import time
import jwt
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
CORE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_DIR))

from api.session_security import (
    SessionSecurityConfig,
    SessionSecurityManager,
    SessionTokens,
    RefreshResult,
    ValidationResult,
    TokenPayload,
    init_security_manager,
    get_security_manager,
)


@pytest.fixture
def security_config():
    """Create a test security configuration."""
    return SessionSecurityConfig(
        jwt_secret="test_secret_key_that_is_at_least_64_characters_long_for_security",
        jwt_expiry_minutes=15,
        refresh_token_expiry_hours=24,
        jwt_issuer="commonground",
        jwt_audience="commonground-ws-reconnect",
        fingerprint_cookie_secure=False,  # Allow HTTP for testing
    )


@pytest.fixture
def security_manager(security_config):
    """Create a SessionSecurityManager for testing."""
    return SessionSecurityManager(security_config)


class TestSessionTokenCreation:
    """Test JWT and session token creation."""

    def test_create_session_tokens_returns_valid_structure(self, security_manager):
        """Test that token creation returns all required fields."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session-123",
            project_id="test-project",
        )

        assert isinstance(tokens, SessionTokens)
        assert tokens.session_id == "test-session-123"
        assert tokens.jwt_token is not None
        assert len(tokens.jwt_token) > 0
        assert tokens.refresh_token is not None
        assert len(tokens.refresh_token) > 0
        assert tokens.fingerprint is not None
        assert len(tokens.fingerprint) > 0
        assert tokens.expires_in == 15 * 60  # 15 minutes in seconds

    def test_jwt_contains_required_claims(self, security_manager, security_config):
        """Test that JWT contains all required RFC 7519 claims."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # Decode without verification to inspect claims
        payload = jwt.decode(
            tokens.jwt_token,
            security_config.jwt_secret,
            algorithms=[security_config.jwt_algorithm],
            audience=security_config.jwt_audience,
        )

        # Standard claims
        assert payload["iss"] == "commonground"
        assert payload["sub"] == "test-session"
        assert payload["aud"] == "commonground-ws-reconnect"
        assert "exp" in payload
        assert "iat" in payload
        assert "nbf" in payload
        assert "jti" in payload

        # Custom claims
        assert payload["pid"] == "test-project"
        assert "ver" in payload
        assert "fph" in payload  # Fingerprint hash

    def test_jwt_header_contains_explicit_type(self, security_manager):
        """Test that JWT header includes typ: session+jwt (RFC 8725 Section 3.11)."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # Get headers without verification
        headers = jwt.get_unverified_header(tokens.jwt_token)
        assert headers.get("typ") == "session+jwt"

    def test_fingerprint_is_cryptographically_random(self, security_manager):
        """Test that fingerprints are unique and properly random."""
        tokens1 = security_manager.create_session_tokens(
            session_id="session-1",
            project_id="project-1",
        )
        tokens2 = security_manager.create_session_tokens(
            session_id="session-2",
            project_id="project-1",
        )

        assert tokens1.fingerprint != tokens2.fingerprint
        assert len(tokens1.fingerprint) >= 32  # At least 32 characters

    def test_existing_fingerprint_reuse_for_reconnection(self, security_manager):
        """Test that existing fingerprint can be reused for reconnection."""
        original_fingerprint = "existing_fingerprint_value"

        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
            existing_fingerprint=original_fingerprint,
        )

        assert tokens.fingerprint == original_fingerprint


class TestJWTValidation:
    """Test JWT validation with fingerprint binding."""

    def test_valid_jwt_with_correct_fingerprint(self, security_manager):
        """Test successful validation with matching fingerprint."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie=tokens.fingerprint,
        )

        assert result.valid is True
        assert result.session_id == "test-session"
        assert result.project_id == "test-project"
        assert result.error is None

    def test_invalid_fingerprint_rejected(self, security_manager):
        """Test that mismatched fingerprint is rejected (OWASP Sidejacking)."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie="wrong_fingerprint_value",
        )

        assert result.valid is False
        assert "Fingerprint mismatch" in result.error

    def test_missing_fingerprint_rejected(self, security_manager):
        """Test that missing fingerprint is rejected."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie=None,
        )

        assert result.valid is False
        assert "Missing fingerprint cookie" in result.error

    def test_expired_jwt_rejected(self, security_manager, security_config):
        """Test that expired JWT is rejected."""
        # Create a token with past expiry
        now = datetime.now(timezone.utc)
        past_exp = int((now - timedelta(hours=1)).timestamp())

        payload = {
            "iss": security_config.jwt_issuer,
            "sub": "test-session",
            "aud": security_config.jwt_audience,
            "exp": past_exp,
            "iat": int((now - timedelta(hours=2)).timestamp()),
            "nbf": int((now - timedelta(hours=2)).timestamp()),
            "jti": "test-jti",
            "pid": "test-project",
            "ver": 1,
            "fph": "test-hash",
        }

        expired_token = jwt.encode(
            payload,
            security_config.jwt_secret,
            algorithm=security_config.jwt_algorithm,
        )

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=expired_token,
            fingerprint_cookie="any",
        )

        assert result.valid is False
        assert "expired" in result.error.lower()

    def test_invalid_signature_rejected(self, security_manager):
        """Test that JWT with invalid signature is rejected."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # Modify the token to invalidate signature
        modified_token = tokens.jwt_token[:-5] + "XXXXX"

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=modified_token,
            fingerprint_cookie=tokens.fingerprint,
        )

        assert result.valid is False

    def test_wrong_audience_rejected(self, security_manager, security_config):
        """Test that JWT with wrong audience is rejected (RFC 8725)."""
        now = datetime.now(timezone.utc)

        payload = {
            "iss": security_config.jwt_issuer,
            "sub": "test-session",
            "aud": "wrong-audience",  # Wrong!
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "jti": "test-jti",
            "pid": "test-project",
            "ver": 1,
            "fph": "test-hash",
        }

        wrong_aud_token = jwt.encode(
            payload,
            security_config.jwt_secret,
            algorithm=security_config.jwt_algorithm,
        )

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=wrong_aud_token,
            fingerprint_cookie="any",
        )

        assert result.valid is False
        assert "audience" in result.error.lower()


class TestRefreshTokenRotation:
    """Test refresh token rotation (Auth0 pattern)."""

    def test_successful_token_refresh(self, security_manager):
        """Test successful refresh returns new tokens."""
        original_tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        refresh_result = security_manager.refresh_tokens(
            refresh_token=original_tokens.refresh_token,
            fingerprint_cookie=original_tokens.fingerprint,
        )

        assert refresh_result is not None
        assert isinstance(refresh_result, RefreshResult)
        assert refresh_result.jwt_token != original_tokens.jwt_token
        assert refresh_result.refresh_token != original_tokens.refresh_token
        assert refresh_result.expires_in > 0

    def test_refresh_token_rotation_single_use(self, security_manager):
        """Test that refresh token can only be used once (rotation)."""
        original_tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # First refresh should succeed
        first_refresh = security_manager.refresh_tokens(
            refresh_token=original_tokens.refresh_token,
            fingerprint_cookie=original_tokens.fingerprint,
        )
        assert first_refresh is not None

        # Second use of same token should fail (reuse detection)
        second_refresh = security_manager.refresh_tokens(
            refresh_token=original_tokens.refresh_token,
            fingerprint_cookie=original_tokens.fingerprint,
        )
        assert second_refresh is None

    def test_refresh_with_wrong_fingerprint_fails(self, security_manager):
        """Test that refresh fails with wrong fingerprint."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        result = security_manager.refresh_tokens(
            refresh_token=tokens.refresh_token,
            fingerprint_cookie="wrong_fingerprint",
        )

        assert result is None

    def test_refresh_with_invalid_token_fails(self, security_manager):
        """Test that refresh fails with invalid token."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        result = security_manager.refresh_tokens(
            refresh_token="invalid_refresh_token",
            fingerprint_cookie=tokens.fingerprint,
        )

        assert result is None


class TestSessionRevocation:
    """Test session and token revocation."""

    def test_revoke_session_invalidates_existing_jwt(self, security_manager):
        """Test that session revocation invalidates all JWTs for that session."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # Verify token is valid before revocation
        pre_revoke = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie=tokens.fingerprint,
        )
        assert pre_revoke.valid is True

        # Revoke the session
        security_manager.revoke_session("test-session")

        # Token should now be invalid (version mismatch)
        post_revoke = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie=tokens.fingerprint,
        )
        assert post_revoke.valid is False
        assert "version" in post_revoke.error.lower()

    def test_revoke_session_removes_refresh_tokens(self, security_manager):
        """Test that session revocation removes all refresh tokens."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # Revoke the session
        security_manager.revoke_session("test-session")

        # Refresh should fail (token removed)
        result = security_manager.refresh_tokens(
            refresh_token=tokens.refresh_token,
            fingerprint_cookie=tokens.fingerprint,
        )
        assert result is None

    def test_revoke_specific_jwt_by_jti(self, security_manager, security_config):
        """Test revoking a specific JWT by its JTI."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # Extract JTI from token
        payload = jwt.decode(
            tokens.jwt_token,
            security_config.jwt_secret,
            algorithms=[security_config.jwt_algorithm],
            audience=security_config.jwt_audience,
        )
        jti = payload["jti"]

        # Revoke by JTI
        security_manager.revoke_token(jti)

        # Token should now be invalid
        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie=tokens.fingerprint,
        )
        assert result.valid is False
        assert "revoked" in result.error.lower()


class TestCookieSettings:
    """Test cookie configuration."""

    def test_cookie_settings_structure(self, security_manager):
        """Test that cookie settings contain required security attributes."""
        settings = security_manager.get_cookie_settings()

        assert "key" in settings
        assert settings["key"] == "__Secure-Fgp"
        assert settings["httponly"] is True
        assert settings["samesite"] == "Strict"
        assert "max_age" in settings
        assert "path" in settings


class TestCleanup:
    """Test cleanup of expired tokens."""

    def test_cleanup_expired_refresh_tokens(self, security_manager):
        """Test that expired refresh tokens are cleaned up."""
        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id="test-project",
        )

        # Manually expire the token
        token_hash = security_manager._hash_refresh_token(tokens.refresh_token)
        security_manager._refresh_tokens[token_hash].expires_at = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        )

        # Run cleanup
        cleaned = security_manager.cleanup_expired()

        assert cleaned == 1
        assert token_hash not in security_manager._refresh_tokens


class TestTokenPayloadModel:
    """Test TokenPayload Pydantic model."""

    def test_token_payload_validation(self):
        """Test that TokenPayload validates correctly."""
        now = int(datetime.now(timezone.utc).timestamp())

        payload = TokenPayload(
            iss="commonground",
            sub="test-session",
            aud="commonground-ws-reconnect",
            exp=now + 900,
            iat=now,
            nbf=now,
            jti="unique-jti",
            pid="test-project",
            ver=1,
            fph="fingerprint-hash",
        )

        assert payload.iss == "commonground"
        assert payload.sub == "test-session"
        assert payload.ver == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_session_id(self, security_manager):
        """Test handling of empty session ID."""
        tokens = security_manager.create_session_tokens(
            session_id="",
            project_id="test-project",
        )

        # Should still work, but session_id will be empty
        assert tokens.session_id == ""

    def test_very_long_project_id(self, security_manager):
        """Test handling of very long project ID."""
        long_project_id = "x" * 1000

        tokens = security_manager.create_session_tokens(
            session_id="test-session",
            project_id=long_project_id,
        )

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie=tokens.fingerprint,
        )

        assert result.valid is True
        assert result.project_id == long_project_id

    def test_unicode_in_ids(self, security_manager):
        """Test handling of unicode characters in IDs."""
        unicode_session = "session-日本語-🎉"
        unicode_project = "project-中文-émoji"

        tokens = security_manager.create_session_tokens(
            session_id=unicode_session,
            project_id=unicode_project,
        )

        result = security_manager.validate_jwt_with_fingerprint(
            jwt_token=tokens.jwt_token,
            fingerprint_cookie=tokens.fingerprint,
        )

        assert result.valid is True
        assert result.session_id == unicode_session
        assert result.project_id == unicode_project
