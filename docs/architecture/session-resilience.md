# Session Resilience Architecture

## Overview

This document describes the WebSocket session resilience architecture for CommonGround, providing:
- **JWT-based session authentication** with HttpOnly cookie fingerprint binding
- **Dual heartbeat mechanism** (server-initiated ping + client-initiated heartbeat)
- **Reconnection grace period** for surviving temporary disconnects
- **Event buffering** for replay on reconnection
- **Auto-refresh tokens** at 90% of JWT lifespan

## Design Goals

1. **Session survives browser refresh** - User doesn't lose work when refreshing
2. **Quick reconnection** - Detect and recover from transient network issues fast
3. **Security** - JWT + fingerprint prevents token theft/replay
4. **Standards compliance** - OWASP JWT guidelines, RFC 8725

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SESSION RESILIENCE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          BROWSER                                      │   │
│  │                                                                       │   │
│  │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │   │
│  │  │  sessionStorage  │   │  HttpOnly Cookie │   │  SessionManager  │  │   │
│  │  │                  │   │  (Hardened)      │   │                  │  │   │
│  │  │  • JWT Token     │   │                  │   │  • Auto-refresh  │  │   │
│  │  │  • Refresh Token │   │  • Fingerprint   │   │    at 90%        │  │   │
│  │  │  • Session ID    │   │    (random str)  │   │  • Heartbeat     │  │   │
│  │  │  • Run ID        │   │                  │   │    sender        │  │   │
│  │  │  • Last Event ID │   │  (NOT accessible │   │  • Reconnection  │  │   │
│  │  │                  │   │   to JavaScript) │   │    handler       │  │   │
│  │  └──────────────────┘   └──────────────────┘   └──────────────────┘  │   │
│  │                                                                       │   │
│  └───────────────────────────────────┬───────────────────────────────────┘   │
│                                      │                                       │
│                    WebSocket + JWT Header + Cookie                           │
│                                      │                                       │
│  ┌───────────────────────────────────▼───────────────────────────────────┐   │
│  │                          SERVER                                        │   │
│  │                                                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │                 SessionSecurityManager                           │  │   │
│  │  │                                                                  │  │   │
│  │  │  • create_session_tokens(session_id, project_id)                 │  │   │
│  │  │  • validate_jwt_with_fingerprint(jwt, cookie)                    │  │   │
│  │  │  • refresh_tokens(refresh_token, fingerprint)                    │  │   │
│  │  │  • revoke_session(session_id)                                    │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │                 ConnectionManager (existing)                     │  │   │
│  │  │                                                                  │  │   │
│  │  │  • register_connection() / unregister_connection()               │  │   │
│  │  │  • reconnect_run() ← Now wired up                               │  │   │
│  │  │  • start_heartbeat() (server → client)                          │  │   │
│  │  │  • handle_client_heartbeat() ← NEW                              │  │   │
│  │  │  • buffer_event() / replay_buffered_events()                     │  │   │
│  │  │  • grace_period_monitor()                                        │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                        │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Dual Heartbeat Mechanism

Both server-initiated and client-initiated heartbeats operate concurrently for maximum resilience:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DUAL HEARTBEAT MECHANISM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SERVER-INITIATED (Existing - Enhanced)                                      │
│  ───────────────────────────────────────                                     │
│                                                                              │
│  Purpose: Detect client disappearance → trigger grace period                 │
│  Interval: Every 30 seconds                                                  │
│  Timeout: 10 seconds for pong response                                       │
│  Max missed: 3 before declaring client dead                                  │
│                                                                              │
│  Server ───── { type: "ping", timestamp: "..." } ────► Client               │
│         ◄──── { type: "pong", timestamp: "...",  ───── Client               │
│                 lastEventId: 42,                                             │
│                 clientTime: 1703779200000 }                                  │
│                                                                              │
│  CLIENT-INITIATED (New)                                                      │
│  ──────────────────────                                                      │
│                                                                              │
│  Purpose: Detect server unresponsiveness → trigger reconnection attempt      │
│  Interval: Every 20 seconds                                                  │
│  Timeout: 10 seconds for ack response                                        │
│  Max missed: 2 before attempting reconnection                                │
│                                                                              │
│  Client ───── { type: "heartbeat",           ────► Server                   │
│                 timestamp: 1703779200000,                                    │
│                 sessionId: "...",                                            │
│                 runId: "..." }                                               │
│         ◄──── { type: "heartbeat_ack",       ───── Server                   │
│                 timestamp: 1703779200000,                                    │
│                 serverTime: "...",                                           │
│                 sessionValid: true }                                         │
│                                                                              │
│  FAILURE DETECTION                                                           │
│  ─────────────────                                                           │
│                                                                              │
│  │ Failure Scenario              │ Server Ping │ Client HB │ Detection │    │
│  │───────────────────────────────│─────────────│───────────│───────────│    │
│  │ Client browser closed         │     ✅      │    ❌     │   Fast    │    │
│  │ Client tab frozen             │     ✅      │    ❌     │   Fast    │    │
│  │ Client network dropped        │     ✅      │    ✅     │   Fast    │    │
│  │ Server process died           │     ❌      │    ✅     │   Fast    │    │
│  │ Server overloaded             │     ❌      │    ✅     │   Fast    │    │
│  │ Network partition             │     ✅      │    ✅     │   Fast    │    │
│  │ TCP zombie connection         │     ✅      │    ✅     │   Fast    │    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## JWT Token Structure

Follows RFC 8725 (JWT Best Current Practices) and OWASP guidelines:

```python
# JWT Claims
{
    # Standard Claims (RFC 7519)
    "iss": "commonground",                    # Issuer
    "sub": "session_abc123",                  # Subject (session ID)
    "aud": "commonground-ws-reconnect",       # Audience - MUST validate
    "exp": 1703779200,                        # Expiry (15 min from now)
    "iat": 1703778300,                        # Issued at
    "nbf": 1703778300,                        # Not before
    "jti": "unique-token-id-xyz",             # JWT ID for revocation

    # Custom Claims
    "pid": "project_456",                     # Project ID
    "ver": 1,                                 # Token version (forced revocation)
    "fph": "sha256-hash-of-fingerprint"       # Fingerprint hash (OWASP)
}

# Header includes explicit typing
{
    "alg": "HS256",
    "typ": "session+jwt"                      # RFC 8725 Section 3.11
}
```

---

## Token Fingerprint Binding (OWASP Token Sidejacking Prevention)

**NOT browser fingerprinting** - this is a server-generated random value:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINGERPRINT BINDING MECHANISM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  On Session Creation:                                                        │
│  ────────────────────                                                        │
│                                                                              │
│  1. Server generates: fingerprint = secrets.token_urlsafe(32)                │
│  2. Server computes:  fingerprint_hash = SHA256(fingerprint)                 │
│  3. Server creates JWT with claim: "fph": fingerprint_hash                   │
│  4. Server sets HttpOnly cookie: __Secure-Fgp = fingerprint                  │
│                                                                              │
│  On Token Validation:                                                        │
│  ────────────────────                                                        │
│                                                                              │
│  1. Extract JWT from Authorization header or query param                     │
│  2. Extract fingerprint from __Secure-Fgp cookie                             │
│  3. Compute: actual_hash = SHA256(cookie_fingerprint)                        │
│  4. Compare: actual_hash == jwt_claims["fph"]                                │
│  5. If mismatch → REJECT (possible token theft)                              │
│                                                                              │
│  Why This Works:                                                             │
│  ───────────────                                                             │
│                                                                              │
│  • XSS can steal JWT from sessionStorage                                     │
│  • XSS CANNOT read HttpOnly cookie                                           │
│  • Stolen JWT is useless without the cookie                                  │
│  • Cookie is automatically sent by browser (same-site)                       │
│                                                                              │
│  Cookie Attributes (Hardened):                                               │
│  ─────────────────────────────                                               │
│                                                                              │
│  Set-Cookie: __Secure-Fgp=<fingerprint>;                                     │
│              HttpOnly;        ← Not accessible to JavaScript                 │
│              Secure;          ← HTTPS only                                   │
│              SameSite=Strict; ← CSRF protection                              │
│              Path=/;                                                         │
│              Max-Age=86400    ← 24 hours (matches refresh token)             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Auto-Refresh at 90% Token Lifespan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TOKEN AUTO-REFRESH TIMELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  JWT Lifespan: 15 minutes                                                    │
│  Refresh Threshold: 90% = 13.5 minutes                                       │
│                                                                              │
│  T=0min        T=13.5min (90%)      T=15min                                  │
│    │               │                   │                                     │
│    ▼               ▼                   ▼                                     │
│    ┌───────────────┬───────────────────┐                                     │
│    │   JWT VALID   │  REFRESH WINDOW   │ EXPIRED                             │
│    │               │                   │                                     │
│    │               │  Auto-refresh     │                                     │
│    │               │  triggered here   │                                     │
│    │               │                   │                                     │
│    └───────────────┴───────────────────┘                                     │
│                                                                              │
│  Client-Side Logic:                                                          │
│  ──────────────────                                                          │
│                                                                              │
│  function scheduleTokenRefresh(jwt) {                                        │
│      const payload = decodeJWT(jwt);                                         │
│      const lifespan = payload.exp - payload.iat;  // 900 seconds             │
│      const refreshAt = payload.iat + (lifespan * 0.9);  // 810 seconds       │
│      const delayMs = (refreshAt - now()) * 1000;                             │
│                                                                              │
│      setTimeout(() => performSilentRefresh(), delayMs);                      │
│  }                                                                           │
│                                                                              │
│  Refresh Endpoint:                                                           │
│  ─────────────────                                                           │
│                                                                              │
│  POST /session/refresh                                                       │
│  Body: { refresh_token: "..." }                                              │
│  Cookie: __Secure-Fgp=<fingerprint> (sent automatically)                     │
│                                                                              │
│  Response: {                                                                 │
│      jwt_token: "<new-jwt>",                                                 │
│      refresh_token: "<new-refresh-token>",  // Rotation!                     │
│      expires_in: 900                                                         │
│  }                                                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Reconnection Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECONNECTION FLOW                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SCENARIO: Browser refresh during active run                                 │
│                                                                              │
│  T=0: User refreshes browser                                                 │
│  ─────────────────────────────                                               │
│                                                                              │
│  Backend:                                                                    │
│  1. WebSocket closes                                                         │
│  2. connection_manager.unregister_connection(session_id)                     │
│  3. RunConnectionState.start_grace_period() → 2 minutes                      │
│  4. Tasks KEEP RUNNING                                                       │
│  5. Events buffered via buffer_event()                                       │
│                                                                              │
│  Frontend (page loads):                                                      │
│  1. Check sessionStorage for existing session                                │
│  2. Found: { jwt, refresh_token, session_id, run_id, lastEventId }           │
│  3. Validate JWT not expired (client-side check)                             │
│                                                                              │
│  T=1-2s: Check run status                                                    │
│  ────────────────────────                                                    │
│                                                                              │
│  GET /run/{run_id}/status                                                    │
│  Response: {                                                                 │
│      exists: true,                                                           │
│      state: "grace_period",                                                  │
│      can_reconnect: true,                                                    │
│      grace_period_expires: "2025-01-01T12:02:00Z",                           │
│      buffered_events: 15                                                     │
│  }                                                                           │
│                                                                              │
│  T=2-3s: Get new session (same fingerprint cookie)                           │
│  ────────────────────────────────────────────────                            │
│                                                                              │
│  POST /session                                                               │
│  Cookie: __Secure-Fgp=<same-fingerprint>                                     │
│  Response: {                                                                 │
│      session_id: "new-session-id",                                           │
│      jwt_token: "<new-jwt>",        // Same fingerprint hash                 │
│      refresh_token: "<new-refresh>"                                          │
│  }                                                                           │
│  Set-Cookie: __Secure-Fgp=<same-value>  // Refresh cookie expiry             │
│                                                                              │
│  T=3-4s: Connect WebSocket and reconnect to run                              │
│  ──────────────────────────────────────────────                              │
│                                                                              │
│  WS /ws/{new-session-id}?token={jwt}                                         │
│  Cookie: __Secure-Fgp=<fingerprint>                                          │
│                                                                              │
│  Client sends:                                                               │
│  {                                                                           │
│      type: "reconnect",                                                      │
│      run_id: "run-123",                                                      │
│      last_event_id: 42                                                       │
│  }                                                                           │
│                                                                              │
│  Server:                                                                     │
│  1. handle_reconnect() validates run can be reconnected                      │
│  2. connection_manager.reconnect_run(run_id, new_session_id, ws, em)         │
│  3. Cancels grace period timer                                               │
│  4. Updates session mapping                                                  │
│  5. Replays buffered events                                                  │
│                                                                              │
│  Server sends:                                                               │
│  {                                                                           │
│      type: "reconnected",                                                    │
│      run_id: "run-123",                                                      │
│      buffered_events: [...],                                                 │
│      run_status: "running"                                                   │
│  }                                                                           │
│                                                                              │
│  T=4s+: Normal operation resumes                                             │
│  ───────────────────────────────                                             │
│                                                                              │
│  • Dual heartbeats restart                                                   │
│  • Token auto-refresh scheduled                                              │
│  • User sees continuous experience                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

```python
# core/api/session_security.py

class SessionSecurityConfig:
    """Configuration for session security."""

    # JWT Settings
    JWT_SECRET: str                           # From environment (required)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 15
    JWT_ISSUER: str = "commonground"
    JWT_AUDIENCE: str = "commonground-ws-reconnect"

    # Refresh Token Settings
    REFRESH_TOKEN_EXPIRY_HOURS: int = 24
    REFRESH_TOKEN_ROTATION: bool = True       # New token on each refresh

    # Auto-Refresh Settings
    AUTO_REFRESH_THRESHOLD: float = 0.9       # Refresh at 90% of lifespan

    # Fingerprint Cookie Settings
    FINGERPRINT_COOKIE_NAME: str = "__Secure-Fgp"
    FINGERPRINT_BYTES: int = 32               # 256 bits of entropy
    FINGERPRINT_COOKIE_SECURE: bool = True
    FINGERPRINT_COOKIE_HTTPONLY: bool = True
    FINGERPRINT_COOKIE_SAMESITE: str = "Strict"
    FINGERPRINT_COOKIE_MAX_AGE: int = 86400   # 24 hours

    # Client Heartbeat Settings (NEW)
    CLIENT_HEARTBEAT_INTERVAL_SECONDS: int = 20
    CLIENT_HEARTBEAT_TIMEOUT_SECONDS: int = 10
    CLIENT_MAX_MISSED_HEARTBEATS: int = 2


# Existing in core/api/connection_manager.py

class ConnectionConfig:
    """Configuration for connection resilience."""

    # Server Heartbeat (ping/pong) - EXISTING
    HEARTBEAT_INTERVAL_SECONDS: float = 30.0
    HEARTBEAT_TIMEOUT_SECONDS: float = 10.0
    MAX_MISSED_HEARTBEATS: int = 3

    # Reconnection - EXISTING
    RECONNECTION_GRACE_PERIOD_SECONDS: float = 120.0

    # Event Buffering - EXISTING
    MAX_BUFFERED_EVENTS: int = 1000
    EVENT_BUFFER_TTL_SECONDS: float = 300.0
```

---

## Message Types

### Server → Client

| Type | Purpose | Payload |
|------|---------|---------|
| `ping` | Server heartbeat | `{ timestamp }` |
| `heartbeat_ack` | Client heartbeat response | `{ timestamp, serverTime, sessionValid }` |
| `connected` | Initial connection confirmed | `{ sessionId }` |
| `reconnected` | Reconnection successful | `{ runId, bufferedEvents, runStatus }` |
| `token_refresh` | New JWT issued | `{ newToken, expiresIn }` |
| `replay_start` | Event replay beginning | `{ runId, eventCount }` |
| `replay_end` | Event replay complete | `{ runId, eventsReplayed }` |
| `session_expired` | Session no longer valid | `{ reason }` |

### Client → Server

| Type | Purpose | Payload |
|------|---------|---------|
| `pong` | Server heartbeat response | `{ timestamp, lastEventId, clientTime }` |
| `heartbeat` | Client heartbeat | `{ timestamp, sessionId, runId }` |
| `reconnect` | Request to reconnect to run | `{ runId, lastEventId }` |
| `start_run` | Start new run (existing) | `{ ... }` |
| `stop_run` | Stop run (existing) | `{ runId }` |

---

## Security Properties

| Property | Mechanism | OWASP/RFC Reference |
|----------|-----------|---------------------|
| Authentication | JWT signature verification | RFC 7519 |
| Authorization | Server-side session lookup | - |
| Replay Prevention | Single-use JTI, rotation | OWASP JWT §Token Sidejacking |
| Token Theft Mitigation | HttpOnly fingerprint cookie | OWASP JWT §Token Sidejacking |
| Algorithm Verification | Explicit alg in decode | RFC 8725 §3.1 |
| Audience Validation | `aud` claim check | RFC 8725 §3.9 |
| Issuer Validation | `iss` claim check | RFC 8725 §3.8 |
| Explicit Typing | `typ: session+jwt` header | RFC 8725 §3.11 |
| Short Expiry | 15 minute JWT | OWASP JWT §Token Storage |
| Forced Revocation | Token version increment | OWASP JWT §Revocation |

---

## File Changes Summary

### New Files

- `core/api/session_security.py` - JWT + fingerprint management
- `frontend/lib/sessionManager.ts` - Token lifecycle + reconnection

### Modified Files

- `core/api/session.py` - Add JWT creation, fingerprint cookie
- `core/api/server.py` - JWT validation, refresh endpoint, reconnect handler
- `core/api/message_handlers.py` - Add `handle_reconnect`, `handle_client_heartbeat`
- `core/api/connection_manager.py` - Add client heartbeat handling
- `frontend/app/stores/sessionStore.ts` - sessionStorage persistence, heartbeat sender
- `frontend/lib/api.ts` - Add `credentials: 'include'`, refresh endpoint
- `core/env.sample` - Add JWT_SECRET, security config

---

## Environment Variables

```bash
# Required
JWT_SECRET=<64+ character random string>

# Optional (with defaults)
JWT_EXPIRY_MINUTES=15
REFRESH_TOKEN_EXPIRY_HOURS=24
CLIENT_HEARTBEAT_INTERVAL_SECONDS=20
```

---

## References

- [OWASP JWT Cheatsheet](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)
- [RFC 8725 - JWT Best Current Practices](https://datatracker.ietf.org/doc/html/rfc8725)
- [Auth0 - Refresh Token Rotation](https://auth0.com/blog/refresh-tokens-what-are-they-and-when-to-use-them/)
