/**
 * Session Manager for WebSocket Session Resilience
 *
 * Implements:
 * - JWT token lifecycle management with auto-refresh at 90% of lifespan
 * - sessionStorage persistence for session survival across page refresh
 * - Dual heartbeat support (server-initiated ping/pong + client-initiated heartbeat)
 * - Reconnection flow for resuming active runs
 *
 * Security:
 * - JWT tokens stored in sessionStorage (cleared on browser close)
 * - HttpOnly fingerprint cookie for token binding (handled by browser automatically)
 * - Token rotation on each refresh (single-use refresh tokens)
 *
 * See docs/architecture/session-resilience.md for full design documentation.
 */

import { config } from '../app/config';

// =============================================================================
// Types
// =============================================================================

export interface SessionTokens {
  session_id: string;
  jwt_token: string;
  refresh_token: string;
  expires_in: number; // seconds
}

export interface StoredSession {
  sessionId: string;
  jwtToken: string;
  refreshToken: string;
  expiresAt: number; // timestamp ms
  issuedAt: number; // timestamp ms
  runId?: string;
  lastEventId?: number;
}

export interface RefreshResult {
  jwt_token: string;
  refresh_token: string;
  expires_in: number;
}

export interface RunStatus {
  run_id: string;
  exists: boolean;
  state?: string;
  can_reconnect: boolean;
  grace_period_expires?: string;
  buffered_events?: number;
  message?: string;
}

export interface HeartbeatConfig {
  intervalMs: number;
  timeoutMs: number;
  maxMissed: number;
}

// =============================================================================
// Storage Keys
// =============================================================================

const STORAGE_KEY = 'cg_session';
const AUTO_REFRESH_THRESHOLD = 0.9; // Refresh at 90% of lifespan

// =============================================================================
// Session Manager Class
// =============================================================================

export class SessionManager {
  private refreshTimerId: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimerId: ReturnType<typeof setInterval> | null = null;
  private missedHeartbeats = 0;
  private lastHeartbeatAck: number | null = null;
  private websocket: WebSocket | null = null;

  private heartbeatConfig: HeartbeatConfig = {
    intervalMs: 20000, // 20 seconds (matches backend CLIENT_HEARTBEAT_INTERVAL_SECONDS)
    timeoutMs: 10000, // 10 seconds
    maxMissed: 2,
  };

  private onSessionExpired?: () => void;
  private onReconnectNeeded?: (runId: string) => void;
  private onHeartbeatFailed?: () => void;

  constructor(callbacks?: {
    onSessionExpired?: () => void;
    onReconnectNeeded?: (runId: string) => void;
    onHeartbeatFailed?: () => void;
  }) {
    this.onSessionExpired = callbacks?.onSessionExpired;
    this.onReconnectNeeded = callbacks?.onReconnectNeeded;
    this.onHeartbeatFailed = callbacks?.onHeartbeatFailed;
  }

  // ---------------------------------------------------------------------------
  // Session Persistence
  // ---------------------------------------------------------------------------

  /**
   * Save session to sessionStorage.
   */
  saveSession(tokens: SessionTokens, runId?: string, lastEventId?: number): void {
    const now = Date.now();
    const session: StoredSession = {
      sessionId: tokens.session_id,
      jwtToken: tokens.jwt_token,
      refreshToken: tokens.refresh_token,
      expiresAt: now + tokens.expires_in * 1000,
      issuedAt: now,
      runId,
      lastEventId,
    };

    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
      console.log('[SessionManager] Session saved to storage', {
        sessionId: tokens.session_id,
        expiresIn: tokens.expires_in,
      });
    } catch (e) {
      console.error('[SessionManager] Failed to save session:', e);
    }
  }

  /**
   * Load session from sessionStorage.
   */
  loadSession(): StoredSession | null {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (!stored) return null;

      const session: StoredSession = JSON.parse(stored);

      // Check if JWT is expired (with small buffer)
      if (Date.now() >= session.expiresAt - 5000) {
        console.log('[SessionManager] Stored session JWT expired');
        // Don't clear - we might be able to refresh
        return session;
      }

      return session;
    } catch (e) {
      console.error('[SessionManager] Failed to load session:', e);
      return null;
    }
  }

  /**
   * Update run info in stored session.
   */
  updateRunInfo(runId: string, lastEventId?: number): void {
    const session = this.loadSession();
    if (session) {
      session.runId = runId;
      if (lastEventId !== undefined) {
        session.lastEventId = lastEventId;
      }
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
      } catch (e) {
        console.error('[SessionManager] Failed to update run info:', e);
      }
    }
  }

  /**
   * Update last event ID in stored session.
   */
  updateLastEventId(lastEventId: number): void {
    const session = this.loadSession();
    if (session) {
      session.lastEventId = lastEventId;
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
      } catch (e) {
        console.error('[SessionManager] Failed to update lastEventId:', e);
      }
    }
  }

  /**
   * Clear stored session.
   */
  clearSession(): void {
    try {
      sessionStorage.removeItem(STORAGE_KEY);
      console.log('[SessionManager] Session cleared from storage');
    } catch (e) {
      console.error('[SessionManager] Failed to clear session:', e);
    }

    this.stopAutoRefresh();
    this.stopHeartbeat();
  }

  // ---------------------------------------------------------------------------
  // Token Refresh
  // ---------------------------------------------------------------------------

  /**
   * Schedule auto-refresh at 90% of token lifespan.
   */
  scheduleAutoRefresh(tokens: SessionTokens): void {
    this.stopAutoRefresh();

    const lifespanMs = tokens.expires_in * 1000;
    const refreshDelayMs = lifespanMs * AUTO_REFRESH_THRESHOLD;

    this.refreshTimerId = setTimeout(async () => {
      console.log('[SessionManager] Auto-refresh triggered at 90% lifespan');
      await this.performSilentRefresh();
    }, refreshDelayMs);

    console.log('[SessionManager] Auto-refresh scheduled', {
      expiresIn: tokens.expires_in,
      refreshIn: refreshDelayMs / 1000,
    });
  }

  /**
   * Stop auto-refresh timer.
   */
  stopAutoRefresh(): void {
    if (this.refreshTimerId) {
      clearTimeout(this.refreshTimerId);
      this.refreshTimerId = null;
    }
  }

  /**
   * Perform silent token refresh.
   */
  async performSilentRefresh(): Promise<boolean> {
    const session = this.loadSession();
    if (!session?.refreshToken) {
      console.warn('[SessionManager] No refresh token available');
      this.onSessionExpired?.();
      return false;
    }

    try {
      const result = await this.refreshTokens(session.refreshToken);

      if (result) {
        // Update stored session with new tokens
        const updatedTokens: SessionTokens = {
          session_id: session.sessionId,
          jwt_token: result.jwt_token,
          refresh_token: result.refresh_token,
          expires_in: result.expires_in,
        };

        this.saveSession(updatedTokens, session.runId, session.lastEventId);
        this.scheduleAutoRefresh(updatedTokens);

        console.log('[SessionManager] Silent refresh successful');
        return true;
      }
    } catch (e) {
      console.error('[SessionManager] Silent refresh failed:', e);
    }

    this.onSessionExpired?.();
    return false;
  }

  /**
   * Call refresh endpoint.
   */
  private async refreshTokens(refreshToken: string): Promise<RefreshResult | null> {
    const apiBase = config.api.baseUrl;

    const response = await fetch(`${apiBase}/session/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include', // Include cookies (fingerprint)
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      console.error('[SessionManager] Refresh request failed:', response.status);
      return null;
    }

    return response.json();
  }

  // ---------------------------------------------------------------------------
  // Client Heartbeat
  // ---------------------------------------------------------------------------

  /**
   * Start client heartbeat sender.
   */
  startHeartbeat(websocket: WebSocket): void {
    this.stopHeartbeat();
    this.websocket = websocket;
    this.missedHeartbeats = 0;
    this.lastHeartbeatAck = Date.now();

    this.heartbeatTimerId = setInterval(() => {
      this.sendHeartbeat();
    }, this.heartbeatConfig.intervalMs);

    console.log('[SessionManager] Client heartbeat started', {
      interval: this.heartbeatConfig.intervalMs,
    });
  }

  /**
   * Stop client heartbeat sender.
   */
  stopHeartbeat(): void {
    if (this.heartbeatTimerId) {
      clearInterval(this.heartbeatTimerId);
      this.heartbeatTimerId = null;
    }
    this.websocket = null;
  }

  /**
   * Send heartbeat to server.
   */
  private sendHeartbeat(): void {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      return;
    }

    const session = this.loadSession();
    const now = Date.now();

    // Check if we've missed heartbeats
    if (this.lastHeartbeatAck) {
      const timeSinceAck = now - this.lastHeartbeatAck;
      if (timeSinceAck > this.heartbeatConfig.intervalMs + this.heartbeatConfig.timeoutMs) {
        this.missedHeartbeats++;
        console.warn('[SessionManager] Heartbeat ack missed', {
          missed: this.missedHeartbeats,
          timeSinceAck,
        });

        if (this.missedHeartbeats >= this.heartbeatConfig.maxMissed) {
          console.error('[SessionManager] Max heartbeats missed - triggering reconnection');
          this.onHeartbeatFailed?.();
          return;
        }
      }
    }

    const heartbeat = {
      type: 'heartbeat',
      timestamp: now,
      sessionId: session?.sessionId,
      runId: session?.runId,
    };

    try {
      this.websocket.send(JSON.stringify(heartbeat));
    } catch (e) {
      console.error('[SessionManager] Failed to send heartbeat:', e);
    }
  }

  /**
   * Handle heartbeat acknowledgment from server.
   */
  handleHeartbeatAck(ack: { timestamp: number; serverTime: string; sessionValid: boolean }): void {
    this.lastHeartbeatAck = Date.now();
    this.missedHeartbeats = 0;

    if (!ack.sessionValid) {
      console.warn('[SessionManager] Server reports session invalid');
      this.onSessionExpired?.();
    }
  }

  // ---------------------------------------------------------------------------
  // Reconnection
  // ---------------------------------------------------------------------------

  /**
   * Check if there's a session that can be reconnected.
   */
  async checkForReconnection(): Promise<{ canReconnect: boolean; runId?: string; runStatus?: RunStatus }> {
    const session = this.loadSession();

    if (!session?.runId) {
      return { canReconnect: false };
    }

    try {
      const runStatus = await this.getRunStatus(session.runId);

      if (runStatus.can_reconnect) {
        return {
          canReconnect: true,
          runId: session.runId,
          runStatus,
        };
      }

      return { canReconnect: false, runStatus };
    } catch (e) {
      console.error('[SessionManager] Failed to check run status:', e);
      return { canReconnect: false };
    }
  }

  /**
   * Get run connection status from server.
   */
  async getRunStatus(runId: string): Promise<RunStatus> {
    const apiBase = config.api.baseUrl;

    const response = await fetch(`${apiBase}/run/${runId}/status`, {
      method: 'GET',
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`Failed to get run status: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Send reconnect message over WebSocket.
   */
  sendReconnectMessage(websocket: WebSocket, runId: string, lastEventId?: number): void {
    if (!runId || runId.trim() === '') {
      console.error('[SessionManager] Cannot send reconnect message: runId is empty');
      return;
    }

    const message = {
      type: 'reconnect',
      run_id: runId,
      last_event_id: lastEventId ?? 0,
    };

    websocket.send(JSON.stringify(message));
    console.log('[SessionManager] Reconnect message sent', { runId, lastEventId });
  }

  // ---------------------------------------------------------------------------
  // Session Creation
  // ---------------------------------------------------------------------------

  /**
   * Create a new session.
   */
  async createSession(projectId: string = 'default'): Promise<SessionTokens> {
    const apiBase = config.api.baseUrl;

    const response = await fetch(`${apiBase}/session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include', // Include cookies for fingerprint
      body: JSON.stringify({ project_id: projectId }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.status}`);
    }

    const tokens: SessionTokens = await response.json();

    // Save to storage and schedule refresh
    this.saveSession(tokens);
    this.scheduleAutoRefresh(tokens);

    return tokens;
  }

  /**
   * Get or create a session, checking for reconnection possibilities.
   * 
   * IMPORTANT: We always create a NEW session_id for WebSocket connection,
   * because the backend removes session_ids from pending_websocket_sessions
   * after the first WebSocket connection. But we preserve runId/lastEventId
   * from the old session for reconnection purposes.
   */
  async getOrCreateSession(projectId: string = 'default'): Promise<{
    tokens: SessionTokens;
    isReconnect: boolean;
    reconnectInfo?: { runId: string; lastEventId?: number };
  }> {
    // Check for existing session to get reconnection info
    const existingSession = this.loadSession();
    let reconnectInfo: { runId: string; lastEventId?: number } | undefined;

    if (existingSession?.runId) {
      // Check if we can reconnect to the existing run
      try {
        const { canReconnect } = await this.checkForReconnection();
        if (canReconnect) {
          reconnectInfo = {
            runId: existingSession.runId,
            lastEventId: existingSession.lastEventId,
          };
          console.log('[SessionManager] Found reconnectable run:', reconnectInfo);
        }
      } catch (e) {
        console.warn('[SessionManager] Failed to check reconnection:', e);
      }
    }

    // Always create a new session for WebSocket connection
    // The backend requires a fresh session_id in pending_websocket_sessions
    const tokens = await this.createSession(projectId);
    
    // Preserve run info in the new session for reconnection
    if (reconnectInfo) {
      this.updateRunInfo(reconnectInfo.runId, reconnectInfo.lastEventId);
    }

    return {
      tokens,
      isReconnect: !!reconnectInfo,
      reconnectInfo,
    };
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let sessionManagerInstance: SessionManager | null = null;

export function getSessionManager(callbacks?: {
  onSessionExpired?: () => void;
  onReconnectNeeded?: (runId: string) => void;
  onHeartbeatFailed?: () => void;
}): SessionManager {
  if (!sessionManagerInstance) {
    sessionManagerInstance = new SessionManager(callbacks);
  }
  return sessionManagerInstance;
}

export function resetSessionManager(): void {
  if (sessionManagerInstance) {
    sessionManagerInstance.clearSession();
    sessionManagerInstance = null;
  }
}
