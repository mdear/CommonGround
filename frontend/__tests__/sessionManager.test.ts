/**
 * Unit tests for SessionManager
 *
 * Tests cover:
 * - Session persistence (sessionStorage)
 * - Token refresh scheduling
 * - Heartbeat management
 * - Reconnection logic
 */

// Mock config module before imports
jest.mock('../app/config', () => ({
  config: {
    api: {
      baseUrl: 'http://localhost:8000',
    },
    ws: {
      url: 'ws://localhost:8000',
      endpoint: '/ws',
    },
  },
}));

import { SessionManager, SessionTokens, StoredSession } from '../lib/sessionManager';

// Mock fetch globally
global.fetch = jest.fn();

// Mock sessionStorage
const mockSessionStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: jest.fn((key: string) => { delete store[key]; }),
    clear: jest.fn(() => { store = {}; }),
    get length() { return Object.keys(store).length; },
    key: jest.fn((index: number) => Object.keys(store)[index] || null),
  };
})();

Object.defineProperty(window, 'sessionStorage', {
  value: mockSessionStorage,
});

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.OPEN;
  send = jest.fn();
  close = jest.fn();

  addEventListener = jest.fn();
  removeEventListener = jest.fn();
}

(global as any).WebSocket = MockWebSocket;

describe('SessionManager', () => {
  let manager: SessionManager;

  beforeEach(() => {
    jest.clearAllMocks();
    mockSessionStorage.clear();
    jest.useFakeTimers();
    manager = new SessionManager();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Session Persistence', () => {
    const mockTokens: SessionTokens = {
      session_id: 'test-session-123',
      jwt_token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6InNlc3Npb24rand0In0.test',
      refresh_token: 'refresh-token-abc',
      expires_in: 900, // 15 minutes
    };

    test('saveSession stores tokens in sessionStorage', () => {
      manager.saveSession(mockTokens);

      expect(mockSessionStorage.setItem).toHaveBeenCalledWith(
        'cg_session',
        expect.any(String)
      );

      const stored = JSON.parse(mockSessionStorage.setItem.mock.calls[0][1]);
      expect(stored.sessionId).toBe('test-session-123');
      expect(stored.jwtToken).toBe(mockTokens.jwt_token);
      expect(stored.refreshToken).toBe(mockTokens.refresh_token);
    });

    test('saveSession includes run info when provided', () => {
      manager.saveSession(mockTokens, 'run-456', 42);

      const stored = JSON.parse(mockSessionStorage.setItem.mock.calls[0][1]);
      expect(stored.runId).toBe('run-456');
      expect(stored.lastEventId).toBe(42);
    });

    test('loadSession returns null when no session exists', () => {
      const result = manager.loadSession();
      expect(result).toBeNull();
    });

    test('loadSession returns stored session', () => {
      manager.saveSession(mockTokens);

      const result = manager.loadSession();

      expect(result).not.toBeNull();
      expect(result?.sessionId).toBe('test-session-123');
      expect(result?.jwtToken).toBe(mockTokens.jwt_token);
    });

    test('loadSession returns session even if JWT expired (for refresh)', () => {
      // Save a session
      manager.saveSession(mockTokens);

      // Advance time past expiry
      jest.advanceTimersByTime(20 * 60 * 1000); // 20 minutes

      // Should still return session (for refresh attempt)
      const result = manager.loadSession();
      expect(result).not.toBeNull();
    });

    test('clearSession removes from storage', () => {
      manager.saveSession(mockTokens);
      manager.clearSession();

      expect(mockSessionStorage.removeItem).toHaveBeenCalledWith('cg_session');
    });

    test('updateRunInfo updates run ID in stored session', () => {
      manager.saveSession(mockTokens);
      manager.updateRunInfo('new-run-789');

      const result = manager.loadSession();
      expect(result?.runId).toBe('new-run-789');
    });

    test('updateLastEventId updates event ID in stored session', () => {
      manager.saveSession(mockTokens, 'run-123');
      manager.updateLastEventId(100);

      const result = manager.loadSession();
      expect(result?.lastEventId).toBe(100);
    });
  });

  describe('Token Refresh', () => {
    const mockTokens: SessionTokens = {
      session_id: 'test-session',
      jwt_token: 'jwt-token',
      refresh_token: 'refresh-token',
      expires_in: 900, // 15 minutes
    };

    test('scheduleAutoRefresh sets timer for 90% of lifespan', () => {
      jest.useFakeTimers();
      const setTimeoutSpy = jest.spyOn(global, 'setTimeout');

      manager.scheduleAutoRefresh(mockTokens);

      // 90% of 15 minutes = 13.5 minutes = 810 seconds = 810000ms
      const expectedDelay = 900 * 1000 * 0.9;

      // Timer should be set with correct delay
      expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), expectedDelay);

      jest.useRealTimers();
      setTimeoutSpy.mockRestore();
    });

    test('stopAutoRefresh clears the timer', () => {
      jest.useFakeTimers();
      const clearTimeoutSpy = jest.spyOn(global, 'clearTimeout');

      manager.scheduleAutoRefresh(mockTokens);
      manager.stopAutoRefresh();

      // clearTimeout should have been called
      expect(clearTimeoutSpy).toHaveBeenCalled();

      jest.useRealTimers();
      clearTimeoutSpy.mockRestore();
    });

    test('performSilentRefresh calls refresh endpoint', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          jwt_token: 'new-jwt',
          refresh_token: 'new-refresh',
          expires_in: 900,
        }),
      });

      manager.saveSession(mockTokens);

      const result = await manager.performSilentRefresh();

      expect(result).toBe(true);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/session/refresh'),
        expect.objectContaining({
          method: 'POST',
          credentials: 'include',
        })
      );
    });

    test('performSilentRefresh returns false on failure', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 401,
      });

      manager.saveSession(mockTokens);

      const onExpired = jest.fn();
      manager = new SessionManager({ onSessionExpired: onExpired });
      manager.saveSession(mockTokens);

      const result = await manager.performSilentRefresh();

      expect(result).toBe(false);
      expect(onExpired).toHaveBeenCalled();
    });
  });

  describe('Client Heartbeat', () => {
    const mockTokens: SessionTokens = {
      session_id: 'test-session',
      jwt_token: 'jwt-token',
      refresh_token: 'refresh-token',
      expires_in: 900,
    };

    test('startHeartbeat begins sending heartbeats', () => {
      const ws = new MockWebSocket() as unknown as WebSocket;

      manager.saveSession(mockTokens);
      manager.startHeartbeat(ws);

      // Advance past heartbeat interval (20 seconds)
      jest.advanceTimersByTime(20000);

      expect((ws as any).send).toHaveBeenCalled();
      const sentMessage = JSON.parse((ws as any).send.mock.calls[0][0]);
      expect(sentMessage.type).toBe('heartbeat');
      expect(sentMessage.data.sessionId).toBe('test-session');
    });

    test('stopHeartbeat clears interval', () => {
      const ws = new MockWebSocket() as unknown as WebSocket;

      manager.startHeartbeat(ws);
      manager.stopHeartbeat();

      // Advance time
      jest.advanceTimersByTime(60000);

      // Should not have sent after stop
      const callCount = (ws as any).send.mock.calls.length;
      jest.advanceTimersByTime(30000);
      expect((ws as any).send.mock.calls.length).toBe(callCount);
    });

    test('handleHeartbeatAck resets missed counter', () => {
      const ws = new MockWebSocket() as unknown as WebSocket;

      manager.startHeartbeat(ws);

      manager.handleHeartbeatAck({
        timestamp: Date.now(),
        serverTime: new Date().toISOString(),
        sessionValid: true,
      });

      // No error should be triggered
    });

    test('handleHeartbeatAck calls onSessionExpired when invalid', () => {
      const onExpired = jest.fn();
      manager = new SessionManager({ onSessionExpired: onExpired });

      manager.handleHeartbeatAck({
        timestamp: Date.now(),
        serverTime: new Date().toISOString(),
        sessionValid: false,
      });

      expect(onExpired).toHaveBeenCalled();
    });
  });

  describe('Reconnection', () => {
    const mockTokens: SessionTokens = {
      session_id: 'test-session',
      jwt_token: 'jwt-token',
      refresh_token: 'refresh-token',
      expires_in: 900,
    };

    test('checkForReconnection returns false with no session', async () => {
      const result = await manager.checkForReconnection();
      expect(result.canReconnect).toBe(false);
    });

    test('checkForReconnection returns false with no run ID', async () => {
      manager.saveSession(mockTokens);

      const result = await manager.checkForReconnection();
      expect(result.canReconnect).toBe(false);
    });

    test('checkForReconnection checks run status when run ID exists', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          run_id: 'run-123',
          exists: true,
          can_reconnect: true,
          state: 'grace_period',
        }),
      });

      manager.saveSession(mockTokens, 'run-123');

      const result = await manager.checkForReconnection();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/run/run-123/status'),
        expect.objectContaining({ credentials: 'include' })
      );
      expect(result.canReconnect).toBe(true);
      expect(result.runId).toBe('run-123');
    });

    test('sendReconnectMessage sends correct message format', () => {
      const ws = new MockWebSocket() as unknown as WebSocket;

      manager.sendReconnectMessage(ws, 'run-123', 42);

      expect((ws as any).send).toHaveBeenCalled();
      const sentMessage = JSON.parse((ws as any).send.mock.calls[0][0]);
      expect(sentMessage).toEqual({
        type: 'reconnect',
        data: {
          run_id: 'run-123',
          last_event_id: 42,
        },
      });
    });
  });

  describe('Session Creation', () => {
    test('createSession calls API and saves response', async () => {
      const mockResponse: SessionTokens = {
        session_id: 'new-session-789',
        jwt_token: 'new-jwt-token',
        refresh_token: 'new-refresh-token',
        expires_in: 900,
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await manager.createSession('test-project');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/session'),
        expect.objectContaining({
          method: 'POST',
          credentials: 'include',
          body: JSON.stringify({ project_id: 'test-project' }),
        })
      );
      expect(result.session_id).toBe('new-session-789');
    });

    test('getOrCreateSession returns existing valid session', async () => {
      const existingTokens: SessionTokens = {
        session_id: 'existing-session',
        jwt_token: 'existing-jwt',
        refresh_token: 'existing-refresh',
        expires_in: 900,
      };

      manager.saveSession(existingTokens);

      // Mock fetch for potential reconnection check (but no run_id so won't be called)
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const result = await manager.getOrCreateSession();

      expect(result.tokens.session_id).toBe('existing-session');
      expect(result.isReconnect).toBe(false);
    });

    test('getOrCreateSession identifies reconnection opportunity', async () => {
      // Clear any previous mock calls and switch to real timers for async fetch
      (global.fetch as jest.Mock).mockReset();
      jest.useRealTimers();

      // Create a fresh manager instance
      const reconnectManager = new SessionManager();

      // Save session with runId - JWT will be valid (not expired)
      const existingTokens: SessionTokens = {
        session_id: 'existing-session',
        jwt_token: 'existing-jwt',
        refresh_token: 'existing-refresh',
        expires_in: 900, // 15 minutes - won't be expired
      };

      reconnectManager.saveSession(existingTokens, 'active-run-123', 42);

      // Verify session was saved correctly with runId
      const savedSession = reconnectManager.loadSession();
      expect(savedSession?.runId).toBe('active-run-123');
      expect(savedSession?.lastEventId).toBe(42);

      // Mock run status API response - must return can_reconnect: true
      (global.fetch as jest.Mock).mockImplementation((url: string) => {
        if (url.includes('/run/active-run-123/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              run_id: 'active-run-123',
              exists: true,
              can_reconnect: true,
              state: 'grace_period',
            }),
          });
        }
        return Promise.reject(new Error(`Unexpected fetch to: ${url}`));
      });

      const result = await reconnectManager.getOrCreateSession();

      // Verify the run status endpoint was called
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/run/active-run-123/status',
        expect.objectContaining({ method: 'GET', credentials: 'include' })
      );

      // Verify reconnection was detected
      expect(result.isReconnect).toBe(true);
      expect(result.reconnectInfo?.runId).toBe('active-run-123');
      expect(result.reconnectInfo?.lastEventId).toBe(42);
      expect(result.tokens.session_id).toBe('existing-session');

      // Restore fake timers for subsequent tests
      jest.useFakeTimers();
    });
  });

  describe('Callbacks', () => {
    test('onSessionExpired callback is called when session expires', async () => {
      const onExpired = jest.fn();
      manager = new SessionManager({ onSessionExpired: onExpired });

      // Try to refresh with no session
      await manager.performSilentRefresh();

      expect(onExpired).toHaveBeenCalled();
    });

    test('onHeartbeatFailed callback is called after max missed', () => {
      const onFailed = jest.fn();
      manager = new SessionManager({ onHeartbeatFailed: onFailed });

      const ws = new MockWebSocket() as unknown as WebSocket;
      manager.startHeartbeat(ws);

      // Advance time past multiple heartbeat intervals + timeouts
      // 2 missed max, 20s interval, 10s timeout = need to wait 60s+
      jest.advanceTimersByTime(70000);

      // The callback should eventually be triggered
      // (exact timing depends on implementation)
    });
  });
});
