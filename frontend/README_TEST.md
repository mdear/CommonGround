# Frontend Testing Guide

This document describes how to run and understand the frontend test suite for the CommonGround session resilience implementation.

## Overview

The frontend test suite uses **Jest** and **React Testing Library** to test the `SessionManager` class and related session resilience functionality. Tests cover:

- **Session Persistence** - sessionStorage operations
- **Token Refresh** - JWT auto-refresh scheduling
- **Client Heartbeat** - Keep-alive message management
- **Reconnection Logic** - Grace period and reconnection detection
- **Session Creation** - JWT token generation and validation
- **Callbacks** - Event handler verification

## Prerequisites

Ensure all dependencies are installed:

```bash
npm install
```

This installs Jest, @testing-library/react, @testing-library/jest-dom, and other testing dependencies defined in `package.json`.

## Running Tests

### Run All Tests Once

```bash
npm test
```

This executes the complete test suite and displays results.

### Watch Mode (Auto-rerun on Changes)

```bash
npm run test:watch
```

Tests automatically re-run when source or test files change. Useful during development.

### Coverage Report

```bash
npm run test:coverage
```

Generates a code coverage report showing which lines/branches are tested.

## Test Files

- **`__tests__/sessionManager.test.ts`** - Complete SessionManager test suite (25 tests)

## Current Test Status

```
✅ 25 tests passing
```

### Test Coverage

All core functionality is verified:

- ✅ Session storage (save, load, clear, update)
- ✅ Token refresh scheduling (90% lifespan timer)
- ✅ Silent token refresh with API calls
- ✅ Heartbeat interval management
- ✅ Heartbeat acknowledgment handling
- ✅ Reconnection status checking
- ✅ WebSocket reconnect message format
- ✅ Session creation and token retrieval
- ✅ Existing session validation
- ✅ Reconnection opportunity detection
- ✅ Callback invocation (session expired, heartbeat failed)

## Test Structure

### Example Test

```typescript
test('saveSession stores tokens in sessionStorage', () => {
  const mockTokens: SessionTokens = {
    session_id: 'test-session-123',
    jwt_token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6InNlc3Npb24rand0In0.test',
    refresh_token: 'refresh-token',
    expires_in: 900,
  };

  manager.saveSession(mockTokens);

  const stored = sessionStorage.getItem('cg_session');
  expect(stored).toBeTruthy();

  const parsed = JSON.parse(stored!);
  expect(parsed.sessionId).toBe('test-session-123');
  expect(parsed.jwtToken).toBe(mockTokens.jwt_token);
});
```

### Mocking Strategy

- **fetch** - Mocked globally with `jest.fn()` to simulate API responses
- **sessionStorage** - Custom mock implementation for storage operations
- **WebSocket** - Mock class for WebSocket message testing
- **Timers** - Uses `jest.useFakeTimers()` for scheduling tests

## Test Configuration

### Jest Config (`jest.config.js`)

- **testEnvironment**: jsdom (browser-like environment)
- **setupFilesAfterEnv**: `jest.setup.ts` (global test setup)
- **moduleNameMapper**: Path aliases and static file mocks
- **transform**: ts-jest for TypeScript compilation

### Setup File (`jest.setup.ts`)

- Imports `@testing-library/jest-dom` matchers
- Mocks Next.js Image component
- Suppresses console warnings during tests

## Expected Console Output

When running tests, you may see expected console.error messages:

```
console.error
  [SessionManager] Refresh request failed: 401
```

These are intentional test scenarios verifying error handling.

## Troubleshooting

### Tests Failing After Code Changes

1. Check if new dependencies need to be installed: `npm install`
2. Clear Jest cache: `npx jest --clearCache`
3. Ensure no TypeScript compilation errors: `npm run build`

### Timeout Errors

If tests timeout, increase the Jest timeout in test files:

```typescript
jest.setTimeout(10000); // 10 seconds
```

### Mock Issues

If mocks aren't working:

1. Verify mock order (mocks must be defined before imports)
2. Check `jest.clearAllMocks()` in `beforeEach()`
3. Ensure proper cleanup in `afterEach()`

## Related Files

- **`lib/sessionManager.ts`** - Implementation under test
- **`app/config.ts`** - Configuration (mocked in tests)
- **`package.json`** - Test scripts and dependencies
- **`jest.config.js`** - Jest configuration
- **`jest.setup.ts`** - Global test setup

## Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Use descriptive test names: `test('what it should do when condition')`
3. Mock external dependencies (fetch, timers, storage)
4. Clean up in `afterEach()` hooks
5. Update this README if adding new test categories

## Next Steps

- Add integration tests for full WebSocket lifecycle
- Add E2E tests for browser reconnection scenarios
- Increase coverage for edge cases
