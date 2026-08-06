// Phase 9 (9d): shared E2E identities.
//
// The deterministic two-owner AUTH_TOKENS map used by BOTH the harness backend
// (fixtures/test-server.ts) and the browser specs. Two distinct owners let the
// unauthorized-cross-owner journey prove ownership isolation end-to-end through
// the REAL server (user2 must NOT reach user1's thread; identical 404 / no leak).
//
// These are OPAQUE test tokens for a local, in-memory auth table — NOT secrets,
// NOT provider credentials. They exist only for the offline E2E harness.

// Owner #1 (the primary-journey user).
export const TOKEN_USER = "e2e-user-token";
export const USER_ID = "e2e-user";

// Owner #2 (used to prove cross-owner ownership isolation).
export const TOKEN_USER2 = "e2e-user2-token";
export const USER2_ID = "e2e-user2";

// An obviously-invalid bearer token (not in the map) for the unauthorized
// invalid-token journey — the app must surface a generic auth failure.
export const TOKEN_INVALID = "definitely-not-a-valid-token";
