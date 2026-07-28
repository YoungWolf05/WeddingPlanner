import { timingSafeEqual } from "node:crypto";

// Phase 5 (5c): bearer-token -> userId authentication for the HTTP service.
//
// SECURITY MODEL
//   - A request authenticates by presenting `Authorization: Bearer <token>`.
//   - The service resolves that opaque token to an authenticated `userId` using
//     a CONFIGURED token->userId map (the local baseline; full SSO/OIDC is out
//     of scope per the roadmap). The resolved `userId` becomes the `ownerId`
//     passed to the ThreadStore for EVERY thread operation.
//   - The token is a SECRET; the `userId` is an identity. The map is the trust
//     anchor. A user-supplied owner/user id (from body/query/header/path) is
//     NEVER trusted — only the token resolves identity.
//   - GENERIC FAILURE: a missing, malformed, or unknown token all yield the
//     same "unauthenticated" result. We never reveal whether a token was
//     well-formed-but-unknown vs. malformed, so an attacker cannot probe the
//     token space by response differences.
//   - CONSTANT-TIME MATCH: tokens are compared with a length-independent
//     constant-time equality so match timing does not leak how many leading
//     bytes were correct. (The map is small and local; this is defense in depth.)

// An immutable token->userId resolver. Constructed from a plain map so tests can
// inject a deterministic table directly (no process env), and production loads
// the map from config (see parseAuthTokens / config.authTokens).
export interface TokenAuthenticator {
  // Resolve a raw bearer token string to a userId, or null if unknown.
  resolveToken(token: string): string | null;
  // Parse an Authorization header value and resolve it to a userId, or null if
  // absent/malformed/unknown. This is the single entry point the server uses.
  authenticate(authorizationHeader: string | undefined): string | null;
  // Number of configured tokens; used by the entrypoint to warn on an empty map.
  readonly size: number;
}

// Case-insensitive `Bearer <token>` extraction. Returns the raw token, or null
// when the header is absent or not a well-formed Bearer credential. Returning
// null here (rather than distinguishing "no header" from "bad scheme") is
// deliberate: the caller maps every null to the SAME generic 401.
export function extractBearerToken(
  authorizationHeader: string | undefined
): string | null {
  if (typeof authorizationHeader !== "string") return null;
  const match = /^Bearer[ \t]+(\S+)$/i.exec(authorizationHeader.trim());
  return match ? match[1]! : null;
}

// Constant-time string equality. Compares UTF-8 byte buffers; when lengths
// differ it still performs a fixed comparison against `a` so the branch time
// does not leak the length relationship, then returns false.
function constantTimeEquals(a: string, b: string): boolean {
  const bufA = Buffer.from(a, "utf8");
  const bufB = Buffer.from(b, "utf8");
  if (bufA.length !== bufB.length) {
    // Compare against self to spend comparable time, then fail.
    timingSafeEqual(bufA, bufA);
    return false;
  }
  return timingSafeEqual(bufA, bufB);
}

// Build a TokenAuthenticator from a token->userId map. The map is copied so
// later external mutation cannot change the authenticator's trust table.
//
// Tokens mapping to an empty/blank userId are REJECTED at construction: an empty
// ownerId would be a security footgun (it could collide across principals), so
// we fail fast rather than silently authenticate to "".
export function createTokenAuthenticator(
  tokenToUserId: Readonly<Record<string, string>>
): TokenAuthenticator {
  const entries: Array<[token: string, userId: string]> = [];
  for (const [token, userId] of Object.entries(tokenToUserId)) {
    if (token.length === 0) {
      throw new Error("Auth config contains an empty token; tokens must be non-empty.");
    }
    if (typeof userId !== "string" || userId.trim() === "") {
      throw new Error(
        `Auth config maps a token to an empty userId; every token must map to a non-empty userId.`
      );
    }
    entries.push([token, userId]);
  }

  function resolveToken(token: string): string | null {
    // Linear scan with constant-time compare per entry. The table is small
    // (local baseline); this avoids a hash-map lookup whose timing could, in
    // principle, correlate with token contents.
    let resolved: string | null = null;
    for (const [candidate, userId] of entries) {
      if (constantTimeEquals(token, candidate)) {
        resolved = userId;
      }
    }
    return resolved;
  }

  return {
    resolveToken,
    authenticate(authorizationHeader) {
      const token = extractBearerToken(authorizationHeader);
      if (token === null) return null;
      return resolveToken(token);
    },
    get size() {
      return entries.length;
    },
  };
}

// Parse the AUTH_TOKENS env value into a token->userId map. Two accepted
// formats, documented in .env.example:
//   1. JSON object: {"token-abc":"user-alice","token-def":"user-bob"}
//   2. Comma-separated pairs: token-abc:user-alice,token-def:user-bob
// A blank/undefined value yields an EMPTY map (no tokens configured). Malformed
// input throws so a misconfiguration fails loudly at startup rather than
// silently disabling auth.
export function parseAuthTokens(raw: string | undefined): Record<string, string> {
  const trimmed = raw?.trim();
  if (!trimmed) return {};

  // JSON object form.
  if (trimmed.startsWith("{")) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      throw new Error(
        "AUTH_TOKENS looks like JSON but failed to parse. Expected a JSON object " +
          'mapping tokens to user ids, e.g. {"token-abc":"user-alice"}.'
      );
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error(
        "AUTH_TOKENS JSON must be an object mapping token strings to user id strings."
      );
    }
    const out: Record<string, string> = {};
    for (const [token, userId] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof userId !== "string") {
        throw new Error(
          `AUTH_TOKENS: user id for a token must be a string (got ${typeof userId}).`
        );
      }
      out[token] = userId;
    }
    return out;
  }

  // Comma-separated `token:userId` pairs.
  const out: Record<string, string> = {};
  for (const pair of trimmed.split(",")) {
    const item = pair.trim();
    if (item === "") continue;
    const idx = item.indexOf(":");
    if (idx <= 0 || idx === item.length - 1) {
      throw new Error(
        'AUTH_TOKENS pair is malformed. Use "token:userId" pairs separated by commas, ' +
          'e.g. token-abc:user-alice,token-def:user-bob.'
      );
    }
    const token = item.slice(0, idx).trim();
    const userId = item.slice(idx + 1).trim();
    if (token === "" || userId === "") {
      throw new Error(
        'AUTH_TOKENS pair has an empty token or userId. Use "token:userId".'
      );
    }
    out[token] = userId;
  }
  return out;
}
