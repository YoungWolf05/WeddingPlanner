import "dotenv/config";

function required(name: string): string {
  const value = process.env[name];
  if (!value || value.trim() === "") {
    throw new Error(
      `Missing required environment variable: ${name}. ` +
        `Copy .env.example to .env and fill it in.`
    );
  }
  return value.trim();
}

// Phase 7 (7d): the default expected embedding vector dimension when
// LITELLM_EMBED_DIM is unset. 768 matches the verified gemini-embedding-001
// alias (docs/capabilities/2026-07-28.md) and MUST stay in lockstep with
// DEFAULT_EMBEDDING_DIM in src/core/knowledge-store.ts — a coupling asserted by
// an offline test. Kept as a plain literal here (not imported from
// knowledge-store.ts) because knowledge-store.ts imports THIS module, and a
// cyclic import would be fragile.
export const DEFAULT_EMBED_DIM = 768;

// Parse LITELLM_EMBED_DIM into a positive-integer expected embedding dimension.
// This is the SINGLE source of truth for the dimension the knowledge store is
// built with AND the dimension the embedding alias must produce (7d
// compatibility checks). Rules, mirroring createKnowledgeStore's own dimension
// validation:
//   - unset / empty / whitespace-only  -> DEFAULT_EMBED_DIM (768) fallback.
//   - a positive integer               -> that value.
//   - anything else (non-numeric, zero, negative, non-integer) -> FAIL LOUD.
// It fails loud (rather than silently falling back) on a present-but-invalid
// value so an operator typo can never silently build the store at the wrong
// dimension; this matches the `required()` fail-loud spirit and the store's
// "must be a positive integer" guard. Exported pure so it is unit-testable
// without mutating process.env.
export function parseEmbedDim(raw: string | undefined): number {
  const trimmed = raw?.trim();
  if (!trimmed) return DEFAULT_EMBED_DIM;
  // Accept ONLY a plain decimal non-negative integer string. `Number()` would
  // otherwise treat hex ("0x10"), exponent ("1e3"), and other non-decimal forms
  // as valid, contradicting the "must be a positive integer" contract below.
  if (!/^\d+$/.test(trimmed)) {
    throw new Error(
      `Invalid LITELLM_EMBED_DIM "${raw}": must be a positive integer ` +
        `(the expected embedding vector dimension).`
    );
  }
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(
      `Invalid LITELLM_EMBED_DIM "${raw}": must be a positive integer ` +
        `(the expected embedding vector dimension).`
    );
  }
  return parsed;
}

export const config = {
  baseURL: required("LITELLM_BASE_URL"),
  apiKey: required("LITELLM_API_KEY"),
  model: process.env.LITELLM_MODEL?.trim() || "claude-sonnet-4-6",
  // Optional embedding alias. Embeddings are a SEPARATE contract from chat
  // aliases (see docs/roadmap.md Phase 4). When unset, the capability probe
  // records embeddings as "N/A — no embedding alias configured" rather than
  // failing; it never assumes a chat alias supports embeddings.
  embedModel: process.env.LITELLM_EMBED_MODEL?.trim() || undefined,
  // Phase 7 (7d): the EXPECTED embedding vector dimension — the SINGLE source of
  // truth read by the knowledge store (its recorded/built dimension), the 7d
  // compatibility expectation, and the live embedding probe. Sourced from
  // LITELLM_EMBED_DIM; defaults to DEFAULT_EMBED_DIM (768) when unset, and fails
  // loud on a present-but-invalid value (see parseEmbedDim). Unlike the chat/
  // embed aliases this is NEVER undefined: an absent env yields the 768 default,
  // preserving full backward-compatibility for the store's prior 768 fallback.
  embedDim: parseEmbedDim(process.env.LITELLM_EMBED_DIM),
  // Phase 5 (5a): filesystem path to the durable LangGraph SQLite checkpoint
  // database. Deliberately NOT under the LITELLM_* namespace — it configures
  // local persistence, not the provider. Optional; when unset, the checkpointer
  // factory falls back to its documented default (./data/checkpoints.sqlite,
  // relative to the process working directory). The parent directory is created
  // on first use. This db file is gitignored and must never be committed.
  checkpointDbPath: process.env.CHECKPOINT_DB_PATH?.trim() || undefined,
  // Phase 7 (7a): filesystem path to the durable, app-owned KNOWLEDGE-BASE
  // SQLite database (documents/chunks + the sqlite-vec vector table). This is a
  // SEPARATE file and a SEPARATE better-sqlite3 connection from the conversation
  // checkpoint DB above — the two never share a handle. Like CHECKPOINT_DB_PATH
  // it is LOCAL persistence, NOT a provider setting, so it is deliberately NOT
  // under the LITELLM_* namespace. Optional; when unset, the knowledge store
  // falls back to its documented default (./data/knowledge.sqlite, relative to
  // the process working directory). The parent directory is created on first
  // use. This db file is gitignored (data/) and must never be committed.
  knowledgeDbPath: process.env.KNOWLEDGE_DB_PATH?.trim() || undefined,
  // Phase 5 (5c): TCP port for the authenticated HTTP service (npm run serve).
  // Optional; defaults to 3000. Parsed lazily by the entrypoint so importing
  // config never fails on a bad port during offline tests.
  servicePort: process.env.SERVICE_PORT?.trim() || undefined,
  // Phase 5 (5c): the bearer-token -> userId map for the HTTP service, as a raw
  // string. Two accepted formats (see .env.example and parseAuthTokens):
  //   - JSON object: {"token-abc":"user-alice"}
  //   - CSV pairs:   token-abc:user-alice,token-def:user-bob
  // Optional at load time (kept raw here; parsed by the entrypoint) so the
  // offline test suite — which injects its own auth table directly — never
  // depends on this being set. The entrypoint fails loudly if it is empty,
  // because a service with no tokens would reject every request. Tokens are
  // SECRETS: never commit real values; .env is gitignored.
  authTokensRaw: process.env.AUTH_TOKENS?.trim() || undefined,
  // Phase 5 (5d): HTTP server hardening timeouts, in milliseconds. All optional;
  // kept RAW here and parsed by the entrypoint (src/run-server.ts), which falls
  // back to the documented defaults (headers 10_000 / request 30_000 / SSE idle
  // 60_000) when unset. Kept raw (not parsed) at load time so importing config
  // never throws on a malformed value during offline tests — those inject tiny
  // timeout values straight into createServer(deps) and never read process env.
  //   - SERVICE_HEADERS_TIMEOUT_MS bounds time to receive request headers.
  //   - SERVICE_REQUEST_TIMEOUT_MS bounds time to receive the ENTIRE request
  //     (headers + body); it also bounds the oversized-body drain-to-EOF.
  //   - SERVICE_SSE_IDLE_TIMEOUT_MS caps an idle chat stream (no token emitted)
  //     before the turn is aborted and a redacted error event ends the stream.
  serviceHeadersTimeoutMs: process.env.SERVICE_HEADERS_TIMEOUT_MS?.trim() || undefined,
  serviceRequestTimeoutMs: process.env.SERVICE_REQUEST_TIMEOUT_MS?.trim() || undefined,
  serviceSseIdleTimeoutMs: process.env.SERVICE_SSE_IDLE_TIMEOUT_MS?.trim() || undefined,
};
