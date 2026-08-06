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

export const config = {
  baseURL: required("LITELLM_BASE_URL"),
  apiKey: required("LITELLM_API_KEY"),
  model: process.env.LITELLM_MODEL?.trim() || "claude-sonnet-4-6",
  // Optional embedding alias. Embeddings are a SEPARATE contract from chat
  // aliases (see docs/roadmap.md Phase 4). When unset, the capability probe
  // records embeddings as "N/A — no embedding alias configured" rather than
  // failing; it never assumes a chat alias supports embeddings.
  embedModel: process.env.LITELLM_EMBED_MODEL?.trim() || undefined,
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
