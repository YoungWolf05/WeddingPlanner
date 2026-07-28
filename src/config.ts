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
};
