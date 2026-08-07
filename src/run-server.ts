import { createConversationalChain } from "./core/chain.js";
import { getSqliteSaver } from "./core/memory.js";
import { createThreadStore } from "./core/threads.js";
import { createTokenAuthenticator, parseAuthTokens } from "./core/auth.js";
import { createHistoryReader, createServer, DEFAULT_TIMEOUTS, parseTimeoutMs, type StreamingChat } from "./core/server.js";
import { createKnowledgeStore } from "./core/knowledge-store.js";
import { createQueryEmbedder } from "./core/retriever.js";
import { createAnswerTurn, type AnswerTurn } from "./core/grounded-turn.js";
import { config } from "./config.js";
import { redactError } from "./core/redaction.js";

// Interpret the OPT-IN SERVICE_GROUNDED flag (see config.serviceGroundedRaw).
// Truthy = "1"/"true"/"yes" (case-insensitive); anything else (including unset)
// is OFF, preserving the existing plain-chat serve behavior.
function isGroundedEnabled(raw: string | undefined): boolean {
  if (!raw) return false;
  return ["1", "true", "yes"].includes(raw.toLowerCase());
}

// Phase 5 (5c): LIVE entrypoint for the authenticated HTTP service.
//
// This is the ONLY file that binds a port and starts listening, and it is
// DELIBERATELY not imported by the offline test suite: tests import
// src/core/server.ts and start their own server on an ephemeral port with
// injected deps + a mocked model. So merely running `npm test` never opens a
// socket or a real LiteLLM connection.
//
// Wiring (production dependencies):
//   - store: the 5b ownership store over the SHARED SQLite connection
//     (getSqliteSaver), so the ownership row and checkpoint state share one
//     durable database and the hard delete stays atomic.
//   - auth: token->userId map parsed from AUTH_TOKENS (fails loudly if empty —
//     a service with no tokens would reject every request).
//   - createChat: a fresh streaming conversational graph per turn, bound to the
//     shared saver so history persists across turns and restarts.

function parsePort(raw: string | undefined): number {
  if (!raw) return 3000;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 0 || port > 65_535) {
    throw new Error(
      `Invalid SERVICE_PORT "${raw}". Must be an integer in 0..65535.`
    );
  }
  return port;
}

// Parse a POSITIVE integer millisecond timeout from env, falling back to the
// approved default when unset. The parser (with its reject-0 rule) lives in
// src/core/server.ts so it is unit-testable offline without importing this
// listening entrypoint.
function main(): void {
  const authTokens = parseAuthTokens(config.authTokensRaw);
  if (Object.keys(authTokens).length === 0) {
    throw new Error(
      "No auth tokens configured. Set AUTH_TOKENS (see .env.example) — the " +
        "service authenticates every request via a bearer token and would " +
        "otherwise reject all traffic."
    );
  }
  const auth = createTokenAuthenticator(authTokens);

  // Share ONE SqliteSaver across the store and every per-turn graph so the
  // ownership table and checkpoint tables live in the same durable connection.
  const saver = getSqliteSaver();
  const store = createThreadStore(saver);

  // R1 hardening timeouts from env, with the approved 5d defaults when unset.
  const timeouts = {
    headersTimeoutMs: parseTimeoutMs(
      config.serviceHeadersTimeoutMs,
      "SERVICE_HEADERS_TIMEOUT_MS",
      DEFAULT_TIMEOUTS.headersTimeoutMs
    ),
    requestTimeoutMs: parseTimeoutMs(
      config.serviceRequestTimeoutMs,
      "SERVICE_REQUEST_TIMEOUT_MS",
      DEFAULT_TIMEOUTS.requestTimeoutMs
    ),
    sseIdleTimeoutMs: parseTimeoutMs(
      config.serviceSseIdleTimeoutMs,
      "SERVICE_SSE_IDLE_TIMEOUT_MS",
      DEFAULT_TIMEOUTS.sseIdleTimeoutMs
    ),
  };

  // Phase 9 (9b): OPT-IN grounded-answer mode. When SERVICE_GROUNDED is truthy,
  // build the grounded-turn seam over the durable knowledge store + the real
  // query embedder (routed through the single embeddings factory) + the default
  // generation model. answerTurn drives the v2 citation/artifact SSE events; when
  // OFF (default) the chat endpoint keeps the plain-chat streaming path. The
  // knowledge store opens a durable file handle here in the LIVE entrypoint (never
  // imported by the offline suite), so this side effect stays out of `npm test`.
  const grounded = isGroundedEnabled(config.serviceGroundedRaw);
  let answerTurn: AnswerTurn | undefined;
  if (grounded) {
    const knowledgeStore = createKnowledgeStore();
    const queryEmbedder = createQueryEmbedder();
    answerTurn = createAnswerTurn({ store: knowledgeStore, queryEmbedder });
  }

  const server = createServer({
    store,
    auth,
    createChat: (): StreamingChat =>
      // For 5c the graph is constructed with streaming: true so streamMode
      // "messages" yields incremental token chunks.
      createConversationalChain({ streaming: true }, saver) as StreamingChat,
    // Conversation-history replay (GET /threads/:id/messages): read a thread's
    // prior messages from the SHARED durable saver via the compiled graph's
    // getState. This serves REAL history for the default plain-chat mode. NOTE:
    // grounded turns (SERVICE_GROUNDED) run answerTurn and do NOT persist to this
    // checkpointer, so the route replays [] for grounded threads — see
    // createHistoryReader's documented caveat.
    readHistory: createHistoryReader(saver),
    ...(answerTurn !== undefined ? { answerTurn } : {}),
    timeouts,
  });

  const port = parsePort(config.servicePort);
  const host = "127.0.0.1";
  server.listen(port, host, () => {
    const address = server.address();
    const boundPort =
      address && typeof address === "object" ? address.port : port;
    console.error(
      `[server] Wedding Planner HTTP service listening on http://${host}:${boundPort} ` +
        `(${auth.size} auth token(s) configured; ` +
        `chat mode: ${grounded ? "grounded (RAG + citations)" : "plain-chat"})`
    );
  });

  const shutdown = (signal: string): void => {
    console.error(`[server] received ${signal}, shutting down…`);
    server.close(() => process.exit(0));
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}

try {
  main();
} catch (err) {
  console.error("[server] failed to start:", redactError(err));
  process.exit(1);
}
