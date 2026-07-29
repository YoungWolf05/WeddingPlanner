# Wedding Planner Chatbot — Agent Instructions

LangChain + TypeScript chatbot routed through a **corporate LiteLLM proxy** (not OpenAI directly). Uses `@langchain/openai` `ChatOpenAI` pointed at the LiteLLM base URL.

## Environment Setup

Copy `.env.example` to `.env` before running anything. Required:

- `LITELLM_API_KEY` — mandatory, app hard-fails without it
- `LITELLM_BASE_URL` — mandatory, app hard-fails without it
- `LITELLM_MODEL` — optional, defaults to `claude-sonnet-4-6`

Available models: `claude-opus-4-8`, `claude-sonnet-4-6`

Phase 5 service env vars (all optional except `AUTH_TOKENS` to actually run
`npm run serve`; `.env.example` is authoritative):

- `CHECKPOINT_DB_PATH` — durable SQLite DB path (default `./data/checkpoints.sqlite`, gitignored). Not under the `LITELLM_` namespace — it configures local persistence, not the provider.
- `SERVICE_PORT` — HTTP service port on 127.0.0.1 (default `3000`).
- `AUTH_TOKENS` — bearer-token → userId map (JSON object or `token:userId` CSV). The resolved userId is the thread OWNER; the service refuses to start with none.
- `SERVICE_HEADERS_TIMEOUT_MS` / `SERVICE_REQUEST_TIMEOUT_MS` / `SERVICE_SSE_IDLE_TIMEOUT_MS` — R1 hardening timeouts (defaults `10000` / `30000` / `60000`).

## Developer Commands

```bash
npm test                         # OFFLINE Vitest suite — deterministic, mocked model, no network
npm run chain "your message"     # single-turn LCEL chain (Phase 1)
npm run memory                   # multi-turn memory demo (Phase 2)
npm run chat                     # streaming terminal REPL (Phase 3)
npm run typecheck                # type-check without emitting (src + vitest configs)
npm run build                    # compile to dist/

# LIVE / opt-in — make real, credentialed LiteLLM calls. NOT part of npm test or CI:
npm run test:connection          # verify LiteLLM reachability — run first after setup
npm run test:capabilities        # probe the chat/embedding capability matrix -> docs/capabilities/<date>.md
npm run eval                     # run the wedding-planning eval dataset -> docs/eval/<date>.md
npm run serve                    # LIVE local durable conversation service (Phase 5); binds SERVICE_PORT on 127.0.0.1
```

`npm run serve` starts the authenticated HTTP + SSE conversation service
(`src/run-server.ts`). It is a LIVE local service: it needs `AUTH_TOKENS`
configured (it refuses to start with none) and a reachable model for actual
chat turns, and it opens the durable SQLite DB. It is NOT run by `npm test`; the
HTTP/auth/ownership/SSE layer is fully covered OFFLINE by importing
`src/core/server.ts` with injected deps + a mocked model, and the listening
entrypoint is never imported by the suite.

`npm test` is fully OFFLINE and CI-safe: the model boundary (`createChatModel`)
is mocked per test, so no credentials or network are used. `test:connection`,
`test:capabilities`, and `eval` are LIVE, opt-in commands that call the real
proxy and are never run by `npm test` or CI. `typecheck` and `build` provide
compilation checks.

## Architecture

```
src/
  config.ts            # env loading — single source of truth for baseURL/apiKey/model/embedModel
  core/
    model.ts           # createChatModel() — the ONLY ChatOpenAI construction site; LiteLLM wiring + tracing hook
    embeddings.ts      # createEmbeddingsModel() — the ONLY OpenAIEmbeddings construction site (separate contract)
    prompts.ts         # WEDDING_PLANNER_SYSTEM_PROMPT + weddingPlannerPrompt (persona: "Aria")
    chain.ts           # Phase 1: createWeddingPlannerChain() (LCEL, stateless)
                       # Phase 2: createConversationalChain() (LangGraph + checkpointer)
    memory.ts          # Phase 5 (5a): durable SQLite checkpointer behind BaseCheckpointSaver
                       #   (createCheckpointer/getCheckpointer/getSqliteSaver); sessionConfig() for thread_id
    threads.ts         # Phase 5 (5b): thread ownership store (createThreadStore); owner-scoped
                       #   CRUD + atomic hard delete + pruneThreads() retention hook (5d)
    auth.ts            # Phase 5 (5c): bearer-token -> userId TokenAuthenticator (constant-time, generic 401)
    sse.ts             # Phase 5 (5c): versioned typed SSE contract (SSE_PROTOCOL_VERSION) + SseWriter
    server.ts          # Phase 5 (5c/5d): node:http service (createServer(deps)); auth-before-dispatch,
                       #   ownership enforcement, SSE chat, body limits, R1 timeouts, R3 proto guard
    repl.ts            # Phase 3 REPL pure logic (ALLOWED_MODELS, command parsing)
    redaction.ts       # shared, always-on secret/PII scrubbing (pure); reused by probe + tracing + eval
    tracing.ts         # Phase 4 (4d) local tracing at the createChatModel() boundary (opt-in, redacted)
    capabilities.ts    # Phase 4 (4c) pure capability-matrix rendering + abort classification
    eval.ts            # Phase 4 (4e) pure eval dataset parsing, property scorers, aggregation + rendering
  run-chain.ts         # CLI entrypoint for Phase 1
  run-memory.ts        # CLI entrypoint for Phase 2
  cli.ts               # Phase 3 streaming terminal REPL and conversation controls
  run-server.ts        # Phase 5 LIVE `npm run serve` entrypoint (binds SERVICE_PORT; NOT imported by tests)
  test-connection.ts   # LIVE smoke test (routed through createChatModel)
  probe-capabilities.ts# LIVE, opt-in capability probe -> docs/capabilities/<date>.md
  run-eval.ts          # LIVE, opt-in eval runner -> docs/eval/<date>.md
evals/
  dataset.jsonl        # versioned wedding-planning eval prompts + deterministic expectations
```

The offline Vitest suite lives under `test/` (outside `src/`, so it is never
emitted to `dist/`). The pure modules above (`redaction.ts`, `capabilities.ts`,
`eval.ts`, `repl.ts`) are I/O-free so the suite exercises them without a live
call; the `probe-capabilities.ts` and `run-eval.ts` scripts own the impure,
live-only I/O and are never imported by tests.

### Durable conversation service (Phase 5) notes

- **Persistence / durability.** Conversation state lives in a durable SQLite
  checkpointer (`src/core/memory.ts`) and the ownership store (`src/core/threads.ts`)
  SHARES that one better-sqlite3 connection (`getSqliteSaver()`), so a thread's
  ownership row and its checkpoint state transact together (atomic hard delete;
  no orphaned checkpoints). State survives a process/server restart — a fresh
  `createServer` over the same DB file resumes threads and history (covered
  end-to-end offline in `test/phase5-hardening.test.ts`).
- **Concurrency model.** Requests are handled on ONE Node event loop, and
  better-sqlite3 is a SYNCHRONOUS single connection, so DB operations are
  serialized — there is no intra-process write interleaving to corrupt state.
  SQLite WAL is set by the saver. This is validated (not just asserted): concurrent
  creates for different users and concurrent chat turns on different threads are
  covered offline. Multi-instance / horizontal concurrency is deliberately NOT
  claimed and is a future (PostgreSQL / Phase 10) concern.
- **Retention.** `ThreadStore.pruneThreads({ olderThanEpochMs, ownerId? })` is a
  callable policy HOOK (owner-scopeable) that atomically hard-deletes expired
  threads + their checkpoint state, reusing the same delete semantics. It is NOT
  a scheduler; operational purge cadence is a Phase 10 concern.
- **App-owned schema versioning / migrations (5e).** The APP-OWNED schema
  (currently the `threads` table + its owner index) is evolved by a real,
  versioned, forward-only migration runner in `src/core/migrations.ts`
  (`runMigrations(db, migrations)`, the ordered `APP_MIGRATIONS` list, and the
  `getAppSchemaVersion(db)` reader), which `createThreadStore` runs at
  construction (replacing the old bare `CREATE TABLE IF NOT EXISTS`). The version
  is recorded in a DEDICATED app-owned table, `app_schema_migrations` (append-only,
  one row per applied migration) — NOT `PRAGMA user_version`. Verified: the pinned
  `@langchain/langgraph-checkpoint-sqlite` 1.0.3 sets only `journal_mode=WAL` and
  never touches `user_version`; a dedicated table is chosen anyway because the DB
  file is SHARED and a future library version could start using `user_version`,
  so an app-namespaced table is collision-proof (a coexistence test also asserts
  `user_version` stays 0). Each migration step runs in its own `db.transaction()`
  (atomic: a throwing `up()` rolls back its DDL and does NOT advance the recorded
  version); steps are additive/data-preserving (no drop-recreate). Forward-only
  safety: if the recorded version exceeds the code's latest known migration (DB
  written by a newer app), the runner FAILS LOUDLY and mutates nothing. **App vs
  library schema boundary:** this runner manages ONLY app tables and MUST NOT
  create/alter/migrate the LangGraph-owned `checkpoints`/`writes` tables — those
  remain the library's responsibility (`SqliteSaver.setup()`), mirrored separately
  by `ensureCheckpointTables` (so the atomic delete/prune DELETEs can be prepared)
  and guarded against library drift by the version-pin coupling test. The runner
  is injectable (tests pass synthetic migration lists) and covered offline in
  `test/phase5-schema-migrations.test.ts` (fresh-init, idempotency, multi-step,
  data preservation, forward-only guard, atomic rollback, library coexistence,
  restart/durability).
- **Hardening (5d).** `createServer(deps)` applies `http.Server` `headersTimeout`
  / `requestTimeout` (R1) and a per-turn SSE idle timeout that aborts a stalled
  stream with a redacted `error` event; the JSON body boundary rejects top-level
  `__proto__`/`constructor`/`prototype` keys (R3, defense-in-depth); the 413
  oversized-body path stays clean and is time-bounded by `requestTimeout` (R4).
  All timeout values are injectable via `deps.timeouts` so tests drive the
  timeout paths deterministically (no real 10s/30s/60s waits).

## Critical: ESM Import Extensions

This is a NodeNext ESM project (`"type": "module"`). All local imports **must use `.js` extension**, even when importing `.ts` source files:

```ts
import { config } from "../config.js"; // correct
import { config } from "../config"; // wrong — will fail at runtime
```

## Roadmap and phase governance

[`docs/roadmap.md`](docs/roadmap.md) is the phase/status source of truth. **Phase 4 — Engineering Baseline and Provider Contract** is complete. **Phase 5 — Durable Conversation Service** is the ACTIVE phase and is in progress: increments 5a (durable SQLite checkpointer), 5b (thread identity & ownership), and 5c (authenticated HTTP service + SSE) are complete, and 5d (hardening, concurrency/restart/retention validation, docs alignment) is in progress. Phase 5 is NOT complete — top-level Phase 5 exit criteria remain unticked and closeout is a separate manager step.

Do not activate or complete a phase without user approval. A completion status requires the approved exit criteria and recorded verification evidence; intent or partial implementation is insufficient.

## Conventions

- All application chat-model construction goes through `createChatModel()` in `src/core/model.ts` — do not instantiate `ChatOpenAI` elsewhere (including `src/test-connection.ts`, which was routed through the factory in 4b). Embeddings construction goes through `createEmbeddingsModel()` in `src/core/embeddings.ts` — do not instantiate `OpenAIEmbeddings` elsewhere. An offline guard test (`test/phase4-model-factory.test.ts`) enforces both single-factory rules.
- System prompt / persona changes belong in `src/core/prompts.ts`.
- Evaluation dataset changes belong in `evals/dataset.jsonl`; property scorers live in `src/core/eval.ts`. Keep prompts benign and PII-free; keep expectations property-based (checkable without an LLM judge).
- Any string that may reach a log, trace, console, or dated evidence file must pass through `src/core/redaction.ts` first (always-on secret/PII scrubbing).
- Service ownership/security (Phase 5): `thread_id` is a server-issued UUID conversation key, NEVER identity or authorization. The `ownerId` for every store operation comes ONLY from the authenticated bearer token (`src/core/auth.ts`) — never from any client-supplied body/query/header/path field. Not-owned and not-found are indistinguishable (identical 404, no existence leak). All client-facing and logged errors are redacted.
- TypeScript strict mode is on. Do not disable strict checks or add `any` casts.
