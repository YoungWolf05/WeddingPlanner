// Phase 9 (9d): DETERMINISTIC, OFFLINE E2E BACKEND HARNESS.
//
// WHAT THIS IS. A thin script that boots the REAL backend HTTP conversation
// service — src/core/server.ts `createServer(deps)` — with a REAL ThreadStore
// over a TEMP sqlite DB OUTSIDE the repo and a FAKE `answerTurn` grounded-turn
// seam that returns SCRIPTED GroundedAnswerResult values keyed off the message
// text. It binds a real port on 127.0.0.1 with a known two-owner AUTH_TOKENS
// map. Playwright drives the built SPA (served by `vite preview`) whose Vite
// proxy forwards /threads* to THIS harness.
//
// WHY THIS EXERCISES THE REAL CODE PATHS. Only the MODEL BOUNDARY is faked: the
// grounded-turn seam (`answerTurn`) returns scripted results instead of calling
// the live LiteLLM proxy + embeddings + Phase 8 RAG. EVERYTHING ELSE is the real
// production code — the REAL auth (src/core/auth.ts), the REAL ownership store
// (src/core/threads.ts: identical-404 for not-owned/nonexistent, generic-401),
// the REAL SSE v2 wire (src/core/sse.ts via SseWriter), and the REAL redaction.
// So the E2E proves the real server + the real v2 SSE wire + the real React
// client interoperate end-to-end, deterministically, with NO network/model call.
//
// NO LIVE MODEL / NO NETWORK. The grounded path in server.ts bypasses `createChat`
// entirely (it runs `answerTurn`), so `createChat` is wired to a function that
// THROWS if ever reached — proving the plain-chat graph is never touched. The
// backend `config` module hard-fails without LITELLM_* env, so we set BENIGN
// PLACEHOLDER values (NOT real secrets, never used to make a call) BEFORE
// importing any backend module. The zero-creds bundle guardrail (9c) already
// proves the browser holds no provider creds; these placeholders live only in
// this Node harness process, never in the served bundle.
//
// TYPE ISOLATION. The web project must NOT type-check backend src/** (separate
// tsconfig). The backend modules are dynamic-imported at RUNTIME and cast to the
// NARROW re-declared surface in ./backend-types.ts (same accepted pattern as the
// 9c SSE-contract mirror). tsx resolves the real modules against the ROOT
// project's node_modules at runtime.
//
// ISOLATION / HYGIENE. The sqlite DB lives under os.tmpdir() (OUTSIDE the repo);
// it is created on start and removed on exit. This harness NEVER writes ./data or
// any repo artifact, and it is NEVER imported by any test suite (offline vitest
// or web vitest) — it is an opt-in Playwright webServer entrypoint only.

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import type {
  AnswerTurn,
  BackendAuthModule,
  BackendGroundedTurnModule,
  BackendMemoryModule,
  BackendServerModule,
  BackendThreadsModule,
  GroundedAnswerResult,
  HistoryMessage,
  HistoryReader,
  HttpServerLike,
  SqliteSaverLike,
  ThreadStore,
  TokenAuthenticator,
  TrustedCitation,
} from "./backend-types.js";
import {
  TOKEN_USER,
  TOKEN_USER2,
  USER_ID,
  USER2_ID,
} from "./auth.js";
import {
  INSUFFICIENT_TRIGGER,
  SCRIPTED_CITATION,
  SUPPORTED_ANSWER,
} from "./scripted.js";

// BENIGN placeholders so the backend `config` (which requires LITELLM_* at import
// time) loads. They are NEVER used: the grounded path never constructs a model,
// and `createChat` is wired to throw. Set BEFORE importing backend modules.
process.env["LITELLM_BASE_URL"] ??= "http://127.0.0.1:9/e2e-placeholder-not-used";
process.env["LITELLM_API_KEY"] ??= "e2e-placeholder-not-a-real-key";

// Backend internals (dynamic-imported AFTER the placeholder env is set, and cast
// to the narrow re-declared surface). These resolve against the ROOT project's
// node_modules (this harness is run with the root's tsx).
//
// The specifier is passed THROUGH a variable so the web TypeScript project does
// NOT statically resolve (and thus type-check) the backend src/** module graph —
// that graph lives under a different tsconfig with different lib/strictness. The
// runtime resolution is unaffected (tsx resolves the real path against the root
// node_modules). We then cast the loaded namespace to the narrow re-declared
// backend surface (backend-types.ts), keeping the harness type-safe with NO `any`.
async function loadBackend<T>(specifier: string): Promise<T> {
  const mod: unknown = await import(specifier);
  return mod as T;
}

const BACKEND = "../../../src/core";
const serverMod = await loadBackend<BackendServerModule>(`${BACKEND}/server.js`);
const memoryMod = await loadBackend<BackendMemoryModule>(`${BACKEND}/memory.js`);
const threadsMod = await loadBackend<BackendThreadsModule>(
  `${BACKEND}/threads.js`
);
const authMod = await loadBackend<BackendAuthModule>(`${BACKEND}/auth.js`);
const groundedTurnMod = await loadBackend<BackendGroundedTurnModule>(
  `${BACKEND}/grounded-turn.js`
);

// The fixed port this harness binds. Overridable via E2E_BACKEND_PORT so the
// Playwright config and the vite-preview proxy target stay in agreement.
const BACKEND_PORT = Number(process.env["E2E_BACKEND_PORT"] ?? "3100");

// --- SCRIPTED grounded results (the faked model boundary) -------------------
// The fake answerTurn keys off the message text so every journey is
// deterministic. The APP-OWNED citation fields below are the EXACT values the
// browser must render — the specs assert equality, which is the end-to-end
// wire-compat proof that the 9c re-declared v2 types match the live backend.

// A scripted, app-owned trusted citation. Every field is app-owned (as the real
// 8b resolver guarantees); `ownerId` is set to the turn's owner and is
// DELIBERATELY dropped from the wire by the server's toSseCitation projection.
function scriptedCitation(ownerId: string): TrustedCitation {
  return {
    ...SCRIPTED_CITATION,
    // The server drops ownerId from the wire; set it to the caller for
    // authorization-consistency of the scripted result.
    ownerId,
  };
}

// A SUPPORTED grounded answer with exactly one trusted citation.
function supportedResult(ownerId: string): GroundedAnswerResult {
  return {
    answer: {
      answer: SUPPORTED_ANSWER,
      citations: [1],
      insufficientEvidence: false,
    },
    resolvedCitations: [scriptedCitation(ownerId)],
    droppedCitations: [],
    markerMap: new Map(),
    retrieved: [],
    contextBlock: "",
    evidenceStatus: "supported",
  };
}

// An INSUFFICIENT-evidence turn: empty answer, NO citations, evidenceStatus
// "insufficient" — so the client renders the distinct insufficient state with
// no fabricated citations.
function insufficientResult(): GroundedAnswerResult {
  return {
    answer: { answer: "", citations: [], insufficientEvidence: true },
    resolvedCitations: [],
    droppedCitations: [],
    markerMap: new Map(),
    retrieved: [],
    contextBlock: "",
    evidenceStatus: "insufficient",
  };
}

// --- IN-HARNESS HISTORY RECORDER (for GET /threads/:id/messages replay) ------
//
// The grounded path (answerTurn) does NOT write to the conversational
// checkpointer, so there is no real persisted history to read back. To exercise
// the REAL history-replay ROUTE end-to-end (real auth + real ownership gate +
// real 404/redaction), the harness records each completed turn's user+assistant
// text per thread and serves it through the injected `readHistory` seam.
//
// CORRELATION. answerTurn receives (ownerId, query) but not the threadId; the
// server calls `store.touchThread(ownerId, threadId)` immediately AFTER a
// successful turn. E2E specs are SEQUENTIAL (one in-flight turn at a time), so we
// stash the last (ownerId, query, answer) from answerTurn and pair it with the
// next touchThread(ownerId, threadId) to append the turn to that thread's
// transcript — a deterministic, offline reconstruction of replayable history.
const historyByThread = new Map<string, HistoryMessage[]>();
let pendingTurn: { ownerId: string; query: string; answer: string } | null =
  null;

function recordPendingTurn(ownerId: string, threadId: string): void {
  if (pendingTurn === null || pendingTurn.ownerId !== ownerId) return;
  const transcript = historyByThread.get(threadId) ?? [];
  transcript.push({ role: "user", text: pendingTurn.query });
  if (pendingTurn.answer !== "") {
    transcript.push({ role: "assistant", text: pendingTurn.answer });
  }
  historyByThread.set(threadId, transcript);
  pendingTurn = null;
}

const readHistory: HistoryReader = async (threadId: string) =>
  historyByThread.get(threadId) ?? [];

// The scripted grounded-turn seam. DETERMINISTIC + keyed off the message text:
//   - contains "insufficient" -> the insufficient-evidence case
//   - otherwise               -> a supported answer + one trusted citation
// The `ownerId` it receives is the AUTHENTICATED owner (from the bearer token,
// resolved by the real auth+server) — never client input — so the scripted
// citation's ownerId matches the caller and stays authorization-consistent.
const scriptedAnswerTurn: AnswerTurn = async ({ query, ownerId, signal }) => {
  if (signal.aborted) {
    // Mirror the production seam's early-abort semantics on a pre-aborted turn.
    throw new groundedTurnMod.TurnAbortedError();
  }
  const result = query.toLowerCase().includes(INSUFFICIENT_TRIGGER)
    ? insufficientResult()
    : supportedResult(ownerId);
  // Stash this turn so the next touchThread pairs it to its thread (see above).
  pendingTurn = { ownerId, query, answer: result.answer.answer };
  return result;
};

// `createChat` must NEVER run on the grounded path. If it does, fail loudly so a
// regression that bypasses the grounded seam is caught immediately.
function forbiddenCreateChat(): never {
  throw new Error(
    "[e2e-harness] createChat must not run: the grounded answerTurn seam is always used"
  );
}

// --- boot -------------------------------------------------------------------

const tempDir = mkdtempSync(path.join(tmpdir(), "wp-e2e-"));
const dbPath = path.join(tempDir, "checkpoints.sqlite");

const saver: SqliteSaverLike = memoryMod.createCheckpointer(dbPath);
const realStore: ThreadStore = threadsMod.createThreadStore(saver);

// Wrap the real store so touchThread ALSO records the just-completed turn into
// the in-harness history map (see recordPendingTurn). All other store methods
// pass straight through to the REAL owner-scoped store (real ownership/404).
const store: ThreadStore = new Proxy(realStore, {
  get(target, prop, receiver) {
    if (prop === "touchThread") {
      return (ownerId: string, threadId: string): boolean => {
        const result = target.touchThread(ownerId, threadId);
        if (result) recordPendingTurn(ownerId, threadId);
        return result;
      };
    }
    return Reflect.get(target, prop, receiver) as unknown;
  },
});

const auth: TokenAuthenticator = authMod.createTokenAuthenticator({
  [TOKEN_USER]: USER_ID,
  [TOKEN_USER2]: USER2_ID,
});

const server: HttpServerLike = serverMod.createServer({
  store,
  auth,
  createChat: forbiddenCreateChat,
  answerTurn: scriptedAnswerTurn,
  // Conversation-history replay (GET /threads/:id/messages) served from the
  // in-harness recorder (the grounded path does not persist to the conversational
  // checkpointer). The REAL route still enforces auth + the ownership gate.
  readHistory,
  // Quiet, redacted-only logger (the server only ever hands it redacted strings).
  log: () => {},
});

function shutdown(): void {
  try {
    server.close();
  } catch {
    /* already closing */
  }
  try {
    saver.db.close();
  } catch {
    /* already closed */
  }
  try {
    rmSync(tempDir, { recursive: true, force: true });
  } catch {
    /* best effort */
  }
}

process.on("SIGINT", () => {
  shutdown();
  process.exit(0);
});
process.on("SIGTERM", () => {
  shutdown();
  process.exit(0);
});
process.on("exit", shutdown);

server.listen(BACKEND_PORT, "127.0.0.1", () => {
  // Single line so Playwright's webServer readiness has a human hint in logs; no
  // secrets. The SPA's proxy targets this port.
  console.log(
    `[e2e-harness] deterministic backend listening on http://127.0.0.1:${String(
      BACKEND_PORT
    )} (temp db ${dbPath})`
  );
});
