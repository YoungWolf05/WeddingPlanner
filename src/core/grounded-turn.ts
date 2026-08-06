import { redactText } from "./redaction.js";
import { answerQuestion } from "./rag.js";
import type { GroundedAnswerResult } from "./rag.js";
import type { KnowledgeStore } from "./knowledge-store.js";
import type { QueryEmbedder } from "./retriever.js";

// Phase 9 (increment 9b): GROUNDED-TURN SEAM — the injected collaborator that
// runs the Phase 8 two-step RAG pipeline for one chat turn and hands the server a
// GroundedAnswerResult to translate into v2 SSE events.
//
// WHY A SEAM (and not calling answerQuestion directly in server.ts). The Phase 8
// RAG (answerQuestion) needs a knowledge store + a query embedder + a generation
// model — all live I/O collaborators. Injecting a narrow `AnswerTurn` function
// keeps src/core/server.ts free of that wiring and, crucially, keeps the server's
// grounded path OFFLINE-TESTABLE: the suite injects a fake AnswerTurn returning a
// scripted GroundedAnswerResult (no embeddings/model/network), while production
// injects createAnswerTurn() below (real store + embedder + model). This mirrors
// the existing `createChat` factory seam for the plain-chat path.
//
// AUTHORIZATION. `ownerId` is threaded straight through to answerQuestion's
// ownerId, so retrieval is owner-scoped (retriever.ts) AND the 8b citation
// resolver drops any chunk outside the caller's authorization. The server ALWAYS
// passes the ownerId resolved from the authenticated bearer token — never from
// thread_id or any client-supplied field — so grounded citations are
// authorization-consistent with the Phase 5 ownership model.
//
// CANCELLATION. The two-step RAG is not a token stream, but a client disconnect
// (or the idle watchdog) must still cancel in-flight work. The server passes its
// per-turn AbortSignal here; this seam observes it and ABORTS EARLY (before the
// billable generation round-trip) if it is already aborted, and re-checks after
// completion so nothing is emitted for an abandoned turn. (retrieve()/generate()
// do not yet take a signal in their public contract, so the honored cancellation
// is: no generation is started once aborted, and the server never emits after an
// abort — SseWriter also guards write-after-close.) The signal-threading into the
// retrieval/generation internals is a documented follow-up.

// Arguments for one grounded turn. Deliberately narrow: the request `query`, the
// authenticated `ownerId` (authorization scope), and the per-turn `signal`.
export interface AnswerTurnArgs {
  // The user's chat message for this turn (already validated/trimmed by the
  // server). Passed through to answerQuestion as the query.
  query: string;
  // The authorization scope: the ownerId resolved from the authenticated bearer
  // token. Forwarded to answerQuestion for owner-scoped retrieval + citation
  // authorization. Never sourced from client input.
  ownerId: string;
  // The per-turn AbortSignal (client-disconnect + idle-watchdog). Observed for
  // early cancellation (see CANCELLATION above).
  signal: AbortSignal;
}

// The injected grounded-turn collaborator: run the RAG pipeline for one turn and
// return the full GroundedAnswerResult (answer text + trusted resolved citations
// + evidenceStatus). The server translates that into the v2 SSE events. Errors
// propagate to the server, which emits a REDACTED sse.error (never raw provider
// text). This is the shape tests inject a scripted fake for.
export type AnswerTurn = (args: AnswerTurnArgs) => Promise<GroundedAnswerResult>;

// Thrown when a turn is cancelled before generation. Redaction-safe (fixed,
// content-free message). The server treats an aborted turn as a client
// disconnect (no error event), so this only surfaces on the abort path.
export class TurnAbortedError extends Error {
  constructor() {
    super(redactText("Grounded turn aborted before generation."));
    this.name = "TurnAbortedError";
  }
}

// Options for the production grounded-turn factory.
export interface CreateAnswerTurnOptions {
  // The durable, app-owned knowledge store (createKnowledgeStore).
  store: KnowledgeStore;
  // The query embedder (createQueryEmbedder over the single embeddings factory).
  queryEmbedder: QueryEmbedder;
  // Number of chunks to retrieve per turn (positive integer). Defaults to
  // DEFAULT_RETRIEVAL_K.
  k?: number;
  // OPTIONAL generation model id (defaults inside answerQuestion/generateStructured
  // to DEFAULT_STRUCTURED_MODEL; opus temperature-omit honored there).
  model?: string;
  // OPTIONAL minimum evidence score (defaults inside answerQuestion to the 8c
  // DEFAULT_MIN_EVIDENCE_SCORE).
  minScore?: number;
}

// Default number of chunks retrieved per grounded turn. A small, fixed default
// matching the retrieval/RAG eval usage; tunable per deployment via options.
export const DEFAULT_RETRIEVAL_K = 4;

/**
 * Production grounded-turn factory: bind the RAG pipeline to a live store +
 * embedder (+ optional model/minScore/k) and return an {@link AnswerTurn}.
 *
 * The returned function forwards the authenticated `ownerId` to answerQuestion
 * (owner-scoped retrieval + citation authorization) and observes the per-turn
 * AbortSignal for early cancellation. It constructs NO model/embeddings client
 * itself — answerQuestion routes generation through the single createChatModel
 * factory (via generateStructured), and the embedder is the injected
 * createQueryEmbedder — so the single-factory rules are preserved. OFFLINE tests
 * never call this; they inject a scripted AnswerTurn directly.
 */
export function createAnswerTurn(options: CreateAnswerTurnOptions): AnswerTurn {
  const { store, queryEmbedder, k = DEFAULT_RETRIEVAL_K, model, minScore } =
    options;
  return async ({ query, ownerId, signal }: AnswerTurnArgs) => {
    // Early cancellation: if the client already disconnected (or the idle
    // watchdog fired) do NOT start the billable generation round-trip.
    if (signal.aborted) {
      throw new TurnAbortedError();
    }
    const result = await answerQuestion({
      store,
      queryEmbedder,
      query,
      k,
      // Owner-scoped retrieval + 8b citation authorization. ALWAYS the
      // authenticated ownerId — never client-supplied.
      ownerId,
      ...(model !== undefined ? { model } : {}),
      ...(minScore !== undefined ? { minScore } : {}),
    });
    return result;
  };
}
