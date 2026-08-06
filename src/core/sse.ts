import type { ServerResponse } from "node:http";
import type { TrustedCitation } from "./citations.js";
import type { EvidenceStatus } from "./rag.js";

// Phase 5 (5c) / Phase 9 (9a): the VERSIONED, typed Server-Sent Events (SSE)
// contract for the streaming chat endpoint.
//
// DESIGN GOALS
//   - INDEPENDENT OF THE TERMINAL TUPLE FORMAT. The REPL (src/core/repl.ts)
//     consumes LangGraph's `[messageChunk, metadata]` tuples directly; that is
//     an internal streaming detail. The HTTP/SSE contract below is a stable,
//     public wire format the browser/client depends on. The server translates
//     graph chunks into these events; clients never see the tuple shape.
//   - VERSIONED so the contract can evolve. The stream ALWAYS opens with an
//     `init` event carrying `version`, so a client can branch on the schema
//     version before interpreting any `token`/`done`/`error` event.
//   - REDACTED failures only. An `error` event's `message` is produced via the
//     shared redaction helper (src/core/redaction.ts) so provider secrets
//     (apiKey/baseURL) and PII never reach the wire.
//
// PHASE 9 (9a) — CONTRACT v2 (typed citations / tool-state / artifacts).
// ----------------------------------------------------------------------
// This increment BUMPS the protocol to v2 and ADDS three new event families —
// `citation`, `tool`, and `artifact` — so the browser UI (Phase 9c) can render
// TRUSTED citations, agent tool progress, and structured artifacts from TYPED
// TRUSTED events (exit criterion 3), over a VERSIONED wire format (exit
// criterion 2). 9a defines the CONTRACT + WRITER ONLY. It does NOT wire these
// events into the server (that is 9b: translating GroundedAnswerResult / the
// agent message stream into these events) and it builds NO frontend (that is
// 9c). The v2 payloads are built FROM the existing Phase 8 TrustedCitation /
// GroundedAnswerResult and Phase 6 agent-message shapes — never from model text.
//
// BACKWARD-COMPATIBILITY CONTRACT (v2 is strictly ADDITIVE).
//   - The v1 event shapes — `init`, `token`, `done`, `error` — are UNCHANGED in
//     both name and payload meaning. A v1-aware client keeps working: it still
//     receives init -> token* -> done, or an error, exactly as before.
//   - The stream STILL opens with `init`, and `init` STILL carries `version`.
//     `version` is now 2, so a client BRANCHES on `init.version` BEFORE
//     interpreting any new event. A v1-only client that reads `version` can
//     detect it is talking to a newer server; even if it does not, the SSE spec
//     requires clients to IGNORE unknown `event:` names, so the additive
//     `citation`/`tool`/`artifact` frames are safely skipped by a v1 client and
//     the init/token/done/error subset still renders correctly.
//   - Additive means: new event NAMES + new payload TYPES only; no existing
//     event's shape or ordering guarantee is removed or repurposed.

// Bump when the event set changes. v2 (Phase 9 / 9a) is a strictly ADDITIVE
// bump: the v1 events (init/token/done/error) are unchanged; the `citation`,
// `tool`, and `artifact` events are new. Clients read this from the initial
// `init` event and branch accordingly (unknown event names are ignored per the
// SSE spec, so a v1 client still works — see the BACKWARD-COMPATIBILITY note).
export const SSE_PROTOCOL_VERSION = 2 as const;

// SSE `event:` names. Kept as a const map so both the writer and tests refer to
// one source of truth (no stringly-typed drift).
//
// v1 events (init/token/done/error) are the original streaming contract. v2
// (9a) ADDS `citation`, `tool`, and `artifact` (all additive — see the module
// header's BACKWARD-COMPATIBILITY note).
export const SSE_EVENT = {
  // --- v1 (unchanged) -------------------------------------------------------
  init: "init",
  token: "token",
  done: "done",
  error: "error",
  // --- v2 additions (Phase 9 / 9a) ------------------------------------------
  // TRUSTED citations for the turn (Phase 8 TrustedCitation objects). Batched.
  citation: "citation",
  // Agent tool lifecycle: a tool-call INTENTION or a tool RESULT/ERROR.
  tool: "tool",
  // A typed structured artifact envelope for the turn.
  artifact: "artifact",
} as const;

export type SseEventName = (typeof SSE_EVENT)[keyof typeof SSE_EVENT];

// The `init` marker: first event on every successful stream. `version` lets a
// client validate the contract before reading tokens.
export interface SseInitEvent {
  version: typeof SSE_PROTOCOL_VERSION;
  // Echo the conversation key so a client can correlate the stream. This is the
  // server-issued thread_id (already authorized for this caller); it is NOT
  // identity and carries no authorization on its own.
  threadId: string;
}

// One incremental chunk of assistant text.
export interface SseTokenEvent {
  text: string;
}

// Turn completed successfully. `text` is the full accumulated reply, so a client
// that missed/coalesced token events still has the final answer.
export interface SseDoneEvent {
  text: string;
}

// A redacted, client-safe failure. `message` has already passed through
// redactError(); never place a raw provider error here.
export interface SseErrorEvent {
  message: string;
}

// --- v2 payloads (Phase 9 / 9a) ---------------------------------------------

// One TRUSTED citation as carried on the wire. This is a WIRE PROJECTION of the
// Phase 8 `TrustedCitation` (src/core/citations.ts) — every field is APP-OWNED,
// copied from the store-backed RetrievedChunk via the app-assigned markerMap,
// NEVER from model text. Only the display-relevant subset is exposed:
//   - marker      : the app-assigned integer the model echoed (correlates a
//                   citation back to the answer's numbered context).
//   - chunkId     : server-derived chunk identity (7a/7c).
//   - documentId  : server-derived owning-document identity.
//   - sourceUri   : the document's app-owned source identity (null only for a
//                   legacy row with no recorded source).
//   - chunkIndex  : the chunk's ordinal position within its document.
//   - score       : the bounded similarity score (convenience for display; NOT
//                   identity). Carried so a UI can show confidence / order.
//   - contentHash : the chunk's content-version hash (convenience; NOT identity;
//                   lets a UI detect citation drift after re-ingestion).
// `ownerId` is DELIBERATELY OMITTED from the wire projection: it is an internal
// authorization field, not display data, and the whole stream is already scoped
// to the authenticated owner (Phase 5). Keeping it off the wire avoids leaking
// the authorization scheme to the client.
export interface SseCitation {
  marker: number;
  chunkId: string;
  documentId: string;
  sourceUri: string | null;
  chunkIndex: number;
  score: number;
  contentHash: string;
}

// The `citation` event: the TRUSTED citations for the turn, emitted as ONE
// BATCHED event. Batching (rather than one event per citation) matches the
// two-step RAG flow — the resolved/authorized citation set is known only once
// the grounded answer completes (see rag.ts `GroundedAnswerResult`) — and keeps
// the wire simple (a single array the UI renders at answer completion). The
// event also carries `evidenceStatus` so the client can render an "insufficient
// evidence" turn DISTINCTLY from a "supported" one: an "insufficient" turn
// carries an EMPTY `citations` array (rag.ts keeps resolvedCitations consistent
// with evidenceStatus), so the UI shows the insufficient-evidence state without
// citations. Every entry is built from the app-owned TrustedCitation, so nothing
// here is model-supplied identity.
export interface SseCitationEvent {
  citations: SseCitation[];
  evidenceStatus: EvidenceStatus;
}

// The two phases of a tool's lifecycle, modeling the Phase 6 agent MESSAGE
// STREAM (see agent.ts): a tool INTENTION is an `AIMessage.tool_calls` entry
// (name + parsed args + id); a RESULT/ERROR is a `ToolMessage` (content/artifact
// linked by `tool_call_id`, `status: "error"` on failure). "call" projects the
// intention; "result" projects the ToolMessage (ok or error).
export type SseToolPhase = "call" | "result";

// A tool RESULT's outcome. "ok" is a normal ToolMessage; "error" is a
// ToolMessage with `status: "error"` (also how the prebuilt ToolNode refuses an
// unknown/unpermitted tool — see agent.ts). Only present on the "result" phase.
export type SseToolStatus = "ok" | "error";

// The `tool` event: a typed, DISCRIMINATED projection of the agent tool
// lifecycle for a UI (Phase 6 exit-criterion-3 shape, now on the SSE wire).
// Discriminated by `phase`:
//   - phase "call"   : a tool-call INTENTION. Carries `name`, `toolCallId`, and
//                      the parsed `args` (a JSON-safe object echoing the model's
//                      structured tool arguments). No `status`/`content`.
//   - phase "result" : a tool RESULT or ERROR. Carries `name`, `toolCallId`,
//                      `status` ("ok"|"error"), and an OPTIONAL redacted
//                      `content` summary of the ToolMessage output.
//
// REDACTION CONTRACT (same as the v1 `error` event — the CALLER redacts):
//   `args` and `content` are the ONLY dynamic free-text fields that could carry
//   caller/model-derived data. Per the module-wide convention, the CALLER (9b's
//   server wiring) MUST redact any such free-text via src/core/redaction.ts
//   BEFORE handing the payload to the writer — the writer only SERIALIZES, it
//   does NOT scrub (see SseWriter.tool). `name`/`toolCallId`/`phase`/`status`
//   are structural and safe. Keeping the redaction responsibility at the call
//   site (next to where the tool output is produced) is consistent with error().
export type SseToolEvent =
  | {
      phase: "call";
      name: string;
      toolCallId: string;
      // Parsed, JSON-safe tool arguments (from AIMessage.tool_calls[].args).
      // Caller-redacted if it can carry dynamic free-text.
      args: Record<string, unknown>;
    }
  | {
      phase: "result";
      name: string;
      toolCallId: string;
      status: SseToolStatus;
      // OPTIONAL redacted summary of the ToolMessage content. Caller-redacted.
      content?: string;
    };

// The `artifact` event: a typed, well-formed ENVELOPE for a structured artifact
// produced during the turn (e.g. a Phase 6 structured-output object such as a
// BudgetPlan / PlanningChecklist). 9a defines the ENVELOPE only; the concrete
// artifact KINDS the server emits are decided by 9b (server wiring) and rendered
// by 9c (frontend). The envelope keeps the wire well-typed and self-describing:
//   - kind : a short, stable discriminator naming the artifact type (e.g.
//            "budget_plan"). The client switches on this to pick a renderer.
//   - data : the JSON-safe structured payload. It is typed as `unknown` here
//            (9a intentionally does not enumerate kinds); 9b/9c narrow it per
//            `kind`. It MUST be a structured, JSON-serializable value — NOT
//            arbitrary un-redacted free text. If any field could carry dynamic
//            free-text, the CALLER redacts it via src/core/redaction.ts before
//            handing it to the writer (same caller-redacts convention as above).
export interface SseArtifactEvent {
  kind: string;
  data: unknown;
}

// --- Compile-time coupling: aliases MUST NOT drift from the union (9a) -------
//
// The `SseToolPhase` / `SseToolStatus` aliases above are documentation-only:
// they name the literal sets that the `SseToolEvent` discriminated union
// actually uses for its `phase` discriminant and its result-phase `status`.
// Because they are declared SEPARATELY from the union, they could silently
// DRIFT (e.g. adding a `phase: "stream"` union member without extending the
// alias, or vice versa). The two `AssertExact` checks below make any such drift
// a COMPILE ERROR (`npm run typecheck`) — they have NO runtime cost (pure type
// space) and introduce NO `any`. Extract the discriminants straight from the
// union with indexed access so the alias is proven to be EXACTLY the union's
// literal set (bidirectional: neither may gain or lose a member unnoticed).

// Exact-equality assertion: resolves to `true` only when `A` and `B` are
// mutually assignable (identical types); any drift makes it `false`, which
// `AssertTrue` then rejects at compile time.
type AssertExact<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? (<T>() => T extends B ? 1 : 2) extends <T>() => T extends A ? 1 : 2
      ? true
      : false
    : false;
type AssertTrue<T extends true> = T;

// `SseToolPhase` must equal the union of every member's `phase` discriminant.
type _AssertPhaseCoupled = AssertTrue<
  AssertExact<SseToolPhase, SseToolEvent["phase"]>
>;
// `SseToolStatus` must equal the `status` field carried by the result phase.
type _AssertStatusCoupled = AssertTrue<
  AssertExact<SseToolStatus, Extract<SseToolEvent, { phase: "result" }>["status"]>
>;

// The HTTP response headers that put a connection into SSE mode. Exposed so the
// server sets them in exactly one place and tests can assert them.
export const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
  // Defensive against proxies that buffer responses (would defeat streaming).
  "X-Accel-Buffering": "no",
} as const;

// Serialize one SSE frame: `event: <name>\n` followed by one `data:` line
// carrying the JSON payload, then the blank-line terminator. JSON is single-line
// (no embedded newlines from JSON.stringify), so one `data:` line is sufficient
// and unambiguous.
export function formatSseFrame(event: SseEventName, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

// Project the app-owned Phase 8 TrustedCitation into the WIRE `SseCitation`
// (see SseCitation). PURE: it copies the display-relevant, APP-OWNED fields
// verbatim (never mutating them) and DROPS the internal `ownerId` authorization
// field from the wire. Nothing here is model-supplied identity — the input is
// already a resolved, trusted, store-backed citation.
export function toSseCitation(citation: TrustedCitation): SseCitation {
  return {
    marker: citation.marker,
    chunkId: citation.chunkId,
    documentId: citation.documentId,
    sourceUri: citation.sourceUri,
    chunkIndex: citation.chunkIndex,
    score: citation.score,
    contentHash: citation.contentHash,
  };
}

// A minimal, typed SSE writer bound to one ServerResponse. Every public method
// maps to exactly one event type in the versioned contract, so callers cannot
// emit an off-contract frame. `end()` after `done`/`error` closes the response.
//
// Writes are guarded: once the underlying socket is finished/destroyed (e.g. the
// client disconnected), further writes are no-ops rather than throwing — this is
// what keeps a mid-stream client disconnect from crashing the turn.
//
// EVENT ORDERING IN A TURN (v2). The stream still OPENS with `init` and
// TERMINATES with `done` (success) or `error` + `end()`. The v2 events sit
// WITHIN that envelope, faithful to how 9b will drive them:
//   init -> (token* interleaved with tool* during generation)
//        -> citation (+ artifact) at/just before completion -> done
// Concretely: `tool` events are emitted MID-STREAM as the agent calls tools and
// receives results/errors; `citation` (and any `artifact`) is emitted at answer
// COMPLETION (the resolved/authorized citation set is known only once the
// grounded answer finishes — see rag.ts), just BEFORE `done`; `done` still
// carries the full accumulated text. None of the v2 methods are terminal — only
// `done`/`error` + `end()` close the stream.
export class SseWriter {
  private closed = false;

  constructor(private readonly res: ServerResponse) {}

  // Send the versioned init marker and flush the SSE headers. Must be called
  // first; it is what transitions the response into event-stream mode.
  init(threadId: string): void {
    this.res.writeHead(200, SSE_HEADERS);
    const payload: SseInitEvent = {
      version: SSE_PROTOCOL_VERSION,
      threadId,
    };
    this.write(SSE_EVENT.init, payload);
  }

  token(text: string): void {
    const payload: SseTokenEvent = { text };
    this.write(SSE_EVENT.token, payload);
  }

  done(text: string): void {
    const payload: SseDoneEvent = { text };
    this.write(SSE_EVENT.done, payload);
  }

  // `message` MUST already be redacted by the caller (redactError). This class
  // does not redact — it only serializes — so the redaction responsibility stays
  // explicit at the call site next to the error handling.
  error(message: string): void {
    const payload: SseErrorEvent = { message };
    this.write(SSE_EVENT.error, payload);
  }

  // Emit the turn's TRUSTED citations as one BATCHED `citation` event (v2). Each
  // TrustedCitation is projected to its wire form via toSseCitation (app-owned
  // fields copied verbatim, `ownerId` dropped from the wire). `evidenceStatus`
  // is carried so the client can render an "insufficient" turn distinctly; an
  // "insufficient" turn passes an EMPTY list (rag.ts keeps resolvedCitations
  // consistent with evidenceStatus). This event carries NO free-text and NO
  // model-supplied identity, so there is nothing for the caller to redact — the
  // inputs are already resolved, trusted, store-backed citations. Emitted at
  // answer completion, just before done() (see EVENT ORDERING above). Non-
  // terminal: it does NOT close the stream.
  citations(
    citations: readonly TrustedCitation[],
    evidenceStatus: EvidenceStatus
  ): void {
    const payload: SseCitationEvent = {
      citations: citations.map(toSseCitation),
      evidenceStatus,
    };
    this.write(SSE_EVENT.citation, payload);
  }

  // Emit one tool-lifecycle `tool` event (v2): a tool-call INTENTION (phase
  // "call") or a tool RESULT/ERROR (phase "result"). See SseToolEvent for the
  // discriminated shape. Emitted MID-STREAM as the agent runs (see EVENT
  // ORDERING above); non-terminal.
  //
  // REDACTION CONTRACT (same as error(): the CALLER redacts). The dynamic
  // free-text fields — `args` on a "call", `content` on a "result" — MUST
  // ALREADY be redacted by the caller (via src/core/redaction.ts) before being
  // handed here. This class SERIALIZES only; it does NOT scrub — the redaction
  // responsibility stays explicit at the call site next to where the tool output
  // is produced (9b), exactly like the error() event's `message`.
  tool(payload: SseToolEvent): void {
    this.write(SSE_EVENT.tool, payload);
  }

  // Emit one typed structured-artifact ENVELOPE `artifact` event (v2). See
  // SseArtifactEvent: `kind` names the artifact type (the client's renderer
  // discriminator) and `data` is the JSON-safe structured payload. Emitted at
  // answer completion (see EVENT ORDERING above); non-terminal.
  //
  // REDACTION CONTRACT (same as error(): the CALLER redacts). `data` MUST be a
  // structured, JSON-serializable value; if any field could carry dynamic
  // free-text, the CALLER redacts it (via src/core/redaction.ts) before handing
  // the envelope here. This class SERIALIZES only; it does NOT scrub.
  artifact(payload: SseArtifactEvent): void {
    this.write(SSE_EVENT.artifact, payload);
  }

  end(): void {
    if (this.closed) return;
    this.closed = true;
    this.res.end();
  }

  private write(event: SseEventName, data: unknown): void {
    if (this.closed || this.res.writableEnded || this.res.destroyed) return;
    this.res.write(formatSseFrame(event, data));
  }
}
