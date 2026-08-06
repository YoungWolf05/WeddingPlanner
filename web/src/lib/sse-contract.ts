// Phase 9 (9c): the BROWSER-SIDE mirror of the versioned typed SSE contract.
//
// CANONICAL SOURCE: src/core/sse.ts (the backend). These types MUST MATCH that
// module's wire shapes EXACTLY. Because the frontend is a SEPARATE npm project
// (its own tsconfig/build, isolated from the backend's `tsc`/vitest so the
// offline backend suite is never polluted), it cannot cleanly import the backend
// `src/` across the project boundary — so the wire types are RE-DECLARED here.
//
// DRIFT RISK + WHERE IT IS CAUGHT. Re-declaration means a backend contract
// change could silently drift from this mirror. That is ACCEPTED for 9c and
// covered by:
//   - the `SSE_PROTOCOL_VERSION` guard in sseClient.ts (a mismatched `version`
//     on the `init` event surfaces a user-safe error rather than misparsing);
//   - 9d Playwright browser E2E, which drives the REAL backend end-to-end and is
//     what actually VERIFIES wire compatibility (this file is only a typing aid).
// Keep this file in lockstep with src/core/sse.ts when the contract evolves.

// The protocol version the client understands. MUST equal SSE_PROTOCOL_VERSION
// in src/core/sse.ts (currently 2). The client branches on the `init` event's
// `version` BEFORE interpreting any other event; a mismatch is surfaced as an
// error (see sseClient.ts) rather than risking a misparse.
export const SSE_PROTOCOL_VERSION = 2 as const;

// SSE `event:` names — mirrors SSE_EVENT in src/core/sse.ts. v1
// (init/token/done/error) plus the v2 additions (citation/tool/artifact).
export const SSE_EVENT = {
  init: "init",
  token: "token",
  done: "done",
  error: "error",
  citation: "citation",
  tool: "tool",
  artifact: "artifact",
} as const;

export type SseEventName = (typeof SSE_EVENT)[keyof typeof SSE_EVENT];

// --- v1 payloads (unchanged from Phase 5) -----------------------------------

// First event on every successful stream. `version` lets the client validate the
// contract before reading any token.
export interface SseInitEvent {
  version: typeof SSE_PROTOCOL_VERSION;
  // Server-issued conversation key. NOT identity / authorization.
  threadId: string;
}

// One incremental chunk of assistant text.
export interface SseTokenEvent {
  text: string;
}

// Turn completed successfully. `text` is the full accumulated reply.
export interface SseDoneEvent {
  text: string;
}

// A redacted, client-safe failure. The backend has ALREADY redacted `message`
// (src/core/redaction.ts); the client only DISPLAYS it — never re-derives it.
export interface SseErrorEvent {
  message: string;
}

// --- v2 payloads (Phase 9 / 9a) ---------------------------------------------

// EvidenceStatus mirror (src/core/rag.ts). An "insufficient" turn carries an
// EMPTY citation list and is rendered DISTINCTLY in the UI.
export type EvidenceStatus = "supported" | "insufficient";

// One TRUSTED citation as carried on the wire — a WIRE PROJECTION of the Phase 8
// TrustedCitation. Every field is APP-OWNED (resolved from the store by the
// backend, never from model text). `ownerId` is deliberately OMITTED from the
// wire (internal authorization field). The UI renders citations ONLY from these
// fields (exit criterion 3): it never fabricates a citation client-side.
export interface SseCitation {
  marker: number;
  chunkId: string;
  documentId: string;
  sourceUri: string | null;
  chunkIndex: number;
  score: number;
  contentHash: string;
}

// The `citation` event: the turn's TRUSTED citations (batched) + the
// app-authoritative `evidenceStatus`. An "insufficient" turn has an empty list.
export interface SseCitationEvent {
  citations: SseCitation[];
  evidenceStatus: EvidenceStatus;
}

// The two phases of a tool's lifecycle (mirrors src/core/sse.ts).
export type SseToolPhase = "call" | "result";

// A tool RESULT's outcome. "error" is a ToolMessage with status: "error" (also
// how the backend's ToolNode refuses an unknown/unpermitted tool).
export type SseToolStatus = "ok" | "error";

// The `tool` event: a typed, DISCRIMINATED projection of the agent tool
// lifecycle. `args` / `content` are the only dynamic free-text fields and have
// ALREADY been redacted by the backend before the wire; the client only renders.
export type SseToolEvent =
  | {
      phase: "call";
      name: string;
      toolCallId: string;
      args: Record<string, unknown>;
    }
  | {
      phase: "result";
      name: string;
      toolCallId: string;
      status: SseToolStatus;
      content?: string;
    };

// The `artifact` event: a typed, self-describing ENVELOPE for a structured
// artifact produced during the turn. `kind` selects the client renderer; `data`
// is a JSON-safe structured payload (typed `unknown` here; narrowed per `kind`
// by the renderer). Any free-text within `data` has ALREADY been redacted by the
// backend.
export interface SseArtifactEvent {
  kind: string;
  data: unknown;
}

// The stable `kind` the backend's grounded path emits (mirrors
// GROUNDED_ANSWER_ARTIFACT_KIND in src/core/server.ts). `data` is
// { answer: string; evidenceStatus: EvidenceStatus }.
export const GROUNDED_ANSWER_ARTIFACT_KIND = "grounded_answer";

// A narrowed view of the grounded-answer artifact `data`. The renderer validates
// an incoming artifact against this shape before treating it as such; anything
// else is rendered generically (kind + JSON).
export interface GroundedAnswerArtifactData {
  answer: string;
  evidenceStatus: EvidenceStatus;
}

// Runtime narrowing for the grounded-answer artifact envelope. Kept as a pure
// predicate so both the renderer and any unit test share one definition.
export function isGroundedAnswerArtifact(
  artifact: SseArtifactEvent
): artifact is { kind: typeof GROUNDED_ANSWER_ARTIFACT_KIND; data: GroundedAnswerArtifactData } {
  if (artifact.kind !== GROUNDED_ANSWER_ARTIFACT_KIND) return false;
  const data = artifact.data;
  if (typeof data !== "object" || data === null) return false;
  const record = data as Record<string, unknown>;
  return (
    typeof record["answer"] === "string" &&
    (record["evidenceStatus"] === "supported" ||
      record["evidenceStatus"] === "insufficient")
  );
}

// A single, discriminated, TYPED event dispatched by the SSE client. The `type`
// discriminant is the SSE `event:` name; `data` is the parsed payload. Consumers
// (the app) switch on `type` — never on stringly-typed raw frames.
export type SseClientEvent =
  | { type: typeof SSE_EVENT.init; data: SseInitEvent }
  | { type: typeof SSE_EVENT.token; data: SseTokenEvent }
  | { type: typeof SSE_EVENT.done; data: SseDoneEvent }
  | { type: typeof SSE_EVENT.error; data: SseErrorEvent }
  | { type: typeof SSE_EVENT.citation; data: SseCitationEvent }
  | { type: typeof SSE_EVENT.tool; data: SseToolEvent }
  | { type: typeof SSE_EVENT.artifact; data: SseArtifactEvent };
