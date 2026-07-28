import type { ServerResponse } from "node:http";

// Phase 5 (5c): the VERSIONED, typed Server-Sent Events (SSE) contract for the
// streaming chat endpoint.
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

// Bump when the event shape changes in a backward-incompatible way. Clients
// read this from the initial `init` event and branch accordingly.
export const SSE_PROTOCOL_VERSION = 1 as const;

// SSE `event:` names. Kept as a const map so both the writer and tests refer to
// one source of truth (no stringly-typed drift).
export const SSE_EVENT = {
  init: "init",
  token: "token",
  done: "done",
  error: "error",
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

// A minimal, typed SSE writer bound to one ServerResponse. Every public method
// maps to exactly one event type in the versioned contract, so callers cannot
// emit an off-contract frame. `end()` after `done`/`error` closes the response.
//
// Writes are guarded: once the underlying socket is finished/destroyed (e.g. the
// client disconnected), further writes are no-ops rather than throwing — this is
// what keeps a mid-stream client disconnect from crashing the turn.
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
