// Phase 9 (9c): OFFLINE unit tests for startChatStream's async reader loop and
// its crit-2 SAFETY branches. The PURE SSE parse helpers (splitSseFrames /
// parseFrameLines / parseSseFrame) are covered in sseClient.test.ts; this file
// drives the END-TO-END reader loop through startChatStream with a FAKE fetch —
// no network, no real server, no timers. The fake fetch returns a Response-like
// object whose `body` is a ReadableStream we construct in-test from encoded
// Uint8Array chunks (TextEncoder), so partial-frame buffering, the version /
// handshake guards, ordered dispatch, and quiet-abort semantics are all
// exercised through the PUBLIC startChatStream contract.

import { afterEach, describe, expect, it, vi } from "vitest";
import { startChatStream } from "../src/lib/sseClient.js";
import type { ChatStreamHandlers } from "../src/lib/sseClient.js";

const encoder = new TextEncoder();

// A minimal, SOUNDLY-typed Response-like value: only the fields startChatStream
// reads (`ok`, `status`, `body`). Typed as `Response` at the fetch boundary via
// the fakeFetch helper so no `any` is needed.
interface ResponseLike {
  ok: boolean;
  status: number;
  body: ReadableStream<Uint8Array> | null;
}

// Build a ReadableStream that emits the given encoded string chunks IN ORDER,
// one per pull, then closes. Deterministic: a chunk is enqueued only when the
// consumer pulls, so read order mirrors emit order with no timing races.
function streamFromChunks(chunks: string[]): ReadableStream<Uint8Array> {
  let index = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index += 1;
      } else {
        controller.close();
      }
    },
  });
}

// A ReadableStream whose reader honors an AbortSignal: after `emitBeforeAbort`
// chunks it aborts the controller and every subsequent read rejects with an
// AbortError (DOMException name === "AbortError"), mirroring how a real
// fetch-body reader surfaces a caller-initiated cancel. Deterministic: the abort
// fires on a specific pull, not on a wall-clock timer.
function abortingStream(
  chunks: string[],
  emitBeforeAbort: number,
  controller: AbortController
): ReadableStream<Uint8Array> {
  let index = 0;
  return new ReadableStream<Uint8Array>({
    pull(streamController) {
      if (index < emitBeforeAbort && index < chunks.length) {
        streamController.enqueue(encoder.encode(chunks[index]));
        index += 1;
        return;
      }
      // Simulate the caller aborting mid-stream: abort the signal and surface an
      // AbortError from this read, exactly as a real reader would.
      controller.abort();
      streamController.error(
        new DOMException("The operation was aborted.", "AbortError")
      );
    },
  });
}

// A fetch stand-in typed as the global `fetch` so the assignment to
// globalThis.fetch is sound (no `any`). It ignores its arguments and returns the
// prepared Response-like value cast through `Response` (structurally sufficient:
// startChatStream reads only ok/status/body).
function installFakeFetch(response: ResponseLike): void {
  const fake: typeof fetch = () => Promise.resolve(response as unknown as Response);
  vi.stubGlobal("fetch", fake);
}

// A fetch stand-in that REJECTS (transport failure / cancel-at-fetch), typed as
// the global fetch so the stub is sound.
function installRejectingFetch(err: unknown): void {
  const fake: typeof fetch = () => Promise.reject(err);
  vi.stubGlobal("fetch", fake);
}

// Record the ORDER of dispatched events (via onEvent) plus per-kind + error
// callbacks, so a test can assert both ordering and the specific typed payloads.
function recordingHandlers(): {
  handlers: ChatStreamHandlers;
  order: string[];
  tokens: string[];
  citationStatuses: string[];
  doneTexts: string[];
  errors: string[];
} {
  const order: string[] = [];
  const tokens: string[] = [];
  const citationStatuses: string[] = [];
  const doneTexts: string[] = [];
  const errors: string[] = [];
  const handlers: ChatStreamHandlers = {
    onEvent: (event) => order.push(event.type),
    onToken: (data) => tokens.push(data.text),
    onCitation: (data) => citationStatuses.push(data.evidenceStatus),
    onDone: (data) => doneTexts.push(data.text),
    onError: (message) => errors.push(message),
  };
  return { handlers, order, tokens, citationStatuses, doneTexts, errors };
}

function baseArgs(
  handlers: ChatStreamHandlers,
  signal: AbortSignal
): Parameters<typeof startChatStream>[0] {
  return {
    baseUrl: "",
    threadId: "t1",
    message: "hello",
    token: "test-token",
    handlers,
    signal,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("startChatStream", () => {
  it("(a) surfaces a user-safe error on a protocol VERSION MISMATCH and does not misparse later events", async () => {
    // First event is init with an UNSUPPORTED version, followed by a token that
    // MUST NOT be dispatched (the client stops at the mismatch).
    installFakeFetch({
      ok: true,
      status: 200,
      body: streamFromChunks([
        'event: init\ndata: {"version":99,"threadId":"t1"}\n\n',
        'event: token\ndata: {"text":"should-not-parse"}\n\n',
      ]),
    });
    const { handlers, order, tokens, errors } = recordingHandlers();
    const controller = new AbortController();

    await startChatStream(baseArgs(handlers, controller.signal));

    // The error path is taken (not a thrown raw error / not silent).
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain("Unsupported server protocol version 99");
    // The safety guard runs BEFORE dispatch, so NOTHING is dispatched at all —
    // not even the offending init reaches onEvent, and the later token is never
    // misparsed (the client returns at the mismatch).
    expect(order).toEqual([]);
    expect(tokens).toHaveLength(0);
  });

  it("(b) surfaces a user-safe error when the FIRST event is NOT init and does not misparse it", async () => {
    // The stream opens with a token (no handshake). The guard rejects it.
    installFakeFetch({
      ok: true,
      status: 200,
      body: streamFromChunks([
        'event: token\ndata: {"text":"premature"}\n\n',
        'event: token\ndata: {"text":"also-premature"}\n\n',
      ]),
    });
    const { handlers, order, tokens, errors } = recordingHandlers();
    const controller = new AbortController();

    await startChatStream(baseArgs(handlers, controller.signal));

    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain("did not start with a valid handshake");
    // The handshake guard runs BEFORE dispatch, so the premature token is NEVER
    // dispatched (no onEvent, no onToken) — the client returns at the guard.
    expect(order).toEqual([]);
    expect(tokens).toHaveLength(0);
  });

  it("(c) buffers a CHUNK-SPLIT frame across two reads and parses it as ONE token event", async () => {
    // A valid init handshake, then a SINGLE token frame delivered in TWO reads:
    // the frame boundary falls mid-payload, proving end-to-end partial buffering.
    installFakeFetch({
      ok: true,
      status: 200,
      body: streamFromChunks([
        'event: init\ndata: {"version":2,"threadId":"t1"}\n\n',
        "event: token\nda",
        'ta: {"text":"hi"}\n\n',
      ]),
    });
    const { handlers, order, tokens, errors } = recordingHandlers();
    const controller = new AbortController();

    await startChatStream(baseArgs(handlers, controller.signal));

    expect(errors).toHaveLength(0);
    // Exactly ONE token event, correctly reassembled from the split frame.
    expect(tokens).toEqual(["hi"]);
    expect(order).toEqual(["init", "token"]);
  });

  it("(d) drives the typed callbacks in ORDER on the happy path init->token->citation->done", async () => {
    installFakeFetch({
      ok: true,
      status: 200,
      body: streamFromChunks([
        'event: init\ndata: {"version":2,"threadId":"t1"}\n\n',
        'event: token\ndata: {"text":"Hello"}\n\n',
        'event: citation\ndata: {"citations":[],"evidenceStatus":"supported"}\n\n',
        'event: done\ndata: {"text":"Hello"}\n\n',
      ]),
    });
    const { handlers, order, tokens, citationStatuses, doneTexts, errors } =
      recordingHandlers();
    const controller = new AbortController();

    await startChatStream(baseArgs(handlers, controller.signal));

    expect(errors).toHaveLength(0);
    expect(order).toEqual(["init", "token", "citation", "done"]);
    expect(tokens).toEqual(["Hello"]);
    expect(citationStatuses).toEqual(["supported"]);
    expect(doneTexts).toEqual(["Hello"]);
  });

  it("(e) resolves QUIETLY on a mid-stream abort — an AbortError is NOT surfaced as a generic error", async () => {
    const controller = new AbortController();
    // Emit a valid init handshake, then abort on the next pull.
    installFakeFetch({
      ok: true,
      status: 200,
      body: abortingStream(
        ['event: init\ndata: {"version":2,"threadId":"t1"}\n\n'],
        1,
        controller
      ),
    });
    const { handlers, order, errors } = recordingHandlers();

    // Must RESOLVE (never reject) despite the AbortError from the reader.
    await expect(
      startChatStream(baseArgs(handlers, controller.signal))
    ).resolves.toBeUndefined();

    // The init was dispatched, then the abort ended the stream QUIETLY: no
    // onError call (a user cancel is not an error), and NOT the generic message.
    expect(order).toEqual(["init"]);
    expect(errors).toHaveLength(0);
  });

  it("(e') resolves quietly when the fetch itself rejects with an AbortError (cancel before body)", async () => {
    installRejectingFetch(
      new DOMException("The operation was aborted.", "AbortError")
    );
    const { handlers, errors } = recordingHandlers();
    const controller = new AbortController();

    await expect(
      startChatStream(baseArgs(handlers, controller.signal))
    ).resolves.toBeUndefined();

    expect(errors).toHaveLength(0);
  });
});
