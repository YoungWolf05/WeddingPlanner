import { describe, it, expect } from "vitest";
import type { ServerResponse } from "node:http";
import {
  SSE_PROTOCOL_VERSION,
  SSE_EVENT,
  SSE_HEADERS,
  formatSseFrame,
  toSseCitation,
  SseWriter,
  type SseInitEvent,
  type SseTokenEvent,
  type SseDoneEvent,
  type SseErrorEvent,
  type SseCitationEvent,
  type SseToolEvent,
  type SseArtifactEvent,
  type SseEventName,
} from "../src/core/sse.js";
import type { TrustedCitation } from "../src/core/citations.js";

// Phase 9 (increment 9a) — SSE EVENT CONTRACT v2.
//
// FULLY OFFLINE + DETERMINISTIC: sse.ts is pure (it only serializes onto a
// ServerResponse), so these tests drive it with a FAKE ServerResponse that
// captures every write() — mirroring the fake-sink idiom used by the Phase 3
// REPL tests and the Phase 5 SSE tests. NO network, NO server binding, NO
// RAG/agent calls: 9a defines the CONTRACT + WRITER only (server wiring is 9b,
// the frontend is 9c). Citation payloads are built from the Phase 8
// TrustedCitation — never from model text.

// --- Fake ServerResponse ----------------------------------------------------

// A minimal fake capturing writeHead + every write() chunk, and modeling the
// three socket-state flags SseWriter guards on (writableEnded / destroyed and
// its own `closed`). Only the surface SseWriter touches is implemented.
interface CapturedHead {
  status: number;
  headers: unknown;
}

class FakeResponse {
  public writableEnded = false;
  public destroyed = false;
  public head: CapturedHead | null = null;
  public chunks: string[] = [];

  writeHead(status: number, headers: unknown): this {
    this.head = { status, headers };
    return this;
  }

  write(chunk: string): boolean {
    this.chunks.push(chunk);
    return true;
  }

  end(): this {
    this.writableEnded = true;
    return this;
  }

  // The concatenated wire text of everything written so far.
  get body(): string {
    return this.chunks.join("");
  }

  asServerResponse(): ServerResponse {
    return this as unknown as ServerResponse;
  }
}

// Parse an SSE body into ordered { event, data } frames (same idiom as the
// Phase 5 SSE tests).
interface SseFrame {
  event: string;
  data: unknown;
}
function parseSse(body: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const block of body.split("\n\n")) {
    const trimmed = block.trim();
    if (trimmed === "") continue;
    let event = "message";
    let dataLine = "";
    for (const line of trimmed.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
    }
    frames.push({
      event,
      data: dataLine === "" ? undefined : JSON.parse(dataLine),
    });
  }
  return frames;
}

// A fully-populated app-owned TrustedCitation fixture (as rag.ts/citations.ts
// would produce). ownerId is present here so we can assert the wire projection
// DROPS it.
function makeCitation(overrides: Partial<TrustedCitation> = {}): TrustedCitation {
  return {
    marker: 1,
    chunkId: "chunk-abc",
    documentId: "doc-xyz",
    sourceUri: "knowledge/corpus/budget.md",
    chunkIndex: 3,
    ownerId: "user-alice",
    contentHash: "sha256:deadbeef",
    score: 0.87,
    ...overrides,
  };
}

// --- 1. Version bump + backward-compatible v2 event set ---------------------

describe("Phase 9 (9a) — protocol version + event set", () => {
  it("SSE_PROTOCOL_VERSION is bumped to 2", () => {
    expect(SSE_PROTOCOL_VERSION).toBe(2);
  });

  it("keeps the v1 event names INTACT (backward-compat regression guard)", () => {
    expect(SSE_EVENT.init).toBe("init");
    expect(SSE_EVENT.token).toBe("token");
    expect(SSE_EVENT.done).toBe("done");
    expect(SSE_EVENT.error).toBe("error");
  });

  it("ADDS the v2 event names (citation/tool/artifact)", () => {
    expect(SSE_EVENT.citation).toBe("citation");
    expect(SSE_EVENT.tool).toBe("tool");
    expect(SSE_EVENT.artifact).toBe("artifact");
  });

  it("SseEventName union covers exactly the seven contract events", () => {
    // A compile-time-shaped runtime check: every value of SSE_EVENT is a
    // SseEventName, and there are exactly seven of them.
    const names: SseEventName[] = Object.values(SSE_EVENT);
    expect(new Set(names).size).toBe(7);
    expect(names.sort()).toEqual(
      ["artifact", "citation", "done", "error", "init", "token", "tool"].sort()
    );
  });
});

// --- v1 shapes unchanged (backward-compat) ----------------------------------

describe("Phase 9 (9a) — v1 event shapes unchanged (additive bump)", () => {
  it("init still carries { version, threadId } with version === 2", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    writer.init("thread-123");

    // Headers flushed exactly once via writeHead with the SSE headers.
    expect(res.head?.status).toBe(200);
    expect(res.head?.headers).toBe(SSE_HEADERS);

    const [frame] = parseSse(res.body);
    expect(frame!.event).toBe("init");
    const data = frame!.data as SseInitEvent;
    expect(data.version).toBe(2);
    expect(data.version).toBe(SSE_PROTOCOL_VERSION);
    expect(data.threadId).toBe("thread-123");
    // No extra fields leaked onto init.
    expect(Object.keys(data).sort()).toEqual(["threadId", "version"]);
  });

  it("token/done/error payload fields are intact", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    writer.token("hello ");
    writer.done("hello there");
    writer.error("redacted reason");

    const frames = parseSse(res.body);
    expect(frames.map((f) => f.event)).toEqual(["token", "done", "error"]);
    expect((frames[0]!.data as SseTokenEvent).text).toBe("hello ");
    expect((frames[1]!.data as SseDoneEvent).text).toBe("hello there");
    expect((frames[2]!.data as SseErrorEvent).message).toBe("redacted reason");
  });
});

// --- 2. formatSseFrame for every new event ----------------------------------

describe("Phase 9 (9a) — formatSseFrame for the v2 events", () => {
  it("citation frame: correct event line, single-line data, blank terminator", () => {
    const payload: SseCitationEvent = {
      citations: [toSseCitation(makeCitation())],
      evidenceStatus: "supported",
    };
    const frame = formatSseFrame(SSE_EVENT.citation, payload);
    expect(frame.startsWith("event: citation\n")).toBe(true);
    expect(frame.endsWith("\n\n")).toBe(true);

    const dataLine = frame.split("\n")[1]!;
    expect(dataLine.startsWith("data: ")).toBe(true);
    const json = dataLine.slice("data: ".length);
    // Single-line data: no embedded newline inside the JSON payload.
    expect(json.includes("\n")).toBe(false);
    expect(JSON.parse(json)).toEqual(payload); // round-trips
  });

  it("tool frame (call phase) round-trips as single-line JSON", () => {
    const payload: SseToolEvent = {
      phase: "call",
      name: "days_until",
      toolCallId: "call_1",
      args: { date: "2028-06-01" },
    };
    const frame = formatSseFrame(SSE_EVENT.tool, payload);
    expect(frame.startsWith("event: tool\n")).toBe(true);
    expect(frame.endsWith("\n\n")).toBe(true);
    const json = frame.split("\n")[1]!.slice("data: ".length);
    expect(json.includes("\n")).toBe(false);
    expect(JSON.parse(json)).toEqual(payload);
  });

  it("tool frame (result phase) round-trips as single-line JSON", () => {
    const payload: SseToolEvent = {
      phase: "result",
      name: "days_until",
      toolCallId: "call_1",
      status: "ok",
      content: "512 days until the wedding",
    };
    const frame = formatSseFrame(SSE_EVENT.tool, payload);
    const json = frame.split("\n")[1]!.slice("data: ".length);
    expect(JSON.parse(json)).toEqual(payload);
  });

  it("artifact frame round-trips as single-line JSON", () => {
    const payload: SseArtifactEvent = {
      kind: "budget_plan",
      data: { total: 30000, allocations: [{ category: "venue", amount: 12000 }] },
    };
    const frame = formatSseFrame(SSE_EVENT.artifact, payload);
    expect(frame.startsWith("event: artifact\n")).toBe(true);
    expect(frame.endsWith("\n\n")).toBe(true);
    const json = frame.split("\n")[1]!.slice("data: ".length);
    expect(json.includes("\n")).toBe(false);
    expect(JSON.parse(json)).toEqual(payload);
  });
});

// --- toSseCitation projection -----------------------------------------------

describe("Phase 9 (9a) — toSseCitation projection", () => {
  it("copies app-owned identity fields verbatim and DROPS ownerId", () => {
    const trusted = makeCitation();
    const wire = toSseCitation(trusted);

    // App-owned identity passed through unmutated.
    expect(wire.marker).toBe(trusted.marker);
    expect(wire.chunkId).toBe(trusted.chunkId);
    expect(wire.documentId).toBe(trusted.documentId);
    expect(wire.sourceUri).toBe(trusted.sourceUri);
    expect(wire.chunkIndex).toBe(trusted.chunkIndex);
    expect(wire.score).toBe(trusted.score);
    expect(wire.contentHash).toBe(trusted.contentHash);

    // ownerId is an internal authorization field — NOT on the wire.
    expect("ownerId" in wire).toBe(false);
    expect(Object.keys(wire).sort()).toEqual(
      ["chunkId", "chunkIndex", "contentHash", "documentId", "marker", "score", "sourceUri"].sort()
    );

    // Does not mutate the input.
    expect(trusted.ownerId).toBe("user-alice");
  });

  it("passes a null sourceUri through unchanged", () => {
    const wire = toSseCitation(makeCitation({ sourceUri: null }));
    expect(wire.sourceUri).toBeNull();
  });
});

// --- 3. SseWriter new methods ------------------------------------------------

describe("Phase 9 (9a) — SseWriter.citations()", () => {
  it("emits ONE batched citation event whose payload carries app-owned identity", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    const c1 = makeCitation({ marker: 1, chunkId: "c1", documentId: "d1", chunkIndex: 0 });
    const c2 = makeCitation({ marker: 2, chunkId: "c2", documentId: "d2", chunkIndex: 5 });

    writer.citations([c1, c2], "supported");

    const frames = parseSse(res.body);
    expect(frames).toHaveLength(1);
    expect(frames[0]!.event).toBe("citation");
    const data = frames[0]!.data as SseCitationEvent;
    expect(data.evidenceStatus).toBe("supported");
    expect(data.citations).toHaveLength(2);

    // App-owned identity passed through EXACTLY as given (not mutated).
    expect(data.citations[0]).toEqual({
      marker: 1,
      chunkId: "c1",
      documentId: "d1",
      sourceUri: "knowledge/corpus/budget.md",
      chunkIndex: 0,
      score: 0.87,
      contentHash: "sha256:deadbeef",
    });
    expect(data.citations[1]!.marker).toBe(2);
    expect(data.citations[1]!.chunkId).toBe("c2");
    expect(data.citations[1]!.chunkIndex).toBe(5);
    // ownerId never reaches the wire.
    expect("ownerId" in data.citations[0]!).toBe(false);
  });

  it("carries evidenceStatus 'insufficient' with an empty citation list", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    writer.citations([], "insufficient");

    const data = parseSse(res.body)[0]!.data as SseCitationEvent;
    expect(data.evidenceStatus).toBe("insufficient");
    expect(data.citations).toEqual([]);
  });
});

describe("Phase 9 (9a) — SseWriter.tool()", () => {
  it("emits a discriminated 'call' payload", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    const payload: SseToolEvent = {
      phase: "call",
      name: "split_budget",
      toolCallId: "call_42",
      args: { total: 30000 },
    };
    writer.tool(payload);

    const frames = parseSse(res.body);
    expect(frames).toHaveLength(1);
    expect(frames[0]!.event).toBe("tool");
    expect(frames[0]!.data).toEqual(payload);
  });

  it("emits a discriminated 'result' (ok) payload", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    const payload: SseToolEvent = {
      phase: "result",
      name: "split_budget",
      toolCallId: "call_42",
      status: "ok",
      content: "allocated across 7 categories",
    };
    writer.tool(payload);
    expect(parseSse(res.body)[0]!.data).toEqual(payload);
  });

  it("emits a discriminated 'result' (error) payload", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    const payload: SseToolEvent = {
      phase: "result",
      name: "unknown_tool",
      toolCallId: "call_99",
      status: "error",
      content: "tool not permitted",
    };
    writer.tool(payload);
    const data = parseSse(res.body)[0]!.data as SseToolEvent;
    expect(data).toEqual(payload);
    expect(data.phase).toBe("result");
    if (data.phase === "result") expect(data.status).toBe("error");
  });
});

describe("Phase 9 (9a) — SseWriter.artifact()", () => {
  it("emits the typed envelope { kind, data } verbatim", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    const payload: SseArtifactEvent = {
      kind: "planning_checklist",
      data: { tasks: ["book venue", "hire caterer"] },
    };
    writer.artifact(payload);

    const frames = parseSse(res.body);
    expect(frames).toHaveLength(1);
    expect(frames[0]!.event).toBe("artifact");
    expect(frames[0]!.data).toEqual(payload);
  });
});

// --- Guard behavior preserved -----------------------------------------------

describe("Phase 9 (9a) — socket guard preserved for the new methods", () => {
  it("new methods are no-ops after end() (do not throw, write nothing)", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    writer.init("t");
    writer.done("final");
    writer.end();

    const before = res.chunks.length;
    expect(() => {
      writer.citations([makeCitation()], "supported");
      writer.tool({ phase: "call", name: "x", toolCallId: "id", args: {} });
      writer.artifact({ kind: "k", data: {} });
    }).not.toThrow();
    // No further frames were written after end().
    expect(res.chunks.length).toBe(before);
  });

  it("new methods are no-ops once the socket is writableEnded", () => {
    const res = new FakeResponse();
    res.writableEnded = true;
    const writer = new SseWriter(res.asServerResponse());

    writer.citations([makeCitation()], "supported");
    writer.tool({ phase: "call", name: "x", toolCallId: "id", args: {} });
    writer.artifact({ kind: "k", data: {} });

    expect(res.chunks).toHaveLength(0);
  });

  it("new methods are no-ops once the socket is destroyed", () => {
    const res = new FakeResponse();
    res.destroyed = true;
    const writer = new SseWriter(res.asServerResponse());

    writer.citations([makeCitation()], "supported");
    writer.tool({ phase: "result", name: "x", toolCallId: "id", status: "ok" });
    writer.artifact({ kind: "k", data: {} });

    expect(res.chunks).toHaveLength(0);
  });
});

// --- Redaction contract (writer serializes only; caller redacts) ------------

describe("Phase 9 (9a) — redaction contract (writer does NOT scrub)", () => {
  // These tests DOCUMENT + ENFORCE the writer-serializes-only (caller-redacts)
  // invariant across EVERY caller-redact free-text field of the v2 contract:
  //   - tool "result" `content`
  //   - tool "call" `args` (free-text values)
  //   - artifact `data` (free-text values)
  // Each proves the writer passes a secret-shaped string through UNMUTATED, so
  // 9b knows it MUST redact these fields at the call site BEFORE calling the
  // writer (via src/core/redaction.ts), exactly like the v1 error() event.

  it("passes tool 'result' content THROUGH unchanged (matching the error() contract)", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    // A string that WOULD be scrubbed if the writer redacted (an API-key-shaped
    // token). The writer must pass it through verbatim: redaction is the CALLER's
    // job (9b), exactly like the v1 error() event.
    const raw = "sk-ABCDEF0123456789ABCDEF0123456789";
    writer.tool({
      phase: "result",
      name: "t",
      toolCallId: "id",
      status: "error",
      content: raw,
    });

    const data = parseSse(res.body)[0]!.data as SseToolEvent;
    if (data.phase !== "result") throw new Error("expected result phase");
    expect(data.content).toBe(raw); // passed through, NOT scrubbed by the writer
  });

  it("passes tool 'call' args secret-shaped free-text THROUGH unchanged", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    // `args` is a caller-redact free-text field on the "call" phase. A
    // secret-shaped value must survive verbatim — the writer NEVER scrubs.
    const secret = "sk-THIS_SHOULD_NOT_BE_SCRUBBED_BY_WRITER";
    writer.tool({
      phase: "call",
      name: "lookup",
      toolCallId: "id",
      args: { query: secret, nested: { token: secret } },
    });

    const data = parseSse(res.body)[0]!.data as SseToolEvent;
    if (data.phase !== "call") throw new Error("expected call phase");
    expect(data.args.query).toBe(secret); // top-level arg passed through
    // Nested free-text is likewise untouched (writer does not walk/scrub).
    expect((data.args.nested as { token: string }).token).toBe(secret);
  });

  it("passes artifact data secret-shaped free-text THROUGH unchanged", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    // `data` is a caller-redact free-text-bearing field on the artifact
    // envelope. A secret-shaped value must survive verbatim.
    const secret = "sk-THIS_SHOULD_NOT_BE_SCRUBBED_BY_WRITER";
    writer.artifact({
      kind: "budget_plan",
      data: { note: secret, allocations: [{ label: secret }] },
    });

    const data = parseSse(res.body)[0]!.data as SseArtifactEvent;
    const payload = data.data as {
      note: string;
      allocations: { label: string }[];
    };
    expect(payload.note).toBe(secret); // top-level field passed through
    expect(payload.allocations[0]!.label).toBe(secret); // nested untouched too
  });

  it("passes error() message THROUGH unchanged (v1 contract regression)", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    const raw = "sk-ABCDEF0123456789ABCDEF0123456789";
    writer.error(raw);
    const data = parseSse(res.body)[0]!.data as SseErrorEvent;
    expect(data.message).toBe(raw);
  });
});

// --- Optional #3: single-line framing hardens crit-3 citation integrity -----

describe("Phase 9 (9a) — data: line stays single-line for adversarial input", () => {
  it("a citation sourceUri containing a newline produces NO raw newline in the data: line", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());
    // An adversarial (or merely messy) app-owned sourceUri with an embedded
    // newline. JSON.stringify escapes it to `\\n`, so the SSE `data:` line stays
    // single-line and the frame is unambiguous — a v2 citation cannot inject a
    // spurious frame boundary. (sourceUri is app-owned, but this proves the
    // framing is robust regardless.)
    writer.citations([makeCitation({ sourceUri: "a\nb" })], "supported");

    // Exactly one frame; its single data: line has no embedded raw newline.
    const [, dataLine] = res.body.split("\n");
    expect(dataLine!.startsWith("data: ")).toBe(true);
    const json = dataLine!.slice("data: ".length);
    expect(json.includes("\n")).toBe(false); // escaped, not a raw newline

    // The value still round-trips to the original string with its newline.
    const frames = parseSse(res.body);
    const data = frames[0]!.data as SseCitationEvent;
    expect(data.citations[0]!.sourceUri).toBe("a\nb");
  });
});

// --- Ordering / shape sanity for a representative turn ----------------------

describe("Phase 9 (9a) — a representative turn composes to well-formed frames", () => {
  it("init -> token* -> tool* -> citation -> done serializes in order", () => {
    const res = new FakeResponse();
    const writer = new SseWriter(res.asServerResponse());

    writer.init("thread-9");
    writer.token("Your ");
    writer.token("wedding ");
    writer.tool({
      phase: "call",
      name: "days_until",
      toolCallId: "c1",
      args: { date: "2028-06-01" },
    });
    writer.tool({
      phase: "result",
      name: "days_until",
      toolCallId: "c1",
      status: "ok",
      content: "512 days",
    });
    writer.token("is in 512 days.");
    writer.citations([makeCitation()], "supported");
    writer.done("Your wedding is in 512 days.");
    writer.end();

    const frames = parseSse(res.body);
    expect(frames.map((f) => f.event)).toEqual([
      "init",
      "token",
      "token",
      "tool",
      "tool",
      "token",
      "citation",
      "done",
    ]);

    // init opens with version 2; done carries the full text.
    expect((frames[0]!.data as SseInitEvent).version).toBe(2);
    expect((frames.at(-1)!.data as SseDoneEvent).text).toBe(
      "Your wedding is in 512 days."
    );

    // The whole body is a sequence of well-formed frames terminated by blank
    // lines (no dangling partial frame).
    expect(res.body.endsWith("\n\n")).toBe(true);
    // Every non-empty block parsed to a frame (no unparseable data).
    for (const f of frames) expect(f.data).not.toBeUndefined();
  });
});
