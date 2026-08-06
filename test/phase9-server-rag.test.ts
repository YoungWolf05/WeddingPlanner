import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { AddressInfo } from "node:net";
import type { Server } from "node:http";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";

// Phase 9 (9b) — SERVER RAG/AGENT WIRING (v2 SSE events).
//
// FULLY OFFLINE + DETERMINISTIC (same discipline as the Phase 5 server suites):
//   - The grounded path is driven by an INJECTED scripted `answerTurn` returning
//     a hand-built GroundedAnswerResult — NO embeddings, NO model, NO network.
//   - The tool-event translator is unit-tested directly over AIMessage/ToolMessage.
//   - Servers listen on an EPHEMERAL port (127.0.0.1:0); requests use fetch.
//   - The ThreadStore is the REAL temp-SQLite store so ownership/auth are
//     genuinely enforced (proving 9b did not weaken Phase 5). DBs live under
//     os.tmpdir(), removed in afterEach — no repo artifacts.
//   - The listening entrypoint (src/run-server.ts) is NEVER imported here.
//
// NOTE: NO vi.mock of the model boundary is needed for the grounded path — it
// bypasses `createChat` entirely and runs through the injected answerTurn.

const { createServer, GROUNDED_ANSWER_ARTIFACT_KIND } = await import(
  "../src/core/server.js"
);
const { createCheckpointer, sessionConfig } = await import(
  "../src/core/memory.js"
);
const { createThreadStore } = await import("../src/core/threads.js");
const { createTokenAuthenticator } = await import("../src/core/auth.js");
const { SSE_PROTOCOL_VERSION } = await import("../src/core/sse.js");
const { toolEventsForMessage } = await import("../src/core/agent-sse.js");
const { config } = await import("../src/config.js");
type GroundedAnswerResult = import("../src/core/rag.js").GroundedAnswerResult;
type AnswerTurn = import("../src/core/grounded-turn.js").AnswerTurn;
type TrustedCitation = import("../src/core/citations.js").TrustedCitation;
type StreamingChat = import("../src/core/server.js").StreamingChat;

function makeSaver(dbPath: string): SqliteSaver {
  const saver = createCheckpointer(dbPath);
  if (!(saver instanceof SqliteSaver)) throw new Error("expected a SqliteSaver");
  return saver;
}

const TOKEN_ALICE = "test-token-alice";
const TOKEN_BOB = "test-token-bob";
const USER_ALICE = "user-alice";
const USER_BOB = "user-bob";

function makeAuth() {
  return createTokenAuthenticator({
    [TOKEN_ALICE]: USER_ALICE,
    [TOKEN_BOB]: USER_BOB,
  });
}

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
}

async function readJson(res: Response): Promise<Record<string, any>> {
  return (await res.json()) as Record<string, any>;
}

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
    frames.push({ event, data: dataLine === "" ? undefined : JSON.parse(dataLine) });
  }
  return frames;
}

// A `createChat` that must NEVER be called on the grounded path — if it is, the
// test fails loudly (proving the grounded path bypasses the plain-chat graph).
function forbiddenCreateChat(): StreamingChat {
  throw new Error("createChat must not run on the grounded path");
}

// Build a TrustedCitation with sensible app-owned defaults.
function citation(over: Partial<TrustedCitation> = {}): TrustedCitation {
  return {
    marker: 1,
    chunkId: "chunk-1",
    documentId: "doc-1",
    sourceUri: "knowledge/corpus/venues.md",
    chunkIndex: 0,
    ownerId: USER_ALICE,
    contentHash: "hash-abc",
    score: 0.87,
    ...over,
  };
}

// Build a scripted GroundedAnswerResult for the injected answerTurn.
function groundedResult(over: Partial<GroundedAnswerResult> = {}): GroundedAnswerResult {
  return {
    answer: { answer: "Consider an outdoor garden venue.", citations: [1], insufficientEvidence: false },
    resolvedCitations: [citation()],
    droppedCitations: [],
    markerMap: new Map(),
    retrieved: [],
    contextBlock: "",
    evidenceStatus: "supported",
    ...over,
  };
}

describe("Phase 9 (9b) — grounded chat turn over v2 SSE", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  async function startServer(answerTurn: AnswerTurn, log?: (l: string) => void) {
    const store = createThreadStore(saver);
    server = createServer({
      store,
      auth: makeAuth(),
      createChat: forbiddenCreateChat,
      answerTurn,
      log: log ?? (() => {}),
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    return store;
  }

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-9b-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
  });
  afterEach(async () => {
    if (server) await new Promise<void>((r) => server.close(() => r()));
    try {
      saver.db.close();
    } catch {
      /* already closed */
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  it("happy path: init(v2) -> token(answer) -> citation -> artifact -> done -> end", async () => {
    const scripted = groundedResult();
    let receivedOwnerId: string | undefined;
    let receivedQuery: string | undefined;
    const store = await startServer(async ({ query, ownerId }) => {
      receivedOwnerId = ownerId;
      receivedQuery = query;
      return scripted;
    });

    const t = store.createThread(USER_ALICE, { title: "grounded" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "recommend a venue" }),
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const frames = parseSse(await res.text());
    const order = frames.map((f) => f.event);

    // init is first and versioned v2.
    expect(frames[0]!.event).toBe("init");
    expect((frames[0]!.data as { version: number }).version).toBe(SSE_PROTOCOL_VERSION);
    expect((frames[0]!.data as { threadId: string }).threadId).toBe(t.id);

    // Event ORDER: init -> token -> citation -> artifact -> done.
    expect(order).toEqual(["init", "token", "citation", "artifact", "done"]);

    // token carries the full answer.
    const tokenFrame = frames.find((f) => f.event === "token")!;
    expect((tokenFrame.data as { text: string }).text).toBe(
      "Consider an outdoor garden venue."
    );

    // done carries the same full text.
    const doneFrame = frames.at(-1)!;
    expect((doneFrame.data as { text: string }).text).toBe(
      "Consider an outdoor garden venue."
    );

    // The authenticated ownerId + the message flowed into answerTurn.
    expect(receivedOwnerId).toBe(USER_ALICE);
    expect(receivedQuery).toBe("recommend a venue");
  });

  it("citation event carries ONLY app-owned TrustedCitation fields (ownerId NOT on the wire) + evidenceStatus", async () => {
    const store = await startServer(async () =>
      groundedResult({
        resolvedCitations: [
          citation({ marker: 3, chunkId: "c-3", documentId: "d-3", chunkIndex: 2 }),
        ],
        evidenceStatus: "supported",
      })
    );

    const t = store.createThread(USER_ALICE, { title: "cite" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "cite something" }),
    });
    const frames = parseSse(await res.text());
    const citeFrame = frames.find((f) => f.event === "citation")!;
    const data = citeFrame.data as {
      citations: Array<Record<string, unknown>>;
      evidenceStatus: string;
    };
    expect(data.evidenceStatus).toBe("supported");
    expect(data.citations).toHaveLength(1);
    const c = data.citations[0]!;
    // App-owned wire fields present.
    expect(c.marker).toBe(3);
    expect(c.chunkId).toBe("c-3");
    expect(c.documentId).toBe("d-3");
    expect(c.chunkIndex).toBe(2);
    expect(c.score).toBe(0.87);
    expect(c.contentHash).toBe("hash-abc");
    // ownerId is DELIBERATELY dropped from the wire projection.
    expect(c).not.toHaveProperty("ownerId");
  });

  it("dropped/hallucinated markers NEVER reach the wire (only resolvedCitations are emitted)", async () => {
    // The result has a dropped marker (99) but only marker 1 resolved. The wire
    // citation event must contain ONLY the resolved citation, never the dropped.
    const store = await startServer(async () =>
      groundedResult({
        answer: { answer: "grounded", citations: [1, 99], insufficientEvidence: false },
        resolvedCitations: [citation({ marker: 1 })],
        droppedCitations: [{ marker: 99, reason: "unknown_marker" }],
      })
    );

    const t = store.createThread(USER_ALICE, { title: "drop" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "q" }),
    });
    const body = await res.text();
    // The dropped marker 99 must not appear anywhere on the wire.
    expect(body).not.toContain("99");
    const frames = parseSse(body);
    const citeFrame = frames.find((f) => f.event === "citation")!;
    const markers = (citeFrame.data as { citations: Array<{ marker: number }> }).citations.map(
      (x) => x.marker
    );
    expect(markers).toEqual([1]);
  });

  it("insufficient evidence: empty citations, evidenceStatus 'insufficient', no fabricated citation", async () => {
    const store = await startServer(async () =>
      groundedResult({
        answer: { answer: "", citations: [], insufficientEvidence: true },
        resolvedCitations: [],
        evidenceStatus: "insufficient",
      })
    );

    const t = store.createThread(USER_ALICE, { title: "insuff" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "obscure question" }),
    });
    const frames = parseSse(await res.text());
    const citeFrame = frames.find((f) => f.event === "citation")!;
    const data = citeFrame.data as {
      citations: unknown[];
      evidenceStatus: string;
    };
    expect(data.evidenceStatus).toBe("insufficient");
    expect(data.citations).toEqual([]);
    // Empty answer => no token frame (nothing to show), but citation+artifact+done still emitted.
    expect(frames.some((f) => f.event === "token")).toBe(false);
    expect(frames.some((f) => f.event === "artifact")).toBe(true);
    expect(frames.at(-1)!.event).toBe("done");
  });

  it("artifact event: kind = grounded_answer with redacted answer + evidenceStatus", async () => {
    const store = await startServer(async () =>
      groundedResult({ evidenceStatus: "supported" })
    );
    const t = store.createThread(USER_ALICE, { title: "art" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "q" }),
    });
    const frames = parseSse(await res.text());
    const artFrame = frames.find((f) => f.event === "artifact")!;
    const data = artFrame.data as { kind: string; data: Record<string, unknown> };
    expect(data.kind).toBe(GROUNDED_ANSWER_ARTIFACT_KIND);
    expect(data.data.evidenceStatus).toBe("supported");
    expect(data.data.answer).toBe("Consider an outdoor garden venue.");
  });

  it("REDACTION: a secret/PII-shaped answer is redacted on the token AND artifact", async () => {
    const leaky = `Book it at ${config.apiKey} via ${config.baseURL} email planner@example.com`;
    const store = await startServer(async () =>
      groundedResult({
        answer: { answer: leaky, citations: [], insufficientEvidence: false },
        resolvedCitations: [citation()],
      })
    );
    const t = store.createThread(USER_ALICE, { title: "leak" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "q" }),
    });
    const body = await res.text();
    expect(body).not.toContain(config.apiKey);
    expect(body).not.toContain(config.baseURL);
    expect(body).not.toContain("planner@example.com");
    expect(body).toContain("[redacted-key]");
  });

  it("REDACTION: an answerTurn error surfaces as a REDACTED sse.error (no secret/PII)", async () => {
    const leaky = `boom apiKey=${config.apiKey} url=${config.baseURL} contact planner@example.com`;
    const logs: string[] = [];
    const store = await startServer(async () => {
      throw new Error(leaky);
    }, (l) => logs.push(l));

    const t = store.createThread(USER_ALICE, { title: "err" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "q" }),
    });
    const frames = parseSse(await res.text());
    const errFrame = frames.find((f) => f.event === "error");
    expect(errFrame).toBeDefined();
    const message = (errFrame!.data as { message: string }).message;
    expect(message).not.toContain(config.apiKey);
    expect(message).not.toContain(config.baseURL);
    expect(message).not.toContain("planner@example.com");
    expect(message).toContain("[redacted-key]");
    // No done frame on the error path.
    expect(frames.some((f) => f.event === "done")).toBe(false);
    // The server log is redacted too.
    const joined = logs.join("\n");
    expect(joined).not.toContain(config.apiKey);
    expect(joined).not.toContain(config.baseURL);
  });
});

// ---------------------------------------------------------------------------
// AUTH / OWNERSHIP REGRESSION (unchanged from Phase 5) on the grounded server.
// Prove 9b did not weaken auth/ownership: the ownership gate runs BEFORE any
// grounded work, ownerId comes ONLY from the token, and not-owned == not-found.
// ---------------------------------------------------------------------------
describe("Phase 9 (9b) — auth/ownership regression on the grounded server", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  // answerTurn tracks whether it ran, so we can assert it NEVER runs for a
  // denied/unauthenticated request (no grounded work, no state created).
  let answerTurnRan: boolean;

  beforeEach(async () => {
    answerTurnRan = false;
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-9b-auth-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
    const store = createThreadStore(saver);
    server = createServer({
      store,
      auth: makeAuth(),
      createChat: forbiddenCreateChat,
      answerTurn: async () => {
        answerTurnRan = true;
        return groundedResult();
      },
      log: () => {},
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });
  afterEach(async () => {
    if (server) await new Promise<void>((r) => server.close(() => r()));
    try {
      saver.db.close();
    } catch {
      /* already closed */
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  it("unauthenticated chat => 401 generic; answerTurn never runs", async () => {
    const res = await fetch(`${baseUrl}/threads/some-id/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hi" }),
    });
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual({ error: "Unauthorized" });
    expect(answerTurnRan).toBe(false);
  });

  it("not-owned/nonexistent thread => identical 404, NO SSE headers, NO state, answerTurn never runs", async () => {
    const store = createThreadStore(saver);
    const alice = store.createThread(USER_ALICE, { title: "alice-only" });
    const randomId = "33333333-3333-4333-8333-333333333333";

    // Bob (authenticated) tries Alice's thread AND a random id: identical 404.
    const bobOwned = await fetch(`${baseUrl}/threads/${alice.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_BOB),
      body: JSON.stringify({ message: "hi" }),
    });
    const bobRandom = await fetch(`${baseUrl}/threads/${randomId}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_BOB),
      body: JSON.stringify({ message: "hi" }),
    });
    expect(bobOwned.status).toBe(404);
    expect(bobRandom.status).toBe(404);
    expect(bobOwned.headers.get("content-type")).not.toContain("text/event-stream");
    expect(await bobOwned.json()).toEqual(await bobRandom.json());

    // No grounded work ran; no checkpoint state created for either thread_id.
    expect(answerTurnRan).toBe(false);
    expect(await saver.getTuple(sessionConfig(randomId))).toBeUndefined();
  });

  it("ownerId comes from the TOKEN not client input: spoofed body owner ignored", async () => {
    const store = createThreadStore(saver);
    const alice = store.createThread(USER_ALICE, { title: "a" });
    // Bob tries to reach Alice's thread by spoofing an owner id in the body — the
    // ownership gate keys off the token (Bob), so this is a 404.
    const res = await fetch(`${baseUrl}/threads/${alice.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_BOB),
      body: JSON.stringify({ message: "hi", ownerId: USER_ALICE }),
    });
    expect(res.status).toBe(404);
    expect(answerTurnRan).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Cancellation + idle timeout on the grounded path.
// ---------------------------------------------------------------------------
describe("Phase 9 (9b) — grounded cancellation & idle timeout", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-9b-cx-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
  });
  afterEach(async () => {
    if (server) await new Promise<void>((r) => server.close(() => r()));
    try {
      saver.db.close();
    } catch {
      /* already closed */
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  it("a stalled grounded turn is aborted at the idle timeout: redacted error, no citation/done, server survives", async () => {
    const store = createThreadStore(saver);
    let signalAborted = false;
    server = createServer({
      store,
      auth: makeAuth(),
      createChat: forbiddenCreateChat,
      log: () => {},
      timeouts: { sseIdleTimeoutMs: 40 },
      // answerTurn NEVER resolves until the injected signal aborts, so the idle
      // watchdog must fire and tear the turn down.
      answerTurn: ({ signal }) =>
        new Promise((_resolve) => {
          const onAbort = () => {
            signalAborted = true;
            // The idle watchdog has already fired and aborted the turn, so
            // resolve with a grounded result here; the server's timedOut/aborted
            // guards suppress any emission, so this resolution is intentionally
            // ignored (no citation/done frame is sent).
            _resolve(groundedResult());
          };
          if (signal.aborted) onAbort();
          else signal.addEventListener("abort", onAbort);
        }),
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

    const t = store.createThread(USER_ALICE, { title: "stall" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "hi" }),
    });
    const frames = parseSse(await res.text());
    // init -> error (timed out), and NO citation/done (the turn was torn down).
    expect(frames[0]!.event).toBe("init");
    const errFrame = frames.find((f) => f.event === "error");
    expect(errFrame).toBeDefined();
    expect((errFrame!.data as { message: string }).message).toContain("timed out");
    expect(frames.some((f) => f.event === "citation")).toBe(false);
    expect(frames.some((f) => f.event === "done")).toBe(false);
    expect(signalAborted).toBe(true);

    // Server still serves.
    expect((await fetch(`${baseUrl}/healthz`)).status).toBe(200);
  });

  it("client disconnect before generation aborts (answerTurn sees an aborted signal; no emit after close)", async () => {
    const store = createThreadStore(saver);
    let sawAbortedSignal = false;
    const firstReadReady = { resolve: (): void => {} };
    const gate = new Promise<void>((r) => {
      firstReadReady.resolve = r;
    });

    server = createServer({
      store,
      auth: makeAuth(),
      createChat: forbiddenCreateChat,
      log: () => {},
      answerTurn: async ({ signal }) => {
        // Wait until the client has disconnected (signal aborts), then observe it.
        await new Promise<void>((resolve) => {
          const onAbort = () => {
            sawAbortedSignal = signal.aborted;
            resolve();
          };
          if (signal.aborted) onAbort();
          else signal.addEventListener("abort", onAbort);
        });
        firstReadReady.resolve();
        return groundedResult();
      },
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

    const t = store.createThread(USER_ALICE, { title: "disc" });
    const clientAbort = new AbortController();
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "hi" }),
      signal: clientAbort.signal,
    });
    const reader = res.body!.getReader();
    await reader.read(); // init frame
    // Disconnect mid-turn (before answerTurn resolves).
    clientAbort.abort();
    await reader.read().catch(() => {});
    reader.cancel().catch(() => {});

    // Wait (bounded) for the server to observe abort inside answerTurn.
    await Promise.race([gate, new Promise((r) => setTimeout(r, 1000))]);
    expect(sawAbortedSignal).toBe(true);

    // Server did not crash.
    expect((await fetch(`${baseUrl}/healthz`)).status).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// Tool-event translator (agent-sse.ts) — the crit-3 tool progress/errors path.
// Unit-tested directly (the live agent-into-endpoint hookup is deferred; the
// translator + its redaction are the demonstrable mechanism for 9b).
// ---------------------------------------------------------------------------
describe("Phase 9 (9b) — agent tool-event translator", () => {
  it("AIMessage.tool_calls -> one 'call' event per tool call (name, id, parsed args)", () => {
    // NOTE: non-digit-sequence string values are used here so the pass-through
    // (non-secret text survives) is what's under test; digit sequences like a
    // dashed date would legitimately match redactText's phone pattern (proven
    // separately in the redaction tests). Numeric/boolean args are structural.
    const msg = new AIMessage({
      content: "",
      tool_calls: [
        { name: "lookup", args: { topic: "outdoor venues" }, id: "call-1", type: "tool_call" },
        { name: "split_budget", args: { total: 1000, parts: 4 }, id: "call-2", type: "tool_call" },
      ],
    });
    const events = toolEventsForMessage(msg);
    expect(events).toHaveLength(2);
    expect(events[0]).toEqual({
      phase: "call",
      name: "lookup",
      toolCallId: "call-1",
      args: { topic: "outdoor venues" },
    });
    expect(events[1]).toEqual({
      phase: "call",
      name: "split_budget",
      toolCallId: "call-2",
      args: { total: 1000, parts: 4 },
    });
  });

  it("ToolMessage (ok) -> a 'result' event with status ok + redacted content", () => {
    const tm = new ToolMessage({
      content: "45 days away",
      tool_call_id: "call-1",
      name: "days_until",
    });
    const events = toolEventsForMessage(tm);
    expect(events).toHaveLength(1);
    expect(events[0]).toEqual({
      phase: "result",
      name: "days_until",
      toolCallId: "call-1",
      status: "ok",
      content: "45 days away",
    });
  });

  it("ToolMessage with status 'error' -> a 'result' event with status error", () => {
    const tm = new ToolMessage({
      content: "that date is not a real calendar date",
      tool_call_id: "call-9",
      name: "days_until",
      status: "error",
    });
    const events = toolEventsForMessage(tm);
    expect(events[0]).toMatchObject({
      phase: "result",
      status: "error",
      toolCallId: "call-9",
      name: "days_until",
    });
  });

  it("REDACTION: a secret/PII-shaped value in tool args AND result content is scrubbed", () => {
    const callMsg = new AIMessage({
      content: "",
      tool_calls: [
        {
          name: "lookup",
          args: { note: `key ${config.apiKey} at ${config.baseURL} mail planner@example.com` },
          id: "call-1",
          type: "tool_call",
        },
      ],
    });
    const callEvents = toolEventsForMessage(callMsg);
    const firstCall = callEvents[0]!;
    if (firstCall.phase !== "call") throw new Error("expected a call event");
    const argsNote = firstCall.args.note as string;
    expect(argsNote).not.toContain(config.apiKey);
    expect(argsNote).not.toContain(config.baseURL);
    expect(argsNote).not.toContain("planner@example.com");
    expect(argsNote).toContain("[redacted-key]");

    const resultMsg = new ToolMessage({
      content: `result key=${config.apiKey} url=${config.baseURL} planner@example.com`,
      tool_call_id: "call-1",
      name: "lookup",
    });
    const resultEvents = toolEventsForMessage(resultMsg);
    const firstResult = resultEvents[0]!;
    if (firstResult.phase !== "result") throw new Error("expected a result event");
    const content = firstResult.content ?? "";
    expect(content).not.toContain(config.apiKey);
    expect(content).not.toContain(config.baseURL);
    expect(content).not.toContain("planner@example.com");
    expect(content).toContain("[redacted-key]");
  });

  it("REDACTION: a secret buried in a NESTED args field is scrubbed", () => {
    const msg = new AIMessage({
      content: "",
      tool_calls: [
        {
          name: "nested",
          args: { outer: { inner: [`leak ${config.apiKey}`] }, count: 3 },
          id: "call-1",
          type: "tool_call",
        },
      ],
    });
    const events = toolEventsForMessage(msg);
    const ev = events[0]!;
    if (ev.phase !== "call") throw new Error("expected a call event");
    const args = ev.args;
    expect(JSON.stringify(args)).not.toContain(config.apiKey);
    expect(JSON.stringify(args)).toContain("[redacted-key]");
    // Non-string primitives are preserved structurally.
    expect(args.count).toBe(3);
  });

  it("non-tool messages (final answer, human, system) yield no tool events", () => {
    expect(toolEventsForMessage(new AIMessage({ content: "final answer" }))).toEqual([]);
    expect(toolEventsForMessage(new HumanMessage("hi"))).toEqual([]);
    expect(toolEventsForMessage(new SystemMessage("persona"))).toEqual([]);
  });
});
