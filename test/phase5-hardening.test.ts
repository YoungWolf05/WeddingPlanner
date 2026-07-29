import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { AddressInfo } from "node:net";
import type { Server } from "node:http";
import {
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { recordedCalls, resetRecordedCalls } from "./helpers/fake-model.js";

// Phase 5 (5d) — hardening, concurrency, restart-durability, and retention.
//
// FULLY OFFLINE + DETERMINISTIC (same discipline as the 5c suite):
//   - The model boundary is mocked (fake-model), so the REAL conversational
//     graph + REAL SQLite checkpointer run without any network/creds.
//   - Servers listen on an EPHEMERAL port (127.0.0.1:0); requests use fetch.
//   - Every SQLite DB lives in a per-test temp dir under os.tmpdir(), removed in
//     afterEach — no db artifacts land in the repo.
//   - Timeout values are INJECTED as tiny numbers so the R1 timeout paths are
//     exercised deterministically, with NO real 10s/30s/60s wall-clock waits.
//   - The listening entrypoint (src/run-server.ts) is NEVER imported here.
vi.mock("../src/core/model.js", async () => {
  const { makeFakeChatModel } = await import("./helpers/fake-model.js");
  return {
    createChatModel: () =>
      makeFakeChatModel({
        responses: [
          "reply one",
          "reply two",
          "reply three",
          "reply four",
          "reply five",
          "reply six",
        ],
        sleep: 0,
      }),
  };
});

const { createServer, DEFAULT_TIMEOUTS, resolveTimeouts, parseTimeoutMs } =
  await import("../src/core/server.js");
const { createConversationalChain } = await import("../src/core/chain.js");
const { createCheckpointer, sessionConfig } = await import(
  "../src/core/memory.js"
);
const { createThreadStore } = await import("../src/core/threads.js");
type ThreadStore = import("../src/core/threads.js").ThreadStore;
const { createTokenAuthenticator } = await import("../src/core/auth.js");

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

// Read a JSON response body as an arbitrary record (tests assert on fields).
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

// Human-message texts recorded for a given model call index.
function humanTextsOf(call: BaseMessage[]): string[] {
  return call
    .filter((m): m is HumanMessage => m instanceof HumanMessage)
    .map((m) => String(m.content));
}

// ---------------------------------------------------------------------------
// R1 — server-level timeouts are applied to the real http.Server
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — R1 server-level timeouts", () => {
  let tempDir: string;
  let saver: SqliteSaver;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-5d-to-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
  });
  afterEach(async () => {
    try {
      saver.db.close();
    } catch {
      /* already closed */
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  it("resolveTimeouts merges partials onto the approved defaults", () => {
    expect(DEFAULT_TIMEOUTS).toEqual({
      headersTimeoutMs: 10_000,
      requestTimeoutMs: 30_000,
      sseIdleTimeoutMs: 60_000,
    });
    expect(resolveTimeouts()).toEqual(DEFAULT_TIMEOUTS);
    expect(resolveTimeouts({ requestTimeoutMs: 1234 })).toEqual({
      headersTimeoutMs: 10_000,
      requestTimeoutMs: 1234,
      sseIdleTimeoutMs: 60_000,
    });
  });

  it("createServer applies headersTimeout/requestTimeout from injected options", () => {
    const store = createThreadStore(saver);
    const server = createServer({
      store,
      auth: makeAuth(),
      createChat: () => createConversationalChain({ streaming: true }, saver),
      log: () => {},
      timeouts: { headersTimeoutMs: 1234, requestTimeoutMs: 5678 },
    });
    expect(server.headersTimeout).toBe(1234);
    expect(server.requestTimeout).toBe(5678);
  });

  it("createServer falls back to the approved defaults when timeouts omitted", () => {
    const store = createThreadStore(saver);
    const server = createServer({
      store,
      auth: makeAuth(),
      createChat: () => createConversationalChain({ streaming: true }, saver),
      log: () => {},
    });
    expect(server.headersTimeout).toBe(DEFAULT_TIMEOUTS.headersTimeoutMs);
    expect(server.requestTimeout).toBe(DEFAULT_TIMEOUTS.requestTimeoutMs);
  });
});

// ---------------------------------------------------------------------------
// N-2 — parseTimeoutMs: reject 0 (silent-disable) as loudly as negatives.
// Pure unit test; the same parser backs all three SERVICE_*_TIMEOUT_MS envs, so
// one parser covers headers/request/idle (see src/run-server.ts wiring).
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — N-2 parseTimeoutMs rejects 0", () => {
  const NAME = "SERVICE_HEADERS_TIMEOUT_MS";

  it("returns the fallback when unset/empty (no override)", () => {
    expect(parseTimeoutMs(undefined, NAME, 10_000)).toBe(10_000);
    expect(parseTimeoutMs("", NAME, 30_000)).toBe(30_000);
  });

  it("accepts a positive integer", () => {
    expect(parseTimeoutMs("5000", NAME, 10_000)).toBe(5_000);
    expect(parseTimeoutMs("1", NAME, 10_000)).toBe(1);
  });

  it("REJECTS 0 loudly (0 would silently DISABLE the timeout in Node)", () => {
    expect(() => parseTimeoutMs("0", NAME, 10_000)).toThrow(
      /Must be a positive integer/
    );
    // The offending value + env name are surfaced for a loud, actionable error.
    expect(() => parseTimeoutMs("0", NAME, 10_000)).toThrow(NAME);
  });

  it("still rejects negatives, non-integers, and non-numeric values", () => {
    for (const bad of ["-1", "-1000", "1.5", "12.34", "abc", "10ms", "NaN"]) {
      expect(() => parseTimeoutMs(bad, NAME, 10_000)).toThrow(
        /Must be a positive integer/
      );
    }
  });

  it("applies identically for the request and idle timeout env names (shared parser)", () => {
    // The reject-0 rule holds regardless of which SERVICE_*_TIMEOUT_MS env it
    // parses; run-server.ts routes all three through this one function.
    expect(() =>
      parseTimeoutMs("0", "SERVICE_REQUEST_TIMEOUT_MS", 30_000)
    ).toThrow(/Must be a positive integer/);
    expect(() =>
      parseTimeoutMs("0", "SERVICE_SSE_IDLE_TIMEOUT_MS", 60_000)
    ).toThrow(/Must be a positive integer/);
    expect(parseTimeoutMs("45000", "SERVICE_SSE_IDLE_TIMEOUT_MS", 60_000)).toBe(
      45_000
    );
  });
});

// ---------------------------------------------------------------------------
// R1 — SSE idle-stream timeout aborts a stalled turn with a redacted error
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — R1 SSE idle timeout", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-5d-idle-"));
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

  it("a stalled stream (no token) is aborted at the idle timeout: redacted error event, no done, stream ends, server survives", async () => {
    const store = createThreadStore(saver);
    const auth = makeAuth();

    // A fake graph that NEVER emits a token and only unblocks when the injected
    // signal aborts — i.e. it stalls until the idle watchdog fires.
    let signalAborted = false;
    server = createServer({
      store,
      auth,
      log: () => {},
      // Tiny idle timeout so the watchdog fires promptly and deterministically.
      timeouts: { sseIdleTimeoutMs: 40 },
      createChat: () => ({
        async stream(_input, options): Promise<AsyncIterable<[BaseMessage, unknown]>> {
          const signal = options.signal;
          async function* gen(): AsyncGenerator<[BaseMessage, unknown]> {
            await new Promise<void>((resolve) => {
              if (signal.aborted) {
                signalAborted = true;
                resolve();
                return;
              }
              signal.addEventListener("abort", () => {
                signalAborted = true;
                resolve();
              });
            });
            // Emit nothing further.
          }
          return gen();
        },
      }),
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

    const t = store.createThread(USER_ALICE, { title: "stall" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "hello" }),
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const frames = parseSse(await res.text());
    // init -> error, and NO done frame.
    expect(frames[0]!.event).toBe("init");
    const errorFrame = frames.find((f) => f.event === "error");
    expect(errorFrame).toBeDefined();
    expect((errorFrame!.data as { message: string }).message).toContain(
      "timed out"
    );
    expect(frames.some((f) => f.event === "done")).toBe(false);

    // The in-flight run was aborted (integrates with the AbortController).
    expect(signalAborted).toBe(true);

    // Server did not crash: it still serves.
    const health = await fetch(`${baseUrl}/healthz`);
    expect(health.status).toBe(200);
  });

  it("N-1: a late buffered chunk after the idle timeout does NOT re-arm the timer or emit a second idle-timeout (one error, one end, no token)", async () => {
    // Regression guard for N-1. The fake stalls until the watchdog aborts, then
    // emits ONE MORE buffered chunk (as a real graph might before it honors the
    // abort) and STALLS AGAIN long enough that a RE-ARMED tiny timer would fire.
    // Without the `!timedOut` guard the token branch would call armIdleTimer()
    // again, producing a SECOND `idle-timed-out` log line + redundant abort. With
    // the fix there is exactly one.
    const store = createThreadStore(saver);
    const logs: string[] = [];
    let abortCount = 0;

    server = createServer({
      store,
      auth: makeAuth(),
      log: (line) => logs.push(line),
      timeouts: { sseIdleTimeoutMs: 30 },
      createChat: () => ({
        async stream(_input, options): Promise<AsyncIterable<[BaseMessage, unknown]>> {
          const signal = options.signal;
          signal.addEventListener("abort", () => {
            abortCount += 1;
          });
          async function* gen(): AsyncGenerator<[BaseMessage, unknown]> {
            // 1) Stall until the idle watchdog aborts (~30ms).
            await new Promise<void>((resolve) => {
              if (signal.aborted) resolve();
              else signal.addEventListener("abort", () => resolve());
            });
            // 2) Emit one LATE buffered chunk AFTER the timeout has fired.
            yield [new AIMessageChunk({ content: "late-token" }), {}];
            // 3) Stall again past a re-armed 30ms timer's window so the N-1
            //    regression (second idle-timeout) would deterministically occur.
            await new Promise<void>((resolve) => setTimeout(resolve, 80));
          }
          return gen();
        },
      }),
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

    const t = store.createThread(USER_ALICE, { title: "late-chunk" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "hello" }),
    });
    const frames = parseSse(await res.text());

    // Wire contract: init -> exactly one error -> end; NO token, NO done. The
    // late token was NOT written after the stream ended.
    expect(frames[0]!.event).toBe("init");
    const errorFrames = frames.filter((f) => f.event === "error");
    expect(errorFrames.length).toBe(1);
    expect((errorFrames[0]!.data as { message: string }).message).toContain(
      "timed out"
    );
    expect(frames.some((f) => f.event === "token")).toBe(false);
    expect(frames.some((f) => f.event === "done")).toBe(false);

    // Wait past the window in which a RE-ARMED tiny timer would have fired, then
    // assert exactly ONE idle-timeout log line (no re-arm) and no redundant abort
    // storm from the watchdog path.
    await new Promise<void>((r) => setTimeout(r, 150));
    const idleTimeoutLogs = logs.filter((l) => l.includes("idle-timed-out"));
    expect(idleTimeoutLogs.length).toBe(1);
    // The controller is aborted exactly once (idempotent abort; single teardown).
    expect(abortCount).toBe(1);

    // Server still serves.
    const health = await fetch(`${baseUrl}/healthz`);
    expect(health.status).toBe(200);
  });

  it("an actively-streaming turn is NOT cut off (idle timer resets on each token)", async () => {
    // With a real (mocked) model streaming char-by-char and a modest idle
    // timeout, the reset-on-token logic keeps the turn alive to a clean `done`.
    const store = createThreadStore(saver);
    server = createServer({
      store,
      auth: makeAuth(),
      log: () => {},
      timeouts: { sseIdleTimeoutMs: 1_000 },
      createChat: () => createConversationalChain({ streaming: true }, saver),
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

    const t = store.createThread(USER_ALICE, { title: "active" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "plan" }),
    });
    const frames = parseSse(await res.text());
    expect(frames.some((f) => f.event === "error")).toBe(false);
    expect(frames.at(-1)!.event).toBe("done");
    const accumulated = frames
      .filter((f) => f.event === "token")
      .map((f) => (f.data as { text: string }).text)
      .join("");
    expect(accumulated).toBe("reply one");
  });
});

// ---------------------------------------------------------------------------
// R3 — prototype-pollution hardening at the JSON body boundary
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — R3 prototype-pollution guard", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-5d-proto-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
    const store = createThreadStore(saver);
    server = createServer({
      store,
      auth: makeAuth(),
      createChat: () => createConversationalChain({ streaming: true }, saver),
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

  it("a NORMAL body still works (201)", async () => {
    const res = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ title: "normal" }),
    });
    expect(res.status).toBe(201);
    expect((await readJson(res)).thread.title).toBe("normal");
  });

  it("a body with a top-level __proto__ key => 400 and does NOT pollute Object.prototype", async () => {
    const res = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      // Raw string: JSON.stringify of an object would drop __proto__.
      body: '{"__proto__":{"polluted":true},"title":"x"}',
    });
    expect(res.status).toBe(400);
    // No prototype pollution occurred.
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    expect(
      (Object.prototype as unknown as Record<string, unknown>).polluted
    ).toBeUndefined();
  });

  it("top-level constructor and prototype keys are also rejected (400)", async () => {
    const withConstructor = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: '{"constructor":{"prototype":{"polluted2":true}}}',
    });
    expect(withConstructor.status).toBe(400);

    const withPrototype = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: '{"prototype":1}',
    });
    expect(withPrototype.status).toBe(400);

    expect(
      (Object.prototype as unknown as Record<string, unknown>).polluted2
    ).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// R4 — the clean 413 still holds for an oversized body (bounded by R1 timeout)
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — R4 oversized body still returns 413", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-5d-413-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
    const store = createThreadStore(saver);
    server = createServer({
      store,
      auth: makeAuth(),
      createChat: () => createConversationalChain({ streaming: true }, saver),
      log: () => {},
      // The requestTimeout is what BOUNDS a slow oversized upload's drain; a
      // normal (fast) oversized body still gets the clean 413 below.
      timeouts: { requestTimeoutMs: 5_000 },
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

  it("a normal oversized body returns a clean 413; the request timeout bounds the drain", async () => {
    const store = createThreadStore(saver);
    const t = store.createThread(USER_ALICE, { title: "big" });
    const huge = "x".repeat(70 * 1024); // > 64 KiB cap
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: huge }),
    });
    expect(res.status).toBe(413);
    // The server-level requestTimeout is what bounds a SLOW oversized upload.
    expect(server.requestTimeout).toBe(5_000);
  });
});

// ---------------------------------------------------------------------------
// Concurrency — validate the shared single connection under concurrent ops
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — concurrency", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  async function drain(res: Response): Promise<void> {
    await res.text();
  }

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-5d-conc-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
    const store = createThreadStore(saver);
    server = createServer({
      store,
      auth: makeAuth(),
      createChat: () => createConversationalChain({ streaming: true }, saver),
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

  it("many concurrent createThread calls for DIFFERENT users all succeed and stay owner-scoped", async () => {
    const N = 6;
    const requests: Promise<Response>[] = [];
    for (let i = 0; i < N; i++) {
      requests.push(
        fetch(`${baseUrl}/threads`, {
          method: "POST",
          headers: authHeaders(TOKEN_ALICE),
          body: JSON.stringify({ title: `alice-${i}` }),
        })
      );
      requests.push(
        fetch(`${baseUrl}/threads`, {
          method: "POST",
          headers: authHeaders(TOKEN_BOB),
          body: JSON.stringify({ title: `bob-${i}` }),
        })
      );
    }
    const responses = await Promise.all(requests);
    // Every create succeeded despite the synchronous single connection.
    expect(responses.every((r) => r.status === 201)).toBe(true);

    const created = await Promise.all(responses.map((r) => readJson(r)));
    const ids = created.map((c) => c.thread.id);
    // All ids unique (server-issued UUIDs), no corruption/collision.
    expect(new Set(ids).size).toBe(ids.length);

    // Owner scoping holds: Alice sees exactly her N, Bob exactly his N, disjoint.
    const aList = (
      await readJson(
        await fetch(`${baseUrl}/threads`, { headers: authHeaders(TOKEN_ALICE) })
      )
    ).threads as Array<{ id: string; ownerId: string }>;
    const bList = (
      await readJson(
        await fetch(`${baseUrl}/threads`, { headers: authHeaders(TOKEN_BOB) })
      )
    ).threads as Array<{ id: string; ownerId: string }>;
    expect(aList.length).toBe(N);
    expect(bList.length).toBe(N);
    expect(aList.every((t) => t.ownerId === USER_ALICE)).toBe(true);
    expect(bList.every((t) => t.ownerId === USER_BOB)).toBe(true);
    const aIds = new Set(aList.map((t) => t.id));
    expect(bList.some((t) => aIds.has(t.id))).toBe(false);
  });

  it("concurrent chat turns on DIFFERENT threads do not corrupt each other's history", async () => {
    // Two distinct threads/owners. Run turn 1 on BOTH concurrently, then a
    // follow-up turn on each and assert each thread's recorded model input
    // contains ONLY its own prior messages — no cross-thread bleed.
    const tA = (
      await readJson(
        await fetch(`${baseUrl}/threads`, {
          method: "POST",
          headers: authHeaders(TOKEN_ALICE),
          body: JSON.stringify({ title: "A" }),
        })
      )
    ).thread;
    const tB = (
      await readJson(
        await fetch(`${baseUrl}/threads`, {
          method: "POST",
          headers: authHeaders(TOKEN_BOB),
          body: JSON.stringify({ title: "B" }),
        })
      )
    ).thread;

    // Concurrent turn 1 on each thread.
    await Promise.all([
      fetch(`${baseUrl}/threads/${tA.id}/chat`, {
        method: "POST",
        headers: authHeaders(TOKEN_ALICE),
        body: JSON.stringify({ message: "alice-one" }),
      }).then(drain),
      fetch(`${baseUrl}/threads/${tB.id}/chat`, {
        method: "POST",
        headers: authHeaders(TOKEN_BOB),
        body: JSON.stringify({ message: "bob-one" }),
      }).then(drain),
    ]);

    // Follow-up on A: its model input must include alice-one + alice-two, but
    // NEVER bob-one.
    await drain(
      await fetch(`${baseUrl}/threads/${tA.id}/chat`, {
        method: "POST",
        headers: authHeaders(TOKEN_ALICE),
        body: JSON.stringify({ message: "alice-two" }),
      })
    );
    const callA = recordedCalls.at(-1)!;
    expect(callA[0]).toBeInstanceOf(SystemMessage);
    const humansA = humanTextsOf(callA);
    expect(humansA).toContain("alice-one");
    expect(humansA).toContain("alice-two");
    expect(humansA).not.toContain("bob-one");

    // Follow-up on B: symmetric.
    await drain(
      await fetch(`${baseUrl}/threads/${tB.id}/chat`, {
        method: "POST",
        headers: authHeaders(TOKEN_BOB),
        body: JSON.stringify({ message: "bob-two" }),
      })
    );
    const callB = recordedCalls.at(-1)!;
    const humansB = humanTextsOf(callB);
    expect(humansB).toContain("bob-one");
    expect(humansB).toContain("bob-two");
    expect(humansB).not.toContain("alice-one");
  });
});

// ---------------------------------------------------------------------------
// Restart durability THROUGH THE SERVICE (headline Phase 5 exit-criterion)
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — restart durability through the HTTP service", () => {
  let tempDir: string;
  let dbPath: string;
  const savers: SqliteSaver[] = [];
  const servers: Server[] = [];

  async function startInstance(
    saverForInstance: SqliteSaver
  ): Promise<{ server: Server; baseUrl: string }> {
    const store = createThreadStore(saverForInstance);
    const server = createServer({
      store,
      auth: makeAuth(),
      createChat: () =>
        createConversationalChain({ streaming: true }, saverForInstance),
      log: () => {},
    });
    servers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    return { server, baseUrl };
  }

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-5d-restart-"));
    dbPath = path.join(tempDir, "checkpoints.sqlite");
    savers.length = 0;
    servers.length = 0;
  });
  afterEach(async () => {
    for (const s of servers) {
      await new Promise<void>((r) => s.close(() => r())).catch(() => {});
    }
    for (const sv of savers) {
      try {
        sv.db.close();
      } catch {
        /* already closed */
      }
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  it("a thread created + streamed via service A is resumable via a fresh service B on the same DB; cross-user still denied", async () => {
    // --- Instance A: create a thread + run a chat turn via SSE, then tear down.
    const saverA = makeSaver(dbPath);
    savers.push(saverA);
    const a = await startInstance(saverA);

    const created = await fetch(`${a.baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ title: "durable-svc" }),
    });
    const { thread } = await readJson(created);

    const turn1 = await fetch(`${a.baseUrl}/threads/${thread.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "remember: 120 guests" }),
    });
    await turn1.text(); // drain to completion (done)

    // Tear down instance A completely: close the server AND the DB handle, so
    // instance B opens the SAME file fresh (Windows-safe: no shared handle).
    await new Promise<void>((r) => a.server.close(() => r()));
    saverA.db.close();

    // --- Instance B: fresh saver + store + server over the SAME db file.
    const saverB = makeSaver(dbPath);
    savers.push(saverB);
    const b = await startInstance(saverB);

    // 1) The thread still exists for its owner after restart.
    const got = await fetch(`${b.baseUrl}/threads/${thread.id}`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(got.status).toBe(200);
    expect((await readJson(got)).thread.title).toBe("durable-svc");

    // 2) A new chat turn sees PRIOR history (the mocked model receives it).
    const before = recordedCalls.length;
    const turn2 = await fetch(`${b.baseUrl}/threads/${thread.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "what did I say?" }),
    });
    await turn2.text();
    expect(recordedCalls.length).toBe(before + 1);
    const humans = humanTextsOf(recordedCalls.at(-1)!);
    expect(humans).toContain("remember: 120 guests"); // survived the restart
    expect(humans).toContain("what did I say?");

    // 3) Cross-user access is STILL denied after restart (no existence leak).
    const bobGet = await fetch(`${b.baseUrl}/threads/${thread.id}`, {
      headers: authHeaders(TOKEN_BOB),
    });
    expect(bobGet.status).toBe(404);
    const bobList = await fetch(`${b.baseUrl}/threads`, {
      headers: authHeaders(TOKEN_BOB),
    });
    expect((await readJson(bobList)).threads).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Retention hook — pruneThreads(policy): atomic, owner-scopeable, no orphans
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — retention hook (pruneThreads)", () => {
  let tempDir: string;
  let saver: SqliteSaver;

  const OWNER_A = "owner-alice";
  const OWNER_B = "owner-bob";
  const OLD = 1_000; // epoch-ms well below the cutoff
  const CUTOFF = 2_000;

  // Seed real checkpoint state for a thread_id via one mocked conversational turn.
  async function seedCheckpoint(threadId: string): Promise<void> {
    const graph = createConversationalChain({}, saver);
    await graph.invoke(
      { messages: [new HumanMessage("seed")] },
      sessionConfig(threadId)
    );
  }
  // Directly set updated_at to inject a deterministic age (test-only seeding).
  function setUpdatedAt(id: string, ts: number): void {
    saver.db
      .prepare(`UPDATE threads SET updated_at = ? WHERE id = ?`)
      .run(ts, id);
  }
  function checkpointRowCount(threadId: string): number {
    return (
      saver.db
        .prepare("SELECT COUNT(*) AS n FROM checkpoints WHERE thread_id = ?")
        .get(threadId) as { n: number }
    ).n;
  }

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-5d-ret-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
  });
  afterEach(async () => {
    try {
      saver.db.close();
    } catch {
      /* already closed */
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  it("prunes only threads older than the cutoff (across all owners) and removes their checkpoint state atomically", async () => {
    const store = createThreadStore(saver);
    const oldA = store.createThread(OWNER_A, { title: "old-A" });
    const newA = store.createThread(OWNER_A, { title: "new-A" });
    const oldB = store.createThread(OWNER_B, { title: "old-B" });
    await seedCheckpoint(oldA.id);
    await seedCheckpoint(newA.id);
    await seedCheckpoint(oldB.id);
    setUpdatedAt(oldA.id, OLD);
    setUpdatedAt(oldB.id, OLD);
    // newA keeps its natural (large) createdAt/updatedAt, above the cutoff.

    // Global prune (no owner): both old threads go; the new one stays.
    const deleted = store.pruneThreads({ olderThanEpochMs: CUTOFF });
    expect(deleted).toBe(2);

    // Ownership rows: only newA remains.
    expect(store.getThread(OWNER_A, oldA.id)).toBeNull();
    expect(store.getThread(OWNER_B, oldB.id)).toBeNull();
    expect(store.getThread(OWNER_A, newA.id)?.id).toBe(newA.id);

    // Checkpoint state gone for pruned (atomic: row + checkpoints together),
    // intact for the kept thread.
    expect(await saver.getTuple(sessionConfig(oldA.id))).toBeUndefined();
    expect(await saver.getTuple(sessionConfig(oldB.id))).toBeUndefined();
    expect(checkpointRowCount(oldA.id)).toBe(0);
    expect(checkpointRowCount(oldB.id)).toBe(0);
    expect(await saver.getTuple(sessionConfig(newA.id))).toBeDefined();
    expect(checkpointRowCount(newA.id)).toBeGreaterThan(0);
  });

  it("owner-scoped prune deletes ONLY that owner's expired threads (no cross-owner deletion)", async () => {
    const store = createThreadStore(saver);
    const oldA = store.createThread(OWNER_A, { title: "old-A" });
    const oldB = store.createThread(OWNER_B, { title: "old-B" });
    await seedCheckpoint(oldA.id);
    await seedCheckpoint(oldB.id);
    setUpdatedAt(oldA.id, OLD);
    setUpdatedAt(oldB.id, OLD);

    // Owner-scoped prune for A: only A's expired thread is removed.
    const deleted = store.pruneThreads({ olderThanEpochMs: CUTOFF, ownerId: OWNER_A });
    expect(deleted).toBe(1);

    expect(store.getThread(OWNER_A, oldA.id)).toBeNull();
    expect(await saver.getTuple(sessionConfig(oldA.id))).toBeUndefined();

    // B's equally-old thread is UNTOUCHED (row + checkpoint intact).
    expect(store.getThread(OWNER_B, oldB.id)?.id).toBe(oldB.id);
    expect(await saver.getTuple(sessionConfig(oldB.id))).toBeDefined();
    expect(checkpointRowCount(oldB.id)).toBeGreaterThan(0);
  });

  it("prunes nothing when no thread is older than the cutoff (returns 0, no side effects)", async () => {
    const store = createThreadStore(saver);
    const t = store.createThread(OWNER_A, { title: "fresh" });
    await seedCheckpoint(t.id);

    const deleted = store.pruneThreads({ olderThanEpochMs: CUTOFF });
    expect(deleted).toBe(0);
    expect(store.getThread(OWNER_A, t.id)?.id).toBe(t.id);
    expect(await saver.getTuple(sessionConfig(t.id))).toBeDefined();
  });

  it("owner-scoped prune with an owner that has no expired threads returns 0", async () => {
    const store = createThreadStore(saver);
    const oldA = store.createThread(OWNER_A, { title: "old-A" });
    setUpdatedAt(oldA.id, OLD);

    // Scope to B (who has nothing expired): deletes nothing, and A's old thread
    // remains because the scope excluded it.
    const deleted = store.pruneThreads({ olderThanEpochMs: CUTOFF, ownerId: OWNER_B });
    expect(deleted).toBe(0);
    expect(store.getThread(OWNER_A, oldA.id)?.id).toBe(oldA.id);
  });
});

// ---------------------------------------------------------------------------
// Repo cleanliness — no ./data DB artifact is created by this suite
// ---------------------------------------------------------------------------
describe("Phase 5 (5d) — repo cleanliness guard", () => {
  it("the hardening suite does not create a ./data DB in the repo", async () => {
    const testDir = path.dirname(fileURLToPath(import.meta.url));
    const repoRoot = path.resolve(testDir, "..");
    const dataDir = path.join(repoRoot, "data");
    if (existsSync(dataDir)) {
      const entries = await readdir(dataDir);
      expect(entries.filter((e) => e.includes("checkpoints"))).toEqual([]);
    } else {
      expect(existsSync(dataDir)).toBe(false);
    }
  });
});
