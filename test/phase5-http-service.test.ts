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

// Phase 5 (5c) — authenticated HTTP service + SSE streaming.
//
// FULLY OFFLINE + DETERMINISTIC:
//   - The model boundary is mocked (fake-model), so the REAL conversational
//     graph + REAL SQLite checkpointer run without any network/creds. This is
//     what lets the SSE happy-path + persistence tests exercise genuine history
//     accumulation through the shared saver.
//   - Each test starts the server on an EPHEMERAL port (127.0.0.1:0) and makes
//     real HTTP requests via global fetch.
//   - Every SQLite DB lives in a per-test temp dir under os.tmpdir(), removed in
//     afterEach — the suite writes NO db artifacts into the repo.
//   - The listening entrypoint (src/run-server.ts) is NEVER imported here; we
//     import src/core/server.ts and inject deps directly.
vi.mock("../src/core/model.js", async () => {
  const { makeFakeChatModel } = await import("./helpers/fake-model.js");
  return {
    createChatModel: () =>
      makeFakeChatModel({
        // Distinct, deterministic replies; sleep:0 so streaming is fast and
        // flake-free. FakeListChatModel streams one char per chunk.
        responses: ["hello there", "second reply", "third reply", "fourth reply"],
        sleep: 0,
      }),
  };
});

const { createServer, createHistoryReader, mapMessagesToHistory } = await import(
  "../src/core/server.js"
);
const { AIMessage: AIMessageCtor } = await import("@langchain/core/messages");
const { createConversationalChain } = await import("../src/core/chain.js");
const { createCheckpointer, sessionConfig } = await import(
  "../src/core/memory.js"
);
const { createThreadStore } = await import("../src/core/threads.js");
type ThreadStore = import("../src/core/threads.js").ThreadStore;
const { createTokenAuthenticator } = await import("../src/core/auth.js");
const { SSE_PROTOCOL_VERSION } = await import("../src/core/sse.js");
const { config } = await import("../src/config.js");

// createCheckpointer is typed as BaseCheckpointSaver; narrow to SqliteSaver so
// tests can close saver.db and query it directly (same pattern as 5b tests).
function makeSaver(dbPath: string): SqliteSaver {
  const saver = createCheckpointer(dbPath);
  if (!(saver instanceof SqliteSaver)) throw new Error("expected a SqliteSaver");
  return saver;
}

// Read a JSON response body as an arbitrary record (tests assert on fields).
async function readJson(res: Response): Promise<Record<string, any>> {
  return (await res.json()) as Record<string, any>;
}

// --- Auth fixtures (injected directly; NEVER read from process env) ----------
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

// Parse an SSE response body into a list of { event, data } frames.
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

describe("Phase 5 (5c) — authenticated HTTP service", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  // Start a server with the real graph (mocked model) over a fresh temp saver.
  async function startServer(): Promise<void> {
    const store = createThreadStore(saver);
    const auth = makeAuth();
    server = createServer({
      store,
      auth,
      createChat: () => createConversationalChain({ streaming: true }, saver),
      // Silence server-side logging in tests.
      log: () => {},
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;
  }

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-http-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
  });

  afterEach(async () => {
    if (server) {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    try {
      saver.db.close();
    } catch {
      // already closed
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  function authHeaders(token: string): Record<string, string> {
    return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
  }

  // --- AUTH -----------------------------------------------------------------
  describe("authentication", () => {
    it("rejects a request with NO token (401, generic)", async () => {
      await startServer();
      const res = await fetch(`${baseUrl}/threads`, { method: "GET" });
      expect(res.status).toBe(401);
      const body = await res.json();
      expect(body).toEqual({ error: "Unauthorized" });
    });

    it("rejects a MALFORMED Authorization header (401)", async () => {
      await startServer();
      const res = await fetch(`${baseUrl}/threads`, {
        method: "GET",
        headers: { Authorization: "NotBearer whatever" },
      });
      expect(res.status).toBe(401);
      expect(await res.json()).toEqual({ error: "Unauthorized" });
    });

    it("rejects an UNKNOWN token — indistinguishable from malformed (same 401 body)", async () => {
      await startServer();
      const malformed = await fetch(`${baseUrl}/threads`, {
        method: "GET",
        headers: { Authorization: "Bearer" },
      });
      const unknown = await fetch(`${baseUrl}/threads`, {
        method: "GET",
        headers: { Authorization: "Bearer totally-unknown-token" },
      });
      expect(malformed.status).toBe(401);
      expect(unknown.status).toBe(401);
      // Same status AND same body: no unknown-vs-malformed distinction leaks.
      expect(await malformed.json()).toEqual(await unknown.json());
    });

    it("ALLOWS a valid token", async () => {
      await startServer();
      const res = await fetch(`${baseUrl}/threads`, {
        method: "GET",
        headers: authHeaders(TOKEN_ALICE),
      });
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ threads: [] });
    });

    it("/healthz needs NO auth", async () => {
      await startServer();
      const res = await fetch(`${baseUrl}/healthz`);
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ status: "ok" });
    });
  });

  // --- CRUD happy path ------------------------------------------------------
  it("CRUD happy path: create -> list -> get -> delete -> 404", async () => {
    await startServer();

    // create
    const created = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ title: "Venue ideas" }),
    });
    expect(created.status).toBe(201);
    const { thread } = await readJson(created);
    expect(thread.id).toMatch(/[0-9a-f-]{36}/);
    expect(thread.ownerId).toBe(USER_ALICE);
    expect(thread.title).toBe("Venue ideas");

    // list shows it
    const listed = await fetch(`${baseUrl}/threads`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    const { threads } = await readJson(listed);
    expect(threads.map((t: { id: string }) => t.id)).toContain(thread.id);

    // get returns it
    const got = await fetch(`${baseUrl}/threads/${thread.id}`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(got.status).toBe(200);
    expect((await readJson(got)).thread.id).toBe(thread.id);

    // delete
    const del = await fetch(`${baseUrl}/threads/${thread.id}`, {
      method: "DELETE",
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(del.status).toBe(204);

    // subsequent get => 404
    const gone = await fetch(`${baseUrl}/threads/${thread.id}`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(gone.status).toBe(404);
  });

  // --- Cross-user isolation & no existence leak -----------------------------
  it("cross-user: B's GET/DELETE/chat on A's thread == the SAME 404 as a random id", async () => {
    await startServer();

    // Alice creates a thread.
    const created = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ title: "private" }),
    });
    const { thread } = await readJson(created);

    const randomId = "11111111-1111-4111-8111-111111111111";

    // Bob GET on A's thread and on a random id: identical 404 status + body.
    const bGetOwned = await fetch(`${baseUrl}/threads/${thread.id}`, {
      headers: authHeaders(TOKEN_BOB),
    });
    const bGetRandom = await fetch(`${baseUrl}/threads/${randomId}`, {
      headers: authHeaders(TOKEN_BOB),
    });
    expect(bGetOwned.status).toBe(404);
    expect(bGetRandom.status).toBe(404);
    expect(await bGetOwned.json()).toEqual(await bGetRandom.json());

    // Bob DELETE on A's thread and on a random id: identical 404.
    const bDelOwned = await fetch(`${baseUrl}/threads/${thread.id}`, {
      method: "DELETE",
      headers: authHeaders(TOKEN_BOB),
    });
    const bDelRandom = await fetch(`${baseUrl}/threads/${randomId}`, {
      method: "DELETE",
      headers: authHeaders(TOKEN_BOB),
    });
    expect(bDelOwned.status).toBe(404);
    expect(bDelRandom.status).toBe(404);
    expect(await bDelOwned.json()).toEqual(await bDelRandom.json());

    // Bob chat on A's thread and on a random id: identical 404 (no SSE).
    const bChatOwned = await fetch(`${baseUrl}/threads/${thread.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_BOB),
      body: JSON.stringify({ message: "hi" }),
    });
    const bChatRandom = await fetch(`${baseUrl}/threads/${randomId}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_BOB),
      body: JSON.stringify({ message: "hi" }),
    });
    expect(bChatOwned.status).toBe(404);
    expect(bChatRandom.status).toBe(404);
    expect(bChatOwned.headers.get("content-type")).not.toContain(
      "text/event-stream"
    );
    expect(await bChatOwned.json()).toEqual(await bChatRandom.json());

    // Bob's list does NOT include A's thread.
    const bList = await fetch(`${baseUrl}/threads`, {
      headers: authHeaders(TOKEN_BOB),
    });
    const { threads } = await readJson(bList);
    expect(threads.some((t: { id: string }) => t.id === thread.id)).toBe(false);

    // Bob's delete did NOT affect Alice's thread — Alice can still get it.
    const aStillGet = await fetch(`${baseUrl}/threads/${thread.id}`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(aStillGet.status).toBe(200);
  });

  it("ownership comes from AUTH, not client input: spoofed owner/user id is ignored", async () => {
    await startServer();

    // Alice creates a thread.
    const created = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ title: "alice-only" }),
    });
    const { thread } = await readJson(created);

    // Bob tries to spoof Alice's identity via body/query/header. None of these
    // must grant access to Alice's thread.
    const spoofBody = await fetch(`${baseUrl}/threads/${thread.id}`, {
      method: "GET",
      headers: {
        ...authHeaders(TOKEN_BOB),
        "X-Owner-Id": USER_ALICE,
        "X-User-Id": USER_ALICE,
      },
    });
    expect(spoofBody.status).toBe(404);

    const spoofQuery = await fetch(
      `${baseUrl}/threads/${thread.id}?ownerId=${USER_ALICE}&userId=${USER_ALICE}`,
      { headers: authHeaders(TOKEN_BOB) }
    );
    expect(spoofQuery.status).toBe(404);

    // Even a create with an ownerId in the body is owned by the AUTHENTICATED
    // user (Bob), not the spoofed id.
    const createdByBob = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_BOB),
      body: JSON.stringify({ title: "x", ownerId: USER_ALICE, userId: USER_ALICE }),
    });
    const bobThread = (await readJson(createdByBob)).thread;
    expect(bobThread.ownerId).toBe(USER_BOB);
  });

  // --- SSE chat happy path + persistence ------------------------------------
  it("SSE chat: init(version) -> token(s) -> done; tokens == mocked reply; text/event-stream", async () => {
    await startServer();

    const created = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({}),
    });
    const { thread } = await readJson(created);

    const res = await fetch(`${baseUrl}/threads/${thread.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "plan my wedding" }),
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const frames = parseSse(await res.text());

    // First frame is the versioned init marker.
    expect(frames[0]!.event).toBe("init");
    expect((frames[0]!.data as { version: number }).version).toBe(
      SSE_PROTOCOL_VERSION
    );
    expect((frames[0]!.data as { threadId: string }).threadId).toBe(thread.id);

    // Last frame is done; middle frames are tokens.
    const last = frames.at(-1)!;
    expect(last.event).toBe("done");

    const tokenFrames = frames.filter((f) => f.event === "token");
    expect(tokenFrames.length).toBeGreaterThan(0);
    const accumulated = tokenFrames
      .map((f) => (f.data as { text: string }).text)
      .join("");
    // First mocked reply.
    expect(accumulated).toBe("hello there");
    // done carries the full accumulated reply too.
    expect((last.data as { text: string }).text).toBe("hello there");
  });

  it("SSE chat persistence: a second turn sees prior history via the shared saver", async () => {
    await startServer();

    const created = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({}),
    });
    const { thread } = await readJson(created);

    // Turn 1.
    const r1 = await fetch(`${baseUrl}/threads/${thread.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "We have 120 guests." }),
    });
    await r1.text(); // drain

    // Turn 2 on the SAME thread.
    const r2 = await fetch(`${baseUrl}/threads/${thread.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "What's next?" }),
    });
    await r2.text(); // drain

    // The mocked model recorded the messages it received per call. Turn 2 must
    // include turn 1's history: [system, human(120), ai(reply1), human(next)].
    expect(recordedCalls.length).toBe(2);
    const second = recordedCalls[1]!;
    expect(second[0]).toBeInstanceOf(SystemMessage);
    const humanTexts = second
      .filter((m) => m instanceof HumanMessage)
      .map((m) => m.content);
    expect(humanTexts).toContain("We have 120 guests.");
    expect(humanTexts).toContain("What's next?");
  });

  it("chat on a nonexistent/not-owned thread => 404 BEFORE any stream or checkpoint write", async () => {
    await startServer();

    const randomId = "22222222-2222-4222-8222-222222222222";
    const res = await fetch(`${baseUrl}/threads/${randomId}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "hello" }),
    });
    expect(res.status).toBe(404);
    // Plain JSON, NOT an event stream.
    expect(res.headers.get("content-type")).not.toContain("text/event-stream");
    expect(await res.json()).toEqual({ error: "Not found" });

    // No checkpoint state was written for that thread_id.
    const tuple = await saver.getTuple(sessionConfig(randomId));
    expect(tuple).toBeUndefined();

    // The model was never invoked.
    expect(recordedCalls.length).toBe(0);
  });

  // --- Validation -----------------------------------------------------------
  describe("validation", () => {
    async function createAliceThread(): Promise<string> {
      const created = await fetch(`${baseUrl}/threads`, {
        method: "POST",
        headers: authHeaders(TOKEN_ALICE),
        body: JSON.stringify({}),
      });
      return (await readJson(created)).thread.id;
    }

    it("chat with MISSING message => 400", async () => {
      await startServer();
      const id = await createAliceThread();
      const res = await fetch(`${baseUrl}/threads/${id}/chat`, {
        method: "POST",
        headers: authHeaders(TOKEN_ALICE),
        body: JSON.stringify({}),
      });
      expect(res.status).toBe(400);
    });

    it("chat with EMPTY message => 400", async () => {
      await startServer();
      const id = await createAliceThread();
      const res = await fetch(`${baseUrl}/threads/${id}/chat`, {
        method: "POST",
        headers: authHeaders(TOKEN_ALICE),
        body: JSON.stringify({ message: "   " }),
      });
      expect(res.status).toBe(400);
    });

    it("MALFORMED JSON => 400", async () => {
      await startServer();
      const res = await fetch(`${baseUrl}/threads`, {
        method: "POST",
        headers: authHeaders(TOKEN_ALICE),
        body: "{not valid json",
      });
      expect(res.status).toBe(400);
    });

    it("OVERSIZED body => 413", async () => {
      await startServer();
      const id = await createAliceThread();
      const huge = "x".repeat(70 * 1024); // > 64 KiB cap
      const res = await fetch(`${baseUrl}/threads/${id}/chat`, {
        method: "POST",
        headers: authHeaders(TOKEN_ALICE),
        body: JSON.stringify({ message: huge }),
      });
      expect(res.status).toBe(413);
    });

    it("UNKNOWN route => 404", async () => {
      await startServer();
      const res = await fetch(`${baseUrl}/nope`, {
        headers: authHeaders(TOKEN_ALICE),
      });
      expect(res.status).toBe(404);
    });

    it("WRONG method on /threads/:id => 405", async () => {
      await startServer();
      const id = await createAliceThread();
      const res = await fetch(`${baseUrl}/threads/${id}`, {
        method: "PUT",
        headers: authHeaders(TOKEN_ALICE),
        body: JSON.stringify({}),
      });
      expect(res.status).toBe(405);
    });
  });

  // --- repo cleanliness -----------------------------------------------------
  it("does not create a ./data DB in the repo", async () => {
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

// --- Cancellation + redaction: injected controllable fake graph -------------
//
// These two behaviors are about the SERVER's stream orchestration, so we inject
// a hand-rolled StreamingChat with FULL control over token cadence and errors —
// no real timers, no flakiness. The store is still the real temp-SQLite store so
// ownership is genuinely enforced.
describe("Phase 5 (5c) — cancellation & redaction", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-http-cx-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
  });

  afterEach(async () => {
    if (server) {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    try {
      saver.db.close();
    } catch {
      // already closed
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  it("client disconnect mid-stream aborts the in-flight run (no crash; signal fires)", async () => {
    const store = createThreadStore(saver);
    const auth = makeAuth();

    // A gate the test resolves to release tokens one at a time; the fake aborts
    // cleanly when the injected signal fires.
    let abortObserved = false;
    const firstTokenSent = new Promise<void>((resolveFirst) => {
      server = createServer({
        store,
        auth,
        log: () => {},
        createChat: () => ({
          async stream(_input, options) {
            const signal = options.signal;
            async function* gen(): AsyncGenerator<[BaseMessage, unknown]> {
              // Emit one token, then wait until the client aborts.
              yield [new AIMessageChunk({ content: "partial" }), {}];
              resolveFirst();
              await new Promise<void>((resolve) => {
                if (signal.aborted) {
                  abortObserved = true;
                  resolve();
                  return;
                }
                signal.addEventListener("abort", () => {
                  abortObserved = true;
                  resolve();
                });
              });
              // If we ever get here, emit nothing further.
            }
            return gen();
          },
        }),
      });
    });

    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;

    const t = store.createThread(USER_ALICE, { title: "cancel" });

    // Start the chat with an AbortController we control; abort after the first
    // token has been streamed. Aborting the fetch closes the connection, which
    // the server observes as req 'close' and turns into an AbortController.abort
    // on the in-flight run.
    const clientAbort = new AbortController();
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN_ALICE}`, "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hi" }),
      signal: clientAbort.signal,
    });
    const reader = res.body!.getReader();
    // Read the first chunk (init + first token) so we know the stream started.
    await reader.read();
    await firstTokenSent;

    // Disconnect the client mid-stream.
    clientAbort.abort();
    // Reading after abort rejects; swallow it — the connection is intentionally
    // torn down. The load-bearing assertion is that the SERVER observed abort.
    await reader.read().catch(() => {});
    reader.cancel().catch(() => {});

    // Poll (bounded) until the server's 'close' handler fires the run's
    // AbortController — deterministic, no fixed sleep flakiness.
    for (let i = 0; i < 100 && !abortObserved; i++) {
      await new Promise((r) => setTimeout(r, 10));
    }
    expect(abortObserved).toBe(true);

    // Server did not crash: it can still serve requests.
    const health = await fetch(`${baseUrl}/healthz`);
    expect(health.status).toBe(200);
  });

  it("mid-stream error is REDACTED in the SSE error event (no secret/PII)", async () => {
    const store = createThreadStore(saver);
    const auth = makeAuth();

    // The fake throws a message packed with the fake apiKey/baseURL + an email.
    const leaky =
      `boom apiKey=${config.apiKey} url=${config.baseURL} contact planner@example.com`;

    const logs: string[] = [];
    server = createServer({
      store,
      auth,
      log: (line) => logs.push(line),
      createChat: () => ({
        async stream(): Promise<AsyncIterable<[BaseMessage, unknown]>> {
          async function* gen(): AsyncGenerator<[BaseMessage, unknown]> {
            yield [new AIMessageChunk({ content: "start" }), {}];
            throw new Error(leaky);
          }
          return gen();
        },
      }),
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;

    const t = store.createThread(USER_ALICE, { title: "leaky" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN_ALICE}`, "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hi" }),
    });
    const body = await res.text();
    const frames = parseSse(body);
    const errorFrame = frames.find((f) => f.event === "error");
    expect(errorFrame).toBeDefined();
    const message = (errorFrame!.data as { message: string }).message;

    // The redacted message must NOT contain the secret or PII substrings.
    expect(message).not.toContain(config.apiKey);
    expect(message).not.toContain(config.baseURL);
    expect(message).not.toContain("planner@example.com");
    // ...and it SHOULD contain the redaction placeholders (proof it ran).
    expect(message).toContain("[redacted-key]");

    // The server log line is redacted too.
    const joinedLogs = logs.join("\n");
    expect(joinedLogs).not.toContain(config.apiKey);
    expect(joinedLogs).not.toContain(config.baseURL);
    expect(joinedLogs).not.toContain("planner@example.com");
  });

  it("best-effort touch: a touchThread failure on the success path does NOT emit an error (tokens + done still delivered)", async () => {
    // R2 fix: touchThread runs AFTER sse.done()/sse.end() and is wrapped so a
    // throw can never turn a fully-successful stream into a spurious error event
    // or suppress `done`. We decorate the REAL store so ownership is genuinely
    // enforced (getThread/createThread behave normally) but touchThread throws
    // with a leaky message — proving BOTH best-effort handling AND that the
    // swallowed failure is redacted in the log.
    const auth = makeAuth();
    const realStore = createThreadStore(saver);
    const leakyTouchMessage =
      `touch DB error apiKey=${config.apiKey} url=${config.baseURL} planner@example.com`;
    const store: ThreadStore = {
      createThread: (ownerId, opts) => realStore.createThread(ownerId, opts),
      listThreads: (ownerId) => realStore.listThreads(ownerId),
      getThread: (ownerId, threadId) => realStore.getThread(ownerId, threadId),
      deleteThread: (ownerId, threadId) =>
        realStore.deleteThread(ownerId, threadId),
      pruneThreads: (policy) => realStore.pruneThreads(policy),
      touchThread: () => {
        throw new Error(leakyTouchMessage);
      },
    };

    const logs: string[] = [];
    server = createServer({
      store,
      auth,
      log: (line) => logs.push(line),
      // Real graph + mocked model -> deterministic token stream + a real done.
      createChat: () => createConversationalChain({ streaming: true }, saver),
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;

    const t = store.createThread(USER_ALICE, { title: "touch-throws" });
    const res = await fetch(`${baseUrl}/threads/${t.id}/chat`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${TOKEN_ALICE}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: "hello" }),
    });

    // Still a successful SSE stream.
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");

    const frames = parseSse(await res.text());
    // init -> token(s) -> done, and NO error frame despite the touch throwing.
    expect(frames[0]!.event).toBe("init");
    const tokenFrames = frames.filter((f) => f.event === "token");
    expect(tokenFrames.length).toBeGreaterThan(0);
    expect(frames.some((f) => f.event === "error")).toBe(false);
    const last = frames.at(-1)!;
    expect(last.event).toBe("done");

    // The accumulated tokens equal the mocked reply, and done carries it too.
    const accumulated = tokenFrames
      .map((f) => (f.data as { text: string }).text)
      .join("");
    expect(accumulated).toBe("hello there");
    expect((last.data as { text: string }).text).toBe("hello there");

    // The swallowed touch failure was logged AND redacted (no secret/PII).
    const joinedLogs = logs.join("\n");
    expect(joinedLogs).toContain("touchThread failed");
    expect(joinedLogs).not.toContain(config.apiKey);
    expect(joinedLogs).not.toContain(config.baseURL);
    expect(joinedLogs).not.toContain("planner@example.com");
  });
});

// --- Conversation-history replay: GET /threads/:id/messages -----------------
//
// The history seam is INJECTED as a deterministic FAKE (no real graph/checkpointer
// I/O) so these tests are fully offline. The store is the REAL temp-SQLite store,
// so ownership is genuinely enforced. We assert:
//   - happy path: owner + owned thread -> 200 { messages } in order, roles/text
//     correct, and REDACTION applied to a secret-shaped message on the wire;
//   - empty thread (owned, no history) -> 200 { messages: [] } (NOT 404);
//   - auth: unauthenticated -> generic 401;
//   - ownership: not-owned/nonexistent -> the SAME 404, and the history seam is
//     NOT invoked (ownership gate runs first); ownerId from token only (spoofed
//     body/query owner ignored);
//   - method guard: non-GET -> 405;
//   - routing: /messages does not collide with /:id or /:id/chat; deeper paths 404;
//   - a checkpointer read that throws -> a REDACTED 500 (no leak).
type HistoryMessage = import("../src/core/server.js").HistoryMessage;
type ServerDeps = import("../src/core/server.js").ServerDeps;

describe("Phase (history) — GET /threads/:id/messages", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;
  let store: ThreadStore;
  // Records the thread ids the injected history seam was asked for, so we can
  // assert the ownership gate runs BEFORE any history read.
  let historyCalls: string[];

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-hist-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
    store = createThreadStore(saver);
    historyCalls = [];
  });

  afterEach(async () => {
    if (server) {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    try {
      saver.db.close();
    } catch {
      // already closed
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  function authHeaders(token: string): Record<string, string> {
    return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
  }

  // Start a server with an injected history seam. The `history` map is keyed by
  // thread id; a missing id yields []. `throwFor` forces the seam to throw for a
  // given id (the checkpointer-read-failure path). Every lookup is recorded.
  async function startServer(opts: {
    history?: Record<string, HistoryMessage[]>;
    throwFor?: string;
    throwMessage?: string;
    logs?: string[];
  }): Promise<void> {
    const auth = makeAuth();
    const readHistory: NonNullable<ServerDeps["readHistory"]> = async (
      threadId: string
    ) => {
      historyCalls.push(threadId);
      if (opts.throwFor !== undefined && threadId === opts.throwFor) {
        throw new Error(opts.throwMessage ?? "checkpointer read failed");
      }
      return opts.history?.[threadId] ?? [];
    };
    server = createServer({
      store,
      auth,
      // The plain-chat graph must never run on this read-only route.
      createChat: () => {
        throw new Error("createChat must not run for the messages route");
      },
      readHistory,
      log: opts.logs ? (line) => opts.logs!.push(line) : () => {},
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;
  }

  it("happy path: owner + owned thread -> 200 { messages } in order, roles/text correct, redacted", async () => {
    const t = store.createThread(USER_ALICE, { title: "with-history" });
    // A user + assistant pair, plus an assistant message carrying a secret-shaped
    // substring to prove redaction runs on the wire. The wire shape is produced
    // by the SAME mapMessagesToHistory the production reader uses (System/tool
    // skipped, content flattened, text REDACTED), so this asserts the real
    // server-side redaction path — not a test-local scrub.
    await startServer({
      history: {
        [t.id]: mapMessagesToHistory([
          new HumanMessage("We have 120 guests."),
          new AIMessageCtor("Great, here are venue ideas."),
          new AIMessageCtor(`contact planner@example.com key=${config.apiKey}`),
        ]),
      },
    });

    const res = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("application/json");
    const body = await readJson(res);
    const messages = body["messages"] as HistoryMessage[];
    expect(messages.length).toBe(3);
    // Order + roles + text preserved.
    expect(messages[0]).toEqual({ role: "user", text: "We have 120 guests." });
    expect(messages[1]).toEqual({
      role: "assistant",
      text: "Great, here are venue ideas.",
    });
    // Redaction applied on the wire: no secret/PII leaks; placeholders present.
    const last = messages[2]!;
    expect(last.role).toBe("assistant");
    expect(last.text).not.toContain(config.apiKey);
    expect(last.text).not.toContain("planner@example.com");
    expect(last.text).toContain("[redacted-key]");
    expect(last.text).toContain("[redacted-email]");
  });

  it("empty thread (owned, no history) -> 200 { messages: [] }, NOT 404", async () => {
    const t = store.createThread(USER_ALICE, { title: "empty" });
    await startServer({ history: {} });
    const res = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(res.status).toBe(200);
    expect(await readJson(res)).toEqual({ messages: [] });
    // The seam WAS invoked for an owned thread.
    expect(historyCalls).toEqual([t.id]);
  });

  it("unauthenticated -> generic 401 (seam NOT invoked)", async () => {
    const t = store.createThread(USER_ALICE, { title: "x" });
    await startServer({ history: { [t.id]: [{ role: "user", text: "hi" }] } });
    const res = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
      method: "GET",
    });
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual({ error: "Unauthorized" });
    expect(historyCalls).toEqual([]);
  });

  it("not-owned/nonexistent -> identical 404, and the history seam is NOT invoked (ownership gate first)", async () => {
    const t = store.createThread(USER_ALICE, { title: "alice-only" });
    const randomId = "33333333-3333-4333-8333-333333333333";
    await startServer({
      history: { [t.id]: [{ role: "user", text: "secret history" }] },
    });

    // Bob on Alice's thread and on a random id: identical 404 status + body.
    const bobOnAlice = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
      headers: authHeaders(TOKEN_BOB),
    });
    const bobOnRandom = await fetch(`${baseUrl}/threads/${randomId}/messages`, {
      headers: authHeaders(TOKEN_BOB),
    });
    expect(bobOnAlice.status).toBe(404);
    expect(bobOnRandom.status).toBe(404);
    expect(await bobOnAlice.json()).toEqual(await bobOnRandom.json());
    // CRITICAL: the ownership gate ran first — the seam was never asked for
    // either id (no history read for a thread the caller cannot access).
    expect(historyCalls).toEqual([]);
  });

  it("ownerId from token only: a spoofed body/query/header owner is ignored", async () => {
    const t = store.createThread(USER_ALICE, { title: "alice-only" });
    await startServer({
      history: { [t.id]: [{ role: "user", text: "history" }] },
    });

    // Bob tries to spoof Alice via query + headers. Still a 404 (Bob doesn't own
    // it) and the seam is never invoked.
    const spoofQuery = await fetch(
      `${baseUrl}/threads/${t.id}/messages?ownerId=${USER_ALICE}&userId=${USER_ALICE}`,
      {
        headers: {
          ...authHeaders(TOKEN_BOB),
          "X-Owner-Id": USER_ALICE,
          "X-User-Id": USER_ALICE,
        },
      }
    );
    expect(spoofQuery.status).toBe(404);
    expect(historyCalls).toEqual([]);
  });

  it("method guard: non-GET on /messages -> 405", async () => {
    const t = store.createThread(USER_ALICE, { title: "x" });
    await startServer({ history: {} });
    for (const method of ["POST", "PUT", "DELETE"]) {
      const res = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
        method,
        headers: authHeaders(TOKEN_ALICE),
        ...(method === "GET" ? {} : { body: JSON.stringify({}) }),
      });
      expect(res.status).toBe(405);
    }
    // A 405 short-circuits before any history read.
    expect(historyCalls).toEqual([]);
  });

  it("routing: /messages does not collide with /:id or /:id/chat; deeper sub-paths -> 404", async () => {
    const t = store.createThread(USER_ALICE, { title: "route" });
    await startServer({
      history: { [t.id]: [{ role: "user", text: "hi" }] },
    });

    // GET /:id still returns the thread record (NOT the messages list).
    const item = await fetch(`${baseUrl}/threads/${t.id}`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(item.status).toBe(200);
    expect((await readJson(item))["thread"]).toBeDefined();

    // GET /:id/messages returns the messages list (NOT the thread record).
    const msgs = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(msgs.status).toBe(200);
    expect((await readJson(msgs))["messages"]).toEqual([
      { role: "user", text: "hi" },
    ]);

    // A deeper/unknown sub-path is not a known route -> 404.
    const deeper = await fetch(`${baseUrl}/threads/${t.id}/messages/extra`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(deeper.status).toBe(404);
  });

  it("checkpointer read throwing -> a REDACTED 500 (no secret/PII leak)", async () => {
    const t = store.createThread(USER_ALICE, { title: "boom" });
    const logs: string[] = [];
    const leaky = `read boom apiKey=${config.apiKey} url=${config.baseURL} planner@example.com`;
    await startServer({ throwFor: t.id, throwMessage: leaky, logs });

    const res = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(res.status).toBe(500);
    const body = await readJson(res);
    // Generic error body — no provider/DB internals on the wire.
    expect(body).toEqual({ error: "Internal error" });

    // The log line is redacted (no secret/PII).
    const joined = logs.join("\n");
    expect(joined).toContain("history read failed");
    expect(joined).not.toContain(config.apiKey);
    expect(joined).not.toContain(config.baseURL);
    expect(joined).not.toContain("planner@example.com");
  });

  it("no readHistory seam wired -> 200 { messages: [] } for an owned thread (never an error)", async () => {
    const t = store.createThread(USER_ALICE, { title: "unwired" });
    const auth = makeAuth();
    server = createServer({
      store,
      auth,
      createChat: () => {
        throw new Error("createChat must not run for the messages route");
      },
      // readHistory intentionally omitted.
      log: () => {},
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;

    const res = await fetch(`${baseUrl}/threads/${t.id}/messages`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(res.status).toBe(200);
    expect(await readJson(res)).toEqual({ messages: [] });
  });
});

// --- End-to-end history via the REAL graph + saver + createHistoryReader -----
//
// Beyond the injected-fake seam tests above, this proves the PRODUCTION reader
// (createHistoryReader over the shared saver) reads back messages that a REAL
// chat turn (mocked model, real conversational graph + real SQLite checkpointer)
// actually persisted — the exact plumbing `npm run serve` uses.
describe("Phase (history) — createHistoryReader over the real graph/saver", () => {
  let tempDir: string;
  let saver: SqliteSaver;
  let server: Server;
  let baseUrl: string;

  async function startServer(): Promise<void> {
    const store = createThreadStore(saver);
    const auth = makeAuth();
    server = createServer({
      store,
      auth,
      createChat: () => createConversationalChain({ streaming: true }, saver),
      readHistory: createHistoryReader(saver),
      log: () => {},
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const addr = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${addr.port}`;
  }

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-hist-real-"));
    saver = makeSaver(path.join(tempDir, "checkpoints.sqlite"));
  });

  afterEach(async () => {
    if (server) {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    try {
      saver.db.close();
    } catch {
      // already closed
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  function authHeaders(token: string): Record<string, string> {
    return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
  }

  it("a completed chat turn is replayed as [user, assistant] via GET /messages", async () => {
    await startServer();
    const created = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({}),
    });
    const { thread } = await readJson(created);

    // Run one chat turn (mocked model reply "hello there").
    const chat = await fetch(`${baseUrl}/threads/${thread.id}/chat`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({ message: "plan my wedding" }),
    });
    await chat.text(); // drain the SSE stream to completion

    // Now replay history. The SystemMessage persona is SKIPPED; we see exactly
    // the user message and the assistant reply, in order.
    const res = await fetch(`${baseUrl}/threads/${thread.id}/messages`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(res.status).toBe(200);
    const messages = (await readJson(res))["messages"] as HistoryMessage[];
    expect(messages).toEqual([
      { role: "user", text: "plan my wedding" },
      { role: "assistant", text: "hello there" },
    ]);
  });

  it("a brand-new owned thread with no turns replays as { messages: [] }", async () => {
    await startServer();
    const created = await fetch(`${baseUrl}/threads`, {
      method: "POST",
      headers: authHeaders(TOKEN_ALICE),
      body: JSON.stringify({}),
    });
    const { thread } = await readJson(created);

    const res = await fetch(`${baseUrl}/threads/${thread.id}/messages`, {
      headers: authHeaders(TOKEN_ALICE),
    });
    expect(res.status).toBe(200);
    expect(await readJson(res)).toEqual({ messages: [] });
  });
});
