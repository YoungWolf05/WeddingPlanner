import { createServer as createHttpServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { HumanMessage, type BaseMessage } from "@langchain/core/messages";
import { createConversationalChain } from "./chain.js";
import { sessionConfig } from "./memory.js";
import { extractChunkText } from "./repl.js";
import { redactError } from "./redaction.js";
import { SseWriter } from "./sse.js";
import type { ThreadStore } from "./threads.js";
import type { TokenAuthenticator } from "./auth.js";

// Phase 5 (5c): the thin AUTHENTICATED HTTP service.
//
// This module is the PURE-ISH request-handling core: routing, auth resolution,
// ownership enforcement, body parsing/limits, and SSE streaming. It constructs
// no network binding itself — `createServer(deps)` returns an http.Server the
// caller binds. The listening entrypoint lives in src/run-server.ts, so tests
// import THIS module, start the server on an ephemeral port (127.0.0.1:0), and
// make real HTTP requests without any live LiteLLM call.
//
// DEPENDENCY INJECTION (offline testability): every side-effecting collaborator
// is injected via ServerDeps — the ThreadStore (owner-scoped persistence), the
// TokenAuthenticator (token->userId), and a `createChat` factory that builds the
// streaming graph. Tests inject a temp-SQLite store and a mocked-model graph
// factory; production wires the real store + createConversationalChain.
//
// SECURITY INVARIANTS (see per-handler comments):
//   - EVERY route except /healthz requires a valid bearer token (else 401).
//   - The AUTHENTICATED userId is the ownerId for every store call. No
//     user-supplied owner/user id (body/query/header/path) is ever trusted.
//   - No existence leak: get/delete/chat on a not-owned OR nonexistent thread
//     returns the SAME 404 (identical status + body).
//   - Every client-facing error and every log line passes through redactError.

// Maximum accepted request body size, in bytes. A body exceeding this is
// rejected with 413 and the connection is not drained further. Chat messages are
// short; this cap prevents a memory-exhaustion vector via an unbounded upload.
export const MAX_BODY_BYTES = 64 * 1024; // 64 KiB

// Maximum accepted chat message length (characters). Bounds a single turn's
// prompt independently of the raw-body byte cap.
export const MAX_MESSAGE_CHARS = 8_000;

// The streaming graph the chat handler drives. Structurally the compiled
// LangGraph returned by createConversationalChain; kept as a minimal shape so
// tests can inject a fake with the same `.stream(...)` contract.
export interface StreamingChat {
  stream(
    input: { messages: BaseMessage[] },
    options: {
      configurable: { thread_id: string };
      streamMode: "messages";
      signal: AbortSignal;
    }
  ): Promise<AsyncIterable<[BaseMessage, unknown]>>;
}

// Injected collaborators. Keeping the graph behind a factory (not a singleton)
// lets each chat turn get a fresh streaming graph bound to the SHARED saver, so
// persistence works and cancellation is per-request.
export interface ServerDeps {
  store: ThreadStore;
  auth: TokenAuthenticator;
  // Build a streaming conversational graph. Production passes a closure over the
  // shared saver: () => createConversationalChain({ streaming: true }, saver).
  // Tests pass a factory returning a mocked-model graph.
  createChat: () => StreamingChat;
  // Optional structured logger for server-side diagnostics. Defaults to a
  // console.error sink. Whatever is passed, the server only ever hands it
  // ALREADY-REDACTED strings.
  log?: (line: string) => void;
}

// A tiny structured result for the JSON (non-SSE) handlers, so routing stays
// declarative and each handler returns data rather than writing the socket.
interface JsonResult {
  status: number;
  // undefined body => empty response (used for 204).
  body?: unknown;
}

const NOT_FOUND: JsonResult = { status: 404, body: { error: "Not found" } };
const UNAUTHORIZED: JsonResult = {
  status: 401,
  // Generic: never reveal whether the token was missing, malformed, or unknown.
  body: { error: "Unauthorized" },
};

// Parse the JSON request body with a hard byte cap. Resolves to:
//   - { ok: true, value } on a parsed JSON body (or undefined for empty body),
//   - { ok: false, status: 413 } when the body exceeds MAX_BODY_BYTES,
//   - { ok: false, status: 400 } on malformed JSON.
// The cap is enforced DURING streaming so an oversized upload is abandoned early
// (we destroy the request) rather than buffered in full.
type BodyResult =
  | { ok: true; value: unknown }
  | { ok: false; status: 400 | 413 };

function readJsonBody(req: IncomingMessage): Promise<BodyResult> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let settled = false;

    const finish = (result: BodyResult): void => {
      if (settled) return;
      settled = true;
      resolve(result);
    };

    let overflowed = false;

    req.on("data", (chunk: Buffer) => {
      if (settled || overflowed) return;
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        // Over the cap: stop BUFFERING (release what we have so a huge upload
        // can't exhaust memory) but keep DRAINING the socket to EOF. We do NOT
        // destroy the connection — destroying would reset the socket before the
        // 413 response could be flushed, so the client would see a socket error
        // instead of a clean 413. Resolving on 'end' (below) lets the handler
        // write a proper 413 after the client finishes (or the stream ends).
        overflowed = true;
        chunks.length = 0;
        return;
      }
      chunks.push(chunk);
    });

    req.on("end", () => {
      if (settled) return;
      if (overflowed) {
        finish({ ok: false, status: 413 });
        return;
      }
      const raw = Buffer.concat(chunks).toString("utf8").trim();
      if (raw === "") {
        finish({ ok: true, value: undefined });
        return;
      }
      try {
        finish({ ok: true, value: JSON.parse(raw) });
      } catch {
        finish({ ok: false, status: 400 });
      }
    });

    req.on("error", () => {
      // Underlying socket error while reading — treat as a bad request.
      finish({ ok: false, status: 400 });
    });
  });
}

// Write a JSON result to the response. Single place that serializes JSON so the
// Content-Type and empty-body (204) handling are consistent.
function writeJson(res: ServerResponse, result: JsonResult): void {
  if (result.body === undefined) {
    res.writeHead(result.status);
    res.end();
    return;
  }
  const payload = JSON.stringify(result.body);
  res.writeHead(result.status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

// Route match for a thread path. Returns the thread id for /threads/:id and
// /threads/:id/chat, plus whether it is the chat sub-route.
interface ThreadRoute {
  threadId: string;
  isChat: boolean;
}

// Parse /threads, /threads/:id, /threads/:id/chat from a pathname. Returns:
//   - "collection" for exactly /threads,
//   - a ThreadRoute for /threads/:id or /threads/:id/chat,
//   - null if the path is not under /threads.
// The :id segment is URL-decoded and validated as non-empty. It is only ever a
// conversation key scoped by the authenticated ownerId — never identity.
function matchThreadRoute(
  pathname: string
): "collection" | ThreadRoute | null {
  if (pathname === "/threads") return "collection";
  const prefix = "/threads/";
  if (!pathname.startsWith(prefix)) return null;
  const rest = pathname.slice(prefix.length);
  if (rest === "") return null;
  const segments = rest.split("/");
  const idSegment = segments[0]!;
  let threadId: string;
  try {
    threadId = decodeURIComponent(idSegment);
  } catch {
    return null; // malformed percent-encoding
  }
  if (threadId === "") return null;
  if (segments.length === 1) return { threadId, isChat: false };
  if (segments.length === 2 && segments[1] === "chat") {
    return { threadId, isChat: true };
  }
  return null; // deeper/unknown sub-paths
}

// The request handler factory. Returned function is what http.Server invokes per
// request; it never throws (all paths resolve to a written response).
export function createRequestHandler(deps: ServerDeps) {
  const log = deps.log ?? ((line: string) => console.error(line));

  return async function handle(
    req: IncomingMessage,
    res: ServerResponse
  ): Promise<void> {
    try {
      await route(req, res, deps, log);
    } catch (err) {
      // Last-resort guard: any unexpected throw becomes a redacted 500 (or, if
      // headers were already sent for an SSE stream, a redacted error event).
      const reason = redactError(err);
      log(`[server] unhandled error: ${reason}`);
      if (!res.headersSent) {
        writeJson(res, { status: 500, body: { error: "Internal error" } });
      } else if (!res.writableEnded) {
        res.end();
      }
    }
  };
}

async function route(
  req: IncomingMessage,
  res: ServerResponse,
  deps: ServerDeps,
  log: (line: string) => void
): Promise<void> {
  const method = req.method ?? "GET";
  // Parse only the pathname; query is intentionally IGNORED for identity — a
  // caller cannot smuggle an owner id via the query string.
  const url = new URL(req.url ?? "/", "http://localhost");
  const pathname = url.pathname;

  // Health check: the ONLY unauthenticated route. No thread access, no secrets.
  if (pathname === "/healthz") {
    if (method !== "GET") {
      writeJson(res, { status: 405, body: { error: "Method not allowed" } });
      return;
    }
    writeJson(res, { status: 200, body: { status: "ok" } });
    return;
  }

  const threadRoute = matchThreadRoute(pathname);
  if (threadRoute === null) {
    // Not a known route at all.
    writeJson(res, NOT_FOUND);
    return;
  }

  // AUTHENTICATION GATE — applies to every /threads* route. A missing,
  // malformed, or unknown token all resolve to the SAME generic 401.
  const authHeader = req.headers["authorization"];
  const ownerId = deps.auth.authenticate(
    Array.isArray(authHeader) ? authHeader[0] : authHeader
  );
  if (ownerId === null) {
    writeJson(res, UNAUTHORIZED);
    return;
  }

  if (threadRoute === "collection") {
    await handleCollection(method, req, res, deps, ownerId);
    return;
  }

  if (threadRoute.isChat) {
    if (method !== "POST") {
      writeJson(res, { status: 405, body: { error: "Method not allowed" } });
      return;
    }
    await handleChat(req, res, deps, ownerId, threadRoute.threadId, log);
    return;
  }

  await handleThreadItem(method, res, deps, ownerId, threadRoute.threadId);
}

// POST /threads (create) and GET /threads (list).
async function handleCollection(
  method: string,
  req: IncomingMessage,
  res: ServerResponse,
  deps: ServerDeps,
  ownerId: string
): Promise<void> {
  if (method === "GET") {
    // Owner-scoped list: the store only ever returns THIS ownerId's threads.
    const threads = deps.store.listThreads(ownerId);
    writeJson(res, { status: 200, body: { threads } });
    return;
  }

  if (method === "POST") {
    const body = await readJsonBody(req);
    if (!body.ok) {
      writeJson(res, {
        status: body.status,
        body: { error: body.status === 413 ? "Payload too large" : "Bad request" },
      });
      return;
    }
    // Optional { title? }. Validate the type but do NOT read any owner/user id
    // from the body — ownership comes solely from the authenticated token.
    const title = extractOptionalTitle(body.value);
    if (title === INVALID) {
      writeJson(res, { status: 400, body: { error: "Invalid title" } });
      return;
    }
    const thread = deps.store.createThread(
      ownerId,
      title === undefined ? undefined : { title }
    );
    writeJson(res, { status: 201, body: { thread } });
    return;
  }

  writeJson(res, { status: 405, body: { error: "Method not allowed" } });
}

// GET /threads/:id (get one) and DELETE /threads/:id (hard delete).
async function handleThreadItem(
  method: string,
  res: ServerResponse,
  deps: ServerDeps,
  ownerId: string,
  threadId: string
): Promise<void> {
  if (method === "GET") {
    // Owner-scoped get. A thread owned by someone else returns null from the
    // store (5b), which we surface as the SAME 404 as a nonexistent id — no
    // existence leak.
    const thread = deps.store.getThread(ownerId, threadId);
    if (thread === null) {
      writeJson(res, NOT_FOUND);
      return;
    }
    writeJson(res, { status: 200, body: { thread } });
    return;
  }

  if (method === "DELETE") {
    // Owner-scoped hard delete (atomic ownership row + checkpoint state, 5b).
    // A non-owner/nonexistent delete returns false -> the SAME 404.
    const deleted = deps.store.deleteThread(ownerId, threadId);
    if (!deleted) {
      writeJson(res, NOT_FOUND);
      return;
    }
    writeJson(res, { status: 204 });
    return;
  }

  writeJson(res, { status: 405, body: { error: "Method not allowed" } });
}

// Sentinel for an invalid title value (present but wrong type).
const INVALID = Symbol("invalid");

// Validate the optional `title` field on a create body. Returns:
//   - undefined when absent/null (no title),
//   - the trimmed string when a valid non-empty string,
//   - INVALID when present but not a string / too long.
function extractOptionalTitle(body: unknown): string | undefined | typeof INVALID {
  if (body === undefined) return undefined;
  if (typeof body !== "object" || body === null) return INVALID;
  const title = (body as { title?: unknown }).title;
  if (title === undefined || title === null) return undefined;
  if (typeof title !== "string") return INVALID;
  const trimmed = title.trim();
  if (trimmed === "") return undefined;
  if (trimmed.length > 200) return INVALID;
  return trimmed;
}

// POST /threads/:id/chat — stream a chat turn over SSE for an OWNED thread.
async function handleChat(
  req: IncomingMessage,
  res: ServerResponse,
  deps: ServerDeps,
  ownerId: string,
  threadId: string,
  log: (line: string) => void
): Promise<void> {
  // 1) Parse + validate the body BEFORE any ownership check or stream start.
  const body = await readJsonBody(req);
  if (!body.ok) {
    writeJson(res, {
      status: body.status,
      body: { error: body.status === 413 ? "Payload too large" : "Bad request" },
    });
    return;
  }
  const message = extractChatMessage(body.value);
  if (message === null) {
    writeJson(res, {
      status: 400,
      body: { error: "Body must include a non-empty 'message' string" },
    });
    return;
  }

  // 2) OWNERSHIP GATE BEFORE STREAMING. If the thread is not owned by this
  // authenticated user (or does not exist), return the SAME 404 as a
  // nonexistent thread — as a plain JSON response, with NO SSE headers and,
  // critically, WITHOUT creating any conversation state. getThread performs no
  // checkpoint write, so a 404 here leaves the store untouched.
  const thread = deps.store.getThread(ownerId, threadId);
  if (thread === null) {
    writeJson(res, NOT_FOUND);
    return;
  }

  // 3) Wire cancellation: a client disconnect aborts the in-flight turn. We
  // listen on BOTH the request and the response 'close' events: during an
  // active SSE stream the response socket is what tears down when the client
  // goes away, while req 'close' covers a disconnect before/around body read.
  // Either one aborts the shared controller (abort is idempotent).
  const controller = new AbortController();
  const onClose = (): void => controller.abort();
  req.on("close", onClose);
  res.on("close", onClose);

  const sse = new SseWriter(res);
  let accumulated = "";
  try {
    // Emit the versioned init marker (also flushes SSE headers).
    sse.init(threadId);

    // Drive the graph with streamMode "messages" (same contract as the REPL),
    // but translate chunks into the versioned SSE token events.
    const graph = deps.createChat();
    const stream = await graph.stream(
      { messages: [new HumanMessage(message)] },
      {
        ...sessionConfig(threadId),
        streamMode: "messages",
        signal: controller.signal,
      }
    );

    for await (const [chunk] of stream) {
      const piece = extractChunkText(chunk.content);
      if (piece.length > 0) {
        accumulated += piece;
        sse.token(piece);
      }
    }

    // The turn succeeded: emit `done` and close the stream FIRST, so the client
    // receives the completed reply promptly and the response is fully finalized.
    sse.done(accumulated);
    sse.end();

    // Touch the thread so listThreads ordering reflects recent activity. This is
    // pure post-turn bookkeeping (updates updatedAt for list ordering) and must
    // be genuinely BEST EFFORT: it runs AFTER the response is already complete
    // and is wrapped in its own try/catch, so a touch failure (e.g. a DB error)
    // can NEVER turn a successful turn into a spurious error event or suppress
    // the `done` event. Any failure is logged through the redacted logger.
    try {
      deps.store.touchThread(ownerId, threadId);
    } catch (touchErr) {
      log(`[server] touchThread failed after successful turn: ${redactError(touchErr)}`);
    }
  } catch (err) {
    // A client-initiated abort is NOT a server error: the socket is gone, so
    // just stop. Any real mid-stream failure becomes a REDACTED error event so
    // provider secrets/PII never reach the wire or the logs.
    if (controller.signal.aborted) {
      log("[server] chat turn aborted by client disconnect");
      sse.end();
    } else {
      const reason = redactError(err);
      log(`[server] chat turn failed: ${reason}`);
      sse.error(reason);
      sse.end();
    }
  } finally {
    req.removeListener("close", onClose);
    res.removeListener("close", onClose);
  }
}

// Validate the chat body's `message`. Returns the trimmed message, or null when
// absent/empty/too long/not a string. Never reads any owner/user id from the
// body — identity comes only from the authenticated token.
function extractChatMessage(body: unknown): string | null {
  if (typeof body !== "object" || body === null) return null;
  const message = (body as { message?: unknown }).message;
  if (typeof message !== "string") return null;
  const trimmed = message.trim();
  if (trimmed === "") return null;
  if (trimmed.length > MAX_MESSAGE_CHARS) return null;
  return trimmed;
}

// Build (but do NOT start) an http.Server wired to the request handler. The
// caller (src/run-server.ts, or a test) calls `.listen(port, host)`. Kept
// separate from binding so the network side effect is never triggered by import.
export function createServer(deps: ServerDeps): Server {
  return createHttpServer(createRequestHandler(deps));
}
