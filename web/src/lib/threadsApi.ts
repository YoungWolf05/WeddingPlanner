// Phase 9 (9c): the thread REST client (create / list / get).
//
// MIRRORS src/core/server.ts routing EXACTLY (canonical source):
//   - POST   /threads            body { title? }        -> 201 { thread }
//   - GET    /threads                                   -> 200 { threads }
//   - GET    /threads/:id                               -> 200 { thread } | 404
//   - GET    /threads/:id/messages                      -> 200 { messages } | 404
//   - DELETE /threads/:id                               -> 204            | 404
//   - POST   /threads/:id/chat   (SSE — see sseClient.ts, not here)
//
// AUTH. Every call sends `Authorization: Bearer <token>` — the ONLY credential.
// The backend derives ownerId from the token; the client NEVER sends an
// owner/user id in any body/query/header/path field. No provider credential is
// ever referenced. `thread_id` is a server-issued conversation key, not identity.

// The Thread record shape as returned on the wire — mirrors the `Thread`
// interface in src/core/threads.ts (rowToThread projection).
export interface Thread {
  id: string;
  ownerId: string;
  title: string | null;
  createdAt: number;
  updatedAt: number;
}

// A typed error for a failed thread API call, carrying the HTTP status so the UI
// can react (e.g. 401 -> prompt for a token). The message is user-safe; response
// bodies are already redacted server-side and are NOT echoed verbatim.
export class ThreadApiError extends Error {
  constructor(
    readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "ThreadApiError";
  }
}

export interface ThreadsApiConfig {
  // Backend base path. Empty string = same-origin (dev proxy / prod).
  baseUrl?: string;
  token: string;
}

function authHeaders(token: string): HeadersInit {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    Authorization: `Bearer ${token}`,
  };
}

// Map a non-ok status to a concise, user-safe message.
function statusMessage(status: number): string {
  switch (status) {
    case 401:
      return "Unauthorized. Check your access token.";
    case 404:
      return "Conversation not found.";
    case 400:
      return "Invalid request.";
    case 413:
      return "Request too large.";
    default:
      return "The request failed. Please try again.";
  }
}

async function requestJson<T>(
  url: string,
  init: RequestInit,
  extract: (body: unknown) => T
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new ThreadApiError(0, "Network error. Is the service running?");
  }
  if (!response.ok) {
    throw new ThreadApiError(response.status, statusMessage(response.status));
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ThreadApiError(response.status, "Malformed server response.");
  }
  return extract(body);
}

function asThread(value: unknown): Thread {
  if (typeof value !== "object" || value === null) {
    throw new ThreadApiError(0, "Malformed thread in server response.");
  }
  const r = value as Record<string, unknown>;
  if (
    typeof r["id"] !== "string" ||
    typeof r["ownerId"] !== "string" ||
    typeof r["createdAt"] !== "number" ||
    typeof r["updatedAt"] !== "number" ||
    !(typeof r["title"] === "string" || r["title"] === null)
  ) {
    throw new ThreadApiError(0, "Malformed thread in server response.");
  }
  return {
    id: r["id"],
    ownerId: r["ownerId"],
    title: r["title"],
    createdAt: r["createdAt"],
    updatedAt: r["updatedAt"],
  };
}

// POST /threads — create a thread. Optional title. Returns the new record.
export async function createThread(
  config: ThreadsApiConfig,
  opts?: { title?: string }
): Promise<Thread> {
  const { baseUrl = "", token } = config;
  const body: Record<string, unknown> = {};
  if (opts?.title !== undefined) body["title"] = opts.title;
  return requestJson(
    `${baseUrl}/threads`,
    { method: "POST", headers: authHeaders(token), body: JSON.stringify(body) },
    (parsed) => {
      const record = parsed as Record<string, unknown>;
      return asThread(record["thread"]);
    }
  );
}

// GET /threads — list the authenticated owner's threads (most-recent first).
export async function listThreads(config: ThreadsApiConfig): Promise<Thread[]> {
  const { baseUrl = "", token } = config;
  return requestJson(
    `${baseUrl}/threads`,
    { method: "GET", headers: authHeaders(token) },
    (parsed) => {
      const record = parsed as Record<string, unknown>;
      const threads = record["threads"];
      if (!Array.isArray(threads)) {
        throw new ThreadApiError(0, "Malformed thread list in server response.");
      }
      return threads.map(asThread);
    }
  );
}

// GET /threads/:id — fetch one owned thread. 404 (not-owned/not-found) throws a
// ThreadApiError(404). Used on resume / reconnect to re-fetch thread state.
export async function getThread(
  config: ThreadsApiConfig,
  threadId: string
): Promise<Thread> {
  const { baseUrl = "", token } = config;
  return requestJson(
    `${baseUrl}/threads/${encodeURIComponent(threadId)}`,
    { method: "GET", headers: authHeaders(token) },
    (parsed) => {
      const record = parsed as Record<string, unknown>;
      return asThread(record["thread"]);
    }
  );
}

// One prior conversation message from the history-replay route. Mirrors the
// backend HistoryMessage wire shape (src/core/server.ts): a TEXT-FIRST record —
// role + already-redacted plain text. Historical citations/tools/artifacts are
// DEFERRED, so this carries neither.
export interface HistoryMessage {
  role: "user" | "assistant";
  text: string;
}

// Defensively validate ONE wire history entry: an object with a role of exactly
// "user" | "assistant" and a string text. Anything else is malformed.
function asHistoryMessage(value: unknown): HistoryMessage {
  if (typeof value !== "object" || value === null) {
    throw new ThreadApiError(0, "Malformed message in server response.");
  }
  const r = value as Record<string, unknown>;
  const role = r["role"];
  if (
    (role !== "user" && role !== "assistant") ||
    typeof r["text"] !== "string"
  ) {
    throw new ThreadApiError(0, "Malformed message in server response.");
  }
  return { role, text: r["text"] };
}

// GET /threads/:id/messages — replay an owned thread's prior messages
// (chronological). 404 (not-owned/not-found) throws a ThreadApiError(404); the
// wire shape is validated defensively (an array of { role, text }). Used on
// resume/reconnect to HYDRATE the transcript. Read-only: it never mutates state.
export async function getThreadMessages(
  config: ThreadsApiConfig,
  threadId: string
): Promise<HistoryMessage[]> {
  const { baseUrl = "", token } = config;
  return requestJson(
    `${baseUrl}/threads/${encodeURIComponent(threadId)}/messages`,
    { method: "GET", headers: authHeaders(token) },
    (parsed) => {
      const record = parsed as Record<string, unknown>;
      const messages = record["messages"];
      if (!Array.isArray(messages)) {
        throw new ThreadApiError(
          0,
          "Malformed message list in server response."
        );
      }
      return messages.map(asHistoryMessage);
    }
  );
}
