// Phase 9 (9c): the fetch-based SSE client for the streaming chat endpoint.
//
// WHY fetch + ReadableStream (not EventSource). The chat endpoint is a POST
// (it carries the chat message in a JSON body) AND it requires an
// `Authorization: Bearer <token>` header — the ONLY auth. The browser
// EventSource API can neither POST nor set an Authorization header, so we use
// fetch() with a POST + the bearer header + a JSON body, then read the
// `text/event-stream` response body via a ReadableStream reader and parse the
// SSE frames MANUALLY. This is also the SEAM 9d exercises for cancel / retry /
// reconnect (see startChatStream + the AbortController below).
//
// SECURITY. The bearer token is the ONLY credential the client holds; the
// backend derives ownerId from it (never sent by the client). No provider
// credential (LITELLM_*) is ever referenced here or anywhere in the frontend.

import {
  SSE_EVENT,
  SSE_PROTOCOL_VERSION,
  type SseArtifactEvent,
  type SseCitationEvent,
  type SseClientEvent,
  type SseDoneEvent,
  type SseErrorEvent,
  type SseInitEvent,
  type SseToolEvent,
  type SseTokenEvent,
} from "./sse-contract.js";

// Handlers the caller supplies to react to the typed stream. Each is optional so
// a consumer subscribes only to what it renders. All events are TYPED — the app
// never sees a raw frame.
export interface ChatStreamHandlers {
  onEvent?: (event: SseClientEvent) => void;
  onInit?: (data: SseInitEvent) => void;
  onToken?: (data: SseTokenEvent) => void;
  onCitation?: (data: SseCitationEvent) => void;
  onTool?: (data: SseToolEvent) => void;
  onArtifact?: (data: SseArtifactEvent) => void;
  onDone?: (data: SseDoneEvent) => void;
  // A user-safe error. Sources: a backend `error` SSE frame (already redacted),
  // a protocol-version mismatch, a non-2xx HTTP status, or a transport failure.
  // NOT called for a caller-initiated cancel (that resolves quietly).
  onError?: (message: string) => void;
}

export interface StartChatStreamArgs {
  // Backend base path. Empty string = same-origin (the dev proxy / prod). The
  // client only ever talks to the backend service — never a provider.
  baseUrl?: string;
  threadId: string;
  message: string;
  // The bearer token. Sent as `Authorization: Bearer <token>`. The ONLY auth.
  token: string;
  handlers: ChatStreamHandlers;
  // Cancel seam: abort this to stop an in-flight stream (see cancel/retry below).
  signal: AbortSignal;
}

// A generic, user-safe error message for cases where we must not surface raw
// details (transport internals could otherwise leak environment specifics).
const GENERIC_ERROR = "The connection to the assistant failed. Please retry.";

// Build the chat SSE URL for a thread. Mirrors src/core/server.ts routing:
// POST /threads/:id/chat. The id is URL-encoded (it is a server-issued UUID, but
// encode defensively so it is always a single safe path segment).
export function chatUrl(baseUrl: string, threadId: string): string {
  return `${baseUrl}/threads/${encodeURIComponent(threadId)}/chat`;
}

// Parse the payload of one SSE frame into a typed SseClientEvent, or null if the
// event name is unknown (per the SSE spec, unknown events are IGNORED — this
// keeps a future additive contract bump from breaking an older client) or the
// JSON is malformed. Kept PURE + exported so it is unit-testable in isolation.
export function parseSseFrame(
  eventName: string,
  data: string
): SseClientEvent | null {
  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    return null;
  }
  switch (eventName) {
    case SSE_EVENT.init:
      return { type: SSE_EVENT.init, data: payload as SseInitEvent };
    case SSE_EVENT.token:
      return { type: SSE_EVENT.token, data: payload as SseTokenEvent };
    case SSE_EVENT.done:
      return { type: SSE_EVENT.done, data: payload as SseDoneEvent };
    case SSE_EVENT.error:
      return { type: SSE_EVENT.error, data: payload as SseErrorEvent };
    case SSE_EVENT.citation:
      return { type: SSE_EVENT.citation, data: payload as SseCitationEvent };
    case SSE_EVENT.tool:
      return { type: SSE_EVENT.tool, data: payload as SseToolEvent };
    case SSE_EVENT.artifact:
      return { type: SSE_EVENT.artifact, data: payload as SseArtifactEvent };
    default:
      return null; // unknown event name — ignore (SSE spec / forward-compat).
  }
}

// Split a raw SSE text buffer into complete frames (separated by a blank line)
// and a trailing remainder. PURE + exported for unit testing. Frames are
// separated by "\n\n"; a partial trailing frame (no terminator yet) is returned
// as `rest` to be prepended to the next chunk.
export function splitSseFrames(buffer: string): {
  frames: string[];
  rest: string;
} {
  // Normalize CRLF to LF so a frame terminator is uniformly "\n\n".
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  return { frames: parts, rest };
}

// Extract the `event:` name (default "message" per the SSE spec) and the joined
// `data:` payload from one raw frame's lines. PURE + exported for unit testing.
export function parseFrameLines(frame: string): {
  eventName: string;
  data: string;
} {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const rawLine of frame.split("\n")) {
    const line = rawLine.trimEnd();
    if (line === "" || line.startsWith(":")) continue; // blank / comment line.
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    // Per spec, a single leading space after the colon is stripped.
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventName = value;
    else if (field === "data") dataLines.push(value);
  }
  return { eventName, data: dataLines.join("\n") };
}

// Dispatch one typed event to the matching handlers. `onEvent` fires for every
// event; the specific `on<Kind>` fires for its kind. An `error` frame is routed
// through onError so the app renders the backend's already-redacted message.
function dispatch(event: SseClientEvent, handlers: ChatStreamHandlers): void {
  handlers.onEvent?.(event);
  switch (event.type) {
    case SSE_EVENT.init:
      handlers.onInit?.(event.data);
      break;
    case SSE_EVENT.token:
      handlers.onToken?.(event.data);
      break;
    case SSE_EVENT.citation:
      handlers.onCitation?.(event.data);
      break;
    case SSE_EVENT.tool:
      handlers.onTool?.(event.data);
      break;
    case SSE_EVENT.artifact:
      handlers.onArtifact?.(event.data);
      break;
    case SSE_EVENT.done:
      handlers.onDone?.(event.data);
      break;
    case SSE_EVENT.error:
      handlers.onError?.(event.data.message);
      break;
  }
}

// Start a chat stream. Resolves when the stream ends (done / error / EOF) or is
// canceled; it never rejects — failures are surfaced via handlers.onError so the
// UI has a single, user-safe error path.
//
// CANCEL. Abort the passed `signal` (via an AbortController the caller owns) to
// stop the in-flight stream: fetch is aborted, the reader is released, and this
// resolves quietly WITHOUT calling onError (a user cancel is not an error).
//
// RETRY. The caller re-invokes startChatStream with a FRESH AbortController and
// the same message (see App's retry) — the stream is stateless per turn.
//
// RECONNECT. On a dropped stream (EOF before `done`, or a transport error), the
// caller may re-fetch thread state and/or re-issue the message. The backend turn
// is bounded by the SSE idle timeout, so a stalled turn ends deterministically.
export async function startChatStream(args: StartChatStreamArgs): Promise<void> {
  const { baseUrl = "", threadId, message, token, handlers, signal } = args;

  let versionValidated = false;

  let response: Response;
  try {
    response = await fetch(chatUrl(baseUrl, threadId), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ message }),
      signal,
    });
  } catch (err) {
    // A caller-initiated cancel surfaces as an AbortError — resolve quietly.
    if (isAbortError(err)) return;
    handlers.onError?.(GENERIC_ERROR);
    return;
  }

  if (!response.ok) {
    // Non-2xx (e.g. 401 unauthorized, 404 not-owned/not-found, 400 bad request,
    // 413 too large). The backend body is redacted JSON; surface a concise,
    // user-safe message keyed off the status so the UI can react (e.g. 401).
    handlers.onError?.(httpErrorMessage(response.status));
    return;
  }

  const body = response.body;
  if (body === null) {
    handlers.onError?.(GENERIC_ERROR);
    return;
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { frames, rest } = splitSseFrames(buffer);
      buffer = rest;
      for (const frame of frames) {
        if (frame.trim() === "") continue;
        const { eventName, data } = parseFrameLines(frame);
        const event = parseSseFrame(eventName, data);
        if (event === null) continue; // unknown/malformed — ignore.

        // PROTOCOL-VERSION GUARD. The stream MUST open with `init`, and its
        // `version` MUST match what this client understands. On a mismatch we
        // surface a user-safe error and STOP — better than misparsing a newer
        // (or older) contract. (A version-less first event is also rejected.)
        if (event.type === SSE_EVENT.init) {
          if (event.data.version !== SSE_PROTOCOL_VERSION) {
            handlers.onError?.(
              `Unsupported server protocol version ${String(
                event.data.version
              )}. This client expects version ${String(SSE_PROTOCOL_VERSION)}.`
            );
            return;
          }
          versionValidated = true;
        } else if (!versionValidated) {
          // A non-init event before a validated init: refuse to interpret it.
          handlers.onError?.(
            "The assistant stream did not start with a valid handshake."
          );
          return;
        }

        dispatch(event, handlers);
      }
    }
  } catch (err) {
    if (isAbortError(err)) return; // user cancel — quiet.
    handlers.onError?.(GENERIC_ERROR);
    return;
  } finally {
    reader.releaseLock();
  }
}

// Map an HTTP status to a concise, user-safe message. The backend response
// bodies are already redacted; we key off the STATUS only (never echo a body).
function httpErrorMessage(status: number): string {
  switch (status) {
    case 401:
      return "Unauthorized. Check your access token and try again.";
    case 404:
      return "That conversation was not found.";
    case 400:
      return "The message could not be sent (bad request).";
    case 413:
      return "The message is too large to send.";
    default:
      return GENERIC_ERROR;
  }
}

// True if an error is a fetch/stream AbortError (caller-initiated cancel).
function isAbortError(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "name" in err &&
    (err as { name?: unknown }).name === "AbortError"
  );
}
