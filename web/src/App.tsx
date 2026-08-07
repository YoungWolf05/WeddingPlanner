// Phase 9 (9c): the Wedding Planner browser SPA — a THIN client over the stable
// v2 SSE contract.
//
// WHAT THIS ORCHESTRATES (targets exit criteria 1/2/3):
//   - AUTH (crit 1): a bearer token (held in memory + mirrored to sessionStorage
//     for the tab session) is sent as `Authorization: Bearer <token>` on every
//     call. The backend derives ownerId from it; the client NEVER sends an
//     owner/user id. NO provider credential is referenced anywhere.
//   - THREADS: create / list / resume via the REST client (owner-scoped
//     server-side).
//   - STREAMING CHAT (crit 2): POST the message to the chat SSE endpoint, render
//     init -> token* -> citation -> artifact -> done, with tool events mid-stream.
//     CANCEL aborts the in-flight stream; RETRY re-issues the last message;
//     RECONNECT re-fetches thread state + lets the user re-issue.
//   - TYPED RENDERING (crit 3): citations / tool progress / artifacts /
//     evidenceStatus render ONLY from the typed trusted events (see the
//     conversation reducer + the presentational components).

import { useCallback, useEffect, useRef, useState } from "react";
import { Composer } from "./components/Composer.js";
import { ThreadList } from "./components/ThreadList.js";
import { TokenGate } from "./components/TokenGate.js";
import { Transcript } from "./components/Transcript.js";
import {
  newAssistantTurn,
  turnReducers,
  type AssistantTurn,
  type ChatMessage,
} from "./lib/conversation.js";
import { startChatStream } from "./lib/sseClient.js";
import { deriveThreadTitle, UNTITLED_PLACEHOLDER } from "./lib/title.js";
import {
  createThread,
  getThread,
  listThreads,
  ThreadApiError,
  type Thread,
} from "./lib/threadsApi.js";

// sessionStorage key for the bearer token. sessionStorage (not localStorage) so
// the token lives only for the tab session. It is NEVER in the bundle.
const TOKEN_STORAGE_KEY = "wp.bearerToken";

// Same-origin backend base (dev proxy in dev; same host in prod). The client
// only ever talks to the backend service — never a provider.
const BASE_URL = "";

// A small monotonic id generator for message keys (avoids relying on a specific
// crypto API being present in every test environment).
let idCounter = 0;
function nextId(prefix: string): string {
  idCounter += 1;
  return `${prefix}-${String(idCounter)}`;
}

export function App(): React.ReactElement {
  const [token, setToken] = useState<string | null>(() =>
    readStoredToken()
  );

  if (token === null) {
    return (
      <div className="app app--gate">
        <div className="gate">
          <div className="gate__brand">
            <span className="wordmark__mark wordmark__mark--large" aria-hidden="true">
              A
            </span>
            <h1 className="gate__title">Wedding Planner</h1>
            <p className="gate__lede">
              Meet Aria — your calm, well-sourced companion for planning the day.
            </p>
          </div>
          <TokenGate
            onSubmit={(t) => {
              storeToken(t);
              setToken(t);
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <Conversation
      token={token}
      onSignOut={() => {
        clearStoredToken();
        setToken(null);
      }}
    />
  );
}

interface ConversationProps {
  token: string;
  onSignOut: () => void;
}

function Conversation(props: ConversationProps): React.ReactElement {
  const { token, onSignOut } = props;

  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  // BUG 1 (lazy-create): a brand-new conversation is a UI-only DRAFT until the
  // first message is sent. `draft` true means "New conversation" is prepared
  // (transcript cleared, composer enabled) but NO server row exists yet — the
  // thread is created with a derived title on the first send so the title
  // PERSISTS (the backend only accepts a title at creation; there is no update
  // route). A truly-nothing-selected state is `draft === false && currentThreadId
  // === null`, which still blocks the composer.
  const [draft, setDraft] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  // The AbortController for the in-flight stream (CANCEL seam). A ref so cancel
  // can reach the live controller without a re-render dependency.
  const controllerRef = useRef<AbortController | null>(null);
  // Guards the lazy thread-creation on the first send so rapid double-sends
  // cannot create two server threads (a ref: it must gate synchronously, before
  // any re-render, and is not itself rendered).
  const creatingRef = useRef(false);

  // Update the current (last) assistant turn via a pure reducer.
  const updateCurrentTurn = useCallback(
    (turnId: string, fn: (turn: AssistantTurn) => AssistantTurn): void => {
      setMessages((prev) =>
        prev.map((m) =>
          m.role === "assistant" && m.id === turnId ? fn(m) : m
        )
      );
    },
    []
  );

  const refreshThreads = useCallback(async (): Promise<void> => {
    try {
      const list = await listThreads({ baseUrl: BASE_URL, token });
      setThreads(list);
      setNotice(null);
    } catch (err) {
      handleApiError(err, onSignOut, setNotice);
    }
  }, [token, onSignOut]);

  // Load threads on mount / token change.
  useEffect(() => {
    void refreshThreads();
  }, [refreshThreads]);

  // "+ New conversation" — prepare a UI DRAFT. No server call yet: the thread is
  // created lazily on the first send (with a title derived from that message) so
  // the title persists. Cancels any in-flight stream and resets the transcript.
  const handleCreate = useCallback((): void => {
    controllerRef.current?.abort();
    setDraft(true);
    setCurrentThreadId(null);
    setMessages([]);
    setNotice(null);
  }, []);

  // RESUME / RECONNECT: select a thread and re-fetch its state. (History
  // rehydration is bounded by the backend contract; for 9c we re-fetch the
  // thread record to confirm ownership/existence and reset the transcript. Full
  // history replay is a backend affordance not yet exposed and is left to 9d /
  // a later increment — documented in web/README.md.)
  const handleSelect = useCallback(
    async (threadId: string): Promise<void> => {
      // Cancel any in-flight stream before switching context.
      controllerRef.current?.abort();
      try {
        const thread = await getThread({ baseUrl: BASE_URL, token }, threadId);
        setDraft(false); // selecting a real thread discards any pending draft.
        setCurrentThreadId(thread.id);
        setMessages([]);
        setNotice(null);
      } catch (err) {
        handleApiError(err, onSignOut, setNotice);
      }
    },
    [token, onSignOut]
  );

  // Start a chat stream on a KNOWN thread id. Shared internals of SEND / RETRY:
  // appends a user message (unless retrying) + a fresh assistant turn, then wires
  // the typed handlers into the reducer.
  const streamTurn = useCallback(
    (threadId: string, message: string, appendUser: boolean): void => {
      const turnId = nextId("assistant");

      setMessages((prev) => {
        const next: ChatMessage[] = [...prev];
        if (appendUser) {
          next.push({ role: "user", id: nextId("user"), text: message });
        }
        next.push(newAssistantTurn(turnId, message));
        return next;
      });

      const controller = new AbortController();
      controllerRef.current = controller;
      setStreaming(true);

      void startChatStream({
        baseUrl: BASE_URL,
        threadId,
        message,
        token,
        signal: controller.signal,
        handlers: {
          onToken: (data) =>
            updateCurrentTurn(turnId, (t) =>
              turnReducers.appendToken(t, data.text)
            ),
          onCitation: (data) =>
            updateCurrentTurn(turnId, (t) =>
              turnReducers.setCitations(t, data.citations, data.evidenceStatus)
            ),
          onTool: (data) =>
            updateCurrentTurn(turnId, (t) => turnReducers.addTool(t, data)),
          onArtifact: (data) =>
            updateCurrentTurn(turnId, (t) => turnReducers.addArtifact(t, data)),
          onDone: (data) => {
            updateCurrentTurn(turnId, (t) =>
              turnReducers.finishDone(t, data.text)
            );
          },
          onError: (m) =>
            updateCurrentTurn(turnId, (t) => turnReducers.fail(t, m)),
        },
      }).finally(() => {
        // If the turn is still "streaming" here, the stream ended without a
        // done/error frame (EOF / cancel). Distinguish a user cancel (signal
        // aborted) from an unexpected drop so RECONNECT/RETRY reads sensibly.
        updateCurrentTurn(turnId, (t) => {
          if (t.status !== "streaming") return t;
          return controller.signal.aborted
            ? turnReducers.cancel(t)
            : turnReducers.fail(
                t,
                "The stream ended unexpectedly. You can retry."
              );
        });
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
        setStreaming(false);
        // Refresh thread list so recency ordering reflects the completed turn.
        void refreshThreads();
      });
    },
    [token, updateCurrentTurn, refreshThreads]
  );

  // SEND. Starts a turn on the current thread. If the conversation is a DRAFT
  // (no server row yet), it lazily creates the thread FIRST — using a title
  // derived from this first message so the title PERSISTS — then streams the
  // turn on the new id. Subsequent sends stream directly. Guarded so a rapid
  // double-send cannot create two threads or overlap streams.
  const runTurn = useCallback(
    (message: string): boolean | Promise<boolean> => {
      if (streaming || creatingRef.current) return false;

      // Existing thread: stream directly (accepted synchronously).
      if (currentThreadId !== null && !draft) {
        streamTurn(currentThreadId, message, true);
        return true;
      }

      // No thread and no draft -> nothing to send to (rejected).
      if (!draft) return false;

      // Draft: create the server thread with a derived title, then stream. The
      // returned promise resolves false on failure so the composer restores the
      // typed message (nothing is lost).
      creatingRef.current = true;
      const title = deriveThreadTitle(message) ?? UNTITLED_PLACEHOLDER;
      return (async (): Promise<boolean> => {
        try {
          const thread = await createThread(
            { baseUrl: BASE_URL, token },
            { title }
          );
          setThreads((prev) => [thread, ...prev]);
          setDraft(false);
          setCurrentThreadId(thread.id);
          setNotice(null);
          // Stream the first turn on the freshly created thread.
          streamTurn(thread.id, message, true);
          return true;
        } catch (err) {
          // Keep the draft so the user can retry; surface a user-safe notice
          // (401 signs out). The composer restores the message on `false`.
          handleApiError(err, onSignOut, setNotice);
          return false;
        } finally {
          creatingRef.current = false;
        }
      })();
    },
    [streaming, currentThreadId, draft, token, streamTurn, onSignOut]
  );

  // CANCEL: abort the in-flight stream. The finally-block marks the turn
  // canceled; SseWriter/backends stop the turn server-side via the disconnect.
  const handleCancel = useCallback((): void => {
    controllerRef.current?.abort();
  }, []);

  // RETRY: re-issue the last message WITHOUT appending a duplicate user bubble.
  // Retry only ever fires on an EXISTING turn, so the thread already exists.
  const handleRetry = useCallback(
    (sourceMessage: string): void => {
      if (currentThreadId === null || streaming) return;
      streamTurn(currentThreadId, sourceMessage, false);
    },
    [currentThreadId, streaming, streamTurn]
  );

  // The composer is usable when there is a real thread OR an unsaved draft; a
  // truly-nothing state (no thread, no draft) still blocks input.
  const composerDisabled = currentThreadId === null && !draft;

  return (
    <div className="app app--conversation">
      <header className="app__header">
        <div className="wordmark">
          <span className="wordmark__mark" aria-hidden="true">
            A
          </span>
          <span className="wordmark__text">
            <span className="wordmark__name">Wedding Planner</span>
            <span className="wordmark__tagline">with Aria</span>
          </span>
        </div>
        <button
          type="button"
          onClick={onSignOut}
          data-testid="sign-out"
          className="btn btn--ghost"
        >
          Sign out
        </button>
      </header>

      {notice !== null ? (
        <div className="app__notice" data-testid="notice" role="alert">
          {notice}
        </div>
      ) : null}

      <div className="app__body">
        <ThreadList
          threads={threads}
          currentThreadId={currentThreadId}
          onSelect={(id) => void handleSelect(id)}
          onCreate={handleCreate}
          onRefresh={() => void refreshThreads()}
          busy={streaming}
          draft={draft}
        />

        <main className="app__main">
          <Transcript
            messages={messages}
            onRetry={handleRetry}
            streaming={streaming}
          />
          <Composer
            onSend={(m) => runTurn(m)}
            onCancel={handleCancel}
            streaming={streaming}
            disabled={composerDisabled}
          />
        </main>
      </div>
    </div>
  );
}

// --- token storage helpers (session-scoped; never bundled) ------------------

function readStoredToken(): string | null {
  try {
    const stored = sessionStorage.getItem(TOKEN_STORAGE_KEY);
    return stored !== null && stored.trim() !== "" ? stored : null;
  } catch {
    return null; // storage unavailable (e.g. private mode) — in-memory only.
  }
}

function storeToken(token: string): void {
  try {
    sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  } catch {
    // Non-fatal: the token still lives in React state for this session.
  }
}

function clearStoredToken(): void {
  try {
    sessionStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // Non-fatal.
  }
}

// Centralized API-error handling: an auth failure (401) signs the user out (the
// stored token is invalid); everything else shows a user-safe notice.
function handleApiError(
  err: unknown,
  onSignOut: () => void,
  setNotice: (message: string) => void
): void {
  if (err instanceof ThreadApiError && err.status === 401) {
    onSignOut();
    return;
  }
  const message =
    err instanceof ThreadApiError
      ? err.message
      : "Something went wrong. Please try again.";
  setNotice(message);
}
