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
        <h1>Wedding Planner</h1>
        <TokenGate
          onSubmit={(t) => {
            storeToken(t);
            setToken(t);
          }}
        />
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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  // The AbortController for the in-flight stream (CANCEL seam). A ref so cancel
  // can reach the live controller without a re-render dependency.
  const controllerRef = useRef<AbortController | null>(null);

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

  const handleCreate = useCallback(async (): Promise<void> => {
    try {
      const thread = await createThread({ baseUrl: BASE_URL, token });
      setThreads((prev) => [thread, ...prev]);
      setCurrentThreadId(thread.id);
      setMessages([]);
      setNotice(null);
    } catch (err) {
      handleApiError(err, onSignOut, setNotice);
    }
  }, [token, onSignOut]);

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
        setCurrentThreadId(thread.id);
        setMessages([]);
        setNotice(null);
      } catch (err) {
        handleApiError(err, onSignOut, setNotice);
      }
    },
    [token, onSignOut]
  );

  // Start a chat stream for `message` on the current thread. Shared by SEND and
  // RETRY. Appends a user message (unless retrying) + a fresh assistant turn,
  // then wires the typed handlers into the reducer.
  const runTurn = useCallback(
    (message: string, opts?: { appendUser?: boolean }): void => {
      const threadId = currentThreadId;
      if (threadId === null || streaming) return;

      const appendUser = opts?.appendUser ?? true;
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
    [
      currentThreadId,
      streaming,
      token,
      updateCurrentTurn,
      refreshThreads,
    ]
  );

  // CANCEL: abort the in-flight stream. The finally-block marks the turn
  // canceled; SseWriter/backends stop the turn server-side via the disconnect.
  const handleCancel = useCallback((): void => {
    controllerRef.current?.abort();
  }, []);

  // RETRY: re-issue the last message WITHOUT appending a duplicate user bubble.
  const handleRetry = useCallback(
    (sourceMessage: string): void => {
      runTurn(sourceMessage, { appendUser: false });
    },
    [runTurn]
  );

  return (
    <div className="app app--conversation">
      <header className="app__header">
        <h1>Wedding Planner</h1>
        <button type="button" onClick={onSignOut} data-testid="sign-out">
          Sign out
        </button>
      </header>

      {notice !== null ? (
        <div className="app__notice" data-testid="notice">
          {notice}
        </div>
      ) : null}

      <div className="app__body">
        <ThreadList
          threads={threads}
          currentThreadId={currentThreadId}
          onSelect={(id) => void handleSelect(id)}
          onCreate={() => void handleCreate()}
          onRefresh={() => void refreshThreads()}
          busy={streaming}
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
            disabled={currentThreadId === null}
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
