// Phase 9 (9c): the in-memory conversation VIEW MODEL the UI renders.
//
// This is a pure, framework-agnostic reducer over the TYPED SSE events (see
// sse-contract.ts). Keeping the turn state as a pure reducer means the rendering
// is a deterministic function of the trusted event stream — citations, tool
// progress, artifacts, and evidenceStatus are all derived ONLY from typed events
// (exit criterion 3), never fabricated. It is also unit-testable without React.

import type {
  SseArtifactEvent,
  SseCitation,
  SseToolEvent,
  EvidenceStatus,
} from "./sse-contract.js";

// A single chat message shown in the transcript.
export interface UserMessage {
  role: "user";
  id: string;
  text: string;
}

// The assistant's turn, accumulated from the typed event stream. `status`
// tracks the streaming lifecycle so the UI can show streaming / cancel / retry.
export interface AssistantTurn {
  role: "assistant";
  id: string;
  // Incrementally accumulated token text (or the single grounded answer token).
  text: string;
  // TRUSTED citations from the `citation` event (app-owned fields only).
  citations: SseCitation[];
  // Present once a `citation` event arrived; the app-authoritative evidence
  // state. null before it arrives.
  evidenceStatus: EvidenceStatus | null;
  // Tool-progress entries from `tool` events, in arrival order.
  toolEvents: SseToolEvent[];
  // Structured artifacts from `artifact` events, in arrival order.
  artifacts: SseArtifactEvent[];
  // Lifecycle: streaming -> done | error | canceled.
  status: "streaming" | "done" | "error" | "canceled";
  // A user-safe, already-redacted error message (only when status === "error").
  error: string | null;
  // The user message that produced this turn, retained so RETRY can re-issue it.
  sourceMessage: string;
}

export type ChatMessage = UserMessage | AssistantTurn;

// Create a fresh assistant turn in the streaming state for a given user message.
export function newAssistantTurn(id: string, sourceMessage: string): AssistantTurn {
  return {
    role: "assistant",
    id,
    text: "",
    citations: [],
    evidenceStatus: null,
    toolEvents: [],
    artifacts: [],
    status: "streaming",
    error: null,
    sourceMessage,
  };
}

// Pure updates for an assistant turn (return a NEW object; never mutate). React
// state setters compose these; unit tests exercise them directly.
export const turnReducers = {
  appendToken(turn: AssistantTurn, text: string): AssistantTurn {
    return { ...turn, text: turn.text + text };
  },
  setCitations(
    turn: AssistantTurn,
    citations: SseCitation[],
    evidenceStatus: EvidenceStatus
  ): AssistantTurn {
    return { ...turn, citations, evidenceStatus };
  },
  addTool(turn: AssistantTurn, tool: SseToolEvent): AssistantTurn {
    return { ...turn, toolEvents: [...turn.toolEvents, tool] };
  },
  addArtifact(turn: AssistantTurn, artifact: SseArtifactEvent): AssistantTurn {
    return { ...turn, artifacts: [...turn.artifacts, artifact] };
  },
  // Finalize on `done`. If the turn produced no incremental token text, fall
  // back to the done event's full accumulated text.
  finishDone(turn: AssistantTurn, fullText: string): AssistantTurn {
    return {
      ...turn,
      text: turn.text.length > 0 ? turn.text : fullText,
      status: "done",
    };
  },
  fail(turn: AssistantTurn, message: string): AssistantTurn {
    return { ...turn, status: "error", error: message };
  },
  cancel(turn: AssistantTurn): AssistantTurn {
    return { ...turn, status: "canceled" };
  },
} as const;
