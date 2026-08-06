import { AIMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import { redactText } from "./redaction.js";
import type { SseToolEvent } from "./sse.js";

// Phase 9 (increment 9b): AGENT MESSAGE -> v2 `tool` SSE EVENT translator.
//
// This is the PURE, deterministic bridge that turns the Phase 6 agent's typed
// LangGraph message stream (see src/core/agent.ts) into the v2 `tool` SSE
// payloads defined by the 9a contract (SseToolEvent in src/core/sse.ts). It is
// the concrete "tool progress/errors from typed events" mechanism that exit
// criterion 3 asks for, on the SSE wire.
//
// WHY A SEPARATE, UNIT-TESTED MODULE. The 9a writer SERIALIZES ONLY — it does
// NOT scrub. The redaction contract for `tool` events puts the responsibility on
// the CALLER: the dynamic free-text fields (a call's `args`, a result's
// `content`) MUST already be redacted via src/core/redaction.ts before reaching
// SseWriter.tool(). This translator is exactly that call-site redaction, kept
// pure and I/O-free so the offline suite can prove — deterministically, without a
// live model — that a secret/PII-shaped string in tool args/content is REDACTED
// on the wire.
//
// SCOPE (9b): this increment WIRES the RAG grounded-answer path end-to-end into
// the streaming chat handler (see src/core/server.ts). The FULL live-agent
// tool-loop hookup into the endpoint is deferred to a later increment; the
// translator below is the EMISSION PATH for tool events (the mechanism), driven
// behind the injected agent seam and proven by test/phase9-server-rag.test.ts.
// Providing + testing the translator now keeps crit 3's tool-progress guarantee
// genuinely demonstrable without prematurely coupling the react-agent into the
// endpoint alongside RAG.

// The projection of ONE agent message into ZERO-OR-MORE v2 `tool` events:
//   - an AIMessage carrying `tool_calls` -> one "call" event per tool_call
//     (the tool INTENTION: name + toolCallId + REDACTED parsed args);
//   - a ToolMessage -> one "result" event (the tool RESULT/ERROR: name +
//     toolCallId + status ok|error + REDACTED content summary);
//   - anything else (a plain AIMessage final answer, a HumanMessage, a
//     SystemMessage) -> no `tool` events (that text is the token stream's job).
//
// Every returned SseToolEvent is ALREADY REDACTED and safe to hand straight to
// SseWriter.tool(). PURE + deterministic.
export function toolEventsForMessage(message: BaseMessage): SseToolEvent[] {
  if (message instanceof AIMessage) {
    return toolCallEvents(message);
  }
  if (message instanceof ToolMessage) {
    const event = toolResultEvent(message);
    return event === null ? [] : [event];
  }
  return [];
}

// Project an AIMessage's tool_calls into "call" events. A tool_call is an
// INTENTION (name + parsed args + id). The parsed `args` object is the ONLY
// dynamic free-text field, so every string VALUE is redacted before it reaches
// the wire (per the caller-redacts contract). Keys are structural and left as-is.
function toolCallEvents(message: AIMessage): SseToolEvent[] {
  const calls = message.tool_calls ?? [];
  const events: SseToolEvent[] = [];
  for (const call of calls) {
    // A tool_call id is required to correlate a later result; skip a malformed
    // call with no id rather than emit an uncorrelatable event.
    if (typeof call.id !== "string" || call.id === "") continue;
    events.push({
      phase: "call",
      name: call.name,
      toolCallId: call.id,
      args: redactArgs(call.args),
    });
  }
  return events;
}

// Project a ToolMessage into a "result" event. `status: "error"` (also how the
// prebuilt ToolNode refuses an unknown/unpermitted tool — see agent.ts) maps to
// the wire status "error"; anything else is "ok". The ToolMessage `content` is
// dynamic free-text (tool output / an error message), so it is REDACTED into a
// single-line summary before the wire.
function toolResultEvent(message: ToolMessage): SseToolEvent | null {
  const toolCallId = message.tool_call_id;
  if (typeof toolCallId !== "string" || toolCallId === "") return null;
  const status = message.status === "error" ? "error" : "ok";
  const content = contentToText(message.content);
  const event: SseToolEvent = {
    phase: "result",
    // `name` is optional on a ToolMessage in the type surface; fall back to a
    // fixed, non-dynamic placeholder so the wire field is always present.
    name: message.name ?? "tool",
    toolCallId,
    status,
  };
  // Only attach `content` when there is something to show, and always redacted.
  if (content !== "") {
    event.content = redactText(content);
  }
  return event;
}

// Redact a tool_call args object for the wire. The args are JSON-safe structured
// values from the model; any STRING value could carry dynamic free-text (and, in
// an adversarial case, a secret/PII-shaped string), so each string is passed
// through redactText. Non-string primitives (number/boolean/null) are structural
// and copied verbatim; nested objects/arrays are recursed so a secret buried in a
// nested field is still scrubbed. The result stays a JSON-safe Record.
function redactArgs(args: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    out[key] = redactValue(value);
  }
  return out;
}

function redactValue(value: unknown): unknown {
  if (typeof value === "string") return redactText(value);
  if (Array.isArray(value)) return value.map(redactValue);
  if (value !== null && typeof value === "object") {
    return redactArgs(value as Record<string, unknown>);
  }
  // number | boolean | null | undefined -> structural, safe as-is.
  return value;
}

// Normalize LangChain message content (string | complex-part array) to a plain
// string for a redacted summary. Mirrors agent.ts's messageToText join over text
// parts; non-text parts collapse to "" by design (a tool result summary is text).
function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) =>
        typeof part === "string"
          ? part
          : part !== null &&
              typeof part === "object" &&
              (part as { type?: unknown }).type === "text"
            ? String((part as { text?: unknown }).text ?? "")
            : ""
      )
      .join("");
  }
  return "";
}
