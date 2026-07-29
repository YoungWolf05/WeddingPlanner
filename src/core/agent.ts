// NOTE on createReactAgent: in the installed @langchain/langgraph (1.4.7) the
// createReactAgent SYMBOL is itself @deprecated — its JSDoc steers toward
// `import { createAgent } from "langchain"` (the langchain meta-package). It
// remains fully functional and typechecks/runs correctly today, and we use it
// INTENTIONALLY here: per the accepted Phase 6 decision we deliberately did NOT
// add the langchain meta-package. Migrating to langchain's createAgent is a
// deliberate FUTURE decision, not done now to avoid that new dependency. Note we
// pass `prompt` (see below) — the CURRENT, non-deprecated option — rather than
// the deprecated messageModifier/stateModifier.
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
} from "@langchain/core/messages";
import { createChatModel, type ModelOptions } from "./model.js";
import { isTemperatureOmitModel } from "./structured.js";
import { WEDDING_PLANNER_SYSTEM_PROMPT } from "./prompts.js";
import { weddingTools } from "./tools.js";

// Phase 6 (increment 6c): the wedding-planning TOOL-LOOP AGENT.
//
// This module wires the SAFE, read-only 6b tools (`weddingTools`: days_until +
// split_budget) into a LangGraph PREBUILT react (tool-loop) agent. It is the
// agent FOUNDATION only — this increment deliberately does NOT touch the HTTP
// service or CLI (that wiring is a later step), and it leaves the plain-chat
// conversational graph (`createConversationalChain` in chain.ts) UNTOUCHED.
//
// Two carry-forward invariants from earlier phases are honored:
//   1. Model construction goes through createChatModel() (the 4b single-factory
//      rule) — this module NEVER constructs ChatOpenAI directly.
//   2. The opus temperature-omit constraint (shared with 6a's structured path):
//      for claude-opus-4-8 the model is built with `temperature: null` so the
//      field is OMITTED from the provider payload (opus deprecates an explicit
//      temperature on its tool/structured path and errors when one is sent).
//      The decision reuses the SHARED predicate `isTemperatureOmitModel` from
//      structured.ts, so the "opus => omit temperature" rule lives in ONE place.

// Default agent model. claude-sonnet-4-6 is fully supported and works well for
// tool calling whether or not a temperature is sent, so it is the safe default
// when a caller does not pick a model. (A fixed constant rather than
// config.model so the agent has a stable, documented default.)
export const DEFAULT_AGENT_MODEL = "claude-sonnet-4-6";

// Options accepted by createWeddingAgent.
export interface WeddingAgentOptions {
  // Model id to use. Defaults to DEFAULT_AGENT_MODEL.
  model?: string;
}

// PURE decision function (offline-unit-testable) mapping a model id to the exact
// ModelOptions used to build the AGENT's chat model. This mirrors 6a's
// decideStructuredModelOptions and shares the SAME opus predicate, so there is
// exactly one definition of the "opus => omit temperature" rule:
//   - claude-opus-4-8  -> { model, temperature: null }  (temperature OMITTED)
//   - anything else     -> { model }                     (factory default temp)
// Returning `temperature: null` (rather than a number) is what makes
// createChatModel omit the field entirely — see ModelOptions in model.ts.
export function decideAgentModelOptions(model: string): ModelOptions {
  if (isTemperatureOmitModel(model)) {
    return { model, temperature: null };
  }
  return { model };
}

/**
 * The compiled wedding tool-loop agent type. `createReactAgent` returns a
 * compiled LangGraph you can `.invoke({ messages })` / `.stream(...)`. We derive
 * the type from the factory's return so it stays exact without restating the
 * library's generic surface.
 */
export type WeddingAgent = ReturnType<typeof createReactAgent>;

/**
 * Build the wedding-planning tool-loop agent.
 *
 * Wires the SAFE 6b tools into a LangGraph prebuilt react agent:
 *   - `llm`    — a chat model from createChatModel(), honoring the opus
 *                temperature-omit rule via decideAgentModelOptions().
 *   - `tools`  — the read-only weddingTools ([days_until, split_budget]).
 *   - `prompt` — WEDDING_PLANNER_SYSTEM_PROMPT (persona "Aria"), passed as the
 *                current `prompt` option (NOT the deprecated
 *                messageModifier/stateModifier).
 *
 * The returned value is the compiled graph: call `.invoke({ messages: [...] })`
 * or `.stream(...)`. Use runWeddingAgent() for a thin "final text" convenience.
 */
export function createWeddingAgent(
  options: WeddingAgentOptions = {}
): WeddingAgent {
  const modelId = options.model ?? DEFAULT_AGENT_MODEL;
  const llm = createChatModel(decideAgentModelOptions(modelId));
  return createReactAgent({
    llm,
    // `weddingTools` is a readonly tuple; createReactAgent takes a mutable
    // array, so we spread into a fresh array (the tools themselves are shared).
    tools: [...weddingTools],
    prompt: WEDDING_PLANNER_SYSTEM_PROMPT,
  });
}

/**
 * Thin convenience runner: invoke `agent` with a single user message and return
 * the FINAL assistant text.
 *
 * The agent's terminal message is the last message in the returned state
 * (an AIMessage with no further tool_calls). Its `content` is normalized to a
 * plain string (LangChain content can be a string or an array of parts).
 */
export async function runWeddingAgent(
  agent: WeddingAgent,
  userMessage: string
): Promise<string> {
  const result = await agent.invoke({
    messages: [new HumanMessage(userMessage)],
  });
  const messages = result.messages as BaseMessage[];
  const last = messages[messages.length - 1];
  if (last === undefined) {
    throw new Error("Wedding agent produced no messages");
  }
  return messageToText(last);
}

// Normalize a message's content to a plain string. LangChain message content is
// `string | MessageContentComplex[]`; for the agent's final text answer we join
// the text parts (ignoring non-text parts) so callers always get a string.
function messageToText(message: BaseMessage): string {
  const content = (message as AIMessage).content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      // Non-text terminal content parts collapse to "" by design: a react-agent
      // final answer is text, so any non-text part is intentionally dropped.
      .map((part) =>
        typeof part === "string"
          ? part
          : part.type === "text"
            ? part.text
            : ""
      )
      .join("");
  }
  return "";
}
