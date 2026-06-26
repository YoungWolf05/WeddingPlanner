import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda, type Runnable } from "@langchain/core/runnables";
import { SystemMessage, type BaseMessage } from "@langchain/core/messages";
import {
  START,
  END,
  StateGraph,
  MessagesAnnotation,
} from "@langchain/langgraph";
import { createChatModel, type ModelOptions } from "./model.js";
import { weddingPlannerPrompt, WEDDING_PLANNER_SYSTEM_PROMPT } from "./prompts.js";
import { checkpointer } from "./memory.js";

/** Input shape accepted by the wedding planner chain. */
export interface ChainInput {
  /** The user's current message. */
  input: string;
  /** Prior conversation messages. Defaults to empty (no memory yet). */
  history?: BaseMessage[];
}

/**
 * Builds the core LCEL chain for the wedding planner:
 *
 *   prompt -> model -> string output parser
 *
 * The result is a Runnable that takes { input, history } and returns a string.
 * This is the backbone that later phases (memory, RAG, tools) build on.
 */
export function createWeddingPlannerChain(
  options: ModelOptions = {}
): Runnable<ChainInput, string> {
  const model = createChatModel(options);

  // Normalize the input so `history` is always present for the prompt's
  // MessagesPlaceholder, defaulting to an empty list (no memory yet).
  const normalizeInput = RunnableLambda.from(
    (input: ChainInput): { input: string; history: BaseMessage[] } => ({
      input: input.input,
      history: input.history ?? [],
    })
  );

  return normalizeInput
    .pipe(weddingPlannerPrompt)
    .pipe(model)
    .pipe(new StringOutputParser());
}

/**
 * Builds a conversational wedding planner graph with multi-turn memory.
 *
 * Uses a LangGraph StateGraph with a checkpointer — the modern replacement for
 * the deprecated RunnableWithMessageHistory. The graph holds the running list
 * of messages; the checkpointer persists that state per session (thread_id).
 *
 * Invoke it with new messages and a session config, e.g.:
 *
 *   import { sessionConfig } from "./memory.js";
 *   const graph = createConversationalChain();
 *   const result = await graph.invoke(
 *     { messages: [new HumanMessage("What's my budget?")] },
 *     sessionConfig("user-123")
 *   );
 *   const reply = result.messages.at(-1)?.content;
 *
 * The system persona is injected at call time so it is never persisted as part
 * of the stored history (keeping the saved state clean).
 */
export function createConversationalChain(options: ModelOptions = {}) {
  const model = createChatModel(options);

  // Single node: prepend the system prompt, call the model, return its reply.
  // The returned message is appended to state.messages by MessagesAnnotation.
  async function callModel(state: typeof MessagesAnnotation.State) {
    const messages = [
      new SystemMessage(WEDDING_PLANNER_SYSTEM_PROMPT),
      ...state.messages,
    ];
    const response = await model.invoke(messages);
    return { messages: [response] };
  }

  const workflow = new StateGraph(MessagesAnnotation)
    .addNode("model", callModel)
    .addEdge(START, "model")
    .addEdge("model", END);

  return workflow.compile({ checkpointer });
}

/** The compiled conversational graph type. */
export type ConversationalChain = ReturnType<typeof createConversationalChain>;
