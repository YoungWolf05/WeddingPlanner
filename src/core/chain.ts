import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda, type Runnable } from "@langchain/core/runnables";
import { SystemMessage, type BaseMessage } from "@langchain/core/messages";
import {
  START,
  END,
  StateGraph,
  MessagesAnnotation,
  type BaseCheckpointSaver,
} from "@langchain/langgraph";
import { createChatModel, type ModelOptions } from "./model.js";
import { weddingPlannerPrompt, WEDDING_PLANNER_SYSTEM_PROMPT } from "./prompts.js";
import { getCheckpointer } from "./memory.js";

export interface ChainInput {
  input: string;
  history?: BaseMessage[];
}

// Phase 1: core LCEL chain — prompt -> model -> string output parser.
export function createWeddingPlannerChain(
  options: ModelOptions = {}
): Runnable<ChainInput, string> {
  const model = createChatModel(options);

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

// Phase 2: conversational graph with multi-turn memory via a LangGraph
// checkpointer (keyed by session/thread_id). System persona injected per call.
//
// Phase 5 (5a): the checkpointer is now durable (SQLite). By default the graph
// uses the shared checkpointer (getCheckpointer, constructed lazily); an explicit
// `saver` may be passed to bind an isolated checkpointer (e.g. a temp-file SQLite
// instance in tests) without changing default application behavior.
export function createConversationalChain(
  options: ModelOptions = {},
  saver?: BaseCheckpointSaver
) {
  const model = createChatModel(options);
  // Default to the shared durable checkpointer, constructed lazily on first use.
  const checkpointer = saver ?? getCheckpointer();

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

export type ConversationalChain = ReturnType<typeof createConversationalChain>;
