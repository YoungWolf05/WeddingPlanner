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
export function createConversationalChain(options: ModelOptions = {}) {
  const model = createChatModel(options);

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
