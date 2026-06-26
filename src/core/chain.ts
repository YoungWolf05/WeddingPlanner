import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda, type Runnable } from "@langchain/core/runnables";
import type { BaseMessage } from "@langchain/core/messages";
import { createChatModel, type ModelOptions } from "./model.js";
import { weddingPlannerPrompt } from "./prompts.js";

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
