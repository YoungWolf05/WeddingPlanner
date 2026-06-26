import { ChatOpenAI } from "@langchain/openai";
import { config } from "../config.js";

/**
 * Options for building a chat model. All fields are optional and fall back to
 * the values loaded from the environment via `config`.
 */
export interface ModelOptions {
  /** Model name on the LiteLLM endpoint (e.g. "claude-sonnet-4-6"). */
  model?: string;
  /** Sampling temperature. Lower = more deterministic. */
  temperature?: number;
  /** Stream tokens as they are generated. */
  streaming?: boolean;
}

/**
 * Factory that builds a ChatOpenAI client configured to talk to the corporate
 * LiteLLM endpoint. LiteLLM is OpenAI-compatible, so we just override the base
 * URL on the underlying OpenAI client.
 *
 * Centralizing this here means every chain, tool, and agent in the app shares
 * one consistent model configuration.
 */
export function createChatModel(options: ModelOptions = {}): ChatOpenAI {
  return new ChatOpenAI({
    model: options.model ?? config.model,
    apiKey: config.apiKey,
    temperature: options.temperature ?? 0.7,
    streaming: options.streaming ?? false,
    configuration: {
      baseURL: config.baseURL,
    },
  });
}
