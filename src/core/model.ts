import { ChatOpenAI } from "@langchain/openai";
import { config } from "../config.js";

export interface ModelOptions {
  model?: string;
  temperature?: number;
  streaming?: boolean;
}

// Builds a ChatOpenAI client pointed at the corporate LiteLLM endpoint.
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
