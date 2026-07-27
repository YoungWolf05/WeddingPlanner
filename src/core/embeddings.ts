import { OpenAIEmbeddings } from "@langchain/openai";
import { config } from "../config.js";

export interface EmbeddingsOptions {
  model?: string;
}

// Builds an OpenAIEmbeddings client pointed at the corporate LiteLLM endpoint.
//
// Embeddings are a SEPARATE provider contract from chat (see docs/roadmap.md
// Phase 4): a chat alias is never assumed to work as an embedding alias. This
// factory mirrors the LiteLLM wiring of createChatModel() in model.ts (same
// apiKey and baseURL, sourced from config) but for the embeddings endpoint.
//
// The embedding alias is required here: callers must supply one explicitly or
// have LITELLM_EMBED_MODEL configured. Unlike the chat model, there is no
// sensible default alias, so we fail loudly instead of guessing.
export function createEmbeddingsModel(
  options: EmbeddingsOptions = {}
): OpenAIEmbeddings {
  const model = options.model ?? config.embedModel;
  if (!model) {
    throw new Error(
      "No embedding alias configured. Pass options.model or set " +
        "LITELLM_EMBED_MODEL in your environment."
    );
  }

  return new OpenAIEmbeddings({
    model,
    apiKey: config.apiKey,
    configuration: {
      baseURL: config.baseURL,
    },
  });
}
