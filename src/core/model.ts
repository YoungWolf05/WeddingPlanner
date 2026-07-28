import { ChatOpenAI } from "@langchain/openai";
import { config } from "../config.js";
import { createTracingHandler } from "./tracing.js";

export interface ModelOptions {
  model?: string;
  temperature?: number;
  streaming?: boolean;
}

// Builds a ChatOpenAI client pointed at the corporate LiteLLM endpoint.
//
// Phase 4 (increment 4d): local tracing is integrated HERE so ALL application
// model construction is traced consistently. Tracing is OFF by default; when
// enabled (LITELLM_TRACE=1 / TRACE=1) a LangChain callback handler is attached
// to the single ChatOpenAI instance via its `callbacks` option. We deliberately
// use a callback handler rather than wrapping/proxying the Runnable so:
//   - the return type stays ChatOpenAI (callers rely on bindTools /
//     withStructuredOutput / pipe / stream / invoke),
//   - invoke/stream/abort semantics are unchanged,
//   - no SECOND provider client is constructed (the 4b single-factory guard
//     holds — the handler constructs no ChatOpenAI).
// When tracing is disabled, createTracingHandler() returns undefined and no
// handler is attached, so behavior is byte-for-byte unchanged.
export function createChatModel(options: ModelOptions = {}): ChatOpenAI {
  const model = options.model ?? config.model;
  const streaming = options.streaming ?? false;
  const handler = createTracingHandler({ model, streaming });
  return new ChatOpenAI({
    model,
    apiKey: config.apiKey,
    temperature: options.temperature ?? 0.7,
    streaming,
    configuration: {
      baseURL: config.baseURL,
    },
    ...(handler ? { callbacks: [handler] } : {}),
  });
}
