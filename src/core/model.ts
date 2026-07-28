import { ChatOpenAI } from "@langchain/openai";
import { config } from "../config.js";
import { createTracingHandler } from "./tracing.js";

export interface ModelOptions {
  model?: string;
  // Temperature handling has THREE distinct cases:
  //   - omitted (undefined) => keep the factory default of 0.7 (unchanged
  //     behavior for every existing caller),
  //   - a number           => use that exact value,
  //   - null               => OMIT the temperature field entirely from the
  //     ChatOpenAI constructor payload, so the provider/model applies its own
  //     default. This exists for models (e.g. claude-opus-4-8 on its
  //     structured-output path) that DEPRECATE an explicit temperature and
  //     error when one is sent. Passing null is the opt-in "send no
  //     temperature" mode.
  temperature?: number | null;
  streaming?: boolean;
}

// The exact plain params object handed to `new ChatOpenAI(...)`. `temperature`
// is intentionally optional here: when omission is requested the key is ABSENT
// (not `undefined`), which is what makes the omit behavior unambiguous and
// directly assertable in tests.
export interface ChatModelParams {
  model: string;
  apiKey: string;
  temperature?: number;
  streaming: boolean;
  configuration: { baseURL: string };
}

// Pure builder for the ChatOpenAI constructor params. Deliberately I/O-free and
// client-free (constructs NO provider client) so it is unit-testable offline and
// keeps the single ChatOpenAI construction site inside createChatModel().
//
// Temperature rule (see ModelOptions.temperature):
//   - undefined => default 0.7,
//   - number    => that value,
//   - null      => the `temperature` key is omitted from the returned object.
export function buildChatModelParams(
  options: ModelOptions,
  deps: { apiKey: string; baseURL: string; model: string; streaming: boolean }
): ChatModelParams {
  const params: ChatModelParams = {
    model: deps.model,
    apiKey: deps.apiKey,
    streaming: deps.streaming,
    configuration: { baseURL: deps.baseURL },
  };
  // Only attach `temperature` when it should actually be sent. `null` means
  // "omit", so we leave the key off entirely in that case.
  if (options.temperature !== null) {
    params.temperature = options.temperature ?? 0.7;
  }
  return params;
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
  const params = buildChatModelParams(options, {
    apiKey: config.apiKey,
    baseURL: config.baseURL,
    model,
    streaming,
  });
  return new ChatOpenAI({
    ...params,
    ...(handler ? { callbacks: [handler] } : {}),
  });
}
