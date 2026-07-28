# Wedding Planner Chatbot — Agent Instructions

LangChain + TypeScript chatbot routed through a **corporate LiteLLM proxy** (not OpenAI directly). Uses `@langchain/openai` `ChatOpenAI` pointed at the LiteLLM base URL.

## Environment Setup

Copy `.env.example` to `.env` before running anything. Required:

- `LITELLM_API_KEY` — mandatory, app hard-fails without it
- `LITELLM_BASE_URL` — mandatory, app hard-fails without it
- `LITELLM_MODEL` — optional, defaults to `claude-sonnet-4-6`

Available models: `claude-opus-4-8`, `claude-sonnet-4-6`

## Developer Commands

```bash
npm test                         # OFFLINE Vitest suite — deterministic, mocked model, no network
npm run chain "your message"     # single-turn LCEL chain (Phase 1)
npm run memory                   # multi-turn memory demo (Phase 2)
npm run chat                     # streaming terminal REPL (Phase 3)
npm run typecheck                # type-check without emitting (src + vitest configs)
npm run build                    # compile to dist/

# LIVE / opt-in — make real, credentialed LiteLLM calls. NOT part of npm test or CI:
npm run test:connection          # verify LiteLLM reachability — run first after setup
npm run test:capabilities        # probe the chat/embedding capability matrix -> docs/capabilities/<date>.md
npm run eval                     # run the wedding-planning eval dataset -> docs/eval/<date>.md
```

`npm test` is fully OFFLINE and CI-safe: the model boundary (`createChatModel`)
is mocked per test, so no credentials or network are used. `test:connection`,
`test:capabilities`, and `eval` are LIVE, opt-in commands that call the real
proxy and are never run by `npm test` or CI. `typecheck` and `build` provide
compilation checks.

## Architecture

```
src/
  config.ts            # env loading — single source of truth for baseURL/apiKey/model/embedModel
  core/
    model.ts           # createChatModel() — the ONLY ChatOpenAI construction site; LiteLLM wiring + tracing hook
    embeddings.ts      # createEmbeddingsModel() — the ONLY OpenAIEmbeddings construction site (separate contract)
    prompts.ts         # WEDDING_PLANNER_SYSTEM_PROMPT + weddingPlannerPrompt (persona: "Aria")
    chain.ts           # Phase 1: createWeddingPlannerChain() (LCEL, stateless)
                       # Phase 2: createConversationalChain() (LangGraph + checkpointer)
    memory.ts          # MemorySaver (in-RAM, resets on exit); sessionConfig() for thread_id
    repl.ts            # Phase 3 REPL pure logic (ALLOWED_MODELS, command parsing)
    redaction.ts       # shared, always-on secret/PII scrubbing (pure); reused by probe + tracing + eval
    tracing.ts         # Phase 4 (4d) local tracing at the createChatModel() boundary (opt-in, redacted)
    capabilities.ts    # Phase 4 (4c) pure capability-matrix rendering + abort classification
    eval.ts            # Phase 4 (4e) pure eval dataset parsing, property scorers, aggregation + rendering
  run-chain.ts         # CLI entrypoint for Phase 1
  run-memory.ts        # CLI entrypoint for Phase 2
  cli.ts               # Phase 3 streaming terminal REPL and conversation controls
  test-connection.ts   # LIVE smoke test (routed through createChatModel)
  probe-capabilities.ts# LIVE, opt-in capability probe -> docs/capabilities/<date>.md
  run-eval.ts          # LIVE, opt-in eval runner -> docs/eval/<date>.md
evals/
  dataset.jsonl        # versioned wedding-planning eval prompts + deterministic expectations
```

The offline Vitest suite lives under `test/` (outside `src/`, so it is never
emitted to `dist/`). The pure modules above (`redaction.ts`, `capabilities.ts`,
`eval.ts`, `repl.ts`) are I/O-free so the suite exercises them without a live
call; the `probe-capabilities.ts` and `run-eval.ts` scripts own the impure,
live-only I/O and are never imported by tests.

## Critical: ESM Import Extensions

This is a NodeNext ESM project (`"type": "module"`). All local imports **must use `.js` extension**, even when importing `.ts` source files:

```ts
import { config } from "../config.js"; // correct
import { config } from "../config"; // wrong — will fail at runtime
```

## Roadmap and phase governance

[`docs/roadmap.md`](docs/roadmap.md) is the phase/status source of truth. The current completed milestone is **Phase 4 — Engineering Baseline and Provider Contract**. No phase is active; **Phase 5 — Durable Conversation Service** is the next proposed phase.

Do not activate or complete a phase without user approval. A completion status requires the approved exit criteria and recorded verification evidence; intent or partial implementation is insufficient.

## Conventions

- All application chat-model construction goes through `createChatModel()` in `src/core/model.ts` — do not instantiate `ChatOpenAI` elsewhere (including `src/test-connection.ts`, which was routed through the factory in 4b). Embeddings construction goes through `createEmbeddingsModel()` in `src/core/embeddings.ts` — do not instantiate `OpenAIEmbeddings` elsewhere. An offline guard test (`test/phase4-model-factory.test.ts`) enforces both single-factory rules.
- System prompt / persona changes belong in `src/core/prompts.ts`.
- Evaluation dataset changes belong in `evals/dataset.jsonl`; property scorers live in `src/core/eval.ts`. Keep prompts benign and PII-free; keep expectations property-based (checkable without an LLM judge).
- Any string that may reach a log, trace, console, or dated evidence file must pass through `src/core/redaction.ts` first (always-on secret/PII scrubbing).
- TypeScript strict mode is on. Do not disable strict checks or add `any` casts.
