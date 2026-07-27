# Wedding Planner Chatbot — Agent Instructions

LangChain + TypeScript chatbot routed through a **corporate LiteLLM proxy** (not OpenAI directly). Uses `@langchain/openai` `ChatOpenAI` pointed at the LiteLLM base URL.

## Environment Setup

Copy `.env.example` to `.env` before running anything. Required:

- `LITELLM_API_KEY` — mandatory, app hard-fails without it
- `LITELLM_BASE_URL` — mandatory, app hard-fails without it
- `LITELLM_MODEL` — optional, defaults to `claude-sonnet-4-6`

Available models: `claude-opus-4-8`, `claude-sonnet-4-6`, `gpt-5.1-chat`

## Developer Commands

```bash
npm run test:connection          # verify LiteLLM reachability — run first after setup
npm run chain "your message"     # single-turn LCEL chain (Phase 1)
npm run memory                   # multi-turn memory demo (Phase 2)
npm run chat                     # streaming terminal REPL (Phase 3)
npm run typecheck                # type-check without emitting
npm run build                    # compile to dist/
```

No test framework is configured. `test:connection` is the only automated runtime connectivity smoke check; `typecheck` and `build` provide compilation checks.

## Architecture

```
src/
  config.ts          # env loading — single source of truth for baseURL/apiKey/model
  core/
    model.ts         # createChatModel() — application model factory and LiteLLM wiring
    prompts.ts       # WEDDING_PLANNER_SYSTEM_PROMPT + weddingPlannerPrompt (persona: "Aria")
    chain.ts         # Phase 1: createWeddingPlannerChain() (LCEL, stateless)
                     # Phase 2: createConversationalChain() (LangGraph + checkpointer)
    memory.ts        # MemorySaver (in-RAM, resets on exit); sessionConfig() for thread_id
  run-chain.ts       # CLI entrypoint for Phase 1
  run-memory.ts      # CLI entrypoint for Phase 2
  cli.ts             # Phase 3 streaming terminal REPL and conversation controls
  test-connection.ts # smoke test; temporary direct ChatOpenAI exception (Phase 4 debt)
```

## Critical: ESM Import Extensions

This is a NodeNext ESM project (`"type": "module"`). All local imports **must use `.js` extension**, even when importing `.ts` source files:

```ts
import { config } from "../config.js"; // correct
import { config } from "../config"; // wrong — will fail at runtime
```

## Roadmap and phase governance

[`docs/roadmap.md`](docs/roadmap.md) is the phase/status source of truth. The current completed milestone is **Phase 3 — Streaming Terminal Product Prototype**. No phase is active; **Phase 4 — Engineering Baseline and Provider Contract** is the next proposed phase.

Do not activate or complete a phase without user approval. A completion status requires the approved exit criteria and recorded verification evidence; intent or partial implementation is insufficient.

## Conventions

- All application LLM construction goes through `createChatModel()` in `src/core/model.ts` — do not instantiate `ChatOpenAI` elsewhere. The direct construction in `src/test-connection.ts` is a temporary Phase 4 cleanup exception, not a pattern for future application code.
- System prompt / persona changes belong in `src/core/prompts.ts`.
- TypeScript strict mode is on. Do not disable strict checks or add `any` casts.
