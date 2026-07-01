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
npm run typecheck                # type-check without emitting
npm run build                    # compile to dist/
```

No test framework is configured. `test:connection` is the only automated check.

## Architecture

```
src/
  config.ts          # env loading — single source of truth for baseURL/apiKey/model
  core/
    model.ts         # createChatModel() — all LiteLLM wiring lives here
    prompts.ts       # WEDDING_PLANNER_SYSTEM_PROMPT + weddingPlannerPrompt (persona: "Aria")
    chain.ts         # Phase 1: createWeddingPlannerChain() (LCEL, stateless)
                     # Phase 2: createConversationalChain() (LangGraph + checkpointer)
    memory.ts        # MemorySaver (in-RAM, resets on exit); sessionConfig() for thread_id
  run-chain.ts       # CLI entrypoint for Phase 1
  run-memory.ts      # CLI entrypoint for Phase 2
  test-connection.ts # connectivity smoke test
```

## Critical: ESM Import Extensions

This is a NodeNext ESM project (`"type": "module"`). All local imports **must use `.js` extension**, even when importing `.ts` source files:

```ts
import { config } from "../config.js";   // correct
import { config } from "../config";      // wrong — will fail at runtime
```

## Phased Design

Code comments reference phases (Phase 1, Phase 2, … Phase 8). Each phase builds on the previous:
- **Phase 1** — stateless LCEL chain
- **Phase 2** — multi-turn LangGraph graph with `MemorySaver`
- **Phase 8** — swap `MemorySaver` in `src/core/memory.ts` for `SqliteSaver` or Redis (interface is stable, only that file changes)

Preserve this pattern when adding phases.

## Conventions

- All LLM construction goes through `createChatModel()` in `src/core/model.ts` — do not instantiate `ChatOpenAI` elsewhere.
- System prompt / persona changes belong in `src/core/prompts.ts`.
- TypeScript strict mode is on. Do not disable strict checks or add `any` casts.
