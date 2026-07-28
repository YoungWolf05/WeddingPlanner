# Wedding Planner Product Roadmap

## Status

| Field | Value |
| --- | --- |
| Last reviewed | 2026-07-28 |
| Current completed milestone | **Phase 4 — Engineering Baseline and Provider Contract** |
| Active phase | **None** |
| Next proposed phase | **Phase 5 — Durable Conversation Service** |

Phase 4 is complete. Phase 5 is proposed and not active until the user approves its activation.

## Status vocabulary

- **Planned** — recorded future scope that has not been activated; sequencing and details may still require approval.
- **Active** — explicitly activated by the user; implementation or verification is in progress.
- **Complete** — approved exit criteria are met and verification evidence is recorded.
- **Complete (prototype)** — the prototype exit criteria are met; explicitly recorded hardening debt remains and must not be mistaken for production readiness.
- **Blocked** — an active phase cannot proceed; the blocker and required decision or dependency must be recorded.

## Governance

- This document is the source of truth for phase scope and status.
- Activating or completing a phase requires user approval. Intent, partial implementation, or a proposed next phase does not change status.
- Completion requires all approved exit criteria and recorded verification evidence.
- The engineering manager reads this roadmap when scoping work but remains read-only. Roadmap changes are delegated to `feature-implementer` and reviewed by `code-reviewer`.
- Architecture decisions belong in separate architecture decision records (ADRs), not in roadmap scratch notes.

## Roadmap

### Phase 1 — Stateless Chain Foundation

**Status:** Complete

**Goal:** Establish a single-turn wedding-planning assistant through the corporate LiteLLM proxy.

**Key deliverables**

- LCEL prompt → model → string parser chain with the Aria system persona.
- Central model factory and environment-backed LiteLLM configuration.
- Single-message command-line entrypoint.

**Exit criteria**

- [x] A user message can be passed through the stateless chain.
- [x] Local model construction for the chain uses `createChatModel()`.
- [x] The Phase 1 entrypoint is exposed as `npm run chain "<message>"`.

**Implementation evidence and notes**

- `src/core/chain.ts` implements `createWeddingPlannerChain()`; `src/core/model.ts` and `src/config.ts` supply LiteLLM model configuration.
- `src/core/prompts.ts` defines the Aria persona, and `src/run-chain.ts` is mapped to the `chain` script in `package.json`.
- The command identifies the implemented runtime path; it is not an automated regression test and requires live LiteLLM access.

### Phase 2 — Stateful Conversation Prototype

**Status:** Complete (prototype)

**Goal:** Demonstrate multi-turn conversation state keyed by a LangGraph thread identifier.

**Key deliverables**

- LangGraph conversation graph that reuses the Phase 1 model factory and persona.
- Process-local `MemorySaver` checkpointer and `thread_id` session configuration.
- Multi-turn recall demonstration.

**Exit criteria**

- [x] Multiple turns on one `thread_id` share conversation history.
- [x] The graph compiles with a checkpointer and invokes the configured chat model.
- [x] The demonstration is exposed as `npm run memory`.

**Implementation evidence and notes**

- `src/core/chain.ts` implements `createConversationalChain()`; `src/core/memory.ts` provides `MemorySaver` and `sessionConfig()`.
- `src/run-memory.ts` is mapped to the `memory` script in `package.json` and demonstrates recall on one thread.
- State is volatile and process-local. Durable storage, ownership, retention, concurrency, and migration behavior are not implemented; they are Phase 5 scope.
- The demo command is not automated verification and requires live LiteLLM access.

### Phase 3 — Streaming Terminal Product Prototype

**Status:** Complete (prototype)

**Goal:** Provide an interactive terminal experience with streaming responses and basic conversation controls.

**Key deliverables**

- Streaming terminal REPL backed by the Phase 2 graph.
- `/new`, `/model <name>`, and `/exit` controls.
- Per-turn cancellation, recoverable turn errors, and model allow-list handling.

**Exit criteria**

- [x] The terminal accepts repeated user turns and streams model output.
- [x] Users can start a new thread, switch an allowed model, exit, and interrupt an in-flight turn.
- [x] The REPL is exposed as `npm run chat`.

**Implementation evidence and notes**

- `src/cli.ts` implements the REPL, streaming, commands, model switching, and interruption behavior; `package.json` maps it to `npm run chat`.
- Automated REPL coverage does not exist and is explicitly Phase 4 hardening debt. The chat command is a live manual path, not an automated test.

**Completed-phase verification recorded on 2026-07-27**

- Repository source at baseline commit `f2f9678` and `package.json` scripts were inspected against the Phase 1–3 criteria.
- `npm run typecheck` and `npm run build` passed during this roadmap review.
- No live LiteLLM command was run. The repository has no test framework or lint script, so this evidence proves compilation and implementation presence, not automated end-to-end behavior.

### Phase 4 — Engineering Baseline and Provider Contract

**Status:** Complete

**Goal:** Create a dependable engineering baseline and measure the real LiteLLM/provider contract before building provider-dependent features.

**Increment plan**

Phase 4 is delivered as ordered, independently verifiable sub-increments. One sub-increment is worked at a time; each must be independently verifiable before the next begins.

- **4a — Test framework and Phase 1–3 regression coverage.** Status: Complete. Introduce Vitest and deterministic, mocked-model, offline tests covering the Phase 1 chain, Phase 2 thread memory, and Phase 3 REPL controls/streaming/cancellation. Tests must run in CI without live proxy calls. Delivered: Vitest with deterministic offline Phase 1–3 regression tests (32 tests); reviewed.
- **4b — `createChatModel()` centralization.** Status: Complete. Route all chat-model construction through the model factory, including cleanup of the direct `ChatOpenAI` construction in `src/test-connection.ts`. Delivered: `src/test-connection.ts` routed through `createChatModel()`; offline guard test (`test/phase4-model-factory.test.ts`) enforces single-factory `ChatOpenAI` construction; 34 offline tests; `npm run typecheck` and `npm run build` pass; reviewed.
- **4c — LiteLLM capability matrix and embedding assessment.** Status: Complete. Verify the chat-model capability matrix for each supported chat alias and separately assess embedding aliases/endpoints. Exposed as an explicit opt-in live command; not run in default CI. Delivered: opt-in `npm run test:capabilities` live probe plus a dated capability matrix ([`docs/capabilities/2026-07-27.md`](capabilities/2026-07-27.md)); single-factory guards extended so only `src/core/model.ts` constructs `ChatOpenAI` and only `src/core/embeddings.ts` constructs `OpenAIEmbeddings`; host-only secret redaction with fully masked key; 55 offline tests; `npm run typecheck` and `npm run build` pass; reviewed. See "4c findings and open items" below for caveats.
- **4c.1 — Alias reconciliation.** Status: Complete. Remove the invalid `gpt-5.1-chat` alias confirmed rejected by the key/proxy and reconcile the documented alias list. Delivered: `gpt-5.1-chat` removed from the allow-list (`src/core/repl.ts`), `AGENTS.md`, `.env.example`, and the probe alias set, with tests locking its rejection and pinning `ALLOWED_MODELS` to `["claude-opus-4-8", "claude-sonnet-4-6"]`; 57 offline tests; `npm run typecheck` and `npm run build` pass; reviewed.
- **4d — Local tracing with redaction.** Status: Complete. Add local, self-contained tracing with secret/PII redaction and documented observability boundaries. No external SaaS tracing backend. Delivered: opt-in local tracing integrated at the `createChatModel()` boundary via a LangChain callback handler (off by default; enabled with `LITELLM_TRACE=1`, `TRACE=1` alias, `LITELLM_TRACE` authoritative including `=0` disabling even when `TRACE=1`); metadata-only by default (timestamp, model alias, operation, latency, usage when present, outcome, streaming flag, redacted error reason) with optional content capture off by default and redacted when enabled; pluggable sinks (default JSONL to gitignored `logs/`, optional stderr console sink) whose failures are swallowed and never crash a run; always-on, non-disableable redaction scrubs the LiteLLM `apiKey`/`baseURL` plus basic PII (email/phone) with cap-after-scrub so truncation cannot leak a secret; shared redaction extracted to `src/core/redaction.ts` and reused by the probe (which now additionally redacts PII and redacts its top-level stderr path). Trace output goes to gitignored `logs/` only — no secrets/PII, local only. 110 offline tests; `npm run typecheck` and `npm run build` pass; reviewed (Ready, no blocking/recommended findings).
- **4e — Evaluation dataset, documentation alignment, and closeout.** Status: Complete. Add a small versioned wedding-planning evaluation dataset (~10–15 prompts) with a mostly deterministic baseline, align roadmap/developer/environment/operational documentation (including `AGENTS.md` architecture/tracing alignment), and complete Phase 4 closeout. Delivered: a temperature-omit factory option (`createChatModel({ temperature: null })`); a definitive re-probe resolving `claude-opus-4-8` structured output to `Supported` when temperature is omitted (previously undetermined because the factory-injected temperature tripped a model deprecation on the structured path); `gemini-embedding-001` embeddings verified `Supported` at 768 dimensions ([`docs/capabilities/2026-07-28.md`](capabilities/2026-07-28.md)); a versioned eval dataset at `evals/dataset.jsonl` with pure offline property scorers (`src/core/eval.ts`); an opt-in `npm run eval` live runner and a dated baseline ([`docs/eval/2026-07-28.md`](eval/2026-07-28.md), claude-sonnet-4-6, 12/12 items, 100%); documentation alignment and evidence signposts; 149 offline tests; `npm run typecheck` and `npm run build` pass; reviewed.

**Approved decisions (rationale for future readers)**

- **Test framework:** Vitest.
- **Test strategy:** Deterministic, mocked-model, offline unit/regression tests that run in CI without live proxy calls.
- **Capability checks:** Live LiteLLM capability checks are an explicit opt-in command, not part of default CI.
- **Tracing:** Local, self-contained tracing with secret/PII redaction first; no external SaaS backend.
- **Evaluation set:** A small committed dataset (~10–15 prompts) with a mostly deterministic baseline.
- **Delivery:** Incremental, one sub-increment at a time (4a → 4e), each independently verifiable.

**Key deliverables**

- Test framework and automated Phase 1–3 regression coverage, including the streaming REPL.
- Full `createChatModel()` centralization, including cleanup of direct model construction in `src/test-connection.ts`.
- A chat-model capability matrix verified against the real LiteLLM proxy for each supported chat alias: invoke, stream, abort, usage metadata, tool calls, and structured output.
- A separate embedding alias/endpoint assessment through the proxy. Embeddings are evaluated against embedding aliases, not required of chat aliases.
- Explicit `Supported`, `Unsupported`, `Degraded`, or `N/A` results for every assessed capability.
- An embedding-readiness gate: Phase 7 and Phase 8 cannot activate until an approved embedding alias and endpoint have a dated `Supported` verification.
- Tracing with secret/PII redaction and documented observability boundaries.
- A small, versioned wedding-planning evaluation dataset and repeatable baseline.
- Alignment of roadmap, developer commands, environment guidance, and operational documentation.

**Exit criteria**

- [x] Tests deterministically cover the Phase 1 chain, Phase 2 thread memory, and Phase 3 REPL controls/streaming/cancellation without requiring uncontrolled live calls. (4a; offline Vitest suite, 149 tests.)
- [x] All chat model clients, including connectivity checks, are created through the model factory. (4b; enforced by the single-factory guard test `test/phase4-model-factory.test.ts`.)
- [x] The chat-model matrix records dated `Supported`, `Unsupported`, `Degraded`, or `N/A` results for every supported chat alias and assessed chat capability. (4c/4e; [`docs/capabilities/2026-07-28.md`](capabilities/2026-07-28.md) records dated states for `claude-opus-4-8` and `claude-sonnet-4-6` — note `claude-opus-4-8` structured output is `Supported` only when temperature is omitted.)
- [x] The separate embedding assessment records the proxy endpoint, candidate embedding aliases, and a dated state for each assessment. (4e; `gemini-embedding-001` = `Supported` at 768 dimensions in [`docs/capabilities/2026-07-28.md`](capabilities/2026-07-28.md) — the embedding-readiness gate is now MET.)
- [x] Traces demonstrate useful diagnostics without exposing credentials or designated sensitive data. (4d; opt-in local tracing at the `createChatModel()` boundary with always-on secret/PII redaction, JSONL/console sinks, off by default.)
- [x] The evaluation baseline runs repeatably and documentation matches actual commands and behavior. (4e; `npm run eval` over `evals/dataset.jsonl` with a dated baseline at [`docs/eval/2026-07-28.md`](eval/2026-07-28.md).)

**Implementation evidence and notes**

- Model construction is centralized (4b): `src/config.ts` and `src/core/model.ts` centralize application model setup, and `src/test-connection.ts` now constructs its model via `createChatModel()`; an offline guard test (`test/phase4-model-factory.test.ts`) enforces single-factory `ChatOpenAI` construction.
- `src/cli.ts` supports model switching, but this is not a verified provider capability contract.
- Chat-model and embedding assessments are separate contracts. Chat aliases are not required or evaluated as embedding aliases; embedding results attach only to candidate embedding alias/endpoint pairs.
- Vitest is in place with deterministic offline Phase 1–3 regression coverage (4a); the full suite is 110 offline tests. `npm run test:connection` remains the only automated runtime connectivity smoke check and makes a live request.
- Local tracing (4d) is delivered: `src/core/tracing.ts` wires an opt-in LangChain callback handler at the `createChatModel()` boundary (off by default; `LITELLM_TRACE` authoritative, `TRACE` alias), emitting metadata-only records by default with optional redacted content capture, to pluggable sinks (default JSONL to gitignored `logs/`, optional stderr) whose failures are swallowed. Redaction is shared in `src/core/redaction.ts` (always-on secret + PII scrubbing with cap-after-scrub) and reused by `src/probe-capabilities.ts`. Trace output stays local in gitignored `logs/` and carries no secrets/PII.
- Capability assessment (4c) is delivered: `src/core/capabilities.ts` provides pure capability rendering and `classifyAbortOutcome` logic, `src/core/embeddings.ts` is the embeddings factory, and `src/probe-capabilities.ts` backs the opt-in `npm run test:capabilities` live probe (not in `npm test`/CI). Offline coverage adds capability rendering and abort-classification tests. The architecture guard is extended so only `src/core/model.ts` constructs `ChatOpenAI` and only `src/core/embeddings.ts` constructs `OpenAIEmbeddings`. A dated matrix is recorded at [`docs/capabilities/2026-07-27.md`](capabilities/2026-07-27.md).

**4c findings and open items (resolved at Phase 4 closeout)**

- **`claude-opus-4-8` structured output — RESOLVED (`Supported` with temperature omitted).** A definitive re-probe via a temperature-omitting path confirmed structured output is `Supported` for `claude-opus-4-8` (dated evidence: [`docs/capabilities/2026-07-28.md`](capabilities/2026-07-28.md)). The earlier `Error` was caused by the factory-injected `temperature` tripping a model deprecation on the structured path, not a feature rejection. `claude-sonnet-4-6` remains fully `Supported` across invoke, streaming, abort, usage, tool-calling, and structured output. **Carry-forward constraint for Phase 6:** application structured-output calls to `claude-opus-4-8` must omit temperature (construct the model via `createChatModel({ temperature: null })`) or the model errors on its structured-output path.
- **`gpt-5.1-chat` reconciliation — RESOLVED (4c.1).** The alias was confirmed invalid on this key/proxy ("Invalid model name") and removed from the allow-list (`src/core/repl.ts`), documentation (`AGENTS.md`), `.env.example`, and the probe alias set, with tests locking its rejection (`isAllowedModel("gpt-5.1-chat") === false`; `ALLOWED_MODELS` is exactly `["claude-opus-4-8", "claude-sonnet-4-6"]`). The historical matrix at [`docs/capabilities/2026-07-27.md`](capabilities/2026-07-27.md) preserves the `Error` result from when three aliases were probed.
- **Embedding-readiness gate — RESOLVED / MET.** `gemini-embedding-001` is verified `Supported` at 768 dimensions (dated evidence: [`docs/capabilities/2026-07-28.md`](capabilities/2026-07-28.md)). The gate no longer blocks Phase 7 or Phase 8 activation. For those phases, `LITELLM_EMBED_MODEL` should be set to a verified embedding alias.

### Phase 5 — Durable Conversation Service

**Status:** Planned

**Goal:** Move volatile conversation state behind a secure, durable service boundary.

**Key deliverables**

- Persistent LangGraph checkpointer with migration and operational strategy.
- Server-issued UUID thread identifiers, authenticated ownership/access enforcement, and retention/deletion behavior.
- An API or Agent Server boundary with a versioned, typed event stream.
- Restart, concurrency, isolation, deletion, and migration validation.

**Exit criteria**

- [ ] Authorized users can resume their own threads after process restart, while cross-user access is denied.
- [ ] Thread creation uses server-issued UUIDs; authentication context is not derived from or conflated with `thread_id`.
- [ ] Retention/deletion and schema migration behavior are implemented and tested.
- [ ] Concurrent turns and service restarts preserve documented consistency guarantees.
- [ ] The typed service and event contracts are versioned and validated independently of a UI.

**Implementation evidence and notes**

- The current `MemorySaver` in `src/core/memory.ts` is process-local and does not satisfy this phase.
- `src/core/memory.ts` has a stale legacy comment assigning persistence to Phase 8. This roadmap's Phase 5 supersedes that comment; correct it as Phase 5 implementation housekeeping. Recording this debt does not activate Phase 5.

### Phase 6 — Structured Domain Data and Safe Tools

**Status:** Planned

**Goal:** Add validated wedding-domain data and deterministic tools with explicit safety boundaries.

**Key deliverables**

- Zod schemas for approved domain inputs, outputs, and tool contracts.
- Provider/tool-based structured output selected from Phase 4 capability evidence.
- Deterministic budget and date/timeline tools.
- `createAgent` only when a real model-driven tool loop exists.
- Tool permissions, input validation, timeouts, bounded errors, and auditable execution state.
- Model-specific capability and contract tests.

**Exit criteria**

- [ ] Structured outputs and tool inputs are runtime-validated, with safe handling of refusal, malformed data, and provider errors.
- [ ] Budget/date tools are deterministic and have boundary, timeout, and failure tests.
- [ ] Only permitted tools execute, and tool state/errors are represented in the typed event contract.
- [ ] Every enabled model alias passes the required tool-call and structured-output contract tests.

**Implementation evidence and notes**

- No structured LLM output or tool/agent execution is currently implemented.

### Phase 7 — Knowledge Ingestion and Retrieval

**Status:** Planned

**Goal:** Build a durable, testable retrieval foundation before generating citation-bearing answers.

**Key deliverables**

- Idempotent ingestion with stable document/chunk IDs and trusted metadata.
- Persistent vector store and an embedding alias verified through the LiteLLM proxy.
- Explicit add, update, and delete semantics without duplicate chunks.
- Retrieval-only evaluation dataset and metrics.

**Exit criteria**

- [ ] Re-ingesting unchanged content produces no duplicate documents or chunks.
- [ ] Updates and deletes produce deterministic index state while preserving documented identity rules.
- [ ] Metadata can support later authorization and trusted citations without relying on model-generated identifiers.
- [ ] The embedding alias and vector dimensions are verified through the proxy and covered by compatibility checks.
- [ ] Retrieval quality meets an approved baseline before answer-generation tuning begins.

**Implementation evidence and notes**

- No ingestion pipeline, embedding integration, retriever, or persistent vector store currently exists.

### Phase 8 — Grounded Answers and Trusted Citations

**Status:** Planned

**Goal:** Generate answers grounded in retrieved evidence with citations the application can trust.

**Key deliverables**

- Deterministic two-step retrieval-then-generation RAG as the initial design.
- Trusted citation objects tied by application code to retrieved document/chunk IDs.
- Explicit insufficient-evidence behavior.
- Groundedness, citation correctness, and prompt-injection evaluations.
- Agentic retrieval only after a measured need demonstrates an advantage over two-step RAG.

**Exit criteria**

- [ ] Citation objects resolve to retrieved, authorized source IDs and are never accepted solely from model text.
- [ ] Answers distinguish supported claims from insufficient evidence.
- [ ] Evaluation covers groundedness, citation precision/recall, malicious source instructions, and missing evidence.
- [ ] Any move to agentic retrieval is supported by comparative measurements and a separate ADR.

**Implementation evidence and notes**

- RAG and citation behavior are not currently implemented.

### Phase 9 — Web Interface

**Status:** Planned

**Goal:** Deliver a secure browser experience over the stable service and event contracts.

**Key deliverables**

- Authenticated server-side credential handling; LiteLLM credentials never reach the browser.
- Stable typed Server-Sent Events (SSE) or equivalent event protocol.
- Thread creation/resumption and streaming, cancellation, retry, and reconnect behavior.
- Citation rendering, tool state, and structured artifacts.
- Browser end-to-end coverage, including unauthorized access attempts.

**Exit criteria**

- [ ] The browser contains no provider credentials and all thread operations enforce authenticated ownership.
- [ ] Streaming/cancel/retry/reconnect behavior conforms to the versioned event contract.
- [ ] Citations, tool progress/errors, and artifacts render from typed trusted events.
- [ ] Browser end-to-end tests cover primary journeys, recovery paths, and unauthorized access.

**Implementation evidence and notes**

- No web UI or browser-facing service currently exists.

### Phase 10 — Production Hardening and Continuous Evaluation

**Status:** Planned

**Goal:** Establish production reliability, privacy, scalability, and continuous quality controls.

**Key deliverables**

- Memory trimming/summarization with regression evaluation.
- LiteLLM aliases, routing, and fallback policy, owned by LiteLLM where possible.
- Rate limits, abuse controls, retention/privacy enforcement, and user feedback capture.
- Offline and online evaluation loops.
- Load, concurrency, dependency-failure, cost, and latency validation with operational thresholds.

**Exit criteria**

- [ ] Memory controls remain within approved context/cost budgets without unacceptable quality loss.
- [ ] Routing/fallback behavior and failure modes are tested without duplicating provider routing in application code.
- [ ] Rate limits, privacy, retention, deletion, and feedback controls are auditable and exercised.
- [ ] Offline/online evaluations and production alerts have owners, thresholds, and response procedures.
- [ ] Load and failure tests meet approved reliability, cost, and latency targets.

**Implementation evidence and notes**

- These production controls and evaluations are not currently implemented.

## Dependencies and ordering principles

1. Verify proxy and chat-model capabilities before depending on tools, schemas, or structured outputs; assess embeddings separately against embedding aliases/endpoints.
2. Design trusted document identity and metadata before exposing citations.
3. Establish retrieval-only evaluation before tuning answer generation.
4. Establish durable persistence and thread ownership before building the web interface.
5. Stabilize and version the typed event contract before frontend integration.
6. Keep authentication/authorization context separate from `thread_id`; a thread identifier is not proof of identity or access.
7. Let LiteLLM own provider aliases, routing, and fallback where possible; avoid duplicating that policy in application code.
8. Do not activate Phase 7 or Phase 8 until an approved embedding alias and endpoint have been verified through the proxy. This gate is now MET: `gemini-embedding-001` is verified `Supported` at 768 dimensions ([`docs/capabilities/2026-07-28.md`](capabilities/2026-07-28.md)); set `LITELLM_EMBED_MODEL` to a verified embedding alias for those phases.

## Phase-transition record

Add one concise row only after approval; do not use this table as a session log.

| Date | Phase | Approved transition | Approval reference | Exit criteria and verification evidence |
| --- | --- | --- | --- | --- |
| 2026-07-28 | Phase 4 | Active → Complete; Phase 5 becomes next proposed phase | User-approved Phase 4 closeout | All Phase 4 exit criteria ticked above; [`docs/capabilities/2026-07-28.md`](capabilities/2026-07-28.md), [`docs/eval/2026-07-28.md`](eval/2026-07-28.md); 149 offline tests, `npm run typecheck` and `npm run build` pass |
