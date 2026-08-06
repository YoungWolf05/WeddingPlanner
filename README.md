# Wedding Planner Chatbot

See the [canonical product roadmap](docs/roadmap.md) for phase scope, status, and governance.

Latest live evidence: [capability matrices](docs/capabilities/), [tool-call & structured-output contract matrices](docs/contracts/), [embedding & dimension compatibility](docs/embeddings/), [evaluation baselines](docs/eval/), and [retrieval baselines](docs/retrieval/) (each folder's `README.md` links the most recent dated snapshot).

Phase 6 adds validated wedding-domain structured output (`BudgetPlan` / `PlanningChecklist`) and two SAFE, read-only tools (`days_until`, `split_budget`) behind a tool-loop agent. The agent is exercised offline and via the opt-in `npm run test:contracts` probe; wiring it into the CLI/HTTP service is deferred to a later phase.

Phase 7 adds a durable, app-owned knowledge base (sqlite-vec) with idempotent, source-addressed ingestion. The embedding alias and its vector dimension are verified through the proxy by the opt-in `npm run test:embedding` probe (LIVE — real credentialed calls; NOT part of `npm test` or CI), whose pure compatibility logic is unit-tested offline. The expected dimension is the single source of truth `LITELLM_EMBED_DIM` (default 768), read by both the knowledge store and the probe.

Phase 7 also adds the retriever (`src/core/retriever.ts`) and a retrieval-only evaluation. `retrieve()` embeds a query, KNN-searches the knowledge store, and resolves each hit to TRUSTED, APP-OWNED metadata (documentId, sourceUri, chunkIndex, ownerId, contentHash) pulled from the store — never model-generated — the basis for later authorization and trusted citations. A small curated, benign, PII-free wedding corpus lives under [`knowledge/`](knowledge/README.md) and a versioned retrieval-eval dataset at `evals/retrieval.jsonl` references relevance by stable `source_uri`. The opt-in `npm run eval:retrieval` command (LIVE — real credentialed calls; NOT part of `npm test` or CI) ingests the corpus into an ephemeral store, runs each query, computes recall@k / precision@k / MRR / nDCG@k, checks a PROPOSED baseline (pending user approval at Phase 7 closeout), and writes dated evidence to `docs/retrieval/<date>.md`. The pure metrics/parser/renderers are unit-tested offline.

## Running the conversation service

`npm run serve` starts the durable, authenticated HTTP + SSE conversation service (Phase 5). Copy `.env.example` to `.env` first; set `AUTH_TOKENS` (the service refuses to start with none) and, for real chat turns, the `LITELLM_*` credentials. Optional: `SERVICE_PORT` (default `3000`, bound on `127.0.0.1`), `CHECKPOINT_DB_PATH` (durable SQLite DB), and the `SERVICE_*_TIMEOUT_MS` hardening timeouts. See `.env.example` and [AGENTS.md](AGENTS.md) for the full list and the service's ownership/security model.
