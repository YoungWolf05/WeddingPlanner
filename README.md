# Wedding Planner Chatbot

See the [canonical product roadmap](docs/roadmap.md) for phase scope, status, and governance.

Latest live evidence: [capability matrices](docs/capabilities/), [tool-call & structured-output contract matrices](docs/contracts/), and [evaluation baselines](docs/eval/) (each folder's `README.md` links the most recent dated snapshot).

Phase 6 adds validated wedding-domain structured output (`BudgetPlan` / `PlanningChecklist`) and two SAFE, read-only tools (`days_until`, `split_budget`) behind a tool-loop agent. The agent is exercised offline and via the opt-in `npm run test:contracts` probe; wiring it into the CLI/HTTP service is deferred to a later phase.

## Running the conversation service

`npm run serve` starts the durable, authenticated HTTP + SSE conversation service (Phase 5). Copy `.env.example` to `.env` first; set `AUTH_TOKENS` (the service refuses to start with none) and, for real chat turns, the `LITELLM_*` credentials. Optional: `SERVICE_PORT` (default `3000`, bound on `127.0.0.1`), `CHECKPOINT_DB_PATH` (durable SQLite DB), and the `SERVICE_*_TIMEOUT_MS` hardening timeouts. See `.env.example` and [AGENTS.md](AGENTS.md) for the full list and the service's ownership/security model.
