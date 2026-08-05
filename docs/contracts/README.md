# Tool-call & structured-output contract matrices

Dated per-alias contract matrices produced by the opt-in `npm run test:contracts`
command (LIVE — makes real, credentialed LiteLLM calls; NOT part of `npm test`
or CI). Each file is an **immutable historical snapshot** of one live probe run
against the proxy. For every enabled model alias it records the two Phase 6 exit
criterion (4) contract checks:

- **Tool call** — the model, given the SAFE `weddingTools`, returns a well-formed
  tool call for a permitted tool (`days_until` / `split_budget`) with parseable
  args.
- **Structured output** — `generateBudgetPlan` returns a schema-valid `BudgetPlan`.

Each cell is `Supported`, `Unsupported`, `Degraded`, `Error`, or `N/A`. Opus is
probed with temperature OMITTED (Phase 4 carry-forward: `claude-opus-4-8`
structured/tool paths reject an explicit temperature). Secrets are redacted
(host-only base URL, masked key).

The pure classification/rendering logic lives in
[`src/core/contracts.ts`](../../src/core/contracts.ts) and the live probe in
[`src/probe-contracts.ts`](../../src/probe-contracts.ts). The pure logic is
unit-tested offline in `test/phase6-contract-probe-render.test.ts`.

## Latest matrix

- [2026-08-05](2026-08-05.md) — `claude-opus-4-8` and `claude-sonnet-4-6` both
  `Supported` for tool call and structured output.

To refresh, run `npm run test:contracts`, which writes a new dated file. Older
dated files are retained unchanged as historical snapshots; do not edit their
recorded results.
