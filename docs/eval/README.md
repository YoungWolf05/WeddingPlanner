# Evaluation baselines

Dated wedding-planning evaluation baselines produced by the opt-in `npm run eval`
command. Each file is an **immutable historical snapshot** of one live run
against the LiteLLM proxy: it records the model, an aggregate score, a
per-category summary, and per-item pass/fail with the deterministic property
reasons. Secrets are redacted (host-only base URL, masked key).

The dataset lives at [`evals/dataset.jsonl`](../../evals/dataset.jsonl) and is
graded by the pure, offline scorers in
[`src/core/eval.ts`](../../src/core/eval.ts) (no LLM judge). The same scorers run
in the offline test suite (`npm test`), so these baselines are reproducible.

## Latest baseline

- [2026-07-28](2026-07-28.md) — `claude-sonnet-4-6`, 12/12 items passed (100%).

Older dated files below the latest are retained unchanged as historical
snapshots; do not edit their recorded results. To refresh the baseline, run
`npm run eval`, which writes a new dated file (overwriting only a same-day run).
