# Retrieval baselines

Dated retrieval-only evaluation baselines produced by the opt-in
`npm run eval:retrieval` command (LIVE — makes real, credentialed LiteLLM
embedding calls; NOT part of `npm test` or CI). Each file is an **immutable
historical snapshot** of one live run against the proxy, targeting Phase 7 exit
criterion 5: "retrieval quality meets an approved baseline before
answer-generation tuning begins".

The command builds an EPHEMERAL knowledge store in a temp dir OUTSIDE the repo,
ingests the curated corpus under [`knowledge/`](../../knowledge/README.md) with
the real embedding model, runs each query in
[`evals/retrieval.jsonl`](../../evals/retrieval.jsonl) through the retriever, and
records:

- **Aggregate** — mean `recall@k`, `precision@k`, `MRR`, and `nDCG@k` over all
  queries.
- **Baseline gate** — PASS/FAIL of each aggregate metric against the PROPOSED
  thresholds. These thresholds are **PROPOSED and pending user approval** at
  Phase 7 closeout (exit criterion 5 requires an approved baseline); they are
  inputs to the gate, never hard-coded as "met".
- **Per-query** — recall@k / precision@k / reciprocal rank / nDCG@k per query.

Secrets are redacted (host-only base URL, masked key; any error reason passes
through the shared redaction). Metrics operate on DOCUMENT-LEVEL relevance and
the dataset references relevance by stable `source_uri`, so it does not go stale
when chunking is tuned.

The pure metrics/parsing/rendering live in
[`src/core/retrieval-eval.ts`](../../src/core/retrieval-eval.ts) and the
retriever in [`src/core/retriever.ts`](../../src/core/retriever.ts); both are
unit-tested offline (`test/phase7-retrieval-eval.test.ts`,
`test/phase7-retriever.test.ts`). The live runner is
[`src/run-retrieval-eval.ts`](../../src/run-retrieval-eval.ts).

## Latest baseline

- [2026-08-06](2026-08-06.md) — `gemini-embedding-001` @ 768 dims, 12 corpus
  documents, 15 queries, k=5: recall@5 **1.000**, precision@5 **0.227**
  (theoretical max on this mostly-single-relevant dataset), MRR **1.000**,
  nDCG@5 **1.000**. **PASS** against the PROPOSED thresholds. The thresholds are
  still PROPOSED and require USER APPROVAL at Phase 7 closeout (exit criterion 5).

To refresh, run `npm run eval:retrieval`, which writes a new dated file
(overwriting only a same-day run). Older dated files are retained unchanged as
historical snapshots; do not edit their recorded results.
