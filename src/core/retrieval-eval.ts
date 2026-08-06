// Phase 7 (increment 7e) — RETRIEVAL-ONLY EVALUATION: PURE logic.
//
// This module holds ONLY pure, deterministic, offline logic:
//   - the retrieval-eval dataset item schema + a strict JSONL parser/validator,
//   - the retrieval metric functions (recall@k, precision@k, MRR, nDCG@k),
//   - an aggregator (per-query + mean metrics),
//   - a baseline-gate function (evaluateBaseline) + PROPOSED default thresholds,
//   - pure renderers for a dated Markdown evidence body + a console summary.
//
// It performs NO network calls, reads NO credentials, and does NO file I/O, so
// the offline Vitest suite imports it directly and exercises every metric against
// hand-computed fixtures. All live I/O (loading evals/retrieval.jsonl, building
// an ephemeral store, ingesting the curated corpus, embedding queries, calling
// the retriever, writing the dated docs/retrieval/<date>.md) lives in
// src/run-retrieval-eval.ts. This mirrors the eval.ts / run-eval.ts split.
//
// RELEVANCE MODEL (documented): metrics operate on DOCUMENT-LEVEL relevance.
// A retrieved ranked list of chunks is first reduced to its ranked list of
// DISTINCT documentIds (first occurrence wins — the best rank a document
// achieves), and metrics compare that ranked documentId list against the item's
// set of relevant documentIds. Document-level relevance is used because:
//   - a query is judged "answered" when the RIGHT SOURCE is retrieved, and
//   - documentIds are derived from the stable source_uri (7c source-addressed
//     identity), whereas chunkIds also depend on the chunking parameters, so
//     document-level expectations stay stable across chunk-size tuning.
// See DATASET STABILITY below for how the dataset references relevance stably.

import { isEmbeddingDimensionCompatible } from "./embedding-compat.js";

// ---------------------------------------------------------------------------
// Dataset schema + STABILITY
// ---------------------------------------------------------------------------
//
// DATASET STABILITY (documented): the dataset references relevance by
// `relevantSourceUris` — the STABLE, human-readable source identity of each
// corpus document (7c: source_uri IS the document's app-owned identity). The
// live runner derives each document_id deterministically as
// sha256(normalizeSourceUri(source_uri)) (knowledge-store.computeDocumentId), so
// referencing by source_uri is EXACTLY equivalent to referencing by document_id
// but is stable, readable, and auditable by a human against the curated corpus.
// Because chunking parameters do not affect a source_uri, the dataset does NOT
// go stale when the chunking config is tuned. This keeps expectations defensible
// (a reviewer can see why source X answers query Q) and non-brittle.

// One versioned retrieval-eval dataset item. `id` is stable + unique.
// `relevantSourceUris` is the non-empty set of corpus source URIs that a good
// retriever should surface for `query` (order-independent; treated as a set).
export interface RetrievalEvalItem {
  id: string;
  query: string;
  relevantSourceUris: string[];
}

// ---------------------------------------------------------------------------
// Dataset parsing / validation (pure) — mirrors eval.ts parseDataset strictness.
// ---------------------------------------------------------------------------

// Only these keys are honored; an unknown key is rejected so a typo cannot
// silently disable an expectation.
const KNOWN_ITEM_KEYS: ReadonlySet<string> = new Set([
  "id",
  "query",
  "relevantSourceUris",
]);

function validateItem(raw: unknown, index: number): RetrievalEvalItem {
  const where = `retrieval dataset item #${index}`;
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error(`${where}: expected a JSON object`);
  }
  const rec = raw as Record<string, unknown>;

  for (const key of Object.keys(rec)) {
    if (!KNOWN_ITEM_KEYS.has(key)) {
      throw new Error(`${where}: unknown key "${key}"`);
    }
  }

  const id = rec["id"];
  if (typeof id !== "string" || id.trim() === "") {
    throw new Error(`${where}: "id" must be a non-empty string`);
  }
  const query = rec["query"];
  if (typeof query !== "string" || query.trim() === "") {
    throw new Error(`${where} (id=${id}): "query" must be a non-empty string`);
  }
  const relRaw = rec["relevantSourceUris"];
  if (
    !Array.isArray(relRaw) ||
    relRaw.length === 0 ||
    !relRaw.every((s) => typeof s === "string" && s.trim() !== "")
  ) {
    throw new Error(
      `${where} (id=${id}): "relevantSourceUris" must be a non-empty array of ` +
        `non-empty strings`
    );
  }
  // Reject duplicate source URIs within a single item (a set, not a multiset).
  const rel = relRaw as string[];
  const relSet = new Set(rel);
  if (relSet.size !== rel.length) {
    throw new Error(
      `${where} (id=${id}): "relevantSourceUris" contains duplicate entries`
    );
  }

  return { id, query, relevantSourceUris: rel };
}

// Parse the retrieval dataset from raw JSONL text (one JSON object per non-blank
// line). Validates every item, rejects duplicate ids, and returns ordered items.
// Pure: the caller reads the file and hands the text in.
export function parseRetrievalDataset(jsonl: string): RetrievalEvalItem[] {
  const lines = jsonl.split(/\r?\n/);
  const items: RetrievalEvalItem[] = [];
  const seen = new Set<string>();
  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (trimmed === "") return; // skip blank lines
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch (err) {
      throw new Error(
        `retrieval dataset line ${i + 1}: invalid JSON (${
          err instanceof Error ? err.message : String(err)
        })`
      );
    }
    const item = validateItem(parsed, items.length);
    if (seen.has(item.id)) {
      throw new Error(`duplicate retrieval dataset id: ${item.id}`);
    }
    seen.add(item.id);
    items.push(item);
  });
  if (items.length === 0) {
    throw new Error("retrieval dataset is empty");
  }
  return items;
}

// ---------------------------------------------------------------------------
// Metric primitives (pure) — operate on a RANKED list of ids vs a relevant SET.
// ---------------------------------------------------------------------------
//
// `ranked` is best-first (index 0 == top result). `relevant` is the set of
// relevant ids. `k` is the cutoff (>= 1). Ids beyond position k are ignored by
// the @k metrics. All functions are deterministic and LLM-free.

function topK(ranked: readonly string[], k: number): readonly string[] {
  return ranked.slice(0, Math.max(0, k));
}

/**
 * recall@k = |relevant ∩ retrieved_topk| / |relevant|.
 * The fraction of the relevant documents that appear in the top-k. Range [0,1].
 * With an empty relevant set the metric is undefined; callers never pass one
 * (the parser forbids empty relevance), so we define it as 0 defensively.
 */
export function recallAtK(
  ranked: readonly string[],
  relevant: ReadonlySet<string>,
  k: number
): number {
  if (relevant.size === 0) return 0;
  const top = topK(ranked, k);
  let hit = 0;
  for (const id of relevant) {
    if (top.includes(id)) hit += 1;
  }
  return hit / relevant.size;
}

/**
 * precision@k = |relevant ∩ retrieved_topk| / k.
 * The fraction of the top-k results that are relevant. Range [0,1]. Denominator
 * is the CUTOFF k (not the number actually returned) — the standard definition —
 * so under-filling the top-k lowers precision, as it should.
 */
export function precisionAtK(
  ranked: readonly string[],
  relevant: ReadonlySet<string>,
  k: number
): number {
  if (k <= 0) return 0;
  const top = topK(ranked, k);
  let hit = 0;
  for (const id of top) {
    if (relevant.has(id)) hit += 1;
  }
  return hit / k;
}

/**
 * Reciprocal Rank = 1 / (rank of the FIRST relevant result), 1-indexed; 0 when
 * no relevant result appears anywhere in `ranked`. MRR is the MEAN of this over
 * a dataset (see aggregate). Unbounded-by-k by definition (scans the full list),
 * but we cap the scan at the full ranked list the retriever returned.
 */
export function reciprocalRank(
  ranked: readonly string[],
  relevant: ReadonlySet<string>
): number {
  for (let i = 0; i < ranked.length; i++) {
    if (relevant.has(ranked[i]!)) return 1 / (i + 1);
  }
  return 0;
}

/**
 * nDCG@k with BINARY relevance (gain 1 if relevant, else 0) and the standard
 * log2 discount:
 *
 *     DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
 *     IDCG@k = DCG@k of the ideal ranking (all relevant items first)
 *     nDCG@k = DCG@k / IDCG@k        (0 when IDCG@k == 0)
 *
 * Range [0,1]. With binary relevance IDCG@k is the DCG of min(|relevant|, k)
 * ones at the top. Deterministic; no LLM judge.
 */
export function ndcgAtK(
  ranked: readonly string[],
  relevant: ReadonlySet<string>,
  k: number
): number {
  const top = topK(ranked, k);
  let dcg = 0;
  for (let i = 0; i < top.length; i++) {
    if (relevant.has(top[i]!)) {
      dcg += 1 / Math.log2(i + 2); // i is 0-indexed -> position i+1 -> log2(i+2)
    }
  }
  const idealHits = Math.min(relevant.size, k);
  let idcg = 0;
  for (let i = 0; i < idealHits; i++) {
    idcg += 1 / Math.log2(i + 2);
  }
  return idcg === 0 ? 0 : dcg / idcg;
}

// ---------------------------------------------------------------------------
// Per-query scoring + aggregation (pure)
// ---------------------------------------------------------------------------

// The metric names, in a fixed display order.
export const RETRIEVAL_METRICS = [
  "recallAtK",
  "precisionAtK",
  "mrr",
  "ndcgAtK",
] as const;
export type RetrievalMetricName = (typeof RETRIEVAL_METRICS)[number];

// One query's computed metrics. `mrr` here is the per-query reciprocal rank; the
// dataset MEAN of it is the MRR reported in the aggregate.
export interface QueryMetrics {
  recallAtK: number;
  precisionAtK: number;
  mrr: number; // per-query reciprocal rank
  ndcgAtK: number;
}

// The scored outcome for one query: the item id, the k used, the ranked
// documentIds the retriever produced (already reduced to distinct docs), and the
// metrics. An optional redacted error reason is carried when the live retrieval
// for this query failed (the runner records a zeroed score so it counts honestly).
export interface QueryScore {
  id: string;
  k: number;
  rankedDocumentIds: string[];
  metrics: QueryMetrics;
  errorReason?: string;
}

// Reduce a ranked list of chunk-level documentIds to a ranked list of DISTINCT
// documentIds (first occurrence wins). This is the document-level projection the
// metrics operate on (see RELEVANCE MODEL at the top). Exported for the runner
// and tests.
export function distinctInOrder(ids: readonly string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const id of ids) {
    if (!seen.has(id)) {
      seen.add(id);
      out.push(id);
    }
  }
  return out;
}

// Compute all four metrics for one query given the ranked DISTINCT documentIds
// and the relevant documentId set at cutoff `k`. Pure.
export function scoreQuery(
  id: string,
  rankedDocumentIds: readonly string[],
  relevantDocumentIds: ReadonlySet<string>,
  k: number
): QueryScore {
  const distinct = distinctInOrder(rankedDocumentIds);
  return {
    id,
    k,
    rankedDocumentIds: distinct,
    metrics: {
      recallAtK: recallAtK(distinct, relevantDocumentIds, k),
      precisionAtK: precisionAtK(distinct, relevantDocumentIds, k),
      mrr: reciprocalRank(distinct, relevantDocumentIds),
      ndcgAtK: ndcgAtK(distinct, relevantDocumentIds, k),
    },
  };
}

// The dataset-level aggregate: mean of each metric across all queries, plus the
// cutoff k and the query count. `mrr` is the MEAN reciprocal rank (i.e. MRR).
export interface RetrievalAggregate {
  k: number;
  queryCount: number;
  meanRecallAtK: number;
  meanPrecisionAtK: number;
  mrr: number;
  meanNdcgAtK: number;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// Aggregate per-query scores into the dataset means. Pure + deterministic. The
// cutoff k is taken from the scores (they all share one k in a run); an empty
// input yields zeros with k defaulted to 0.
export function aggregateRetrieval(scores: QueryScore[]): RetrievalAggregate {
  return {
    k: scores[0]?.k ?? 0,
    queryCount: scores.length,
    meanRecallAtK: mean(scores.map((s) => s.metrics.recallAtK)),
    meanPrecisionAtK: mean(scores.map((s) => s.metrics.precisionAtK)),
    mrr: mean(scores.map((s) => s.metrics.mrr)),
    meanNdcgAtK: mean(scores.map((s) => s.metrics.ndcgAtK)),
  };
}

// ---------------------------------------------------------------------------
// Baseline gate (pure)
// ---------------------------------------------------------------------------

// The thresholds an aggregate must MEET-OR-EXCEED to pass the baseline gate.
// Every field is a fraction in [0,1] on the corresponding aggregate mean.
export interface BaselineThresholds {
  meanRecallAtK: number;
  meanPrecisionAtK: number;
  mrr: number;
  meanNdcgAtK: number;
}

// PROPOSED default baseline thresholds — PENDING USER APPROVAL AT CLOSEOUT.
//
// These are DELIBERATELY CONSERVATIVE, defensible starting points for a small
// curated corpus with a single embedding model and NO reranking (7e stops at
// retrieval). They are NOT ratified: exit criterion 5 requires the user to
// APPROVE a baseline at Phase 7 closeout. Rationale for each proposed value:
//   - meanRecallAtK 0.80   : most queries should surface their relevant source
//                            within the top-k on a clean curated corpus.
//   - meanPrecisionAtK 0.18: precision@k is CAPPED BY CONSTRUCTION on this
//                            dataset. Most queries have a SINGLE relevant
//                            document, so with k=5 the maximum precision@5 for a
//                            single-relevant query is 1/5 = 0.20; the dataset's
//                            two dual-relevant queries lift the theoretical mean
//                            ceiling only to ~0.227. A floor of 0.30 would be
//                            mathematically UNREACHABLE, so the proposed floor is
//                            0.18 (just under the achievable ceiling) — it
//                            rewards putting the relevant doc(s) in the top-k
//                            without demanding impossible precision. If the k or
//                            the relevance density changes, revisit this value.
//   - mrr 0.70             : the first relevant source should usually rank very
//                            near the top (rank 1-2).
//   - meanNdcgAtK 0.75     : relevant sources should be ranked high, not merely
//                            present.
// A reviewer may raise/lower any of these at closeout; they are inputs, never
// hard-coded "met" facts.
export const PROPOSED_BASELINE_THRESHOLDS: BaselineThresholds = {
  meanRecallAtK: 0.8,
  meanPrecisionAtK: 0.18,
  mrr: 0.7,
  meanNdcgAtK: 0.75,
};

// One metric's gate outcome.
export interface MetricGate {
  metric: keyof BaselineThresholds;
  value: number;
  threshold: number;
  passed: boolean;
}

// The full baseline-gate verdict.
export interface BaselineResult {
  passed: boolean;
  perMetric: MetricGate[];
}

// Evaluate an aggregate against thresholds. A metric passes iff value >=
// threshold; the gate passes iff EVERY metric passes. Pure + deterministic.
// Thresholds are an INPUT (never hard-coded here as "met").
export function evaluateBaseline(
  aggregate: RetrievalAggregate,
  thresholds: BaselineThresholds
): BaselineResult {
  const perMetric: MetricGate[] = [
    {
      metric: "meanRecallAtK",
      value: aggregate.meanRecallAtK,
      threshold: thresholds.meanRecallAtK,
      passed: aggregate.meanRecallAtK >= thresholds.meanRecallAtK,
    },
    {
      metric: "meanPrecisionAtK",
      value: aggregate.meanPrecisionAtK,
      threshold: thresholds.meanPrecisionAtK,
      passed: aggregate.meanPrecisionAtK >= thresholds.meanPrecisionAtK,
    },
    {
      metric: "mrr",
      value: aggregate.mrr,
      threshold: thresholds.mrr,
      passed: aggregate.mrr >= thresholds.mrr,
    },
    {
      metric: "meanNdcgAtK",
      value: aggregate.meanNdcgAtK,
      threshold: thresholds.meanNdcgAtK,
      passed: aggregate.meanNdcgAtK >= thresholds.meanNdcgAtK,
    },
  ];
  return { passed: perMetric.every((m) => m.passed), perMetric };
}

// ---------------------------------------------------------------------------
// Rendering (pure) — mirrors eval.ts / embedding-compat.ts renderers.
// ---------------------------------------------------------------------------

// Run metadata for the dated evidence file. Redaction (host-only, masked key)
// happens UPSTREAM in the runner; this module renders exactly what it is given.
export interface RetrievalRunMeta {
  runTimestampUtc: string;
  embedModel: string;
  embedDim: number;
  baseUrlHost: string;
  maskedKey: string;
  k: number;
  corpusDocumentCount: number;
}

function fixed(value: number): string {
  return value.toFixed(3);
}

function mdCell(value: string): string {
  return value.replace(/\|/g, "\\|");
}

// Re-export the dimension predicate so a caller wiring the runner has a single
// import surface for retrieval-eval concerns (used to sanity-check the store dim
// vs the observed query dim before a run). Pure passthrough.
export { isEmbeddingDimensionCompatible };

// Render the dated Markdown evidence body: run metadata, the aggregate table,
// the baseline-gate table (marked PROPOSED / pending approval), and a per-query
// table. Deterministic given the same inputs. Contains NO secrets/PII — the
// runner redacts host/key/errors upstream.
export function renderRetrievalMarkdown(
  meta: RetrievalRunMeta,
  scores: QueryScore[],
  aggregate: RetrievalAggregate,
  baseline: BaselineResult
): string {
  const lines: string[] = [];

  lines.push("# Wedding Planner Retrieval Evaluation Baseline");
  lines.push("");
  lines.push(`- **Run (UTC):** ${meta.runTimestampUtc}`);
  lines.push(`- **Embedding model:** ${meta.embedModel}`);
  lines.push(`- **Embedding dim:** ${meta.embedDim}`);
  lines.push(`- **Base URL host:** ${meta.baseUrlHost}`);
  lines.push(`- **API key:** ${meta.maskedKey} (masked)`);
  lines.push(`- **k (cutoff):** ${meta.k}`);
  lines.push(`- **Corpus documents:** ${meta.corpusDocumentCount}`);
  lines.push(`- **Queries:** ${aggregate.queryCount}`);
  lines.push("");

  lines.push("## Aggregate (mean over queries)");
  lines.push("");
  lines.push("| Metric | Value |");
  lines.push("| --- | --- |");
  lines.push(`| recall@${aggregate.k} | ${fixed(aggregate.meanRecallAtK)} |`);
  lines.push(`| precision@${aggregate.k} | ${fixed(aggregate.meanPrecisionAtK)} |`);
  lines.push(`| MRR | ${fixed(aggregate.mrr)} |`);
  lines.push(`| nDCG@${aggregate.k} | ${fixed(aggregate.meanNdcgAtK)} |`);
  lines.push("");

  lines.push("## Baseline gate (PROPOSED thresholds — pending user approval)");
  lines.push("");
  lines.push(
    `**Result: ${baseline.passed ? "PASS" : "FAIL"}** against the PROPOSED ` +
      `thresholds below. These thresholds are NOT yet ratified; exit criterion 5 ` +
      `requires user approval of a baseline at Phase 7 closeout.`
  );
  lines.push("");
  lines.push("| Metric | Value | Threshold | Result |");
  lines.push("| --- | --- | --- | --- |");
  for (const g of baseline.perMetric) {
    lines.push(
      `| ${g.metric} | ${fixed(g.value)} | ${fixed(g.threshold)} | ${
        g.passed ? "PASS" : "FAIL"
      } |`
    );
  }
  lines.push("");

  lines.push("## Per-query results");
  lines.push("");
  lines.push(
    `| ID | recall@${meta.k} | precision@${meta.k} | RR | nDCG@${meta.k} | Notes |`
  );
  lines.push("| --- | --- | --- | --- | --- | --- |");
  for (const s of scores) {
    const notes = s.errorReason ? `error: ${s.errorReason}` : "-";
    lines.push(
      `| ${mdCell(s.id)} | ${fixed(s.metrics.recallAtK)} | ` +
        `${fixed(s.metrics.precisionAtK)} | ${fixed(s.metrics.mrr)} | ` +
        `${fixed(s.metrics.ndcgAtK)} | ${mdCell(notes)} |`
    );
  }
  lines.push("");

  lines.push("## Method");
  lines.push("");
  lines.push(
    "Metrics are produced by deterministic, LLM-free functions in " +
      "`src/core/retrieval-eval.ts` over DOCUMENT-LEVEL relevance (the ranked " +
      "chunk list is reduced to distinct source documents). Relevance is " +
      "referenced in `evals/retrieval.jsonl` by stable `source_uri` (the 7c " +
      "app-owned document identity), so the dataset does not go stale when " +
      "chunking is tuned. The same pure functions grade this live run and the " +
      "offline test suite."
  );
  lines.push("");

  return lines.join("\n");
}

// Render a concise console summary of a run. Deterministic.
export function renderRetrievalConsoleSummary(
  aggregate: RetrievalAggregate,
  baseline: BaselineResult
): string {
  const lines: string[] = [];
  lines.push(`Queries: ${aggregate.queryCount} (k=${aggregate.k})`);
  lines.push(`  recall@${aggregate.k}    : ${fixed(aggregate.meanRecallAtK)}`);
  lines.push(`  precision@${aggregate.k} : ${fixed(aggregate.meanPrecisionAtK)}`);
  lines.push(`  MRR              : ${fixed(aggregate.mrr)}`);
  lines.push(`  nDCG@${aggregate.k}      : ${fixed(aggregate.meanNdcgAtK)}`);
  lines.push(
    `Baseline gate (PROPOSED thresholds): ${baseline.passed ? "PASS" : "FAIL"}`
  );
  for (const g of baseline.perMetric) {
    lines.push(
      `  ${g.metric}: ${fixed(g.value)} vs ${fixed(g.threshold)} -> ${
        g.passed ? "PASS" : "FAIL"
      }`
    );
  }
  return lines.join("\n");
}
