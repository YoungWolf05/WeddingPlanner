import { describe, it, expect } from "vitest";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  parseRetrievalDataset,
  recallAtK,
  precisionAtK,
  reciprocalRank,
  ndcgAtK,
  distinctInOrder,
  scoreQuery,
  aggregateRetrieval,
  evaluateBaseline,
  renderRetrievalMarkdown,
  renderRetrievalConsoleSummary,
  PROPOSED_BASELINE_THRESHOLDS,
  type QueryScore,
  type RetrievalRunMeta,
  type RetrievalAggregate,
} from "../src/core/retrieval-eval.js";

// Phase 7 (7e) — RETRIEVAL-EVAL pure logic coverage.
//
// Fully OFFLINE + DETERMINISTIC: pure parser + metric math on hand-computed
// fixtures + aggregator + baseline gate + renderer determinism. NO network, NO
// credentials, NO doc writes. The live runner (src/run-retrieval-eval.ts) is
// never imported here.

const rel = (...ids: string[]): Set<string> => new Set(ids);

describe("Phase 7 (7e) — parseRetrievalDataset (strict)", () => {
  it("parses valid JSONL into ordered items", () => {
    const jsonl = [
      JSON.stringify({ id: "q1", query: "budget split", relevantSourceUris: ["a.md"] }),
      "",
      JSON.stringify({ id: "q2", query: "timeline", relevantSourceUris: ["b.md", "c.md"] }),
    ].join("\n");
    const items = parseRetrievalDataset(jsonl);
    expect(items.map((i) => i.id)).toEqual(["q1", "q2"]);
    expect(items[1]!.relevantSourceUris).toEqual(["b.md", "c.md"]);
  });

  it("rejects an unknown key with the item index", () => {
    const jsonl = JSON.stringify({
      id: "q1",
      query: "x",
      relevantSourceUris: ["a.md"],
      typoKey: 1,
    });
    expect(() => parseRetrievalDataset(jsonl)).toThrow(/#0.*unknown key "typoKey"/);
  });

  it("rejects a missing/empty id", () => {
    const jsonl = JSON.stringify({ id: "  ", query: "x", relevantSourceUris: ["a.md"] });
    expect(() => parseRetrievalDataset(jsonl)).toThrow(/"id" must be a non-empty string/);
  });

  it("rejects an empty query", () => {
    const jsonl = JSON.stringify({ id: "q1", query: "", relevantSourceUris: ["a.md"] });
    expect(() => parseRetrievalDataset(jsonl)).toThrow(/"query" must be a non-empty string/);
  });

  it("rejects an empty / non-array relevantSourceUris", () => {
    expect(() =>
      parseRetrievalDataset(JSON.stringify({ id: "q1", query: "x", relevantSourceUris: [] }))
    ).toThrow(/"relevantSourceUris" must be a non-empty array/);
    expect(() =>
      parseRetrievalDataset(JSON.stringify({ id: "q1", query: "x", relevantSourceUris: "a.md" }))
    ).toThrow(/"relevantSourceUris" must be a non-empty array/);
  });

  it("rejects duplicate relevance entries within an item", () => {
    const jsonl = JSON.stringify({
      id: "q1",
      query: "x",
      relevantSourceUris: ["a.md", "a.md"],
    });
    expect(() => parseRetrievalDataset(jsonl)).toThrow(/duplicate entries/);
  });

  it("rejects duplicate ids across items", () => {
    const jsonl = [
      JSON.stringify({ id: "dup", query: "x", relevantSourceUris: ["a.md"] }),
      JSON.stringify({ id: "dup", query: "y", relevantSourceUris: ["b.md"] }),
    ].join("\n");
    expect(() => parseRetrievalDataset(jsonl)).toThrow(/duplicate retrieval dataset id: dup/);
  });

  it("reports the line number for invalid JSON", () => {
    const jsonl = [
      JSON.stringify({ id: "q1", query: "x", relevantSourceUris: ["a.md"] }),
      "{ not json",
    ].join("\n");
    expect(() => parseRetrievalDataset(jsonl)).toThrow(/line 2: invalid JSON/);
  });

  it("rejects an empty dataset", () => {
    expect(() => parseRetrievalDataset("\n\n")).toThrow(/retrieval dataset is empty/);
  });
});

describe("Phase 7 (7e) — recall@k (hand-computed)", () => {
  it("all relevant in top-k => 1", () => {
    expect(recallAtK(["a", "b", "c"], rel("a", "b"), 3)).toBe(1);
  });
  it("half of relevant in top-k => 0.5", () => {
    // relevant {a,b}; top-2 = [x,a] contains a only => 1/2.
    expect(recallAtK(["x", "a", "b"], rel("a", "b"), 2)).toBe(0.5);
  });
  it("none relevant => 0", () => {
    expect(recallAtK(["x", "y"], rel("a"), 2)).toBe(0);
  });
});

describe("Phase 7 (7e) — precision@k (hand-computed)", () => {
  it("2 relevant of top-4 with k=4 => 0.5", () => {
    expect(precisionAtK(["a", "x", "b", "y"], rel("a", "b"), 4)).toBe(0.5);
  });
  it("denominator is the cutoff k, not the number returned", () => {
    // Only 2 results but k=4: 1 relevant / 4 = 0.25.
    expect(precisionAtK(["a", "x"], rel("a"), 4)).toBe(0.25);
  });
  it("k=1 top result relevant => 1", () => {
    expect(precisionAtK(["a", "b"], rel("a"), 1)).toBe(1);
  });
});

describe("Phase 7 (7e) — reciprocal rank (hand-computed)", () => {
  it("first relevant at rank 1 => 1", () => {
    expect(reciprocalRank(["a", "b"], rel("a"))).toBe(1);
  });
  it("first relevant at rank 3 => 1/3", () => {
    expect(reciprocalRank(["x", "y", "a"], rel("a"))).toBeCloseTo(1 / 3, 10);
  });
  it("no relevant => 0", () => {
    expect(reciprocalRank(["x", "y"], rel("a"))).toBe(0);
  });
});

describe("Phase 7 (7e) — nDCG@k (hand-computed)", () => {
  it("single relevant at rank 1 => 1 (DCG==IDCG)", () => {
    expect(ndcgAtK(["a", "x", "y"], rel("a"), 3)).toBe(1);
  });

  it("single relevant at rank 2 => 1/log2(3) normalized by ideal 1/log2(2)=1", () => {
    // DCG = 1/log2(3); IDCG (one relevant at top) = 1/log2(2) = 1.
    const expected = 1 / Math.log2(3);
    expect(ndcgAtK(["x", "a", "y"], rel("a"), 3)).toBeCloseTo(expected, 10);
  });

  it("two relevant, one at rank1 one at rank3 (k=3)", () => {
    // DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    // IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309... = 1.6309...
    const dcg = 1 / Math.log2(2) + 1 / Math.log2(4);
    const idcg = 1 / Math.log2(2) + 1 / Math.log2(3);
    expect(ndcgAtK(["a", "x", "b"], rel("a", "b"), 3)).toBeCloseTo(dcg / idcg, 10);
  });

  it("no relevant => 0", () => {
    expect(ndcgAtK(["x", "y"], rel("a"), 2)).toBe(0);
  });
});

describe("Phase 7 (7e) — distinctInOrder (document-level projection)", () => {
  it("keeps first occurrence and preserves order", () => {
    expect(distinctInOrder(["a", "a", "b", "a", "c", "b"])).toEqual(["a", "b", "c"]);
  });
  it("empty stays empty", () => {
    expect(distinctInOrder([])).toEqual([]);
  });
});

describe("Phase 7 (7e) — scoreQuery + aggregateRetrieval", () => {
  it("reduces chunk-level docIds to distinct docs before scoring", () => {
    // ranked chunk docIds [a,a,b]; distinct = [a,b]; relevant {b} => RR = 1/2.
    const s = scoreQuery("q", ["a", "a", "b"], rel("b"), 3);
    expect(s.rankedDocumentIds).toEqual(["a", "b"]);
    expect(s.metrics.mrr).toBeCloseTo(0.5, 10);
  });

  it("mean of per-query metrics is correct", () => {
    const s1 = scoreQuery("q1", ["a"], rel("a"), 3); // recall 1, RR 1
    const s2 = scoreQuery("q2", ["x"], rel("a"), 3); // recall 0, RR 0
    const agg = aggregateRetrieval([s1, s2]);
    expect(agg.queryCount).toBe(2);
    expect(agg.k).toBe(3);
    expect(agg.meanRecallAtK).toBeCloseTo(0.5, 10);
    expect(agg.mrr).toBeCloseTo(0.5, 10);
  });

  it("empty aggregate yields zeros", () => {
    const agg = aggregateRetrieval([]);
    expect(agg.queryCount).toBe(0);
    expect(agg.meanRecallAtK).toBe(0);
    expect(agg.mrr).toBe(0);
  });
});

describe("Phase 7 (7e) — evaluateBaseline (pass/fail against thresholds)", () => {
  const perfect: RetrievalAggregate = {
    k: 5,
    queryCount: 3,
    meanRecallAtK: 1,
    meanPrecisionAtK: 1,
    mrr: 1,
    meanNdcgAtK: 1,
  };

  it("passes when every metric meets-or-exceeds its threshold", () => {
    const res = evaluateBaseline(perfect, PROPOSED_BASELINE_THRESHOLDS);
    expect(res.passed).toBe(true);
    expect(res.perMetric.every((m) => m.passed)).toBe(true);
  });

  it("fails when any single metric is below threshold", () => {
    const agg: RetrievalAggregate = { ...perfect, mrr: 0.1 };
    const res = evaluateBaseline(agg, PROPOSED_BASELINE_THRESHOLDS);
    expect(res.passed).toBe(false);
    const mrrGate = res.perMetric.find((m) => m.metric === "mrr")!;
    expect(mrrGate.passed).toBe(false);
    expect(mrrGate.value).toBe(0.1);
    expect(mrrGate.threshold).toBe(PROPOSED_BASELINE_THRESHOLDS.mrr);
  });

  it("value exactly equal to threshold passes (>=)", () => {
    const agg: RetrievalAggregate = {
      k: 5,
      queryCount: 1,
      meanRecallAtK: PROPOSED_BASELINE_THRESHOLDS.meanRecallAtK,
      meanPrecisionAtK: PROPOSED_BASELINE_THRESHOLDS.meanPrecisionAtK,
      mrr: PROPOSED_BASELINE_THRESHOLDS.mrr,
      meanNdcgAtK: PROPOSED_BASELINE_THRESHOLDS.meanNdcgAtK,
    };
    expect(evaluateBaseline(agg, PROPOSED_BASELINE_THRESHOLDS).passed).toBe(true);
  });
});

describe("Phase 7 (7e) — renderers (deterministic, host-only, masked key)", () => {
  const meta: RetrievalRunMeta = {
    runTimestampUtc: "2026-08-06T00:00:00.000Z",
    embedModel: "gemini-embedding-001",
    embedDim: 768,
    baseUrlHost: "proxy.example.internal",
    maskedKey: "sk-…(redacted)",
    k: 5,
    corpusDocumentCount: 12,
  };
  const scores: QueryScore[] = [
    scoreQuery("q1", ["a"], rel("a"), 5),
    scoreQuery("q2", ["x", "b"], rel("b"), 5),
  ];
  const aggregate = aggregateRetrieval(scores);
  const baseline = evaluateBaseline(aggregate, PROPOSED_BASELINE_THRESHOLDS);

  it("markdown is deterministic and carries no scheme/key body", () => {
    const md1 = renderRetrievalMarkdown(meta, scores, aggregate, baseline);
    const md2 = renderRetrievalMarkdown(meta, scores, aggregate, baseline);
    expect(md1).toBe(md2);
    expect(md1).toContain("proxy.example.internal");
    expect(md1).toContain("sk-…(redacted)");
    expect(md1).not.toContain("https://");
    expect(md1).toContain("PROPOSED");
    expect(md1).toContain("q1");
    expect(md1).toContain("q2");
  });

  it("console summary is deterministic and states the gate", () => {
    const c1 = renderRetrievalConsoleSummary(aggregate, baseline);
    const c2 = renderRetrievalConsoleSummary(aggregate, baseline);
    expect(c1).toBe(c2);
    expect(c1).toMatch(/Baseline gate .*: (PASS|FAIL)/);
  });
});

// ---------------------------------------------------------------------------
// evals/retrieval.jsonl validity + internal consistency (parse-only).
// ---------------------------------------------------------------------------

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "..");

describe("Phase 7 (7e) — evals/retrieval.jsonl is valid and internally consistent", () => {
  it("parses and references only existing corpus source URIs", async () => {
    const jsonl = await readFile(
      path.join(repoRoot, "evals", "retrieval.jsonl"),
      "utf8"
    );
    const items = parseRetrievalDataset(jsonl);
    expect(items.length).toBeGreaterThanOrEqual(10);

    // Build the set of actual corpus source URIs from the committed files.
    const corpusDir = path.join(repoRoot, "knowledge", "corpus");
    const entries = await readdir(corpusDir, { withFileTypes: true });
    const actualSources = new Set(
      entries
        .filter((e) => e.isFile() && e.name.endsWith(".md"))
        .map((e) => `knowledge/corpus/${e.name}`)
    );
    expect(actualSources.size).toBeGreaterThanOrEqual(6);

    // Every referenced relevance key must point at a real corpus document.
    for (const item of items) {
      for (const uri of item.relevantSourceUris) {
        expect(actualSources.has(uri)).toBe(true);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Structural guard: the LIVE runner must NEVER be imported by the offline suite.
// ---------------------------------------------------------------------------

async function collectTestFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) return collectTestFiles(full);
      if (entry.isFile() && entry.name.endsWith(".ts")) return [full];
      return [];
    })
  );
  return files.flat();
}

describe("Phase 7 (7e) — run-retrieval-eval entrypoint is not imported by tests", () => {
  it("no test file imports src/run-retrieval-eval.ts", async () => {
    const files = await collectTestFiles(testDir);
    expect(files.length).toBeGreaterThan(0);
    const offenders: string[] = [];
    for (const file of files) {
      const raw = await readFile(file, "utf8");
      if (/run-retrieval-eval(\.js)?["']/.test(raw)) {
        offenders.push(path.relative(testDir, file));
      }
    }
    expect(
      offenders,
      offenders.length === 0
        ? ""
        : `The LIVE runner src/run-retrieval-eval.ts must not be imported by ` +
            `tests (it makes real proxy calls). Offending file(s): ` +
            offenders.join(", ")
    ).toEqual([]);
  });
});
