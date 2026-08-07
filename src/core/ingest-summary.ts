import type { IngestResult } from "./ingestion.js";

// PURE, I/O-free rendering of a knowledge-ingestion run's results
// (IngestResult[] from ingestDocuments) into a deterministic, human-readable
// summary. This is the offline-unit-testable core of the LIVE `npm run ingest`
// entrypoint (src/run-ingest.ts): the runner does the live embedding + DB I/O
// and delegates ALL result presentation here. No network, no filesystem, no
// secrets — every string is derived solely from the injected results plus the
// caller-supplied source-URI labels.

// The four ingestion outcomes, in the order they are reported in the counts
// block. Kept in lockstep with IngestResult's discriminants.
const STATUS_ORDER = ["created", "updated", "unchanged", "skipped"] as const;
type IngestStatus = (typeof STATUS_ORDER)[number];

// One line of per-document detail. `sourceUri` is the app-owned identity label
// the caller ingested with (the loader's repo-relative path), correlated to a
// result by array position (ingestDocuments preserves input order).
export interface IngestSummaryEntry {
  sourceUri: string;
  result: IngestResult;
}

// Deterministic per-status counts + totals for a run. Exported so callers/tests
// can assert on the aggregation without parsing rendered text.
export interface IngestSummaryCounts {
  created: number;
  updated: number;
  unchanged: number;
  skipped: number;
  totalDocuments: number;
  totalChunks: number;
}

// Aggregate raw per-status counts, total document count, and total chunk count
// from a run's entries. PURE. Totals are simple sums over all entries regardless
// of status (chunkCount is meaningful for every IngestResult variant — see
// ingestion.ts). Exported for direct assertion in tests.
export function aggregateIngestCounts(
  entries: readonly IngestSummaryEntry[]
): IngestSummaryCounts {
  const counts: IngestSummaryCounts = {
    created: 0,
    updated: 0,
    unchanged: 0,
    skipped: 0,
    totalDocuments: entries.length,
    totalChunks: 0,
  };
  for (const entry of entries) {
    const status: IngestStatus = entry.result.status;
    counts[status] += 1;
    counts.totalChunks += entry.result.chunkCount;
  }
  return counts;
}

/**
 * Render a DETERMINISTIC, human-readable ingestion summary from the ordered
 * (sourceUri, IngestResult) entries of a run.
 *
 * The output has three stable sections:
 *   1. A header line.
 *   2. A per-document detail block, ONE line per entry IN THE GIVEN ORDER
 *      (ingestDocuments preserves input order; the loader sorts by sourceUri),
 *      each `  <sourceUri> -> <status> (<n> chunk[s])`.
 *   3. A counts block: per-status tallies (created/updated/unchanged/skipped),
 *      total documents, total chunks, and an IDEMPOTENCY note that makes it
 *      obvious a re-run over an unchanged corpus is safe (all "unchanged", no
 *      re-embed / no duplicates — the Phase 7b/7c guarantee).
 *
 * PURE: no I/O, no clock, no randomness, no secrets. Given the same entries it
 * always returns the same string.
 */
export function renderIngestSummary(
  results: readonly IngestResult[],
  sourceUris: readonly string[]
): string {
  if (results.length !== sourceUris.length) {
    // Programming error at the call site, not user data — surface loudly. This
    // message carries no result/secret content.
    throw new Error(
      `renderIngestSummary: results (${results.length}) and sourceUris ` +
        `(${sourceUris.length}) must be the same length.`
    );
  }
  const entries: IngestSummaryEntry[] = results.map((result, i) => ({
    sourceUri: sourceUris[i]!,
    result,
  }));
  const counts = aggregateIngestCounts(entries);

  const lines: string[] = [];
  lines.push("Knowledge ingestion summary");
  lines.push("---------------------------");

  // Per-document detail (input/sorted order preserved).
  if (entries.length === 0) {
    lines.push("  (no corpus documents found)");
  } else {
    for (const { sourceUri, result } of entries) {
      const chunkWord = result.chunkCount === 1 ? "chunk" : "chunks";
      lines.push(
        `  ${sourceUri} -> ${result.status} (${result.chunkCount} ${chunkWord})`
      );
    }
  }

  lines.push("");
  // Counts block (fixed status order).
  for (const status of STATUS_ORDER) {
    lines.push(`${status.padEnd(9)}: ${counts[status]}`);
  }
  lines.push(`documents: ${counts.totalDocuments}`);
  lines.push(`chunks   : ${counts.totalChunks}`);
  lines.push("");
  lines.push(
    "Ingestion is idempotent: re-running on an unchanged corpus yields all " +
      '"unchanged" (no re-embedding, no duplicates); a changed file yields ' +
      '"updated". Re-runs are safe.'
  );

  return lines.join("\n");
}
