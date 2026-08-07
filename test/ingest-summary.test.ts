import { describe, it, expect, afterEach } from "vitest";
import { mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readdir, readFile } from "node:fs/promises";
import {
  renderIngestSummary,
  aggregateIngestCounts,
  type IngestSummaryEntry,
} from "../src/core/ingest-summary.js";
import { loadCorpusDocuments } from "../src/core/corpus.js";
import type { IngestResult } from "../src/core/ingestion.js";

// OFFLINE, deterministic coverage for the PURE pieces of the LIVE `npm run ingest`
// entrypoint: the ingestion summary renderer/aggregator (src/core/ingest-summary.ts)
// and the shared corpus loader (src/core/corpus.ts). No network, no ./data, no
// docs writes; the live runner (src/run-ingest.ts) is NOT imported here (a
// structural guard at the bottom enforces that).

// A representative results fixture mixing all four statuses, in a fixed order.
const RESULTS: IngestResult[] = [
  { status: "created", documentId: "d1", chunkCount: 3 },
  { status: "updated", documentId: "d2", chunkCount: 5 },
  { status: "unchanged", documentId: "d3", chunkCount: 2 },
  { status: "skipped", documentId: "d4", chunkCount: 0 },
  { status: "created", documentId: "d5", chunkCount: 1 },
];
const SOURCE_URIS = [
  "knowledge/corpus/a.md",
  "knowledge/corpus/b.md",
  "knowledge/corpus/c.md",
  "knowledge/corpus/d.md",
  "knowledge/corpus/e.md",
];

describe("aggregateIngestCounts — deterministic aggregation", () => {
  it("tallies per-status counts, total documents, and total chunks", () => {
    const entries: IngestSummaryEntry[] = RESULTS.map((result, i) => ({
      sourceUri: SOURCE_URIS[i]!,
      result,
    }));
    const counts = aggregateIngestCounts(entries);
    expect(counts).toEqual({
      created: 2,
      updated: 1,
      unchanged: 1,
      skipped: 1,
      totalDocuments: 5,
      totalChunks: 11, // 3 + 5 + 2 + 0 + 1
    });
  });

  it("returns all-zero counts for an empty run", () => {
    expect(aggregateIngestCounts([])).toEqual({
      created: 0,
      updated: 0,
      unchanged: 0,
      skipped: 0,
      totalDocuments: 0,
      totalChunks: 0,
    });
  });
});

describe("renderIngestSummary — deterministic, no secrets", () => {
  it("renders per-document lines in the given order + a counts block", () => {
    const out = renderIngestSummary(RESULTS, SOURCE_URIS);

    // Per-document lines, in order, with singular/plural chunk wording.
    expect(out).toContain("  knowledge/corpus/a.md -> created (3 chunks)");
    expect(out).toContain("  knowledge/corpus/b.md -> updated (5 chunks)");
    expect(out).toContain("  knowledge/corpus/c.md -> unchanged (2 chunks)");
    expect(out).toContain("  knowledge/corpus/d.md -> skipped (0 chunks)");
    expect(out).toContain("  knowledge/corpus/e.md -> created (1 chunk)");

    // Ordering: a before b before c before d before e.
    const ia = out.indexOf("a.md");
    const ib = out.indexOf("b.md");
    const ic = out.indexOf("c.md");
    const id = out.indexOf("d.md");
    const ie = out.indexOf("e.md");
    expect(ia).toBeGreaterThanOrEqual(0);
    expect(ia).toBeLessThan(ib);
    expect(ib).toBeLessThan(ic);
    expect(ic).toBeLessThan(id);
    expect(id).toBeLessThan(ie);

    // Counts block.
    expect(out).toContain("created  : 2");
    expect(out).toContain("updated  : 1");
    expect(out).toContain("unchanged: 1");
    expect(out).toContain("skipped  : 1");
    expect(out).toContain("documents: 5");
    expect(out).toContain("chunks   : 11");

    // Idempotency messaging is present.
    expect(out.toLowerCase()).toContain("idempotent");
    expect(out).toContain('"unchanged"');
    expect(out).toContain('"updated"');
  });

  it("is fully deterministic (same input -> byte-identical output)", () => {
    expect(renderIngestSummary(RESULTS, SOURCE_URIS)).toBe(
      renderIngestSummary(RESULTS, SOURCE_URIS)
    );
  });

  it("handles an empty run with a clear placeholder + zero counts", () => {
    const out = renderIngestSummary([], []);
    expect(out).toContain("(no corpus documents found)");
    expect(out).toContain("created  : 0");
    expect(out).toContain("documents: 0");
    expect(out).toContain("chunks   : 0");
  });

  it("throws when results and sourceUris lengths differ (call-site bug)", () => {
    expect(() => renderIngestSummary(RESULTS, ["only-one"])).toThrow(
      /same length/
    );
  });

  it("contains no secret-like tokens in its output", () => {
    const out = renderIngestSummary(RESULTS, SOURCE_URIS);
    expect(out).not.toMatch(/sk-[A-Za-z0-9]/);
    expect(out).not.toMatch(/Bearer/i);
  });
});

// --- Shared corpus loader (src/core/corpus.ts) over a TEMP dir ---------------

describe("loadCorpusDocuments — over a temp dir (no repo/./data reads)", () => {
  const tempRoots: string[] = [];

  afterEach(async () => {
    for (const root of tempRoots.splice(0)) {
      await rm(root, { recursive: true, force: true });
    }
  });

  async function makeTempCorpus(): Promise<string> {
    const root = await mkdtemp(path.join(tmpdir(), "wp-ingest-corpus-"));
    tempRoots.push(root);
    return root;
  }

  it("loads only .md files, normalizes sourceUri, and sorts deterministically", async () => {
    const dir = await makeTempCorpus();
    // Intentionally out-of-order filenames + non-.md + a subdirectory to ignore.
    await writeFile(path.join(dir, "venues.md"), "venue content", "utf8");
    await writeFile(path.join(dir, "budget.md"), "budget content", "utf8");
    await writeFile(path.join(dir, "notes.txt"), "should be ignored", "utf8");
    await writeFile(path.join(dir, "README"), "no extension, ignored", "utf8");
    await mkdir(path.join(dir, "sub"), { recursive: true });
    await writeFile(path.join(dir, "sub", "nested.md"), "nested ignored", "utf8");

    const docs = await loadCorpusDocuments(dir);

    // Only the two top-level .md files, sorted by sourceUri (budget < venues).
    expect(docs.map((d) => d.sourceUri)).toEqual([
      "knowledge/corpus/budget.md",
      "knowledge/corpus/venues.md",
    ]);
    expect(docs.map((d) => d.content)).toEqual([
      "budget content",
      "venue content",
    ]);
  });

  it("sourceUri is independent of the absolute temp directory (stable identity)", async () => {
    const dirA = await makeTempCorpus();
    const dirB = await makeTempCorpus();
    await writeFile(path.join(dirA, "x.md"), "same", "utf8");
    await writeFile(path.join(dirB, "x.md"), "same", "utf8");

    const [a] = await loadCorpusDocuments(dirA);
    const [b] = await loadCorpusDocuments(dirB);
    expect(a!.sourceUri).toBe("knowledge/corpus/x.md");
    expect(b!.sourceUri).toBe(a!.sourceUri);
  });

  it("returns [] for an empty corpus directory", async () => {
    const dir = await makeTempCorpus();
    expect(await loadCorpusDocuments(dir)).toEqual([]);
  });
});

// --- Structural guard: the LIVE runner is never imported by the suite --------

const thisFile = fileURLToPath(import.meta.url);
const testDir = path.dirname(thisFile);

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

describe("run-ingest entrypoint is not imported by tests", () => {
  it("no test file imports src/run-ingest.ts", async () => {
    const files = await collectTestFiles(testDir);
    expect(files.length).toBeGreaterThan(0);
    const offenders: string[] = [];
    for (const file of files) {
      const raw = await readFile(file, "utf8");
      if (/run-ingest(\.js)?["']/.test(raw)) {
        offenders.push(path.relative(testDir, file));
      }
    }
    expect(
      offenders,
      offenders.length === 0
        ? ""
        : `The LIVE runner src/run-ingest.ts must not be imported by tests ` +
            `(it makes real proxy calls). Offending file(s): ` +
            offenders.join(", ")
    ).toEqual([]);
  });
});
