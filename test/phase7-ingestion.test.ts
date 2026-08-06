import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import {
  createKnowledgeStore,
  computeDocumentId,
  computeContentHash,
  computeChunkId,
  type KnowledgeStore,
} from "../src/core/knowledge-store.js";
import {
  ingestDocument,
  ingestDocuments,
  EmbeddingDimensionError,
  EmbeddingCountError,
  type DocumentEmbedder,
} from "../src/core/ingestion.js";

// Phase 7 (7b) — IDEMPOTENT INGESTION.
//
// Fully OFFLINE + DETERMINISTIC:
//   - The embedder is a deterministic FAKE (hash-seeded vectors). NO model,
//     NO network is ever touched.
//   - Every knowledge DB lives in a per-test temp dir under os.tmpdir() and is
//     removed in afterEach (WAL/-shm sidecars vanish with the temp dir), so
//     `npm test` NEVER writes DB artifacts into the repo working tree.

// A small store dimension keeps the fake vectors tiny + fast while exercising
// the exact same code paths as the production 768-dim store.
const DIM = 16;

// Deterministic vector for a given text: SHA-256(text) bytes mapped to [0,1),
// so identical text always yields the identical vector (KNN distance 0), and
// different text yields a different vector. Pure + reproducible.
function vectorFor(text: string, dim = DIM): number[] {
  const digest = createHash("sha256").update(text, "utf8").digest();
  return Array.from({ length: dim }, (_, i) => digest[i % digest.length]! / 256);
}

// A deterministic fake embedder with an observable call counter and per-call
// batch-size log — used to PROVE the embedder is NOT called on an unchanged
// re-ingest (cost + exit-criterion-1 evidence).
interface SpyEmbedder extends DocumentEmbedder {
  callCount: number;
  embeddedTexts: string[];
}
function makeFakeEmbedder(dim = DIM): SpyEmbedder {
  const spy: SpyEmbedder = {
    callCount: 0,
    embeddedTexts: [],
    embedDocuments(texts: string[]): Promise<number[][]> {
      spy.callCount += 1;
      spy.embeddedTexts.push(...texts);
      return Promise.resolve(texts.map((t) => vectorFor(t, dim)));
    },
  };
  return spy;
}

// A broken embedder that returns WRONG-length vectors (dimension guard test).
function makeWrongDimEmbedder(wrongDim: number): DocumentEmbedder {
  return {
    embedDocuments(texts: string[]): Promise<number[][]> {
      return Promise.resolve(texts.map(() => vectorFor("x", wrongDim)));
    },
  };
}

// A broken embedder that returns the WRONG NUMBER of vectors (count guard test).
function makeWrongCountEmbedder(dim = DIM): DocumentEmbedder {
  return {
    embedDocuments(texts: string[]): Promise<number[][]> {
      // Drop the last vector, violating the one-vector-per-chunk contract.
      return Promise.resolve(texts.slice(1).map((t) => vectorFor(t, dim)));
    },
  };
}

// A GATED embedder whose embedDocuments blocks on an awaitable `gate` until the
// test calls `release()`. This lets two concurrent ingestDocument calls both
// reach (and PARK at) the embed `await` — deterministically interleaving them at
// exactly the point that matters for the FIX A constraint-race, with NO real
// timers or racy scheduling. `callCount` proves BOTH calls embedded.
interface GatedEmbedder extends DocumentEmbedder {
  release: () => void;
  callCount: number;
}
function makeGatedEmbedder(dim = DIM): GatedEmbedder {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const embedder: GatedEmbedder = {
    callCount: 0,
    release,
    async embedDocuments(texts: string[]): Promise<number[][]> {
      embedder.callCount += 1;
      await gate;
      return texts.map((t) => vectorFor(t, dim));
    },
  };
  return embedder;
}

function countDocuments(store: KnowledgeStore): number {
  return (store.db.prepare(`SELECT COUNT(*) AS n FROM documents`).get() as { n: number }).n;
}
function countChunks(store: KnowledgeStore): number {
  return (store.db.prepare(`SELECT COUNT(*) AS n FROM chunks`).get() as { n: number }).n;
}
function countVectors(store: KnowledgeStore): number {
  return (store.db.prepare(`SELECT COUNT(*) AS n FROM chunk_vectors`).get() as { n: number }).n;
}

// A long, multi-paragraph body that reliably chunks into MANY chunks at a small
// chunk size — so "no duplicates on re-ingest" is a meaningful assertion.
const LONG_CONTENT = [
  "Wedding budget overview and planning notes.",
  "Allocate the venue deposit first, then catering per head.",
  "Photography and videography come next, followed by florals.",
  "Reserve vendors at least six months before a peak-season date.",
  "Track every payment against the master spreadsheet weekly.",
].join("\n\n");

describe("Phase 7 (7b) — idempotent ingestion", () => {
  let tempDir: string;
  let dbPath: string;
  const openStores: KnowledgeStore[] = [];

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-ingestion-"));
    dbPath = path.join(tempDir, "knowledge.sqlite");
    openStores.length = 0;
  });

  afterEach(async () => {
    for (const store of openStores) {
      try {
        store.close();
      } catch {
        // already closed
      }
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  function makeStore(): KnowledgeStore {
    const store = createKnowledgeStore({ dbPath, embeddingDim: DIM });
    openStores.push(store);
    return store;
  }

  it("FRESH INGEST: inserts exactly one document, N chunks, N vectors; ids match the 7a scheme; vectors linked + dim recorded; KNN round-trips", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();
    const chunking = { chunkSize: 60, chunkOverlap: 15 };

    const result = await ingestDocument({
      store,
      embedder,
      content: LONG_CONTENT,
      sourceUri: "kb://guide",
      chunking,
    });

    expect(result.status).toBe("created");
    expect(result.documentId).toBe(computeDocumentId(LONG_CONTENT));
    expect(result.chunkCount).toBeGreaterThan(1);

    // Exactly one document; N chunks; N vectors.
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(result.chunkCount);
    expect(countVectors(store)).toBe(result.chunkCount);

    const chunks = store.listChunks(result.documentId);
    expect(chunks).toHaveLength(result.chunkCount);
    chunks.forEach((chunk, index) => {
      // chunk_index is contiguous from 0.
      expect(chunk.chunkIndex).toBe(index);
      // chunk_id matches the 7a content-hash scheme exactly.
      expect(chunk.chunkId).toBe(
        computeChunkId(result.documentId, index, computeContentHash(chunk.text))
      );
      // Vector linked + dimension recorded.
      expect(chunk.vecRowid).not.toBeNull();
      expect(chunk.embeddingDim).toBe(DIM);
    });

    // KNN over an inserted vector round-trips to that exact chunk (distance ~0).
    const target = chunks[0]!;
    const hits = store.searchChunksByVector(vectorFor(target.text), 1);
    expect(hits).toHaveLength(1);
    expect(hits[0]!.chunkId).toBe(target.chunkId);
    expect(hits[0]!.distance).toBeCloseTo(0, 5);

    // One embedder call, embedding exactly the N chunk texts.
    expect(embedder.callCount).toBe(1);
    expect(embedder.embeddedTexts).toEqual(chunks.map((c) => c.text));
  });

  it("IDEMPOTENCY (exit criterion 1): re-ingesting UNCHANGED content is a NO-OP — no duplicate docs/chunks/vectors AND the embedder is NOT called again", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();
    const chunking = { chunkSize: 60, chunkOverlap: 15 };

    const first = await ingestDocument({
      store,
      embedder,
      content: LONG_CONTENT,
      sourceUri: "kb://guide",
      chunking,
    });
    expect(first.status).toBe("created");

    const docsAfterFirst = countDocuments(store);
    const chunksAfterFirst = countChunks(store);
    const vectorsAfterFirst = countVectors(store);
    const embedderCallsAfterFirst = embedder.callCount;
    expect(docsAfterFirst).toBe(1);
    expect(embedderCallsAfterFirst).toBe(1);

    // Re-ingest the SAME content (even via a DIFFERENT source_uri).
    const second = await ingestDocument({
      store,
      embedder,
      content: LONG_CONTENT,
      sourceUri: "kb://different-uri",
      chunking,
    });

    // No-op: reported unchanged, same id + chunk count.
    expect(second.status).toBe("unchanged");
    expect(second.documentId).toBe(first.documentId);
    expect(second.chunkCount).toBe(first.chunkCount);

    // Counts are UNCHANGED — no duplicates of any kind.
    expect(countDocuments(store)).toBe(docsAfterFirst);
    expect(countChunks(store)).toBe(chunksAfterFirst);
    expect(countVectors(store)).toBe(vectorsAfterFirst);

    // PROOF the embedder was NOT called again on the unchanged re-ingest.
    expect(embedder.callCount).toBe(embedderCallsAfterFirst);
  });

  it("DISTINCT contents → two documents; identical normalized content from a DIFFERENT source_uri dedupes to ONE (content-addressed identity)", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();

    const a = await ingestDocument({
      store,
      embedder,
      content: "First distinct document.\nWith two lines.",
      sourceUri: "kb://a",
    });
    const b = await ingestDocument({
      store,
      embedder,
      content: "Second, entirely different, body.",
      sourceUri: "kb://b",
    });
    expect(a.status).toBe("created");
    expect(b.status).toBe("created");
    expect(a.documentId).not.toBe(b.documentId);
    expect(countDocuments(store)).toBe(2);

    // DOCUMENTED BEHAVIOR: identity is content-addressed, NOT uri-addressed.
    // Content that is byte-identical AFTER normalization (here a CRLF variant of
    // `a`) from a DIFFERENT uri dedupes to the SAME single document — no third
    // document is created.
    const dedup = await ingestDocument({
      store,
      embedder,
      content: "First distinct document.\r\nWith two lines.",
      sourceUri: "kb://a-mirror",
    });
    // Same content → same id → unchanged no-op.
    expect(dedup.status).toBe("unchanged");
    expect(dedup.documentId).toBe(a.documentId);
    expect(countDocuments(store)).toBe(2);
  });

  it("NEW content yields a NEW document (changed-content-same-URI replacement is deferred to 7c)", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();

    const v1 = await ingestDocument({
      store,
      embedder,
      content: "Venue shortlist version one.",
      sourceUri: "kb://venues",
    });
    // Genuinely NEW content on the SAME uri → a NEW content-addressed document.
    // 7b does NOT replace/delete the prior document (that is 7c); it simply adds
    // the new one. This asserts the 7b scope boundary explicitly.
    const v2 = await ingestDocument({
      store,
      embedder,
      content: "Venue shortlist version TWO, revised.",
      sourceUri: "kb://venues",
    });
    expect(v1.status).toBe("created");
    expect(v2.status).toBe("created");
    expect(v2.documentId).not.toBe(v1.documentId);
    expect(countDocuments(store)).toBe(2);
  });

  it("DIMENSION MISMATCH: a wrong-length embedding → typed EmbeddingDimensionError and NOTHING persisted (transactional rollback / pre-write guard)", async () => {
    const store = makeStore();
    const badEmbedder = makeWrongDimEmbedder(DIM + 4);

    await expect(
      ingestDocument({
        store,
        embedder: badEmbedder,
        content: "Content that will chunk and then fail the dimension guard.",
        sourceUri: "kb://bad",
      })
    ).rejects.toBeInstanceOf(EmbeddingDimensionError);

    // Nothing persisted — no partial document/chunks/vectors.
    expect(countDocuments(store)).toBe(0);
    expect(countChunks(store)).toBe(0);
    expect(countVectors(store)).toBe(0);
  });

  it("DIMENSION MISMATCH error is typed + redaction-safe with structured fields", async () => {
    const err = new EmbeddingDimensionError(768, 512);
    expect(err).toBeInstanceOf(EmbeddingDimensionError);
    expect(err.name).toBe("EmbeddingDimensionError");
    expect(err.expectedDim).toBe(768);
    expect(err.actualDim).toBe(512);
    expect(err.message).toMatch(/768/);
    expect(err.message).toMatch(/512/);
    // Single-line, bounded (redactText collapses whitespace).
    expect(err.message).not.toMatch(/\n/);
  });

  it("COUNT MISMATCH: an embedder returning the wrong number of vectors → typed EmbeddingCountError, nothing persisted", async () => {
    const store = makeStore();
    const badEmbedder = makeWrongCountEmbedder();

    await expect(
      ingestDocument({
        store,
        embedder: badEmbedder,
        content: LONG_CONTENT,
        sourceUri: "kb://count",
        chunking: { chunkSize: 60, chunkOverlap: 15 },
      })
    ).rejects.toBeInstanceOf(EmbeddingCountError);

    expect(countDocuments(store)).toBe(0);
    expect(countChunks(store)).toBe(0);
    expect(countVectors(store)).toBe(0);
  });

  it("WHITESPACE-ONLY / EMPTY content: SKIPPED — nothing persisted, embedder NOT called; re-ingest still skipped (idempotent)", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();

    // Whitespace-only content chunks to ZERO chunks → skipped, nothing written.
    const first = await ingestDocument({
      store,
      embedder,
      content: "   \n\n  \t ",
      sourceUri: "kb://empty",
    });
    expect(first.status).toBe("skipped");
    expect(first.chunkCount).toBe(0);
    expect(first.documentId).toBe(computeDocumentId("   \n\n  \t "));
    // NOTHING persisted: no document, chunk, or vector rows.
    expect(countDocuments(store)).toBe(0);
    expect(countChunks(store)).toBe(0);
    expect(countVectors(store)).toBe(0);
    // Embedder never called — there was no material to embed.
    expect(embedder.callCount).toBe(0);

    // Re-ingesting the SAME empty content is still skipped and still writes zero.
    const second = await ingestDocument({
      store,
      embedder,
      content: "   \n\n  \t ",
      sourceUri: "kb://empty",
    });
    expect(second.status).toBe("skipped");
    expect(embedder.callCount).toBe(0);
    expect(countDocuments(store)).toBe(0);
    expect(countChunks(store)).toBe(0);
    expect(countVectors(store)).toBe(0);

    // A literal empty string is likewise skipped and persists nothing.
    const empty = await ingestDocument({
      store,
      embedder,
      content: "",
      sourceUri: "kb://literally-empty",
    });
    expect(empty.status).toBe("skipped");
    expect(empty.chunkCount).toBe(0);
    expect(countDocuments(store)).toBe(0);
    expect(embedder.callCount).toBe(0);
  });

  it("BATCH ingestDocuments: ingests in order, each independently idempotent", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();

    const results = await ingestDocuments({
      store,
      embedder,
      documents: [
        { content: "Doc one body.", sourceUri: "kb://1" },
        { content: "Doc two body.", sourceUri: "kb://2" },
        // A duplicate of doc one → unchanged no-op.
        { content: "Doc one body.", sourceUri: "kb://1-again" },
      ],
    });

    expect(results.map((r) => r.status)).toEqual([
      "created",
      "created",
      "unchanged",
    ]);
    expect(countDocuments(store)).toBe(2);
    // Two distinct documents were embedded (one call each); the duplicate did
    // not trigger an embed.
    expect(embedder.callCount).toBe(2);
  });

  it("CONCURRENT first-ingest of IDENTICAL new content: one 'created', one 'unchanged' — no duplicate rows, no raw constraint error (FIX A)", async () => {
    const store = makeStore();
    // A GATED embedder forces BOTH ingests to park at the embed await together,
    // so both pass the pre-embed existence check (nothing is persisted yet) and
    // both then attempt the insert — deterministically exercising the race.
    const embedder = makeGatedEmbedder();
    const chunking = { chunkSize: 60, chunkOverlap: 15 };

    const p1 = ingestDocument({
      store,
      embedder,
      content: LONG_CONTENT,
      sourceUri: "kb://writer-a",
      chunking,
    });
    const p2 = ingestDocument({
      store,
      embedder,
      content: LONG_CONTENT,
      sourceUri: "kb://writer-b",
      chunking,
    });

    // Let the event loop run both calls up to (and parked at) the embed await,
    // then release the gate so their continuations resume in FIFO order.
    await Promise.resolve();
    embedder.release();

    // BOTH promises resolve — no raw/unredacted better-sqlite3 error is thrown.
    const [r1, r2] = await Promise.all([p1, p2]);

    // Both embedded (both got past the existence check before either persisted).
    expect(embedder.callCount).toBe(2);

    // Exactly one 'created' and one 'unchanged', in either resolution order.
    const statuses = [r1.status, r2.status].sort();
    expect(statuses).toEqual(["created", "unchanged"]);
    expect(r1.documentId).toBe(r2.documentId);

    const created = r1.status === "created" ? r1 : r2;
    const loser = r1.status === "created" ? r2 : r1;
    // The loser's chunkCount comes from the STORE (source of truth), matching
    // the persisted document exactly.
    expect(loser.chunkCount).toBe(created.chunkCount);

    // Exactly ONE document, N chunks, N vectors — no duplicates from the race.
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(created.chunkCount);
    expect(countVectors(store)).toBe(created.chunkCount);
    expect(store.listChunks(created.documentId)).toHaveLength(created.chunkCount);
  });

  it("ATOMIC ROLLBACK: a fault DURING the write transaction (after pre-write guards) persists ZERO rows (FIX C)", async () => {
    const store = makeStore();
    // Wrap the real store so the pre-write count + dimension guards PASS (the
    // real embedder is untouched), but the SECOND in-transaction insertChunk
    // throws — a fault injected strictly INSIDE the atomic write, after the
    // document row and chunk 0 have already been inserted. This is the only path
    // that genuinely exercises the transaction boundary (the dimension/count
    // guards short-circuit BEFORE the transaction opens, so they never do).
    let insertChunkCalls = 0;
    const faultyStore: KnowledgeStore = {
      ...store,
      insertChunk(input) {
        insertChunkCalls += 1;
        if (insertChunkCalls >= 2) {
          throw new Error("forced mid-transaction insertChunk failure");
        }
        return store.insertChunk(input);
      },
    };

    await expect(
      ingestDocument({
        store: faultyStore,
        embedder: makeFakeEmbedder(),
        content: LONG_CONTENT,
        sourceUri: "kb://rollback",
        chunking: { chunkSize: 60, chunkOverlap: 15 },
      })
    ).rejects.toThrow(/forced mid-transaction/);

    // The fault fired only after >= 2 chunks were attempted (so we truly were
    // mid-transaction, past the document + first-chunk inserts).
    expect(insertChunkCalls).toBeGreaterThanOrEqual(2);

    // Atomic rollback leaves the store completely clean — no partial document,
    // chunks, or vectors survive the aborted transaction.
    expect(countDocuments(store)).toBe(0);
    expect(countChunks(store)).toBe(0);
    expect(countVectors(store)).toBe(0);
  });

  it("DURABILITY: a re-opened store over the SAME file still treats prior content as unchanged (no re-embed)", async () => {
    const embedder = makeFakeEmbedder();
    const content = "Persisted knowledge body for durability.";

    const storeA = makeStore();
    const created = await ingestDocument({
      store: storeA,
      embedder,
      content,
      sourceUri: "kb://persist",
    });
    expect(created.status).toBe("created");
    const callsAfterCreate = embedder.callCount;
    storeA.close();
    openStores.length = 0;

    // Re-open the SAME file in a fresh store; re-ingest is a no-op.
    const storeB = makeStore();
    const again = await ingestDocument({
      store: storeB,
      embedder,
      content,
      sourceUri: "kb://persist",
    });
    expect(again.status).toBe("unchanged");
    expect(again.documentId).toBe(created.documentId);
    expect(embedder.callCount).toBe(callsAfterCreate);
    expect(countDocuments(storeB)).toBe(1);
  });
});

describe("Phase 7 (7b) — repo cleanliness guard", () => {
  it("the ingestion test does not create a ./data DB in the repo", async () => {
    const testDir = path.dirname(fileURLToPath(import.meta.url));
    const repoRoot = path.resolve(testDir, "..");
    const dataDir = path.join(repoRoot, "data");

    if (existsSync(dataDir)) {
      const entries = await readdir(dataDir);
      const dbArtifacts = entries.filter((e) => e.includes("knowledge"));
      expect(dbArtifacts).toEqual([]);
    } else {
      expect(existsSync(dataDir)).toBe(false);
    }
  });
});
