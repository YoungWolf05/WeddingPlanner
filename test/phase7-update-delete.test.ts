import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
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
  InvalidSourceUriError,
  type KnowledgeStore,
} from "../src/core/knowledge-store.js";
import {
  ingestDocument,
  deleteDocument,
  deleteSource,
  type DocumentEmbedder,
} from "../src/core/ingestion.js";
import { KNOWLEDGE_MIGRATIONS } from "../src/core/knowledge-store.js";

// Phase 7 (7c) — UPDATE / DELETE via SOURCE-ADDRESSED IDENTITY.
//
// Fully OFFLINE + DETERMINISTIC:
//   - The embedder is a deterministic FAKE (hash-seeded vectors). NO model,
//     NO network is ever touched.
//   - Every knowledge DB lives in a per-test temp dir under os.tmpdir() and is
//     removed in afterEach (WAL/-shm sidecars vanish with the temp dir), so
//     `npm test` NEVER writes DB artifacts into the repo working tree.
//   - Clock-sensitive assertions (created_at preserved / updated_at bumped) use
//     Vitest fake timers, so they are deterministic — never wall-clock flaky.

const DIM = 16;

function vectorFor(text: string, dim = DIM): number[] {
  const digest = createHash("sha256").update(text, "utf8").digest();
  return Array.from({ length: dim }, (_, i) => digest[i % digest.length]! / 256);
}

interface SpyEmbedder extends DocumentEmbedder {
  callCount: number;
}
function makeFakeEmbedder(dim = DIM): SpyEmbedder {
  const spy: SpyEmbedder = {
    callCount: 0,
    embedDocuments(texts: string[]): Promise<number[][]> {
      spy.callCount += 1;
      return Promise.resolve(texts.map((t) => vectorFor(t, dim)));
    },
  };
  return spy;
}

// A GATED embedder: both concurrent ingests park at the embed await until the
// test calls release(), deterministically interleaving them at the exact point
// the create-vs-update convergence matters — no real timers, no racy scheduling.
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
// A vec0 row is an ORPHAN if no chunk references its rowid. Zero orphans is the
// core determinism guarantee for update (full chunk replace) and delete.
function countOrphanVectors(store: KnowledgeStore): number {
  return (
    store.db
      .prepare(
        `SELECT COUNT(*) AS n FROM chunk_vectors v
           LEFT JOIN chunks c ON c.vec_rowid = v.rowid
          WHERE c.chunk_id IS NULL`
      )
      .get() as { n: number }
  ).n;
}
function columnsOf(store: KnowledgeStore, table: string): string[] {
  return (
    store.db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>
  ).map((c) => c.name);
}

// Two DIFFERENT contents that both chunk into MANY (and a different number of)
// chunks at a small chunk size — so "old chunks fully gone / new fully present"
// is a meaningful assertion.
const CONTENT_ONE = [
  "Wedding budget overview and planning notes.",
  "Allocate the venue deposit first, then catering per head.",
  "Photography and videography come next, followed by florals.",
].join("\n\n");
const CONTENT_TWO = [
  "Revised wedding budget with an expanded vendor list.",
  "Venue deposit, catering, and a larger photography package.",
  "Add videography, a florist, a band, and a coordinator.",
  "Reserve everything at least eight months before the date.",
  "Reconcile every payment against the master spreadsheet weekly.",
].join("\n\n");
const CHUNKING = { chunkSize: 60, chunkOverlap: 15 };

describe("Phase 7 (7c) — update / delete semantics", () => {
  let tempDir: string;
  let dbPath: string;
  const openStores: KnowledgeStore[] = [];

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-update-delete-"));
    dbPath = path.join(tempDir, "knowledge.sqlite");
    openStores.length = 0;
  });

  afterEach(async () => {
    // Restore real timers FIRST so the fs cleanup below is never under fake time.
    vi.useRealTimers();
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

  it("UPDATE determinism (crit 2): created → unchanged (no re-embed, updated_at unchanged) → updated (created_at preserved, updated_at bumped, chunks/vectors fully replaced, zero orphans)", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();
    vi.useFakeTimers();

    // (1) CREATE at t=1000.
    vi.setSystemTime(1000);
    const created = await ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://guide",
      chunking: CHUNKING,
    });
    expect(created.status).toBe("created");
    const n1 = created.chunkCount;
    expect(n1).toBeGreaterThan(1);
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(n1);
    expect(countVectors(store)).toBe(n1);
    const docId = created.documentId;
    expect(docId).toBe(computeDocumentId("kb://guide"));
    const oldChunkIds = store.listChunks(docId).map((c) => c.chunkId);
    const embedCallsAfterCreate = embedder.callCount; // 1

    // (2) UNCHANGED re-ingest at t=2000: same source + same content → no-op. The
    // embedder is NOT called again and updated_at does NOT move (no write ran),
    // proving the clock advancing did not touch the row.
    vi.setSystemTime(2000);
    const unchanged = await ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://guide",
      chunking: CHUNKING,
    });
    expect(unchanged.status).toBe("unchanged");
    expect(embedder.callCount).toBe(embedCallsAfterCreate); // not re-embedded
    const afterUnchanged = store.getDocument(docId)!;
    expect(afterUnchanged.createdAt).toBe(1000);
    expect(afterUnchanged.updatedAt).toBe(1000); // NOT bumped to 2000

    // (3) UPDATE at t=3000: same source, DIFFERENT content → in-place update.
    vi.setSystemTime(3000);
    const updated = await ingestDocument({
      store,
      embedder,
      content: CONTENT_TWO,
      sourceUri: "kb://guide",
      chunking: CHUNKING,
    });
    expect(updated.status).toBe("updated");
    expect(updated.documentId).toBe(docId); // SAME document_id
    expect(embedder.callCount).toBe(embedCallsAfterCreate + 1); // re-embedded once
    const n2 = updated.chunkCount;

    const doc = store.getDocument(docId)!;
    // content_hash moved to the new version; created_at PRESERVED, updated_at BUMPED.
    expect(doc.contentHash).toBe(computeContentHash(CONTENT_TWO));
    expect(doc.contentHash).not.toBe(computeContentHash(CONTENT_ONE));
    expect(doc.createdAt).toBe(1000);
    expect(doc.updatedAt).toBe(3000);

    // Exactly N2 chunks + N2 vectors; still ONE document; ZERO orphan vectors.
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(n2);
    expect(countVectors(store)).toBe(n2);
    expect(countOrphanVectors(store)).toBe(0);

    // ALL old chunks are GONE (full replace, not a diff).
    for (const oldId of oldChunkIds) {
      expect(store.getChunk(oldId)).toBeNull();
    }
    const newChunks = store.listChunks(docId);
    expect(newChunks).toHaveLength(n2);
    newChunks.forEach((c, i) => {
      expect(c.chunkIndex).toBe(i);
      expect(c.vecRowid).not.toBeNull();
    });

    // Repeating the SAME update converges (idempotent) — now "unchanged", no dups.
    vi.setSystemTime(4000);
    const repeat = await ingestDocument({
      store,
      embedder,
      content: CONTENT_TWO,
      sourceUri: "kb://guide",
      chunking: CHUNKING,
    });
    expect(repeat.status).toBe("unchanged");
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(n2);
    expect(countVectors(store)).toBe(n2);
    expect(countOrphanVectors(store)).toBe(0);
    expect(store.getDocument(docId)!.updatedAt).toBe(3000); // still not re-bumped
  });

  it("DELETE determinism (crit 2): deleteSource removes document + all chunks + all vectors; nonexistent → false, no side effects; re-ingest is a clean 'created'", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();

    const created = await ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://del",
      chunking: CHUNKING,
    });
    expect(created.status).toBe("created");
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(created.chunkCount);
    expect(countVectors(store)).toBe(created.chunkCount);

    // deleteSource removes EVERYTHING for that source, atomically.
    expect(deleteSource(store, "kb://del")).toBe(true);
    expect(countDocuments(store)).toBe(0);
    expect(countChunks(store)).toBe(0);
    expect(countVectors(store)).toBe(0);
    expect(countOrphanVectors(store)).toBe(0);
    expect(store.getDocumentBySource("kb://del")).toBeNull();

    // Deleting a nonexistent source is a clean no-op returning false.
    expect(deleteSource(store, "kb://never")).toBe(false);
    expect(countDocuments(store)).toBe(0);

    // Lifecycle: re-ingesting the SAME source after delete is a fresh 'created'.
    const again = await ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://del",
      chunking: CHUNKING,
    });
    expect(again.status).toBe("created");
    expect(again.documentId).toBe(created.documentId); // same source → same id
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(again.chunkCount);
    expect(countVectors(store)).toBe(again.chunkCount);
  });

  it("DELETE by id: deleteDocument(store, documentId) cascades chunks + vectors; nonexistent id → false; empty source → typed error", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();

    const created = await ingestDocument({
      store,
      embedder,
      content: CONTENT_TWO,
      sourceUri: "kb://byid",
      chunking: CHUNKING,
    });
    expect(deleteDocument(store, created.documentId)).toBe(true);
    expect(countDocuments(store)).toBe(0);
    expect(countChunks(store)).toBe(0);
    expect(countVectors(store)).toBe(0);

    // Nonexistent id → false no-op.
    expect(deleteDocument(store, "no-such-id")).toBe(false);

    // deleteSource with an empty/whitespace source throws the typed error.
    expect(() => deleteSource(store, "   ")).toThrow(InvalidSourceUriError);
  });

  it("EMPTY-CONTENT edge (crit): empty content on a NEW source → skipped, nothing persisted; empty content on an EXISTING source → skipped, existing document/chunks/vectors UNCHANGED (never wiped)", async () => {
    const store = makeStore();
    const embedder = makeFakeEmbedder();

    // Empty on a brand-new source: skipped, nothing written, embedder untouched.
    const newSkip = await ingestDocument({
      store,
      embedder,
      content: "   \n\n  \t ",
      sourceUri: "kb://blank",
      chunking: CHUNKING,
    });
    expect(newSkip.status).toBe("skipped");
    expect(newSkip.chunkCount).toBe(0);
    expect(countDocuments(store)).toBe(0);
    expect(embedder.callCount).toBe(0);

    // Seed a real document, then re-ingest EMPTY content over the SAME source.
    const created = await ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://live",
      chunking: CHUNKING,
    });
    expect(created.status).toBe("created");
    const docsBefore = countDocuments(store);
    const chunksBefore = countChunks(store);
    const vectorsBefore = countVectors(store);
    const contentHashBefore = store.getDocument(created.documentId)!.contentHash;
    const embedCallsBefore = embedder.callCount;

    const overSkip = await ingestDocument({
      store,
      embedder,
      content: "\n\t   \n",
      sourceUri: "kb://live",
      chunking: CHUNKING,
    });
    // Skipped, and it reports the EXISTING (unchanged) chunk count — NOT zero.
    expect(overSkip.status).toBe("skipped");
    expect(overSkip.chunkCount).toBe(created.chunkCount);
    // The existing document, chunks, and vectors are COMPLETELY UNCHANGED.
    expect(countDocuments(store)).toBe(docsBefore);
    expect(countChunks(store)).toBe(chunksBefore);
    expect(countVectors(store)).toBe(vectorsBefore);
    expect(countOrphanVectors(store)).toBe(0);
    expect(store.getDocument(created.documentId)!.contentHash).toBe(contentHashBefore);
    // The embedder was NOT called for the skip.
    expect(embedder.callCount).toBe(embedCallsBefore);
  });

  it("CONCURRENCY: two concurrent ingests of the SAME source with the SAME new content → one 'created' + one 'unchanged', exactly one document, no orphans, no raw error", async () => {
    const store = makeStore();
    const embedder = makeGatedEmbedder();

    const p1 = ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://converge",
      chunking: CHUNKING,
    });
    const p2 = ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://converge",
      chunking: CHUNKING,
    });

    await Promise.resolve();
    embedder.release();
    const [r1, r2] = await Promise.all([p1, p2]);

    // Both embedded (both passed the pre-embed existence check).
    expect(embedder.callCount).toBe(2);
    expect([r1.status, r2.status].sort()).toEqual(["created", "unchanged"]);
    expect(r1.documentId).toBe(r2.documentId);

    // Deterministic converged state: exactly one document, N chunks, N vectors,
    // ZERO orphan vectors.
    const created = r1.status === "created" ? r1 : r2;
    expect(countDocuments(store)).toBe(1);
    expect(countChunks(store)).toBe(created.chunkCount);
    expect(countVectors(store)).toBe(created.chunkCount);
    expect(countOrphanVectors(store)).toBe(0);
  });

  it("CONCURRENCY: two concurrent ingests of the SAME source with DIFFERENT contents → one 'created' + one 'updated', exactly one COMPLETE version persisted, no orphan/duplicate chunks or vectors", async () => {
    const store = makeStore();
    const embedder = makeGatedEmbedder();

    const p1 = ingestDocument({
      store,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://race2",
      chunking: CHUNKING,
    });
    const p2 = ingestDocument({
      store,
      embedder,
      content: CONTENT_TWO,
      sourceUri: "kb://race2",
      chunking: CHUNKING,
    });

    await Promise.resolve();
    embedder.release();
    const [r1, r2] = await Promise.all([p1, p2]);

    expect(embedder.callCount).toBe(2);
    // Serialized on the single connection: the first writer CREATEs, the second
    // re-reads inside its transaction and UPDATEs to its content.
    expect([r1.status, r2.status].sort()).toEqual(["created", "updated"]);
    expect(r1.documentId).toBe(r2.documentId);
    const docId = r1.documentId;

    // Exactly ONE document holding exactly ONE complete version (either C1 or C2,
    // whichever committed last) — never a mix, never orphans/duplicates.
    expect(countDocuments(store)).toBe(1);
    const finalHash = store.getDocument(docId)!.contentHash;
    expect([computeContentHash(CONTENT_ONE), computeContentHash(CONTENT_TWO)]).toContain(
      finalHash
    );
    const persistedChunks = store.listChunks(docId);
    expect(countChunks(store)).toBe(persistedChunks.length);
    expect(countVectors(store)).toBe(persistedChunks.length);
    expect(countOrphanVectors(store)).toBe(0);
    // Every persisted chunk belongs to THIS document (no cross-version leftovers).
    persistedChunks.forEach((c, i) => {
      expect(c.documentId).toBe(docId);
      expect(c.chunkIndex).toBe(i);
      expect(c.vecRowid).not.toBeNull();
    });
  });

  it("MIGRATION / DURABILITY: 7c added NO destructive migration (source_uri stays nullable; app schema version unchanged); a reopened DB reflects prior update AND delete", async () => {
    // No new knowledge migration was introduced for 7c (app-layer enforcement of
    // required source, not a schema rewrite).
    expect(KNOWLEDGE_MIGRATIONS).toHaveLength(1);

    const embedder = makeFakeEmbedder();
    const storeA = makeStore();

    // source_uri column is STILL nullable at the SQL level (additive-only).
    const info = storeA.db
      .prepare(`PRAGMA table_info(documents)`)
      .all() as Array<{ name: string; notnull: number }>;
    const sourceCol = info.find((c) => c.name === "source_uri")!;
    expect(sourceCol.notnull).toBe(0);
    expect(columnsOf(storeA, "documents")).toContain("source_uri");
    expect(storeA.getAppSchemaVersion()).toBe(1);

    // Create then UPDATE one source; create then DELETE another.
    const kept = await ingestDocument({
      store: storeA,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://kept",
      chunking: CHUNKING,
    });
    await ingestDocument({
      store: storeA,
      embedder,
      content: CONTENT_TWO,
      sourceUri: "kb://kept",
      chunking: CHUNKING,
    });
    const removed = await ingestDocument({
      store: storeA,
      embedder,
      content: CONTENT_ONE,
      sourceUri: "kb://removed",
      chunking: CHUNKING,
    });
    expect(deleteSource(storeA, "kb://removed")).toBe(true);
    storeA.close();
    openStores.length = 0;

    // Reopen the SAME file: the update survives, the delete survives.
    const storeB = makeStore();
    expect(storeB.getAppSchemaVersion()).toBe(1);
    const keptDoc = storeB.getDocument(kept.documentId);
    expect(keptDoc).not.toBeNull();
    expect(keptDoc!.contentHash).toBe(computeContentHash(CONTENT_TWO)); // updated
    expect(storeB.getDocument(removed.documentId)).toBeNull(); // deleted
    expect(countDocuments(storeB)).toBe(1);
    expect(countOrphanVectors(storeB)).toBe(0);
    // Re-ingesting the unchanged kept source is a no-op against the reopened DB.
    const noop = await ingestDocument({
      store: storeB,
      embedder,
      content: CONTENT_TWO,
      sourceUri: "kb://kept",
      chunking: CHUNKING,
    });
    expect(noop.status).toBe("unchanged");
  });
});

describe("Phase 7 (7c) — repo cleanliness guard", () => {
  it("the update/delete test does not create a ./data DB in the repo", async () => {
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
