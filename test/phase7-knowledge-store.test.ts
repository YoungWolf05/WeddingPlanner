import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import type { Database as BetterSqlite3Database } from "better-sqlite3";
import {
  createKnowledgeStore,
  computeDocumentId,
  computeContentHash,
  computeChunkId,
  normalizeContent,
  normalizeSourceUri,
  InvalidSourceUriError,
  resolveKnowledgeDbPath,
  KNOWLEDGE_MIGRATIONS,
  DEFAULT_EMBEDDING_DIM,
  type KnowledgeStore,
} from "../src/core/knowledge-store.js";
import { createCheckpointer } from "../src/core/memory.js";
import { createThreadStore } from "../src/core/threads.js";
import { config } from "../src/config.js";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import {
  getAppSchemaVersion,
  runMigrations,
  MIGRATIONS_TABLE,
  type Migration,
} from "../src/core/migrations.js";

// Phase 7 (7a) — PERSISTENCE & SCHEMA FOUNDATION for the knowledge base.
//
// DETERMINISTIC and fully OFFLINE:
//   - No model/network is touched. 7a stores schema + primitives only; embedding
//     vectors used here are synthetic Float32Arrays, never model-generated.
//   - Every SQLite database lives in a per-test temp directory under os.tmpdir()
//     and is removed in afterEach, so `npm test` NEVER writes DB artifacts into
//     the repo working tree.
//
// These tests exercise BOTH the real KNOWLEDGE_MIGRATIONS list (through
// createKnowledgeStore, the production integration point) AND the shared,
// generic runMigrations mechanism (proving Phase 7 reuses Phase 5's forward-only,
// atomic, dedicated-version-table semantics on a DIFFERENT file).

type Db = BetterSqlite3Database;

function tableExists(db: Db, name: string): boolean {
  return (
    db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`
      )
      .get(name) !== undefined
  );
}
function indexExists(db: Db, name: string): boolean {
  return (
    db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?`
      )
      .get(name) !== undefined
  );
}
function columnsOf(db: Db, table: string): string[] {
  return (
    db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>
  ).map((c) => c.name);
}

// A deterministic synthetic embedding: a fixed pseudo-pattern for a given seed,
// so identical seeds produce identical vectors (nearest-neighbour distance 0).
function fakeEmbedding(seed: number, dim = DEFAULT_EMBEDDING_DIM): Float32Array {
  return Float32Array.from({ length: dim }, (_, i) => ((seed + i) % 13) / 13);
}

describe("Phase 7 (7a) — knowledge store schema + migrations", () => {
  let tempDir: string;
  let dbPath: string;
  const openStores: KnowledgeStore[] = [];

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-knowledge-"));
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

  function makeStore(p: string = dbPath): KnowledgeStore {
    const store = createKnowledgeStore({ dbPath: p });
    openStores.push(store);
    return store;
  }

  it("resolveKnowledgeDbPath: resolves file paths to absolute; passes through :memory:", () => {
    expect(path.isAbsolute(resolveKnowledgeDbPath("./data/knowledge.sqlite"))).toBe(true);
    expect(resolveKnowledgeDbPath(":memory:")).toBe(":memory:");
  });

  it("resolveKnowledgeDbPath: honors config.knowledgeDbPath (KNOWLEDGE_DB_PATH) when no arg; explicit arg overrides it", () => {
    // config is the single source of truth (KNOWLEDGE_DB_PATH -> config), mirroring
    // resolveCheckpointDbPath. Set the configured path to a TEMP string and assert
    // resolution honors it when the arg is omitted, and that an explicit arg wins.
    // No DB is opened here — this asserts pure path RESOLUTION only — so no real
    // ./data file is ever created. Capture/restore the field so the test is hermetic.
    const configured = path.join(tempDir, "configured-knowledge.sqlite");
    const explicit = path.join(tempDir, "explicit-knowledge.sqlite");
    const prior = config.knowledgeDbPath;
    try {
      config.knowledgeDbPath = configured;
      // No arg -> configured path (resolved absolute).
      expect(resolveKnowledgeDbPath()).toBe(path.resolve(configured));
      // Explicit arg overrides the configured path.
      expect(resolveKnowledgeDbPath(explicit)).toBe(path.resolve(explicit));

      // When the configured path is unset, resolution falls back to the default.
      config.knowledgeDbPath = undefined;
      expect(resolveKnowledgeDbPath()).toBe(path.resolve("./data/knowledge.sqlite"));
    } finally {
      config.knowledgeDbPath = prior;
    }
  });

  it("FRESH INIT: creates documents/chunks/knowledge_meta + vec table; records app schema version in the dedicated table (NOT user_version)", () => {
    const store = makeStore();
    const db = store.db;

    // App-owned relational schema present.
    expect(tableExists(db, "documents")).toBe(true);
    expect(tableExists(db, "chunks")).toBe(true);
    expect(tableExists(db, "knowledge_meta")).toBe(true);
    expect(indexExists(db, "idx_chunks_document_id")).toBe(true);

    // Exact column sets.
    expect(columnsOf(db, "documents").sort()).toEqual(
      ["document_id", "source_uri", "content_hash", "owner_id", "created_at", "updated_at"].sort()
    );
    expect(columnsOf(db, "chunks").sort()).toEqual(
      [
        "chunk_id",
        "document_id",
        "chunk_index",
        "content_hash",
        "text",
        "embedding_dim",
        "vec_rowid",
        "created_at",
        "updated_at",
      ].sort()
    );

    // The sqlite-vec virtual table exists and reports the expected dimension.
    expect(tableExists(db, "chunk_vectors")).toBe(true);
    expect(store.embeddingDim).toBe(DEFAULT_EMBEDDING_DIM);
    const dimRow = db
      .prepare(`SELECT value FROM knowledge_meta WHERE key = 'embedding_dim'`)
      .get() as { value: string };
    expect(Number(dimRow.value)).toBe(DEFAULT_EMBEDDING_DIM);

    // App schema version recorded in the dedicated table (mirrors Phase 5).
    expect(store.getAppSchemaVersion()).toBe(KNOWLEDGE_MIGRATIONS.length);
    expect(tableExists(db, MIGRATIONS_TABLE)).toBe(true);
    const applied = db
      .prepare(`SELECT version FROM ${MIGRATIONS_TABLE} ORDER BY version`)
      .all() as Array<{ version: number }>;
    expect(applied.map((r) => r.version)).toEqual(
      KNOWLEDGE_MIGRATIONS.map((m) => m.version)
    );

    // Version marker is collision-proof: PRAGMA user_version stays 0.
    expect(db.pragma("user_version", { simple: true })).toBe(0);
  });

  it("IDEMPOTENCY / REOPEN: reopening the same file is a migrator no-op; version unchanged; state intact", () => {
    // Instance A: init + seed a document and a chunk with an embedding.
    const storeA = makeStore();
    const doc = storeA.insertDocument({ content: "hello world", sourceUri: "kb://a" });
    storeA.insertChunk({ documentId: doc.documentId, chunkIndex: 0, text: "hello world", embedding: fakeEmbedding(1) });
    const versionA = storeA.getAppSchemaVersion();
    const rowsA = (
      storeA.db.prepare(`SELECT COUNT(*) AS n FROM ${MIGRATIONS_TABLE}`).get() as { n: number }
    ).n;
    storeA.close();

    // Instance B: reopen the SAME file.
    const storeB = makeStore();
    expect(storeB.getAppSchemaVersion()).toBe(versionA);
    const rowsB = (
      storeB.db.prepare(`SELECT COUNT(*) AS n FROM ${MIGRATIONS_TABLE}`).get() as { n: number }
    ).n;
    expect(rowsB).toBe(rowsA);

    // Persisted state survived the reopen.
    expect(storeB.getDocument(doc.documentId)?.sourceUri).toBe("kb://a");
    expect(storeB.listChunks(doc.documentId)).toHaveLength(1);
  });

  it("SOURCE-ADDRESSED IDENTITY (7c): document_id = f(source_uri), INDEPENDENT of content; content_hash tracks content; normalizeSourceUri (NFC/trim, no case-fold); empty source rejected", () => {
    const store = makeStore();

    // document_id is derived from the SOURCE URI, not the content. Two different
    // contents under the SAME source share ONE document_id.
    const idA = computeDocumentId("kb://identity");
    expect(computeDocumentId("kb://identity")).toBe(idA); // deterministic

    // Insert derives document_id from source_uri and content_hash from content.
    const doc = store.insertDocument({
      content: "line1\r\nline2",
      sourceUri: "kb://identity",
    });
    expect(doc.documentId).toBe(idA);
    expect(doc.contentHash).toBe(computeContentHash("line1\nline2"));
    // document_id (hash of source_uri) and content_hash (hash of content) are now
    // genuinely DIFFERENT values — identity is decoupled from content.
    expect(doc.documentId).not.toBe(doc.contentHash);

    // DIFFERENT source_uri → DIFFERENT document_id (even with identical content).
    expect(computeDocumentId("kb://other")).not.toBe(idA);

    // normalizeSourceUri: NFC + trim, NO case-folding.
    //  - trim: surrounding whitespace does not change identity.
    expect(computeDocumentId("  kb://identity  ")).toBe(idA);
    expect(normalizeSourceUri("  kb://identity  ")).toBe("kb://identity");
    //  - NFC: canonically-equivalent forms map to one identity. "é" as U+00E9
    //    vs "e" + U+0301 combining accent normalize to the SAME id.
    const precomposed = "kb://caf\u00e9";
    const decomposed = "kb://cafe\u0301";
    expect(normalizeSourceUri(decomposed)).toBe(precomposed.normalize("NFC"));
    expect(computeDocumentId(decomposed)).toBe(computeDocumentId(precomposed));
    //  - NO case-fold: case is significant (paths/URIs are case-sensitive).
    expect(computeDocumentId("kb://Identity")).not.toBe(idA);

    // Empty / whitespace-only source is REJECTED with a typed error, at both the
    // pure helper and insertDocument, before any write.
    expect(() => normalizeSourceUri("   ")).toThrow(InvalidSourceUriError);
    expect(() => computeDocumentId("")).toThrow(InvalidSourceUriError);
    expect(() =>
      store.insertDocument({ content: "orphan content", sourceUri: "   " })
    ).toThrow(InvalidSourceUriError);
    // Nothing was persisted by the rejected insert.
    expect(
      (store.db.prepare(`SELECT COUNT(*) AS n FROM documents`).get() as { n: number }).n
    ).toBe(1); // only the "kb://identity" doc above

    // normalizeContent remains the single documented CONTENT normalization.
    expect(normalizeContent("a\r\nb\rc")).toBe("a\nb\nc");
  });

  it("UNIQUE / IDENTITY constraints: duplicate (document_id, chunk_index) is rejected; deterministic chunk_id", () => {
    const store = makeStore();
    const doc = store.insertDocument({ content: "doc for chunks", sourceUri: "kb://chunks" });

    const c0 = store.insertChunk({ documentId: doc.documentId, chunkIndex: 0, text: "chunk A" });
    // chunk_id matches the pure deterministic helper.
    expect(c0.chunkId).toBe(
      computeChunkId(doc.documentId, 0, computeContentHash("chunk A"))
    );

    // Duplicate (document_id, chunk_index) must fail even with different text.
    expect(() =>
      store.insertChunk({ documentId: doc.documentId, chunkIndex: 0, text: "different text" })
    ).toThrow(/UNIQUE constraint failed/i);

    // A different index is fine.
    expect(() =>
      store.insertChunk({ documentId: doc.documentId, chunkIndex: 1, text: "chunk B" })
    ).not.toThrow();
    expect(store.listChunks(doc.documentId).map((c) => c.chunkIndex)).toEqual([0, 1]);
  });

  it("REFERENTIAL INTEGRITY: a chunk for a missing document is rejected (FK on)", () => {
    const store = makeStore();
    expect(() =>
      store.insertChunk({ documentId: "no-such-doc", chunkIndex: 0, text: "orphan" })
    ).toThrow(/FOREIGN KEY constraint failed/i);
  });

  it("DELETE-BY-DOCUMENT: cascades chunks and removes their vectors atomically", () => {
    const store = makeStore();
    const doc = store.insertDocument({ content: "to be deleted", sourceUri: "kb://del" });
    const c0 = store.insertChunk({ documentId: doc.documentId, chunkIndex: 0, text: "a", embedding: fakeEmbedding(2) });
    store.insertChunk({ documentId: doc.documentId, chunkIndex: 1, text: "b", embedding: fakeEmbedding(3) });

    // Vectors present.
    const vecCountBefore = (
      store.db.prepare(`SELECT COUNT(*) AS n FROM chunk_vectors`).get() as { n: number }
    ).n;
    expect(vecCountBefore).toBe(2);
    expect(c0.vecRowid).not.toBeNull();

    // Delete removes the document, its chunks (cascade) and its vectors.
    expect(store.deleteDocument(doc.documentId)).toBe(true);
    expect(store.getDocument(doc.documentId)).toBeNull();
    expect(store.listChunks(doc.documentId)).toHaveLength(0);
    const vecCountAfter = (
      store.db.prepare(`SELECT COUNT(*) AS n FROM chunk_vectors`).get() as { n: number }
    ).n;
    expect(vecCountAfter).toBe(0);

    // Deleting a non-existent document returns false (no-op).
    expect(store.deleteDocument("no-such-doc")).toBe(false);
  });

  it("sqlite-vec: extension loads; a 768-dim vector inserts; a KNN query round-trips", () => {
    const store = makeStore();
    const doc = store.insertDocument({ content: "vector doc", sourceUri: "kb://vec" });

    // Store two chunks with distinct embeddings; one identical to the query.
    const near = fakeEmbedding(5);
    const far = Float32Array.from({ length: DEFAULT_EMBEDDING_DIM }, () => 0.5);
    store.insertChunk({ documentId: doc.documentId, chunkIndex: 0, text: "near", embedding: near });
    store.insertChunk({ documentId: doc.documentId, chunkIndex: 1, text: "far", embedding: far });

    const hits = store.searchChunksByVector(near, 2);
    expect(hits).toHaveLength(2);
    // The exact-match chunk is nearest (distance 0) and ranked first.
    expect(hits[0]!.distance).toBeCloseTo(0, 5);
    expect(hits[0]!.documentId).toBe(doc.documentId);
    // Distances are non-decreasing (KNN ordering).
    expect(hits[1]!.distance).toBeGreaterThan(hits[0]!.distance);
  });

  it("sqlite-vec: a wrong-dimension embedding is rejected (foundation for 7d)", () => {
    const store = makeStore();
    const doc = store.insertDocument({ content: "dim doc", sourceUri: "kb://dim" });
    // Store-level validation rejects a wrong-length embedding before touching the DB.
    expect(() =>
      store.insertChunk({
        documentId: doc.documentId,
        chunkIndex: 0,
        text: "bad dim",
        embedding: new Float32Array(512),
      })
    ).toThrow(/dimension mismatch/i);
    // No chunk row leaked.
    expect(store.listChunks(doc.documentId)).toHaveLength(0);
  });

  it("EMBEDDING-DIM compatibility: reopening with a different dimension fails loudly (7d groundwork)", () => {
    const store = makeStore();
    expect(store.embeddingDim).toBe(768);
    store.close();
    openStores.length = 0; // already closed; avoid double-close in afterEach

    // Reopen the SAME file requesting a different dimension.
    expect(() => createKnowledgeStore({ dbPath, embeddingDim: 1024 })).toThrow(
      /embedding-dimension mismatch/i
    );

    // Reopening with the ORIGINAL dimension is fine.
    const reopened = createKnowledgeStore({ dbPath, embeddingDim: 768 });
    openStores.push(reopened);
    expect(reopened.embeddingDim).toBe(768);
  });

  it("CONFIG-DEFAULTED DIMENSION (7d): the store defaults its dimension from config.embedDim; an explicit override still wins", () => {
    // 7d wires config.embedDim (the single source of truth, LITELLM_EMBED_DIM ->
    // 768 default) into createKnowledgeStore's default so the store dimension,
    // the compatibility expectation, and the probe all read ONE value. Capture/
    // restore config.embedDim so the test is hermetic (mirrors the config.
    // knowledgeDbPath capture/restore pattern above).
    const prior = config.embedDim;
    try {
      // A fresh store with NO explicit embeddingDim picks up config.embedDim.
      config.embedDim = 32;
      const defaulted = createKnowledgeStore({ dbPath });
      openStores.push(defaulted);
      expect(defaulted.embeddingDim).toBe(32);
      const meta = defaulted.db
        .prepare(`SELECT value FROM knowledge_meta WHERE key = 'embedding_dim'`)
        .get() as { value: string };
      expect(Number(meta.value)).toBe(32);

      // An explicit override STILL wins over config.embedDim (back-compat), on a
      // SEPARATE fresh file.
      const otherPath = path.join(tempDir, "override-knowledge.sqlite");
      const overridden = createKnowledgeStore({ dbPath: otherPath, embeddingDim: 24 });
      openStores.push(overridden);
      expect(overridden.embeddingDim).toBe(24);
    } finally {
      config.embedDim = prior;
    }
  });

  it("CONFIG-DEFAULTED DIMENSION (7d): the reopen-mismatch guard still fires against the config default", () => {
    // Build at the default (config.embedDim = 768 in the offline env), then
    // reopen while config.embedDim has changed: the recorded dimension WINS and
    // the mismatch throws loudly — the store enforces compatibility regardless of
    // where the requested dimension comes from.
    const built = makeStore();
    expect(built.embeddingDim).toBe(768);
    built.close();
    openStores.length = 0; // already closed; avoid double-close in afterEach

    const prior = config.embedDim;
    try {
      config.embedDim = 256; // as if the operator changed LITELLM_EMBED_DIM
      expect(() => createKnowledgeStore({ dbPath })).toThrow(
        /embedding-dimension mismatch/i
      );
    } finally {
      config.embedDim = prior;
    }
  });

  it("CUSTOM DIMENSION: a store built at a non-default dimension round-trips a vector at that size", () => {
    const store = createKnowledgeStore({ dbPath, embeddingDim: 16 });
    openStores.push(store);
    expect(store.embeddingDim).toBe(16);
    const doc = store.insertDocument({ content: "small dim", sourceUri: "kb://small" });
    const v = Float32Array.from({ length: 16 }, (_, i) => i / 16);
    store.insertChunk({ documentId: doc.documentId, chunkIndex: 0, text: "x", embedding: v });
    const hits = store.searchChunksByVector(v, 1);
    expect(hits).toHaveLength(1);
    expect(hits[0]!.distance).toBeCloseTo(0, 5);
    // A 768-dim vector is now rejected against this 16-dim store.
    expect(() =>
      store.insertChunk({
        documentId: doc.documentId,
        chunkIndex: 1,
        text: "y",
        embedding: new Float32Array(768),
      })
    ).toThrow(/dimension mismatch/i);
  });

  it("FORWARD-ONLY GUARD: a knowledge DB recorded newer than the code fails loudly and mutates nothing", () => {
    const store = makeStore();
    const db = store.db;

    // Forge a newer recorded version (as if written by a newer app build).
    const future = KNOWLEDGE_MIGRATIONS.length + 5;
    db.prepare(
      `INSERT INTO ${MIGRATIONS_TABLE} (version, applied_at) VALUES (?, ?)`
    ).run(future, Date.now());
    expect(getAppSchemaVersion(db)).toBe(future);

    const schemaBefore = db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY name`
      )
      .all();

    expect(() => runMigrations(db, KNOWLEDGE_MIGRATIONS)).toThrow(
      /newer than this application supports/i
    );

    expect(getAppSchemaVersion(db)).toBe(future);
    const schemaAfter = db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY name`
      )
      .all();
    expect(schemaAfter).toEqual(schemaBefore);
  });

  it("MULTI-STEP + ROLLBACK: the shared runner applies a synthetic knowledge migration and rolls back a throwing step", () => {
    const store = makeStore();
    const db = store.db;
    const base = store.getAppSchemaVersion();

    // A synthetic forward migration appended after the real ones: add an
    // additive nullable column (data-preserving).
    const synthetic: Migration[] = [
      ...KNOWLEDGE_MIGRATIONS,
      {
        version: base + 1,
        up(d) {
          d.exec(`ALTER TABLE documents ADD COLUMN archived INTEGER;`);
        },
      },
    ];
    runMigrations(db, synthetic);
    expect(getAppSchemaVersion(db)).toBe(base + 1);
    expect(columnsOf(db, "documents")).toContain("archived");

    // A throwing step rolls back: partial DDL and version advance both undone.
    const boom = new Error("synthetic knowledge migration failure");
    const throwing: Migration[] = [
      ...synthetic,
      {
        version: base + 2,
        up(d) {
          d.exec(`CREATE TABLE IF NOT EXISTS partial_k (id INTEGER PRIMARY KEY);`);
          throw boom;
        },
      },
    ];
    expect(() => runMigrations(db, throwing)).toThrow(boom);
    expect(getAppSchemaVersion(db)).toBe(base + 1);
    expect(tableExists(db, "partial_k")).toBe(false);
  });

  it("SEPARATION: opening the knowledge store does NOT create conversation checkpoint tables (distinct files)", () => {
    const store = makeStore();
    const db = store.db;
    // The knowledge DB must not carry the conversation library/app tables.
    expect(tableExists(db, "checkpoints")).toBe(false);
    expect(tableExists(db, "writes")).toBe(false);
    expect(tableExists(db, "threads")).toBe(false);
  });

  it("SEPARATION: a conversation checkpoint store does NOT create knowledge tables (distinct files)", () => {
    const checkpointPath = path.join(tempDir, "checkpoints.sqlite");
    const saver = SqliteSaver.fromConnString(checkpointPath);
    try {
      createThreadStore(saver);
      // The conversation DB has its own app + library tables, but NONE of the
      // knowledge tables.
      expect(tableExists(saver.db, "threads")).toBe(true);
      expect(tableExists(saver.db, "documents")).toBe(false);
      expect(tableExists(saver.db, "chunks")).toBe(false);
      expect(tableExists(saver.db, "chunk_vectors")).toBe(false);
      expect(tableExists(saver.db, "knowledge_meta")).toBe(false);
    } finally {
      saver.db.close();
    }

    // And the knowledge store, opened on ITS OWN file, is independent.
    const store = makeStore();
    expect(tableExists(store.db, "documents")).toBe(true);
    expect(tableExists(store.db, "checkpoints")).toBe(false);
  });

  it("SEPARATION (deep): a raw connection to the knowledge file sees only knowledge tables", () => {
    const store = makeStore();
    store.close();
    openStores.length = 0;

    // Independent read-only introspection connection over the same file.
    const raw = new Database(dbPath, { readonly: true });
    try {
      const names = (
        raw
          .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name`)
          .all() as Array<{ name: string }>
      ).map((r) => r.name);
      expect(names).toContain("documents");
      expect(names).toContain("chunks");
      expect(names).toContain("knowledge_meta");
      expect(names).not.toContain("checkpoints");
      expect(names).not.toContain("writes");
      expect(names).not.toContain("threads");
    } finally {
      raw.close();
    }
  });
});

describe("Phase 7 (7a) — repo cleanliness guard", () => {
  it("the knowledge-store test does not create a ./data DB in the repo", async () => {
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
