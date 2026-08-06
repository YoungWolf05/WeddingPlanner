import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
// RUNTIME import of the better-sqlite3 driver. This is a DELIBERATE, sanctioned
// second connection: Phase 7's knowledge base is a SEPARATE database FILE with
// its OWN lifecycle, opened and owned HERE. It is NOT the conversation
// checkpoint connection (that one is constructed only via SqliteSaver in
// memory.ts and shared with threads.ts). The two connections/files never mix.
// The single-factory guard (test/phase4-model-factory.test.ts) governs
// ChatOpenAI/OpenAIEmbeddings construction, not better-sqlite3, so opening our
// own knowledge connection here is within convention.
import Database from "better-sqlite3";
import type { Database as BetterSqlite3Database } from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { config } from "../config.js";
import { redactError, redactText } from "./redaction.js";
// Phase 5 (5e) app-owned schema versioning + forward-migration runner. The
// runner is GENERIC and injectable (it takes the migration list as a parameter),
// so Phase 7 REUSES the exact same forward-only, atomic, dedicated-version-table
// mechanism with its OWN, knowledge-specific migration list — on its OWN file
// and its OWN `app_schema_migrations` bookkeeping table (per-connection). We do
// NOT reuse the conversation DB's APP_MIGRATIONS.
import {
  getAppSchemaVersion,
  runMigrations,
  type Migration,
} from "./migrations.js";

// Phase 7 (7a): PERSISTENCE & SCHEMA FOUNDATION for the knowledge base.
//
// WHAT THIS MODULE OWNS
// ---------------------
// A durable, APP-OWNED knowledge store backed by its OWN SQLite file (default
// ./data/knowledge.sqlite; overridable via KNOWLEDGE_DB_PATH). It establishes:
//   1. The sqlite-vec extension loaded on this connection + a `vec0` virtual
//      table for fixed-dimension float vectors (default 768 dims, matching
//      gemini-embedding-001).
//   2. The app-owned relational schema — `documents`, `chunks`, `knowledge_meta`
//      — created through the Phase-5 versioned, forward-only migration runner.
//   3. Trusted, APP-OWNED identity: `document_id` and `chunk_id` are derived from
//      content hashes computed HERE (never model-generated), so later
//      authorization + trusted citations (exit criterion 3) rest on stable,
//      server-owned identifiers.
//
// WHAT THIS MODULE DOES NOT DO (deferred to 7b-7e)
// ------------------------------------------------
// No chunking policy, no embedding calls, no ingestion orchestration, no
// retrieval/answer pipeline, no eval. This is schema + low-level storage
// primitives ONLY. Accessors here are the minimum needed for 7a to be testable
// and for 7b (idempotent ingestion) to build on.

// The concrete better-sqlite3 Database type. Aliased like threads.ts/migrations.ts.
type Db = BetterSqlite3Database;

// Default knowledge DB path when KNOWLEDGE_DB_PATH is unset. Relative to the
// process working directory; lands under the gitignored ./data directory,
// exactly like the checkpoint DB. The parent directory is created on first use.
export const DEFAULT_KNOWLEDGE_DB_PATH = "./data/knowledge.sqlite";

// Default embedding dimension. 768 matches the verified gemini-embedding-001
// alias (docs/capabilities/2026-07-28.md). Configurable so 7d can do
// compatibility checks; the value is recorded in the DB (see knowledge_meta).
export const DEFAULT_EMBEDDING_DIM = 768;

// The fixed-dimension vector table (sqlite-vec `vec0`). Its shadow tables are
// library-owned, so — mirroring how threads.ts keeps the library's checkpoint
// tables OUTSIDE the app migration runner (ensureCheckpointTables) — this vec0
// table is ensured OUTSIDE runMigrations too. The APP-owned relational tables go
// through the migration runner; this virtual table is created idempotently.
const VECTOR_TABLE = "chunk_vectors";
const DOCUMENTS_TABLE = "documents";
const CHUNKS_TABLE = "chunks";
const META_TABLE = "knowledge_meta";
const META_EMBEDDING_DIM_KEY = "embedding_dim";

// Knowledge-base APP migrations. A STATIC, ordered, forward-only list in the
// SAME shape as APP_MIGRATIONS (migrations.ts) but for the KNOWLEDGE file. Append
// new steps with the next integer version; never renumber or mutate a released
// step. Migration 1 creates the app-owned relational schema. It is deliberately
// dimension-INDEPENDENT — the vector table's dimension lives in the vec0 DDL and
// is recorded in knowledge_meta (see ensureVectorTable), not encoded in a
// relational migration.
export const KNOWLEDGE_MIGRATIONS: Migration[] = [
  {
    version: 1,
    up(db) {
      // App-owned key/value metadata (e.g. the vector dimension the DB was built
      // with, so 7d can reject an incompatible embedding model).
      db.exec(`
CREATE TABLE IF NOT EXISTS ${META_TABLE} (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);`);
      // documents: one row per ingested source. document_id is the app-owned
      // stable identity = the content hash of the normalized source content
      // (see computeDocumentId). owner_id is nullable now but present so later
      // authorization + trusted citations attach WITHOUT a schema change and
      // WITHOUT ever trusting a model-generated identifier.
      db.exec(`
CREATE TABLE IF NOT EXISTS ${DOCUMENTS_TABLE} (
  document_id TEXT PRIMARY KEY,
  source_uri TEXT,
  content_hash TEXT NOT NULL,
  owner_id TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);`);
      // chunks: deterministic chunk_id (see computeChunkId), FK to its document
      // with ON DELETE CASCADE (delete-by-document removes chunks atomically),
      // and a UNIQUE(document_id, chunk_index) constraint that makes a duplicate
      // (document, index) impossible — the foundation for 7b idempotency and 7c
      // delete semantics. embedding_dim records the vector dimension used for
      // this chunk; vec_rowid links to the vec0 virtual table rowid (nullable
      // until an embedding is stored in 7b).
      db.exec(`
CREATE TABLE IF NOT EXISTS ${CHUNKS_TABLE} (
  chunk_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  text TEXT NOT NULL,
  embedding_dim INTEGER NOT NULL,
  vec_rowid INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE (document_id, chunk_index),
  FOREIGN KEY (document_id) REFERENCES ${DOCUMENTS_TABLE} (document_id) ON DELETE CASCADE
);`);
      db.exec(`
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
  ON ${CHUNKS_TABLE} (document_id);`);
    },
  },
];

// ---------------------------------------------------------------------------
// APP-OWNED IDENTITY (content hashing). All identifiers are derived HERE from
// content — never generated by a model — so trusted citations/authorization rest
// on stable, server-owned IDs (exit criterion 3 groundwork).
// ---------------------------------------------------------------------------

// Normalize source content BEFORE hashing so trivial, non-semantic differences
// (line-ending style, unstable Unicode composition) do NOT change identity. This
// is the single, documented normalization used for BOTH document and chunk
// content hashes. Kept intentionally conservative: normalize Unicode to NFC and
// collapse CRLF/CR to LF. It does NOT trim or reflow, so meaningful whitespace is
// preserved. Exported for 7b to reuse the EXACT same normalization.
export function normalizeContent(content: string): string {
  return content.normalize("NFC").replace(/\r\n?/g, "\n");
}

// SHA-256 hex of the normalized content. The lowercase hex digest is the stable
// content hash used as both `documents.content_hash` and the basis of the
// document_id.
export function computeContentHash(content: string): string {
  return createHash("sha256").update(normalizeContent(content), "utf8").digest("hex");
}

// The app-owned stable document identity = the content hash of the normalized
// source content. Identical normalized content ALWAYS yields the same
// document_id (deterministic), which is what makes 7b re-ingestion idempotent.
//
// DESIGN DECISION (content-addressed identity): a document's identity is
// CONTENT-ADDRESSED — document_id == content_hash == sha256(normalized content).
// `source_uri` is METADATA ONLY and is deliberately NOT part of identity, so two
// sources with byte-for-byte identical normalized content dedupe to ONE document
// by design (7b relies on this). Recorded here so 7c does not rediscover it.
export function computeDocumentId(content: string): string {
  return computeContentHash(content);
}

// Deterministic chunk identity = SHA-256 of (document_id : chunk_index :
// chunk_content_hash). Binding all three means a chunk's id is stable for
// identical content at a fixed position, and a change to any of them yields a
// new id. chunk_content_hash is the content hash of the chunk text.
export function computeChunkId(
  documentId: string,
  chunkIndex: number,
  chunkContentHash: string
): string {
  return createHash("sha256")
    .update(`${documentId}:${chunkIndex}:${chunkContentHash}`, "utf8")
    .digest("hex");
}

// ---------------------------------------------------------------------------
// Public record + input types.
// ---------------------------------------------------------------------------

// A stored knowledge document (source-level identity + trusted metadata).
export interface KnowledgeDocument {
  documentId: string;
  sourceUri: string | null;
  contentHash: string;
  ownerId: string | null;
  createdAt: number;
  updatedAt: number;
}

// A stored knowledge chunk. `vecRowid` links to the vec0 virtual-table rowid
// (null when no embedding has been stored yet).
export interface KnowledgeChunk {
  chunkId: string;
  documentId: string;
  chunkIndex: number;
  contentHash: string;
  text: string;
  embeddingDim: number;
  vecRowid: number | null;
  createdAt: number;
  updatedAt: number;
}

// Input to insertDocument. `content` is the raw source content: the store
// derives the app-owned document_id + content_hash from it (never trusts a
// caller/model id). `ownerId` is optional (authorization-ready, nullable now).
export interface InsertDocumentInput {
  content: string;
  sourceUri?: string | null;
  ownerId?: string | null;
}

// Input to insertChunk. The store derives chunk_id + content_hash from
// (documentId, chunkIndex, text). `embedding`, when provided, is stored in the
// vec0 table and linked via vec_rowid; when omitted, vec_rowid stays null (7b
// populates embeddings). Length MUST equal the store's embeddingDim.
export interface InsertChunkInput {
  documentId: string;
  chunkIndex: number;
  text: string;
  embedding?: Float32Array | number[];
}

// One nearest-neighbor hit from the low-level vector search primitive.
export interface VectorSearchHit {
  chunkId: string;
  documentId: string;
  distance: number;
}

// The knowledge store contract. Storage-agnostic in spirit (a future pgvector
// backend would implement the same surface), though 7a is SQLite/sqlite-vec.
export interface KnowledgeStore {
  // The concrete better-sqlite3 handle this store owns. Exposed (like
  // SqliteSaver.db) for schema introspection and future low-level needs; callers
  // should prefer the typed primitives below.
  readonly db: Db;
  // The vector dimension this DB was built with (recorded in knowledge_meta).
  readonly embeddingDim: number;
  // Current app schema version of the KNOWLEDGE DB (passthrough to the shared
  // migrations reader), for tests/operators.
  getAppSchemaVersion(): number;
  // Insert a document. Derives document_id + content_hash from `content`. Throws
  // on a duplicate document_id (a storage primitive; idempotent UPSERT is 7b).
  insertDocument(input: InsertDocumentInput): KnowledgeDocument;
  // Fetch a document by its app-owned id, or null if absent.
  getDocument(documentId: string): KnowledgeDocument | null;
  // Hard-delete a document and (via ON DELETE CASCADE + vec cleanup) all of its
  // chunks and their vectors, atomically. Returns true if a row was removed.
  deleteDocument(documentId: string): boolean;
  // Insert a chunk (and its optional embedding vector), atomically linking the
  // vec rowid. Throws on a duplicate (document_id, chunk_index) or an
  // embedding-length mismatch.
  insertChunk(input: InsertChunkInput): KnowledgeChunk;
  // Fetch a chunk by its deterministic id, or null.
  getChunk(chunkId: string): KnowledgeChunk | null;
  // List a document's chunks in chunk_index order.
  listChunks(documentId: string): KnowledgeChunk[];
  // LOW-LEVEL vector KNN primitive (NOT the 7d retriever): return up to `k`
  // nearest chunks to `embedding` by L2 distance. No ranking policy, no
  // embedding calls — just the storage-layer nearest-neighbor query 7d builds on.
  searchChunksByVector(embedding: Float32Array | number[], k: number): VectorSearchHit[];
  // Close the underlying connection (release the file handle/lock).
  close(): void;
}

// Options for createKnowledgeStore. `dbPath` is injectable for tests (temp
// paths), mirroring createCheckpointer(dbPath?). `embeddingDim` overrides the
// default 768 (must match the DB's recorded dimension on reopen).
export interface KnowledgeStoreOptions {
  dbPath?: string;
  embeddingDim?: number;
}

// Raw row shapes as returned by better-sqlite3.
interface DocumentRow {
  document_id: string;
  source_uri: string | null;
  content_hash: string;
  owner_id: string | null;
  created_at: number;
  updated_at: number;
}
interface ChunkRow {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  content_hash: string;
  text: string;
  embedding_dim: number;
  vec_rowid: number | null;
  created_at: number;
  updated_at: number;
}

function rowToDocument(row: DocumentRow): KnowledgeDocument {
  return {
    documentId: row.document_id,
    sourceUri: row.source_uri,
    contentHash: row.content_hash,
    ownerId: row.owner_id,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}
function rowToChunk(row: ChunkRow): KnowledgeChunk {
  return {
    chunkId: row.chunk_id,
    documentId: row.document_id,
    chunkIndex: row.chunk_index,
    contentHash: row.content_hash,
    text: row.text,
    embeddingDim: row.embedding_dim,
    vecRowid: row.vec_rowid,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

// Resolve the knowledge DB path to an absolute path. Precedence mirrors
// resolveCheckpointDbPath in memory.ts exactly: an explicit `dbPath` argument
// (used by tests/injection) overrides the configured KNOWLEDGE_DB_PATH env
// (config.knowledgeDbPath), which in turn overrides the documented default
// (DEFAULT_KNOWLEDGE_DB_PATH). The special in-memory token ":memory:" is passed
// through unresolved (resolve() would otherwise turn it into a bogus filesystem
// path). Kept pure for tests.
export function resolveKnowledgeDbPath(dbPath?: string): string {
  const raw = dbPath ?? config.knowledgeDbPath ?? DEFAULT_KNOWLEDGE_DB_PATH;
  return raw === ":memory:" ? raw : resolve(raw);
}

// Encode an embedding as a little-endian float32 blob for the vec0 table,
// validating its length against the store's dimension first. Copies into a fresh
// buffer so a subarray's byteOffset can never corrupt the bytes.
function encodeEmbedding(
  embedding: Float32Array | number[],
  expectedDim: number
): Buffer {
  const f32 =
    embedding instanceof Float32Array ? embedding : Float32Array.from(embedding);
  if (f32.length !== expectedDim) {
    throw new Error(
      redactText(
        `Embedding dimension mismatch: expected ${expectedDim} but received ` +
          `${f32.length}. The knowledge base is fixed at ${expectedDim} dimensions.`
      )
    );
  }
  return Buffer.from(f32.buffer.slice(f32.byteOffset, f32.byteOffset + f32.byteLength));
}

// Load the sqlite-vec extension on our OWN connection. Wrapped so a load failure
// (e.g. a missing platform package) surfaces as a loud, REDACTED error rather
// than a raw driver message — this is the documented Windows environment risk.
function loadVectorExtension(db: Db): void {
  try {
    sqliteVec.load(db);
  } catch (err) {
    throw new Error(
      `Failed to load the sqlite-vec extension on the knowledge database. ` +
        `Ensure the sqlite-vec package and its platform binary are installed. ` +
        `Cause: ${redactError(err)}`
    );
  }
}

// Ensure the app-recorded embedding dimension and the vec0 virtual table exist
// and AGREE with the requested dimension. On a fresh DB the requested dimension
// is recorded and the vec0 table is created with it. On reopen, the recorded
// dimension WINS and a mismatch throws loudly (foundation for 7d compatibility
// checks) — the vec0 table's dimension is immutable once created.
function ensureVectorSchema(db: Db, requestedDim: number): number {
  const existing = db
    .prepare(`SELECT value FROM ${META_TABLE} WHERE key = ?`)
    .get(META_EMBEDDING_DIM_KEY) as { value: string } | undefined;

  let effectiveDim: number;
  if (existing === undefined) {
    db.prepare(`INSERT INTO ${META_TABLE} (key, value) VALUES (?, ?)`).run(
      META_EMBEDDING_DIM_KEY,
      String(requestedDim)
    );
    effectiveDim = requestedDim;
  } else {
    effectiveDim = Number(existing.value);
    if (effectiveDim !== requestedDim) {
      throw new Error(
        redactText(
          `Knowledge base embedding-dimension mismatch: the database was built ` +
            `with ${effectiveDim} dimensions but was opened requesting ` +
            `${requestedDim}. Reopen with the original dimension or rebuild the store.`
        )
      );
    }
  }
  // vec0 shadow tables are library-owned; create the virtual table IF NOT
  // EXISTS, mirroring ensureCheckpointTables' out-of-migration approach.
  db.exec(
    `CREATE VIRTUAL TABLE IF NOT EXISTS ${VECTOR_TABLE} USING vec0(embedding float[${effectiveDim}]);`
  );
  return effectiveDim;
}

// Build a durable knowledge store over its OWN better-sqlite3 connection.
//
// Lifecycle mirrors createCheckpointer: the parent directory is created (the
// driver will not open a DB in a missing directory), the connection is opened
// eagerly, the sqlite-vec extension is loaded, foreign keys are enabled, the
// app schema is migrated, and the vector schema is ensured. Callers that create
// short-lived stores (tests) MUST call close() to release the file handle/lock
// (important on Windows).
export function createKnowledgeStore(
  options: KnowledgeStoreOptions = {}
): KnowledgeStore {
  const requestedDim = options.embeddingDim ?? DEFAULT_EMBEDDING_DIM;
  if (!Number.isInteger(requestedDim) || requestedDim <= 0) {
    throw new Error(
      `Invalid embedding dimension ${String(requestedDim)}: must be a positive integer.`
    );
  }
  const absolutePath = resolveKnowledgeDbPath(options.dbPath);
  if (absolutePath !== ":memory:") {
    mkdirSync(dirname(absolutePath), { recursive: true });
  }

  const db: Db = new Database(absolutePath);
  // If ANY setup step below throws (extension load, migration, dimension
  // mismatch), close the freshly-opened connection before rethrowing so a failed
  // construction never leaks a file handle / lock (critical on Windows, where a
  // held handle blocks even deleting the temp DB).
  let embeddingDim: number;
  try {
    // Enforce referential integrity (SQLite defaults foreign_keys OFF). Must be
    // set per-connection and OUTSIDE a transaction. This powers the ON DELETE
    // CASCADE from documents -> chunks.
    db.pragma("foreign_keys = ON");
    // Enable WAL for read-concurrency/durability parity with the conversation
    // checkpoint connection (SqliteSaver sets journal_mode=WAL there). WAL is a
    // per-connection-persisted, file-level journal mode and is meaningless for a
    // ":memory:" database (SQLite silently keeps such a DB in "memory" mode), so
    // only request it for on-disk files. The *.sqlite-wal/*.sqlite-shm sidecars
    // are already gitignored.
    if (absolutePath !== ":memory:") {
      db.pragma("journal_mode = WAL");
    }
    // Load the vector extension BEFORE any vec0 DDL/DML.
    loadVectorExtension(db);

    // App-owned relational schema via the SHARED, generic, forward-only
    // migration runner (its own file + its own app_schema_migrations table on
    // this connection). Then ensure the (library-owned) vec0 table + recorded
    // dimension.
    runMigrations(db, KNOWLEDGE_MIGRATIONS);
    embeddingDim = ensureVectorSchema(db, requestedDim);
  } catch (err) {
    db.close();
    throw err;
  }

  // Prepared statements (compiled once).
  const insertDocumentStmt = db.prepare(
    `INSERT INTO ${DOCUMENTS_TABLE}
       (document_id, source_uri, content_hash, owner_id, created_at, updated_at)
     VALUES (@document_id, @source_uri, @content_hash, @owner_id, @created_at, @updated_at)`
  );
  const getDocumentStmt = db.prepare<[documentId: string], DocumentRow>(
    `SELECT document_id, source_uri, content_hash, owner_id, created_at, updated_at
       FROM ${DOCUMENTS_TABLE} WHERE document_id = ?`
  );
  const deleteDocumentStmt = db.prepare(
    `DELETE FROM ${DOCUMENTS_TABLE} WHERE document_id = ?`
  );
  const insertChunkStmt = db.prepare(
    `INSERT INTO ${CHUNKS_TABLE}
       (chunk_id, document_id, chunk_index, content_hash, text, embedding_dim,
        vec_rowid, created_at, updated_at)
     VALUES (@chunk_id, @document_id, @chunk_index, @content_hash, @text,
             @embedding_dim, @vec_rowid, @created_at, @updated_at)`
  );
  const getChunkStmt = db.prepare<[chunkId: string], ChunkRow>(
    `SELECT chunk_id, document_id, chunk_index, content_hash, text, embedding_dim,
            vec_rowid, created_at, updated_at
       FROM ${CHUNKS_TABLE} WHERE chunk_id = ?`
  );
  const listChunksStmt = db.prepare<[documentId: string], ChunkRow>(
    `SELECT chunk_id, document_id, chunk_index, content_hash, text, embedding_dim,
            vec_rowid, created_at, updated_at
       FROM ${CHUNKS_TABLE} WHERE document_id = ? ORDER BY chunk_index ASC`
  );
  const updateChunkVecRowidStmt = db.prepare(
    `UPDATE ${CHUNKS_TABLE} SET vec_rowid = @vec_rowid WHERE chunk_id = @chunk_id`
  );
  const insertVectorStmt = db.prepare(
    `INSERT INTO ${VECTOR_TABLE} (rowid, embedding) VALUES (?, ?)`
  );
  const deleteVectorStmt = db.prepare(
    `DELETE FROM ${VECTOR_TABLE} WHERE rowid = ?`
  );
  // Collect a document's chunk vec rowids so a document delete also removes its
  // vectors from the vec0 table (no ON DELETE CASCADE crosses into a virtual
  // table, so we clean the vectors explicitly inside the same transaction).
  const selectVecRowidsForDocumentStmt = db.prepare<
    [documentId: string],
    { vec_rowid: number }
  >(
    `SELECT vec_rowid FROM ${CHUNKS_TABLE}
       WHERE document_id = ? AND vec_rowid IS NOT NULL`
  );
  const searchStmt = db.prepare<
    [embedding: Buffer, k: number],
    { chunkId: string; documentId: string; distance: number }
  >(
    `SELECT c.chunk_id AS chunkId, c.document_id AS documentId, v.distance AS distance
       FROM ${VECTOR_TABLE} v
       JOIN ${CHUNKS_TABLE} c ON c.vec_rowid = v.rowid
      WHERE v.embedding MATCH ? AND k = ?
      ORDER BY v.distance`
  );

  return {
    db,
    embeddingDim,

    getAppSchemaVersion() {
      return getAppSchemaVersion(db);
    },

    insertDocument(input) {
      const now = Date.now();
      const documentId = computeDocumentId(input.content);
      const contentHash = computeContentHash(input.content);
      const doc: KnowledgeDocument = {
        documentId,
        sourceUri: input.sourceUri ?? null,
        contentHash,
        ownerId: input.ownerId ?? null,
        createdAt: now,
        updatedAt: now,
      };
      insertDocumentStmt.run({
        document_id: doc.documentId,
        source_uri: doc.sourceUri,
        content_hash: doc.contentHash,
        owner_id: doc.ownerId,
        created_at: doc.createdAt,
        updated_at: doc.updatedAt,
      });
      return doc;
    },

    getDocument(documentId) {
      const row = getDocumentStmt.get(documentId);
      return row ? rowToDocument(row) : null;
    },

    deleteDocument(documentId) {
      // Atomic hard delete: remove the document row (chunks vanish via ON DELETE
      // CASCADE) AND explicitly delete each chunk's vector from the vec0 table
      // (a virtual table is not reached by a relational FK cascade). All in one
      // transaction so no orphaned vectors can remain.
      const deleted = db.transaction(() => {
        const vecRows = selectVecRowidsForDocumentStmt.all(documentId);
        const info = deleteDocumentStmt.run(documentId);
        if (info.changes === 0) return false;
        for (const { vec_rowid } of vecRows) {
          deleteVectorStmt.run(BigInt(vec_rowid));
        }
        return true;
      })();
      return deleted;
    },

    insertChunk(input) {
      const now = Date.now();
      const contentHash = computeContentHash(input.text);
      const chunkId = computeChunkId(input.documentId, input.chunkIndex, contentHash);
      // Validate the embedding length BEFORE opening the transaction so a bad
      // input fails cleanly without touching the DB.
      const encoded =
        input.embedding === undefined
          ? undefined
          : encodeEmbedding(input.embedding, embeddingDim);

      const chunk: KnowledgeChunk = {
        chunkId,
        documentId: input.documentId,
        chunkIndex: input.chunkIndex,
        contentHash,
        text: input.text,
        embeddingDim,
        vecRowid: null,
        createdAt: now,
        updatedAt: now,
      };

      db.transaction(() => {
        const info = insertChunkStmt.run({
          chunk_id: chunk.chunkId,
          document_id: chunk.documentId,
          chunk_index: chunk.chunkIndex,
          content_hash: chunk.contentHash,
          text: chunk.text,
          embedding_dim: chunk.embeddingDim,
          vec_rowid: null,
          created_at: chunk.createdAt,
          updated_at: chunk.updatedAt,
        });
        if (encoded !== undefined) {
          // Use the chunk's own integer rowid as the vec0 rowid: a clean 1:1
          // link. vec0 requires a genuine integer rowid, which better-sqlite3
          // only guarantees when bound as a BigInt.
          const rowid = info.lastInsertRowid;
          const rowidBig = typeof rowid === "bigint" ? rowid : BigInt(rowid);
          insertVectorStmt.run(rowidBig, encoded);
          updateChunkVecRowidStmt.run({
            vec_rowid: Number(rowidBig),
            chunk_id: chunk.chunkId,
          });
          chunk.vecRowid = Number(rowidBig);
        }
      })();

      return chunk;
    },

    getChunk(chunkId) {
      const row = getChunkStmt.get(chunkId);
      return row ? rowToChunk(row) : null;
    },

    listChunks(documentId) {
      return listChunksStmt.all(documentId).map(rowToChunk);
    },

    searchChunksByVector(embedding, k) {
      const encoded = encodeEmbedding(embedding, embeddingDim);
      return searchStmt.all(encoded, k);
    },

    close() {
      db.close();
    },
  };
}
