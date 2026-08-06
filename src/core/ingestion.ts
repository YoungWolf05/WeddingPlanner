import { redactError, redactText } from "./redaction.js";
import { createEmbeddingsModel, type EmbeddingsOptions } from "./embeddings.js";
import { chunkText, type ChunkingOptions } from "./chunking.js";
import {
  computeContentHash,
  computeDocumentId,
  type KnowledgeStore,
} from "./knowledge-store.js";

// Phase 7 (7c): SOURCE-ADDRESSED UPSERT + UPDATE/DELETE INGESTION ORCHESTRATOR.
//
// WHAT THIS MODULE OWNS
// ---------------------
// The deterministic pipeline that turns a source (REQUIRED uri + raw content)
// into normalized → chunked → embedded → persisted documents/chunks/vectors, now
// with first-class UPDATE and DELETE folded into identity (Phase 7 exit criteria
// 1 and 2). Identity is the SOURCE; content is its VERSION.
//
// UPSERT-BY-SOURCE STATE MACHINE
//   1. Compute the SOURCE-addressed `document_id = computeDocumentId(sourceUri)`
//      (7a helper, now hashing the normalized source_uri) and the content
//      VERSION hash `newContentHash = computeContentHash(content)`.
//   2. Look up the document by that id:
//      - NOT FOUND                         → CREATE  (chunk→embed→persist), "created".
//      - FOUND, stored hash == new hash     → UNCHANGED (no chunk, no embed, no
//                                             write), "unchanged".
//      - FOUND, stored hash != new hash     → UPDATE IN PLACE: chunk→embed, then in
//                                             ONE transaction delete the old
//                                             chunks+vectors, insert the new ones,
//                                             and bump the document's content_hash +
//                                             updated_at (created_at preserved),
//                                             "updated".
//   3. EMPTY / whitespace-only content chunks to ZERO chunks → "skipped": the
//      embedder is called ZERO times, NOTHING is written, and an EXISTING
//      document for that source is left UNTOUCHED (a skip must never wipe
//      content — explicit removal is the delete API, not an empty re-ingest).
//
// IDENTITY / UPDATE SEMANTICS (documented, 7c source-addressed)
//   `document_id == sha256(normalizeSourceUri(source_uri))`, INDEPENDENT of
//   content. Two DIFFERENT source URIs with byte-identical content are TWO
//   distinct documents (no content dedup). The SAME source re-ingested with
//   changed content is an in-place UPDATE of the SAME document_id (full chunk
//   REPLACE, not diff, for determinism). This SUPERSEDES the 7a/7b
//   content-addressed model.
//
// DELETE API
//   `deleteDocument(store, documentId)` / `deleteSource(store, sourceUri)` are
//   thin, typed, atomic removals (document + all chunks + all vec0 vectors).
//   Deleting a nonexistent document/source is a clean no-op returning false.
//
// SCOPE BOUNDARY (deferred; do NOT implement here)
//   - 7d: LIVE embedding + full embedding-model/dimension COMPATIBILITY policy.
//     The dimension guard below is only a local vector-length check.
//   - 7e: retriever/ranking, eval, curated corpus.

// Minimal typed contract for the embedding provider used by ingestion. Kept
// deliberately narrow (just `embedDocuments`) so production injects the real
// LiteLLM-backed model via {@link createDocumentEmbedder} while OFFLINE tests
// inject a deterministic FAKE embedder — no network in the test path.
export interface DocumentEmbedder {
  // Embed a batch of chunk texts, returning one vector per input text IN ORDER.
  embedDocuments(texts: string[]): Promise<number[][]>;
}

// Discriminated ingestion result:
//   - `created`   = a new source's document (+ its chunks and vectors) was
//                   persisted. Also the winner of a concurrent first-ingest race
//                   for the same NEW source (see FIX A below).
//   - `unchanged` = the source already existed and its stored content_hash
//                   equals the new content's hash, so ingestion was a no-op
//                   (nothing written, embedder NOT called).
//   - `updated`   = the source already existed with DIFFERENT content, so its
//                   chunks + vectors were fully REPLACED and the document row's
//                   content_hash + updated_at were bumped in ONE transaction
//                   (created_at preserved). `chunkCount` is the NEW chunk count.
//   - `skipped`   = the normalized content is empty or whitespace-only, so it
//                   chunks to ZERO chunks. NOTHING is persisted and the embedder
//                   is NOT called. A skip NEVER wipes an existing document for
//                   that source (explicit removal is the delete API). For a brand
//                   new source `chunkCount` is 0; when skipping over an EXISTING
//                   document it reports that document's current (unchanged) chunk
//                   count.
// `documentId` is the SOURCE-addressed id in ALL cases (computed even when
// skipped, so callers can correlate).
export type IngestResult =
  | { status: "created"; documentId: string; chunkCount: number }
  | { status: "unchanged"; documentId: string; chunkCount: number }
  | { status: "updated"; documentId: string; chunkCount: number }
  | { status: "skipped"; documentId: string; chunkCount: number };

/**
 * Typed, redaction-safe error thrown when an embedding vector's length does not
 * match the store's fixed `embedding_dim`. Raised BEFORE the write transaction
 * opens, so nothing is persisted (strictly stronger than a rollback). The
 * structured fields let callers branch without parsing the message.
 *
 * NOTE: 7b's guard is a local LENGTH check only; full embedding-model
 * compatibility policy is deferred to 7d.
 */
export class EmbeddingDimensionError extends Error {
  readonly expectedDim: number;
  readonly actualDim: number;

  constructor(expectedDim: number, actualDim: number) {
    super(
      redactText(
        `Embedding dimension mismatch during ingestion: the knowledge base is ` +
          `fixed at ${expectedDim} dimensions but the embedder returned a ` +
          `${actualDim}-dimension vector. Nothing was written.`
      )
    );
    this.name = "EmbeddingDimensionError";
    this.expectedDim = expectedDim;
    this.actualDim = actualDim;
  }
}

/**
 * Typed, redaction-safe error thrown when the embedder returns a DIFFERENT
 * number of vectors than the number of chunks it was asked to embed (a broken
 * embedder contract). Raised before any write, so nothing is persisted.
 */
export class EmbeddingCountError extends Error {
  readonly expectedCount: number;
  readonly actualCount: number;

  constructor(expectedCount: number, actualCount: number) {
    super(
      redactText(
        `Embedder contract violation during ingestion: expected ` +
          `${expectedCount} vectors (one per chunk) but received ` +
          `${actualCount}. Nothing was written.`
      )
    );
    this.name = "EmbeddingCountError";
    this.expectedCount = expectedCount;
    this.actualCount = actualCount;
  }
}

/**
 * Typed, redaction-safe error thrown when the atomic write transaction fails on
 * a SQLite CONSTRAINT violation that we CANNOT explain as the benign
 * concurrent-first-ingest race (i.e. the document did NOT materialize on
 * re-check). The underlying driver message is routed through {@link redactError}
 * so no raw better-sqlite3 string (SQL text, bound values) can leak to a log or
 * caller, honoring the always-on redaction convention.
 *
 * NOTE: 7c makes the create-vs-update DECISION atomically INSIDE the write
 * transaction (a re-read on the single, synchronous connection), so a concurrent
 * ingest of the same source converges deterministically WITHOUT a constraint
 * race — the winner CREATEs and the follower sees the row and resolves to
 * `unchanged`/`updated`. This error is defense-in-depth for a constraint failure
 * that STILL surfaces and does not resolve to an existing document.
 */
export class IngestionWriteError extends Error {
  constructor(cause: unknown) {
    super(
      redactText(
        `Ingestion write failed on a database constraint and did not resolve to ` +
          `an existing document. Nothing was persisted. Cause: ${redactError(cause)}`
      )
    );
    this.name = "IngestionWriteError";
  }
}

// Detect a better-sqlite3 CONSTRAINT violation WITHOUT importing the driver's
// error type: better-sqlite3 raises a `SqliteError` whose `.code` is a string
// beginning with `SQLITE_CONSTRAINT` (e.g. `SQLITE_CONSTRAINT_PRIMARYKEY` for a
// duplicate documents PK, `SQLITE_CONSTRAINT_UNIQUE` for a duplicate
// (document_id, chunk_index)). We match ONLY that family so unrelated errors are
// never mistaken for a benign convergence race.
function isSqliteConstraintError(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    typeof (err as { code: unknown }).code === "string" &&
    (err as { code: string }).code.startsWith("SQLITE_CONSTRAINT")
  );
}

// Options for a single-document ingestion.
export interface IngestDocumentOptions {
  // The target knowledge store (7a). Owns the DB connection + transaction.
  store: KnowledgeStore;
  // The injected embedder (production adapter or a test fake).
  embedder: DocumentEmbedder;
  // Raw source content; normalized + hashed internally as the content VERSION.
  content: string;
  // REQUIRED (7c): the source IS identity. Normalized + validated via
  // normalizeSourceUri; document_id is derived from it. Empty/whitespace-only
  // throws InvalidSourceUriError (before any embed or write).
  sourceUri: string;
  // Authorization-ready owner metadata (nullable now; 7a schema column).
  ownerId?: string | null;
  // Injectable chunking parameters; omitted fields use the documented defaults.
  chunking?: ChunkingOptions;
}

/**
 * Ingest ONE source, upserting BY SOURCE (7c).
 *
 * State machine on `document_id = computeDocumentId(sourceUri)` (source-addressed):
 *   - NOT FOUND                     → CREATE, `{ status: "created" }`.
 *   - FOUND, content hash unchanged → NO-OP, `{ status: "unchanged" }` (embedder
 *                                     NOT called, nothing written).
 *   - FOUND, content hash changed   → UPDATE IN PLACE, `{ status: "updated" }`:
 *                                     old chunks+vectors deleted and new ones
 *                                     inserted with the document row's
 *                                     content_hash + updated_at bumped, all in
 *                                     ONE transaction (created_at preserved,
 *                                     full REPLACE not diff).
 *   - EMPTY / whitespace content    → `{ status: "skipped" }`: embedder-free,
 *                                     nothing written, and an EXISTING document
 *                                     for that source is left UNTOUCHED (a skip
 *                                     never wipes content).
 *
 * CONCURRENCY (FIX A, 7c-adapted): the existence check and the embed are
 * separated by an `await`, so two concurrent ingests of the SAME source can both
 * embed. The create-vs-update DECISION and the write are then made TOGETHER
 * inside ONE `store.db.transaction()`, which re-reads the freshest committed
 * state. Because better-sqlite3 is a synchronous single connection, transactions
 * are fully serialized: the first writer CREATEs; the follower re-reads, sees the
 * row, and resolves to `unchanged` (identical content) or `updated` (different
 * content) — never a duplicate, never an orphan, never a raw constraint error. A
 * SQLITE_CONSTRAINT that still somehow surfaces is caught and, if it resolves to
 * an existing document, reported idempotently; otherwise re-thrown as a TYPED,
 * REDACTED {@link IngestionWriteError}. Non-constraint errors propagate (the
 * atomic transaction has already rolled back cleanly).
 */
export async function ingestDocument(
  options: IngestDocumentOptions
): Promise<IngestResult> {
  const { store, embedder, content } = options;
  // (1) SOURCE-addressed identity (7c). computeDocumentId validates + normalizes
  // the REQUIRED source_uri and throws InvalidSourceUriError on empty — BEFORE
  // any embed or write. content_hash is the SEPARATE content VERSION marker.
  const documentId = computeDocumentId(options.sourceUri);
  const newContentHash = computeContentHash(content);

  // (2) UNCHANGED fast-path: if the source already exists AND its stored content
  // hash equals the new content's hash, this is a strict no-op — DO NOT chunk,
  // DO NOT call the embedder, DO NOT write. This is what keeps re-ingesting the
  // same source cheap (no embedding cost) and duplicate-free.
  const existing = store.getDocument(documentId);
  if (existing !== null && existing.contentHash === newContentHash) {
    return {
      status: "unchanged",
      documentId,
      chunkCount: store.listChunks(documentId).length,
    };
  }

  // (3) CREATE or UPDATE: chunk deterministically. `chunkText` normalizes
  // internally with the SAME normalization used to derive the content hash, so
  // boundaries are stable.
  const chunks = chunkText(content, options.chunking);

  // SKIP (product decision): empty or whitespace-only content normalizes to zero
  // chunks. Do NOT call the embedder and do NOT persist ANYTHING — a zero-chunk
  // document is permanently unretrievable. Crucially, a skip NEVER mutates or
  // deletes an EXISTING document for this source: wiping content is the explicit
  // job of the delete API, not of an empty re-ingest. Report `skipped`,
  // reflecting the existing (unchanged) chunk count when a document is present.
  if (chunks.length === 0) {
    return {
      status: "skipped",
      documentId,
      chunkCount: existing !== null ? store.listChunks(documentId).length : 0,
    };
  }

  // Embed the chunks (order-preserving), then contract- and dimension-guard the
  // vectors BEFORE opening the write transaction, so a bad embedder response
  // persists NOTHING (no partial document/chunks/vectors on either path).
  const vectors: number[][] = await embedder.embedDocuments(chunks);
  if (vectors.length !== chunks.length) {
    throw new EmbeddingCountError(chunks.length, vectors.length);
  }
  for (const vector of vectors) {
    if (vector.length !== store.embeddingDim) {
      throw new EmbeddingDimensionError(store.embeddingDim, vector.length);
    }
  }

  const insertChunks = (): void => {
    for (let index = 0; index < chunks.length; index++) {
      store.insertChunk({
        documentId,
        chunkIndex: index,
        text: chunks[index]!,
        embedding: vectors[index],
      });
    }
  };

  // Persist ATOMICALLY. The create-vs-update decision is made INSIDE the
  // transaction on a FRESH read (`store.getDocument`), so on the single,
  // synchronous connection it is race-free: nested store primitives run as
  // savepoints, and any failure rolls the WHOLE thing back — no orphaned or
  // duplicate document/chunks/vectors. Full chunk REPLACE (delete-all-then-
  // insert) keeps the update deterministic.
  try {
    return store.db.transaction((): IngestResult => {
      const current = store.getDocument(documentId);
      if (current === null) {
        // CREATE: brand-new source.
        store.insertDocument({
          content,
          sourceUri: options.sourceUri,
          ownerId: options.ownerId ?? null,
        });
        insertChunks();
        return { status: "created", documentId, chunkCount: chunks.length };
      }
      if (current.contentHash === newContentHash) {
        // A concurrent writer already produced identical content between our
        // pre-embed read and here: converge to a no-op.
        return {
          status: "unchanged",
          documentId,
          chunkCount: store.listChunks(documentId).length,
        };
      }
      // UPDATE IN PLACE: full replace — drop old chunks+vectors, bump the
      // document's content_hash + updated_at (created_at preserved), insert new.
      store.deleteChunks(documentId);
      store.touchDocumentContent(documentId, content);
      insertChunks();
      return { status: "updated", documentId, chunkCount: chunks.length };
    })();
  } catch (err) {
    // Defense-in-depth: a SQLITE_CONSTRAINT that still surfaces despite the
    // in-transaction decision means a concurrent writer already persisted this
    // source. Resolve idempotently using the STORE as source of truth; if the
    // document did NOT materialize, rethrow as a TYPED, REDACTED error so no raw
    // driver string leaks. Any NON-constraint error propagates (the transaction
    // has already rolled back atomically; store-originated errors are redacted).
    if (isSqliteConstraintError(err)) {
      const persisted = store.getDocument(documentId);
      if (persisted !== null) {
        return {
          status: persisted.contentHash === newContentHash ? "unchanged" : "updated",
          documentId,
          chunkCount: store.listChunks(documentId).length,
        };
      }
      throw new IngestionWriteError(err);
    }
    throw err;
  }
}

// One entry for a batch ingestion. `sourceUri` is REQUIRED (7c identity).
export interface IngestDocumentInput {
  content: string;
  sourceUri: string;
  ownerId?: string | null;
}

// Options for a batch ingestion. A single `store`/`embedder`/`chunking` is
// shared across all documents.
export interface IngestDocumentsOptions {
  store: KnowledgeStore;
  embedder: DocumentEmbedder;
  documents: readonly IngestDocumentInput[];
  chunking?: ChunkingOptions;
}

/**
 * Ingest a batch of documents SEQUENTIALLY, each through {@link ingestDocument}.
 * Sequential (not concurrent) execution keeps ordering deterministic and honors
 * the single-connection, serialized-write model of the underlying SQLite store.
 * Every element is independently idempotent. Results are returned in input order.
 */
export async function ingestDocuments(
  options: IngestDocumentsOptions
): Promise<IngestResult[]> {
  const results: IngestResult[] = [];
  for (const doc of options.documents) {
    results.push(
      await ingestDocument({
        store: options.store,
        embedder: options.embedder,
        content: doc.content,
        sourceUri: doc.sourceUri,
        ownerId: doc.ownerId ?? null,
        chunking: options.chunking,
      })
    );
  }
  return results;
}

// ---------------------------------------------------------------------------
// DELETE API (7c). Thin, typed, atomic removals for callers. Each removes the
// document, ALL of its chunks, and ALL of their vec0 vectors in one transaction
// (the store primitives guarantee zero orphan vectors). Deleting something that
// does not exist is a clean no-op returning false.
// ---------------------------------------------------------------------------

/**
 * Delete a document (and its chunks + vectors) by its app-owned `documentId`.
 * Returns true if a document was removed, false if there was no such document.
 * Atomic and deterministic (delegates to the store's cascading hard delete).
 */
export function deleteDocument(
  store: KnowledgeStore,
  documentId: string
): boolean {
  return store.deleteDocument(documentId);
}

/**
 * Delete the document owning `sourceUri` (7c source-scoped delete), plus its
 * chunks + vectors. Returns true if removed, false if no document exists for that
 * source. Throws InvalidSourceUriError on an empty/whitespace-only source (the
 * source is identity and must be well-formed even to address a delete).
 */
export function deleteSource(
  store: KnowledgeStore,
  sourceUri: string
): boolean {
  return store.deleteDocumentBySource(sourceUri);
}

/**
 * Production adapter: build a {@link DocumentEmbedder} backed by the real
 * LiteLLM-routed embedding model. This is a THIN wrapper over the single
 * OpenAIEmbeddings construction site (`createEmbeddingsModel` in
 * `src/core/embeddings.ts`) — it constructs no client itself, honoring the
 * single-factory rule. OFFLINE tests never call this; they inject a fake
 * embedder directly.
 */
export function createDocumentEmbedder(
  options: EmbeddingsOptions = {}
): DocumentEmbedder {
  const model = createEmbeddingsModel(options);
  return {
    async embedDocuments(texts: string[]): Promise<number[][]> {
      try {
        return await model.embedDocuments(texts);
      } catch (err) {
        // Surface a redacted, single-line reason (never a raw provider error
        // that could carry the endpoint/key or request content).
        throw new Error(
          `Embedding request failed during ingestion. Cause: ${redactError(err)}`
        );
      }
    },
  };
}
