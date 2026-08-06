import { redactError, redactText } from "./redaction.js";
import { createEmbeddingsModel, type EmbeddingsOptions } from "./embeddings.js";
import { chunkText, type ChunkingOptions } from "./chunking.js";
import { computeDocumentId, type KnowledgeStore } from "./knowledge-store.js";

// Phase 7 (7b): IDEMPOTENT INGESTION ORCHESTRATOR.
//
// WHAT THIS MODULE OWNS
// ---------------------
// The deterministic pipeline that turns a source (uri + raw content) into
// normalized → chunked → embedded → persisted documents/chunks/vectors, and is a
// NO-OP when the SAME content is ingested again (Phase 7 exit criterion 1).
//
// PIPELINE
//   1. Compute the content-addressed `document_id` from `content` (reusing the
//      7a `computeDocumentId`, which normalizes + SHA-256 hashes) — never a
//      caller/model id.
//   2. IDEMPOTENCY CHECK: if a document with that id already exists, return
//      `unchanged` immediately. The embedder is NOT called and NOTHING is
//      written (this is what makes re-ingestion cheap AND duplicate-free).
//   3. Otherwise chunk the content deterministically (`chunkText`), embed the
//      chunks through the INJECTED embedder, DIMENSION-GUARD every vector, then
//      persist the document + chunks + vectors in ONE transaction.
//
// IDENTITY / DEDUP SEMANTICS (documented, from 7a)
//   Identity is CONTENT-ADDRESSED: `document_id == sha256(normalizeContent)`.
//   `source_uri` is metadata ONLY and is NOT part of identity, so two DIFFERENT
//   source URIs carrying byte-for-byte identical normalized content dedupe to
//   ONE document by design. Conversely, genuinely NEW content yields a NEW
//   document_id and a fresh insert.
//
// SCOPE BOUNDARY (deferred; do NOT implement here)
//   - 7c: update/delete semantics when the SAME `source_uri` is re-ingested with
//     CHANGED content (which produces a NEW content-addressed document_id, i.e. a
//     new document). 7b intentionally does NOT reconcile/replace the prior
//     document for that URI or delete stale chunks. TODO(7c): add source-URI
//     replacement / stale-document reconciliation here.
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
//   - `created`   = a new document (+ its chunks and vectors) was persisted.
//   - `unchanged` = an identical content-addressed document already existed and
//                   ingestion was a no-op (nothing written, embedder not called).
//                   This ALSO covers the concurrent-first-ingest race where two
//                   ingests of identical NEW content run at once: exactly one
//                   wins the insert (`created`) and the loser resolves cleanly to
//                   `unchanged` against the persisted document (see FIX A below).
//   - `skipped`   = the normalized content is empty or whitespace-only, so it
//                   chunks to ZERO chunks. NOTHING is persisted (no document,
//                   chunk, or vector row) and the embedder is NOT called. This is
//                   a deliberate product decision (see FIX B below): a zero-chunk
//                   document is unretrievable dead weight, so we refuse to mint
//                   one. Re-ingesting the same empty content is still `skipped`
//                   (idempotent) and still writes nothing.
// `documentId` is the content-addressed id in ALL cases (computed even when
// skipped, so callers can correlate). `chunkCount` is the number of persisted
// chunks — always 0 for `skipped`.
export type IngestResult =
  | { status: "created"; documentId: string; chunkCount: number }
  | { status: "unchanged"; documentId: string; chunkCount: number }
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
 * NOTE: the EXPECTED race (identical NEW content ingested concurrently) is NOT
 * an error — it resolves to `{ status: "unchanged" }` (see {@link ingestDocument}
 * FIX A). This error is only for a constraint failure that stays unexplained.
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
// never mistaken for the idempotency race.
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
  // Raw source content; normalized + hashed internally for identity.
  content: string;
  // Provenance metadata only (NOT identity). Defaults to null.
  sourceUri?: string | null;
  // Authorization-ready owner metadata (nullable now; 7a schema column).
  ownerId?: string | null;
  // Injectable chunking parameters; omitted fields use the documented defaults.
  chunking?: ChunkingOptions;
}

/**
 * Ingest ONE document idempotently.
 *
 * Returns `{ status: "unchanged" }` (calling the embedder ZERO times and writing
 * nothing) when a document with the same content-addressed id already exists.
 * Returns `{ status: "skipped" }` (again embedder-free and writing nothing) when
 * the content is empty/whitespace-only and therefore chunks to zero chunks —
 * minting a zero-chunk document is a deliberate NON-decision (FIX B). Otherwise
 * chunks, embeds, dimension-guards, and persists the document + chunks + vectors
 * in a single transaction and returns `{ status: "created" }`.
 *
 * CONCURRENCY (FIX A): the existence check and the embed step are separated by an
 * `await`, so two concurrent ingests of identical NEW content can both pass the
 * check and both reach the insert. The UNIQUE/PK constraints guarantee only one
 * physically wins; the loser catches the SQLITE_CONSTRAINT violation, re-reads
 * the now-persisted document, and resolves to `{ status: "unchanged" }` using the
 * STORE as the source of truth for `chunkCount`. Non-constraint errors are never
 * swallowed — they propagate (the atomic transaction rolls back cleanly first).
 */
export async function ingestDocument(
  options: IngestDocumentOptions
): Promise<IngestResult> {
  const { store, embedder, content } = options;
  const documentId = computeDocumentId(content);

  // (2) IDEMPOTENCY: content-addressed identity check. If the document already
  // exists, this is a strict no-op — DO NOT chunk, DO NOT call the embedder, DO
  // NOT write. This is the core of exit criterion 1 (no duplicate documents or
  // chunks on re-ingest) and the reason re-ingestion incurs no embedding cost.
  const existing = store.getDocument(documentId);
  if (existing !== null) {
    return {
      status: "unchanged",
      documentId,
      chunkCount: store.listChunks(documentId).length,
    };
  }

  // (3) New content: chunk deterministically. `chunkText` normalizes internally
  // with the SAME normalization used to derive the id, so boundaries are stable.
  const chunks = chunkText(content, options.chunking);

  // FIX B (product decision): empty or whitespace-only content normalizes to
  // zero chunks. Do NOT call the embedder and do NOT persist ANYTHING — a
  // zero-chunk document is permanently unretrievable and would only accrete dead
  // rows for 7c/7e to carry. Report it explicitly as `skipped` (idempotent: the
  // next ingest of the same empty content re-computes zero chunks and skips
  // again, having written nothing).
  if (chunks.length === 0) {
    return { status: "skipped", documentId, chunkCount: 0 };
  }

  // Embed the chunks (order-preserving), then contract- and dimension-guard the
  // vectors BEFORE opening the write transaction, so a bad embedder response
  // persists NOTHING (no partial document/chunks/vectors).
  const vectors: number[][] = await embedder.embedDocuments(chunks);
  if (vectors.length !== chunks.length) {
    throw new EmbeddingCountError(chunks.length, vectors.length);
  }
  for (const vector of vectors) {
    if (vector.length !== store.embeddingDim) {
      throw new EmbeddingDimensionError(store.embeddingDim, vector.length);
    }
  }

  // Persist document + chunks + vectors ATOMICALLY. Nested store primitives run
  // as savepoints inside this outer transaction, so any failure (including the
  // store's own defense-in-depth dimension check inside insertChunk) rolls the
  // WHOLE thing back — no orphaned document, chunks, or vectors. The store's
  // UNIQUE(document_id, chunk_index) makes a duplicate chunk index impossible.
  try {
    store.db.transaction(() => {
      store.insertDocument({
        content,
        sourceUri: options.sourceUri ?? null,
        ownerId: options.ownerId ?? null,
      });
      for (let index = 0; index < chunks.length; index++) {
        store.insertChunk({
          documentId,
          chunkIndex: index,
          text: chunks[index]!,
          embedding: vectors[index],
        });
      }
    })();
  } catch (err) {
    // FIX A: a SQLITE_CONSTRAINT violation here means a CONCURRENT ingest of
    // identical content won the race and already persisted this exact document
    // (documents PK / chunks UNIQUE(document_id, chunk_index)). Resolve to the
    // idempotent outcome using the STORE as the source of truth — never the
    // loser's local `chunks` array. If the document did NOT materialize, this is
    // an unexpected constraint failure: rethrow as a TYPED, REDACTED error so no
    // raw driver string leaks. Any NON-constraint error is unrelated and must
    // propagate (the transaction has already rolled back atomically, so the
    // store is clean); store-originated errors are already redacted.
    if (isSqliteConstraintError(err)) {
      const persisted = store.getDocument(documentId);
      if (persisted !== null) {
        return {
          status: "unchanged",
          documentId,
          chunkCount: store.listChunks(documentId).length,
        };
      }
      throw new IngestionWriteError(err);
    }
    throw err;
  }

  return { status: "created", documentId, chunkCount: chunks.length };
}

// One entry for a batch ingestion.
export interface IngestDocumentInput {
  content: string;
  sourceUri?: string | null;
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
        sourceUri: doc.sourceUri ?? null,
        ownerId: doc.ownerId ?? null,
        chunking: options.chunking,
      })
    );
  }
  return results;
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
