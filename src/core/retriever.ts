import { redactText, redactError } from "./redaction.js";
import { config } from "../config.js";
import { createEmbeddingsModel, type EmbeddingsOptions } from "./embeddings.js";
import { isEmbeddingDimensionCompatible } from "./embedding-compat.js";
import type { KnowledgeStore } from "./knowledge-store.js";

// Phase 7 (7e): RETRIEVER — pure retrieval logic over an injected query embedder
// and a knowledge store. NO direct network access and NO credential handling
// live here: production injects an embedder built from the single embeddings
// factory (createEmbeddingsModel via createQueryEmbedder), while OFFLINE tests
// inject a deterministic fake QueryEmbedder. This mirrors the ingestion module's
// DocumentEmbedder seam exactly.
//
// WHAT THIS MODULE OWNS
// ---------------------
// A single `retrieve()` entry point that:
//   1. Embeds the query text via the injected QueryEmbedder.
//   2. Dimension-checks the query vector against the store's embeddingDim,
//      REUSING the 7d `isEmbeddingDimensionCompatible` predicate (never a
//      duplicated guard), throwing a TYPED, REDACTED error on a mismatch.
//   3. Calls the store's low-level KNN primitive (searchChunksByVector).
//   4. RESOLVES each hit into a typed result carrying the TRUSTED, APP-OWNED
//      metadata needed for later citations + authorization: chunkId, documentId,
//      sourceUri, chunkIndex, ownerId, contentHash, distance, and a similarity
//      score derived from the L2 distance.
//
// EXIT CRITERION 3 (trusted metadata; no model-generated identifiers).
// -------------------------------------------------------------------
// The metadata on every RetrievedChunk is pulled FROM THE STORE (the chunk row
// and its owning document row) — NEVER from model output or from the query text.
// document_id / chunk_id are server-derived (7a/7c), source_uri is the
// app-owned identity, and owner_id is the app-owned authorization field. This
// resolution (KNN hit -> app-owned metadata) is the concrete demonstration that
// later authorization + trusted citations rest on server-owned identifiers, not
// on anything a model produced.
//
// SCOPE BOUNDARY (do NOT implement here)
// --------------------------------------
// No ranking policy beyond the store's L2-distance order. No reranking, no
// agentic retrieval, no answer generation / RAG / citations-in-text — those are
// later phases (8+). 7e stops at retrieval + resolving trusted metadata.

// Minimal typed contract for the query embedder. Deliberately narrow (just
// `embedQuery`) so production injects the real LiteLLM-routed model via
// {@link createQueryEmbedder} while OFFLINE tests inject a deterministic FAKE —
// no network in the test path. Mirrors ingestion's DocumentEmbedder.
export interface QueryEmbedder {
  // Embed a single query string into one vector.
  embedQuery(text: string): Promise<number[]>;
}

// A single retrieved chunk, resolved to its TRUSTED, APP-OWNED metadata. Every
// field here is server-owned (from the store), NEVER model-generated:
//   - chunkId / documentId : deterministic, server-derived identity (7a/7c).
//   - sourceUri            : the document's app-owned identity/source (7c). Null
//                            only for a legacy row with no recorded source.
//   - chunkIndex           : the chunk's ordinal position within its document.
//   - ownerId              : the app-owned authorization field (nullable now;
//                            the seam for later authz). From the DOCUMENT row.
//   - contentHash          : the chunk's content-version hash.
//   - distance             : raw L2 distance from the store's KNN primitive.
//   - score                : a monotonic similarity in (0, 1] derived from
//                            distance (see distanceToScore).
export interface RetrievedChunk {
  chunkId: string;
  documentId: string;
  sourceUri: string | null;
  chunkIndex: number;
  ownerId: string | null;
  contentHash: string;
  distance: number;
  score: number;
}

// Options for a single retrieval. `store` + `queryEmbedder` are injected I/O
// seams; `query`/`k` are the request. `ownerId`, when provided, restricts
// results to documents owned by that owner (authorization-ready — see below).
export interface RetrieveOptions {
  store: KnowledgeStore;
  queryEmbedder: QueryEmbedder;
  // The query text. EMPTY / whitespace-only is REJECTED (see EmptyQueryError).
  query: string;
  // The number of results to return. Must be a positive integer (see
  // InvalidKError). If k exceeds the corpus size the store simply returns all.
  k: number;
  // OPTIONAL owner-scoped retrieval (the authorization seam). When provided,
  // results are restricted to chunks whose OWNING DOCUMENT has this ownerId.
  // See the k-underfill note on the filtering strategy below.
  ownerId?: string | null;
}

// Typed, redaction-safe error thrown when the query is empty or whitespace-only.
// DESIGN DECISION (documented): an empty query is a caller error, not a valid
// "return nothing" request — embedding an empty string is meaningless and would
// waste a live embedding call, so we REJECT it before embedding. Thrown BEFORE
// any embed or store access.
export class EmptyQueryError extends Error {
  constructor() {
    super(
      redactText(
        "A non-empty query is required for retrieval: the query cannot be empty " +
          "or whitespace-only."
      )
    );
    this.name = "EmptyQueryError";
  }
}

// Typed, redaction-safe error thrown when `k` is not a positive integer. Thrown
// before any embed or store access.
export class InvalidKError extends Error {
  readonly k: number;
  constructor(k: number) {
    super(
      redactText(
        `Invalid retrieval k=${String(k)}: k must be a positive integer.`
      )
    );
    this.name = "InvalidKError";
    this.k = k;
  }
}

// Typed, redaction-safe error thrown when the QUERY vector's dimension does not
// match the store's fixed embeddingDim. This is the retrieval-side mirror of
// ingestion's dimension guard, but it REUSES the 7d predicate
// (isEmbeddingDimensionCompatible) rather than duplicating the equality logic.
// Structured fields let callers branch without parsing the message.
export class QueryEmbeddingDimensionError extends Error {
  readonly expectedDim: number;
  readonly actualDim: number;
  constructor(expectedDim: number, actualDim: number) {
    super(
      redactText(
        `Query embedding dimension mismatch during retrieval: the knowledge ` +
          `base is fixed at ${expectedDim} dimensions but the query embedder ` +
          `returned a ${actualDim}-dimension vector. Nothing was retrieved.`
      )
    );
    this.name = "QueryEmbeddingDimensionError";
    this.expectedDim = expectedDim;
    this.actualDim = actualDim;
  }
}

// L2 distance -> similarity score transform.
//
// The store's KNN primitive returns a raw NON-NEGATIVE L2 (Euclidean) distance
// where SMALLER is more similar (0 == identical vector). For a bounded,
// human-friendly "higher is better" score we map distance d >= 0 into (0, 1]
// with the monotonically-decreasing transform:
//
//     score = 1 / (1 + d)
//
// Properties (all documented, all tested):
//   - d == 0            -> score == 1   (a perfect match).
//   - d increases       -> score strictly DECREASES toward 0 (never reaches 0).
//   - order-preserving  -> ranking by descending score is IDENTICAL to ranking
//                          by ascending distance, so the score never reorders
//                          the store's distance order.
// This is a presentational convenience, NOT a probability or a cosine score; it
// exists so downstream/eval code can threshold on a bounded value. A negative or
// non-finite distance (which the store never produces) clamps to 0 defensively.
export function distanceToScore(distance: number): number {
  if (!Number.isFinite(distance) || distance < 0) return 0;
  return 1 / (1 + distance);
}

/**
 * Retrieve the `k` nearest chunks to `query`, resolved to their trusted,
 * app-owned metadata (best-first, length <= k).
 *
 * Pipeline:
 *   1. Validate `k` (positive integer) and `query` (non-empty) — BEFORE any I/O.
 *   2. Embed the query with the injected {@link QueryEmbedder}.
 *   3. Dimension-check the query vector against `store.embeddingDim` via the 7d
 *      {@link isEmbeddingDimensionCompatible} predicate; a mismatch throws a
 *      typed, redacted {@link QueryEmbeddingDimensionError}.
 *   4. Run the store's L2 KNN primitive and resolve each hit into a
 *      {@link RetrievedChunk} carrying store-owned metadata (exit criterion 3).
 *
 * Edge cases (documented):
 *   - EMPTY / whitespace-only query -> throws {@link EmptyQueryError}.
 *   - k <= 0 / non-integer          -> throws {@link InvalidKError}.
 *   - EMPTY store                   -> returns [] (the KNN yields no hits).
 *   - k LARGER than the corpus      -> returns ALL chunks (the store caps at the
 *                                      corpus size; result length may be < k).
 *
 * OWNER-SCOPED RETRIEVAL (authorization seam) + k-UNDERFILL CAVEAT:
 *   When `ownerId` is provided, results are restricted to chunks whose OWNING
 *   DOCUMENT has that ownerId. Because filtering happens AFTER the KNN (SQLite
 *   vec0 has no owner column), a naive `MATCH ... k = k` then filter could
 *   UNDER-FILL k when other owners' chunks occupy the top-k. To make owner
 *   scoping correct WITHOUT changing the store schema, we fetch a LARGER
 *   candidate window (k * OWNER_CANDIDATE_MULTIPLIER, capped) and filter it down
 *   to at most k. This is a pragmatic best-effort widen, DOCUMENTED as such: it
 *   is not a hard guarantee of exactly k for a corpus dominated by other owners
 *   (a dedicated owner-partitioned index is a later-phase concern). The
 *   unfiltered path is unaffected.
 */
export async function retrieve(
  options: RetrieveOptions
): Promise<RetrievedChunk[]> {
  const { store, queryEmbedder, query, k, ownerId } = options;

  // (1) Validate inputs BEFORE any embed or store access, so a bad request never
  // wastes a live embedding call or touches the DB.
  if (!Number.isInteger(k) || k <= 0) {
    throw new InvalidKError(k);
  }
  if (typeof query !== "string" || query.trim() === "") {
    throw new EmptyQueryError();
  }

  // (2) Embed the query (injected seam). A production embedder already redacts
  // its own failures; guard here too so nothing raw can escape this boundary.
  let queryVector: number[];
  try {
    queryVector = await queryEmbedder.embedQuery(query);
  } catch (err) {
    throw new Error(
      redactText(`Query embedding failed during retrieval. Cause: ${redactError(err)}`)
    );
  }

  // (3) Dimension guard — REUSE the 7d predicate (no duplicated equality logic).
  if (!isEmbeddingDimensionCompatible(store.embeddingDim, queryVector.length)) {
    throw new QueryEmbeddingDimensionError(store.embeddingDim, queryVector.length);
  }

  // (4) KNN. For the owner-scoped path, over-fetch a candidate window then filter
  // (see the k-underfill caveat above). The store returns best-first (ascending
  // L2 distance), which we preserve.
  const requestK = ownerId != null ? candidateWindow(k) : k;
  const hits = store.searchChunksByVector(queryVector, requestK);

  const results: RetrievedChunk[] = [];
  for (const hit of hits) {
    // Resolve TRUSTED, APP-OWNED metadata from the STORE (exit criterion 3):
    // the chunk row gives chunkIndex/contentHash; its owning document row gives
    // sourceUri + ownerId. None of this comes from model text or the query.
    const chunk = store.getChunk(hit.chunkId);
    if (chunk === null) continue; // defensive: a vector without its chunk row.
    const document = store.getDocument(chunk.documentId);
    const resolvedOwnerId = document?.ownerId ?? null;

    // Owner-scoped filter (authorization seam): drop chunks not owned by the
    // requested owner. Applied post-KNN over the widened candidate window.
    if (ownerId != null && resolvedOwnerId !== ownerId) continue;

    results.push({
      chunkId: chunk.chunkId,
      documentId: chunk.documentId,
      sourceUri: document?.sourceUri ?? null,
      chunkIndex: chunk.chunkIndex,
      ownerId: resolvedOwnerId,
      contentHash: chunk.contentHash,
      distance: hit.distance,
      score: distanceToScore(hit.distance),
    });

    // Never return more than k (the widened window is only for owner filtering).
    if (results.length >= k) break;
  }

  return results;
}

// The multiplier + cap used to widen the KNN candidate window for owner-scoped
// retrieval (see the k-underfill caveat). Exported for tests/documentation.
export const OWNER_CANDIDATE_MULTIPLIER = 5;
export const OWNER_CANDIDATE_CAP = 200;

function candidateWindow(k: number): number {
  return Math.min(k * OWNER_CANDIDATE_MULTIPLIER, OWNER_CANDIDATE_CAP);
}

/**
 * Production adapter: build a {@link QueryEmbedder} backed by the real
 * LiteLLM-routed embedding model. A THIN wrapper over the single OpenAIEmbeddings
 * construction site (`createEmbeddingsModel` in `src/core/embeddings.ts`) — it
 * constructs no client itself, honoring the single-factory rule. Mirrors
 * ingestion's `createDocumentEmbedder`. OFFLINE tests never call this; they inject
 * a fake embedder directly.
 *
 * DIMENSION IS REQUESTED EXPLICITLY HERE. The query vector MUST match the
 * knowledge store's FIXED dimension (`store.embeddingDim`, sourced from
 * `config.embedDim`) or `retrieve()` throws QueryEmbeddingDimensionError, so this
 * adapter passes `dimensions: config.embedDim` to the factory (a Matryoshka-
 * truncation request — e.g. 768 via gemini-embedding-001), symmetric with
 * ingestion's createDocumentEmbedder. The factory no longer defaults `dimensions`
 * (see the WIRE CONTRACT in embeddings.ts); the explicit request at this
 * store-reading call site is what keeps live retrieval producing store-compatible
 * query vectors. A caller may still override `options.dimensions`.
 */
export function createQueryEmbedder(
  options: EmbeddingsOptions = {}
): QueryEmbedder {
  const model = createEmbeddingsModel({
    dimensions: config.embedDim,
    ...options,
  });
  return {
    async embedQuery(text: string): Promise<number[]> {
      try {
        return await model.embedQuery(text);
      } catch (err) {
        // Surface a redacted, single-line reason (never a raw provider error
        // that could carry the endpoint/key or request content).
        throw new Error(
          `Embedding request failed during retrieval. Cause: ${redactError(err)}`
        );
      }
    },
  };
}
