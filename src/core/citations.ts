import { redactText } from "./redaction.js";
import type { RetrievedChunk } from "./retriever.js";

// Phase 8 (increment 8b): TRUSTED CITATION RESOLUTION — the definitive citation-
// trust guarantee, targeting EXIT CRITERION 1: "Citation objects resolve to
// retrieved, authorized source IDs and are never accepted solely from model
// text." This is a PURE, deterministic, I/O-free library addition built on 8a.
//
// WHAT THIS MODULE OWNS
// ---------------------
// It turns the RAW, model-emitted citation MARKERS (integers from
// GroundedAnswer.citations) into TRUSTED, AUTHORIZED citation OBJECTS whose every
// identity field is copied FROM the app-owned marker -> RetrievedChunk map that
// 8a's answerQuestion produced. It NEVER reads an identifier from model text —
// the ONLY thing consumed from the model is an integer marker, which is used to
// INDEX the app-owned markerMap. If the model invents a marker the app did not
// assign, the marker is DROPPED, never honored. This is the crux of exit
// criterion 1: the model cannot introduce a citation to something the app did not
// retrieve and mark.
//
// THE TRUST BOUNDARY (why this is safe)
// -------------------------------------
//   - The markerMap keys are APP-ASSIGNED integers (assigned in retrieval order
//     by buildGroundedContext); its values are RetrievedChunk records whose
//     metadata is trusted, server-owned data resolved from the store by the
//     retriever (exit criterion 3). See src/core/retriever.ts.
//   - The model's citations are integers per groundedAnswerSchema (the schema
//     REJECTS strings / non-integers / negatives), so there is NO code path here
//     that could read a chunk/document id from model output — by construction the
//     resolver only ever consumes an integer marker.
//   - Every field on the emitted TrustedCitation is copied FROM the RetrievedChunk
//     found in the markerMap for that marker. Model text is used solely to SELECT
//     which app-owned chunk (if any) to cite.

// A resolved, trusted citation. Every IDENTITY field is APP-OWNED — copied from
// the RetrievedChunk stored in the app-assigned markerMap, NEVER from model
// output. `marker` is the app-assigned integer the model echoed (kept so callers
// can correlate a citation back to the answer's citation list / numbered
// context). `score` and `contentHash` are carried through from the same
// store-backed RetrievedChunk purely as a convenience for later citation display
// / eval (8d) — they are ALSO app-owned, not identity, and not model-derived.
export interface TrustedCitation {
  // The app-assigned integer marker the model cited (indexes the markerMap).
  marker: number;
  // Server-derived chunk identity (7a/7c) — from the store, via the markerMap.
  chunkId: string;
  // Server-derived owning-document identity — from the store.
  documentId: string;
  // The document's app-owned source identity (null only for a legacy row with
  // no recorded source) — from the store.
  sourceUri: string | null;
  // The chunk's ordinal position within its document — from the store.
  chunkIndex: number;
  // The app-owned authorization field (nullable) — from the DOCUMENT row.
  ownerId: string | null;
  // Convenience (NOT identity): the chunk's content-version hash — from the
  // store. Useful to detect citation drift when the corpus is re-ingested.
  contentHash: string;
  // Convenience (NOT identity): the similarity score derived from the KNN
  // distance — carried for later display / low-score eval.
  score: number;
}

// The documented, closed set of reasons a raw marker is DROPPED rather than
// emitted as a TrustedCitation:
//   - "unknown_marker": the marker is not a key of the app-owned markerMap. The
//     app never assigned it (out-of-range / hallucinated / a plausible-looking
//     but invalid index). This is the crux of exit criterion 1 — a marker the
//     app didn't assign is never honored.
//   - "unauthorized": under an ownerId scope, the resolved chunk is owned by a
//     DIFFERENT, non-null owner (a null-owner PUBLIC chunk is kept). Defense-in-
//     depth atop the retriever's ownerId filter (see AUTHORIZATION below).
export type DroppedCitationReason = "unknown_marker" | "unauthorized";

// A raw marker that was NOT emitted as a trusted citation, with the typed reason.
// `marker` is a safe integer; there is no dynamic free-text to leak here (the
// reason is a fixed const), but any human-facing message we ever build from this
// still routes through redaction per the always-redact convention.
export interface DroppedCitation {
  marker: number;
  reason: DroppedCitationReason;
}

// Arguments for {@link resolveCitations}.
export interface ResolveCitationsArgs {
  // The RAW integer markers the model emitted (GroundedAnswer.citations). Only
  // integers reach here (the schema enforces it); non-integers are impossible by
  // construction, but a defensive guard treats any non-integer as an unknown
  // marker rather than trusting it.
  citations: number[];
  // The APP-OWNED marker -> RetrievedChunk map produced by 8a (the ONLY source
  // of truth for which markers exist and what trusted metadata they carry).
  markerMap: Map<number, RetrievedChunk>;
  // OPTIONAL owner scope. When provided, a resolved citation is DROPPED as
  // "unauthorized" ONLY when its chunk.ownerId is a DIFFERENT, non-null owner; a
  // null-owner (unowned/PUBLIC) chunk is KEPT under any scope (see AUTHORIZATION).
  // When null/absent, ownership is NOT enforced here (matches retrieval scope).
  ownerId?: string | null;
}

// The result of resolution: the trusted citations kept, and the raw markers
// dropped (with reasons), for observability / eval.
export interface ResolveCitationsResult {
  resolved: TrustedCitation[];
  dropped: DroppedCitation[];
}

/**
 * Resolve raw model-emitted citation MARKERS into TRUSTED, AUTHORIZED citation
 * OBJECTS (Phase 8 / 8b). PURE + deterministic + I/O-free.
 *
 * For each raw marker, IN THE ORDER THE MODEL CITED THEM:
 *   1. UNKNOWN-MARKER DROP (the crux of exit criterion 1). Look the marker up in
 *      the APP-OWNED `markerMap`. If it is not an integer key of that map
 *      (out-of-range, hallucinated, or otherwise never app-assigned), DROP it
 *      with reason "unknown_marker". The app-owned markerMap is the SINGLE source
 *      of truth for which citations may exist; a marker the app didn't assign is
 *      NEVER honored.
 *   2. AUTHORIZATION DROP (defense-in-depth), PUBLIC-UNOWNED rule. If an
 *      `ownerId` scope is provided, DROP a resolved citation ONLY when its
 *      chunk.ownerId is a DIFFERENT, non-null owner (chunk.ownerId != null &&
 *      chunk.ownerId !== ownerId) with reason "unauthorized". retrieve() already
 *      owner-scopes the candidate set with the SAME rule, so a cross-owner chunk
 *      should not be in the markerMap at all; this guard is a SECOND, independent
 *      barrier so 8b can never emit a citation outside the caller's authorization
 *      even if retrieval scope were bypassed/changed.
 *        - NULL-OWNER (UNOWNED/PUBLIC) CHUNK UNDER A SCOPE: a chunk with
 *          ownerId === null is PUBLIC knowledge (e.g. the ingested corpus) and is
 *          KEPT under ANY ownerId scope. ONLY a chunk owned by a DIFFERENT,
 *          non-null owner is dropped as "unauthorized" — this preserves owned-doc
 *          isolation while making unowned corpus visible under an owner scope,
 *          exactly matching the retriever's public-unowned filter.
 *   3. DEDUPE. If the model cites the same marker more than once, it resolves to
 *      exactly ONE TrustedCitation. Dedupe is by marker. A repeated marker after
 *      its first (kept) occurrence is simply skipped — it is NOT re-emitted and
 *      NOT recorded as a drop (it is a duplicate of a decision already made, not
 *      a new failure).
 *
 * ORDERING (documented): the `resolved` list preserves FIRST-CITED ORDER — the
 * order in which the model first mentioned each surviving marker. This mirrors
 * how a human reader encounters citations in the answer text. (Ascending-marker
 * order was the alternative; first-cited was chosen and is tested.) The `dropped`
 * list is in the order drops were encountered.
 *
 * @returns { resolved, dropped } — never throws; deterministic for fixed inputs.
 */
export function resolveCitations(
  args: ResolveCitationsArgs
): ResolveCitationsResult {
  const { citations, markerMap, ownerId } = args;

  const resolved: TrustedCitation[] = [];
  const dropped: DroppedCitation[] = [];
  // Markers already decided (kept OR dropped) so a repeated marker resolves once
  // and a repeated FAILURE is not recorded twice.
  const seen = new Set<number>();

  for (const marker of citations) {
    // Defensive: only integers should arrive (the schema enforces it). Anything
    // else is treated as an unknown marker, never trusted. Guard against NaN /
    // floats slipping through a custom generateFn.
    if (!Number.isInteger(marker)) {
      // Use a stable key for the seen-set so a repeated bad value dedupes too.
      if (!seen.has(marker)) {
        seen.add(marker);
        dropped.push({ marker, reason: "unknown_marker" });
      }
      continue;
    }

    if (seen.has(marker)) continue; // duplicate of an already-decided marker.
    seen.add(marker);

    // (1) UNKNOWN-MARKER DROP — the app-owned markerMap is the source of truth.
    const chunk = markerMap.get(marker);
    if (chunk === undefined) {
      dropped.push({ marker, reason: "unknown_marker" });
      continue;
    }

    // (2) AUTHORIZATION DROP (defense-in-depth), PUBLIC-UNOWNED rule. Only
    // enforced under a scope, and aligned EXACTLY with the retriever's filter:
    // a null-owner (unowned/PUBLIC) chunk is KEPT under any ownerId scope (public
    // knowledge); ONLY a chunk owned by a DIFFERENT, non-null owner is dropped as
    // "unauthorized", preserving owned-doc isolation.
    if (ownerId != null && chunk.ownerId != null && chunk.ownerId !== ownerId) {
      dropped.push({ marker, reason: "unauthorized" });
      continue;
    }

    // KEEP. Every identity field is copied FROM the app-owned RetrievedChunk in
    // the markerMap — NEVER from model text (the model only supplied `marker`).
    resolved.push({
      marker,
      chunkId: chunk.chunkId,
      documentId: chunk.documentId,
      sourceUri: chunk.sourceUri,
      chunkIndex: chunk.chunkIndex,
      ownerId: chunk.ownerId,
      contentHash: chunk.contentHash,
      score: chunk.score,
    });
  }

  return { resolved, dropped };
}

// Build a concise, redacted, human-facing summary of a resolution outcome
// (e.g. for a future trace/log line). Marker numbers and fixed reason consts are
// inherently safe, but per the AGENTS.md always-redact convention any string
// destined for a log/trace/evidence artifact passes through redactText first.
export function summarizeCitationResolution(
  result: ResolveCitationsResult
): string {
  const kept = result.resolved.map((c) => c.marker).join(", ");
  const droppedParts = result.dropped
    .map((d) => `${String(d.marker)}:${d.reason}`)
    .join(", ");
  return redactText(
    `Citations resolved: kept [${kept}]; dropped [${droppedParts}].`
  );
}
