import type { RetrievedChunk } from "./retriever.js";

// Phase 8 (increment 8c): INSUFFICIENT-EVIDENCE POLICY — the deterministic,
// PURE evidence-sufficiency gate that lets the RAG pipeline distinguish an answer
// SUPPORTED BY CITED EVIDENCE from one that must be reported as INSUFFICIENT
// EVIDENCE. This module owns the LOW-SCORE cutoff and the "usable evidence"
// definition; src/core/rag.ts wires it into answerQuestion (pre-generation gate +
// post-generation reconciliation). Targets EXIT CRITERION 2: "Answers distinguish
// supported claims from insufficient evidence."
//
// WHY A SEPARATE, PURE MODULE
// ---------------------------
// The cutoff is a POLICY decision that must be tunable and independently testable
// offline (no store / embedder / model). Keeping it here as I/O-free pure logic
// means the offline suite exercises the exact thresholding the pipeline uses, and
// the 8d eval / Phase 8 closeout can ratify the default without touching the
// pipeline code.
//
// WHAT "SCORE" MEANS (why thresholding on it is principled)
// ---------------------------------------------------------
// Each RetrievedChunk carries a bounded similarity `score = 1 / (1 + distance)`
// (see distanceToScore in src/core/retriever.ts): it lives in (0, 1], equals 1
// for a perfect match, and is STRICTLY MONOTONICALLY DECREASING in the raw L2
// distance — so it is ORDER-PRESERVING with the store's distance ranking.
// Because of that monotonicity, applying a minimum-score cutoff is a PRINCIPLED
// TAIL CUT of the ranked list (it removes the least-similar chunks) and NEVER
// reorders results — it only decides where the "too weak to ground on" boundary
// falls.

// The PROPOSED default minimum similarity score a retrieved chunk must clear to
// be treated as USABLE evidence for grounding an answer.
//
// PROPOSED — PENDING EVAL / CLOSEOUT RATIFICATION. This is an EXPLICIT, exported,
// tunable default (a parameter on the pipeline, see AnswerQuestionOptions.minScore),
// NOT a silent magic number buried in a comparison. It is deliberately treated
// the way Phase 7 treated PROPOSED_BASELINE_THRESHOLDS: an input to be ratified
// by the 8d retrieval/answer eval and confirmed at Phase 8 closeout — it is NOT
// asserted as "the correct value" here.
//
// Rationale for the STARTING point: score = 1/(1+distance), so this default of
// 0.5 corresponds to an L2 distance of 1.0 (score 0.5 <=> distance 1.0). It is a
// conservative-but-permissive starting cutoff: near-duplicate/strong matches
// (small distance, score -> 1) always clear it, while clearly-weak tail matches
// (distance > 1, score < 0.5) are excluded. The eval (8d) is what will move this
// to its ratified value.
export const DEFAULT_MIN_EVIDENCE_SCORE = 0.5;

// A retrieved chunk paired with its store-resolved text (the shape the context
// builder consumes). Mirrors the internal pair produced by resolveChunkTexts in
// src/core/rag.ts; declared here so the gate can operate on the same value the
// pipeline builds context from.
export interface EvidencePair {
  chunk: RetrievedChunk;
  text: string;
}

/**
 * Pure evidence gate (Phase 8 / 8c): keep only the pairs whose retrieved chunk
 * clears `minScore`.
 *
 * The comparison is `chunk.score >= minScore` — INCLUSIVE at the boundary, so a
 * chunk whose score is EXACTLY the threshold is KEPT (a chunk is "usable" when it
 * is at least as similar as the cutoff). Because `score` is order-preserving with
 * distance (see the module header), this is a principled tail cut of the ranked
 * list, not a reordering: the surviving pairs retain their input (retrieval)
 * order, so the app-assigned marker order downstream is unchanged.
 *
 * PURE + deterministic + I/O-free: no store, no model, no clock. Returns a NEW
 * array (does not mutate the input).
 *
 * @param pairs    retrieved chunk + resolved text, in retrieval order.
 * @param minScore the inclusive minimum similarity a chunk must reach to be usable.
 * @returns the subset of `pairs` whose chunk.score >= minScore, in input order.
 */
export function filterUsableEvidence(
  pairs: EvidencePair[],
  minScore: number
): EvidencePair[] {
  return pairs.filter((pair) => pair.chunk.score >= minScore);
}
