import { describe, it, expect } from "vitest";

// Phase 8 (increment 8c) — PURE evidence-gate coverage (no store/embedder/model).
//
// Exercises filterUsableEvidence + the DEFAULT_MIN_EVIDENCE_SCORE constant in
// isolation: the >= minScore cutoff, the exact-boundary case, all-below ->
// empty, order preservation (score is order-preserving with distance), and
// injectability of the threshold.

import {
  filterUsableEvidence,
  DEFAULT_MIN_EVIDENCE_SCORE,
  type EvidencePair,
} from "../src/core/evidence.js";
import type { RetrievedChunk } from "../src/core/retriever.js";

function makeChunk(chunkId: string, score: number): RetrievedChunk {
  return {
    chunkId,
    documentId: `doc-${chunkId}`,
    sourceUri: `knowledge/corpus/${chunkId}.md`,
    chunkIndex: 0,
    ownerId: null,
    contentHash: `hash-${chunkId}`,
    // distance is not consulted by the gate; keep it consistent-ish for realism.
    distance: score > 0 ? 1 / score - 1 : Number.POSITIVE_INFINITY,
    score,
  };
}

function pair(chunkId: string, score: number): EvidencePair {
  return { chunk: makeChunk(chunkId, score), text: `text-${chunkId}` };
}

describe("Phase 8 (8c) — DEFAULT_MIN_EVIDENCE_SCORE", () => {
  it("is an exported bounded (0,1] constant (PROPOSED, ratified at closeout)", () => {
    expect(DEFAULT_MIN_EVIDENCE_SCORE).toBeGreaterThan(0);
    expect(DEFAULT_MIN_EVIDENCE_SCORE).toBeLessThanOrEqual(1);
  });
});

describe("Phase 8 (8c) — filterUsableEvidence (pure low-score gate)", () => {
  it("keeps only chunks whose score >= minScore (straddling the cutoff)", () => {
    const pairs = [pair("hi", 0.9), pair("lo", 0.1), pair("mid", 0.6)];
    const usable = filterUsableEvidence(pairs, 0.5);
    expect(usable.map((p) => p.chunk.chunkId)).toEqual(["hi", "mid"]);
  });

  it("KEEPS a chunk whose score is EXACTLY minScore (inclusive boundary)", () => {
    const pairs = [pair("boundary", 0.5), pair("just-below", 0.4999)];
    const usable = filterUsableEvidence(pairs, 0.5);
    expect(usable.map((p) => p.chunk.chunkId)).toEqual(["boundary"]);
  });

  it("all chunks below threshold -> empty usable set", () => {
    const pairs = [pair("a", 0.2), pair("b", 0.3), pair("c", 0.49)];
    expect(filterUsableEvidence(pairs, 0.5)).toEqual([]);
  });

  it("empty input -> empty output", () => {
    expect(filterUsableEvidence([], 0.5)).toEqual([]);
  });

  it("preserves input (retrieval) order among survivors (no reordering)", () => {
    // Deliberately NOT sorted by score: the gate must keep input order.
    const pairs = [pair("first", 0.6), pair("second", 0.9), pair("third", 0.55)];
    const usable = filterUsableEvidence(pairs, 0.5);
    expect(usable.map((p) => p.chunk.chunkId)).toEqual(["first", "second", "third"]);
  });

  it("does not mutate the input array", () => {
    const pairs = [pair("a", 0.9), pair("b", 0.1)];
    const snapshot = [...pairs];
    filterUsableEvidence(pairs, 0.5);
    expect(pairs).toEqual(snapshot);
  });

  it("is injectable: a higher minScore removes chunks a lower one kept", () => {
    const pairs = [pair("a", 0.6), pair("b", 0.9)];
    expect(filterUsableEvidence(pairs, 0.5).map((p) => p.chunk.chunkId)).toEqual([
      "a",
      "b",
    ]);
    expect(filterUsableEvidence(pairs, 0.7).map((p) => p.chunk.chunkId)).toEqual([
      "b",
    ]);
  });
});
