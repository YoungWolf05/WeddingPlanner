import { describe, it, expect } from "vitest";

// Phase 8 (increment 8b) — TRUSTED CITATION RESOLUTION offline coverage.
//
// Fully OFFLINE + DETERMINISTIC + PURE (no network / store / embedder / model).
// resolveCitations is I/O-free logic over an app-owned marker -> RetrievedChunk
// map, so these tests exercise EXIT CRITERION 1 directly: citation objects
// resolve to retrieved, AUTHORIZED source IDs and are NEVER accepted solely from
// model text (the model only supplies an integer marker that indexes the map).

import {
  resolveCitations,
  summarizeCitationResolution,
  type TrustedCitation,
} from "../src/core/citations.js";
import type { RetrievedChunk } from "../src/core/retriever.js";

// ---- Deterministic fixtures -------------------------------------------------

function makeChunk(
  overrides: Partial<RetrievedChunk> & { chunkId: string }
): RetrievedChunk {
  return {
    documentId: `doc-${overrides.chunkId}`,
    sourceUri: `knowledge/corpus/${overrides.chunkId}.md`,
    chunkIndex: 0,
    ownerId: null,
    contentHash: `hash-${overrides.chunkId}`,
    distance: 0,
    score: 1,
    ...overrides,
  };
}

// Build an app-owned markerMap (1-based markers, retrieval order) from chunks.
function makeMarkerMap(chunks: RetrievedChunk[]): Map<number, RetrievedChunk> {
  const map = new Map<number, RetrievedChunk>();
  chunks.forEach((chunk, i) => map.set(i + 1, chunk));
  return map;
}

// ---- Happy path: identity comes FROM the store-backed markerMap --------------

describe("Phase 8 (8b) — resolveCitations happy path (identity is app-owned)", () => {
  it("resolves markers [1,2] to TrustedCitations whose IDENTITY equals the markerMap chunks", () => {
    const c0 = makeChunk({
      chunkId: "chunk-a",
      documentId: "document-a",
      sourceUri: "knowledge/corpus/a.md",
      chunkIndex: 3,
      ownerId: "owner-1",
      contentHash: "hash-a",
      score: 0.9,
    });
    const c1 = makeChunk({
      chunkId: "chunk-b",
      documentId: "document-b",
      sourceUri: "knowledge/corpus/b.md",
      chunkIndex: 7,
      ownerId: "owner-1",
      contentHash: "hash-b",
      score: 0.5,
    });
    const markerMap = makeMarkerMap([c0, c1]);

    const { resolved, dropped } = resolveCitations({
      citations: [1, 2],
      markerMap,
    });

    expect(dropped).toEqual([]);
    expect(resolved).toHaveLength(2);

    // Every identity field EQUALS the store-backed RetrievedChunk (NOT any model
    // string) — this is the crux of "resolve to retrieved source IDs".
    const [r0, r1] = resolved as [TrustedCitation, TrustedCitation];
    expect(r0).toEqual({
      marker: 1,
      chunkId: c0.chunkId,
      documentId: c0.documentId,
      sourceUri: c0.sourceUri,
      chunkIndex: c0.chunkIndex,
      ownerId: c0.ownerId,
      contentHash: c0.contentHash,
      score: c0.score,
    });
    expect(r1.chunkId).toBe(c1.chunkId);
    expect(r1.documentId).toBe(c1.documentId);
    expect(r1.sourceUri).toBe(c1.sourceUri);
    expect(r1.chunkIndex).toBe(c1.chunkIndex);
    expect(r1.ownerId).toBe(c1.ownerId);
    expect(r1.contentHash).toBe(c1.contentHash);
    expect(r1.score).toBe(c1.score);
  });

  it("carries a null sourceUri straight through from the store chunk", () => {
    const c0 = makeChunk({ chunkId: "chunk-a", sourceUri: null });
    const { resolved } = resolveCitations({
      citations: [1],
      markerMap: makeMarkerMap([c0]),
    });
    expect(resolved[0]!.sourceUri).toBeNull();
  });
});

// ---- Hallucinated / unknown markers are DROPPED (the crux of criterion 1) ---

describe("Phase 8 (8b) — hallucinated markers are dropped (unknown_marker)", () => {
  it("drops a marker the app never assigned (out of range) and never resolves it", () => {
    const c0 = makeChunk({ chunkId: "chunk-a" });
    const markerMap = makeMarkerMap([c0]); // only marker 1 exists

    // 99 is wildly out of range; 2 is a PLAUSIBLE-LOOKING index that is still
    // not app-assigned (only 1 chunk was retrieved) — both must be dropped.
    const { resolved, dropped } = resolveCitations({
      citations: [1, 2, 99],
      markerMap,
    });

    expect(resolved.map((r) => r.marker)).toEqual([1]);
    expect(dropped).toEqual([
      { marker: 2, reason: "unknown_marker" },
      { marker: 99, reason: "unknown_marker" },
    ]);
    // The model could NOT introduce a citation the app didn't assign.
    expect(resolved.some((r) => r.marker === 2 || r.marker === 99)).toBe(false);
  });

  it("drops marker 0 (markers are 1-based; 0 is never app-assigned)", () => {
    const c0 = makeChunk({ chunkId: "chunk-a" });
    const { resolved, dropped } = resolveCitations({
      citations: [0],
      markerMap: makeMarkerMap([c0]),
    });
    expect(resolved).toEqual([]);
    expect(dropped).toEqual([{ marker: 0, reason: "unknown_marker" }]);
  });

  it("only consumes integer markers — there is NO path that reads an id from model output", () => {
    // groundedAnswerSchema guarantees citations are integers, so a chunk/document
    // id string can never reach the resolver. Defensively, a non-integer that
    // somehow slips through a custom generateFn is treated as unknown, never
    // trusted (proving identity is only ever taken from the markerMap chunk).
    const c0 = makeChunk({ chunkId: "chunk-a" });
    // Deliberately hostile input: non-integer values that the schema would
    // reject upstream but which must still be treated as unknown here.
    const { resolved, dropped } = resolveCitations({
      citations: [1.5, Number.NaN],
      markerMap: makeMarkerMap([c0]),
    });
    expect(resolved).toEqual([]);
    expect(dropped.map((d) => d.reason)).toEqual(["unknown_marker", "unknown_marker"]);
  });
});

// ---- Authorization (defense-in-depth) --------------------------------------

describe("Phase 8 (8b) — authorization drops (unauthorized), defense-in-depth", () => {
  it("under an ownerId scope: matching owner kept, mismatched owner dropped", () => {
    const mine = makeChunk({ chunkId: "chunk-mine", ownerId: "owner-1" });
    const theirs = makeChunk({ chunkId: "chunk-theirs", ownerId: "owner-2" });
    const markerMap = makeMarkerMap([mine, theirs]);

    const { resolved, dropped } = resolveCitations({
      citations: [1, 2],
      markerMap,
      ownerId: "owner-1",
    });

    expect(resolved.map((r) => r.chunkId)).toEqual(["chunk-mine"]);
    expect(dropped).toEqual([{ marker: 2, reason: "unauthorized" }]);
  });

  it("null-owner (PUBLIC) chunk UNDER A SCOPE is KEPT (Option 1 public-unowned rule)", () => {
    // NEW rule (flipped from the OLD "null-owner under scope -> unauthorized"):
    // a null-owner chunk is PUBLIC knowledge and is visible under ANY owner scope,
    // exactly matching the retriever's public-unowned filter.
    const nullOwner = makeChunk({ chunkId: "chunk-null", ownerId: null });
    const { resolved, dropped } = resolveCitations({
      citations: [1],
      markerMap: makeMarkerMap([nullOwner]),
      ownerId: "owner-1",
    });
    expect(resolved.map((r) => r.chunkId)).toEqual(["chunk-null"]);
    expect(dropped).toEqual([]);
  });

  it("SCOPED mix: PUBLIC (null) kept, owner-A kept, owner-B dropped as unauthorized", () => {
    // The citation-side mirror of the retriever's public-unowned regression test:
    // under owner-A, a public chunk + the requester's chunk resolve, and only a
    // DIFFERENT non-null owner's chunk is dropped (owned-doc isolation preserved).
    const pub = makeChunk({ chunkId: "chunk-public", ownerId: null });
    const mine = makeChunk({ chunkId: "chunk-a", ownerId: "owner-A" });
    const theirs = makeChunk({ chunkId: "chunk-b", ownerId: "owner-B" });
    const markerMap = makeMarkerMap([pub, mine, theirs]);

    const { resolved, dropped } = resolveCitations({
      citations: [1, 2, 3],
      markerMap,
      ownerId: "owner-A",
    });

    expect(resolved.map((r) => r.chunkId)).toEqual(["chunk-public", "chunk-a"]);
    expect(dropped).toEqual([{ marker: 3, reason: "unauthorized" }]);
  });

  it("NO scope (ownerId absent): ownership NOT enforced (matches retrieval scope)", () => {
    const nullOwner = makeChunk({ chunkId: "chunk-null", ownerId: null });
    const other = makeChunk({ chunkId: "chunk-other", ownerId: "owner-2" });
    const markerMap = makeMarkerMap([nullOwner, other]);

    // No ownerId -> both kept (a public/unowned store + no scope is allowed).
    const { resolved, dropped } = resolveCitations({
      citations: [1, 2],
      markerMap,
    });
    expect(resolved.map((r) => r.chunkId)).toEqual(["chunk-null", "chunk-other"]);
    expect(dropped).toEqual([]);
  });

  it("ownerId: null is treated the same as absent (no enforcement)", () => {
    const other = makeChunk({ chunkId: "chunk-other", ownerId: "owner-2" });
    const { resolved, dropped } = resolveCitations({
      citations: [1],
      markerMap: makeMarkerMap([other]),
      ownerId: null,
    });
    expect(resolved.map((r) => r.chunkId)).toEqual(["chunk-other"]);
    expect(dropped).toEqual([]);
  });
});

// ---- Dedupe + ordering ------------------------------------------------------

describe("Phase 8 (8b) — dedupe and first-cited ordering", () => {
  it("resolves a repeated marker exactly once (dedupe by marker)", () => {
    const c0 = makeChunk({ chunkId: "chunk-a" });
    const c1 = makeChunk({ chunkId: "chunk-b" });
    const markerMap = makeMarkerMap([c0, c1]);

    const { resolved, dropped } = resolveCitations({
      citations: [2, 1, 2, 1, 2],
      markerMap,
    });

    // FIRST-CITED order: 2 was cited before 1, and each appears once.
    expect(resolved.map((r) => r.marker)).toEqual([2, 1]);
    expect(dropped).toEqual([]);
  });

  it("a repeated UNKNOWN marker is recorded as dropped only once", () => {
    const c0 = makeChunk({ chunkId: "chunk-a" });
    const { resolved, dropped } = resolveCitations({
      citations: [99, 99, 1],
      markerMap: makeMarkerMap([c0]),
    });
    expect(resolved.map((r) => r.marker)).toEqual([1]);
    expect(dropped).toEqual([{ marker: 99, reason: "unknown_marker" }]);
  });

  it("resolved list preserves FIRST-CITED order, not ascending-marker order", () => {
    const chunks = [
      makeChunk({ chunkId: "c1" }),
      makeChunk({ chunkId: "c2" }),
      makeChunk({ chunkId: "c3" }),
    ];
    const { resolved } = resolveCitations({
      citations: [3, 1, 2],
      markerMap: makeMarkerMap(chunks),
    });
    expect(resolved.map((r) => r.marker)).toEqual([3, 1, 2]);
  });
});

// ---- Empty input ------------------------------------------------------------

describe("Phase 8 (8b) — empty citations", () => {
  it("empty citations -> empty resolved + empty dropped", () => {
    const c0 = makeChunk({ chunkId: "chunk-a" });
    const { resolved, dropped } = resolveCitations({
      citations: [],
      markerMap: makeMarkerMap([c0]),
    });
    expect(resolved).toEqual([]);
    expect(dropped).toEqual([]);
  });

  it("empty markerMap -> every marker dropped as unknown", () => {
    const { resolved, dropped } = resolveCitations({
      citations: [1, 2],
      markerMap: new Map(),
    });
    expect(resolved).toEqual([]);
    expect(dropped).toEqual([
      { marker: 1, reason: "unknown_marker" },
      { marker: 2, reason: "unknown_marker" },
    ]);
  });
});

// ---- Redacted summary -------------------------------------------------------

describe("Phase 8 (8b) — summarizeCitationResolution", () => {
  it("produces a single-line redacted summary of kept + dropped markers", () => {
    const c0 = makeChunk({ chunkId: "chunk-a" });
    const result = resolveCitations({
      citations: [1, 99],
      markerMap: makeMarkerMap([c0]),
    });
    const summary = summarizeCitationResolution(result);
    expect(summary).toContain("kept [1]");
    expect(summary).toContain("99:unknown_marker");
    expect(summary).not.toContain("\n");
  });
});
