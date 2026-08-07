import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import {
  createKnowledgeStore,
  computeDocumentId,
  type KnowledgeStore,
} from "../src/core/knowledge-store.js";
import { ingestDocuments, type DocumentEmbedder } from "../src/core/ingestion.js";
import {
  retrieve,
  distanceToScore,
  createQueryEmbedder,
  EmptyQueryError,
  InvalidKError,
  QueryEmbeddingDimensionError,
  type QueryEmbedder,
} from "../src/core/retriever.js";

// Phase 7 (7e) — RETRIEVER offline coverage.
//
// Fully OFFLINE + DETERMINISTIC:
//   - A REAL knowledge store + REAL sqlite-vec over a per-test temp DB.
//   - A deterministic FAKE DocumentEmbedder (for ingest) and a deterministic
//     FAKE QueryEmbedder (for retrieve). NO model, NO network is ever touched.
//   - Fake embeddings are crafted so the nearest neighbor is DETERMINISTIC and
//     asserted. Every DB lives under os.tmpdir() and is removed in afterEach, so
//     `npm test` NEVER writes DB artifacts into the repo working tree.

// Small store dimension keeps fake vectors tiny + fast; same code paths as 768.
const DIM = 8;

// A one-hot vector: index `slot` = 1, all others 0. Two different slots are at
// L2 distance sqrt(2); identical slots are at distance 0. This makes the nearest
// neighbor for a given query slot fully deterministic and hand-checkable.
function oneHot(slot: number, dim = DIM): number[] {
  const v = new Array<number>(dim).fill(0);
  v[slot % dim] = 1;
  return v;
}

// The corpus: each source maps to a distinct one-hot slot. We embed the WHOLE
// document as a single chunk by using a large chunk size (each doc is tiny), so
// documentId <-> chunk <-> slot is 1:1 and assertions are exact.
const CORPUS: { sourceUri: string; content: string; slot: number; owner: string | null }[] = [
  { sourceUri: "knowledge/corpus/a.md", content: "alpha document about venues", slot: 0, owner: "owner-1" },
  { sourceUri: "knowledge/corpus/b.md", content: "bravo document about budget", slot: 1, owner: "owner-1" },
  { sourceUri: "knowledge/corpus/c.md", content: "charlie document about catering", slot: 2, owner: "owner-2" },
  { sourceUri: "knowledge/corpus/d.md", content: "delta document about timeline", slot: 3, owner: null },
];

// Map a document's content back to its slot so the fake DOCUMENT embedder returns
// the right one-hot vector for whichever chunk text it is handed.
function slotForContent(content: string): number {
  const found = CORPUS.find((d) => content.includes(d.content));
  return found ? found.slot : DIM - 1;
}

// Deterministic fake document embedder: one-hot per chunk based on its content.
function makeDocEmbedder(): DocumentEmbedder {
  return {
    embedDocuments(texts: string[]): Promise<number[][]> {
      return Promise.resolve(texts.map((t) => oneHot(slotForContent(t))));
    },
  };
}

// A deterministic fake query embedder that returns a chosen one-hot slot.
function makeQueryEmbedder(slot: number): QueryEmbedder {
  return {
    embedQuery(_text: string): Promise<number[]> {
      return Promise.resolve(oneHot(slot));
    },
  };
}

// A fake query embedder that returns a WRONG-dimension vector (dimension guard).
function makeWrongDimQueryEmbedder(wrongDim: number): QueryEmbedder {
  return {
    embedQuery(_text: string): Promise<number[]> {
      return Promise.resolve(new Array<number>(wrongDim).fill(0));
    },
  };
}

let tempDir: string;
let store: KnowledgeStore;

async function ingestCorpus(s: KnowledgeStore): Promise<void> {
  await ingestDocuments({
    store: s,
    embedder: makeDocEmbedder(),
    documents: CORPUS.map((d) => ({
      content: d.content,
      sourceUri: d.sourceUri,
      ownerId: d.owner,
    })),
    // Large chunk size so each tiny doc is exactly one chunk (1:1 doc<->slot).
    chunking: { chunkSize: 1000, chunkOverlap: 0 },
  });
}

beforeEach(async () => {
  tempDir = await mkdtemp(path.join(tmpdir(), "wp-retriever-"));
  store = createKnowledgeStore({ dbPath: path.join(tempDir, "k.sqlite"), embeddingDim: DIM });
});

afterEach(async () => {
  store.close();
  await rm(tempDir, { recursive: true, force: true });
});

describe("Phase 7 (7e) — distanceToScore (L2 -> similarity transform)", () => {
  it("distance 0 maps to score 1", () => {
    expect(distanceToScore(0)).toBe(1);
  });
  it("is strictly decreasing in distance and stays in (0,1]", () => {
    const s1 = distanceToScore(1);
    const s2 = distanceToScore(2);
    expect(s1).toBeGreaterThan(s2);
    expect(s1).toBeLessThan(1);
    expect(s2).toBeGreaterThan(0);
  });
  it("clamps negative / non-finite distance to 0 defensively", () => {
    expect(distanceToScore(-1)).toBe(0);
    expect(distanceToScore(Number.NaN)).toBe(0);
    expect(distanceToScore(Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe("Phase 7 (7e) — retrieve (core)", () => {
  it("returns the nearest document first with TRUSTED, app-owned metadata", async () => {
    await ingestCorpus(store);
    // Query at slot 2 -> nearest is corpus[2] (charlie / c.md).
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(2),
      query: "anything about catering",
      k: 3,
    });
    expect(results.length).toBeGreaterThan(0);
    const top = results[0]!;
    // The nearest neighbor is the exact matching document (distance 0, score 1).
    expect(top.sourceUri).toBe("knowledge/corpus/c.md");
    expect(top.documentId).toBe(computeDocumentId("knowledge/corpus/c.md"));
    expect(top.chunkIndex).toBe(0);
    expect(top.ownerId).toBe("owner-2");
    expect(top.distance).toBeCloseTo(0, 6);
    expect(top.score).toBeCloseTo(1, 6);
    // contentHash is the STORE's chunk content hash, pulled from the store (not
    // model text): it must equal what the store recorded for that chunk.
    const chunk = store.getChunk(top.chunkId);
    expect(chunk).not.toBeNull();
    expect(top.contentHash).toBe(chunk!.contentHash);
    expect(top.documentId).toBe(chunk!.documentId);
  });

  it("orders results best-first (ascending distance == descending score)", async () => {
    await ingestCorpus(store);
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "venues",
      k: 4,
    });
    // Distances must be non-decreasing and scores non-increasing.
    for (let i = 1; i < results.length; i++) {
      expect(results[i]!.distance).toBeGreaterThanOrEqual(results[i - 1]!.distance);
      expect(results[i]!.score).toBeLessThanOrEqual(results[i - 1]!.score);
    }
    // The exact match ranks first.
    expect(results[0]!.sourceUri).toBe("knowledge/corpus/a.md");
  });

  it("caps result length at k", async () => {
    await ingestCorpus(store);
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(1),
      query: "budget",
      k: 2,
    });
    expect(results.length).toBe(2);
  });

  it("returns ALL chunks when k exceeds the corpus size", async () => {
    await ingestCorpus(store);
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(1),
      query: "budget",
      k: 100,
    });
    expect(results.length).toBe(CORPUS.length);
  });

  it("returns [] for an EMPTY store", async () => {
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "nothing ingested yet",
      k: 5,
    });
    expect(results).toEqual([]);
  });

  it("rejects an empty / whitespace-only query with EmptyQueryError (no embed)", async () => {
    await ingestCorpus(store);
    let embedCalled = false;
    const spyEmbedder: QueryEmbedder = {
      embedQuery(_t: string): Promise<number[]> {
        embedCalled = true;
        return Promise.resolve(oneHot(0));
      },
    };
    await expect(
      retrieve({ store, queryEmbedder: spyEmbedder, query: "   ", k: 3 })
    ).rejects.toBeInstanceOf(EmptyQueryError);
    expect(embedCalled).toBe(false);
  });

  it("rejects k <= 0 and non-integer k with InvalidKError (no embed)", async () => {
    await ingestCorpus(store);
    let embedCalled = false;
    const spyEmbedder: QueryEmbedder = {
      embedQuery(_t: string): Promise<number[]> {
        embedCalled = true;
        return Promise.resolve(oneHot(0));
      },
    };
    await expect(
      retrieve({ store, queryEmbedder: spyEmbedder, query: "q", k: 0 })
    ).rejects.toBeInstanceOf(InvalidKError);
    await expect(
      retrieve({ store, queryEmbedder: spyEmbedder, query: "q", k: -3 })
    ).rejects.toBeInstanceOf(InvalidKError);
    await expect(
      retrieve({ store, queryEmbedder: spyEmbedder, query: "q", k: 2.5 })
    ).rejects.toBeInstanceOf(InvalidKError);
    expect(embedCalled).toBe(false);
  });

  it("throws a typed, redacted QueryEmbeddingDimensionError on a dimension mismatch", async () => {
    await ingestCorpus(store);
    let thrown: unknown;
    try {
      await retrieve({
        store,
        queryEmbedder: makeWrongDimQueryEmbedder(DIM + 3),
        query: "q",
        k: 3,
      });
    } catch (err) {
      thrown = err;
    }
    expect(thrown).toBeInstanceOf(QueryEmbeddingDimensionError);
    const e = thrown as QueryEmbeddingDimensionError;
    expect(e.expectedDim).toBe(DIM);
    expect(e.actualDim).toBe(DIM + 3);
    // Nothing sensitive is leaked; the message is a bounded, single-line reason.
    expect(e.message).not.toContain("\n");
    expect(e.message).toContain("dimension mismatch");
  });

  it("owner-scoped retrieval restricts results to the requester's OWNED docs + PUBLIC docs", async () => {
    await ingestCorpus(store);
    // Query at slot 0 (owner-1's a.md is nearest). Scope to owner-2: owner-2's
    // OWNED docs (c.md) AND the PUBLIC/unowned d.md are visible (Option 1
    // public-unowned rule); NO owner-1 docs (cross-owner isolation preserved).
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "venues",
      k: 5,
      ownerId: "owner-2",
    });
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      // Every hit is either owner-2's own doc or a PUBLIC (null-owner) doc.
      expect(r.ownerId === "owner-2" || r.ownerId === null).toBe(true);
    }
    const uris = results.map((r) => r.sourceUri);
    expect(uris).toContain("knowledge/corpus/c.md"); // owner-2's owned doc
    expect(uris).toContain("knowledge/corpus/d.md"); // PUBLIC (unowned) doc — NEW
    // owner-1's OWNED docs remain private (isolation preserved).
    expect(uris).not.toContain("knowledge/corpus/a.md");
    expect(uris).not.toContain("knowledge/corpus/b.md");
  });

  it("owner-scoped retrieval still returns PUBLIC docs when the owner has no OWNED docs", async () => {
    await ingestCorpus(store);
    // An owner with zero OWNED documents still sees PUBLIC (unowned) docs: d.md is
    // the only unowned doc, so exactly it is returned (NEW public-unowned rule;
    // OLD rule returned []).
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "venues",
      k: 5,
      ownerId: "owner-does-not-exist",
    });
    expect(results.map((r) => r.sourceUri)).toEqual(["knowledge/corpus/d.md"]);
    expect(results.every((r) => r.ownerId === null)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// PUBLIC-UNOWNED authorization rule (Option 1 fix). A regression guard for the
// ingest -> grounded-serve bug: the corpus is ingested UNOWNED (ownerId=null),
// but the grounded server ALWAYS scopes retrieval to a non-null authenticated
// ownerId. The OLD filter dropped null-owner chunks under a scope, so grounded
// serve returned ZERO corpus hits. NEW rule: an UNOWNED (null) document is
// PUBLIC — visible under ANY owner scope — while OWNED docs stay private to their
// owner. These tests exercise the SCOPED path (the eval runners retrieve
// UNSCOPED, which is exactly why the eval-green evidence could not catch this).
//
// A dedicated 3-owner corpus: one PUBLIC doc, one owned by owner-A, one by
// owner-B, each on its own one-hot slot so KNN order is deterministic.
const OWNERSHIP_CORPUS: {
  sourceUri: string;
  content: string;
  slot: number;
  owner: string | null;
}[] = [
  { sourceUri: "knowledge/corpus/public.md", content: "public corpus knowledge about venues", slot: 0, owner: null },
  { sourceUri: "knowledge/corpus/owner-a.md", content: "owner A private note about budget", slot: 1, owner: "owner-A" },
  { sourceUri: "knowledge/corpus/owner-b.md", content: "owner B private note about catering", slot: 2, owner: "owner-B" },
];

function slotForOwnershipContent(content: string): number {
  const found = OWNERSHIP_CORPUS.find((d) => content.includes(d.content));
  return found ? found.slot : DIM - 1;
}

function makeOwnershipDocEmbedder(): DocumentEmbedder {
  return {
    embedDocuments(texts: string[]): Promise<number[][]> {
      return Promise.resolve(texts.map((t) => oneHot(slotForOwnershipContent(t))));
    },
  };
}

async function ingestOwnershipCorpus(s: KnowledgeStore): Promise<void> {
  await ingestDocuments({
    store: s,
    embedder: makeOwnershipDocEmbedder(),
    documents: OWNERSHIP_CORPUS.map((d) => ({
      content: d.content,
      sourceUri: d.sourceUri,
      ownerId: d.owner,
    })),
    chunking: { chunkSize: 1000, chunkOverlap: 0 },
  });
}

describe("Phase 7 (7e) — PUBLIC-unowned authorization rule (Option 1 fix)", () => {
  it("under owner-A scope: PUBLIC doc IS returned, owner-A doc IS returned, owner-B doc is NOT (isolation preserved)", async () => {
    await ingestOwnershipCorpus(store);
    // k >= corpus size so authorization (not k-underfill) is the only reason a
    // doc could be absent. Every doc is at distance <= sqrt(2), so all are
    // candidates; the owner filter is what excludes owner-B.
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "venues",
      k: 5,
      ownerId: "owner-A",
    });
    const uris = results.map((r) => r.sourceUri);
    // NEW behavior (the regression guard): the PUBLIC (null-owner) doc is visible
    // under a non-null scope.
    expect(uris).toContain("knowledge/corpus/public.md");
    // The requester's own owned doc is visible.
    expect(uris).toContain("knowledge/corpus/owner-a.md");
    // Cross-owner isolation preserved: owner-B's owned doc is NEVER visible.
    expect(uris).not.toContain("knowledge/corpus/owner-b.md");
    // Every returned chunk is either public (null) or owned by the requester.
    for (const r of results) {
      expect(r.ownerId === null || r.ownerId === "owner-A").toBe(true);
    }
  });

  it("with NO ownerId scope: all three docs (public + owner-A + owner-B) are returned (unchanged)", async () => {
    await ingestOwnershipCorpus(store);
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "venues",
      k: 5,
    });
    const uris = results.map((r) => r.sourceUri);
    expect(uris).toContain("knowledge/corpus/public.md");
    expect(uris).toContain("knowledge/corpus/owner-a.md");
    expect(uris).toContain("knowledge/corpus/owner-b.md");
    expect(results.length).toBe(OWNERSHIP_CORPUS.length);
  });

  it("a PURELY-PUBLIC corpus is fully retrievable under a NON-NULL owner scope (the exact ingest->serve scenario)", async () => {
    // Mirror the real bug: ingest ONLY unowned (null-owner) corpus docs, then
    // retrieve under an authenticated (non-null) ownerId — as the grounded server
    // does. Every public doc MUST be retrievable (OLD rule returned zero).
    await ingestDocuments({
      store,
      embedder: makeOwnershipDocEmbedder(),
      documents: [
        { content: "public corpus knowledge about venues", sourceUri: "knowledge/corpus/public.md", ownerId: null },
        { content: "owner A private note about budget", sourceUri: "knowledge/corpus/pub2.md", ownerId: null },
      ],
      chunking: { chunkSize: 1000, chunkOverlap: 0 },
    });
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "anything grounded",
      k: 5,
      ownerId: "some-authenticated-user",
    });
    expect(results.length).toBe(2);
    for (const r of results) {
      expect(r.ownerId).toBeNull();
    }
    expect(results.map((r) => r.sourceUri)).toContain("knowledge/corpus/public.md");
  });
});

describe("Phase 7 (7e) — createQueryEmbedder (production adapter is a factory)", () => {
  it("is a function that returns a QueryEmbedder shape (no live call here)", () => {
    // We do NOT invoke embedQuery (that would need a live proxy); we only assert
    // the adapter constructs without a network call and exposes the contract.
    const embedder = createQueryEmbedder({ model: "some-embedding-alias" });
    expect(typeof embedder.embedQuery).toBe("function");
  });
});
