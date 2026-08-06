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

  it("owner-scoped retrieval restricts results to the requested owner", async () => {
    await ingestCorpus(store);
    // Query at slot 0 (owner-1's a.md is nearest). Scope to owner-2: only
    // owner-2's documents may appear (c.md), and NONE from owner-1 or the
    // unowned d.md.
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "venues",
      k: 5,
      ownerId: "owner-2",
    });
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(r.ownerId).toBe("owner-2");
    }
    const uris = results.map((r) => r.sourceUri);
    expect(uris).toContain("knowledge/corpus/c.md");
    expect(uris).not.toContain("knowledge/corpus/a.md");
    expect(uris).not.toContain("knowledge/corpus/b.md");
    expect(uris).not.toContain("knowledge/corpus/d.md");
  });

  it("owner-scoped retrieval returns [] when the owner has no documents", async () => {
    await ingestCorpus(store);
    const results = await retrieve({
      store,
      queryEmbedder: makeQueryEmbedder(0),
      query: "venues",
      k: 5,
      ownerId: "owner-does-not-exist",
    });
    expect(results).toEqual([]);
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
