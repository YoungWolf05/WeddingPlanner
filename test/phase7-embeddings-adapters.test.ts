import { describe, it, expect, vi, beforeEach } from "vitest";
import type { EmbeddingsOptions } from "../src/core/embeddings.js";
import { config } from "../src/config.js";

// Phase 7 (7e) — OFFLINE coverage that the STORE-WRITING adapters request an
// EXPLICIT dimension equal to the store's fixed dimension (config.embedDim).
//
// The store-writing adapters (createDocumentEmbedder in ingestion.ts and
// createQueryEmbedder in retriever.ts) wrap createEmbeddingsModel, so we MOCK the
// single factory to CAPTURE the exact EmbeddingsOptions each adapter passes. This
// makes the assertion direct and network-free: we never construct a real
// OpenAIEmbeddings and never make a live call.
//
// Why this matters: the factory no longer defaults `dimensions` (opt-in by
// design). Vectors written to / queried against the fixed-dimension knowledge
// store MUST be config.embedDim (e.g. 768 via gemini-embedding-001 Matryoshka
// truncation), so these adapters must make the request EXPLICIT. This test is the
// code-level confirmation that live ingestion/retrieval still produce
// store-compatible 768-dim vectors, keeping the committed retrieval evidence
// representative.

// Capture every options object passed to the mocked factory.
const factoryCalls: EmbeddingsOptions[] = [];

vi.mock("../src/core/embeddings.js", () => ({
  createEmbeddingsModel: (options: EmbeddingsOptions = {}) => {
    factoryCalls.push(options);
    // Return a minimal stand-in with the embedder methods the adapters wrap. The
    // adapters only call these on live use, which we never do here.
    return {
      embedDocuments: async (_texts: string[]): Promise<number[][]> => [],
      embedQuery: async (_text: string): Promise<number[]> => [],
    };
  },
}));

// Import the adapters AFTER the mock is registered (vi.mock is hoisted, so these
// static imports already see the mocked factory).
const { createDocumentEmbedder } = await import("../src/core/ingestion.js");
const { createQueryEmbedder } = await import("../src/core/retriever.js");

beforeEach(() => {
  factoryCalls.length = 0;
});

describe("Phase 7 (7e) — store-writing adapters request an explicit store dimension", () => {
  it("createDocumentEmbedder requests dimensions === config.embedDim by default", () => {
    createDocumentEmbedder({ model: "some-embed" });
    expect(factoryCalls).toHaveLength(1);
    expect(factoryCalls[0]!.dimensions).toBe(config.embedDim);
    expect(factoryCalls[0]!.model).toBe("some-embed");
  });

  it("createQueryEmbedder requests dimensions === config.embedDim by default", () => {
    createQueryEmbedder({ model: "some-embed" });
    expect(factoryCalls).toHaveLength(1);
    expect(factoryCalls[0]!.dimensions).toBe(config.embedDim);
    expect(factoryCalls[0]!.model).toBe("some-embed");
  });

  it("createDocumentEmbedder still requests the store dimension when no options given", () => {
    createDocumentEmbedder();
    expect(factoryCalls).toHaveLength(1);
    expect(factoryCalls[0]!.dimensions).toBe(config.embedDim);
  });

  it("createQueryEmbedder still requests the store dimension when no options given", () => {
    createQueryEmbedder();
    expect(factoryCalls).toHaveLength(1);
    expect(factoryCalls[0]!.dimensions).toBe(config.embedDim);
  });

  it("an explicit caller-provided dimensions overrides the store-dimension default", () => {
    // Defense-in-depth: a caller that deliberately asks for a different size wins
    // (the adapter spreads caller options AFTER the default).
    createDocumentEmbedder({ model: "some-embed", dimensions: 256 });
    expect(factoryCalls[0]!.dimensions).toBe(256);
  });
});
