import { describe, it, expect } from "vitest";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createEmbeddingsModel } from "../src/core/embeddings.js";
import { config } from "../src/config.js";

// Phase 7 (7e) — OFFLINE coverage for the embeddings factory WIRE CONTRACT.
//
// Fully OFFLINE + DETERMINISTIC: constructing an OpenAIEmbeddings instance makes
// NO network call (the live call only happens on embedQuery/embedDocuments, which
// we never invoke here). We inspect the constructed instance's config the same
// way it is exposed on the object (`dimensions`, `encodingFormat`, `model`).
//
// This file pins the corrected 7e factory wiring:
//   - `encodingFormat: "float"` is sent UNCONDITIONALLY (the decode fix);
//   - `dimensions` is OPT-IN — sent ONLY when the caller provides it, NEVER
//     defaulted to config.embedDim, so an omitting caller (the 7d probe) observes
//     the alias's NATIVE dimension.
//
// The store-writing adapters' EXPLICIT dimension request is asserted separately
// in test/phase7-embeddings-adapters.test.ts (which mocks the factory to capture
// the exact options), because that assertion needs to intercept the factory call.

describe("Phase 7 (7e) — createEmbeddingsModel WIRE CONTRACT", () => {
  it("sets encodingFormat 'float' unconditionally", () => {
    const withDim = createEmbeddingsModel({ model: "some-embed", dimensions: 768 });
    const withoutDim = createEmbeddingsModel({ model: "some-embed" });
    expect(withDim.encodingFormat).toBe("float");
    expect(withoutDim.encodingFormat).toBe("float");
  });

  it("does NOT force a default dimensions when the caller omits it (native size)", () => {
    // The critical regression guard: omitting dimensions must leave it unset so
    // the SDK sends no `dimensions` and the alias returns its NATIVE dimension.
    // It must NOT be silently defaulted to config.embedDim.
    const model = createEmbeddingsModel({ model: "some-embed" });
    expect(model.dimensions).toBeUndefined();
    // Guard the specific regression: it is not the config default either.
    expect(model.dimensions).not.toBe(config.embedDim);
  });

  it("passes through an explicit dimensions when the caller provides it", () => {
    const model = createEmbeddingsModel({ model: "some-embed", dimensions: 1536 });
    expect(model.dimensions).toBe(1536);
  });

  it("constructs a real OpenAIEmbeddings pointed at the configured alias", () => {
    const model = createEmbeddingsModel({ model: "some-embed" });
    expect(model).toBeInstanceOf(OpenAIEmbeddings);
    expect(model.model).toBe("some-embed");
  });

  it("throws (does not silently guess) when no alias is configured", () => {
    // The offline env leaves LITELLM_EMBED_MODEL unset, so an omitted model must
    // fail loudly rather than default to some alias.
    expect(() => createEmbeddingsModel()).toThrow(/No embedding alias configured/i);
  });
});
