import { OpenAIEmbeddings } from "@langchain/openai";
import { config } from "../config.js";

export interface EmbeddingsOptions {
  model?: string;
  // OPTIONAL requested output dimension. When provided, it is sent to the alias
  // (a Matryoshka/truncation request); when OMITTED, the factory sends NO
  // `dimensions` and the alias returns its NATIVE dimension. This is opt-in by
  // DESIGN (see WIRE CONTRACT below): the factory NEVER defaults it to
  // config.embedDim, so it never silently truncates for callers that did not ask.
  // Callers that must match a fixed store dimension pass it EXPLICITLY (the 7e
  // ingestion + query embedder adapters pass dimensions: config.embedDim); the
  // 7d embedding/capability probe OMITS it so it can observe (and verify) the
  // alias's native dimension against config.embedDim.
  dimensions?: number;
}

// Builds an OpenAIEmbeddings client pointed at the corporate LiteLLM endpoint.
//
// Embeddings are a SEPARATE provider contract from chat (see docs/roadmap.md
// Phase 4): a chat alias is never assumed to work as an embedding alias. This
// factory mirrors the LiteLLM wiring of createChatModel() in model.ts (same
// apiKey and baseURL, sourced from config) but for the embeddings endpoint.
//
// The embedding alias is required here: callers must supply one explicitly or
// have LITELLM_EMBED_MODEL configured. Unlike the chat model, there is no
// sensible default alias, so we fail loudly instead of guessing.
//
// WIRE CONTRACT (7e): two independent settings, with DIFFERENT default policies:
//   - encodingFormat "float" is set UNCONDITIONALLY and is REQUIRED because the
//     proxy returns float arrays. `@langchain/openai` (via the openai SDK)
//     requests `encoding_format: "base64"` by DEFAULT and decodes the base64
//     field. The corporate LiteLLM proxy returns embeddings as a FLOAT ARRAY even
//     when base64 is requested, so the default base64 decode path misreads the
//     response and yields ALL-ZERO vectors (verified live: the raw /embeddings
//     response is a meaningful unit-norm float array, but the base64-decoded
//     langchain result is zeros). Requesting "float" makes langchain read the
//     float array directly and produce correct vectors. Discovered during the 7e
//     live retrieval eval; the 7d dimension probe passed only because it checks
//     the vector LENGTH, not its content.
//   - dimensions is OPTIONAL and OPT-IN: it is sent ONLY when the caller provides
//     options.dimensions, and is OMITTED otherwise (the SDK then sends no
//     `dimensions` and the alias returns its NATIVE dimension). Requesting a
//     dimension ASSUMES a truncation-capable (Matryoshka) alias, so the factory
//     must NOT force it on everyone: coupling all future aliases to Matryoshka
//     support (and making the 7d probe's dimension check tautological — verifying
//     the proxy HONORED the request rather than the alias's native size) is
//     exactly the bug this design avoids. Instead:
//       * callers that must match the knowledge store's FIXED dimension pass it
//         EXPLICITLY (the 7e adapters createDocumentEmbedder /
//         createQueryEmbedder pass dimensions: config.embedDim, e.g. 768 via
//         gemini-embedding-001 Matryoshka truncation), and
//       * the 7d embedding/capability probe OMITS it so it observes the alias's
//         NATIVE dimension and can classify native-vs-expected (config.embedDim),
//         re-enabling detection of a truly wrong-dimension alias.
export function createEmbeddingsModel(
  options: EmbeddingsOptions = {}
): OpenAIEmbeddings {
  const model = options.model ?? config.embedModel;
  if (!model) {
    throw new Error(
      "No embedding alias configured. Pass options.model or set " +
        "LITELLM_EMBED_MODEL in your environment."
    );
  }

  return new OpenAIEmbeddings({
    model,
    apiKey: config.apiKey,
    // OPT-IN dimension (see WIRE CONTRACT): pass through ONLY when the caller
    // provided it. When undefined the SDK omits `dimensions` entirely, so the
    // alias returns its native dimension — we deliberately do NOT default this to
    // config.embedDim here.
    dimensions: options.dimensions,
    // See WIRE CONTRACT above: float (not the default base64) is REQUIRED for
    // this proxy to return non-zero vectors.
    encodingFormat: "float",
    configuration: {
      baseURL: config.baseURL,
    },
  });
}
