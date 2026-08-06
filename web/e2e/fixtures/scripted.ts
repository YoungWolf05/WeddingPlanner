// Phase 9 (9d): the SCRIPTED grounded-answer values, shared by the harness
// (which emits them through the REAL server) and the specs (which assert the
// browser renders them). Centralizing them here is the crux of the WIRE-COMPAT
// proof: the specs assert the rendered citation fields EQUAL these exact
// app-owned values, so a drift between the 9c re-declared v2 wire types and the
// live backend projection would fail the E2E.

// The supported-answer text the harness scripts (contains the [1] marker).
export const SUPPORTED_ANSWER =
  "Consider an outdoor garden venue for a spring wedding. [1]";

// The single app-owned citation the harness emits for a supported turn. These
// are the EXACT fields the browser must render (marker/sourceUri/documentId/
// chunkIndex/score). `ownerId` is intentionally NOT here — the server drops it
// from the wire, so the browser never sees it (asserted by its absence in the
// rendered DOM).
export const SCRIPTED_CITATION = {
  marker: 1,
  chunkId: "e2e-chunk-0001",
  documentId: "e2e-doc-venues",
  sourceUri: "knowledge/corpus/venues.md",
  chunkIndex: 0,
  contentHash: "e2e-contenthash-abc123",
  score: 0.912,
} as const;

// The trigger substring that steers the harness to the insufficient-evidence
// case (case-insensitive match on the message text).
export const INSUFFICIENT_TRIGGER = "insufficient";
