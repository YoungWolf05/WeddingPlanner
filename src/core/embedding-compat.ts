// Phase 7 (increment 7d) — EMBEDDING & DIMENSION COMPATIBILITY: PURE logic.
//
// This module contains only pure, deterministic, offline logic: the compatibility
// state vocabulary, the named dimension predicate, the classifier that turns an
// observed probe outcome into a state, and the functions that render a report
// into a console table + a dated Markdown evidence body.
//
// It performs NO network calls and reads NO credentials, so the offline Vitest
// suite imports it directly (like src/core/capabilities.ts / src/core/contracts.ts).
// All live probing I/O lives in src/probe-embedding.ts, which imports these
// functions. Redaction (host-only base URL, masked key, redacted error notes)
// happens UPSTREAM in the probe; this module renders exactly what it is given and
// never touches secrets.
//
// This is the explicit, NAMED compatibility check backing Phase 7 exit criterion
// 4 ("the embedding alias and vector dimensions are verified through the proxy
// and covered by compatibility checks"). The knowledge store already ENFORCES the
// dimension at write time (encodeEmbedding length guard) and on reopen
// (ensureVectorSchema mismatch throws); this module names and reports the check
// so a live probe can VERIFY the alias produces the expected dimension.

// One recorded compatibility state for an (alias, expected-dimension) assessment.
//
// - Compatible       — the alias produced a NON-EMPTY vector whose dimension
//                      EQUALS the expected/knowledge-store dimension.
// - DimensionMismatch — the alias produced a vector of a DIFFERENT dimension
//                      (a definitive "no", distinct from an infrastructure error).
// - Unverified       — no alias configured, or the alias was not probed (no
//                      observed dimension), so compatibility is simply unknown.
// - Error            — a network/auth/unexpected probe failure (the reason is
//                      redacted), distinct from a definitive DimensionMismatch.
export type EmbeddingCompatState =
  | "Compatible"
  | "DimensionMismatch"
  | "Unverified"
  | "Error";

// The reusable, NAMED dimension predicate. The knowledge store enforces the same
// equality at write time (encodeEmbedding) and on reopen (ensureVectorSchema);
// this is the explicit, unit-tested check the ingestion/probe wiring reads.
// Compatible iff BOTH dimensions are positive integers AND equal. A non-positive
// or non-integer input (e.g. an empty 0-length vector) is never compatible.
export function isEmbeddingDimensionCompatible(
  expectedDim: number,
  observedDim: number
): boolean {
  return (
    Number.isInteger(expectedDim) &&
    Number.isInteger(observedDim) &&
    expectedDim > 0 &&
    observedDim > 0 &&
    expectedDim === observedDim
  );
}

// The observable inputs to the compatibility classification, decoupled from the
// live I/O so the decision is deterministic and unit-testable offline.
export interface EmbeddingCompatInput {
  // The embedding alias probed, or null/undefined when none is configured.
  alias: string | null | undefined;
  // The expected dimension (the knowledge store's / config.embedDim value).
  expectedDim: number;
  // The dimension actually observed from the alias, when it was probed.
  observedDim?: number;
  // A concise, ALREADY-REDACTED probe error reason, when the probe failed. Its
  // presence takes precedence: an Error is distinct from a definitive mismatch.
  error?: string;
}

// The classifier's verdict.
export interface EmbeddingCompatClassification {
  state: EmbeddingCompatState;
  expectedDim: number;
  observedDim?: number;
  // A concise, already-redacted human note (never a secret/PII).
  note?: string;
}

// Classify an embedding-compatibility outcome. SINGLE source of truth, exercised
// by offline unit tests.
//
// Rules (evaluated in this order):
//   - A probe error present       -> Error (network/auth/unexpected). Distinct
//                                    from a definitive DimensionMismatch.
//   - No alias configured          -> Unverified.
//   - Alias but no observed dim    -> Unverified (not probed).
//   - observed == expected (>0)    -> Compatible.
//   - Otherwise                    -> DimensionMismatch.
export function classifyEmbeddingCompatibility(
  input: EmbeddingCompatInput
): EmbeddingCompatClassification {
  const { alias, expectedDim, observedDim, error } = input;

  if (error !== undefined) {
    return { state: "Error", expectedDim, observedDim, note: error };
  }
  if (!alias) {
    return {
      state: "Unverified",
      expectedDim,
      note: "no embedding alias configured (set LITELLM_EMBED_MODEL)",
    };
  }
  if (observedDim === undefined) {
    return {
      state: "Unverified",
      expectedDim,
      note: "embedding alias not probed (no observed dimension)",
    };
  }
  if (isEmbeddingDimensionCompatible(expectedDim, observedDim)) {
    return { state: "Compatible", expectedDim, observedDim };
  }
  return {
    state: "DimensionMismatch",
    expectedDim,
    observedDim,
    note:
      `alias produced ${observedDim} dimension(s) but the knowledge store ` +
      `expects ${expectedDim}`,
  };
}

// The full report the probe produces and the renderers consume. Analogous to
// CapabilityMatrix: it carries the run metadata (already redacted upstream) plus
// the single alias assessment.
export interface EmbeddingCompatReport {
  // ISO-8601 UTC timestamp of the run.
  runTimestampUtc: string;
  // Base URL HOST only (no scheme secrets, no key). Redaction happens upstream.
  baseUrlHost: string;
  // Redacted key marker (e.g. "sk-…(redacted)") — never any key-body character.
  maskedKey: string;
  // The embedding alias assessed, or null when none is configured.
  alias: string | null;
  // The expected dimension the store is built with.
  expectedDim: number;
  // The observed dimension, when the alias was probed.
  observedDim?: number;
  // The classified compatibility state.
  state: EmbeddingCompatState;
  // A concise, already-redacted note (never a secret/PII).
  note?: string;
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

// Legend lines shared by console and Markdown output.
export const EMBEDDING_COMPAT_LEGEND: ReadonlyArray<
  [EmbeddingCompatState, string]
> = [
  [
    "Compatible",
    "the alias produced a non-empty vector whose dimension equals the expected/knowledge-store dimension.",
  ],
  [
    "DimensionMismatch",
    "the alias produced a vector of a DIFFERENT dimension than expected (a definitive incompatibility).",
  ],
  [
    "Unverified",
    "no embedding alias configured, or the alias was not probed — compatibility is unknown.",
  ],
  [
    "Error",
    "network/auth/unexpected probe failure, distinct from a definitive mismatch (reason is redacted).",
  ],
];

// Render the observed dimension cell (or a placeholder when unprobed).
function observedCell(report: EmbeddingCompatReport): string {
  return report.observedDim !== undefined ? String(report.observedDim) : "-";
}

// Render the human-readable, single-line console summary of the assessment.
// Deterministic given the same report.
export function renderEmbeddingCompatConsole(
  report: EmbeddingCompatReport
): string {
  const alias = report.alias ?? "(none configured)";
  const observed = observedCell(report);
  const note = report.note ? ` — ${report.note}` : "";
  return (
    `Embedding compatibility: ${alias} => ${report.state} ` +
    `(expected ${report.expectedDim}, observed ${observed})${note}`
  );
}

// Escape a value for safe use inside a Markdown table cell.
function mdCell(value: string): string {
  return value.replace(/\|/g, "\\|");
}

// Render the full dated Markdown evidence file body.
//
// Includes: title, UTC run timestamp, base URL host (no key), masked key, a
// single-row assessment table (alias | state | expected dim | observed dim |
// note) and the legend. It never emits raw model output or secrets — those
// responsibilities live in the redaction done upstream in the probe.
export function renderEmbeddingCompatMarkdown(
  report: EmbeddingCompatReport
): string {
  const lines: string[] = [];

  lines.push("# LiteLLM Embedding & Dimension Compatibility");
  lines.push("");
  lines.push(`- **Run (UTC):** ${report.runTimestampUtc}`);
  lines.push(`- **Base URL host:** ${report.baseUrlHost}`);
  lines.push(`- **API key:** ${report.maskedKey} (masked)`);
  lines.push("");
  lines.push(
    "Verifies the embedding alias and vector dimensions THROUGH the proxy " +
      "(Phase 7 exit criterion 4): the alias must produce a non-empty vector " +
      "whose dimension equals the dimension the knowledge store is built with " +
      "(`config.embedDim`)."
  );
  lines.push("");

  const aliasCell = report.alias ?? "(none configured)";
  const noteCell = report.note ?? "-";
  lines.push("| Alias | State | Expected dim | Observed dim | Note |");
  lines.push("| --- | --- | --- | --- | --- |");
  lines.push(
    `| ${mdCell(aliasCell)} | ${report.state} | ${report.expectedDim} | ` +
      `${observedCell(report)} | ${mdCell(noteCell)} |`
  );
  lines.push("");

  lines.push("## Legend");
  lines.push("");
  for (const [state, meaning] of EMBEDDING_COMPAT_LEGEND) {
    lines.push(`- **${state}** — ${meaning}`);
  }
  lines.push("");

  return lines.join("\n");
}
