// Phase 4 (increment 4c) — LiteLLM capability matrix: PURE logic.
//
// This module contains only pure, deterministic, offline logic: the result
// vocabulary, the ordered lists of aliases/capabilities, and the functions that
// aggregate a results object into a human-readable console table and a dated
// Markdown evidence file body.
//
// It performs NO network calls and reads NO credentials, so it is safe to import
// from the offline Vitest suite. All live probing I/O lives in
// src/probe-capabilities.ts, which imports these functions.

// One recorded state per (alias, capability) or per embedding assessment.
//
// - Supported   — capability worked as expected.
// - Unsupported — proxy/model rejected it or does not support it (a definitive
//                 "no", e.g. tool calls or structured output unavailable).
// - Degraded    — capability partially worked (see per-capability definitions).
// - Error       — network/auth/unexpected failure, distinct from Unsupported.
// - N/A         — not applicable (e.g. embeddings with no embedding alias).
export type CapabilityState =
  | "Supported"
  | "Unsupported"
  | "Degraded"
  | "Error"
  | "N/A";

// The chat capabilities probed, in stable display order.
export const CHAT_CAPABILITIES = [
  "invoke",
  "streaming",
  "abort",
  "usageMetadata",
  "toolCalling",
  "structuredOutput",
] as const;

export type ChatCapability = (typeof CHAT_CAPABILITIES)[number];

// Short human-facing column headers for the chat capabilities.
export const CHAT_CAPABILITY_LABELS: Record<ChatCapability, string> = {
  invoke: "Invoke",
  streaming: "Streaming",
  abort: "Abort",
  usageMetadata: "Usage meta",
  toolCalling: "Tool calling",
  structuredOutput: "Structured out",
};

// A single probe outcome: the classified state plus an optional concise,
// already-redacted note (e.g. a redacted error reason or a Degraded rationale).
export interface ProbeResult {
  state: CapabilityState;
  note?: string;
}

// All chat capability results for one alias.
export type AliasCapabilityResults = Record<ChatCapability, ProbeResult>;

// The full result set the probe produces and the renderers consume.
export interface CapabilityMatrix {
  // ISO-8601 UTC timestamp of the run.
  runTimestampUtc: string;
  // Base URL HOST only (no scheme secrets, no key). Redaction happens upstream.
  baseUrlHost: string;
  // Redacted key marker (e.g. "sk-…(redacted)") — never any key-body character.
  maskedKey: string;
  // Per chat alias -> per capability result. Insertion order = display order.
  chat: Record<string, AliasCapabilityResults>;
  // Separate embeddings assessment (chat aliases are never listed here).
  embeddings: EmbeddingsAssessment;
}

export interface EmbeddingsAssessment {
  // The embedding alias probed, or null when none is configured.
  alias: string | null;
  result: ProbeResult;
  // Vector dimension when a successful embedding was produced.
  dimensions?: number;
}

// ---------------------------------------------------------------------------
// Abort-outcome classification (pure)
// ---------------------------------------------------------------------------

// The observable outcome of the abort probe, decoupled from the live I/O so the
// classification is deterministic and unit-testable offline.
export interface AbortOutcome {
  // Did the stream complete WITHOUT throwing (i.e. abort was ignored)?
  completed: boolean;
  // Did our per-probe timeout fire (the stream ran past its budget)?
  timedOut: boolean;
  // Was controller.signal.aborted true at the time the error surfaced?
  signalAborted: boolean;
  // Did the surfaced error actually look like a cancellation
  // (err.name === "AbortError" or /abort/i on the message)?
  isAbortError: boolean;
  // Milliseconds from stream start to the outcome.
  elapsedMs: number;
  // Whether a non-abort error's message looked like a definitive feature-no.
  errorLooksUnsupported: boolean;
}

// How long (ms) a genuine cancellation may take before we call it Degraded.
export const ABORT_SLOW_THRESHOLD_MS = 5_000;

// Classify the abort probe outcome. This is the SINGLE source of truth for the
// abort state and is exercised by offline unit tests.
//
// Rules (see finding B1):
//   - Stream completed without throwing        -> Unsupported (abort ignored).
//   - Our timeout fired                         -> Unsupported (abort ignored).
//   - A GENUINE cancellation (signal aborted    -> Supported, or Degraded if the
//     AND the error is an abort error)             cancellation was slow.
//   - Any OTHER error (invalid model, auth, …)  -> Unsupported if it looks like a
//     is NOT a successful abort                     feature-no, else Error. It is
//                                                   NEVER reported as Supported.
export function classifyAbortOutcome(outcome: AbortOutcome): ProbeResult {
  if (outcome.completed) {
    return { state: "Unsupported", note: "stream completed despite abort()" };
  }
  if (outcome.timedOut) {
    return {
      state: "Unsupported",
      note: "abort ignored; stream ran past its timeout budget",
    };
  }
  // From here the probe caught an error. Only a real cancellation counts as a
  // successful abort.
  const genuineAbort = outcome.signalAborted && outcome.isAbortError;
  if (genuineAbort) {
    if (outcome.elapsedMs > ABORT_SLOW_THRESHOLD_MS) {
      return {
        state: "Degraded",
        note: `cancelled but slowly (~${outcome.elapsedMs}ms)`,
      };
    }
    return { state: "Supported" };
  }
  // Not an abort: an unrelated request failure. Never Supported.
  if (outcome.errorLooksUnsupported) {
    return { state: "Unsupported" };
  }
  return { state: "Error" };
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

// Legend lines shared by console and Markdown output.
export const LEGEND: ReadonlyArray<[CapabilityState, string]> = [
  ["Supported", "capability worked as expected."],
  [
    "Unsupported",
    "proxy/model definitively rejected or does not support the capability.",
  ],
  [
    "Degraded",
    "capability partially worked (e.g. response returned but usage metadata missing, or a stream produced only a single chunk).",
  ],
  [
    "Error",
    "network/auth/unexpected failure, distinct from Unsupported (reason is redacted).",
  ],
  ["N/A", "not applicable (e.g. embeddings when no embedding alias is configured)."],
];

// Pad a cell to a fixed width (left-aligned) for the fixed-width console table.
function pad(value: string, width: number): string {
  return value.length >= width ? value : value + " ".repeat(width - value.length);
}

// Render the chat matrix as a fixed-width, human-readable console table.
// Deterministic given the same matrix (aliases and capabilities are ordered).
export function renderConsoleTable(matrix: CapabilityMatrix): string {
  const aliases = Object.keys(matrix.chat);
  const capLabels = CHAT_CAPABILITIES.map((c) => CHAT_CAPABILITY_LABELS[c]);

  // Column 0 is the alias name; the rest are capabilities.
  const aliasColWidth = Math.max(
    "Alias".length,
    ...aliases.map((a) => a.length)
  );
  const capColWidths = CHAT_CAPABILITIES.map((cap, i) =>
    Math.max(
      capLabels[i]!.length,
      ...aliases.map((a) => matrix.chat[a]![cap].state.length)
    )
  );

  const lines: string[] = [];

  const header =
    pad("Alias", aliasColWidth) +
    " | " +
    capLabels.map((label, i) => pad(label, capColWidths[i]!)).join(" | ");
  lines.push(header);
  lines.push("-".repeat(header.length));

  for (const alias of aliases) {
    const row =
      pad(alias, aliasColWidth) +
      " | " +
      CHAT_CAPABILITIES.map((cap, i) =>
        pad(matrix.chat[alias]![cap].state, capColWidths[i]!)
      ).join(" | ");
    lines.push(row);
  }

  return lines.join("\n");
}

// Render the console-facing embeddings summary line(s).
export function renderConsoleEmbeddings(matrix: CapabilityMatrix): string {
  const { embeddings } = matrix;
  const alias = embeddings.alias ?? "(none configured)";
  const dims =
    embeddings.dimensions !== undefined ? ` dims=${embeddings.dimensions}` : "";
  const note = embeddings.result.note ? ` — ${embeddings.result.note}` : "";
  return `Embedding alias: ${alias} => ${embeddings.result.state}${dims}${note}`;
}

// Escape a value for safe use inside a Markdown table cell.
function mdCell(value: string): string {
  return value.replace(/\|/g, "\\|");
}

// Render the full dated Markdown evidence file body.
//
// Includes: title, UTC run timestamp, base URL host (no key), the chat matrix
// table, a separate embeddings section, the legend, and concise per-cell notes
// for any result that is not plainly Supported. It never emits raw model output
// or secrets — those responsibilities live in the redaction done upstream.
export function renderMarkdown(matrix: CapabilityMatrix): string {
  const aliases = Object.keys(matrix.chat);
  const lines: string[] = [];

  lines.push("# LiteLLM Capability Matrix");
  lines.push("");
  lines.push(`- **Run (UTC):** ${matrix.runTimestampUtc}`);
  lines.push(`- **Base URL host:** ${matrix.baseUrlHost}`);
  lines.push(`- **API key:** ${matrix.maskedKey} (masked)`);
  lines.push("");

  // --- Chat capability matrix ------------------------------------------------
  lines.push("## Chat capability matrix");
  lines.push("");

  const headerCells = ["Alias", ...CHAT_CAPABILITIES.map((c) => CHAT_CAPABILITY_LABELS[c])];
  lines.push(`| ${headerCells.map(mdCell).join(" | ")} |`);
  lines.push(`| ${headerCells.map(() => "---").join(" | ")} |`);

  for (const alias of aliases) {
    const cells = [
      alias,
      ...CHAT_CAPABILITIES.map((cap) => matrix.chat[alias]![cap].state),
    ];
    lines.push(`| ${cells.map(mdCell).join(" | ")} |`);
  }
  lines.push("");

  // Per-cell notes for anything that is not plainly Supported.
  const notes: string[] = [];
  for (const alias of aliases) {
    for (const cap of CHAT_CAPABILITIES) {
      const result = matrix.chat[alias]![cap];
      if (result.state !== "Supported") {
        const label = CHAT_CAPABILITY_LABELS[cap];
        const note = result.note ? `: ${result.note}` : "";
        notes.push(`- **${mdCell(alias)} / ${label}** — ${result.state}${mdCell(note)}`);
      }
    }
  }
  if (notes.length > 0) {
    lines.push("### Chat notes");
    lines.push("");
    lines.push(...notes);
    lines.push("");
  }

  // --- Embeddings (separate contract) ---------------------------------------
  lines.push("## Embeddings assessment");
  lines.push("");
  lines.push(
    "Embeddings are a separate provider contract from chat aliases; chat " +
      "aliases are not required or evaluated as embedding aliases."
  );
  lines.push("");

  const { embeddings } = matrix;
  const aliasCell = embeddings.alias ?? "(none configured)";
  const dimsCell =
    embeddings.dimensions !== undefined ? String(embeddings.dimensions) : "-";
  lines.push("| Embedding alias | State | Dimensions |");
  lines.push("| --- | --- | --- |");
  lines.push(
    `| ${mdCell(aliasCell)} | ${embeddings.result.state} | ${dimsCell} |`
  );
  lines.push("");
  if (embeddings.result.note) {
    lines.push(`- **Note:** ${mdCell(embeddings.result.note)}`);
    lines.push("");
  }

  // --- Legend ----------------------------------------------------------------
  lines.push("## Legend");
  lines.push("");
  for (const [state, meaning] of LEGEND) {
    lines.push(`- **${state}** — ${meaning}`);
  }
  lines.push("");

  // --- Per-capability Degraded definitions -----------------------------------
  lines.push("## Degraded definitions (per capability)");
  lines.push("");
  lines.push(
    "- **Invoke** — Degraded if a response returned but had empty text content."
  );
  lines.push(
    "- **Streaming** — Degraded if the stream produced only a single chunk (no incremental streaming)."
  );
  lines.push(
    "- **Abort** — Degraded if cancellation eventually took effect but not promptly."
  );
  lines.push(
    "- **Usage meta** — Degraded if a response returned but `usage_metadata` was absent or lacked token counts."
  );
  lines.push(
    "- **Tool calling** — Degraded if the model acknowledged the tool but returned a malformed/partial tool call."
  );
  lines.push(
    "- **Structured out** — Degraded if JSON-ish output returned but failed schema validation; Unsupported if the feature is unavailable."
  );
  lines.push("");

  return lines.join("\n");
}
