import type { CapabilityState, ProbeResult } from "./capabilities.js";

// Phase 6 (increment 6d) — per-alias TOOL-CALL + STRUCTURED-OUTPUT contract
// matrix: PURE logic.
//
// This module contains only pure, deterministic, offline logic: the ordered
// list of contract capabilities, the classification of an observed probe
// outcome into a state, and the functions that render a results object into a
// console table + a dated Markdown evidence body. It performs NO network calls
// and reads NO credentials, so the offline Vitest suite imports it directly.
// All live probing I/O lives in src/probe-contracts.ts, which imports these.
//
// It deliberately REUSES the Phase 4 `CapabilityState` / `ProbeResult`
// vocabulary from src/core/capabilities.ts so the two evidence artifacts speak
// the same language (Supported / Unsupported / Degraded / Error / N/A).

// The two contract checks probed, in stable display order. These are the Phase 6
// exit-criterion (4) requirements: "every enabled model alias passes the
// required tool-call and structured-output contract tests".
export const CONTRACT_CAPABILITIES = ["toolCall", "structuredOutput"] as const;

export type ContractCapability = (typeof CONTRACT_CAPABILITIES)[number];

export const CONTRACT_CAPABILITY_LABELS: Record<ContractCapability, string> = {
  toolCall: "Tool call",
  structuredOutput: "Structured output",
};

// The SAFE tools an alias is permitted to call. A well-formed tool-call contract
// result MUST name one of these; a call to anything else is NOT a pass (see
// classifyToolCallOutcome). Mirrors `weddingTools` in src/core/tools.ts.
export const PERMITTED_TOOL_NAMES = ["days_until", "split_budget"] as const;

// All contract results for one alias.
export type AliasContractResults = Record<ContractCapability, ProbeResult>;

// The full result set the probe produces and the renderers consume.
export interface ContractMatrix {
  // ISO-8601 UTC timestamp of the run.
  runTimestampUtc: string;
  // Base URL HOST only (no scheme secrets, no key). Redaction happens upstream.
  baseUrlHost: string;
  // Redacted key marker (e.g. "sk-…(redacted)") — never any key-body character.
  maskedKey: string;
  // Per alias -> per contract capability result. Insertion order = display order.
  aliases: Record<string, AliasContractResults>;
}

// ---------------------------------------------------------------------------
// Outcome classification (pure)
// ---------------------------------------------------------------------------

// The OBSERVABLE outcome of the tool-call contract probe, decoupled from the
// live I/O so the classification is deterministic and unit-testable offline.
export interface ToolCallOutcome {
  // Did the request throw (network/auth/provider/timeout)?
  errored: boolean;
  // If it errored, did the error look like a definitive feature-no?
  errorLooksUnsupported: boolean;
  // Did the model return at least one tool call?
  hasToolCall: boolean;
  // Name of the first tool call (when hasToolCall).
  toolName?: string;
  // Were the first tool call's args a parseable object?
  argsParseable: boolean;
  // Was the first tool call's name one of PERMITTED_TOOL_NAMES?
  permitted: boolean;
}

// Classify the tool-call contract outcome. SINGLE source of truth, exercised by
// offline unit tests.
//
// Rules:
//   - Threw            -> Unsupported if it looked like a feature-no, else Error.
//   - No tool call      -> Unsupported (the model did not use the tool).
//   - Non-permitted name-> Degraded (a tool call, but not one of the safe tools).
//   - Unparseable args  -> Degraded (tool call present but malformed).
//   - Otherwise         -> Supported.
export function classifyToolCallOutcome(outcome: ToolCallOutcome): ProbeResult {
  if (outcome.errored) {
    return outcome.errorLooksUnsupported
      ? { state: "Unsupported" }
      : { state: "Error" };
  }
  if (!outcome.hasToolCall) {
    return { state: "Unsupported", note: "model returned no tool call" };
  }
  if (!outcome.permitted) {
    return {
      state: "Degraded",
      note: `tool call for a non-permitted tool (${outcome.toolName ?? "?"})`,
    };
  }
  if (!outcome.argsParseable) {
    return {
      state: "Degraded",
      note: `tool call present but args were not parseable (${outcome.toolName ?? "?"})`,
    };
  }
  return { state: "Supported" };
}

// The OBSERVABLE outcome of the structured-output contract probe.
export interface StructuredOutputOutcome {
  // Did the request throw (network/auth/provider/timeout/refusal)?
  errored: boolean;
  // If it errored, did the error look like a definitive feature-no?
  errorLooksUnsupported: boolean;
  // Did a schema-VALID BudgetPlan come back (only meaningful when !errored)?
  schemaValid: boolean;
}

// Classify the structured-output contract outcome.
//
// Rules:
//   - Threw            -> Unsupported if it looked like a feature-no, else Error.
//   - Not schema-valid  -> Degraded (output returned but failed validation).
//   - Otherwise         -> Supported.
export function classifyStructuredOutcome(
  outcome: StructuredOutputOutcome
): ProbeResult {
  if (outcome.errored) {
    return outcome.errorLooksUnsupported
      ? { state: "Unsupported" }
      : { state: "Error" };
  }
  if (!outcome.schemaValid) {
    return {
      state: "Degraded",
      note: "output returned but failed schema validation",
    };
  }
  return { state: "Supported" };
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

// Legend lines shared by console and Markdown output. Uses the Phase 4 state
// vocabulary; the meanings are tailored to the two contract checks.
export const CONTRACT_LEGEND: ReadonlyArray<[CapabilityState, string]> = [
  ["Supported", "the contract held (a well-formed tool call for a permitted tool / a schema-valid BudgetPlan)."],
  [
    "Unsupported",
    "the model produced no tool call / no structured output, or the provider definitively rejected the feature.",
  ],
  [
    "Degraded",
    "partially held (a tool call for a non-permitted tool or with unparseable args; or structured output that failed schema validation).",
  ],
  [
    "Error",
    "network/auth/unexpected failure or timeout, distinct from Unsupported (reason is redacted).",
  ],
  ["N/A", "not applicable."],
];

// Pad a cell to a fixed width (left-aligned) for the fixed-width console table.
function pad(value: string, width: number): string {
  return value.length >= width ? value : value + " ".repeat(width - value.length);
}

// Render the contract matrix as a fixed-width, human-readable console table.
// Deterministic given the same matrix (aliases and capabilities are ordered).
export function renderContractConsoleTable(matrix: ContractMatrix): string {
  const aliases = Object.keys(matrix.aliases);
  const capLabels = CONTRACT_CAPABILITIES.map((c) => CONTRACT_CAPABILITY_LABELS[c]);

  const aliasColWidth = Math.max("Alias".length, ...aliases.map((a) => a.length));
  const capColWidths = CONTRACT_CAPABILITIES.map((cap, i) =>
    Math.max(
      capLabels[i]!.length,
      ...aliases.map((a) => matrix.aliases[a]![cap].state.length)
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
      CONTRACT_CAPABILITIES.map((cap, i) =>
        pad(matrix.aliases[alias]![cap].state, capColWidths[i]!)
      ).join(" | ");
    lines.push(row);
  }

  return lines.join("\n");
}

// Escape a value for safe use inside a Markdown table cell.
function mdCell(value: string): string {
  return value.replace(/\|/g, "\\|");
}

// Render the full dated Markdown evidence file body.
//
// Includes: title, UTC run timestamp, base URL host (no key), masked key, the
// per-alias contract table, per-cell notes for any non-Supported result, and
// the legend. It never emits raw model output or secrets — redaction is done
// upstream in the probe.
export function renderContractMarkdown(matrix: ContractMatrix): string {
  const aliases = Object.keys(matrix.aliases);
  const lines: string[] = [];

  lines.push("# LiteLLM Tool-Call & Structured-Output Contract Matrix");
  lines.push("");
  lines.push(`- **Run (UTC):** ${matrix.runTimestampUtc}`);
  lines.push(`- **Base URL host:** ${matrix.baseUrlHost}`);
  lines.push(`- **API key:** ${matrix.maskedKey} (masked)`);
  lines.push("");
  lines.push(
    "Per enabled model alias, the contract checks required by Phase 6 exit " +
      "criterion (4): a well-formed **tool call** for a permitted safe tool " +
      "(`days_until` / `split_budget`) and a schema-valid **structured output** " +
      "(`BudgetPlan`). Opus is probed with temperature OMITTED (Phase 4 " +
      "carry-forward)."
  );
  lines.push("");

  const headerCells = [
    "Alias",
    ...CONTRACT_CAPABILITIES.map((c) => CONTRACT_CAPABILITY_LABELS[c]),
  ];
  lines.push(`| ${headerCells.map(mdCell).join(" | ")} |`);
  lines.push(`| ${headerCells.map(() => "---").join(" | ")} |`);
  for (const alias of aliases) {
    const cells = [
      alias,
      ...CONTRACT_CAPABILITIES.map((cap) => matrix.aliases[alias]![cap].state),
    ];
    lines.push(`| ${cells.map(mdCell).join(" | ")} |`);
  }
  lines.push("");

  // Per-cell notes for anything not plainly Supported.
  const notes: string[] = [];
  for (const alias of aliases) {
    for (const cap of CONTRACT_CAPABILITIES) {
      const result = matrix.aliases[alias]![cap];
      if (result.state !== "Supported") {
        const label = CONTRACT_CAPABILITY_LABELS[cap];
        const note = result.note ? `: ${result.note}` : "";
        notes.push(
          `- **${mdCell(alias)} / ${label}** — ${result.state}${mdCell(note)}`
        );
      }
    }
  }
  if (notes.length > 0) {
    lines.push("## Notes");
    lines.push("");
    lines.push(...notes);
    lines.push("");
  }

  lines.push("## Legend");
  lines.push("");
  for (const [state, meaning] of CONTRACT_LEGEND) {
    lines.push(`- **${state}** — ${meaning}`);
  }
  lines.push("");

  return lines.join("\n");
}
