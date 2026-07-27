import { describe, it, expect } from "vitest";
import {
  ABORT_SLOW_THRESHOLD_MS,
  CHAT_CAPABILITIES,
  CHAT_CAPABILITY_LABELS,
  LEGEND,
  classifyAbortOutcome,
  renderConsoleTable,
  renderConsoleEmbeddings,
  renderMarkdown,
  type AbortOutcome,
  type AliasCapabilityResults,
  type CapabilityMatrix,
  type CapabilityState,
} from "../src/core/capabilities.js";

// Phase 4 (increment 4c) — OFFLINE rendering/aggregation test.
//
// Validates the PURE result-formatting logic in src/core/capabilities.ts using
// a synthetic results object. Makes NO live call, imports NO probe I/O module,
// and needs no credentials. The live probe (src/probe-capabilities.ts) is never
// imported here, so `npm test` cannot trigger a network call.

// Build a full AliasCapabilityResults from a per-capability state map, defaulting
// unspecified capabilities to Supported.
function aliasResults(
  overrides: Partial<Record<(typeof CHAT_CAPABILITIES)[number], CapabilityState>>
): AliasCapabilityResults {
  const base = {} as AliasCapabilityResults;
  for (const cap of CHAT_CAPABILITIES) {
    base[cap] = { state: overrides[cap] ?? "Supported" };
  }
  return base;
}

// A synthetic matrix exercising every state across aliases and capabilities.
function syntheticMatrix(): CapabilityMatrix {
  return {
    runTimestampUtc: "2026-07-27T12:34:56.000Z",
    baseUrlHost: "litellm.example.internal",
    maskedKey: "sk-a...",
    chat: {
      "claude-opus-4-8": aliasResults({}),
      "claude-sonnet-4-6": aliasResults({
        toolCalling: "Unsupported",
        structuredOutput: "Degraded",
      }),
      // A clearly SYNTHETIC third alias (not a real supported alias) purely to
      // prove the renderer/aggregator handles more than two aliases and every
      // state. The real supported set is claude-opus-4-8 + claude-sonnet-4-6.
      "synthetic-alias-x": aliasResults({
        streaming: "Degraded",
        abort: "Error",
      }),
    },
    embeddings: {
      alias: "text-embedding-3-large",
      result: { state: "Supported" },
      dimensions: 3072,
    },
  };
}

describe("Phase 4 — capability matrix rendering", () => {
  it("console table has a header row, a separator, and one row per alias", () => {
    const matrix = syntheticMatrix();
    const table = renderConsoleTable(matrix);
    const lines = table.split("\n");

    // header + separator + 3 alias rows
    expect(lines).toHaveLength(5);
    expect(lines[0]).toContain("Alias");
    for (const cap of CHAT_CAPABILITIES) {
      expect(lines[0]).toContain(CHAT_CAPABILITY_LABELS[cap]);
    }
    expect(lines[1]).toMatch(/^-+$/);

    // Each alias appears with its own row.
    for (const alias of Object.keys(matrix.chat)) {
      const row = lines.find((l) => l.startsWith(alias));
      expect(row, `row for ${alias}`).toBeTruthy();
    }
  });

  it("console table renders the correct state in each cell", () => {
    const matrix = syntheticMatrix();
    const lines = renderConsoleTable(matrix).split("\n");

    const sonnetRow = lines.find((l) => l.startsWith("claude-sonnet-4-6"))!;
    expect(sonnetRow).toContain("Unsupported");
    expect(sonnetRow).toContain("Degraded");

    const syntheticRow = lines.find((l) => l.startsWith("synthetic-alias-x"))!;
    expect(syntheticRow).toContain("Error");
    expect(syntheticRow).toContain("Degraded");
  });

  it("console embeddings line shows alias, state, and dimensions", () => {
    const line = renderConsoleEmbeddings(syntheticMatrix());
    expect(line).toContain("text-embedding-3-large");
    expect(line).toContain("Supported");
    expect(line).toContain("dims=3072");
  });

  it("console embeddings line handles the N/A (no alias) case", () => {
    const matrix = syntheticMatrix();
    matrix.embeddings = {
      alias: null,
      result: { state: "N/A", note: "no embedding alias configured" },
    };
    const line = renderConsoleEmbeddings(matrix);
    expect(line).toContain("(none configured)");
    expect(line).toContain("N/A");
    expect(line).toContain("no embedding alias configured");
  });

  it("markdown includes title, UTC timestamp, host, and masked key (no secret)", () => {
    const md = renderMarkdown(syntheticMatrix());
    expect(md).toContain("# LiteLLM Capability Matrix");
    expect(md).toContain("2026-07-27T12:34:56.000Z");
    expect(md).toContain("litellm.example.internal");
    expect(md).toContain("sk-a...");
    // The masked key never expands to a full-looking secret.
    expect(md).not.toMatch(/sk-[A-Za-z0-9]{20,}/);
  });

  it("markdown renders a chat matrix table with a header and per-alias rows", () => {
    const md = renderMarkdown(syntheticMatrix());
    expect(md).toContain("## Chat capability matrix");
    // Header cells present.
    expect(md).toContain("| Alias |");
    for (const alias of ["claude-opus-4-8", "claude-sonnet-4-6", "synthetic-alias-x"]) {
      expect(md).toContain(`| ${alias} |`);
    }
  });

  it("markdown lists per-cell notes only for non-Supported results", () => {
    const md = renderMarkdown(syntheticMatrix());
    expect(md).toContain("### Chat notes");
    expect(md).toContain("claude-sonnet-4-6 / Tool calling");
    expect(md).toContain("synthetic-alias-x / Abort");
    // A fully-Supported alias should not appear in the notes section.
    const notesSection = md.slice(md.indexOf("### Chat notes"));
    expect(notesSection).not.toContain("claude-opus-4-8 / Invoke");
  });

  it("markdown has a separate embeddings section", () => {
    const md = renderMarkdown(syntheticMatrix());
    expect(md).toContain("## Embeddings assessment");
    expect(md).toContain("separate provider contract");
    expect(md).toContain("text-embedding-3-large");
    expect(md).toContain("3072");
  });

  it("markdown includes a full legend defining every state", () => {
    const md = renderMarkdown(syntheticMatrix());
    expect(md).toContain("## Legend");
    for (const [state] of LEGEND) {
      expect(md).toContain(`**${state}**`);
    }
  });

  it("markdown embeddings N/A case records the no-alias note", () => {
    const matrix = syntheticMatrix();
    matrix.embeddings = {
      alias: null,
      result: { state: "N/A", note: "no embedding alias configured" },
    };
    const md = renderMarkdown(matrix);
    expect(md).toContain("(none configured)");
    expect(md).toContain("| N/A |");
    expect(md).toContain("no embedding alias configured");
  });

  it("escapes pipe characters in notes so the Markdown table stays valid", () => {
    const matrix = syntheticMatrix();
    const overridden = { ...matrix.chat["claude-opus-4-8"]! };
    overridden.invoke = { state: "Error", note: "a | b piped reason" };
    matrix.chat = { "claude-opus-4-8": overridden };
    const md = renderMarkdown(matrix);
    expect(md).toContain("a \\| b piped reason");
  });
});

// Deterministic, offline coverage for the corrected abort classification
// (finding B1). classifyAbortOutcome() is a pure function, so we can pin every
// branch without any network/timers.
describe("Phase 4 — classifyAbortOutcome (finding B1)", () => {
  // A baseline "genuine prompt cancellation" outcome; individual tests override.
  function outcome(overrides: Partial<AbortOutcome> = {}): AbortOutcome {
    return {
      completed: false,
      timedOut: false,
      signalAborted: true,
      isAbortError: true,
      elapsedMs: 200,
      errorLooksUnsupported: false,
      ...overrides,
    };
  }

  it("a genuine, prompt cancellation is Supported", () => {
    expect(classifyAbortOutcome(outcome())).toEqual({ state: "Supported" });
  });

  it("a genuine but slow cancellation is Degraded", () => {
    const result = classifyAbortOutcome(
      outcome({ elapsedMs: ABORT_SLOW_THRESHOLD_MS + 1 })
    );
    expect(result.state).toBe("Degraded");
    expect(result.note).toMatch(/slowly/);
  });

  it("a fast NON-abort error is NOT Supported — Error by default (finding B1)", () => {
    // e.g. an invalid model name: signal never aborted, error is not an
    // abort. This is the exact false-positive the finding calls out.
    const result = classifyAbortOutcome(
      outcome({ signalAborted: false, isAbortError: false })
    );
    expect(result.state).toBe("Error");
  });

  it("a NON-abort error that looks unsupported is Unsupported, not Supported", () => {
    const result = classifyAbortOutcome(
      outcome({
        signalAborted: false,
        isAbortError: false,
        errorLooksUnsupported: true,
      })
    );
    expect(result.state).toBe("Unsupported");
  });

  it("signal aborted but the error is not an abort error is NOT a success", () => {
    // Defensive: an unrelated error surfacing after we happened to abort must
    // not be counted as a successful cancellation.
    const result = classifyAbortOutcome(
      outcome({ signalAborted: true, isAbortError: false })
    );
    expect(result.state).toBe("Error");
  });

  it("an abort error without the signal set is NOT a success", () => {
    const result = classifyAbortOutcome(
      outcome({ signalAborted: false, isAbortError: true })
    );
    expect(result.state).toBe("Error");
  });

  it("a completed stream (abort ignored) is Unsupported", () => {
    const result = classifyAbortOutcome(
      outcome({ completed: true, signalAborted: true, isAbortError: false })
    );
    expect(result.state).toBe("Unsupported");
    expect(result.note).toMatch(/completed despite abort/);
  });

  it("a timed-out stream (abort ignored past budget) is Unsupported", () => {
    const result = classifyAbortOutcome(
      outcome({ timedOut: true, signalAborted: false, isAbortError: false })
    );
    expect(result.state).toBe("Unsupported");
    expect(result.note).toMatch(/ran past its timeout/);
  });
});
