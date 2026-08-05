import { describe, it, expect } from "vitest";
import {
  CONTRACT_CAPABILITIES,
  CONTRACT_CAPABILITY_LABELS,
  CONTRACT_LEGEND,
  PERMITTED_TOOL_NAMES,
  classifyToolCallOutcome,
  classifyStructuredOutcome,
  renderContractConsoleTable,
  renderContractMarkdown,
  type AliasContractResults,
  type ContractMatrix,
  type ToolCallOutcome,
  type StructuredOutputOutcome,
} from "../src/core/contracts.js";
import type { CapabilityState } from "../src/core/capabilities.js";

// Phase 6 (increment 6d) — OFFLINE rendering/classification test for the
// tool-call + structured-output CONTRACT probe.
//
// Validates the PURE logic in src/core/contracts.ts with synthetic outcomes.
// Makes NO live call, imports NO probe I/O module, needs no credentials. The
// live probe (src/probe-contracts.ts) is never imported here, so `npm test`
// cannot trigger a network call.

function aliasResults(
  overrides: Partial<Record<(typeof CONTRACT_CAPABILITIES)[number], CapabilityState>>
): AliasContractResults {
  const base = {} as AliasContractResults;
  for (const cap of CONTRACT_CAPABILITIES) {
    base[cap] = { state: overrides[cap] ?? "Supported" };
  }
  return base;
}

function syntheticMatrix(): ContractMatrix {
  return {
    runTimestampUtc: "2026-08-05T12:34:56.000Z",
    baseUrlHost: "litellm.example.internal",
    maskedKey: "sk-…(redacted)",
    aliases: {
      "claude-opus-4-8": aliasResults({}),
      "claude-sonnet-4-6": aliasResults({ structuredOutput: "Degraded" }),
    },
  };
}

describe("Phase 6 — contract matrix rendering", () => {
  it("console table has a header, a separator, and one row per alias", () => {
    const lines = renderContractConsoleTable(syntheticMatrix()).split("\n");
    expect(lines).toHaveLength(4); // header + separator + 2 alias rows
    expect(lines[0]).toContain("Alias");
    for (const cap of CONTRACT_CAPABILITIES) {
      expect(lines[0]).toContain(CONTRACT_CAPABILITY_LABELS[cap]);
    }
    expect(lines[1]).toMatch(/^-+$/);
    expect(lines.find((l) => l.startsWith("claude-opus-4-8"))).toBeTruthy();
    expect(lines.find((l) => l.startsWith("claude-sonnet-4-6"))).toBeTruthy();
  });

  it("console table renders the correct state in each cell", () => {
    const lines = renderContractConsoleTable(syntheticMatrix()).split("\n");
    const sonnetRow = lines.find((l) => l.startsWith("claude-sonnet-4-6"))!;
    expect(sonnetRow).toContain("Degraded");
    const opusRow = lines.find((l) => l.startsWith("claude-opus-4-8"))!;
    expect(opusRow).toContain("Supported");
  });

  it("markdown includes title, UTC timestamp, host, masked key (no secret)", () => {
    const md = renderContractMarkdown(syntheticMatrix());
    expect(md).toContain("# LiteLLM Tool-Call & Structured-Output Contract Matrix");
    expect(md).toContain("2026-08-05T12:34:56.000Z");
    expect(md).toContain("litellm.example.internal");
    expect(md).toContain("sk-…(redacted)");
    expect(md).not.toMatch(/sk-[A-Za-z0-9]{20,}/);
  });

  it("markdown renders a table with a header and per-alias rows", () => {
    const md = renderContractMarkdown(syntheticMatrix());
    expect(md).toContain("| Alias |");
    for (const alias of ["claude-opus-4-8", "claude-sonnet-4-6"]) {
      expect(md).toContain(`| ${alias} |`);
    }
  });

  it("markdown lists per-cell notes only for non-Supported results", () => {
    const matrix = syntheticMatrix();
    matrix.aliases["claude-sonnet-4-6"]!.structuredOutput = {
      state: "Degraded",
      note: "output returned but failed schema validation",
    };
    const md = renderContractMarkdown(matrix);
    expect(md).toContain("## Notes");
    expect(md).toContain("claude-sonnet-4-6 / Structured output");
    const notesSection = md.slice(md.indexOf("## Notes"));
    expect(notesSection).not.toContain("claude-opus-4-8 / Tool call");
  });

  it("markdown includes a full legend defining every state", () => {
    const md = renderContractMarkdown(syntheticMatrix());
    expect(md).toContain("## Legend");
    for (const [state] of CONTRACT_LEGEND) {
      expect(md).toContain(`**${state}**`);
    }
  });

  it("escapes pipe characters in notes so the Markdown table stays valid", () => {
    const matrix = syntheticMatrix();
    matrix.aliases["claude-opus-4-8"]!.toolCall = {
      state: "Error",
      note: "a | b piped reason",
    };
    const md = renderContractMarkdown(matrix);
    expect(md).toContain("a \\| b piped reason");
  });
});

describe("Phase 6 — classifyToolCallOutcome (pure)", () => {
  function base(overrides: Partial<ToolCallOutcome> = {}): ToolCallOutcome {
    return {
      errored: false,
      errorLooksUnsupported: false,
      hasToolCall: true,
      toolName: "days_until",
      argsParseable: true,
      permitted: true,
      ...overrides,
    };
  }

  it("a well-formed permitted tool call is Supported", () => {
    expect(classifyToolCallOutcome(base())).toEqual({ state: "Supported" });
  });

  it("no tool call is Unsupported", () => {
    const r = classifyToolCallOutcome(
      base({ hasToolCall: false, toolName: undefined })
    );
    expect(r.state).toBe("Unsupported");
    expect(r.note).toMatch(/no tool call/i);
  });

  it("a tool call for a NON-permitted tool is Degraded (not Supported)", () => {
    const r = classifyToolCallOutcome(
      base({ toolName: "delete_everything", permitted: false })
    );
    expect(r.state).toBe("Degraded");
    expect(r.note).toMatch(/non-permitted/i);
  });

  it("a tool call with unparseable args is Degraded", () => {
    const r = classifyToolCallOutcome(base({ argsParseable: false }));
    expect(r.state).toBe("Degraded");
    expect(r.note).toMatch(/not parseable/i);
  });

  it("a thrown request that looks like a feature-no is Unsupported", () => {
    const r = classifyToolCallOutcome(
      base({ errored: true, errorLooksUnsupported: true, hasToolCall: false })
    );
    expect(r.state).toBe("Unsupported");
  });

  it("a thrown request that is incidental is Error", () => {
    const r = classifyToolCallOutcome(
      base({ errored: true, errorLooksUnsupported: false, hasToolCall: false })
    );
    expect(r.state).toBe("Error");
  });

  it("the permitted set matches the exported registry names", () => {
    expect([...PERMITTED_TOOL_NAMES].sort()).toEqual([
      "days_until",
      "split_budget",
    ]);
  });
});

describe("Phase 6 — classifyStructuredOutcome (pure)", () => {
  function base(
    overrides: Partial<StructuredOutputOutcome> = {}
  ): StructuredOutputOutcome {
    return {
      errored: false,
      errorLooksUnsupported: false,
      schemaValid: true,
      ...overrides,
    };
  }

  it("a schema-valid plan is Supported", () => {
    expect(classifyStructuredOutcome(base())).toEqual({ state: "Supported" });
  });

  it("a returned-but-invalid plan is Degraded", () => {
    const r = classifyStructuredOutcome(base({ schemaValid: false }));
    expect(r.state).toBe("Degraded");
    expect(r.note).toMatch(/schema validation/i);
  });

  it("a thrown feature-no is Unsupported; an incidental throw is Error", () => {
    expect(
      classifyStructuredOutcome(
        base({ errored: true, errorLooksUnsupported: true, schemaValid: false })
      ).state
    ).toBe("Unsupported");
    expect(
      classifyStructuredOutcome(
        base({ errored: true, errorLooksUnsupported: false, schemaValid: false })
      ).state
    ).toBe("Error");
  });
});
