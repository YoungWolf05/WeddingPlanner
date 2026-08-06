import { describe, it, expect } from "vitest";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  EMBEDDING_COMPAT_LEGEND,
  classifyEmbeddingCompatibility,
  isEmbeddingDimensionCompatible,
  renderEmbeddingCompatConsole,
  renderEmbeddingCompatMarkdown,
  type EmbeddingCompatReport,
} from "../src/core/embedding-compat.js";
import {
  config,
  parseEmbedDim,
  DEFAULT_EMBED_DIM,
} from "../src/config.js";
import { DEFAULT_EMBEDDING_DIM } from "../src/core/knowledge-store.js";

// Phase 7 (increment 7d) — OFFLINE compatibility coverage for the embedding &
// dimension verification.
//
// Validates the PURE logic in src/core/embedding-compat.ts and the config.embedDim
// parsing with synthetic inputs. Makes NO live call, imports NO probe I/O module,
// needs no credentials. The live probe (src/probe-embedding.ts) is never imported
// here, so `npm test` cannot trigger a network call or write docs/embeddings.

const EXPECTED = 768;

describe("Phase 7 (7d) — isEmbeddingDimensionCompatible (predicate)", () => {
  it("equal positive dimensions are compatible", () => {
    expect(isEmbeddingDimensionCompatible(768, 768)).toBe(true);
    expect(isEmbeddingDimensionCompatible(1536, 1536)).toBe(true);
  });

  it("different dimensions are NOT compatible (both directions)", () => {
    expect(isEmbeddingDimensionCompatible(768, 1536)).toBe(false);
    expect(isEmbeddingDimensionCompatible(1536, 768)).toBe(false);
  });

  it("zero, negative, and non-integer dimensions are never compatible", () => {
    expect(isEmbeddingDimensionCompatible(768, 0)).toBe(false);
    expect(isEmbeddingDimensionCompatible(0, 0)).toBe(false);
    expect(isEmbeddingDimensionCompatible(768, -768)).toBe(false);
    expect(isEmbeddingDimensionCompatible(768, 768.5)).toBe(false);
    expect(isEmbeddingDimensionCompatible(Number.NaN, 768)).toBe(false);
  });
});

describe("Phase 7 (7d) — classifyEmbeddingCompatibility (pure)", () => {
  it("observed == expected is Compatible (no note)", () => {
    const r = classifyEmbeddingCompatibility({
      alias: "gemini-embedding-001",
      expectedDim: EXPECTED,
      observedDim: EXPECTED,
    });
    expect(r.state).toBe("Compatible");
    expect(r.expectedDim).toBe(EXPECTED);
    expect(r.observedDim).toBe(EXPECTED);
    expect(r.note).toBeUndefined();
  });

  it("observed > expected is DimensionMismatch", () => {
    const r = classifyEmbeddingCompatibility({
      alias: "some-embed",
      expectedDim: 768,
      observedDim: 1536,
    });
    expect(r.state).toBe("DimensionMismatch");
    expect(r.observedDim).toBe(1536);
    expect(r.note).toMatch(/1536 dimension/i);
    expect(r.note).toMatch(/expects 768/i);
  });

  it("observed < expected is DimensionMismatch", () => {
    const r = classifyEmbeddingCompatibility({
      alias: "some-embed",
      expectedDim: 1536,
      observedDim: 768,
    });
    expect(r.state).toBe("DimensionMismatch");
  });

  it("an empty (0-length) vector is DimensionMismatch, not Compatible", () => {
    const r = classifyEmbeddingCompatibility({
      alias: "some-embed",
      expectedDim: 768,
      observedDim: 0,
    });
    expect(r.state).toBe("DimensionMismatch");
  });

  it("no alias configured is Unverified", () => {
    for (const alias of [null, undefined, ""] as const) {
      const r = classifyEmbeddingCompatibility({ alias, expectedDim: EXPECTED });
      expect(r.state).toBe("Unverified");
      expect(r.note).toMatch(/no embedding alias/i);
    }
  });

  it("an alias with no observed dimension is Unverified (not probed)", () => {
    const r = classifyEmbeddingCompatibility({
      alias: "gemini-embedding-001",
      expectedDim: EXPECTED,
    });
    expect(r.state).toBe("Unverified");
    expect(r.note).toMatch(/not probed/i);
  });

  it("a probe error is Error and carries the (already-redacted) note", () => {
    const r = classifyEmbeddingCompatibility({
      alias: "gemini-embedding-001",
      expectedDim: EXPECTED,
      error: "connection reset by peer",
    });
    expect(r.state).toBe("Error");
    expect(r.note).toBe("connection reset by peer");
  });

  it("an error takes precedence even when an observed dimension is present", () => {
    // The transport failed AFTER a partial read; the definitive signal is Error,
    // never a mismatch verdict on a possibly-bogus dimension.
    const r = classifyEmbeddingCompatibility({
      alias: "gemini-embedding-001",
      expectedDim: EXPECTED,
      observedDim: 10,
      error: "timed out",
    });
    expect(r.state).toBe("Error");
    expect(r.note).toBe("timed out");
  });
});

// A synthetic report exercising the renderers deterministically.
function report(
  overrides: Partial<EmbeddingCompatReport> = {}
): EmbeddingCompatReport {
  return {
    runTimestampUtc: "2026-08-06T12:34:56.000Z",
    baseUrlHost: "litellm.example.internal",
    maskedKey: "sk-…(redacted)",
    alias: "gemini-embedding-001",
    expectedDim: 768,
    observedDim: 768,
    state: "Compatible",
    ...overrides,
  };
}

describe("Phase 7 (7d) — renderEmbeddingCompatConsole", () => {
  it("renders alias, state, expected + observed dims", () => {
    const line = renderEmbeddingCompatConsole(report());
    expect(line).toContain("gemini-embedding-001");
    expect(line).toContain("Compatible");
    expect(line).toContain("expected 768");
    expect(line).toContain("observed 768");
  });

  it("handles the no-alias / unprobed case", () => {
    const line = renderEmbeddingCompatConsole(
      report({ alias: null, observedDim: undefined, state: "Unverified", note: "no embedding alias" })
    );
    expect(line).toContain("(none configured)");
    expect(line).toContain("Unverified");
    expect(line).toContain("observed -");
    expect(line).toContain("no embedding alias");
  });
});

describe("Phase 7 (7d) — renderEmbeddingCompatMarkdown", () => {
  it("includes title, UTC timestamp, host, and masked key (no secret)", () => {
    const md = renderEmbeddingCompatMarkdown(report());
    expect(md).toContain("# LiteLLM Embedding & Dimension Compatibility");
    expect(md).toContain("2026-08-06T12:34:56.000Z");
    expect(md).toContain("litellm.example.internal");
    expect(md).toContain("sk-…(redacted)");
    // The masked key never expands to a full-looking secret.
    expect(md).not.toMatch(/sk-[A-Za-z0-9]{20,}/);
  });

  it("renders the single-row assessment table with all columns", () => {
    const md = renderEmbeddingCompatMarkdown(report());
    expect(md).toContain("| Alias | State | Expected dim | Observed dim | Note |");
    expect(md).toContain("| gemini-embedding-001 | Compatible | 768 | 768 | - |");
  });

  it("renders a DimensionMismatch row with its note", () => {
    const md = renderEmbeddingCompatMarkdown(
      report({
        state: "DimensionMismatch",
        observedDim: 1536,
        note: "alias produced 1536 dimension(s) but the knowledge store expects 768",
      })
    );
    expect(md).toContain("| gemini-embedding-001 | DimensionMismatch | 768 | 1536 |");
    expect(md).toContain("1536 dimension");
  });

  it("includes a full legend defining every state", () => {
    const md = renderEmbeddingCompatMarkdown(report());
    expect(md).toContain("## Legend");
    for (const [state] of EMBEDDING_COMPAT_LEGEND) {
      expect(md).toContain(`**${state}**`);
    }
  });

  it("escapes pipe characters in the note so the Markdown table stays valid", () => {
    const md = renderEmbeddingCompatMarkdown(
      report({ state: "Error", observedDim: undefined, note: "a | b piped reason" })
    );
    expect(md).toContain("a \\| b piped reason");
  });

  it("renders EXACTLY what it is given — redaction is the caller's job", () => {
    // Contract: the pure module never scrubs; the PROBE redacts before calling
    // render. If a caller (wrongly) passes an unredacted secret in the note, the
    // module renders it verbatim. We assert this contract so the redaction
    // boundary is unambiguous: verification that no secret leaks is a PROBE-level
    // responsibility (probe-embedding.ts routes every error through redactError).
    const md = renderEmbeddingCompatMarkdown(
      report({ state: "Error", observedDim: undefined, note: "sk-PLACEHOLDER-not-scrubbed-here" })
    );
    expect(md).toContain("sk-PLACEHOLDER-not-scrubbed-here");
  });
});

describe("Phase 7 (7d) — config.embedDim parsing (parseEmbedDim)", () => {
  it("defaults to 768 when unset / empty / whitespace-only", () => {
    expect(parseEmbedDim(undefined)).toBe(768);
    expect(parseEmbedDim("")).toBe(768);
    expect(parseEmbedDim("   ")).toBe(768);
    expect(DEFAULT_EMBED_DIM).toBe(768);
  });

  it("parses a positive integer", () => {
    expect(parseEmbedDim("768")).toBe(768);
    expect(parseEmbedDim("1536")).toBe(1536);
    expect(parseEmbedDim("  1024  ")).toBe(1024);
  });

  it("FAILS LOUD on a present-but-invalid value (non-integer, zero, negative, non-numeric)", () => {
    for (const bad of ["0", "-1", "-5", "768.5", "12.0", "abc", "12x", "NaN"]) {
      expect(() => parseEmbedDim(bad), `parseEmbedDim(${bad})`).toThrow(
        /must be a positive integer/i
      );
    }
  });

  it("FAILS LOUD on non-decimal integer forms Number() would otherwise accept", () => {
    // `Number("0x10")` === 16, `Number("1e3")` === 1000, `Number("0o17")` etc.
    // are all integers, but they are NOT the plain decimal integer the contract
    // promises, so the strict /^\d+$/ gate must reject them.
    for (const bad of ["0x10", "1e3", "0b101", "0o17"]) {
      expect(() => parseEmbedDim(bad), `parseEmbedDim(${bad})`).toThrow(
        /must be a positive integer/i
      );
    }
  });

  it("FAILS LOUD when digits are present but whitespace/junk is embedded internally", () => {
    // Only surrounding whitespace is trimmed; internal separators/junk remain and
    // must be rejected (a valid dimension is a contiguous run of decimal digits).
    for (const bad of ["1 3", "7 7", "1_536", "+768"]) {
      expect(() => parseEmbedDim(bad), `parseEmbedDim(${bad})`).toThrow(
        /must be a positive integer/i
      );
    }
  });

  it("the live config.embedDim resolves to the default in the offline env", () => {
    // test/setup/env.ts actively deletes LITELLM_EMBED_DIM from the environment,
    // so the offline suite is deterministic regardless of the surrounding shell/
    // .env and the single source of truth is the documented 768 default.
    expect(config.embedDim).toBe(768);
  });

  it("DEFAULT_EMBED_DIM stays in lockstep with the store's DEFAULT_EMBEDDING_DIM", () => {
    // The two constants live in different modules to avoid a cyclic import; this
    // pins them together so they can never silently drift.
    expect(DEFAULT_EMBED_DIM).toBe(DEFAULT_EMBEDDING_DIM);
  });
});

// -----------------------------------------------------------------------------
// Structural guard: the offline suite must NEVER import the LIVE probe, which is
// the ONLY module that performs an embedding call and writes docs/embeddings/.
// Mirrors test/phase5-server-entrypoint-guard.test.ts. Because no test imports
// src/probe-embedding.ts, `npm test` cannot make a live embedding call nor create
// the docs/embeddings evidence directory — that is exclusively the probe's job.
// -----------------------------------------------------------------------------
const testDir = path.dirname(fileURLToPath(import.meta.url));

async function collectTestFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) return collectTestFiles(full);
      if (entry.isFile() && entry.name.endsWith(".ts")) return [full];
      return [];
    })
  );
  return files.flat();
}

describe("Phase 7 (7d) — the live embedding probe is not imported by tests", () => {
  it("no test file imports src/probe-embedding.ts", async () => {
    const files = await collectTestFiles(testDir);
    expect(files.length).toBeGreaterThan(0);

    const offenders: string[] = [];
    for (const file of files) {
      const raw = await readFile(file, "utf8");
      if (/probe-embedding(\.js)?["']/.test(raw)) {
        offenders.push(path.relative(testDir, file));
      }
    }

    expect(
      offenders,
      offenders.length === 0
        ? ""
        : `The LIVE probe src/probe-embedding.ts must not be imported by tests ` +
            `(it makes a live embedding call and writes docs/embeddings). ` +
            `Offending test file(s): ${offenders.join(", ")}`
    ).toEqual([]);
  });
});
