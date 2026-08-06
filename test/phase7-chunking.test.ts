import { describe, it, expect } from "vitest";
import {
  chunkText,
  DEFAULT_CHUNK_SIZE,
  DEFAULT_CHUNK_OVERLAP,
  DEFAULT_SEPARATORS,
} from "../src/core/chunking.js";
import {
  computeChunkId,
  computeContentHash,
  computeDocumentId,
  normalizeContent,
} from "../src/core/knowledge-store.js";

// Phase 7 (7b) — DETERMINISTIC chunking.
//
// Fully OFFLINE and pure: no store, no embedder, no network. These tests pin the
// determinism + edge-case contract the ingestion pipeline (and therefore the 7a
// content-hash chunk_id scheme) depends on.

describe("Phase 7 (7b) — chunking determinism + parameters", () => {
  it("exports sensible documented defaults", () => {
    expect(DEFAULT_CHUNK_SIZE).toBe(1000);
    expect(DEFAULT_CHUNK_OVERLAP).toBe(200);
    expect([...DEFAULT_SEPARATORS]).toEqual(["\n\n", "\n", ". ", " ", ""]);
    // The default separator list is frozen (a caller cannot mutate the shared
    // default out from under other callers).
    expect(Object.isFrozen(DEFAULT_SEPARATORS)).toBe(true);
  });

  it("is DETERMINISTIC: identical input + options yield byte-for-byte identical chunks across runs", () => {
    const content =
      "Wedding budget overview.\n\nAllocate the venue first. " +
      "Then catering. Then photography and flowers.\n\n" +
      "Book vendors at least six months ahead for peak season.";
    const a = chunkText(content, { chunkSize: 40, chunkOverlap: 10 });
    const b = chunkText(content, { chunkSize: 40, chunkOverlap: 10 });
    expect(a).toEqual(b);
    // And the derived chunk_ids (the 7a scheme) are identical across runs — the
    // property idempotent ingestion relies on.
    const documentId = computeDocumentId(content);
    const idsA = a.map((text, i) =>
      computeChunkId(documentId, i, computeContentHash(text))
    );
    const idsB = b.map((text, i) =>
      computeChunkId(documentId, i, computeContentHash(text))
    );
    expect(idsA).toEqual(idsB);
  });

  it("normalizes before chunking: CRLF vs LF produce identical chunks", () => {
    const lf = "alpha beta\n\ngamma delta epsilon zeta";
    const crlf = "alpha beta\r\n\r\ngamma delta epsilon zeta";
    expect(chunkText(crlf, { chunkSize: 20, chunkOverlap: 5 })).toEqual(
      chunkText(lf, { chunkSize: 20, chunkOverlap: 5 })
    );
  });

  it("honors chunkSize: no chunk exceeds the configured size (recursion splits oversized runs to characters)", () => {
    // A single unsplittable run longer than chunkSize is split down via the ""
    // separator so every chunk fits the budget.
    const content = "abcdefghijklmnopqrstuvwxyz0123456789";
    const chunks = chunkText(content, { chunkSize: 8, chunkOverlap: 2 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.length).toBeLessThanOrEqual(8);
    }
  });

  it("honors chunkSize on realistic MULTI-SEPARATOR prose: EVERY chunk fits the budget (exercises the paragraph/sentence/word merge path)", () => {
    // Multi-paragraph prose with sentence ('. ') and word (' ') boundaries plus
    // paragraph breaks ('\n\n'). At a small size the paragraphs are too big to
    // keep whole, so the splitter recurses through '\n\n' -> '. ' -> ' ' and
    // MERGES pieces back up to the budget. This exercises the merge path (not the
    // single-unsplittable-run "" case) and asserts the size invariant holds
    // across it.
    const prose = [
      "Set the budget first. Then build the guest list. Confirm both with family.",
      "Pick a venue for the season. Match it to your style. Check the headcount.",
      "Book catering early. Lock photography next. Reserve florals before peak.",
    ].join("\n\n");
    const chunkSize = 50;
    const chunks = chunkText(prose, { chunkSize, chunkOverlap: 10 });
    // The prose is far larger than one chunk, so it must split into several.
    expect(chunks.length).toBeGreaterThan(1);
    // The size invariant holds for EVERY produced chunk across the merge path.
    for (const c of chunks) {
      expect(c.length).toBeLessThanOrEqual(chunkSize);
    }
  });

  it("produces the exact expected chunks + documented overlap for a known word list", () => {
    // Deterministic worked example: five 4-char words, size 10, overlap 4.
    const content = "aaaa bbbb cccc dddd eeee";
    const chunks = chunkText(content, { chunkSize: 10, chunkOverlap: 4 });
    expect(chunks).toEqual([
      "aaaa bbbb",
      "bbbb cccc",
      "cccc dddd",
      "dddd eeee",
    ]);
    // Each adjacent pair shares exactly one overlapping word.
    expect(chunks[0]!.endsWith("bbbb")).toBe(true);
    expect(chunks[1]!.startsWith("bbbb")).toBe(true);
  });

  it("respects the separator ORDER: paragraph boundaries are preferred over spaces", () => {
    // With a generous size the two paragraphs stay intact as separate chunks
    // (split on "\n\n" first), rather than being merged/word-split.
    const content = "paragraph one text.\n\nparagraph two text.";
    const chunks = chunkText(content, { chunkSize: 25, chunkOverlap: 0 });
    expect(chunks).toEqual(["paragraph one text.", "paragraph two text."]);
  });

  describe("edge cases", () => {
    it("empty content → no chunks", () => {
      expect(chunkText("")).toEqual([]);
    });

    it("whitespace-only content → no chunks", () => {
      expect(chunkText("   \n\n \t\r\n  ")).toEqual([]);
    });

    it("content shorter than one chunk → a single trimmed chunk", () => {
      const chunks = chunkText("  short note  ", { chunkSize: 100, chunkOverlap: 20 });
      expect(chunks).toEqual(["short note"]);
    });

    it("content exactly at the boundary stays a single chunk", () => {
      const content = "0123456789"; // length 10
      expect(chunkText(content, { chunkSize: 10, chunkOverlap: 2 })).toEqual([
        "0123456789",
      ]);
    });

    it("overlap >= size is rejected (guard/validation)", () => {
      expect(() => chunkText("x", { chunkSize: 10, chunkOverlap: 10 })).toThrow(
        /chunkOverlap .* must be strictly less than chunkSize/i
      );
      expect(() => chunkText("x", { chunkSize: 10, chunkOverlap: 11 })).toThrow(
        /strictly less than chunkSize/i
      );
    });

    it("non-positive chunkSize is rejected", () => {
      expect(() => chunkText("x", { chunkSize: 0 })).toThrow(/positive integer/i);
      expect(() => chunkText("x", { chunkSize: -5 })).toThrow(/positive integer/i);
      expect(() => chunkText("x", { chunkSize: 1.5 })).toThrow(/positive integer/i);
    });

    it("empty separators list is rejected", () => {
      expect(() => chunkText("x y z", { separators: [] })).toThrow(
        /separators must be a non-empty list/i
      );
    });
  });

  it("chunk texts reconstruct only from NORMALIZED input (boundaries align with the id scheme)", () => {
    // Every emitted chunk is a substring-derived, trimmed slice of the
    // normalized content, so its content hash is stable.
    const content = "one two three four five six seven eight nine ten";
    const normalized = normalizeContent(content);
    for (const c of chunkText(content, { chunkSize: 15, chunkOverlap: 3 })) {
      // Each chunk's non-space tokens all appear in the normalized source.
      for (const token of c.split(" ")) {
        expect(normalized).toContain(token);
      }
    }
  });
});
