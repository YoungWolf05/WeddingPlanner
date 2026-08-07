import { describe, expect, it } from "vitest";
import {
  deriveThreadTitle,
  MAX_TITLE_LENGTH,
  threadTitleLabel,
  UNTITLED_PLACEHOLDER,
} from "../src/lib/title.js";

// BUG 1: the auto-title derivation is a pure function (message -> title). These
// tests pin its whitespace / length / ellipsis rules directly.

describe("deriveThreadTitle", () => {
  it("trims and returns short messages verbatim", () => {
    expect(deriveThreadTitle("  Book a venue  ")).toBe("Book a venue");
  });

  it("collapses internal whitespace (incl. newlines) to single spaces", () => {
    expect(deriveThreadTitle("plan\n\nthe   spring    wedding")).toBe(
      "plan the spring wedding"
    );
  });

  it("returns null for empty / whitespace-only messages", () => {
    expect(deriveThreadTitle("")).toBeNull();
    expect(deriveThreadTitle("   \n\t  ")).toBeNull();
  });

  it("truncates long messages on a word boundary with an ellipsis", () => {
    const long =
      "Help me plan an elegant outdoor garden wedding for next spring with a hundred guests";
    const title = deriveThreadTitle(long);
    expect(title).not.toBeNull();
    const result = title!;
    // The ellipsis is appended and the body stays within budget.
    expect(result.endsWith("…")).toBe(true);
    expect(result.length).toBeLessThanOrEqual(MAX_TITLE_LENGTH + 1);
    // A word boundary cut does not slice a word: no trailing partial word before
    // the ellipsis (the char before the ellipsis is not mid-word whitespace).
    expect(result).not.toContain("  ");
  });

  it("hard-cuts a single very long token (no usable word boundary)", () => {
    const token = "a".repeat(MAX_TITLE_LENGTH + 20);
    const title = deriveThreadTitle(token);
    expect(title).not.toBeNull();
    const result = title!;
    expect(result.endsWith("…")).toBe(true);
    expect(result.length).toBe(MAX_TITLE_LENGTH + 1);
  });
});

describe("threadTitleLabel", () => {
  it("falls back to the placeholder for null / empty titles", () => {
    expect(threadTitleLabel(null)).toBe(UNTITLED_PLACEHOLDER);
    expect(threadTitleLabel(undefined)).toBe(UNTITLED_PLACEHOLDER);
    expect(threadTitleLabel("   ")).toBe(UNTITLED_PLACEHOLDER);
  });

  it("returns the trimmed title when present", () => {
    expect(threadTitleLabel("  Venues  ")).toBe("Venues");
  });

  it("never returns the old '(untitled)' string", () => {
    expect(threadTitleLabel(null)).not.toContain("untitled");
  });
});
