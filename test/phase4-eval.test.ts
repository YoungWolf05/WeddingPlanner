import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  EVAL_CATEGORIES,
  GLOBAL_MIN_LENGTH,
  parseDataset,
  scoreItem,
  aggregateScores,
  hasNumericBreakdown,
  hasListOrSteps,
  mentionsAny,
  steersBackToWeddings,
  nonWhitespaceLength,
  renderEvalMarkdown,
  renderEvalConsoleSummary,
  type EvalItem,
  type ItemRunResult,
} from "../src/core/eval.js";

// Phase 4 (increment 4e) — OFFLINE evaluation dataset + scorer test.
//
// Two responsibilities, both deterministic and fully offline (no network, no
// credentials, no live probe/runner import):
//   1) Dataset integrity — the committed evals/dataset.jsonl parses, has unique
//      ids and required fields, the right size, and spans every category
//      including an off-topic case.
//   2) Scorer correctness — the pure property scorers pass on synthetic GOOD
//      responses and fail on synthetic BAD responses (positive + negative
//      fixtures), so a real live run's scores are trustworthy.
//
// The guarantee that `npm test` makes no network call and writes no docs/eval
// results file is STRUCTURAL: this suite never imports the live runner
// (src/run-eval.ts) — only the pure, I/O-free src/core/eval.ts module. There is
// no runtime assertion for this; it holds by construction (nothing here can
// reach the runner's file I/O or provider calls).

// Resolve the committed dataset relative to THIS test file (working-dir safe).
const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "..");
const datasetPath = path.join(repoRoot, "evals", "dataset.jsonl");

function loadCommittedDataset(): EvalItem[] {
  return parseDataset(readFileSync(datasetPath, "utf8"));
}

describe("Phase 4 — eval dataset integrity", () => {
  const items = loadCommittedDataset();

  it("has ~10-15 items", () => {
    expect(items.length).toBeGreaterThanOrEqual(10);
    expect(items.length).toBeLessThanOrEqual(15);
  });

  it("every item has stable id, category, prompt, and expectations", () => {
    for (const item of items) {
      expect(item.id).toBeTruthy();
      expect(typeof item.id).toBe("string");
      expect(EVAL_CATEGORIES).toContain(item.category);
      expect(item.prompt.trim().length).toBeGreaterThan(0);
      expect(typeof item.expectations).toBe("object");
    }
  });

  it("all ids are unique", () => {
    const ids = items.map((i) => i.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("spans every representative category, including an off-topic case", () => {
    const present = new Set(items.map((i) => i.category));
    for (const category of EVAL_CATEGORIES) {
      expect(present, `missing category: ${category}`).toContain(category);
    }
    const offTopic = items.filter((i) => i.category === "off_topic");
    expect(offTopic.length).toBeGreaterThanOrEqual(1);
    // Every off-topic item must require a steer-back.
    for (const item of offTopic) {
      expect(item.expectations.mustSteerBack).toBe(true);
    }
  });

  it("at least one budget item expects a numeric breakdown", () => {
    const budget = items.filter((i) => i.category === "budget");
    expect(budget.length).toBeGreaterThanOrEqual(1);
    expect(budget.some((i) => i.expectations.expectsNumericBreakdown)).toBe(true);
  });

  it("at least one timeline item expects a list/steps", () => {
    const timeline = items.filter((i) => i.category === "timeline");
    expect(timeline.length).toBeGreaterThanOrEqual(1);
    expect(timeline.some((i) => i.expectations.expectsList)).toBe(true);
  });
});

describe("Phase 4 — parseDataset validation", () => {
  it("rejects duplicate ids", () => {
    const jsonl =
      JSON.stringify({
        id: "dup",
        category: "budget",
        prompt: "p",
        expectations: {},
      }) +
      "\n" +
      JSON.stringify({
        id: "dup",
        category: "timeline",
        prompt: "q",
        expectations: {},
      });
    expect(() => parseDataset(jsonl)).toThrow(/duplicate/i);
  });

  it("rejects an unknown category", () => {
    const jsonl = JSON.stringify({
      id: "x",
      category: "not-a-category",
      prompt: "p",
      expectations: {},
    });
    expect(() => parseDataset(jsonl)).toThrow(/category/i);
  });

  it("rejects an unknown expectation key (typo guard)", () => {
    const jsonl = JSON.stringify({
      id: "x",
      category: "budget",
      prompt: "p",
      expectations: { expectsLists: true },
    });
    expect(() => parseDataset(jsonl)).toThrow(/unknown expectation key/i);
  });

  it("rejects a missing prompt", () => {
    const jsonl = JSON.stringify({
      id: "x",
      category: "budget",
      prompt: "",
      expectations: {},
    });
    expect(() => parseDataset(jsonl)).toThrow(/prompt/i);
  });

  it("rejects invalid JSON with the line number", () => {
    expect(() => parseDataset("{ not json")).toThrow(/line 1/i);
  });

  it("ignores blank lines", () => {
    const jsonl =
      "\n" +
      JSON.stringify({
        id: "a",
        category: "budget",
        prompt: "p",
        expectations: {},
      }) +
      "\n\n";
    expect(parseDataset(jsonl)).toHaveLength(1);
  });
});

describe("Phase 4 — property detectors (unit)", () => {
  it("hasNumericBreakdown detects percentages, symbols, and currency words", () => {
    expect(hasNumericBreakdown("Venue: 40%, catering 30%")).toBe(true);
    expect(hasNumericBreakdown("Venue costs about $12,000")).toBe(true);
    expect(hasNumericBreakdown("Around 5000 USD for the venue")).toBe(true);
    expect(hasNumericBreakdown("Spend a lot on the venue and catering")).toBe(
      false
    );
  });

  it("hasListOrSteps detects bullets and numbered lists but not prose", () => {
    expect(hasListOrSteps("- book venue\n- hire caterer\n- pick a date")).toBe(
      true
    );
    expect(hasListOrSteps("1. Ceremony\n2. Cocktails\n3. Reception")).toBe(true);
    expect(hasListOrSteps("First book a venue - then hire a caterer.")).toBe(
      false
    );
  });

  it("mentionsAny is case-insensitive and matches any single term", () => {
    expect(mentionsAny("The VENUE is lovely", ["venue", "catering"])).toBe(true);
    expect(mentionsAny("nothing relevant here", ["venue", "catering"])).toBe(
      false
    );
  });

  it("steersBackToWeddings requires an explicit WEDDING-ANCHORED redirect", () => {
    // Legitimate steer-backs that name the wedding domain as the thing to
    // return to (tolerant of apostrophe/quote/spacing variants) pass.
    expect(
      steersBackToWeddings(
        "I'm here to help with wedding planning — shall we get back to your big day?"
      )
    ).toBe(true);
    expect(steersBackToWeddings("Let's focus on your wedding instead!")).toBe(
      true
    );
    expect(
      steersBackToWeddings("I'd suggest we stick to wedding planning.")
    ).toBe(true);
    expect(
      steersBackToWeddings("Happy to help you plan your wedding.")
    ).toBe(true);
    expect(
      steersBackToWeddings("I'm your wedding planner, not a coder.")
    ).toBe(true);
    // A substantive off-topic answer with no wedding context does not steer back.
    expect(
      steersBackToWeddings("Sure! The IndexError happens because the list is empty.")
    ).toBe(false);
  });

  it("steersBackToWeddings does NOT false-positive on domain nouns + a generic phrase (finding R1)", () => {
    // The exact previously-false-positive string: a genuinely off-topic
    // technical answer that merely uses wedding-domain nouns (venue/guest)
    // next to a generic preposition ("focus on"). It names no wedding-anchored
    // redirect, so it must NOT count as a steer-back.
    expect(
      steersBackToWeddings(
        "To fix the bug, focus on the venue booking API response and the guest count field."
      )
    ).toBe(false);
    // A couple more off-topic answers using domain nouns + generic redirects.
    expect(
      steersBackToWeddings(
        "Let's get back to the venue booking code and the guest table."
      )
    ).toBe(false);
    expect(
      steersBackToWeddings(
        "The reception buffer overflowed; stick to the venue array indices."
      )
    ).toBe(false);
  });

  it("nonWhitespaceLength ignores whitespace", () => {
    expect(nonWhitespaceLength("  a b\tc\n")).toBe(3);
  });
});

// Synthetic fixture item builder for scorer tests.
function item(
  id: string,
  category: EvalItem["category"],
  expectations: EvalItem["expectations"]
): EvalItem {
  return { id, category, prompt: "synthetic prompt", expectations };
}

// A long-enough filler so the global length floor never accidentally fails a
// positive fixture that is testing a DIFFERENT property.
const FILLER =
  "This is a detailed, helpful wedding planning answer with plenty of substance. ";

describe("Phase 4 — scoreItem positive fixtures (good responses pass)", () => {
  it("budget response with a numeric breakdown and list passes", () => {
    const it0 = item("b", "budget", {
      expectsNumericBreakdown: true,
      expectsList: true,
      mustMention: ["venue", "catering"],
    });
    const response =
      "Here is a budget breakdown:\n" +
      "- Venue: 40% (about $12,000)\n" +
      "- Catering: 30%\n" +
      "- Photography: 15%";
    const score = scoreItem(it0, response);
    expect(score.passed).toBe(true);
    expect(score.properties.every((p) => p.passed)).toBe(true);
  });

  it("timeline response with numbered steps passes", () => {
    const it0 = item("t", "timeline", {
      expectsList: true,
      mustMention: ["ceremony"],
    });
    const response =
      FILLER +
      "\n1. Morning: getting ready\n2. Ceremony at noon\n3. Reception in the evening";
    expect(scoreItem(it0, response).passed).toBe(true);
  });

  it("off-topic response that steers back passes", () => {
    const it0 = item("o", "off_topic", { mustSteerBack: true });
    const response =
      "I'd love to help, but I'm your wedding planning assistant! " +
      "Let's get back to planning your wedding — what can I help you with?";
    expect(scoreItem(it0, response).passed).toBe(true);
  });
});

describe("Phase 4 — scoreItem negative fixtures (bad responses fail)", () => {
  it("budget response without any numbers fails the numericBreakdown property", () => {
    const it0 = item("b", "budget", { expectsNumericBreakdown: true });
    const score = scoreItem(it0, FILLER + "Spend more on the venue than flowers.");
    expect(score.passed).toBe(false);
    const prop = score.properties.find((p) => p.property === "numericBreakdown");
    expect(prop?.passed).toBe(false);
  });

  it("timeline response with no list fails the listOrSteps property", () => {
    const it0 = item("t", "timeline", { expectsList: true });
    const score = scoreItem(
      it0,
      FILLER + "Just do things in a sensible order over the year."
    );
    expect(score.passed).toBe(false);
    expect(
      score.properties.find((p) => p.property === "listOrSteps")?.passed
    ).toBe(false);
  });

  it("off-topic response that actually answers off-topic fails steerBack", () => {
    const it0 = item("o", "off_topic", { mustSteerBack: true });
    // A substantive off-topic answer with NO wedding context (so the FILLER,
    // which mentions weddings, is deliberately not used here).
    const score = scoreItem(
      it0,
      "Sure, I can help debug that. The IndexError is because your list index " +
        "exceeds its length; add a bounds check before accessing the element."
    );
    expect(score.passed).toBe(false);
    expect(
      score.properties.find((p) => p.property === "steerBack")?.passed
    ).toBe(false);
  });

  it("off-topic answer using domain nouns + a generic phrase fails steerBack (finding R1)", () => {
    // Pins finding R1 at the scoreItem level: the off-topic item requires a
    // steer-back, and this genuinely off-topic technical answer merely reuses
    // wedding-domain nouns (venue/guest) next to a generic preposition
    // ("focus on"). It must score as a FAIL, not a false-positive pass.
    const it0 = item("o", "off_topic", { mustSteerBack: true });
    const score = scoreItem(
      it0,
      "To fix the bug, focus on the venue booking API response and the guest count field."
    );
    expect(score.passed).toBe(false);
    expect(
      score.properties.find((p) => p.property === "steerBack")?.passed
    ).toBe(false);
  });

  it("an empty response fails the global length floor", () => {
    const it0 = item("x", "budget", {});
    const score = scoreItem(it0, "");
    expect(score.passed).toBe(false);
    const prop = score.properties.find((p) => p.property === "minLength");
    expect(prop?.passed).toBe(false);
    // Match the exact message form so an incidental "40" elsewhere in a future
    // reason string cannot satisfy this assertion.
    expect(prop?.reason).toContain(`< ${GLOBAL_MIN_LENGTH}`);
  });

  it("a response missing all required mentions fails mustMention", () => {
    const it0 = item("m", "vendor_selection", {
      mustMention: ["photographer", "portfolio"],
    });
    const score = scoreItem(it0, FILLER + "Pick someone whose work you like.");
    expect(
      score.properties.find((p) => p.property === "mustMention")?.passed
    ).toBe(false);
  });
});

describe("Phase 4 — aggregateScores", () => {
  it("computes item, property, and per-category tallies", () => {
    const scores = [
      scoreItem(item("a", "budget", { expectsNumericBreakdown: true }), FILLER + "40%"),
      scoreItem(item("b", "budget", { expectsNumericBreakdown: true }), FILLER + "no numbers"),
      scoreItem(
        item("c", "off_topic", { mustSteerBack: true }),
        FILLER + "Let's get back to your wedding planning."
      ),
    ];
    const agg = aggregateScores(scores);
    expect(agg.itemsTotal).toBe(3);
    expect(agg.itemsPassed).toBe(2);
    expect(agg.propertiesTotal).toBeGreaterThan(0);
    const budget = agg.byCategory.find((c) => c.category === "budget");
    expect(budget).toEqual({ category: "budget", passed: 1, total: 2 });
    // Categories with no items are omitted.
    expect(agg.byCategory.find((c) => c.category === "timeline")).toBeUndefined();
  });
});

describe("Phase 4 — eval rendering (offline, deterministic)", () => {
  const meta = {
    runTimestampUtc: "2026-07-28T10:00:00.000Z",
    model: "claude-sonnet-4-6",
    baseUrlHost: "litellm.example.internal",
    maskedKey: "sk-…(redacted)",
  };

  function sampleResults(): ItemRunResult[] {
    return [
      {
        score: scoreItem(
          item("b", "budget", { expectsNumericBreakdown: true }),
          FILLER + "Venue 40%"
        ),
        prompt: "budget prompt",
        latencyMs: 1234,
      },
      {
        score: scoreItem(item("o", "off_topic", { mustSteerBack: true }), ""),
        prompt: "off-topic prompt",
        latencyMs: 100,
        errorReason: "timed out",
      },
    ];
  }

  it("markdown includes header, aggregate, per-category, and per-item sections", () => {
    const md = renderEvalMarkdown(meta, sampleResults());
    expect(md).toContain("# Wedding Planner Evaluation Baseline");
    expect(md).toContain("2026-07-28T10:00:00.000Z");
    expect(md).toContain("claude-sonnet-4-6");
    expect(md).toContain("litellm.example.internal");
    expect(md).toContain("## Aggregate");
    expect(md).toContain("## Per-category summary");
    expect(md).toContain("## Per-item results");
    // The masked key never expands to a full-looking secret.
    expect(md).not.toMatch(/sk-[A-Za-z0-9]{20,}/);
    // An errored item is rendered as ERROR with its (already-redacted) reason.
    expect(md).toContain("ERROR");
    expect(md).toContain("timed out");
  });

  it("console summary reports items, properties, categories, and errored ids", () => {
    const summary = renderEvalConsoleSummary(sampleResults());
    expect(summary).toContain("Items passed:");
    expect(summary).toContain("Properties passed:");
    expect(summary).toContain("budget:");
    expect(summary).toContain("Errored items: o");
  });
});
