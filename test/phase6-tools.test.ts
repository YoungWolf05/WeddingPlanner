import { describe, it, expect } from "vitest";
import {
  daysUntil,
  splitBudget,
  daysUntilTool,
  splitBudgetTool,
  weddingTools,
  DEFAULT_BUDGET_SPLIT,
  type DaysUntilResult,
} from "../src/core/tools.js";
import {
  budgetPlanSchema,
  validateBudgetAllocation,
  type BudgetPlan,
} from "../src/core/schemas.js";

// Phase 6 (increment 6b) — SAFE, deterministic, pure, read-only domain tools.
//
// Fully OFFLINE and deterministic: these exercise ONLY the pure functions and
// the LangChain tool() wrappers' Zod validation. NO model construction, NO
// network. `daysUntil` is driven with a fixed injected `now` so nothing depends
// on the real clock.

describe("Phase 6 — daysUntil (pure, UTC calendar-day, injectable now)", () => {
  it("future date returns a positive count + direction 'future' (2026-01-01 -> 2026-12-12 = 345)", () => {
    const result = daysUntil("2026-12-12", "2026-01-01T00:00:00Z");
    expect(result).toEqual<DaysUntilResult>({
      targetDate: "2026-12-12",
      daysUntil: 345,
      direction: "future",
    });
  });

  it("exit-criterion date 2026-12-12 from a fixed now computes the exact day count", () => {
    // now = 2026-12-01 -> 11 days until 2026-12-12.
    const result = daysUntil("2026-12-12", "2026-12-01T00:00:00Z");
    expect(result.daysUntil).toBe(11);
    expect(result.direction).toBe("future");
    expect(result.targetDate).toBe("2026-12-12");
  });

  it("today returns 0 + direction 'today'", () => {
    const result = daysUntil("2026-06-15", "2026-06-15T09:30:00Z");
    expect(result.daysUntil).toBe(0);
    expect(result.direction).toBe("today");
  });

  it("past date returns a negative count + direction 'past'", () => {
    const result = daysUntil("2026-01-01", "2026-01-11T00:00:00Z");
    expect(result.daysUntil).toBe(-10);
    expect(result.direction).toBe("past");
  });

  it("leap-year boundary: 2028-02-28 -> 2028-03-01 is 2 days (Feb 29 counted)", () => {
    const result = daysUntil("2028-03-01", "2028-02-28T00:00:00Z");
    expect(result.daysUntil).toBe(2);
    expect(result.direction).toBe("future");
    // And to the leap day itself is exactly 1 day.
    expect(daysUntil("2028-02-29", "2028-02-28T00:00:00Z").daysUntil).toBe(1);
  });

  it("is time-of-day independent (same calendar date, different times => same result)", () => {
    const early = daysUntil("2026-12-12", "2026-12-01T00:00:00Z");
    const late = daysUntil("2026-12-12", "2026-12-01T23:59:59Z");
    const midday = daysUntil("2026-12-12", "2026-12-01T12:34:56Z");
    expect(early).toEqual(late);
    expect(early).toEqual(midday);
    expect(early.daysUntil).toBe(11);
  });

  it("accepts a Date object for now as well as a string", () => {
    const asDate = daysUntil("2026-12-12", new Date("2026-12-01T00:00:00Z"));
    expect(asDate.daysUntil).toBe(11);
  });

  it.each([
    ["2026-13-01", "impossible month"],
    ["2026-02-30", "impossible day (Feb 30)"],
    ["2026-04-31", "impossible day (Apr 31)"],
    ["not-a-date", "non-date text"],
    ["", "empty string"],
    ["2026-1-1", "non-zero-padded"],
    ["2026/12/12", "wrong separators"],
  ])("throws a clear error for invalid date %s (%s)", (bad) => {
    expect(() => daysUntil(bad, "2026-01-01T00:00:00Z")).toThrow(/invalid date/i);
  });
});

describe("Phase 6 — splitBudget (pure, largest-remainder, over-allocation rule)", () => {
  it("default split (no categories): 7 categories, exact 40/25/10/8/7/5/5, sums EXACTLY to total, passes allocation", () => {
    const plan = splitBudget(30000);
    expect(plan.categories).toHaveLength(7);
    expect(plan.total).toBe(30000);

    // Names + display percentages match the documented default set.
    expect(plan.categories.map((c) => c.name)).toEqual([
      "Venue",
      "Catering",
      "Photography",
      "Attire",
      "Flowers",
      "Music",
      "Misc",
    ]);
    expect(plan.categories.map((c) => c.percentage)).toEqual([
      40, 25, 10, 8, 7, 5, 5,
    ]);
    // 30000 * the default percentages -> whole amounts.
    expect(plan.categories.map((c) => c.amount)).toEqual([
      12000, 7500, 3000, 2400, 2100, 1500, 1500,
    ]);

    const sum = plan.categories.reduce((s, c) => s + c.amount, 0);
    expect(sum).toBe(30000);
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
    expect(DEFAULT_BUDGET_SPLIT.reduce((s, c) => s + c.percentage, 0)).toBe(100);
  });

  it("default split with a rounding-prone total sums EXACTLY to total (999 across default set)", () => {
    const plan = splitBudget(999);
    const sum = plan.categories.reduce((s, c) => s + c.amount, 0);
    expect(sum).toBe(999);
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });

  it("custom percentages summing to 100 with a rounding-forcing total (1000 split in thirds) sum EXACTLY to total", () => {
    const plan = splitBudget(1000, [
      { name: "A", percentage: 33.34 },
      { name: "B", percentage: 33.33 },
      { name: "C", percentage: 33.33 },
    ]);
    const sum = plan.categories.reduce((s, c) => s + c.amount, 0);
    // Largest-remainder makes the integer amounts add up to the total exactly.
    expect(sum).toBe(1000);
    // Ideal amounts are 333.4 / 333.3 / 333.3 -> 333 each + 1 leftover to the
    // largest remainder (A) => 334, 333, 333.
    expect(plan.categories.map((c) => c.amount)).toEqual([334, 333, 333]);
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });

  it("rounding-sum-exactness is explicitly preserved for another prone case (10000 split in thirds)", () => {
    const plan = splitBudget(10000, [
      { name: "A", percentage: 33.34 },
      { name: "B", percentage: 33.33 },
      { name: "C", percentage: 33.33 },
    ]);
    const sum = plan.categories.reduce((s, c) => s + c.amount, 0);
    expect(sum).toBe(10000);
  });

  it("custom amounts: percentages are derived and the amount sum is preserved", () => {
    const plan = splitBudget(20000, [
      { name: "Venue", amount: 8000 },
      { name: "Catering", amount: 5000 },
      { name: "Photography", amount: 2000 },
    ]);
    const sum = plan.categories.reduce((s, c) => s + c.amount, 0);
    expect(sum).toBe(15000); // amounts under-allocate; we do not invent money
    expect(plan.categories.map((c) => c.amount)).toEqual([8000, 5000, 2000]);
    // Derived display percentages relative to the 20000 total.
    expect(plan.categories.map((c) => c.percentage)).toEqual([40, 25, 10]);
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });

  it("over-allocation (custom amounts sum > total*1.001) is REJECTED with a clear error", () => {
    expect(() =>
      splitBudget(1000, [
        { name: "Venue", amount: 800 },
        { name: "Catering", amount: 700 },
      ])
    ).toThrow(/must not exceed the total/i);
  });

  // --- B1 regression: fractional over-allocation must surface the DOMAIN
  // over-allocation message, NOT an incidental "percentage must be <= 100"
  // per-field error, and consistently regardless of how it is distributed. ---
  it("B1: single-category fractional over-allocation (1000.9 on total 1000) throws the OVER-ALLOCATION message, not percentage<=100", () => {
    // Previously this threw "produced an invalid budget plan: percentage must
    // be <= 100" because 1000.9 rounds up to 1001 and the display percentage
    // (1001/1000 = 100.1) tripped the per-field check first.
    let message = "";
    try {
      splitBudget(1000, [{ name: "Venue", amount: 1000.9 }]);
      throw new Error("expected splitBudget to throw");
    } catch (e) {
      message = (e as Error).message;
    }
    expect(message).toMatch(/must not exceed the total/i);
    // Explicitly assert it is NOT the misleading per-field percentage error.
    expect(message).not.toMatch(/percentage must be <= 100/i);
  });

  it("B1: another single-category fractional over-allocation (1500.6 on total 1500) throws the over-allocation message", () => {
    // 1500.6 <= 1500*1.001 (=1501.5) so the raw-multiplier guard passes, but
    // Math.round(1500.6)=1501 > 1500 — this specifically exercises the new
    // integer-rounding over-allocation guard.
    expect(() =>
      splitBudget(1500, [{ name: "Venue", amount: 1500.6 }])
    ).toThrow(/must not exceed the total/i);
  });

  it("B1: multi-category fractional over-allocation (500.9 + 500.05 on total 1000) throws the SAME over-allocation message (consistency)", () => {
    // Net 1000.95 rounds to 1001 > 1000. Different distribution, same intended
    // domain error — proving the message no longer depends on distribution.
    expect(() =>
      splitBudget(1000, [
        { name: "A", amount: 500.9 },
        { name: "B", amount: 500.05 },
      ])
    ).toThrow(/must not exceed the total/i);
  });

  it("B1: display percentages never exceed 100 for a valid at-total amount plan", () => {
    // 1000.4 rounds to 1000 (not over-allocated); the single category's display
    // percentage must be exactly 100, never 100.1.
    const plan = splitBudget(1000, [{ name: "Venue", amount: 1000.4 }]);
    expect(plan.categories[0]!.amount).toBe(1000);
    expect(plan.categories[0]!.percentage).toBe(100);
    expect(
      plan.categories.every((c) => (c.percentage ?? 0) <= 100)
    ).toBe(true);
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });

  // --- R2: genuinely exercise the STRICT (validateBudgetAllocation /
  // budgetPlanStrictSchema) branch producing the domain reason. Percentage mode
  // has no amount-mode guard, so percentages at the top of the sum-to-100
  // tolerance whose rounded raw amounts exceed the total reach strict. ---
  it("R2: percentage-mode over-allocation via rounding (60 + 40.1 on total 1000) reaches the strict branch and yields the domain reason", () => {
    // rawAmounts = [600, 401], sum 1001 > roundedTotal 1000. No amount-mode
    // guard runs (percentage mode), so budgetPlanStrictSchema catches it.
    expect(() =>
      splitBudget(1000, [
        { name: "A", percentage: 60 },
        { name: "B", percentage: 40.1 },
      ])
    ).toThrow(/must not exceed the total/i);
  });

  // --- R1: amount-mode whole-unit rounding ("don't invent money"). The plan
  // sums to Math.round(sum of provided amounts), a whole number of units. ---
  it("R1: amount-mode sum rounds to whole units (100.4 + 100.4 => plan sums to Math.round(200.8) = 201)", () => {
    const plan = splitBudget(1000, [
      { name: "A", amount: 100.4 },
      { name: "B", amount: 100.4 },
    ]);
    const sum = plan.categories.reduce((s, c) => s + c.amount, 0);
    expect(sum).toBe(201); // Math.round(200.8)
    // A fractional, NON-over-allocated input still succeeds with a sensible
    // whole-unit plan (largest-remainder gives 101 + 100).
    expect(plan.categories.map((c) => c.amount)).toEqual([101, 100]);
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });

  it("under-allocation (custom amounts sum < total) is ACCEPTED", () => {
    const plan = splitBudget(30000, [
      { name: "Venue", amount: 12000 },
      { name: "Catering", amount: 6000 },
    ]);
    expect(plan.total).toBe(30000);
    const sum = plan.categories.reduce((s, c) => s + c.amount, 0);
    expect(sum).toBe(18000);
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });

  it("amounts just within tolerance are accepted (sum == total exactly)", () => {
    const plan = splitBudget(1000, [
      { name: "Venue", amount: 600 },
      { name: "Catering", amount: 400 },
    ]);
    expect(plan.categories.reduce((s, c) => s + c.amount, 0)).toBe(1000);
  });

  it.each([
    [() => splitBudget(0), /total must be a positive/i, "total <= 0"],
    [() => splitBudget(-5), /total must be a positive/i, "negative total"],
    [
      () => splitBudget(1000, [{ name: "Venue", amount: -1 }]),
      /non-negative/i,
      "negative amount",
    ],
    [
      () => splitBudget(1000, [{ name: "Venue", percentage: 150 }]),
      /between 0 and 100/i,
      "percentage > 100",
    ],
    [
      () => splitBudget(1000, [{ name: "", amount: 100 }]),
      /name must be non-empty/i,
      "empty name",
    ],
  ])("rejects per-field invalid input (%s)", (fn, pattern) => {
    expect(fn).toThrow(pattern as RegExp);
  });

  it("rejects a MIXED percentage+amount category set as ambiguous", () => {
    expect(() =>
      splitBudget(1000, [
        { name: "Venue", percentage: 50 },
        { name: "Catering", amount: 500 },
      ])
    ).toThrow(/EITHER percentages OR amounts/i);
  });

  it("rejects a single category carrying BOTH percentage and amount", () => {
    expect(() =>
      splitBudget(1000, [{ name: "Venue", percentage: 100, amount: 1000 }])
    ).toThrow(/EITHER percentages OR amounts/i);
  });

  it("rejects percentages that do not sum to ~100", () => {
    expect(() =>
      splitBudget(1000, [
        { name: "Venue", percentage: 40 },
        { name: "Catering", percentage: 40 },
      ])
    ).toThrow(/must sum to ~100/i);
  });

  it("sets currency only when provided", () => {
    const withCurrency = splitBudget(1000, undefined, "USD");
    expect(withCurrency.currency).toBe("USD");
    const withoutCurrency = splitBudget(1000);
    expect(withoutCurrency.currency).toBeUndefined();
  });

  it("result validates against the 6a plain budgetPlanSchema (per-field)", () => {
    const plan = splitBudget(30000);
    const parsed = budgetPlanSchema.safeParse(plan);
    expect(parsed.success).toBe(true);
    // Type-level: the returned value is a BudgetPlan.
    const typed: BudgetPlan = plan;
    expect(typed.categories.length).toBeGreaterThan(0);
  });
});

describe("Phase 6 — tool() wrappers (offline; Zod input validation + structured artifact)", () => {
  it("days_until tool returns the structured artifact for valid input", async () => {
    // A tool with responseFormat 'content_and_artifact' returns a ToolMessage
    // whose `artifact` is the structured result.
    const msg = await daysUntilTool.invoke({
      // ToolCall form so we get a ToolMessage (with artifact) back.
      name: "days_until",
      args: { date: "2026-12-12" },
      id: "call-1",
      type: "tool_call",
    });
    expect(msg.artifact).toMatchObject({
      targetDate: "2026-12-12",
      direction: expect.any(String),
    });
    expect(typeof (msg.artifact as DaysUntilResult).daysUntil).toBe("number");
    expect(typeof msg.content).toBe("string");
  });

  it("days_until tool REJECTS a badly-formatted date via the Zod input schema", async () => {
    await expect(
      daysUntilTool.invoke({
        name: "days_until",
        args: { date: "not-a-date" },
        id: "call-2",
        type: "tool_call",
      })
    ).rejects.toThrow();
  });

  it("split_budget tool returns the structured BudgetPlan artifact for valid input", async () => {
    const msg = await splitBudgetTool.invoke({
      name: "split_budget",
      args: { total: 30000 },
      id: "call-3",
      type: "tool_call",
    });
    const plan = msg.artifact as BudgetPlan;
    expect(plan.total).toBe(30000);
    expect(plan.categories).toHaveLength(7);
    expect(plan.categories.reduce((s, c) => s + c.amount, 0)).toBe(30000);
    expect(typeof msg.content).toBe("string");
  });

  it("split_budget tool REJECTS total <= 0 via the Zod input schema", async () => {
    await expect(
      splitBudgetTool.invoke({
        name: "split_budget",
        args: { total: 0 },
        id: "call-4",
        type: "tool_call",
      })
    ).rejects.toThrow();
  });

  it("split_budget tool surfaces a domain error (over-allocation) as a thrown error", async () => {
    await expect(
      splitBudgetTool.invoke({
        name: "split_budget",
        args: {
          total: 1000,
          categories: [
            { name: "Venue", amount: 800 },
            { name: "Catering", amount: 700 },
          ],
        },
        id: "call-5",
        type: "tool_call",
      })
    ).rejects.toThrow();
  });

  it("exposes a tool registry (weddingTools) with correct names for 6c", () => {
    expect(weddingTools.map((t) => t.name)).toEqual([
      "days_until",
      "split_budget",
    ]);
  });
});
