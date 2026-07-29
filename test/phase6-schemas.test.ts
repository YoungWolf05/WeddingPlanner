import { describe, it, expect } from "vitest";
import {
  budgetPlanSchema,
  budgetCategorySchema,
  budgetPlanStrictSchema,
  validateBudgetAllocation,
  planningChecklistSchema,
  checklistItemSchema,
  type BudgetPlan,
  type BudgetCategory,
  type PlanningChecklist,
  type ChecklistItem,
} from "../src/core/schemas.js";

// Phase 6 (increment 6a) — wedding-domain Zod schema validation.
//
// Fully OFFLINE and deterministic: these exercise ONLY the pure Zod schemas,
// no model/network. They assert that valid domain objects parse, that clearly
// invalid ones fail with useful messages, and that the LIGHT refinements accept
// reasonable outputs while rejecting broken ones (without being over-strict).

describe("Phase 6 — budgetPlanSchema", () => {
  const validPlan = {
    total: 30000,
    currency: "USD",
    categories: [
      { name: "Venue", amount: 12000, percentage: 40 },
      { name: "Catering", amount: 9000, percentage: 30 },
      { name: "Photography", amount: 4000 },
    ],
    notes: "First-pass breakdown; leaves headroom for contingency.",
  };

  it("parses a well-formed budget plan and returns the typed value", () => {
    const parsed = budgetPlanSchema.parse(validPlan);
    expect(parsed.total).toBe(30000);
    expect(parsed.categories).toHaveLength(3);
    expect(parsed.categories[0]!.name).toBe("Venue");
  });

  it("accepts a minimal plan: positive total + one category, no optionals", () => {
    const result = budgetPlanSchema.safeParse({
      total: 5000,
      categories: [{ name: "Venue", amount: 0 }],
    });
    expect(result.success).toBe(true);
    // amount === 0 is allowed (placeholder category), currency/notes optional.
    if (result.success) expect(result.data.categories[0]!.amount).toBe(0);
  });

  it("rejects a non-positive total with a useful message", () => {
    const result = budgetPlanSchema.safeParse({
      total: 0,
      categories: [{ name: "Venue", amount: 100 }],
    });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0]!.message).toContain("must be > 0");
    }
  });

  it("rejects a negative category amount", () => {
    const result = budgetPlanSchema.safeParse({
      total: 5000,
      categories: [{ name: "Venue", amount: -1 }],
    });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(
        result.error.issues.some((i) => i.message.includes(">= 0"))
      ).toBe(true);
    }
  });

  it("rejects a missing required field (categories)", () => {
    const result = budgetPlanSchema.safeParse({ total: 5000 });
    expect(result.success).toBe(false);
  });

  it("rejects an empty categories array (not a useful breakdown)", () => {
    const result = budgetPlanSchema.safeParse({ total: 5000, categories: [] });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(
        result.error.issues.some((i) =>
          i.message.includes("at least one category")
        )
      ).toBe(true);
    }
  });

  it("rejects a percentage outside 0..100", () => {
    const result = budgetPlanSchema.safeParse({
      total: 5000,
      categories: [{ name: "Venue", amount: 100, percentage: 150 }],
    });
    expect(result.success).toBe(false);
  });

  it("BASE (LLM-facing) schema ACCEPTS a slightly over-allocated budget", () => {
    // The generation path uses this plain schema and must NOT hard-fail a
    // plausible first-pass budget that over-allocates — the cross-field rule
    // is enforced separately at the domain boundary, not here.
    const result = budgetPlanSchema.safeParse({
      total: 1000,
      categories: [
        { name: "Venue", amount: 800 },
        { name: "Catering", amount: 700 },
      ],
    });
    expect(result.success).toBe(true);
  });
});

describe("Phase 6 — validateBudgetAllocation (domain over-allocation rule)", () => {
  it("REJECTS categories that over-allocate the total, with a reason", () => {
    const plan = budgetPlanSchema.parse({
      total: 1000,
      categories: [
        { name: "Venue", amount: 800 },
        { name: "Catering", amount: 700 },
      ],
    });
    const result = validateBudgetAllocation(plan);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toContain("must not exceed the total");
    }
  });

  it("ALLOWS under-allocation (headroom / contingency)", () => {
    // Sum (18000) is well under total (30000) — a normal, valid plan.
    const plan = budgetPlanSchema.parse({
      total: 30000,
      categories: [
        { name: "Venue", amount: 12000 },
        { name: "Catering", amount: 6000 },
      ],
    });
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });

  it("tolerates tiny floating-point overshoot at the total", () => {
    // Exact sum equal to total (with fp noise) must still be accepted.
    const plan = budgetPlanSchema.parse({
      total: 100,
      categories: [
        { name: "A", amount: 33.33 },
        { name: "B", amount: 33.33 },
        { name: "C", amount: 33.34 },
      ],
    });
    expect(validateBudgetAllocation(plan)).toEqual({ ok: true });
  });
});

describe("Phase 6 — budgetPlanStrictSchema (base schema + domain rule)", () => {
  it("REJECTS clear over-allocation at parse time", () => {
    const result = budgetPlanStrictSchema.safeParse({
      total: 1000,
      categories: [
        { name: "Venue", amount: 800 },
        { name: "Catering", amount: 700 },
      ],
    });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(
        result.error.issues.some((i) =>
          i.message.includes("must not exceed the total")
        )
      ).toBe(true);
    }
  });

  it("ACCEPTS under-allocation and preserves the BudgetPlan shape", () => {
    const result = budgetPlanStrictSchema.safeParse({
      total: 30000,
      categories: [
        { name: "Venue", amount: 12000 },
        { name: "Catering", amount: 6000 },
      ],
    });
    expect(result.success).toBe(true);
    if (result.success) {
      // Type is assignment-compatible with BudgetPlan.
      const plan: BudgetPlan = result.data;
      expect(plan.total).toBe(30000);
    }
  });

  it("still enforces per-field rules (e.g. positive total)", () => {
    const result = budgetPlanStrictSchema.safeParse({
      total: 0,
      categories: [{ name: "Venue", amount: 0 }],
    });
    expect(result.success).toBe(false);
  });
});

describe("Phase 6 — budgetCategorySchema", () => {
  it("rejects an empty category name", () => {
    const result = budgetCategorySchema.safeParse({ name: "", amount: 10 });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0]!.message).toContain("must not be empty");
    }
  });
});

describe("Phase 6 — planningChecklistSchema", () => {
  it("parses a well-formed checklist and defaults `done` to false", () => {
    const parsed = planningChecklistSchema.parse({
      items: [
        { title: "Book the venue", when: "12 months before" },
        { title: "Send invitations", when: "3 months before", done: true },
      ],
    });
    expect(parsed.items).toHaveLength(2);
    // `done` omitted on the first item -> default false.
    expect(parsed.items[0]!.done).toBe(false);
    expect(parsed.items[1]!.done).toBe(true);
  });

  it("accepts an item with only a title (when optional)", () => {
    const result = planningChecklistSchema.safeParse({
      items: [{ title: "Pick a color palette" }],
    });
    expect(result.success).toBe(true);
  });

  it("rejects an empty checklist", () => {
    const result = planningChecklistSchema.safeParse({ items: [] });
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(
        result.error.issues.some((i) =>
          i.message.includes("at least one item")
        )
      ).toBe(true);
    }
  });

  it("rejects an item with an empty title", () => {
    const result = planningChecklistSchema.safeParse({
      items: [{ title: "" }],
    });
    expect(result.success).toBe(false);
  });

  it("rejects a non-boolean `done`", () => {
    const result = checklistItemSchema.safeParse({
      title: "Book venue",
      done: "yes",
    });
    expect(result.success).toBe(false);
  });
});

describe("Phase 6 — inferred types compile (type-level assertions)", () => {
  it("BudgetPlan / BudgetCategory / PlanningChecklist / ChecklistItem line up with the schemas", () => {
    // These are compile-time checks: if z.infer drifts from the intended shape
    // the assignments below stop type-checking (`npm run typecheck`). At runtime
    // they simply construct valid values.
    const category: BudgetCategory = { name: "Venue", amount: 12000 };
    const plan: BudgetPlan = { total: 100, categories: [category] };
    const item: ChecklistItem = { title: "Book venue", done: false };
    const checklist: PlanningChecklist = { items: [item] };

    expect(plan.categories[0]).toEqual(category);
    expect(checklist.items[0]).toEqual(item);
  });
});
