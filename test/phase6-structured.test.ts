import { describe, it, expect, beforeEach, vi } from "vitest";

// Phase 6 (increment 6a) — structured-output helper.
//
// Fully OFFLINE and deterministic: the model boundary (createChatModel) is
// mocked with a fake that exposes `withStructuredOutput(schema).invoke()`
// returning a CANNED object (no network, no credentials). A shared, hoisted
// `control` lets each test choose the canned value / thrown error and inspect
// the exact ModelOptions createChatModel was called with (so the opus
// temperature-omit path is asserted without a live call).

const control = vi.hoisted(() => ({
  // Value returned by the fake structured model's invoke().
  next: undefined as unknown,
  // When set, invoke() throws this instead of returning `next`.
  throwErr: undefined as unknown,
  // Every ModelOptions createChatModel was called with, in order.
  captured: [] as Array<Record<string, unknown>>,
}));

vi.mock("../src/core/model.js", () => ({
  createChatModel: (opts?: Record<string, unknown>) => {
    control.captured.push(opts ?? {});
    return {
      withStructuredOutput: (_schema: unknown) => ({
        invoke: async (_messages: unknown): Promise<unknown> => {
          if (control.throwErr) throw control.throwErr;
          return control.next;
        },
      }),
    };
  },
}));

const {
  generateStructured,
  generateBudgetPlan,
  generatePlanningChecklist,
  decideStructuredModelOptions,
  isTemperatureOmitModel,
  DEFAULT_STRUCTURED_MODEL,
} = await import("../src/core/structured.js");
const { budgetPlanSchema, planningChecklistSchema } = await import(
  "../src/core/schemas.js"
);

const OPUS = "claude-opus-4-8";
const SONNET = "claude-sonnet-4-6";

const validBudget = {
  total: 20000,
  currency: "USD",
  categories: [
    { name: "Venue", amount: 8000 },
    { name: "Catering", amount: 6000 },
  ],
};

beforeEach(() => {
  control.next = undefined;
  control.throwErr = undefined;
  control.captured.length = 0;
});

describe("Phase 6 — decideStructuredModelOptions (pure opus temperature-omit)", () => {
  it("omits temperature (temperature: null) for claude-opus-4-8", () => {
    const opts = decideStructuredModelOptions(OPUS);
    expect(opts.model).toBe(OPUS);
    // null => createChatModel OMITS the temperature field entirely.
    expect(opts.temperature).toBeNull();
  });

  it("does NOT force temperature to null for sonnet (key absent)", () => {
    const opts = decideStructuredModelOptions(SONNET);
    expect(opts.model).toBe(SONNET);
    expect("temperature" in opts).toBe(false);
  });

  it("isTemperatureOmitModel is true for opus, false for sonnet", () => {
    expect(isTemperatureOmitModel(OPUS)).toBe(true);
    expect(isTemperatureOmitModel("CLAUDE-OPUS-4-8")).toBe(true);
    expect(isTemperatureOmitModel(SONNET)).toBe(false);
  });

  it("DEFAULT_STRUCTURED_MODEL is the fully-supported sonnet alias", () => {
    expect(DEFAULT_STRUCTURED_MODEL).toBe(SONNET);
  });
});

describe("Phase 6 — generateStructured (validated typed result)", () => {
  it("returns the validated, typed object from a canned model response", async () => {
    control.next = validBudget;
    const result = await generateStructured(budgetPlanSchema, "plan a budget");
    expect(result.total).toBe(20000);
    expect(result.categories).toHaveLength(2);
  });

  it("defaults to the sonnet model with temperature NOT forced to null", async () => {
    control.next = validBudget;
    await generateStructured(budgetPlanSchema, "plan a budget");
    expect(control.captured).toHaveLength(1);
    const opts = control.captured[0]!;
    expect(opts.model).toBe(SONNET);
    expect("temperature" in opts).toBe(false);
  });

  it("builds the opus model with temperature omitted (null) via the helper", async () => {
    control.next = validBudget;
    await generateStructured(budgetPlanSchema, "plan a budget", {
      model: OPUS,
    });
    expect(control.captured).toHaveLength(1);
    const opts = control.captured[0]!;
    expect(opts.model).toBe(OPUS);
    expect(opts.temperature).toBeNull();
  });

  it("applies Zod defaults on the way out (checklist done -> false)", async () => {
    control.next = { items: [{ title: "Book the venue" }] };
    const result = await generateStructured(
      planningChecklistSchema,
      "make a checklist"
    );
    expect(result.items[0]!.done).toBe(false);
  });

  it("throws a clear, concise error when the model output violates the schema", async () => {
    // Negative total violates a PER-FIELD rule even if the library returned it.
    control.next = { total: -5, categories: [{ name: "Venue", amount: 1 }] };
    let message = "";
    try {
      await generateStructured(budgetPlanSchema, "plan a budget");
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).toMatch(/failed schema validation/i);
    // Cleaner reason from issue messages — NOT the raw zod JSON blob.
    expect(message).toContain("must be > 0");
    expect(message).not.toContain("[{");
  });

  it("does NOT hard-fail a slightly over-allocated budget (generation path uses the plain schema)", async () => {
    // Sum (1500) exceeds total (1000): the OLD refined generation schema would
    // have thrown here. The plain LLM-facing schema must return it successfully;
    // the over-allocation rule is a domain concern (validateBudgetAllocation).
    control.next = {
      total: 1000,
      categories: [
        { name: "Venue", amount: 800 },
        { name: "Catering", amount: 700 },
      ],
    };
    const result = await generateBudgetPlan("plan a budget");
    expect(result.total).toBe(1000);
    expect(result.categories).toHaveLength(2);
  });

  it("fast-fails on empty/whitespace-only string input before calling the model", async () => {
    control.next = validBudget;
    await expect(
      generateStructured(budgetPlanSchema, "   ")
    ).rejects.toThrow(/non-empty input/i);
    // Model was never constructed/invoked.
    expect(control.captured).toHaveLength(0);
  });

  it("fast-fails on an empty message array before calling the model", async () => {
    control.next = validBudget;
    await expect(
      generateStructured(budgetPlanSchema, [])
    ).rejects.toThrow(/at least one input message/i);
    expect(control.captured).toHaveLength(0);
  });

  it("redacts secrets in a thrown model/transport error", async () => {
    // The setup env injects a fake key/baseURL; a provider error that echoes
    // them must be scrubbed before it reaches the caller/logs.
    control.throwErr = new Error(
      "boom key=sk-test-dummy-not-a-real-key url=http://localhost:0/test-fake-litellm"
    );
    let message = "";
    try {
      await generateStructured(budgetPlanSchema, "plan a budget");
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }
    expect(message).toContain("Structured generation failed");
    expect(message).not.toContain("sk-test-dummy-not-a-real-key");
    expect(message).not.toContain("test-fake-litellm");
    expect(message).toContain("[redacted-key]");
    expect(message).toContain("[redacted-url]");
  });

  it("passes a plain string input through as a human message (no throw)", async () => {
    control.next = validBudget;
    // Just assert the happy path works with the string overload + systemPrompt.
    const result = await generateStructured(budgetPlanSchema, "plan a budget", {
      systemPrompt: "You are Aria, a wedding planner.",
    });
    expect(result.total).toBe(20000);
  });
});

describe("Phase 6 — domain convenience wrappers", () => {
  it("generateBudgetPlan returns a validated BudgetPlan", async () => {
    control.next = validBudget;
    const plan = await generateBudgetPlan("budget for 100 guests");
    expect(plan.categories.map((c) => c.name)).toEqual(["Venue", "Catering"]);
  });

  it("generatePlanningChecklist returns a validated PlanningChecklist", async () => {
    control.next = { items: [{ title: "Book venue", when: "12mo before" }] };
    const checklist = await generatePlanningChecklist("what should I do first?");
    expect(checklist.items[0]!.title).toBe("Book venue");
    expect(checklist.items[0]!.done).toBe(false);
  });
});
