import { z } from "zod";

// Phase 6 (increment 6a): typed wedding-domain data schemas.
//
// This module is the single source of truth for the STRUCTURED wedding-domain
// artifacts the assistant can produce (budgets, planning checklists). Each shape
// is defined ONCE as a Zod schema and the corresponding TypeScript type is
// derived from it via `z.infer`, so the runtime validator and the compile-time
// type can never drift apart.
//
// Zod version note: the project depends on `zod` directly (v4). LangChain
// (`@langchain/core` / `@langchain/openai` 1.x) declares `zod: "^3.25.76 || ^4"`
// and its `withStructuredOutput` accepts these v4 schema objects via its zod
// interop layer, so the SAME schema value is used both as the LLM's
// structured-output contract (see src/core/structured.ts) and as the
// defense-in-depth runtime validator.
//
// The schemas are deliberately kept benign and PII-free (budgets and generic
// planning tasks — no names, contacts, or addresses).
//
// SCHEMA LAYERING (LLM-facing vs domain):
//   The LLM-FACING schemas below are PLAIN z.object schemas — per-field
//   constraints ONLY, no cross-field `.refine()`. This matters because
//   `model.withStructuredOutput(schema)` converts the schema to JSON Schema for
//   provider-native structured output, and that conversion SILENTLY DROPS
//   `.refine()` checks. A cross-field refinement on the generation path would
//   therefore give NO generation-time guarantee while still hard-failing
//   plausible model outputs at the post-hoc `safeParse` (e.g. a first-pass
//   budget that over-allocates by a hair). It is also the one real
//   provider-compatibility risk (a ZodEffects schema instead of a plain object).
//   Cross-field BUSINESS RULES therefore live SEPARATELY as an exported domain
//   validator (`validateBudgetAllocation`) / strict schema variant
//   (`budgetPlanStrictSchema`) applied at the domain/tool boundary (6b), NOT on
//   the LLM generation path.

// A single line item within a wedding budget breakdown.
//
//   - name:       human-readable category label (e.g. "Venue", "Catering").
//                 Must be non-empty.
//   - amount:     money allocated to this category, in the plan's currency.
//                 Non-negative (0 is allowed — a category may be a placeholder).
//   - percentage: OPTIONAL share of the total budget this category represents,
//                 0..100. Optional because the model may express a breakdown by
//                 absolute amounts only.
export const budgetCategorySchema = z.object({
  name: z.string().min(1, "category name must not be empty"),
  amount: z.number().nonnegative("category amount must be >= 0"),
  percentage: z
    .number()
    .min(0, "percentage must be >= 0")
    .max(100, "percentage must be <= 100")
    .optional(),
});

/** A single wedding-budget line item (category + allocated amount). */
export type BudgetCategory = z.infer<typeof budgetCategorySchema>;

// The LLM-FACING wedding budget breakdown schema. PLAIN z.object (per-field
// constraints only) so it converts cleanly to JSON Schema for provider-native
// structured output AND is the defense-in-depth `safeParse` contract on the
// generation path. See the SCHEMA LAYERING note above: the cross-field
// over-allocation rule is intentionally NOT here — it lives in
// `validateBudgetAllocation` / `budgetPlanStrictSchema` below.
//
//   - total:      the overall budget. Must be a positive number.
//   - currency:   OPTIONAL ISO-ish currency label (e.g. "USD"). Free-form and
//                 optional so the model is not forced to guess a currency.
//   - categories: at least one line item (a budget with no categories is not a
//                 useful breakdown).
//   - notes:      OPTIONAL free-form planning notes.
export const budgetPlanSchema = z.object({
  total: z.number().positive("total budget must be > 0"),
  currency: z.string().min(1).optional(),
  categories: z
    .array(budgetCategorySchema)
    .min(1, "budget must have at least one category"),
  notes: z.string().optional(),
});

/**
 * A validated wedding budget breakdown: a positive total plus one or more
 * non-negative category allocations. This is the LLM-facing type; the
 * over-allocation business rule is enforced separately at the domain boundary
 * (see `validateBudgetAllocation` / `budgetPlanStrictSchema`).
 */
export type BudgetPlan = z.infer<typeof budgetPlanSchema>;

// Floating-point tolerance for the over-allocation business rule: category
// amounts may sum to at most `total * BUDGET_ALLOCATION_TOLERANCE`. The 0.1%
// slack absorbs rounding in model-produced numbers so an effectively-exact
// allocation is not rejected by fp noise.
export const BUDGET_ALLOCATION_TOLERANCE = 1.001;

/**
 * Domain BUSINESS RULE (kept OFF the LLM generation path): the sum of category
 * amounts must not EXCEED the total budget (within a tiny fp tolerance). This
 * catches clearly broken plans that over-allocate the budget WITHOUT forcing an
 * exact sum — under-allocation is explicitly allowed (contingency / not-yet-
 * assigned funds are normal in a real wedding budget), so a reasonable plan is
 * never rejected for leaving headroom. Intended consumer: the 6b budget tool /
 * domain layer, which applies this AFTER a plan has been generated + parsed.
 *
 * Returns a discriminated result so callers get a clear, non-throwing reason.
 */
export function validateBudgetAllocation(
  plan: BudgetPlan
): { ok: true } | { ok: false; reason: string } {
  const allocated = plan.categories.reduce((sum, c) => sum + c.amount, 0);
  if (allocated <= plan.total * BUDGET_ALLOCATION_TOLERANCE) {
    return { ok: true };
  }
  return {
    ok: false,
    reason: "sum of category amounts must not exceed the total budget",
  };
}

/**
 * STRICT budget schema variant: the LLM-facing `budgetPlanSchema` plus the
 * over-allocation business rule (`validateBudgetAllocation`). Use at the
 * domain/tool boundary when you want a single parse that also enforces the
 * cross-field rule. Deliberately NOT used on the `withStructuredOutput`
 * generation path (see SCHEMA LAYERING note). Its inferred type is
 * assignment-compatible with `BudgetPlan` (a `.refine()` narrows values, not
 * the type).
 */
export const budgetPlanStrictSchema = budgetPlanSchema.refine(
  (plan) => validateBudgetAllocation(plan).ok,
  {
    message: "sum of category amounts must not exceed the total budget",
    path: ["categories"],
  }
);

/** A budget plan that also satisfies the over-allocation business rule. */
export type BudgetPlanStrict = z.infer<typeof budgetPlanStrictSchema>;

// A single actionable planning task within a checklist.
//
//   - title: what needs to happen (e.g. "Book the venue"). Non-empty.
//   - when:  OPTIONAL free-form timing hint (e.g. "12 months before",
//            "2025-06"). Kept free-form so the model can express whatever
//            granularity is natural rather than a rigid date type.
//   - done:  OPTIONAL completion flag; defaults to false so a freshly generated
//            checklist item is treated as outstanding unless stated otherwise.
export const checklistItemSchema = z.object({
  title: z.string().min(1, "checklist item title must not be empty"),
  when: z.string().min(1).optional(),
  done: z.boolean().optional().default(false),
});

/** A single wedding-planning checklist task. */
export type ChecklistItem = z.infer<typeof checklistItemSchema>;

// A validated wedding planning checklist.
//
//   - items: at least one task (an empty checklist is not useful output).
//
// Deliberately a flat list rather than a rigid month-by-month structure: each
// item carries an OPTIONAL `when` timing hint, which is flexible enough to
// represent a timeline without forcing the model into a fixed calendar shape.
export const planningChecklistSchema = z.object({
  items: z
    .array(checklistItemSchema)
    .min(1, "checklist must have at least one item"),
});

/**
 * A validated wedding-planning checklist: one or more actionable tasks, each
 * with an optional timing hint and completion flag.
 */
export type PlanningChecklist = z.infer<typeof planningChecklistSchema>;

// Phase 8 (increment 8a): the GROUNDED-ANSWER structured-output contract.
//
// The model fills this via `withStructuredOutput` (reusing the Phase 6
// structured-output plumbing in src/core/structured.ts) after being given a
// numbered, delimited block of retrieved context (see src/core/prompts.ts and
// src/core/rag.ts). It is the LLM-facing shape for a grounded RAG answer.
//
// CITATION TRUST MODEL — APP-ASSIGNED INTEGER MARKERS, NOT MODEL IDs.
// -------------------------------------------------------------------
// `citations` is a list of APP-ASSIGNED CITATION MARKERS: integers that INDEX
// INTO the app-provided numbered context block (e.g. `[1, 2]` means "context
// entries 1 and 2 support this answer"). The markers are assigned by APP CODE
// when it renders the context — the model only ECHOES the marker numbers it
// relied on. The model MUST NEVER emit chunk/document IDs; those are trusted,
// app-owned identifiers resolved by app code (see the retriever's
// RetrievedChunk metadata), never taken from raw model text. This is the crux of
// Phase 8 exit criterion 1.
//
// 8a establishes this marker contract and returns the raw markers plus the
// app-owned marker->RetrievedChunk map (see src/core/rag.ts). The AUTHORITATIVE
// resolution of markers back to trusted/authorized chunk/document IDs, and the
// dropping/flagging of any marker NOT in the retrieved set, is TODO(8b) — 8a
// deliberately does not validate or drop out-of-range markers here.
//
// The schema is a PLAIN z.object (per-field constraints only, no cross-field
// `.refine()`) so it converts cleanly to JSON Schema for provider-native
// structured output AND doubles as the defense-in-depth `safeParse` contract
// (mirrors the SCHEMA LAYERING note above).
//
//   - answer:              the grounded natural-language answer. MAY be empty
//                          when `insufficientEvidence` is true (nothing to say).
//   - citations:           app-assigned MARKER NUMBERS (non-negative integers)
//                          that index the numbered context block. Deduped by
//                          convention at the app layer; the schema only enforces
//                          the per-element integer/non-negative rule. Defaults to
//                          an empty array so a model that cites nothing is valid.
//   - insufficientEvidence: true when the retrieved context does NOT support a
//                          grounded answer (the model must NOT fabricate). The
//                          deterministic low-score / empty-retrieval POLICY that
//                          also drives this flag is TODO(8c); 8a only handles the
//                          trivial empty-retrieval case (see src/core/rag.ts).
export const groundedAnswerSchema = z.object({
  answer: z.string(),
  citations: z
    .array(
      z
        .number()
        .int("citation marker must be an integer")
        .nonnegative("citation marker must be a non-negative integer")
    )
    .default([]),
  insufficientEvidence: z.boolean(),
});

/**
 * A grounded RAG answer produced via structured output: the natural-language
 * `answer`, the APP-ASSIGNED integer citation `markers` it relied on, and an
 * `insufficientEvidence` flag. Citation markers index the numbered context block
 * and are resolved to trusted, app-owned IDs by app code (TODO(8b)) — they are
 * NEVER trusted source IDs coming from the model.
 */
export type GroundedAnswer = z.infer<typeof groundedAnswerSchema>;
