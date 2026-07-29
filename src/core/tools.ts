import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  budgetPlanSchema,
  budgetPlanStrictSchema,
  validateBudgetAllocation,
  BUDGET_ALLOCATION_TOLERANCE,
  type BudgetPlan,
  type BudgetCategory,
} from "./schemas.js";

// Phase 6 (increment 6b): SAFE, deterministic, pure, read-only domain tools.
//
// This module exposes two wedding-planning helper tools as (1) PURE functions
// and (2) thin LangChain `tool()` wrappers with Zod input schemas. They are the
// tool FOUNDATION only — this increment deliberately does NOT wire an agent /
// tool-loop (that is 6c) and does NOT touch the HTTP service or CLI.
//
// SAFETY CONTRACT (why these are "safe" tools):
//   - NO I/O: no filesystem, no network, no database, no environment mutation.
//   - NO model construction: these never touch createChatModel/ChatOpenAI or
//     OpenAIEmbeddings (the 4b/4c single-factory guards stay satisfied).
//   - PURE + TOTAL(-ish): each pure function is a deterministic computation over
//     its inputs. `daysUntil` takes an injectable `now` so it is deterministic
//     under test; the tool wrapper defaults `now` to the real clock. The only
//     non-total behavior is a THROWN typed error on invalid input (bad date,
//     over-allocated / malformed budget), which is the documented failure mode.
//   - Because they are synchronous pure computations there is nothing to
//     "time out": there is no async boundary, no unbounded loop, and no blocking
//     call. A per-tool timer around sync code would be dead weight (and a source
//     of flakiness), so we deliberately do NOT add one. Robustness instead comes
//     from Zod input validation at the wrapper boundary plus clear typed errors.
//
// The tool wrappers use `responseFormat: "content_and_artifact"` so each returns
// a tuple `[humanReadableSummary, structuredResult]`: the summary string is what
// a model would read, and the structured artifact is the exact typed object
// (`DaysUntilResult` / `BudgetPlan`) for programmatic consumers such as 6c.

// ---------------------------------------------------------------------------
// TOOL 1 — days_until
// ---------------------------------------------------------------------------

/** Which side of "today" the target calendar date falls on. */
export type DaysUntilDirection = "future" | "past" | "today";

/**
 * Structured result of {@link daysUntil}.
 *
 *   - targetDate: the input date NORMALIZED to `YYYY-MM-DD`.
 *   - daysUntil:  SIGNED whole-day difference (target - now), in UTC calendar
 *                 days. Positive = future, negative = past, 0 = today.
 *   - direction:  "future" | "past" | "today", always consistent with the sign
 *                 of `daysUntil`.
 */
export interface DaysUntilResult {
  targetDate: string;
  daysUntil: number;
  direction: DaysUntilDirection;
}

// Strict YYYY-MM-DD shape check. This is a FORMAT gate only — it guarantees the
// four-two-two digit layout; real calendar validity (month 1..12, day valid for
// the month, no 2026-02-30) is enforced afterward by round-tripping through a
// UTC Date. Both checks together reject bad format, impossible months/days, and
// non-existent dates.
const ISO_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

// Milliseconds in one whole day. Used only for a UTC-midnight-to-UTC-midnight
// difference, so no DST/leap-second drift can creep in (both operands are UTC
// epoch millis at 00:00:00Z).
const MS_PER_DAY = 24 * 60 * 60 * 1000;

/**
 * Parse a strict `YYYY-MM-DD` string into the UTC-midnight epoch millis for that
 * calendar day, REJECTING anything that is not a real ISO calendar date.
 *
 * Rejection cases (all throw a clear, non-leaking Error):
 *   - wrong format (not exactly `YYYY-MM-DD`, e.g. "not-a-date", "", "2026/1/1")
 *   - impossible month (e.g. "2026-13-01")
 *   - impossible day for the month (e.g. "2026-02-30", "2026-04-31")
 *
 * Validity is confirmed by constructing the date via `Date.UTC` and checking
 * that its UTC year/month/day round-trip EXACTLY to the parsed integers — a
 * mismatch means JS normalized an out-of-range component (e.g. Feb 30 -> Mar 2),
 * which we treat as invalid.
 */
function parseUtcCalendarDate(dateStr: string): number {
  if (!ISO_DATE_PATTERN.test(dateStr)) {
    throw new Error(
      `Invalid date "${dateStr}": expected an ISO calendar date in YYYY-MM-DD format`
    );
  }
  // Safe integer parses: the regex guarantees pure digit groups.
  const year = Number(dateStr.slice(0, 4));
  const month = Number(dateStr.slice(5, 7)); // 1..12 expected
  const day = Number(dateStr.slice(8, 10)); // 1..31 expected

  const epochMs = Date.UTC(year, month - 1, day);
  const roundTrip = new Date(epochMs);
  if (
    roundTrip.getUTCFullYear() !== year ||
    roundTrip.getUTCMonth() !== month - 1 ||
    roundTrip.getUTCDate() !== day
  ) {
    throw new Error(
      `Invalid date "${dateStr}": not a real calendar date`
    );
  }
  return epochMs;
}

// Reduce an arbitrary `now` (Date or ISO-ish string) to the UTC-midnight epoch
// millis of its CALENDAR DAY, discarding the time-of-day. This is what makes the
// day difference independent of the time on the clock (and thus DST-proof): only
// the UTC year/month/day of `now` are used.
function nowToUtcMidnightMs(now: Date | string): number {
  const d = typeof now === "string" ? new Date(now) : now;
  const ms = d.getTime();
  if (Number.isNaN(ms)) {
    throw new Error("Invalid `now`: not a parseable date");
  }
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
}

/**
 * PURE: compute the SIGNED, UTC calendar-day difference from `now` to
 * `targetDate`.
 *
 * Design (user-approved):
 *   - Date-only, UTC calendar-day semantics: `targetDate` is parsed STRICTLY as
 *     `YYYY-MM-DD` and both endpoints are collapsed to UTC midnight, so the
 *     result is a whole number of days with no time-of-day or DST drift.
 *   - `now` is INJECTABLE (Date or string) for deterministic tests; the tool
 *     wrapper defaults it to `new Date()` (the real clock). Only the UTC DATE
 *     part of `now` is used, so the same calendar day at different times yields
 *     the same answer.
 *   - Invalid/unparseable `targetDate` (bad format, month 13, day 32,
 *     non-existent dates like 2026-02-30) THROWS a clear, non-leaking Error.
 *
 * @param targetDate strict `YYYY-MM-DD` calendar date.
 * @param now        reference "today"; defaults to the real clock. Injectable.
 * @returns {@link DaysUntilResult} with a signed day count + consistent direction.
 */
export function daysUntil(
  targetDate: string,
  now: Date | string = new Date()
): DaysUntilResult {
  const targetMs = parseUtcCalendarDate(targetDate);
  const nowMs = nowToUtcMidnightMs(now);

  // Both operands are exact UTC-midnight epoch millis, so this division is a
  // whole number; Math.round guards against any residual fp representation.
  const diffDays = Math.round((targetMs - nowMs) / MS_PER_DAY);

  const direction: DaysUntilDirection =
    diffDays > 0 ? "future" : diffDays < 0 ? "past" : "today";

  // Re-emit the normalized YYYY-MM-DD (already validated) so the output form is
  // canonical regardless of input spacing/casing (input is already strict, so
  // this is effectively an identity — but it documents the contract).
  const normalized = new Date(targetMs).toISOString().slice(0, 10);

  return { targetDate: normalized, daysUntil: diffDays, direction };
}

// ---------------------------------------------------------------------------
// TOOL 2 — split_budget
// ---------------------------------------------------------------------------

/**
 * DEFAULT wedding-budget category split, used when the caller supplies no
 * categories. Percentages sum to EXACTLY 100, so the resulting plan always
 * satisfies the over-allocation rule (sum == total). Order is preserved in the
 * output.
 */
export const DEFAULT_BUDGET_SPLIT: ReadonlyArray<{
  name: string;
  percentage: number;
}> = [
  { name: "Venue", percentage: 40 },
  { name: "Catering", percentage: 25 },
  { name: "Photography", percentage: 10 },
  { name: "Attire", percentage: 8 },
  { name: "Flowers", percentage: 7 },
  { name: "Music", percentage: 5 },
  { name: "Misc", percentage: 5 },
];

/** A single category as ACCEPTED by {@link splitBudget} (input, not output). */
export interface SplitBudgetCategoryInput {
  name: string;
  percentage?: number;
  amount?: number;
}

// Tolerance for "custom percentages must sum to ~100". Mirrors the spirit of
// BUDGET_ALLOCATION_TOLERANCE (0.1% slack) but applies to the percentage total
// so rounding in caller-provided percentages is not rejected.
//
// N.B. this is DELIBERATELY separate from the 6a BUDGET_ALLOCATION_TOLERANCE:
// this one is an ABSOLUTE slack in PERCENTAGE POINTS on the sum-to-100 check,
// whereas BUDGET_ALLOCATION_TOLERANCE (1.001) is a MULTIPLIER on the total for
// the amount over-allocation check. They govern different modes and units.
const PERCENTAGE_SUM_TOLERANCE = 0.1; // absolute, in percentage points

/**
 * Distribute `total` across integer category amounts that sum EXACTLY to
 * `total`, using the LARGEST-REMAINDER (Hamilton) method.
 *
 * Each `rawAmounts[i]` is the ideal (possibly fractional) amount. We floor each,
 * then hand out the leftover currency units (total - sum of floors) one at a
 * time to the categories with the largest fractional remainders. This guarantees
 * `sum(result) === roundedTotal` with the fairest integer rounding and no drift.
 */
function largestRemainderRound(
  rawAmounts: number[],
  roundedTotal: number
): number[] {
  const floors = rawAmounts.map((a) => Math.floor(a));
  const remainders = rawAmounts.map((a, i) => ({
    index: i,
    frac: a - floors[i]!,
  }));
  const allocated = floors.reduce((s, f) => s + f, 0);
  let leftover = roundedTotal - allocated;

  // Hand out leftover units to the largest fractional remainders first. Ties
  // break by lower index for deterministic output.
  remainders.sort((a, b) => b.frac - a.frac || a.index - b.index);
  const result = floors.slice();
  for (let i = 0; i < remainders.length && leftover > 0; i++) {
    result[remainders[i]!.index]! += 1;
    leftover--;
  }
  return result;
}

/**
 * PURE: build a validated {@link BudgetPlan} by splitting `total` across
 * categories.
 *
 * Behavior (user-approved):
 *   - No `categories` -> use {@link DEFAULT_BUDGET_SPLIT} (Venue 40, Catering 25,
 *     Photography 10, Attire 8, Flowers 7, Music 5, Misc 5). Sums to 100%, so it
 *     always passes the over-allocation rule.
 *   - Custom categories accept EITHER percentages OR amounts, not both:
 *       * PERCENTAGE MODE: every category has `percentage` and no `amount`.
 *         Percentages must sum to ~100 (± PERCENTAGE_SUM_TOLERANCE). Amounts are
 *         derived from `total`.
 *       * AMOUNT MODE: every category has `amount` and no `percentage`. Amounts
 *         are used as the ideal split; percentages are derived from `total`.
 *     A MIXED set (some percentage-only, some amount-only, or a single category
 *     carrying both) is REJECTED as ambiguous — the mode must be consistent.
 *   - ROUNDING: category amounts are rounded to WHOLE currency units via
 *     largest-remainder (Hamilton) so they sum EXACTLY to `Math.round(total)`.
 *     Displayed percentages are recomputed from the final integer amounts and
 *     rounded for display, keeping them internally consistent with the amounts.
 *   - OVER-ALLOCATION: the resulting plan is validated with the 6a
 *     `validateBudgetAllocation` / `budgetPlanStrictSchema`, so an AMOUNT-MODE
 *     split whose amounts exceed `total` (beyond the 0.1% tolerance) is REJECTED.
 *     Under-allocation is allowed (headroom / contingency).
 *   - The returned object is validated against the 6a plain `budgetPlanSchema`
 *     for per-field correctness before it is returned.
 *
 * @param total      overall budget; must be > 0.
 * @param categories optional custom categories (percentage OR amount mode).
 * @param currency   optional currency label; set on the plan only if provided.
 * @returns a validated {@link BudgetPlan} whose category amounts sum to
 *          `Math.round(total)`.
 * @throws  a clear Error on non-positive total, invalid/mixed categories,
 *          percentages not summing to ~100, or over-allocation.
 */
export function splitBudget(
  total: number,
  categories?: SplitBudgetCategoryInput[],
  currency?: string
): BudgetPlan {
  if (!(total > 0) || !Number.isFinite(total)) {
    throw new Error("split_budget: total must be a positive, finite number");
  }

  const roundedTotal = Math.round(total);
  if (roundedTotal <= 0) {
    throw new Error(
      "split_budget: total is too small to split into whole currency units"
    );
  }

  // Resolve the ideal (pre-rounding) amount per category and the working list of
  // { name } entries, honoring the percentage-vs-amount mode rules.
  let names: string[];
  let rawAmounts: number[];

  if (categories === undefined || categories.length === 0) {
    // DEFAULT split: derive ideal amounts from the fixed percentages.
    names = DEFAULT_BUDGET_SPLIT.map((c) => c.name);
    rawAmounts = DEFAULT_BUDGET_SPLIT.map((c) => (c.percentage / 100) * total);
  } else {
    // Validate names up front (mirrors the 6a per-field rule) and classify mode.
    for (const c of categories) {
      if (typeof c.name !== "string" || c.name.trim() === "") {
        throw new Error("split_budget: every category name must be non-empty");
      }
    }

    const hasAnyPercentage = categories.some((c) => c.percentage !== undefined);
    const hasAnyAmount = categories.some((c) => c.amount !== undefined);

    if (hasAnyPercentage && hasAnyAmount) {
      throw new Error(
        "split_budget: provide EITHER percentages OR amounts for all categories, not both"
      );
    }
    if (!hasAnyPercentage && !hasAnyAmount) {
      throw new Error(
        "split_budget: custom categories must specify either a percentage or an amount"
      );
    }

    names = categories.map((c) => c.name);

    if (hasAnyPercentage) {
      // PERCENTAGE MODE: every category must carry a valid percentage.
      const pcts = categories.map((c) => {
        if (c.percentage === undefined) {
          throw new Error(
            "split_budget: in percentage mode every category needs a percentage"
          );
        }
        if (
          !Number.isFinite(c.percentage) ||
          c.percentage < 0 ||
          c.percentage > 100
        ) {
          throw new Error(
            "split_budget: each percentage must be between 0 and 100"
          );
        }
        return c.percentage;
      });
      const pctSum = pcts.reduce((s, p) => s + p, 0);
      if (Math.abs(pctSum - 100) > PERCENTAGE_SUM_TOLERANCE) {
        throw new Error(
          `split_budget: percentages must sum to ~100 (got ${pctSum})`
        );
      }
      rawAmounts = pcts.map((p) => (p / 100) * total);
    } else {
      // AMOUNT MODE: every category must carry a valid non-negative amount.
      rawAmounts = categories.map((c) => {
        if (c.amount === undefined) {
          throw new Error(
            "split_budget: in amount mode every category needs an amount"
          );
        }
        if (!Number.isFinite(c.amount) || c.amount < 0) {
          throw new Error(
            "split_budget: each amount must be a non-negative, finite number"
          );
        }
        return c.amount;
      });

      // Over-allocation guard, using the raw provided amounts so the error
      // reflects the caller's actual input (the 6a rule/tolerance). We check the
      // ROUNDED integer sum against the rounded total, NOT just raw < total*tol:
      // rounding each amount up to whole currency units can push the integer sum
      // above the rounded total (e.g. a single 1000.9 rounds to 1001 > 1000).
      // Rejecting when `Math.round(rawSum) > roundedTotal` here — BEFORE any
      // display-percentage computation — makes the over-allocation error
      // CONSISTENT regardless of how the overshoot is distributed across
      // categories, and it is what guarantees the display-percentage step below
      // can never exceed 100 for an accepted plan (see the percentage note).
      const amountSum = rawAmounts.reduce((s, a) => s + a, 0);
      if (
        amountSum > total * BUDGET_ALLOCATION_TOLERANCE ||
        Math.round(amountSum) > roundedTotal
      ) {
        throw new Error(
          "split_budget: sum of category amounts must not exceed the total budget"
        );
      }
    }
  }

  // AMOUNT MODE may under-allocate: in that case the integer amounts should sum
  // to the ROUNDED SUM OF AMOUNTS, not to the total (we must not invent money).
  // PERCENTAGE / DEFAULT mode always targets the full rounded total.
  //
  // NOTE on `targetSum = Math.round(rawSum)`: this is the rounded sum of the
  // IDEAL (pre-rounding) amounts, which can differ BOTH from the literal
  // provided sum (it is rounded to a whole unit, e.g. 200.8 -> 201) AND from
  // `sum(round(each amount))` (largest-remainder distributes to hit `targetSum`
  // exactly, which is fairer than independently rounding each amount). This is
  // the intentional whole-unit "don't invent money" behavior: the plan sums to
  // `targetSum`, a whole number of currency units near the provided total.
  const rawSum = rawAmounts.reduce((s, a) => s + a, 0);
  const targetSum = Math.round(rawSum);

  const amounts = largestRemainderRound(rawAmounts, targetSum);

  // Recompute display percentages from the FINAL integer amounts so amount and
  // percentage are always internally consistent. Rounded to one decimal place.
  //
  // The denominator is `roundedTotal` (the SAME rounded value used for amounts
  // and reported as `plan.total`), NOT the raw `total`. Using the raw total here
  // was the B1 bug: a legitimately at-total plan whose amount rounds up (e.g.
  // 1001 against a raw total of 1000) computed 100.1% and tripped the per-field
  // `percentage <= 100` check with a misleading error. Since the amount-mode
  // guard above rejects any plan whose integer sum exceeds `roundedTotal`, every
  // accepted plan has each amount <= sum(amounts) <= roundedTotal, so this
  // percentage is always <= 100.
  const outCategories: BudgetCategory[] = names.map((name, i) => {
    const amount = amounts[i]!;
    const percentage = Math.round((amount / roundedTotal) * 1000) / 10;
    return { name, amount, percentage };
  });

  const plan: BudgetPlan = {
    total: roundedTotal,
    categories: outCategories,
    ...(currency !== undefined ? { currency } : {}),
  };

  // DEFENSE IN DEPTH: cross-field over-allocation rule (6a strict schema) FIRST,
  // then per-field validation (6a plain schema). Strict runs first so that if a
  // plan somehow over-allocates it surfaces the DOMAIN over-allocation reason
  // rather than an incidental per-field message; the amount-mode guard above
  // should already have caught over-allocation, so this is defense in depth.
  const strict = budgetPlanStrictSchema.safeParse(plan);
  if (!strict.success) {
    // Reuse the domain validator for a precise reason.
    const reason = validateBudgetAllocation(plan);
    throw new Error(
      `split_budget: ${reason.ok ? "invalid budget plan" : reason.reason}`
    );
  }
  const perField = budgetPlanSchema.safeParse(plan);
  if (!perField.success) {
    const reason = perField.error.issues.map((iss) => iss.message).join("; ");
    throw new Error(`split_budget: produced an invalid budget plan: ${reason}`);
  }

  return perField.data;
}

// ---------------------------------------------------------------------------
// LangChain tool() wrappers (Zod input schemas)
// ---------------------------------------------------------------------------

// Input schema for the days_until tool. A single strict `date` field. We keep
// the format check on the schema (fast, model-facing message) AND in the pure
// function (calendar validity), so a bad date is rejected either way.
const daysUntilInputSchema = z.object({
  date: z
    .string()
    .regex(
      /^\d{4}-\d{2}-\d{2}$/,
      "date must be an ISO calendar date in YYYY-MM-DD format"
    )
    .describe("Target calendar date in YYYY-MM-DD format (e.g. 2026-12-12)."),
});

/**
 * LangChain tool: days_until.
 *
 * Computes the signed, UTC calendar-day difference from TODAY (the real clock)
 * to the given date. Read-only and side-effect-free. Returns a
 * `content_and_artifact` tuple: a short human summary plus the structured
 * {@link DaysUntilResult}.
 */
export const daysUntilTool = tool(
  ({ date }): [string, DaysUntilResult] => {
    const result = daysUntil(date);
    const summary =
      result.direction === "today"
        ? `${result.targetDate} is today.`
        : result.direction === "future"
          ? `${result.daysUntil} day(s) until ${result.targetDate}.`
          : `${result.targetDate} was ${Math.abs(result.daysUntil)} day(s) ago.`;
    return [summary, result];
  },
  {
    name: "days_until",
    description:
      "Return the number of whole calendar days from today until a given date " +
      "(YYYY-MM-DD). The count is signed: positive = future, negative = past, " +
      "0 = today. Uses UTC calendar days (time-of-day is ignored). Read-only; " +
      "no side effects.",
    schema: daysUntilInputSchema,
    responseFormat: "content_and_artifact",
  }
);

// Input schema for the split_budget tool. `total` is required; `categories` and
// `currency` are optional. Category fields mirror the pure function's contract.
const splitBudgetInputSchema = z.object({
  total: z
    .number()
    .positive("total must be > 0")
    .describe("Overall wedding budget to split. Must be a positive number."),
  categories: z
    .array(
      z.object({
        name: z
          .string()
          .min(1, "category name must not be empty")
          .describe("Category label, e.g. \"Venue\"."),
        percentage: z
          .number()
          .min(0)
          .max(100)
          .optional()
          .describe(
            "Share of the total for this category (0..100). Use percentage " +
              "mode OR amount mode consistently, not both."
          ),
        amount: z
          .number()
          .nonnegative()
          .optional()
          .describe(
            "Absolute amount for this category. Use amount mode OR percentage " +
              "mode consistently, not both."
          ),
      })
    )
    .optional()
    .describe(
      "Optional custom categories. Omit to use the default wedding split " +
        "(Venue 40, Catering 25, Photography 10, Attire 8, Flowers 7, Music 5, " +
        "Misc 5). If provided, use EITHER percentages OR amounts for all " +
        "categories."
    ),
  currency: z
    .string()
    .min(1)
    .optional()
    .describe("Optional currency label (e.g. \"USD\")."),
});

/**
 * LangChain tool: split_budget.
 *
 * Splits a total budget across categories (default set if none given) into whole
 * currency amounts that sum exactly to the total, enforcing the over-allocation
 * rule. Read-only and side-effect-free. Returns a `content_and_artifact` tuple:
 * a short human summary plus the structured {@link BudgetPlan}.
 */
export const splitBudgetTool = tool(
  ({ total, categories, currency }): [string, BudgetPlan] => {
    const plan = splitBudget(total, categories, currency);
    const summary =
      `Budget of ${plan.total}${plan.currency ? " " + plan.currency : ""} split ` +
      `across ${plan.categories.length} categories: ` +
      plan.categories.map((c) => `${c.name} ${c.amount}`).join(", ") +
      ".";
    return [summary, plan];
  },
  {
    name: "split_budget",
    description:
      "Split a total wedding budget into category amounts. Omit categories to " +
      "use the default split. If categories are given, provide EITHER " +
      "percentages OR amounts for all of them. Amounts are rounded to whole " +
      "currency units and always sum exactly to the total (or the sum of " +
      "provided amounts). Rejects over-allocation. Read-only; no side effects.",
    schema: splitBudgetInputSchema,
    responseFormat: "content_and_artifact",
  }
);

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/**
 * The SAFE, read-only wedding tools, for the 6c agent/tool-loop to consume. This
 * is intentionally the ONLY aggregated export of the tools; the individual pure
 * functions and tool objects are also exported above for direct/programmatic
 * use and testing.
 */
export const weddingTools = [daysUntilTool, splitBudgetTool] as const;
