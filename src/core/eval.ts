// Phase 4 (increment 4e) — wedding-planning evaluation: PURE logic.
//
// This module holds ONLY pure, deterministic, offline logic:
//   - the dataset item schema and the category vocabulary,
//   - a strict JSONL parser/validator for the versioned dataset,
//   - property-based scorers that, given a model response string and an item's
//     expectations, return a pass/fail verdict per property WITH a reason.
//
// It performs NO network calls, reads NO credentials, and does NO file I/O, so
// the offline Vitest suite imports it directly and exercises the scorers against
// synthetic responses. All live I/O (loading the dataset file, calling the real
// proxy, writing the dated results file) lives in src/run-eval.ts.
//
// Scoring philosophy: expectations are PROPERTY-based, never exact-string. A
// property check asks "does this response have the shape a good wedding-planner
// answer for this category would have?" (e.g. does a budget answer contain a
// numeric breakdown?). Checks are deliberately lenient/robust so a genuinely
// good answer passes, while an off-topic or empty answer fails — no LLM judge.

// ---------------------------------------------------------------------------
// Dataset schema
// ---------------------------------------------------------------------------

// Representative wedding-planning categories the dataset must span. The final
// entry is an explicitly OFF-TOPIC prompt the assistant must steer back from.
export const EVAL_CATEGORIES = [
  "budget",
  "timeline",
  "venue_theme",
  "guest_logistics",
  "cultural_ceremony",
  "vendor_selection",
  "off_topic",
] as const;

export type EvalCategory = (typeof EVAL_CATEGORIES)[number];

// The deterministic, property-based expectations for one item. Every field is
// optional and defaults to "not required"; a scorer only evaluates properties
// that are explicitly requested. All are checkable WITHOUT an LLM judge.
export interface EvalExpectations {
  // The response must contain at least one numeric monetary breakdown
  // (percentages or currency amounts) — used for budget prompts.
  expectsNumericBreakdown?: boolean;
  // The response must present a list or ordered steps (checklist/timeline).
  expectsList?: boolean;
  // The response must mention at least one term from `mustMention` (any one is
  // enough). Case-insensitive, matched on word-ish boundaries.
  mustMention?: string[];
  // The off-topic guardrail: the response must gently steer back to weddings
  // AND must not substantively answer the off-topic request.
  mustSteerBack?: boolean;
  // Minimum non-whitespace character length for a substantive answer. Defaults
  // to a small floor so an empty/near-empty response always fails.
  minLength?: number;
}

// One versioned dataset item. `id` is stable and unique; do not renumber.
export interface EvalItem {
  id: string;
  category: EvalCategory;
  prompt: string;
  expectations: EvalExpectations;
}

// The lowest length (non-whitespace chars) any substantive answer must clear.
// Applied to every item in addition to any per-item minLength.
export const GLOBAL_MIN_LENGTH = 40;

// ---------------------------------------------------------------------------
// Dataset parsing / validation (pure)
// ---------------------------------------------------------------------------

const CATEGORY_SET: ReadonlySet<string> = new Set(EVAL_CATEGORIES);

// Validate one already-JSON-parsed record into an EvalItem, throwing a precise
// error (with the record index) on the first problem. Kept strict so a
// malformed dataset fails loudly rather than silently under-testing.
function validateItem(raw: unknown, index: number): EvalItem {
  const where = `dataset item #${index}`;
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error(`${where}: expected a JSON object`);
  }
  const rec = raw as Record<string, unknown>;

  const id = rec["id"];
  if (typeof id !== "string" || id.trim() === "") {
    throw new Error(`${where}: "id" must be a non-empty string`);
  }
  const category = rec["category"];
  if (typeof category !== "string" || !CATEGORY_SET.has(category)) {
    throw new Error(
      `${where} (id=${id}): "category" must be one of ${EVAL_CATEGORIES.join(", ")}`
    );
  }
  const prompt = rec["prompt"];
  if (typeof prompt !== "string" || prompt.trim() === "") {
    throw new Error(`${where} (id=${id}): "prompt" must be a non-empty string`);
  }
  const expectationsRaw = rec["expectations"];
  if (
    typeof expectationsRaw !== "object" ||
    expectationsRaw === null ||
    Array.isArray(expectationsRaw)
  ) {
    throw new Error(`${where} (id=${id}): "expectations" must be an object`);
  }
  const expectations = validateExpectations(
    expectationsRaw as Record<string, unknown>,
    id
  );

  return { id, category: category as EvalCategory, prompt, expectations };
}

// Validate the expectations sub-object. Only known keys are honored; unknown
// keys are rejected so a typo (e.g. "expectsLists") cannot silently disable a
// check the author intended.
const KNOWN_EXPECTATION_KEYS: ReadonlySet<string> = new Set([
  "expectsNumericBreakdown",
  "expectsList",
  "mustMention",
  "mustSteerBack",
  "minLength",
]);

function validateExpectations(
  raw: Record<string, unknown>,
  id: string
): EvalExpectations {
  for (const key of Object.keys(raw)) {
    if (!KNOWN_EXPECTATION_KEYS.has(key)) {
      throw new Error(`item id=${id}: unknown expectation key "${key}"`);
    }
  }
  const out: EvalExpectations = {};
  const boolKeys = [
    "expectsNumericBreakdown",
    "expectsList",
    "mustSteerBack",
  ] as const;
  for (const key of boolKeys) {
    const v = raw[key];
    if (v !== undefined) {
      if (typeof v !== "boolean") {
        throw new Error(`item id=${id}: "${key}" must be a boolean`);
      }
      out[key] = v;
    }
  }
  if (raw["mustMention"] !== undefined) {
    const v = raw["mustMention"];
    if (
      !Array.isArray(v) ||
      v.length === 0 ||
      !v.every((s) => typeof s === "string" && s.trim() !== "")
    ) {
      throw new Error(
        `item id=${id}: "mustMention" must be a non-empty array of non-empty strings`
      );
    }
    out.mustMention = v as string[];
  }
  if (raw["minLength"] !== undefined) {
    const v = raw["minLength"];
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0) {
      throw new Error(`item id=${id}: "minLength" must be a non-negative number`);
    }
    out.minLength = v;
  }
  return out;
}

// Parse the dataset from raw JSONL text (one JSON object per non-blank line).
// Validates every item, rejects duplicate ids, and returns the ordered items.
// Pure: the caller reads the file and hands the text in.
export function parseDataset(jsonl: string): EvalItem[] {
  const lines = jsonl.split(/\r?\n/);
  const items: EvalItem[] = [];
  const seen = new Set<string>();
  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (trimmed === "") return; // skip blank lines
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch (err) {
      throw new Error(
        `dataset line ${i + 1}: invalid JSON (${
          err instanceof Error ? err.message : String(err)
        })`
      );
    }
    const item = validateItem(parsed, items.length);
    if (seen.has(item.id)) {
      throw new Error(`duplicate dataset id: ${item.id}`);
    }
    seen.add(item.id);
    items.push(item);
  });
  if (items.length === 0) {
    throw new Error("dataset is empty");
  }
  return items;
}

// ---------------------------------------------------------------------------
// Property scorers (pure)
// ---------------------------------------------------------------------------

// The result of checking ONE property against a response.
export interface PropertyResult {
  // A stable machine name for the property (e.g. "numericBreakdown").
  property: string;
  passed: boolean;
  // Human-readable reason (why it passed or failed). Never contains secrets.
  reason: string;
}

// The full score for one item: the item id/category and every property checked.
export interface ItemScore {
  id: string;
  category: EvalCategory;
  properties: PropertyResult[];
  // Convenience: true only if EVERY checked property passed.
  passed: boolean;
}

// --- Individual property detectors (small, testable, lenient) ---------------

// Collapse to lowercase for case-insensitive matching.
function lower(text: string): string {
  return text.toLowerCase();
}

// Non-whitespace length of a response (used for the length floor).
export function nonWhitespaceLength(text: string): number {
  return text.replace(/\s+/g, "").length;
}

// Does the text contain a numeric monetary breakdown? Accepts either a
// percentage (e.g. "40%") or a currency amount (e.g. "$5,000", "5000 USD",
// "£1,200"). Lenient by design: any ONE such token satisfies the property.
const PERCENT_PATTERN = /\b\d{1,3}(?:\.\d+)?\s?%/;
const CURRENCY_SYMBOL_PATTERN = /[$£€₹]\s?\d[\d,]*(?:\.\d+)?/;
const CURRENCY_WORD_PATTERN =
  /\b\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp|inr|dollars?|euros?|pounds?|rupees?)\b/i;

export function hasNumericBreakdown(text: string): boolean {
  return (
    PERCENT_PATTERN.test(text) ||
    CURRENCY_SYMBOL_PATTERN.test(text) ||
    CURRENCY_WORD_PATTERN.test(text)
  );
}

// Does the text present a list or ordered steps? Accepts markdown bullets
// (-, *, +), numbered lines (1. / 1)), or several newline-separated short
// segments. Requires at least TWO list-like markers so a single dash in prose
// does not count.
export function hasListOrSteps(text: string): boolean {
  const lines = text.split(/\r?\n/);
  let bulletCount = 0;
  let numberedCount = 0;
  for (const line of lines) {
    const t = line.trim();
    if (/^[-*+]\s+\S/.test(t)) bulletCount += 1;
    if (/^\d+[.)]\s+\S/.test(t)) numberedCount += 1;
  }
  if (bulletCount >= 2 || numberedCount >= 2) return true;
  // Fall back to inline enumerations like "1) ... 2) ..." on one line.
  const inlineNumbered = (text.match(/\b\d+[.)]\s+\S/g) ?? []).length;
  return inlineNumbered >= 2;
}

// Does the text mention any of the given terms (case-insensitive, substring on
// word-ish boundaries)? Any single hit satisfies the property.
export function mentionsAny(text: string, terms: string[]): boolean {
  const hay = lower(text);
  return terms.some((term) => hay.includes(lower(term.trim())));
}

// Normalize a response for steer-back matching: lowercase, fold curly quotes to
// straight, and collapse runs of whitespace to a single space. This makes the
// anchored phrase patterns tolerant of apostrophe/quote style and spacing
// without loosening WHAT they match.
function normalizeForSteerBack(text: string): string {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019\u201B\u2032]/g, "'") // curly/prime single quotes -> '
    .replace(/[\u2013\u2014]/g, "-") // en/em dash -> hyphen (spacing tolerance)
    .replace(/\s+/g, " ")
    .trim();
}

// Anchored steer-back phrases. The defining property of a REAL steer-back is
// that it names the WEDDING DOMAIN as the thing to return to — not merely any
// domain noun (venue/guest) next to a generic preposition ("focus on"). So
// every pattern requires the word "wedding(s)" (optionally "wedding planning")
// tied to a redirect/refusal intent. `wp` matches "wedding" or "weddings"; the
// apostrophe class tolerates "i'm"/"im"; `\s+` tolerates variable spacing (text
// is pre-collapsed by normalizeForSteerBack).
const WP = "weddings?"; // wedding | weddings
const AP = "['\u2019]?"; // optional apostrophe (straight already folded)
const STEER_BACK_PATTERNS: RegExp[] = [
  // "back to your/the wedding [planning]", "get back to ... wedding"
  new RegExp(`back to (?:your |the |our )?(?:${WP} (?:planning|day)|${WP})`),
  new RegExp(`back to (?:your |the |our )?${WP} planning`),
  // "focus on your/the wedding [planning]"
  new RegExp(`focus on (?:your |the |our )?${WP}`),
  // "stick to wedding planning" / "stick to weddings"
  new RegExp(`stick to (?:${WP} planning|${WP})`),
  // "help you plan your wedding", "help (you) with your wedding [planning]"
  new RegExp(`help (?:you )?(?:plan|with)(?: your| the| our)? ${WP}`),
  new RegExp(`help you plan (?:your |the |our )?${WP}`),
  // "i'm/i am your wedding planning assistant/planner"
  new RegExp(`i${AP}m (?:your |a |the )?${WP} (?:planning )?(?:assistant|planner)`),
  new RegExp(`i am (?:your |a |the )?${WP} (?:planning )?(?:assistant|planner)`),
  // "i'm here to help with (your) wedding [planning]"
  new RegExp(`here to help with (?:your |the |our )?${WP}`),
  // "your wedding planning" as a direct redirect object
  new RegExp(`(?:your|the|our) ${WP} planning`),
  // "let's get back to the wedding", "lets get back to weddings"
  new RegExp(`get back to (?:your |the |our )?${WP}`),
];

// Does the response steer an off-topic request back to wedding planning? True
// only when the response contains an explicit, WEDDING-ANCHORED redirect/refusal
// phrase (see STEER_BACK_PATTERNS) — i.e. it names the wedding domain as the
// thing to return to. A genuinely off-topic answer that merely happens to use a
// domain noun (venue/guest) alongside a generic preposition ("focus on") no
// longer counts, closing finding R1's false positive. Pure and deterministic;
// no LLM judge.
export function steersBackToWeddings(text: string): boolean {
  const normalized = normalizeForSteerBack(text);
  return STEER_BACK_PATTERNS.some((pattern) => pattern.test(normalized));
}

// --- Scoring orchestration --------------------------------------------------

// Score a single response string against one item's expectations. Deterministic
// and offline. Returns a per-property breakdown; the item passes only if every
// checked property passes.
export function scoreItem(item: EvalItem, response: string): ItemScore {
  const properties: PropertyResult[] = [];
  const exp = item.expectations;
  const text = response ?? "";

  // Length floor is ALWAYS checked (global floor, plus any per-item override).
  const minLen = Math.max(GLOBAL_MIN_LENGTH, exp.minLength ?? 0);
  const len = nonWhitespaceLength(text);
  properties.push({
    property: "minLength",
    passed: len >= minLen,
    reason:
      len >= minLen
        ? `response has ${len} non-whitespace chars (>= ${minLen})`
        : `response too short: ${len} non-whitespace chars (< ${minLen})`,
  });

  if (exp.expectsNumericBreakdown) {
    const ok = hasNumericBreakdown(text);
    properties.push({
      property: "numericBreakdown",
      passed: ok,
      reason: ok
        ? "contains a numeric monetary breakdown (percentage or amount)"
        : "no numeric monetary breakdown (percentage or currency amount) found",
    });
  }

  if (exp.expectsList) {
    const ok = hasListOrSteps(text);
    properties.push({
      property: "listOrSteps",
      passed: ok,
      reason: ok
        ? "presents a list or ordered steps"
        : "no list or ordered steps detected (expected bullets or numbered items)",
    });
  }

  if (exp.mustMention && exp.mustMention.length > 0) {
    const ok = mentionsAny(text, exp.mustMention);
    properties.push({
      property: "mustMention",
      passed: ok,
      reason: ok
        ? `mentions at least one expected term (${exp.mustMention.join(", ")})`
        : `mentions none of the expected terms (${exp.mustMention.join(", ")})`,
    });
  }

  if (exp.mustSteerBack) {
    const ok = steersBackToWeddings(text);
    properties.push({
      property: "steerBack",
      passed: ok,
      reason: ok
        ? "steers the off-topic request back to wedding planning"
        : "did not steer the off-topic request back to wedding planning",
    });
  }

  const passed = properties.every((p) => p.passed);
  return { id: item.id, category: item.category, properties, passed };
}

// ---------------------------------------------------------------------------
// Aggregation (pure)
// ---------------------------------------------------------------------------

export interface CategoryAggregate {
  category: EvalCategory;
  passed: number;
  total: number;
}

export interface EvalAggregate {
  itemsPassed: number;
  itemsTotal: number;
  // 0..1 fraction of fully-passing items.
  itemScore: number;
  // Individual property pass rate across ALL checked properties (0..1).
  propertyScore: number;
  propertiesPassed: number;
  propertiesTotal: number;
  byCategory: CategoryAggregate[];
}

// Aggregate per-item scores into overall and per-category summaries. Pure and
// deterministic; category order follows EVAL_CATEGORIES.
export function aggregateScores(scores: ItemScore[]): EvalAggregate {
  const itemsTotal = scores.length;
  const itemsPassed = scores.filter((s) => s.passed).length;
  let propertiesPassed = 0;
  let propertiesTotal = 0;
  for (const s of scores) {
    for (const p of s.properties) {
      propertiesTotal += 1;
      if (p.passed) propertiesPassed += 1;
    }
  }

  const byCategory: CategoryAggregate[] = EVAL_CATEGORIES.map((category) => {
    const inCat = scores.filter((s) => s.category === category);
    return {
      category,
      passed: inCat.filter((s) => s.passed).length,
      total: inCat.length,
    };
  }).filter((agg) => agg.total > 0);

  return {
    itemsPassed,
    itemsTotal,
    itemScore: itemsTotal === 0 ? 0 : itemsPassed / itemsTotal,
    propertyScore: propertiesTotal === 0 ? 0 : propertiesPassed / propertiesTotal,
    propertiesPassed,
    propertiesTotal,
    byCategory,
  };
}

// ---------------------------------------------------------------------------
// Results rendering (pure)
// ---------------------------------------------------------------------------

// One item's live outcome: its score plus optional live-run metadata (latency,
// or a redacted error reason when the call failed). Kept separate from ItemScore
// so the pure scorer stays I/O-free.
export interface ItemRunResult {
  score: ItemScore;
  prompt: string;
  latencyMs?: number;
  // Present only when the live call errored (already redacted by the caller).
  errorReason?: string;
}

export interface EvalRunMeta {
  runTimestampUtc: string;
  model: string;
  baseUrlHost: string;
  maskedKey: string;
}

function pct(fraction: number): string {
  return `${(fraction * 100).toFixed(0)}%`;
}

function mdCell(value: string): string {
  return value.replace(/\|/g, "\\|");
}

// Render the dated Markdown results file body. Contains model output shape
// verdicts (pass/fail + reasons), NOT raw secrets; the caller redacts any error
// text before it reaches here. Deterministic given the same inputs.
export function renderEvalMarkdown(
  meta: EvalRunMeta,
  results: ItemRunResult[]
): string {
  const scores = results.map((r) => r.score);
  const agg = aggregateScores(scores);
  const lines: string[] = [];

  lines.push("# Wedding Planner Evaluation Baseline");
  lines.push("");
  lines.push(`- **Run (UTC):** ${meta.runTimestampUtc}`);
  lines.push(`- **Model:** ${meta.model}`);
  lines.push(`- **Base URL host:** ${meta.baseUrlHost}`);
  lines.push(`- **API key:** ${meta.maskedKey} (masked)`);
  lines.push("");

  lines.push("## Aggregate");
  lines.push("");
  lines.push(
    `- **Items passed:** ${agg.itemsPassed} / ${agg.itemsTotal} (${pct(agg.itemScore)})`
  );
  lines.push(
    `- **Properties passed:** ${agg.propertiesPassed} / ${agg.propertiesTotal} (${pct(agg.propertyScore)})`
  );
  lines.push("");

  lines.push("## Per-category summary");
  lines.push("");
  lines.push("| Category | Items passed | Total |");
  lines.push("| --- | --- | --- |");
  for (const c of agg.byCategory) {
    lines.push(`| ${c.category} | ${c.passed} | ${c.total} |`);
  }
  lines.push("");

  lines.push("## Per-item results");
  lines.push("");
  lines.push("| ID | Category | Result | Latency (ms) | Notes |");
  lines.push("| --- | --- | --- | --- | --- |");
  for (const r of results) {
    const result = r.errorReason
      ? "ERROR"
      : r.score.passed
        ? "PASS"
        : "FAIL";
    const latency = r.latencyMs !== undefined ? String(r.latencyMs) : "-";
    const failed = r.score.properties.filter((p) => !p.passed);
    const notes = r.errorReason
      ? `error: ${r.errorReason}`
      : failed.length === 0
        ? "all properties passed"
        : failed.map((p) => `${p.property}: ${p.reason}`).join("; ");
    lines.push(
      `| ${mdCell(r.score.id)} | ${r.score.category} | ${result} | ${latency} | ${mdCell(
        notes
      )} |`
    );
  }
  lines.push("");

  lines.push("## Scoring method");
  lines.push("");
  lines.push(
    "Scores are produced by deterministic, property-based checks in " +
      "`src/core/eval.ts` (no LLM judge). Each item declares checkable " +
      "expectations (e.g. a numeric budget breakdown, a list/steps, a required " +
      "mention, or an off-topic steer-back) and the same pure scorers grade both " +
      "this live run and the offline test suite."
  );
  lines.push("");

  return lines.join("\n");
}

// Render a concise console summary of a live run. Deterministic.
export function renderEvalConsoleSummary(results: ItemRunResult[]): string {
  const agg = aggregateScores(results.map((r) => r.score));
  const lines: string[] = [];
  lines.push(
    `Items passed: ${agg.itemsPassed}/${agg.itemsTotal} (${pct(agg.itemScore)})`
  );
  lines.push(
    `Properties passed: ${agg.propertiesPassed}/${agg.propertiesTotal} (${pct(
      agg.propertyScore
    )})`
  );
  for (const c of agg.byCategory) {
    lines.push(`  ${c.category}: ${c.passed}/${c.total}`);
  }
  const errored = results.filter((r) => r.errorReason);
  if (errored.length > 0) {
    lines.push(`Errored items: ${errored.map((r) => r.score.id).join(", ")}`);
  }
  return lines.join("\n");
}
