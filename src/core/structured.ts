import { HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import type { z } from "zod";
import { createChatModel, type ModelOptions } from "./model.js";
import { redactError, redactText } from "./redaction.js";
import {
  budgetPlanSchema,
  planningChecklistSchema,
  type BudgetPlan,
  type PlanningChecklist,
} from "./schemas.js";

// Phase 6 (increment 6a): structured-output plumbing.
//
// This module turns an LLM turn into a VALIDATED, typed wedding-domain artifact
// (see src/core/schemas.ts). It is the structured-output FOUNDATION only —
// tools (6b) and the agent/tool-loop (6c) are out of scope here, and this is
// intentionally NOT yet wired into the HTTP service or CLI.
//
// Two invariants are enforced here:
//   1. Model construction goes through createChatModel() (the 4b single-factory
//      rule) — this module NEVER constructs ChatOpenAI directly.
//   2. The opus temperature-omit carry-forward constraint is honored: for
//      claude-opus-4-8, the model is built with `temperature: null` so the
//      field is OMITTED from the provider payload (opus deprecates an explicit
//      temperature on its structured-output path and errors when one is sent).
//      Other models (default claude-sonnet-4-6) are built normally.

// Default structured-output model. claude-sonnet-4-6 is fully "Supported" for
// structured output (docs/capabilities/2026-07-28.md) and works whether or not a
// temperature is sent, so it is the safe default when a caller does not pick a
// model. (Deliberately a fixed constant rather than config.model so structured
// output has a stable, documented default regardless of the chat default.)
export const DEFAULT_STRUCTURED_MODEL = "claude-sonnet-4-6";

// Options accepted by the structured-output helpers.
export interface StructuredOptions {
  // Model id to use. Defaults to DEFAULT_STRUCTURED_MODEL.
  model?: string;
  // Optional system prompt prepended when `input` is a plain string. Ignored
  // when `input` is already a BaseMessage[] (the caller owns the full prompt).
  systemPrompt?: string;
}

// True when `model` is one whose structured-output path DEPRECATES an explicit
// temperature and must be built with the temperature field omitted. Currently
// this is the claude-opus family (claude-opus-4-8). Match is case-insensitive
// and substring-based so alias variants ("claude-opus-4-8", future opus tags)
// are all covered without an exhaustive allow-list.
export function isTemperatureOmitModel(model: string): boolean {
  return model.toLowerCase().includes("opus");
}

// PURE decision function (offline-unit-testable) mapping a model id to the exact
// ModelOptions used to build the structured-output model. This is where the
// opus temperature-omit constraint lives:
//   - claude-opus-4-8  -> { model, temperature: null }  (temperature OMITTED)
//   - anything else     -> { model }                     (factory default temp)
// Returning `temperature: null` (rather than a number) is what makes
// createChatModel omit the field entirely — see ModelOptions in model.ts.
export function decideStructuredModelOptions(model: string): ModelOptions {
  if (isTemperatureOmitModel(model)) {
    return { model, temperature: null };
  }
  return { model };
}

// Normalize helper input into the message array handed to the model. A plain
// string becomes an optional system message + a human message; a caller who
// passes a BaseMessage[] owns the whole prompt and it is used verbatim.
//
// Fast-fails on effectively-empty input (whitespace-only string, or an empty
// message array) with a clear, NON-LEAKING error before the model is ever
// called — an empty prompt cannot yield a useful structured artifact and just
// wastes a round trip. The message carries no user content, so nothing to leak.
function toMessages(
  input: string | BaseMessage[],
  systemPrompt?: string
): BaseMessage[] {
  if (Array.isArray(input)) {
    if (input.length === 0) {
      throw new Error(
        "Structured generation requires at least one input message"
      );
    }
    return input;
  }
  if (input.trim() === "") {
    throw new Error("Structured generation requires non-empty input");
  }
  const messages: BaseMessage[] = [];
  if (systemPrompt) messages.push(new SystemMessage(systemPrompt));
  messages.push(new HumanMessage(input));
  return messages;
}

// Shared message for the REFUSAL / no-output failure path (see generateStructured
// contract). A fixed, content-free template, so it is inherently redaction-safe.
const REFUSAL_MESSAGE =
  "Structured generation refused or returned no structured output";

// True for a model response that is a refusal / no-output REGARDLESS of the
// target schema, so it is classified as refusal BEFORE schema validation runs:
//   - null / undefined                 (no structured output came back at all)
//   - an OpenAI-style refusal payload   ({ refusal: <truthy> } — the model
//                                        explicitly declined)
// These can never be a legitimately-valid structured result. A non-object
// primitive (string/number/etc.) is NOT a refusal here; it falls through to
// schema validation, which rejects it with a precise per-field reason. An empty
// object ({}) is handled SEPARATELY (see isEmptyObject): it is only a refusal
// when it also fails validation, so a schema that legitimately permits {} is not
// misclassified.
function isUnconditionalRefusal(raw: unknown): boolean {
  if (raw === null || raw === undefined) return true;
  if (typeof raw !== "object") return false;
  const obj = raw as Record<string, unknown>;
  return (
    "refusal" in obj &&
    obj["refusal"] !== undefined &&
    obj["refusal"] !== null &&
    obj["refusal"] !== false
  );
}

// True for a non-null object with ZERO own enumerable keys ({}). An empty object
// is treated as a refusal / no-output ONLY when it also FAILS schema validation
// (nothing was populated to validate); a schema that legitimately permits {}
// PASSES validation first and returns the empty object. See generateStructured.
function isEmptyObject(raw: unknown): boolean {
  return (
    typeof raw === "object" &&
    raw !== null &&
    Object.keys(raw as Record<string, unknown>).length === 0
  );
}

// Generate a VALIDATED, typed artifact from the LLM for an arbitrary Zod schema.
//
// Contract:
//   - The model is built via createChatModel(decideStructuredModelOptions(model))
//     so the opus temperature-omit rule is honored automatically.
//   - `model.withStructuredOutput(schema)` requests provider-native structured
//     output where supported; LangChain falls back to tool-based structuring as
//     needed. LangChain returns an object it already coerced toward the schema.
//   - DEFENSE IN DEPTH: the returned object is STILL re-validated with an
//     explicit `schema.safeParse`, so the value this function returns is
//     guaranteed to satisfy the schema at runtime AND is correctly typed as
//     z.infer<schema>. This guards against any provider/library drift.
//   - THREE DISTINCT, NAMED failure paths, each with a clear, redacted message:
//       1. TRANSPORT error   — the invoke() call itself threw (network/auth/
//          provider). Message: "Structured generation failed: …".
//       2. REFUSAL / no output — the model declined or returned nothing usable:
//          null/undefined, an OpenAI-style {refusal} payload, OR an empty object
//          {} THAT IS NOT SCHEMA-VALID. This is a first-class case, DISTINCT from
//          a schema failure. Message: "Structured generation refused or returned
//          no structured output". NOTE the precondition: an empty object is only
//          treated as refusal when it FAILS validation — a schema that
//          legitimately permits {} (e.g. all-optional fields) validates first and
//          the empty object is returned as the typed result.
//       3. SCHEMA-VALIDATION failure — a NON-EMPTY object came back but does not
//          satisfy the schema. Message: "Structured output failed schema
//          validation: …".
//     All three pass through the shared redaction layer (src/core/redaction.ts)
//     so no secret/PII can leak into logs or clients.
export async function generateStructured<T>(
  schema: z.ZodType<T>,
  input: string | BaseMessage[],
  options: StructuredOptions = {}
): Promise<T> {
  // Fast-fail on empty input BEFORE constructing/invoking the model.
  const messages = toMessages(input, options.systemPrompt);
  const modelId = options.model ?? DEFAULT_STRUCTURED_MODEL;
  const model = createChatModel(decideStructuredModelOptions(modelId));
  const structuredModel = model.withStructuredOutput(schema);

  let raw: unknown;
  try {
    raw = await structuredModel.invoke(messages);
  } catch (err) {
    // Redact before surfacing — the provider error may echo the endpoint/key.
    throw new Error(
      `Structured generation failed: ${redactError(err)}`
    );
  }

  // REFUSAL / no-output that is independent of the schema (null/undefined or an
  // OpenAI-style {refusal} payload) is checked BEFORE schema validation: there
  // is no candidate object to validate, so a "schema validation" message would
  // be misleading.
  if (isUnconditionalRefusal(raw)) {
    throw new Error(REFUSAL_MESSAGE);
  }

  const parsed = schema.safeParse(raw);
  if (!parsed.success) {
    // An EMPTY object that fails validation is a refusal / no-output, NOT a
    // schema-validation failure: nothing was populated to validate. (A schema
    // that legitimately permits {} would have PASSED above and returned it, so
    // this branch is only reached for a schema that actually requires fields.)
    if (isEmptyObject(raw)) {
      throw new Error(REFUSAL_MESSAGE);
    }
    // In zod v4 `error.message` is a verbose JSON-serialized issue array; build
    // a concise reason from the per-issue messages instead. Redact it before it
    // reaches any log/client — issue messages can echo offending model values
    // (potentially PII).
    const reason = parsed.error.issues.map((i) => i.message).join("; ");
    throw new Error(
      `Structured output failed schema validation: ${redactText(reason)}`
    );
  }
  return parsed.data;
}

// Domain convenience wrapper: produce a validated wedding BudgetPlan.
export function generateBudgetPlan(
  input: string | BaseMessage[],
  options?: StructuredOptions
): Promise<BudgetPlan> {
  return generateStructured(budgetPlanSchema, input, options);
}

// Domain convenience wrapper: produce a validated wedding PlanningChecklist.
export function generatePlanningChecklist(
  input: string | BaseMessage[],
  options?: StructuredOptions
): Promise<PlanningChecklist> {
  return generateStructured(planningChecklistSchema, input, options);
}
