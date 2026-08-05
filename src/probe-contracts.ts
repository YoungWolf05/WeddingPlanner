import { isAIMessage } from "@langchain/core/messages";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { config } from "./config.js";
import { createChatModel } from "./core/model.js";
import { redactError as redactErrorShared } from "./core/redaction.js";
import { weddingTools } from "./core/tools.js";
import { decideAgentModelOptions } from "./core/agent.js";
import { generateBudgetPlan } from "./core/structured.js";
import { budgetPlanSchema } from "./core/schemas.js";
import {
  CONTRACT_CAPABILITIES,
  PERMITTED_TOOL_NAMES,
  classifyStructuredOutcome,
  classifyToolCallOutcome,
  renderContractConsoleTable,
  renderContractMarkdown,
  type AliasContractResults,
  type ContractMatrix,
} from "./core/contracts.js";
import type { ProbeResult } from "./core/capabilities.js";

// Phase 6 (increment 6d) — LIVE, opt-in TOOL-CALL + STRUCTURED-OUTPUT contract
// probe. Directly targets Phase 6 exit criterion (4): "every enabled model alias
// passes the required tool-call and structured-output contract tests".
//
// Makes live, credentialed, possibly billable calls — this is NOT part of
// `npm test`, `npm run typecheck`, `npm run build`, or CI. Run explicitly:
//   npm run test:contracts
//
// The pure classification/rendering logic lives in src/core/contracts.ts and is
// unit-tested offline. This file owns only the live I/O + redaction, mirroring
// the src/probe-capabilities.ts pattern:
//   - resilient per-alias / per-capability (one failure never aborts the run),
//   - a per-probe timeout so a hung call cannot stall the run,
//   - redaction host-only + masked key; every error reason is redacted.

// The enabled aliases = the current ALLOWED set (see src/core/repl.ts
// ALLOWED_MODELS). Kept as a local literal so the probe has no import cycle with
// the REPL, but it is the SAME set by construction (asserted in AGENTS.md).
const CHAT_ALIASES = ["claude-opus-4-8", "claude-sonnet-4-6"] as const;

// Per-probe timeout so a single hung capability cannot stall the whole run.
const PROBE_TIMEOUT_MS = 30_000;

// --- Redaction (host-only base URL + masked key; mirrors probe-capabilities) --

function baseUrlHost(rawBaseUrl: string): string {
  try {
    return new URL(rawBaseUrl).host;
  } catch {
    return rawBaseUrl.replace(/^[a-z]+:\/\//i, "").split("/")[0] ?? "(unknown)";
  }
}

function maskKey(rawKey: string): string {
  if (rawKey.startsWith("sk-")) return "sk-…(redacted)";
  return "…(redacted)";
}

// Pin the probe's redaction cap to the same 200-char bound as the capability
// probe so evidence lines stay bounded.
const PROBE_REDACT_MAX = 200;
function redactError(err: unknown): string {
  return redactErrorShared(err, PROBE_REDACT_MAX);
}

// --- Timeout wrapper (mirrors probe-capabilities) ---------------------------

class ProbeTimeoutError extends Error {
  constructor(ms: number) {
    super(`probe timed out after ${ms}ms`);
    this.name = "ProbeTimeoutError";
  }
}

async function withTimeout<T>(
  ms: number,
  fn: (signal: AbortSignal) => Promise<T>
): Promise<T> {
  const controller = new AbortController();
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new ProbeTimeoutError(ms));
    }, ms);
  });
  try {
    return await Promise.race([fn(controller.signal), timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// Heuristic (shared spirit with probe-capabilities): does an error indicate the
// FEATURE is unsupported (a definitive "no") rather than an incidental/transient
// failure? Incidental parameter errors (temperature/max_tokens/etc.) are NOT
// feature rejections and stay classified as Error so the redacted note carries
// the real reason for a human.
function looksUnsupported(err: unknown): boolean {
  const msg = (err instanceof Error ? err.message : String(err)).toLowerCase();
  const incidental =
    msg.includes("temperature") ||
    msg.includes("max_tokens") ||
    msg.includes("top_p") ||
    msg.includes("deprecated") ||
    msg.includes("invalid model name") ||
    msg.includes("model not found");
  if (incidental) return false;
  return (
    msg.includes("not support") ||
    msg.includes("does not support") ||
    msg.includes("unsupported") ||
    msg.includes("not available") ||
    msg.includes("no endpoints") ||
    msg.includes("function calling is not") ||
    (msg.includes("tool") && msg.includes("not support")) ||
    (msg.includes("response_format") && msg.includes("not")) ||
    (msg.includes("json") && msg.includes("not support"))
  );
}

// --- Contract probes --------------------------------------------------------
//
// Each probe returns a ProbeResult and NEVER throws: it catches its own errors,
// distinguishes a definitive Unsupported from an Error, and records a redacted
// note.

const PERMITTED = new Set<string>(PERMITTED_TOOL_NAMES);

// Tool-call contract: bind the SAFE weddingTools and prompt for something that
// should trigger `days_until`. Assert a well-formed tool call for a PERMITTED
// tool with parseable args. Opus is built with temperature omitted via the
// shared decideAgentModelOptions (Phase 4 carry-forward), so an incidental
// temperature deprecation cannot masquerade as a tool-calling failure.
async function probeToolCall(alias: string): Promise<ProbeResult> {
  try {
    const model = createChatModel(decideAgentModelOptions(alias));
    const bound = model.bindTools([...weddingTools]);
    const res = await withTimeout(PROBE_TIMEOUT_MS, (signal) =>
      bound.invoke(
        "How many days until 2026-12-12? Use the days_until tool.",
        { signal }
      )
    );
    const toolCalls = isAIMessage(res) ? res.tool_calls ?? [] : [];
    if (toolCalls.length === 0) {
      return classifyToolCallOutcome({
        errored: false,
        errorLooksUnsupported: false,
        hasToolCall: false,
        argsParseable: false,
        permitted: false,
      });
    }
    const call = toolCalls[0]!;
    const argsParseable =
      typeof call.args === "object" && call.args !== null;
    return classifyToolCallOutcome({
      errored: false,
      errorLooksUnsupported: false,
      hasToolCall: true,
      toolName: call.name,
      argsParseable,
      permitted: PERMITTED.has(call.name),
    });
  } catch (err) {
    const result = classifyToolCallOutcome({
      errored: true,
      errorLooksUnsupported: looksUnsupported(err),
      hasToolCall: false,
      argsParseable: false,
      permitted: false,
    });
    return { ...result, note: redactError(err) };
  }
}

// Structured-output contract: ask for a BudgetPlan via generateBudgetPlan (which
// builds the model via decideStructuredModelOptions -> opus temperature omitted)
// and confirm a schema-valid plan. generateBudgetPlan already re-validates with
// safeParse, so reaching a resolved value means the contract held; its distinct
// "failed schema validation" message is mapped to Degraded.
async function probeStructuredOutput(alias: string): Promise<ProbeResult> {
  try {
    const plan = await withTimeout(PROBE_TIMEOUT_MS, async () =>
      generateBudgetPlan(
        "Create a wedding budget plan totalling 30000 USD split across " +
          "Venue, Catering, Photography, and Flowers. Return category amounts.",
        { model: alias }
      )
    );
    // Defense in depth: independently confirm the returned plan is schema-valid.
    const schemaValid = budgetPlanSchema.safeParse(plan).success;
    return classifyStructuredOutcome({
      errored: false,
      errorLooksUnsupported: false,
      schemaValid,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    // generateStructured throws a DISTINCT "failed schema validation" message
    // when a well-formed-but-invalid object came back: that is Degraded, not a
    // transport/feature Error.
    if (/failed schema validation/i.test(msg)) {
      const result = classifyStructuredOutcome({
        errored: false,
        errorLooksUnsupported: false,
        schemaValid: false,
      });
      return { ...result, note: redactError(err) };
    }
    const result = classifyStructuredOutcome({
      errored: true,
      errorLooksUnsupported: looksUnsupported(err),
      schemaValid: false,
    });
    return { ...result, note: redactError(err) };
  }
}

// Probe both contracts for one alias. Resilient: a failing capability is
// recorded, never thrown, so the rest of the alias and run continue.
async function probeAlias(alias: string): Promise<AliasContractResults> {
  process.stderr.write(`  probing ${alias} ...\n`);
  return {
    toolCall: await probeToolCall(alias),
    structuredOutput: await probeStructuredOutput(alias),
  };
}

// --- Orchestration ----------------------------------------------------------

async function main(): Promise<void> {
  const runTimestampUtc = new Date().toISOString();
  const host = baseUrlHost(config.baseURL);
  const masked = maskKey(config.apiKey);

  process.stderr.write("LiteLLM contract probe (LIVE) — opt-in\n");
  process.stderr.write(`Base URL host : ${host}\n`);
  process.stderr.write(`API key       : ${masked} (masked)\n`);
  process.stderr.write(`Run (UTC)     : ${runTimestampUtc}\n\n`);

  const aliases: Record<string, AliasContractResults> = {};
  for (const alias of CHAT_ALIASES) {
    try {
      aliases[alias] = await probeAlias(alias);
    } catch (err) {
      // Defensive: probeAlias should never throw, but if it does, record every
      // capability for this alias as an Error rather than aborting the run.
      const reason = redactError(err);
      aliases[alias] = Object.fromEntries(
        CONTRACT_CAPABILITIES.map((cap) => [
          cap,
          { state: "Error", note: reason } as ProbeResult,
        ])
      ) as AliasContractResults;
    }
  }

  const matrix: ContractMatrix = {
    runTimestampUtc,
    baseUrlHost: host,
    maskedKey: masked,
    aliases,
  };

  process.stdout.write("\n" + renderContractConsoleTable(matrix) + "\n\n");

  const testDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(testDir, "..");
  const outDir = path.join(repoRoot, "docs", "contracts");
  await mkdir(outDir, { recursive: true });
  const dateStamp = runTimestampUtc.slice(0, 10); // YYYY-MM-DD (UTC)
  const outFile = path.join(outDir, `${dateStamp}.md`);
  await writeFile(outFile, renderContractMarkdown(matrix) + "\n", "utf8");

  process.stdout.write(`Evidence written: ${path.relative(repoRoot, outFile)}\n`);
}

main().catch((err) => {
  // Top-level guard: resilient run, but never leak a raw secret/PII. Route the
  // reason through the shared redaction (same 200-char probe cap).
  process.stderr.write("\nContract probe FAILED unexpectedly.\n");
  process.stderr.write(redactError(err) + "\n");
  process.exit(1);
});
