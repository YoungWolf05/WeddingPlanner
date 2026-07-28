import { readFile, mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { config } from "./config.js";
import { createChatModel } from "./core/model.js";
import { WEDDING_PLANNER_SYSTEM_PROMPT } from "./core/prompts.js";
import { redactError } from "./core/redaction.js";
import {
  parseDataset,
  scoreItem,
  renderEvalMarkdown,
  renderEvalConsoleSummary,
  type EvalItem,
  type EvalRunMeta,
  type ItemRunResult,
} from "./core/eval.js";

// Phase 4 (increment 4e) — wedding-planning evaluation runner (LIVE, opt-in).
//
// Runs the committed dataset (evals/dataset.jsonl) against the REAL LiteLLM
// proxy via createChatModel (default model), scores every response with the SAME
// pure scorers the offline suite uses (src/core/eval.ts), writes a dated results
// file under docs/eval/, and prints a console summary.
//
// This makes live, credentialed, possibly billable calls — it is NOT part of
// `npm test`, `npm run typecheck`, `npm run build`, or CI. Run explicitly:
//   npm run eval
//
// Resilience: one failed/slow item never aborts the run. Each item has a hard
// per-item timeout and its own try/catch; a failure is recorded as an ERROR row
// with a REDACTED reason. Only src/core/model.ts constructs the provider client
// (4b guard); this script uses createChatModel().

// Per-item hard timeout so a single hung call cannot stall the whole run.
const ITEM_TIMEOUT_MS = 45_000;

// --- Redaction helpers (mirror the capability probe) ------------------------

// Host-only base URL for evidence; never emit path/query that could carry a
// token, and never emit the scheme+creds form.
function baseUrlHost(rawBaseUrl: string): string {
  try {
    return new URL(rawBaseUrl).host;
  } catch {
    return rawBaseUrl.replace(/^[a-z]+:\/\//i, "").split("/")[0] ?? "(unknown)";
  }
}

// Mask the API key: emit NO key-body character, only a recognized non-secret
// scheme marker when present. Proves a key is configured without leaking a byte.
function maskKey(rawKey: string): string {
  if (rawKey.startsWith("sk-")) return "sk-…(redacted)";
  return "…(redacted)";
}

// Cap evidence error reasons the same way the capability probe does so eval
// evidence lines stay bounded and consistent with the capability matrix.
const EVAL_REDACT_MAX = 200;
function redactEvalError(err: unknown): string {
  return redactError(err, EVAL_REDACT_MAX);
}

// --- Per-item timeout wrapper ----------------------------------------------

class ItemTimeoutError extends Error {
  constructor(ms: number) {
    super(`eval item timed out after ${ms}ms`);
    this.name = "ItemTimeoutError";
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
      reject(new ItemTimeoutError(ms));
    }, ms);
  });
  try {
    return await Promise.race([fn(controller.signal), timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// --- Dataset loading (I/O) --------------------------------------------------

// Resolve paths relative to THIS module so the run is independent of the process
// working directory.
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

async function loadDataset(): Promise<EvalItem[]> {
  const datasetPath = path.join(repoRoot, "evals", "dataset.jsonl");
  const text = await readFile(datasetPath, "utf8");
  return parseDataset(text);
}

// --- Single-item run (resilient) -------------------------------------------

// Run one dataset item against the live model and score it. NEVER throws: a
// failure is captured as an ERROR result with a redacted reason and a
// definitively-failing score so it counts against the baseline honestly.
async function runItem(
  item: EvalItem,
  model: ReturnType<typeof createChatModel>
): Promise<ItemRunResult> {
  const start = Date.now();
  try {
    const res = await withTimeout(ITEM_TIMEOUT_MS, (signal) =>
      model.invoke(
        [
          new SystemMessage(WEDDING_PLANNER_SYSTEM_PROMPT),
          new HumanMessage(item.prompt),
        ],
        { signal }
      )
    );
    const latencyMs = Date.now() - start;
    const text =
      typeof res.content === "string"
        ? res.content
        : JSON.stringify(res.content);
    const score = scoreItem(item, text);
    return { score, prompt: item.prompt, latencyMs };
  } catch (err) {
    const latencyMs = Date.now() - start;
    // A failed call is scored against an empty response so it definitively fails
    // and lowers the aggregate honestly (rather than being silently skipped).
    const score = scoreItem(item, "");
    return {
      score,
      prompt: item.prompt,
      latencyMs,
      errorReason: redactEvalError(err),
    };
  }
}

// --- Orchestration ----------------------------------------------------------

async function main(): Promise<void> {
  const runTimestampUtc = new Date().toISOString();
  const host = baseUrlHost(config.baseURL);
  const masked = maskKey(config.apiKey);

  process.stderr.write("Wedding planner evaluation (LIVE) — opt-in\n");
  process.stderr.write(`Model         : ${config.model}\n`);
  process.stderr.write(`Base URL host : ${host}\n`);
  process.stderr.write(`API key       : ${masked} (masked)\n`);
  process.stderr.write(`Run (UTC)     : ${runTimestampUtc}\n\n`);

  const items = await loadDataset();
  process.stderr.write(`Loaded ${items.length} dataset items.\n\n`);

  // One model instance (default alias) reused across items. temperature: 0 for
  // as-deterministic-as-the-provider-allows scoring of the baseline.
  const model = createChatModel({ temperature: 0 });

  const results: ItemRunResult[] = [];
  for (const item of items) {
    process.stderr.write(`  running ${item.id} (${item.category}) ...\n`);
    const result = await runItem(item, model);
    const status = result.errorReason
      ? "ERROR"
      : result.score.passed
        ? "PASS"
        : "FAIL";
    process.stderr.write(`    -> ${status} (${result.latencyMs ?? "?"}ms)\n`);
    results.push(result);
  }

  const meta: EvalRunMeta = {
    runTimestampUtc,
    model: config.model,
    baseUrlHost: host,
    maskedKey: masked,
  };

  // Console summary.
  process.stdout.write("\n" + renderEvalConsoleSummary(results) + "\n\n");

  // Write the dated Markdown results file. Re-running the same day overwrites it
  // deterministically; the embedded UTC timestamp records the exact run.
  const outDir = path.join(repoRoot, "docs", "eval");
  await mkdir(outDir, { recursive: true });
  const dateStamp = runTimestampUtc.slice(0, 10); // YYYY-MM-DD (UTC)
  const outFile = path.join(outDir, `${dateStamp}.md`);
  await writeFile(outFile, renderEvalMarkdown(meta, results) + "\n", "utf8");

  process.stdout.write(`Results written: ${path.relative(repoRoot, outFile)}\n`);
}

main().catch((err) => {
  // Top-level guard: resilient by item, but if orchestration itself fails, never
  // leak a raw secret/PII — route the reason through the shared redaction.
  process.stderr.write("\nEvaluation run FAILED unexpectedly.\n");
  process.stderr.write(redactEvalError(err) + "\n");
  process.exit(1);
});
