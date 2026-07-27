import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { isAIMessage } from "@langchain/core/messages";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { config } from "./config.js";
import { createChatModel } from "./core/model.js";
import { createEmbeddingsModel } from "./core/embeddings.js";
import {
  CHAT_CAPABILITIES,
  classifyAbortOutcome,
  renderConsoleTable,
  renderConsoleEmbeddings,
  renderMarkdown,
  type AliasCapabilityResults,
  type CapabilityMatrix,
  type EmbeddingsAssessment,
  type ProbeResult,
} from "./core/capabilities.js";

// Phase 4 (increment 4c) — LiveLLM capability probe (LIVE, opt-in).
//
// Measures the real provider/model contract for each supported chat alias and,
// separately, an embedding alias if one is configured. Makes live, credentialed,
// possibly billable calls — this is NOT part of `npm test`, `npm run typecheck`,
// `npm run build`, or CI. Run explicitly: npm run test:capabilities
//
// The pure aggregation/rendering logic lives in src/core/capabilities.ts and is
// unit-tested offline. This file owns only the live I/O and classification.

// Chat aliases to probe (from AGENTS.md). We do NOT hardcode credentials —
// createChatModel() sources apiKey/baseURL from config.
const CHAT_ALIASES = ["claude-opus-4-8", "claude-sonnet-4-6"] as const;

// Per-probe timeout so a single hung capability cannot stall the whole run.
const PROBE_TIMEOUT_MS = 30_000;
// Shorter budget for the abort probe: we expect prompt cancellation.
const ABORT_TIMEOUT_MS = 15_000;

// --- Redaction --------------------------------------------------------------

// Extract only the host from the base URL for evidence; never emit path/query
// that could carry a token, and never emit the scheme+creds form.
function baseUrlHost(rawBaseUrl: string): string {
  try {
    return new URL(rawBaseUrl).host;
  } catch {
    // If it is not a parseable URL, strip any scheme and take the first segment.
    return rawBaseUrl.replace(/^[a-z]+:\/\//i, "").split("/")[0] ?? "(unknown)";
  }
}

// Mask the API key for evidence/console. We emit NO key-body character: only a
// recognized, non-secret scheme marker (e.g. "sk-…") when the key uses a known
// prefix, otherwise a fixed redaction. This proves a key is configured without
// leaking any byte of the secret itself.
function maskKey(rawKey: string): string {
  if (rawKey.startsWith("sk-")) return "sk-…(redacted)";
  return "…(redacted)";
}

// Reduce any error to a concise, redacted single-line reason. Strips the base
// URL and API key if they ever appear in a provider error string.
function redactError(err: unknown): string {
  let message =
    err instanceof Error ? err.message : typeof err === "string" ? err : String(err);
  // Collapse whitespace/newlines to one line and cap length.
  message = message.replace(/\s+/g, " ").trim();
  // Defensive scrub of any secret substrings.
  if (config.apiKey) message = message.split(config.apiKey).join("[redacted-key]");
  if (config.baseURL) message = message.split(config.baseURL).join("[redacted-url]");
  const MAX = 200;
  return message.length > MAX ? message.slice(0, MAX) + "…" : message;
}

// --- Timeout wrapper --------------------------------------------------------

class ProbeTimeoutError extends Error {
  constructor(ms: number) {
    super(`probe timed out after ${ms}ms`);
    this.name = "ProbeTimeoutError";
  }
}

// Run a probe with a hard timeout. The AbortController is passed to the probe so
// it can wire cancellation into the underlying request when it times out.
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

// --- Capability probes ------------------------------------------------------
//
// Each probe returns a ProbeResult and NEVER throws: it catches its own errors,
// distinguishes a definitive Unsupported (provider rejects the feature) from an
// Error (network/auth/unexpected/timeout), and records a redacted note.

// Heuristic: does a provider error indicate the FEATURE is unsupported (a
// definitive "no") rather than an incidental/transient/unexpected failure?
//
// We deliberately keep this conservative. A generic "invalid_request_error" is
// NOT enough — the live run showed such an error caused by an incidental
// `temperature` deprecation on one model, which is an environmental parameter
// constraint, not a structured-output limitation. Misclassifying that as
// Unsupported would poison the contract record. So we only treat an error as
// Unsupported when it explicitly references the feature being unavailable, and
// we explicitly exclude known incidental parameter errors.
function looksUnsupported(err: unknown): boolean {
  const msg = (err instanceof Error ? err.message : String(err)).toLowerCase();

  // Incidental parameter/validation errors are NOT feature rejections; classify
  // them as Error so the note carries the real reason for a human to review.
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

async function probeInvoke(alias: string): Promise<ProbeResult> {
  try {
    const model = createChatModel({ model: alias, temperature: 0 });
    const res = await withTimeout(PROBE_TIMEOUT_MS, (signal) =>
      model.invoke("Reply with the single word: pong.", { signal })
    );
    const text = typeof res.content === "string" ? res.content : JSON.stringify(res.content);
    if (!text || text.trim().length === 0) {
      return { state: "Degraded", note: "response returned but text content was empty" };
    }
    return { state: "Supported" };
  } catch (err) {
    return { state: "Error", note: redactError(err) };
  }
}

async function probeStreaming(alias: string): Promise<ProbeResult> {
  try {
    const model = createChatModel({ model: alias, temperature: 0, streaming: true });
    let chunks = 0;
    await withTimeout(PROBE_TIMEOUT_MS, async (signal) => {
      const stream = await model.stream(
        "Count slowly from one to ten, one number per line.",
        { signal }
      );
      for await (const _chunk of stream) {
        chunks += 1;
      }
    });
    if (chunks === 0) {
      return { state: "Unsupported", note: "stream produced no chunks" };
    }
    if (chunks === 1) {
      return { state: "Degraded", note: "stream produced only a single chunk (no incremental streaming)" };
    }
    return { state: "Supported" };
  } catch (err) {
    return { state: "Error", note: redactError(err) };
  }
}

// Does an error look like a genuine cancellation (AbortError / "abort" text)?
function isAbortError(err: unknown): boolean {
  if (err instanceof Error && err.name === "AbortError") return true;
  const msg = err instanceof Error ? err.message : String(err);
  return /abort/i.test(msg);
}

async function probeAbort(alias: string): Promise<ProbeResult> {
  // Start a stream and abort it almost immediately; verify prompt cancellation.
  //
  // We gather the OBSERVABLE outcome here (I/O) and delegate the state decision
  // to the pure classifyAbortOutcome() so the classification is deterministic
  // and unit-tested offline (finding B1). Crucially, a fast NON-abort error
  // (e.g. an invalid model or auth failure) must NOT be reported as a
  // successful abort — it is Unsupported/Error, never Supported.
  const model = createChatModel({ model: alias, temperature: 0, streaming: true });
  const controller = new AbortController();
  const start = Date.now();
  try {
    const stream = await model.stream(
      "Write a very long, detailed essay about wedding planning.",
      { signal: controller.signal }
    );
    // Abort shortly after the stream begins.
    const aborter = setTimeout(() => controller.abort(), 150);
    try {
      // Enforce an outer bound: if abort does not stop the stream, this rejects
      // with ProbeTimeoutError (its own internal controller, not ours).
      await withTimeout(ABORT_TIMEOUT_MS, async () => {
        for await (const _chunk of stream) {
          // Drain until the abort takes effect.
        }
      });
    } finally {
      clearTimeout(aborter);
    }
    // The loop completed WITHOUT throwing — abort was ignored.
    return classifyAbortOutcome({
      completed: true,
      timedOut: false,
      signalAborted: controller.signal.aborted,
      isAbortError: false,
      elapsedMs: Date.now() - start,
      errorLooksUnsupported: false,
    });
  } catch (err) {
    const result = classifyAbortOutcome({
      completed: false,
      timedOut: err instanceof ProbeTimeoutError,
      signalAborted: controller.signal.aborted,
      isAbortError: isAbortError(err),
      elapsedMs: Date.now() - start,
      errorLooksUnsupported: looksUnsupported(err),
    });
    // Attach a redacted reason for non-abort failures (Unsupported/Error) that
    // the pure classifier intentionally leaves noteless.
    if (!result.note && (result.state === "Error" || result.state === "Unsupported")) {
      return { ...result, note: redactError(err) };
    }
    return result;
  }
}

async function probeUsageMetadata(alias: string): Promise<ProbeResult> {
  try {
    const model = createChatModel({ model: alias, temperature: 0 });
    const res = await withTimeout(PROBE_TIMEOUT_MS, (signal) =>
      model.invoke("Say hello in one short sentence.", { signal })
    );
    const usage = isAIMessage(res) ? res.usage_metadata : undefined;
    if (!usage) {
      return { state: "Degraded", note: "response returned but usage_metadata was absent" };
    }
    const hasTokens =
      typeof usage.input_tokens === "number" &&
      typeof usage.output_tokens === "number" &&
      typeof usage.total_tokens === "number";
    if (!hasTokens) {
      return { state: "Degraded", note: "usage_metadata present but token counts missing" };
    }
    return { state: "Supported" };
  } catch (err) {
    return { state: "Error", note: redactError(err) };
  }
}

async function probeToolCalling(alias: string): Promise<ProbeResult> {
  const addTool = tool(
    ({ a, b }: { a: number; b: number }) => String(a + b),
    {
      name: "add",
      description: "Add two integers and return the sum.",
      schema: z.object({
        a: z.number().describe("first addend"),
        b: z.number().describe("second addend"),
      }),
    }
  );
  try {
    const model = createChatModel({ model: alias, temperature: 0 });
    const bound = model.bindTools([addTool]);
    const res = await withTimeout(PROBE_TIMEOUT_MS, (signal) =>
      bound.invoke("Use the add tool to compute 17 + 25. Call the tool.", { signal })
    );
    const toolCalls = isAIMessage(res) ? res.tool_calls ?? [] : [];
    if (toolCalls.length === 0) {
      return { state: "Unsupported", note: "model returned no tool call" };
    }
    const call = toolCalls[0]!;
    if (call.name !== "add" || typeof call.args?.["a"] !== "number" || typeof call.args?.["b"] !== "number") {
      return { state: "Degraded", note: `tool call present but malformed (name=${call.name})` };
    }
    return { state: "Supported" };
  } catch (err) {
    if (looksUnsupported(err)) {
      return { state: "Unsupported", note: redactError(err) };
    }
    return { state: "Error", note: redactError(err) };
  }
}

async function probeStructuredOutput(alias: string): Promise<ProbeResult> {
  const schema = z.object({
    city: z.string().describe("the city name"),
    country: z.string().describe("the country name"),
  });
  try {
    const model = createChatModel({ model: alias, temperature: 0 });
    const structured = model.withStructuredOutput(schema, { name: "location" });
    const res = await withTimeout(PROBE_TIMEOUT_MS, (signal) =>
      structured.invoke("What city is the Eiffel Tower in? Provide city and country.", { signal })
    );
    const parsed = schema.safeParse(res);
    if (!parsed.success) {
      return { state: "Degraded", note: "output returned but failed schema validation" };
    }
    return { state: "Supported" };
  } catch (err) {
    if (looksUnsupported(err)) {
      return { state: "Unsupported", note: redactError(err) };
    }
    return { state: "Error", note: redactError(err) };
  }
}

// Probe every chat capability for one alias. Resilient: a failing capability is
// recorded, never thrown, so the rest of the alias and run continue.
async function probeAlias(alias: string): Promise<AliasCapabilityResults> {
  process.stderr.write(`  probing ${alias} ...\n`);
  const results: AliasCapabilityResults = {
    invoke: await probeInvoke(alias),
    streaming: await probeStreaming(alias),
    abort: await probeAbort(alias),
    usageMetadata: await probeUsageMetadata(alias),
    toolCalling: await probeToolCalling(alias),
    structuredOutput: await probeStructuredOutput(alias),
  };
  return results;
}

// --- Embeddings (separate contract) ----------------------------------------

async function probeEmbeddings(): Promise<EmbeddingsAssessment> {
  const alias = config.embedModel ?? null;
  if (!alias) {
    return {
      alias: null,
      result: { state: "N/A", note: "no embedding alias configured (set LITELLM_EMBED_MODEL)" },
    };
  }
  process.stderr.write(`  probing embedding alias ${alias} ...\n`);
  try {
    const embeddings = createEmbeddingsModel({ model: alias });
    const vector = await withTimeout(PROBE_TIMEOUT_MS, async () =>
      embeddings.embedQuery("wedding venue with garden and sunset views")
    );
    if (!Array.isArray(vector) || vector.length === 0) {
      return { alias, result: { state: "Degraded", note: "empty embedding vector returned" } };
    }
    return { alias, result: { state: "Supported" }, dimensions: vector.length };
  } catch (err) {
    if (looksUnsupported(err)) {
      return { alias, result: { state: "Unsupported", note: redactError(err) } };
    }
    return { alias, result: { state: "Error", note: redactError(err) } };
  }
}

// --- Orchestration ----------------------------------------------------------

async function main(): Promise<void> {
  const runTimestampUtc = new Date().toISOString();
  const host = baseUrlHost(config.baseURL);
  const masked = maskKey(config.apiKey);

  process.stderr.write("LiteLLM capability probe (LIVE) — Phase 4 / 4c\n");
  process.stderr.write(`Base URL host : ${host}\n`);
  process.stderr.write(`API key       : ${masked} (masked)\n`);
  process.stderr.write(`Run (UTC)     : ${runTimestampUtc}\n\n`);

  const chat: Record<string, AliasCapabilityResults> = {};
  for (const alias of CHAT_ALIASES) {
    try {
      chat[alias] = await probeAlias(alias);
    } catch (err) {
      // Defensive: probeAlias should never throw, but if it does, record every
      // capability for this alias as an Error rather than aborting the run.
      const reason = redactError(err);
      const errored = Object.fromEntries(
        CHAT_CAPABILITIES.map((cap) => [cap, { state: "Error", note: reason } as ProbeResult])
      ) as AliasCapabilityResults;
      chat[alias] = errored;
    }
  }

  const embeddings = await probeEmbeddings();

  const matrix: CapabilityMatrix = {
    runTimestampUtc,
    baseUrlHost: host,
    maskedKey: masked,
    chat,
    embeddings,
  };

  // Print the human-readable table.
  process.stdout.write("\n" + renderConsoleTable(matrix) + "\n\n");
  process.stdout.write(renderConsoleEmbeddings(matrix) + "\n\n");

  // Write the dated Markdown evidence file. If a file for today already exists,
  // we overwrite it deterministically (the embedded UTC run timestamp records
  // the exact run), so re-running the same day yields one authoritative file.
  const testDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(testDir, "..");
  const outDir = path.join(repoRoot, "docs", "capabilities");
  await mkdir(outDir, { recursive: true });
  const dateStamp = runTimestampUtc.slice(0, 10); // YYYY-MM-DD (UTC)
  const outFile = path.join(outDir, `${dateStamp}.md`);
  await writeFile(outFile, renderMarkdown(matrix) + "\n", "utf8");

  process.stdout.write(`Evidence written: ${path.relative(repoRoot, outFile)}\n`);
}

main().catch((err) => {
  // Top-level guard: the run should be resilient, but never leak a raw secret.
  process.stderr.write("\nCapability probe FAILED unexpectedly.\n");
  process.stderr.write((err instanceof Error ? err.message : String(err)) + "\n");
  process.exit(1);
});
