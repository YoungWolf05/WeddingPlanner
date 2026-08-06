import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { config } from "./config.js";
import { createEmbeddingsModel } from "./core/embeddings.js";
import { redactError as redactErrorShared } from "./core/redaction.js";
import {
  classifyEmbeddingCompatibility,
  renderEmbeddingCompatConsole,
  renderEmbeddingCompatMarkdown,
  type EmbeddingCompatReport,
} from "./core/embedding-compat.js";

// Phase 7 (increment 7d) — LIVE, opt-in EMBEDDING & DIMENSION compatibility
// probe. Directly targets Phase 7 exit criterion 4: "the embedding alias and
// vector dimensions are verified through the proxy and covered by compatibility
// checks".
//
// Makes a live, credentialed, possibly billable embedding call — this is NOT
// part of `npm test`, `npm run typecheck`, `npm run build`, or CI. Run
// explicitly:
//   npm run test:embedding
//
// The pure classification/rendering logic lives in src/core/embedding-compat.ts
// and is unit-tested offline. This file owns ONLY the live I/O + redaction,
// mirroring src/probe-capabilities.ts / src/probe-contracts.ts:
//   - a per-probe timeout so a hung call cannot stall the run,
//   - redaction host-only base URL + masked key; every error reason is redacted,
//   - resilient: an alias failure is recorded as Error, never thrown,
//   - a dated Markdown evidence file under docs/embeddings/<UTC-date>.md.
//
// The embedder is constructed ONLY via createEmbeddingsModel (the single
// OpenAIEmbeddings factory). This file performs the ONLY live I/O and must NEVER
// be imported by the offline suite.

// Per-probe timeout so a single hung embedding call cannot stall the run. Same
// budget as the capability probe's embedding step.
const PROBE_TIMEOUT_MS = 30_000;

// --- Redaction (host-only base URL + masked key; mirrors the other probes) ----

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

// Pin the probe's redaction cap to the same 200-char bound as the other probes
// so evidence lines stay bounded.
const PROBE_REDACT_MAX = 200;
function redactError(err: unknown): string {
  return redactErrorShared(err, PROBE_REDACT_MAX);
}

// --- Timeout wrapper (mirrors the other probes) -----------------------------

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

// --- Orchestration ----------------------------------------------------------

async function main(): Promise<void> {
  const runTimestampUtc = new Date().toISOString();
  const host = baseUrlHost(config.baseURL);
  const masked = maskKey(config.apiKey);
  const expectedDim = config.embedDim;
  const alias = config.embedModel ?? null;

  process.stderr.write("LiteLLM embedding compatibility probe (LIVE) — opt-in\n");
  process.stderr.write(`Base URL host : ${host}\n`);
  process.stderr.write(`API key       : ${masked} (masked)\n`);
  process.stderr.write(`Expected dim  : ${expectedDim}\n`);
  process.stderr.write(`Run (UTC)     : ${runTimestampUtc}\n\n`);

  // Build the assessment. When no alias is configured we record an Unverified
  // report rather than throwing (parity with the capability probe's N/A path).
  let observedDim: number | undefined;
  let error: string | undefined;
  if (alias === null) {
    process.stderr.write(
      "  no embedding alias configured (set LITELLM_EMBED_MODEL) — Unverified\n"
    );
  } else {
    process.stderr.write(`  probing embedding alias ${alias} ...\n`);
    try {
      const embeddings = createEmbeddingsModel({ model: alias });
      // A benign, PII-free wedding-domain probe sentence.
      const vector = await withTimeout(PROBE_TIMEOUT_MS, async () =>
        embeddings.embedQuery(
          "wedding venue with a garden ceremony and a sunset reception"
        )
      );
      observedDim = Array.isArray(vector) ? vector.length : 0;
    } catch (err) {
      error = redactError(err);
    }
  }

  const classification = classifyEmbeddingCompatibility({
    alias,
    expectedDim,
    observedDim,
    error,
  });

  const report: EmbeddingCompatReport = {
    runTimestampUtc,
    baseUrlHost: host,
    maskedKey: masked,
    alias,
    expectedDim: classification.expectedDim,
    observedDim: classification.observedDim,
    state: classification.state,
    note: classification.note,
  };

  process.stdout.write("\n" + renderEmbeddingCompatConsole(report) + "\n\n");

  // Write the dated Markdown evidence file. Re-running the same day overwrites it
  // deterministically; the embedded UTC run timestamp records the exact run.
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(scriptDir, "..");
  const outDir = path.join(repoRoot, "docs", "embeddings");
  await mkdir(outDir, { recursive: true });
  const dateStamp = runTimestampUtc.slice(0, 10); // YYYY-MM-DD (UTC)
  const outFile = path.join(outDir, `${dateStamp}.md`);
  await writeFile(outFile, renderEmbeddingCompatMarkdown(report) + "\n", "utf8");

  process.stdout.write(`Evidence written: ${path.relative(repoRoot, outFile)}\n`);
}

main().catch((err) => {
  // Top-level guard: resilient run, but never leak a raw secret/PII. Route the
  // reason through the shared redaction (same 200-char probe cap).
  process.stderr.write("\nEmbedding compatibility probe FAILED unexpectedly.\n");
  process.stderr.write(redactError(err) + "\n");
  process.exit(1);
});
