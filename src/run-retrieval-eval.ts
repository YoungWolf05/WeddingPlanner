import { readFile, readdir, mkdir, writeFile, mkdtemp, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import path from "node:path";
import { config } from "./config.js";
import { redactError as redactErrorShared } from "./core/redaction.js";
import {
  createKnowledgeStore,
  computeDocumentId,
  type KnowledgeStore,
} from "./core/knowledge-store.js";
import { ingestDocuments, createDocumentEmbedder } from "./core/ingestion.js";
import { retrieve, createQueryEmbedder } from "./core/retriever.js";
import {
  parseRetrievalDataset,
  scoreQuery,
  aggregateRetrieval,
  evaluateBaseline,
  renderRetrievalMarkdown,
  renderRetrievalConsoleSummary,
  PROPOSED_BASELINE_THRESHOLDS,
  type RetrievalEvalItem,
  type QueryScore,
  type RetrievalRunMeta,
} from "./core/retrieval-eval.js";

// Phase 7 (increment 7e) — LIVE, opt-in RETRIEVAL EVALUATION runner. Directly
// targets Phase 7 exit criterion 5: "retrieval quality meets an approved
// baseline before answer-generation tuning begins" — it produces the dated proxy
// evidence of retrieval quality vs a (PROPOSED, pending-approval) baseline.
//
// Makes live, credentialed, possibly billable EMBEDDING calls — this is NOT part
// of `npm test`, `npm run typecheck`, `npm run build`, or CI. Run explicitly:
//   npm run eval:retrieval
//
// The pure metrics/parsing/rendering + the retriever core live in
// src/core/retrieval-eval.ts and src/core/retriever.ts and are unit-tested
// offline. This file owns ONLY the live I/O + redaction, mirroring
// src/run-eval.ts and the probes:
//   - builds an EPHEMERAL knowledge store in a temp dir OUTSIDE the repo (never
//     ./data), so an offline repo-cleanliness guard is never tripped,
//   - ingests the curated corpus (knowledge/corpus/*.md) with the REAL
//     createDocumentEmbedder(),
//   - runs each dataset query through retrieve() with the REAL
//     createQueryEmbedder(),
//   - computes metrics, evaluates the baseline gate, prints a summary, and writes
//     dated evidence to docs/retrieval/<UTC-date>.md,
//   - redaction: host-only base URL, masked key, every error reason redacted,
//   - withTimeout on every live call so a hung call cannot stall the run.
//
// Embeddings are constructed ONLY via the factory adapters (createDocumentEmbedder
// / createQueryEmbedder → createEmbeddingsModel), honoring the single-factory
// rule. This file performs the ONLY live I/O and must NEVER be imported by the
// offline suite (a structural guard enforces this).

// Per-call hard timeout so a single hung embedding call cannot stall the run.
const CALL_TIMEOUT_MS = 45_000;

// The retrieval cutoff k used for every query's @k metrics.
const RETRIEVAL_K = 5;

// --- Redaction helpers (mirror run-eval.ts / the probes) --------------------

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

const REDACT_MAX = 200;
function redactError(err: unknown): string {
  return redactErrorShared(err, REDACT_MAX);
}

// --- Timeout wrapper (mirrors run-eval.ts / the probes) ---------------------

class CallTimeoutError extends Error {
  constructor(ms: number) {
    super(`retrieval eval call timed out after ${ms}ms`);
    this.name = "CallTimeoutError";
  }
}

async function withTimeout<T>(ms: number, fn: () => Promise<T>): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new CallTimeoutError(ms)), ms);
  });
  try {
    return await Promise.race([fn(), timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// --- Paths / loading (I/O) --------------------------------------------------

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

async function loadDataset(): Promise<RetrievalEvalItem[]> {
  const datasetPath = path.join(repoRoot, "evals", "retrieval.jsonl");
  const text = await readFile(datasetPath, "utf8");
  return parseRetrievalDataset(text);
}

// One corpus document ready to ingest: its repo-relative source_uri (the stable
// identity key) + its content.
interface CorpusDoc {
  sourceUri: string;
  content: string;
}

// Load every markdown file under knowledge/corpus/ as a corpus document. The
// source_uri is the POSIX-style repo-relative path (matches evals/retrieval.jsonl
// references and how computeDocumentId derives identity).
async function loadCorpus(): Promise<CorpusDoc[]> {
  const corpusDir = path.join(repoRoot, "knowledge", "corpus");
  const entries = await readdir(corpusDir, { withFileTypes: true });
  const docs: CorpusDoc[] = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".md")) continue;
    const abs = path.join(corpusDir, entry.name);
    const content = await readFile(abs, "utf8");
    docs.push({ sourceUri: `knowledge/corpus/${entry.name}`, content });
  }
  // Deterministic ingestion order.
  docs.sort((a, b) => a.sourceUri.localeCompare(b.sourceUri));
  return docs;
}

// --- Orchestration ----------------------------------------------------------

async function main(): Promise<void> {
  const runTimestampUtc = new Date().toISOString();
  const host = baseUrlHost(config.baseURL);
  const masked = maskKey(config.apiKey);
  const embedModel = config.embedModel;

  process.stderr.write("Wedding planner retrieval evaluation (LIVE) — opt-in\n");
  process.stderr.write(`Embedding     : ${embedModel ?? "(none configured)"}\n`);
  process.stderr.write(`Embedding dim : ${config.embedDim}\n`);
  process.stderr.write(`Base URL host : ${host}\n`);
  process.stderr.write(`API key       : ${masked} (masked)\n`);
  process.stderr.write(`k (cutoff)    : ${RETRIEVAL_K}\n`);
  process.stderr.write(`Run (UTC)     : ${runTimestampUtc}\n\n`);

  if (!embedModel) {
    // Embeddings have no sensible default alias; fail loud but redacted.
    throw new Error(
      "No embedding alias configured. Set LITELLM_EMBED_MODEL to run the " +
        "retrieval evaluation (it embeds the corpus and queries through the proxy)."
    );
  }

  const items = await loadDataset();
  const corpus = await loadCorpus();
  process.stderr.write(
    `Loaded ${corpus.length} corpus documents and ${items.length} queries.\n\n`
  );

  // EPHEMERAL knowledge store in a temp dir OUTSIDE the repo — never ./data — so
  // the run leaves no repo artifacts. Removed in the finally block (WAL/-shm
  // sidecars vanish with the temp dir). Its dimension follows config.embedDim.
  const tempRoot = await mkdtemp(path.join(tmpdir(), "wp-retrieval-eval-"));
  const dbPath = path.join(tempRoot, "knowledge.sqlite");
  let store: KnowledgeStore | undefined;
  try {
    store = createKnowledgeStore({ dbPath });

    // Ingest the curated corpus with the REAL document embedder.
    process.stderr.write("Ingesting curated corpus ...\n");
    const documentEmbedder = createDocumentEmbedder({ model: embedModel });
    await withTimeout(CALL_TIMEOUT_MS * corpus.length, () =>
      ingestDocuments({
        store: store!,
        embedder: documentEmbedder,
        documents: corpus.map((d) => ({
          content: d.content,
          sourceUri: d.sourceUri,
        })),
      })
    );
    process.stderr.write(`Ingested ${corpus.length} documents.\n\n`);

    // Run every query through the retriever with the REAL query embedder. Each
    // query is resilient: a failure records a zeroed score with a redacted reason.
    const queryEmbedder = createQueryEmbedder({ model: embedModel });
    const scores: QueryScore[] = [];
    for (const item of items) {
      process.stderr.write(`  querying ${item.id} ...\n`);
      const relevantDocumentIds = new Set(
        item.relevantSourceUris.map((uri) => computeDocumentId(uri))
      );
      try {
        const results = await withTimeout(CALL_TIMEOUT_MS, () =>
          retrieve({
            store: store!,
            queryEmbedder,
            query: item.query,
            k: RETRIEVAL_K,
          })
        );
        const rankedDocumentIds = results.map((r) => r.documentId);
        const score = scoreQuery(
          item.id,
          rankedDocumentIds,
          relevantDocumentIds,
          RETRIEVAL_K
        );
        process.stderr.write(
          `    -> recall ${score.metrics.recallAtK.toFixed(2)} ` +
            `RR ${score.metrics.mrr.toFixed(2)}\n`
        );
        scores.push(score);
      } catch (err) {
        // A failed retrieval counts honestly as a zeroed score (empty ranking).
        const score = scoreQuery(item.id, [], relevantDocumentIds, RETRIEVAL_K);
        score.errorReason = redactError(err);
        process.stderr.write(`    -> ERROR (${score.errorReason})\n`);
        scores.push(score);
      }
    }

    const aggregate = aggregateRetrieval(scores);
    const baseline = evaluateBaseline(aggregate, PROPOSED_BASELINE_THRESHOLDS);

    // Console summary.
    process.stdout.write(
      "\n" + renderRetrievalConsoleSummary(aggregate, baseline) + "\n\n"
    );

    // Write the dated Markdown evidence file (UTC date; same-day overwrite).
    const meta: RetrievalRunMeta = {
      runTimestampUtc,
      embedModel,
      embedDim: config.embedDim,
      baseUrlHost: host,
      maskedKey: masked,
      k: RETRIEVAL_K,
      corpusDocumentCount: corpus.length,
    };
    const outDir = path.join(repoRoot, "docs", "retrieval");
    await mkdir(outDir, { recursive: true });
    const dateStamp = runTimestampUtc.slice(0, 10); // YYYY-MM-DD (UTC)
    const outFile = path.join(outDir, `${dateStamp}.md`);
    await writeFile(
      outFile,
      renderRetrievalMarkdown(meta, scores, aggregate, baseline) + "\n",
      "utf8"
    );
    process.stdout.write(
      `Evidence written: ${path.relative(repoRoot, outFile)}\n`
    );
  } finally {
    // Release the DB handle (critical on Windows) and remove the temp dir so the
    // run never leaves an artifact anywhere.
    if (store) store.close();
    await rm(tempRoot, { recursive: true, force: true });
  }
}

main().catch((err) => {
  // Top-level guard: resilient by query, but if orchestration itself fails, never
  // leak a raw secret/PII — route the reason through the shared redaction.
  process.stderr.write("\nRetrieval evaluation run FAILED unexpectedly.\n");
  process.stderr.write(redactError(err) + "\n");
  process.exit(1);
});
