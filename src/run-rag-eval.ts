import { readFile, readdir, mkdir, writeFile, mkdtemp, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import path from "node:path";
import { config } from "./config.js";
import { redactError as redactErrorShared } from "./core/redaction.js";
import {
  createKnowledgeStore,
  type KnowledgeStore,
} from "./core/knowledge-store.js";
import { ingestDocuments, createDocumentEmbedder } from "./core/ingestion.js";
import { createQueryEmbedder } from "./core/retriever.js";
import { answerQuestion } from "./core/rag.js";
import { DEFAULT_MIN_EVIDENCE_SCORE } from "./core/evidence.js";
import {
  parseRagDataset,
  scoreItem,
  failedItemScore,
  aggregateRag,
  evaluateRagBaseline,
  renderRagMarkdown,
  renderRagConsoleSummary,
  PROPOSED_RAG_BASELINE_THRESHOLDS,
  type RagEvalItem,
  type ItemScore,
  type RagRunMeta,
} from "./core/rag-eval.js";

// Phase 8 (increment 8d) — LIVE, opt-in GROUNDED-ANSWER (RAG) EVALUATION runner.
// Directly targets Phase 8 exit criterion 3: "Evaluation covers groundedness,
// citation precision/recall, malicious source instructions, and missing
// evidence" — it produces the dated proxy evidence of grounded-answer quality vs
// a (PROPOSED, pending-approval) baseline, and is the record against which the 8c
// DEFAULT_MIN_EVIDENCE_SCORE is ratified at closeout.
//
// Makes live, credentialed, BILLABLE calls — BOTH embeddings (corpus + query)
// AND chat generation (the grounded answer). This is NOT part of `npm test`,
// `npm run typecheck`, `npm run build`, or CI. Run explicitly:
//   npm run eval:rag
//
// The pure scorers/parsing/rendering live in src/core/rag-eval.ts and the RAG
// pipeline in src/core/rag.ts; both are unit-tested offline. This file owns ONLY
// the live I/O + redaction, mirroring src/run-retrieval-eval.ts EXACTLY:
//   - builds an EPHEMERAL knowledge store in a temp dir OUTSIDE the repo (never
//     ./data), so an offline repo-cleanliness guard is never tripped,
//   - ingests the curated corpus (knowledge/corpus/*.md) with the REAL
//     createDocumentEmbedder(),
//   - runs each dataset item through answerQuestion() with the REAL
//     createQueryEmbedder() (the default generation model + default minScore =
//     DEFAULT_MIN_EVIDENCE_SCORE, so the run exercises the ratified-at-closeout
//     cutoff),
//   - scores with the pure property scorers, aggregates, evaluates the baseline
//     gate, prints a summary, and writes dated evidence to
//     docs/rag-eval/<UTC-date>.md,
//   - redaction: host-only base URL, masked key, every error reason redacted,
//   - withTimeout on every live call so a hung call cannot stall the run.
//
// Models are constructed ONLY via the factory adapters/createChatModel path:
// createDocumentEmbedder/createQueryEmbedder route through createEmbeddingsModel,
// and answerQuestion routes generation through generateStructured -> createChatModel
// (single-factory rule). This file performs the ONLY live I/O and must NEVER be
// imported by the offline suite (a structural guard enforces this).

// Per-call hard timeout so a single hung embedding/chat call cannot stall the run.
const CALL_TIMEOUT_MS = 60_000;

// The retrieval cutoff k used for every item's answerQuestion call.
const RETRIEVAL_K = 5;

// --- Redaction helpers (mirror run-retrieval-eval.ts) -----------------------

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

// --- Timeout wrapper (mirrors run-retrieval-eval.ts) ------------------------

class CallTimeoutError extends Error {
  constructor(ms: number) {
    super(`rag eval call timed out after ${ms}ms`);
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

async function loadDataset(): Promise<RagEvalItem[]> {
  const datasetPath = path.join(repoRoot, "evals", "rag.jsonl");
  const text = await readFile(datasetPath, "utf8");
  return parseRagDataset(text);
}

// One corpus document ready to ingest: its repo-relative source_uri (the stable
// identity key) + its content.
interface CorpusDoc {
  sourceUri: string;
  content: string;
}

// Load every markdown file under knowledge/corpus/ as a corpus document. The
// source_uri is the POSIX-style repo-relative path (matches evals/rag.jsonl
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
  const chatModel = config.model;
  const minScore = DEFAULT_MIN_EVIDENCE_SCORE;

  process.stderr.write("Wedding planner grounded-answer (RAG) evaluation (LIVE) — opt-in\n");
  process.stderr.write(`Chat model    : ${chatModel}\n`);
  process.stderr.write(`Embedding     : ${embedModel ?? "(none configured)"}\n`);
  process.stderr.write(`Embedding dim : ${config.embedDim}\n`);
  process.stderr.write(`Base URL host : ${host}\n`);
  process.stderr.write(`API key       : ${masked} (masked)\n`);
  process.stderr.write(`k (cutoff)    : ${RETRIEVAL_K}\n`);
  process.stderr.write(`minScore      : ${minScore}\n`);
  process.stderr.write(`Run (UTC)     : ${runTimestampUtc}\n\n`);

  if (!embedModel) {
    // Embeddings have no sensible default alias; fail loud but redacted.
    throw new Error(
      "No embedding alias configured. Set LITELLM_EMBED_MODEL to run the RAG " +
        "evaluation (it embeds the corpus and queries through the proxy)."
    );
  }

  const items = await loadDataset();
  const corpus = await loadCorpus();
  process.stderr.write(
    `Loaded ${corpus.length} corpus documents and ${items.length} items.\n\n`
  );

  // EPHEMERAL knowledge store in a temp dir OUTSIDE the repo — never ./data — so
  // the run leaves no repo artifacts. Removed in the finally block (WAL/-shm
  // sidecars vanish with the temp dir). Its dimension follows config.embedDim.
  const tempRoot = await mkdtemp(path.join(tmpdir(), "wp-rag-eval-"));
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

    // Run every item through the full RAG pipeline with the REAL query embedder +
    // the default generation model. Each item is resilient: a failure records a
    // failing score with a redacted reason so a transport/model error counts
    // honestly rather than silently vanishing.
    const queryEmbedder = createQueryEmbedder({ model: embedModel });
    const scores: ItemScore[] = [];
    for (const item of items) {
      process.stderr.write(`  answering ${item.id} (${item.category}) ...\n`);
      try {
        const result = await withTimeout(CALL_TIMEOUT_MS, () =>
          answerQuestion({
            store: store!,
            queryEmbedder,
            query: item.query,
            k: RETRIEVAL_K,
            minScore,
          })
        );
        const score = scoreItem(item, result);
        process.stderr.write(
          `    -> ${score.passed ? "PASS" : "FAIL"} (${score.evidenceStatus})\n`
        );
        scores.push(score);
      } catch (err) {
        const reason = redactError(err);
        process.stderr.write(`    -> ERROR (${reason})\n`);
        scores.push(failedItemScore(item, reason));
      }
    }

    const aggregate = aggregateRag(scores);
    const baseline = evaluateRagBaseline(aggregate, PROPOSED_RAG_BASELINE_THRESHOLDS);

    // Console summary.
    process.stdout.write(
      "\n" + renderRagConsoleSummary(aggregate, baseline) + "\n\n"
    );

    // Write the dated Markdown evidence file (UTC date; same-day overwrite).
    const meta: RagRunMeta = {
      runTimestampUtc,
      chatModel,
      embedModel,
      embedDim: config.embedDim,
      baseUrlHost: host,
      maskedKey: masked,
      k: RETRIEVAL_K,
      minScore,
      corpusDocumentCount: corpus.length,
    };
    const outDir = path.join(repoRoot, "docs", "rag-eval");
    await mkdir(outDir, { recursive: true });
    const dateStamp = runTimestampUtc.slice(0, 10); // YYYY-MM-DD (UTC)
    const outFile = path.join(outDir, `${dateStamp}.md`);
    await writeFile(
      outFile,
      renderRagMarkdown(meta, scores, aggregate, baseline) + "\n",
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
  // Top-level guard: resilient by item, but if orchestration itself fails, never
  // leak a raw secret/PII — route the reason through the shared redaction.
  process.stderr.write("\nRAG evaluation run FAILED unexpectedly.\n");
  process.stderr.write(redactError(err) + "\n");
  process.exit(1);
});
