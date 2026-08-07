import { fileURLToPath } from "node:url";
import path from "node:path";
import { config } from "./config.js";
import { redactError as redactErrorShared } from "./core/redaction.js";
import {
  createKnowledgeStore,
  resolveKnowledgeDbPath,
  type KnowledgeStore,
} from "./core/knowledge-store.js";
import {
  ingestDocuments,
  createDocumentEmbedder,
  type IngestResult,
} from "./core/ingestion.js";
import { loadCorpusDocuments } from "./core/corpus.js";
import { renderIngestSummary } from "./core/ingest-summary.js";

// LIVE, opt-in KNOWLEDGE INGESTION entrypoint — `npm run ingest`.
//
// PURPOSE
// -------
// Populate the DURABLE, app-owned knowledge store (the SAME SQLite file at
// KNOWLEDGE_DB_PATH that `npm run serve` with SERVICE_GROUNDED=1 reads) from the
// curated corpus under knowledge/corpus/*.md, so grounded answers can serve real
// citations. The ingestion LIBRARY has existed since Phase 7 (idempotent,
// source-addressed upsert/update); the eval runners only ever built THROWAWAY
// temp stores. This entrypoint is the missing bridge to the durable store.
//
// LIVE / COST / SCOPE
// -------------------
// It makes real, credentialed, BILLABLE embedding calls (one per corpus chunk on
// a create/update). It is NOT part of `npm test`, `npm run typecheck`,
// `npm run build`, or CI. Run explicitly:
//   npm run ingest
//
// It mirrors src/run-retrieval-eval.ts / src/run-eval.ts idioms — a redacted
// header (host-only base URL, masked key), a withTimeout bound on the live
// ingest, always-redacted error reporting, and a finally that ALWAYS close()s
// the store (releasing the SQLite handle/lock — important on Windows). The KEY
// DIFFERENCE vs the eval runners: it opens the DURABLE store via
// createKnowledgeStore() (using config.knowledgeDbPath / the documented default),
// NOT an ephemeral temp store.
//
// Embeddings are constructed ONLY via createDocumentEmbedder → createEmbeddingsModel
// (the single-factory rule). The pure, offline-testable pieces (the corpus loader
// and renderIngestSummary) live in src/core/corpus.ts and src/core/ingest-summary.ts
// and are unit-tested; this file owns the ONLY live I/O and must NEVER be imported
// by the offline suite (a structural guard enforces this).
//
// OWNERSHIP: corpus documents are ingested UNOWNED (ownerId left undefined) —
// they are public/shared knowledge, not user-scoped content. Owner-scoped
// ingestion is a deliberate FUTURE concern.
//
// IDEMPOTENCY: re-running on an unchanged corpus yields all "unchanged" (no
// re-embedding, no duplicates — the Phase 7b/7c guarantee); a changed file yields
// "updated". The printed summary makes this explicit so operators see re-runs are
// safe.

// Per-document hard timeout budget. The whole ingest is bounded by this times the
// corpus size, mirroring run-retrieval-eval.ts (CALL_TIMEOUT_MS * corpus.length),
// so a single hung embedding call cannot stall the run forever while still
// allowing a realistic multi-document corpus to complete.
const CALL_TIMEOUT_MS = 45_000;

// --- Redaction helpers (mirror run-retrieval-eval.ts / run-eval.ts) ---------

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

// --- Timeout wrapper (mirrors run-retrieval-eval.ts / run-eval.ts) ----------

class CallTimeoutError extends Error {
  constructor(ms: number) {
    super(`ingestion call timed out after ${ms}ms`);
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

// --- Paths (I/O) ------------------------------------------------------------

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const corpusDir = path.join(repoRoot, "knowledge", "corpus");

// --- Orchestration ----------------------------------------------------------

async function main(): Promise<void> {
  const host = baseUrlHost(config.baseURL);
  const masked = maskKey(config.apiKey);
  const embedModel = config.embedModel;
  // Resolve the DURABLE knowledge DB path exactly as createKnowledgeStore will
  // (dbPath arg omitted → config.knowledgeDbPath ?? default). Printing it aids
  // the operator; it is a local filesystem path (no secret), so it is safe to
  // show verbatim — consistent with how the runners surface operational context.
  const dbPath = resolveKnowledgeDbPath();

  process.stderr.write("Wedding planner knowledge ingestion (LIVE) — opt-in\n");
  process.stderr.write(`Embedding     : ${embedModel ?? "(none configured)"}\n`);
  process.stderr.write(`Embedding dim : ${config.embedDim}\n`);
  process.stderr.write(`Knowledge DB  : ${dbPath}\n`);
  process.stderr.write(`Base URL host : ${host}\n`);
  process.stderr.write(`API key       : ${masked} (masked)\n\n`);

  if (!embedModel) {
    // Embeddings have no sensible default alias; fail loud but redacted (mirrors
    // run-retrieval-eval.ts's wording). This path is offline-safe (no network).
    throw new Error(
      "No embedding alias configured. Set LITELLM_EMBED_MODEL to run ingestion " +
        "(it embeds the corpus through the proxy)."
    );
  }

  const corpus = await loadCorpusDocuments(corpusDir);
  process.stderr.write(`Loaded ${corpus.length} corpus documents.\n\n`);

  // Open the DURABLE store (config.knowledgeDbPath / documented default). This is
  // the SAME DB SERVICE_GROUNDED=1 reads — the whole point of this entrypoint.
  let store: KnowledgeStore | undefined;
  try {
    store = createKnowledgeStore();

    process.stderr.write("Ingesting curated corpus into the durable store ...\n");
    const documentEmbedder = createDocumentEmbedder({ model: embedModel });
    // Bound the whole ingest so a single hung embedding call cannot stall
    // forever. ingestDocuments runs sequentially and is idempotent per document.
    const budgetMs = CALL_TIMEOUT_MS * Math.max(1, corpus.length);
    const results: IngestResult[] = await withTimeout(budgetMs, () =>
      ingestDocuments({
        store: store!,
        embedder: documentEmbedder,
        // ownerId intentionally omitted: corpus docs are unowned/public knowledge.
        documents: corpus.map((d) => ({
          content: d.content,
          sourceUri: d.sourceUri,
        })),
      })
    );

    // Deterministic summary via the pure renderer (same-order sourceUris).
    process.stdout.write(
      "\n" +
        renderIngestSummary(
          results,
          corpus.map((d) => d.sourceUri)
        ) +
        "\n"
    );
  } finally {
    // ALWAYS release the DB handle/lock (critical on Windows) even on failure.
    if (store) store.close();
  }
}

main().catch((err) => {
  // Top-level guard: never leak a raw secret/PII — route the reason through the
  // shared redaction and exit non-zero so an operator/script sees the failure.
  process.stderr.write("\nKnowledge ingestion FAILED.\n");
  process.stderr.write(redactError(err) + "\n");
  process.exit(1);
});
