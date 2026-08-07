import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

// Shared, PURE-ish corpus loader for the curated knowledge corpus
// (knowledge/corpus/*.md). Its ONLY side effect is reading files from a caller-
// supplied directory; it holds NO policy about WHICH directory (callers pass an
// absolute path) and performs NO network / embedding / DB work. It is reused by
// both the retrieval-eval runner (src/run-retrieval-eval.ts) and the durable
// ingestion entrypoint (src/run-ingest.ts) so the two never drift on how a
// corpus file becomes an ingestable document (identity + ordering).

// One corpus document ready to ingest: its repo-relative source_uri (the stable
// identity key that computeDocumentId hashes) + its raw content.
export interface CorpusDoc {
  // POSIX-style, repo-relative path, e.g. "knowledge/corpus/venues.md". This IS
  // the document's app-owned identity input; it matches evals/retrieval.jsonl /
  // evals/rag.jsonl relevance references and how computeDocumentId derives
  // document_id. It is derived deterministically from the file NAME (below), so
  // it is stable regardless of the absolute directory the caller passes.
  sourceUri: string;
  // Raw file content (UTF-8). Normalization/hashing happens downstream in
  // ingestion; the loader passes bytes through unchanged.
  content: string;
}

// The stable repo-relative prefix every corpus document's source_uri carries.
// A file's source_uri IS its identity, so it is built from the fixed corpus
// prefix + the file name — NOT from the absolute directory the caller happens to
// pass — keeping identity independent of where the corpus lives on disk (e.g. an
// ephemeral eval temp dir vs the real repo checkout).
const CORPUS_SOURCE_URI_PREFIX = "knowledge/corpus";

/**
 * Load every markdown (`.md`) file directly under `corpusDir` as a corpus
 * document. Non-`.md` files and subdirectories are ignored. Each document's
 * `sourceUri` is the POSIX repo-relative path `knowledge/corpus/<filename>` (the
 * stable identity key), independent of the absolute `corpusDir` passed in.
 * Results are sorted by `sourceUri` (== by filename) for a DETERMINISTIC
 * ingestion order, so re-runs and tests are stable.
 *
 * Reads files but is otherwise policy-free: it does NOT resolve the corpus
 * location, embed, chunk, or touch a database. Callers own those concerns.
 */
export async function loadCorpusDocuments(corpusDir: string): Promise<CorpusDoc[]> {
  const entries = await readdir(corpusDir, { withFileTypes: true });
  const docs: CorpusDoc[] = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".md")) continue;
    const abs = path.join(corpusDir, entry.name);
    const content = await readFile(abs, "utf8");
    docs.push({
      sourceUri: `${CORPUS_SOURCE_URI_PREFIX}/${entry.name}`,
      content,
    });
  }
  // Deterministic order (stable identity → stable ordering across runs/tests).
  docs.sort((a, b) => a.sourceUri.localeCompare(b.sourceUri));
  return docs;
}
