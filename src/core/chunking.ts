import { normalizeContent } from "./knowledge-store.js";

// Phase 7 (7b): DETERMINISTIC recursive text chunking.
//
// WHAT THIS MODULE OWNS
// ---------------------
// A PURE, deterministic recursive splitter that turns a source document's text
// into an ordered list of overlapping chunks. It is the FIRST stage of the 7b
// ingestion pipeline: chunk boundaries are what `computeChunkId` (7a) hashes, so
// they MUST be stable and reproducible across runs and machines. This module is
// I/O-free and has NO dependency on the store, embeddings, or the network.
//
// DETERMINISM CONTRACT
// --------------------
//   1. The input is FIRST run through `normalizeContent` (7a) — the SAME
//      normalization used to derive `document_id`/`content_hash` — so chunking
//      operates on exactly the bytes that identity is computed from. Line-ending
//      style and Unicode composition can therefore never shift a boundary.
//   2. All operations are plain, locale-independent string operations (no regex
//      locale/Unicode flags, no `Intl`, no randomness), so identical input +
//      identical options always yield byte-for-byte identical chunks.
//
// ALGORITHM (recursive character splitting)
// -----------------------------------------
// A faithful, self-contained implementation of the well-known recursive
// character text-splitter strategy (the same idea as LangChain's
// RecursiveCharacterTextSplitter), kept in-repo rather than pulling the heavy
// `langchain` meta-package for one function. It is small and fully specified:
//
//   - `separators` is an ORDERED list tried from most to least semantic. For a
//     given (sub)text the FIRST separator that occurs in it is chosen; the text
//     is split on that separator and the pieces are greedily merged back into
//     chunks up to `chunkSize`. Any single piece still larger than `chunkSize`
//     is recursively re-split using the REMAINING (finer) separators.
//   - The final separator SHOULD be "" (the empty string), which splits into
//     individual characters and guarantees termination even for text with no
//     natural boundaries (e.g. one enormous "word"). The default list ends in "".
//   - When merging, `chunkOverlap` characters of trailing context are carried
//     into the next chunk, so adjacent chunks share a documented overlap window.
//
// The separator is used as the JOIN delimiter when re-assembling pieces
// (keepSeparator=false semantics), which keeps the implementation simple and the
// output faithful for the default whitespace-oriented separators.

// Default target chunk size, in CHARACTERS (not tokens). 1000 chars is a
// sensible default for the short-form wedding-domain corpus (checklists, budget
// notes, venue/vendor blurbs): large enough to keep a coherent idea together,
// small enough to embed precisely.
export const DEFAULT_CHUNK_SIZE = 1000;

// Default overlap, in CHARACTERS, carried from the end of one chunk into the
// start of the next so context spanning a boundary is not lost at retrieval
// time. Must be strictly LESS than the chunk size (see validation).
export const DEFAULT_CHUNK_OVERLAP = 200;

// Default ordered separator list, coarsest → finest: paragraph break, line
// break, sentence boundary, word boundary, then character. Ending in "" is
// deliberate — it guarantees any oversized atomic run is still split down to
// size. This constant is frozen so a caller cannot mutate the shared default.
export const DEFAULT_SEPARATORS: readonly string[] = Object.freeze([
  "\n\n",
  "\n",
  ". ",
  " ",
  "",
]);

// Injectable chunking parameters. All optional; each falls back to the
// documented default above. Kept injectable so tests pin tiny sizes and so a
// future corpus can tune sizing without touching the algorithm.
export interface ChunkingOptions {
  // Target maximum chunk length in characters. Must be a positive integer.
  chunkSize?: number;
  // Characters of overlap between adjacent chunks. Must be >= 0 and STRICTLY
  // less than `chunkSize`.
  chunkOverlap?: number;
  // Ordered separators, coarsest → finest. Must be non-empty. Should end with ""
  // to guarantee termination on unsplittable text.
  separators?: readonly string[];
}

// Fully-resolved parameters after defaulting + validation.
interface ResolvedChunkingOptions {
  chunkSize: number;
  chunkOverlap: number;
  separators: readonly string[];
}

// Validate and resolve options once, up front. Guards the documented edge cases
// (non-positive size, overlap >= size, empty separator list) with loud, explicit
// errors rather than producing silently-degenerate output. These messages are
// developer-facing configuration errors with no secret/PII content.
function resolveOptions(options: ChunkingOptions): ResolvedChunkingOptions {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const chunkOverlap = options.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP;
  const separators = options.separators ?? DEFAULT_SEPARATORS;

  if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
    throw new Error(
      `Invalid chunkSize ${String(chunkSize)}: must be a positive integer.`
    );
  }
  if (!Number.isInteger(chunkOverlap) || chunkOverlap < 0) {
    throw new Error(
      `Invalid chunkOverlap ${String(chunkOverlap)}: must be a non-negative integer.`
    );
  }
  if (chunkOverlap >= chunkSize) {
    throw new Error(
      `Invalid chunking configuration: chunkOverlap (${chunkOverlap}) must be ` +
        `strictly less than chunkSize (${chunkSize}).`
    );
  }
  if (separators.length === 0) {
    throw new Error(
      "Invalid chunking configuration: separators must be a non-empty list."
    );
  }
  return { chunkSize, chunkOverlap, separators };
}

// Split `text` on a single separator. The empty separator "" splits into
// individual characters (the recursion's base case). A non-empty separator drops
// empty pieces so repeated separators do not create empty chunks. No regex is
// used, so separators need no escaping and behavior is locale-independent.
function splitOnSeparator(text: string, separator: string): string[] {
  if (separator === "") {
    return Array.from(text);
  }
  return text.split(separator).filter((piece) => piece !== "");
}

// Join merged pieces with the separator and trim surrounding whitespace. Returns
// null for an empty/whitespace-only result so it is dropped from the output.
function joinPieces(pieces: string[], separator: string): string | null {
  const joined = pieces.join(separator).trim();
  return joined === "" ? null : joined;
}

// Greedily merge small `splits` into chunks up to `chunkSize`, carrying
// `chunkOverlap` characters of trailing context into each subsequent chunk.
// This is the standard recursive-splitter merge: when adding the next piece
// would exceed the size budget, the current chunk is flushed and the front of
// the working buffer is popped until the retained tail is within the overlap
// budget. Deterministic: no data-dependent tie-breaking beyond the size math.
function mergeSplits(
  splits: string[],
  separator: string,
  opts: ResolvedChunkingOptions
): string[] {
  const sepLen = separator.length;
  const chunks: string[] = [];
  const current: string[] = [];
  let total = 0;

  for (const piece of splits) {
    const pieceLen = piece.length;
    // Would appending this piece (plus a joining separator if the buffer is
    // non-empty) overflow the size budget?
    if (total + pieceLen + (current.length > 0 ? sepLen : 0) > opts.chunkSize) {
      if (current.length > 0) {
        const chunk = joinPieces(current, separator);
        if (chunk !== null) chunks.push(chunk);
        // Pop from the front until the retained tail fits the overlap window
        // (or until making room for the incoming piece). This is what produces
        // the documented overlap between adjacent chunks.
        while (
          total > opts.chunkOverlap ||
          (total + pieceLen + (current.length > 0 ? sepLen : 0) > opts.chunkSize &&
            total > 0)
        ) {
          const removed = current[0]!;
          total -= removed.length + (current.length > 1 ? sepLen : 0);
          current.shift();
        }
      }
    }
    current.push(piece);
    total += pieceLen + (current.length > 1 ? sepLen : 0);
  }

  const last = joinPieces(current, separator);
  if (last !== null) chunks.push(last);
  return chunks;
}

// Recursively split `text` using the ordered `separators`. Chooses the first
// separator present in `text` (defaulting to the last separator if none match),
// splits, then either keeps a piece as-is, merges runs of small pieces, or
// recurses into a still-too-large piece with the finer remaining separators.
function splitRecursive(
  text: string,
  separators: readonly string[],
  opts: ResolvedChunkingOptions
): string[] {
  const finalChunks: string[] = [];

  // Select the separator: the first one that occurs in `text`; "" always
  // "occurs". Remember the finer separators for recursion into oversized pieces.
  let separator = separators[separators.length - 1]!;
  let remaining: readonly string[] = [];
  for (let i = 0; i < separators.length; i++) {
    const candidate = separators[i]!;
    if (candidate === "" || text.includes(candidate)) {
      separator = candidate;
      remaining = separators.slice(i + 1);
      break;
    }
  }

  const splits = splitOnSeparator(text, separator);

  // Buffer of consecutive small pieces to be merged with `mergeSplits`.
  let goodSplits: string[] = [];
  for (const piece of splits) {
    if (piece.length < opts.chunkSize) {
      goodSplits.push(piece);
      continue;
    }
    // Flush any pending small pieces first, preserving order.
    if (goodSplits.length > 0) {
      finalChunks.push(...mergeSplits(goodSplits, separator, opts));
      goodSplits = [];
    }
    // An oversized piece: recurse with finer separators, or (if none remain)
    // keep it whole — the "" base case guarantees this only happens when the
    // caller supplied a separator list that cannot split the text further.
    if (remaining.length === 0) {
      finalChunks.push(piece);
    } else {
      finalChunks.push(...splitRecursive(piece, remaining, opts));
    }
  }
  if (goodSplits.length > 0) {
    finalChunks.push(...mergeSplits(goodSplits, separator, opts));
  }
  return finalChunks;
}

/**
 * Split source `content` into an ordered list of overlapping chunks.
 *
 * The content is FIRST normalized with `normalizeContent` (7a) so chunk
 * boundaries align exactly with the bytes that `document_id`/`chunk_id` are
 * hashed from — making the output deterministic and reproducible.
 *
 * Documented edge cases:
 *   - EMPTY or WHITESPACE-ONLY content → returns `[]` (no chunks). Ingestion
 *     therefore calls the embedder zero times for such input.
 *   - Content SHORTER than one chunk → returns a single trimmed chunk.
 *   - `chunkOverlap >= chunkSize`, non-positive `chunkSize`, or an empty
 *     `separators` list → throws a validation error (see {@link resolveOptions}).
 *
 * @param content raw source text (will be normalized internally).
 * @param options injectable {@link ChunkingOptions}; omitted fields use the
 *                documented defaults.
 */
export function chunkText(
  content: string,
  options: ChunkingOptions = {}
): string[] {
  const opts = resolveOptions(options);
  const normalized = normalizeContent(content);
  // Whitespace-only (or empty) content has no chunkable material.
  if (normalized.trim() === "") {
    return [];
  }
  return splitRecursive(normalized, opts.separators, opts);
}
