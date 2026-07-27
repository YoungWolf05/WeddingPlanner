import { describe, it, expect } from "vitest";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

// Phase 4 — architecture convention guard (4b: ChatOpenAI; 4c: OpenAIEmbeddings).
//
// AGENTS.md convention: "All application LLM construction goes through
// createChatModel() in src/core/model.ts — do not instantiate ChatOpenAI
// elsewhere." Increment 4c adds the embeddings equivalent: OpenAIEmbeddings must
// only be constructed in src/core/embeddings.ts (via createEmbeddingsModel()).
//
// This is a DETERMINISTIC, fully OFFLINE static source scan: it reads the
// TypeScript sources under src/ and fails if `new ChatOpenAI(` appears anywhere
// except src/core/model.ts, or `new OpenAIEmbeddings(` anywhere except
// src/core/embeddings.ts. Comments and string/template literals are stripped
// first, so references inside comments (like these) never false-positive.
//
// It performs no network calls, needs no credentials, and does not import any
// application module — so a regression is caught by `npm test` regardless of
// environment.

// Resolve the repository's src/ directory relative to THIS test file so the
// scan is independent of the process working directory.
const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "..");
const srcDir = path.join(repoRoot, "src");

// The one and only file permitted to construct ChatOpenAI directly.
const ALLOWED_FACTORY = path.join(srcDir, "core", "model.ts");
// The one and only file permitted to construct OpenAIEmbeddings directly.
const ALLOWED_EMBEDDINGS_FACTORY = path.join(srcDir, "core", "embeddings.ts");

// Matches a direct construction `new ChatOpenAI(` allowing arbitrary whitespace
// (including newlines) between the keyword, the class name, and the paren.
const CHAT_OPENAI_CONSTRUCTION = /new\s+ChatOpenAI\s*\(/;
// Same shape for OpenAIEmbeddings — the embeddings equivalent of the rule above.
const OPENAI_EMBEDDINGS_CONSTRUCTION = /new\s+OpenAIEmbeddings\s*\(/;

// Recursively collect every *.ts file under a directory.
async function collectTsFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) return collectTsFiles(full);
      if (entry.isFile() && entry.name.endsWith(".ts")) return [full];
      return [];
    })
  );
  return files.flat();
}

// Remove comments so a `new ChatOpenAI(` appearing only inside a comment/doc
// (like this test's own references) never triggers a false positive. Handles
// block comments (/* */), line comments (//), and skips string/template
// literals so a `//` inside a URL string is not mistaken for a comment start.
function stripComments(source: string): string {
  let out = "";
  let i = 0;
  const n = source.length;
  type Mode = "code" | "line" | "block" | "single" | "double" | "template";
  let mode: Mode = "code";

  while (i < n) {
    const ch = source[i]!;
    const next = i + 1 < n ? source[i + 1]! : "";

    switch (mode) {
      case "code":
        if (ch === "/" && next === "/") {
          mode = "line";
          i += 2;
        } else if (ch === "/" && next === "*") {
          mode = "block";
          i += 2;
        } else if (ch === "'") {
          mode = "single";
          out += ch;
          i += 1;
        } else if (ch === '"') {
          mode = "double";
          out += ch;
          i += 1;
        } else if (ch === "`") {
          mode = "template";
          out += ch;
          i += 1;
        } else {
          out += ch;
          i += 1;
        }
        break;
      case "line":
        if (ch === "\n") {
          mode = "code";
          out += ch;
        }
        i += 1;
        break;
      case "block":
        if (ch === "*" && next === "/") {
          mode = "code";
          i += 2;
        } else {
          i += 1;
        }
        break;
      case "single":
      case "double":
      case "template": {
        const quote =
          mode === "single" ? "'" : mode === "double" ? '"' : "`";
        if (ch === "\\") {
          // Preserve escaped char pairs verbatim.
          out += ch + next;
          i += 2;
        } else if (ch === quote) {
          mode = "code";
          out += ch;
          i += 1;
        } else {
          out += ch;
          i += 1;
        }
        break;
      }
    }
  }
  return out;
}

// Scan every src/ *.ts file (comment/string-stripped) for a direct construction
// matched by `pattern`, excluding the single allowed factory file. Returns the
// repo-relative paths of any offenders.
async function findOffenders(
  pattern: RegExp,
  allowedFactory: string
): Promise<string[]> {
  const files = await collectTsFiles(srcDir);
  // Sanity: the scan actually found sources (guards against a broken path
  // silently passing the assertion).
  expect(files.length).toBeGreaterThan(0);

  const offenders: string[] = [];
  for (const file of files) {
    if (path.resolve(file) === path.resolve(allowedFactory)) continue;
    const raw = await readFile(file, "utf8");
    const code = stripComments(raw);
    if (pattern.test(code)) {
      offenders.push(path.relative(repoRoot, file));
    }
  }
  return offenders;
}

describe("Phase 4 — model factory convention guard", () => {
  it("no source file except src/core/model.ts constructs ChatOpenAI directly", async () => {
    const offenders = await findOffenders(
      CHAT_OPENAI_CONSTRUCTION,
      ALLOWED_FACTORY
    );

    expect(
      offenders,
      offenders.length === 0
        ? ""
        : `Direct \`new ChatOpenAI(\` is only allowed in src/core/model.ts. ` +
            `All application model construction must go through createChatModel(). ` +
            `Offending file(s): ${offenders.join(", ")}`
    ).toEqual([]);
  });

  it("src/core/model.ts (the allowed factory) does construct ChatOpenAI", async () => {
    // Positive control: ensures the scan target and regex stay meaningful — if
    // the factory stops constructing ChatOpenAI, this convention no longer maps
    // to reality and the guard should be revisited.
    const raw = await readFile(ALLOWED_FACTORY, "utf8");
    expect(CHAT_OPENAI_CONSTRUCTION.test(stripComments(raw))).toBe(true);
  });

  it("no source file except src/core/embeddings.ts constructs OpenAIEmbeddings directly", async () => {
    const offenders = await findOffenders(
      OPENAI_EMBEDDINGS_CONSTRUCTION,
      ALLOWED_EMBEDDINGS_FACTORY
    );

    expect(
      offenders,
      offenders.length === 0
        ? ""
        : `Direct \`new OpenAIEmbeddings(\` is only allowed in src/core/embeddings.ts. ` +
            `All embeddings construction must go through createEmbeddingsModel(). ` +
            `Offending file(s): ${offenders.join(", ")}`
    ).toEqual([]);
  });

  it("src/core/embeddings.ts (the allowed factory) does construct OpenAIEmbeddings", async () => {
    // Positive control mirroring the ChatOpenAI one: if the embeddings factory
    // stops constructing OpenAIEmbeddings, this guard no longer maps to reality.
    const raw = await readFile(ALLOWED_EMBEDDINGS_FACTORY, "utf8");
    expect(OPENAI_EMBEDDINGS_CONSTRUCTION.test(stripComments(raw))).toBe(true);
  });
});
