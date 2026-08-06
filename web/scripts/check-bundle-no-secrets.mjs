// Phase 9 (9c): the ZERO-CREDENTIALS-IN-BUNDLE guardrail (exit criterion 1).
//
// This is a HARD, TESTED guardrail: the BUILT browser bundle (web/dist) must
// contain ZERO provider credentials. This scanner walks every file under
// web/dist and FAILS (non-zero exit) if any forbidden pattern is found. It is
// deterministic (no network, no randomness) and is the concrete artifact 9d /
// Phase 9 closeout cite for "the browser contains no provider credentials".
//
// USAGE:
//   node scripts/check-bundle-no-secrets.mjs         (scan an existing web/dist)
//   npm run test:bundle                               (same)
//   npm run build:check                               (vite build, then scan)
//
// WHAT IT FORBIDS (and why):
//   - LITELLM_* env var NAMES  -> a provider config var must never be inlined.
//   - "sk-<...>" style API keys -> a leaked provider/OpenAI-style secret.
//   - apiKey / baseURL literals -> provider client config that must stay server
//                                  side (the browser only talks to the backend).
//   - a caller-supplied EXTRA needle (argv/env) -> e.g. the actual configured
//                                  proxy secret, so CI can assert THAT specific
//                                  value never leaks.
//
// It scans TEXT-LIKE build assets (.js/.mjs/.cjs/.css/.html/.json/.map/.txt).
// Source maps are included on purpose: a secret could hide in a .map even if the
// minified .js masked it.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const distDir = join(here, "..", "dist");

// Text-like extensions worth scanning. Binary assets (images/fonts) are skipped.
const TEXT_EXTS = new Set([
  ".js",
  ".mjs",
  ".cjs",
  ".css",
  ".html",
  ".json",
  ".map",
  ".txt",
  ".svg",
]);

// Forbidden patterns. Each entry: { name, regex } — a match is a FAILURE.
// NOTE: keep these as constructed RegExp (not literals containing the exact
// banned tokens verbatim in a way that could false-positive on this file — this
// scanner never scans itself; it only scans web/dist).
const FORBIDDEN = [
  {
    name: "LITELLM_ env var name",
    regex: /LITELLM_[A-Z_]+/,
  },
  {
    // OpenAI/provider-style secret keys: sk- followed by >=16 key chars.
    name: "sk-* style secret key",
    regex: /\bsk-[A-Za-z0-9_-]{16,}\b/,
  },
  {
    name: "apiKey literal",
    regex: /["']?apiKey["']?\s*[:=]\s*["'][^"']+["']/,
  },
  {
    name: "baseURL literal (provider)",
    // A baseURL string assignment pointing at a non-localhost http(s) origin.
    // The frontend only references a same-origin/relative backend base, so any
    // absolute provider baseURL literal is forbidden.
    regex: /["']?baseURL["']?\s*[:=]\s*["']https?:\/\/(?!localhost|127\.0\.0\.1)[^"']+["']/,
  },
];

function* walk(dir) {
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    return;
  }
  for (const entry of entries) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      yield* walk(full);
    } else if (st.isFile()) {
      yield full;
    }
  }
}

function collectExtraNeedles() {
  const needles = [];
  // From CLI args: node check-bundle-no-secrets.mjs <needle> [<needle>...]
  for (const arg of process.argv.slice(2)) {
    if (arg.trim() !== "") needles.push(arg.trim());
  }
  // From env: BUNDLE_FORBIDDEN_NEEDLES = comma-separated exact strings.
  const fromEnv = process.env.BUNDLE_FORBIDDEN_NEEDLES;
  if (fromEnv) {
    for (const part of fromEnv.split(",")) {
      const trimmed = part.trim();
      if (trimmed !== "") needles.push(trimmed);
    }
  }
  return needles;
}

function main() {
  // Confirm the build output exists (fail loudly rather than silently pass).
  let distStat;
  try {
    distStat = statSync(distDir);
  } catch {
    console.error(
      `[bundle-scan] FAIL: build output not found at ${distDir}. Run \`vite build\` first (or \`npm run build:check\`).`
    );
    process.exit(2);
  }
  if (!distStat.isDirectory()) {
    console.error(`[bundle-scan] FAIL: ${distDir} is not a directory.`);
    process.exit(2);
  }

  const extraNeedles = collectExtraNeedles();
  const findings = [];
  let scanned = 0;

  for (const file of walk(distDir)) {
    if (!TEXT_EXTS.has(extname(file).toLowerCase())) continue;
    scanned += 1;
    const content = readFileSync(file, "utf8");

    for (const { name, regex } of FORBIDDEN) {
      const m = content.match(regex);
      if (m) {
        findings.push({ file, pattern: name, sample: redactSample(m[0]) });
      }
    }
    for (const needle of extraNeedles) {
      if (content.includes(needle)) {
        findings.push({
          file,
          pattern: `caller-supplied needle`,
          sample: "<redacted needle match>",
        });
      }
    }
  }

  if (findings.length > 0) {
    console.error(
      `[bundle-scan] FAIL: found ${findings.length} forbidden pattern match(es) in ${scanned} scanned file(s):`
    );
    for (const f of findings) {
      console.error(`  - ${f.pattern} in ${f.file} :: ${f.sample}`);
    }
    process.exit(1);
  }

  console.log(
    `[bundle-scan] PASS: scanned ${scanned} file(s) under web/dist; no provider credentials or forbidden patterns found` +
      (extraNeedles.length > 0
        ? ` (incl. ${extraNeedles.length} caller-supplied needle(s)).`
        : ".")
  );
  process.exit(0);
}

// Redact all but a short prefix of a matched sample so the failure message never
// echoes a full secret into logs.
function redactSample(sample) {
  const head = sample.slice(0, 6);
  return `${head}…(${sample.length} chars, redacted)`;
}

main();
