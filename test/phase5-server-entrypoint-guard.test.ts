import { describe, it, expect } from "vitest";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

// Phase 5 (5c) — structural guarantee: the LISTENING entrypoint
// src/run-server.ts must NEVER be imported by the offline test suite.
//
// run-server.ts binds a TCP port and starts listening on import (its top-level
// `main()` runs immediately). If any test imported it, `npm test` would open a
// real socket (and attempt to wire the shared default SQLite saver + real
// createConversationalChain), breaking the "fully offline, no repo artifacts"
// guarantee. Tests exercise the service by importing src/core/server.ts and
// starting their own server on an ephemeral port with injected deps.
//
// This is a DETERMINISTIC static scan: it reads every test source and fails if
// any of them references run-server. It imports no application module and makes
// no network call.

const testDir = path.dirname(fileURLToPath(import.meta.url));

async function collectTestFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) return collectTestFiles(full);
      if (entry.isFile() && entry.name.endsWith(".ts")) return [full];
      return [];
    })
  );
  return files.flat();
}

describe("Phase 5 (5c) — run-server entrypoint is not imported by tests", () => {
  it("no test file imports src/run-server.ts", async () => {
    const files = await collectTestFiles(testDir);
    expect(files.length).toBeGreaterThan(0);

    const offenders: string[] = [];
    for (const file of files) {
      const raw = await readFile(file, "utf8");
      // Any reference to the run-server module specifier in an import/require.
      if (/run-server(\.js)?["']/.test(raw)) {
        offenders.push(path.relative(testDir, file));
      }
    }

    expect(
      offenders,
      offenders.length === 0
        ? ""
        : `The listening entrypoint src/run-server.ts must not be imported by ` +
            `tests (it binds a port on import). Offending test file(s): ` +
            offenders.join(", ")
    ).toEqual([]);
  });
});
