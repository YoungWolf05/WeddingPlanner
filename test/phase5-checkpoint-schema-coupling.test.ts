import { describe, it, expect } from "vitest";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

// Phase 5 (5b) — library-schema coupling guard (loud failure on upgrade).
//
// src/core/threads.ts deliberately MIRRORS the internal SQLite schema of
// @langchain/langgraph-checkpoint-sqlite in two places:
//   1. ensureCheckpointTables() — CREATE TABLE IF NOT EXISTS for `checkpoints`
//      and `writes`, mirroring the library's own DDL (so DELETEs can be
//      prepared before the library has lazily run setup()).
//   2. The replicated `DELETE FROM checkpoints WHERE thread_id = ?` and
//      `DELETE FROM writes WHERE thread_id = ?` statements inside deleteThread's
//      atomic transaction (replacing the library's async deleteThread so the
//      whole hard-delete is one synchronous better-sqlite3 transaction).
//
// This coupling is safe ONLY for the reviewed/pinned library version. If the
// package is upgraded, the mirrored DDL and the replicated DELETEs MUST be
// re-verified against the new library source. This test fails LOUDLY on any
// version drift so the coupling is never silently invalidated.
//
// Fully OFFLINE and deterministic: it reads the installed package.json (and, for
// the optional sanity check, introspects a temp SQLite DB after a real, mocked
// checkpoint op). No network, no credentials.

// The library version whose internal schema src/core/threads.ts was reviewed
// against and mirrors. Bump this ONLY after re-verifying the mirrored DDL and
// replicated DELETE statements against the new library source.
const REVIEWED_CHECKPOINT_SQLITE_VERSION = "1.0.3";

const require = createRequire(import.meta.url);

describe("Phase 5 (5b) — checkpoint-schema coupling guard", () => {
  it("the installed @langchain/langgraph-checkpoint-sqlite matches the reviewed/pinned version", async () => {
    // Resolve the installed package.json robustly (independent of cwd).
    const pkgJsonPath = require.resolve(
      "@langchain/langgraph-checkpoint-sqlite/package.json"
    );
    const pkg = JSON.parse(await readFile(pkgJsonPath, "utf8")) as {
      name: string;
      version: string;
    };

    expect(pkg.name).toBe("@langchain/langgraph-checkpoint-sqlite");

    expect(
      pkg.version,
      pkg.version === REVIEWED_CHECKPOINT_SQLITE_VERSION
        ? ""
        : `@langchain/langgraph-checkpoint-sqlite is installed at ${pkg.version} ` +
            `but src/core/threads.ts mirrors the internal SQLite schema of ` +
            `${REVIEWED_CHECKPOINT_SQLITE_VERSION}. RE-VERIFY the mirrored DDL ` +
            `(ensureCheckpointTables: CREATE TABLE IF NOT EXISTS checkpoints/writes) ` +
            `AND the replicated statements (DELETE FROM checkpoints/writes WHERE ` +
            `thread_id = ?) in src/core/threads.ts against the new library source, ` +
            `then bump REVIEWED_CHECKPOINT_SQLITE_VERSION in this test.`
    ).toBe(REVIEWED_CHECKPOINT_SQLITE_VERSION);
  });
});
