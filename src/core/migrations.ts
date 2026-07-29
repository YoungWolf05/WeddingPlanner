// TYPE-ONLY import: brings in better-sqlite3's real type surface (Database)
// without emitting any runtime `require("better-sqlite3")`. The `import type`
// form is erased at compile time, so the "construct the driver only via
// SqliteSaver in memory.ts" convention (and the 4b/4c single-factory guards) is
// unaffected — this module never opens a connection. It only operates on a
// better-sqlite3 handle that memory.ts already opened (saver.db), consistent
// with the shared-connection model from 5b.
import type { Database as BetterSqlite3Database } from "better-sqlite3";

// Phase 5 (5e): APP-OWNED schema versioning + forward-migration runner.
//
// WHY THIS EXISTS
// ---------------
// The Phase 5 durable service persists two DISTINCT schema families into ONE
// shared SQLite file (see memory.ts / threads.ts):
//   1. LIBRARY-OWNED tables — `checkpoints` / `writes` — created and evolved by
//      @langchain/langgraph-checkpoint-sqlite (SqliteSaver.setup()). These are
//      NOT our concern here; the library owns their DDL and any future
//      migration of them, and version drift is caught separately by the
//      library-version-pin coupling guard (phase5-checkpoint-schema-coupling).
//   2. APP-OWNED tables — currently `threads` (+ its owner index), plus this
//      module's own `app_schema_migrations` bookkeeping table, and any future
//      app tables/columns. THESE are what this migration runner manages.
//
// This module replaces the previous bare `CREATE TABLE IF NOT EXISTS` approach
// for the app schema with a real, minimal, versioned FORWARD-migration runner so
// the application's own tables evolve deterministically and safely across
// releases (Phase 5 exit criterion (3): schema-migration behavior implemented
// and tested).
//
// APP/LIBRARY BOUNDARY (critical on a shared file)
// ------------------------------------------------
// The runner MUST NOT create, alter, or migrate the LangGraph-owned
// `checkpoints` / `writes` tables. Migration `up()` steps touch only app tables.
// threads.ts still calls `ensureCheckpointTables()` separately (so the app's
// atomic delete/prune DELETEs can be prepared before the library lazily runs
// setup()); that mirroring is intentionally OUTSIDE this runner and remains the
// library's responsibility to define.
//
// VERSION MARKER DECISION
// -----------------------
// We record the app schema version in a DEDICATED app-owned table,
// `app_schema_migrations`, NOT in `PRAGMA user_version`.
//
// Verification: the installed @langchain/langgraph-checkpoint-sqlite@1.0.3
// (node_modules/.../dist/index.js, SqliteSaver.setup()) sets ONLY
// `journal_mode=WAL` and creates the two checkpoint tables — it does NOT touch
// `PRAGMA user_version`. So user_version is technically free today.
//
// Rationale for STILL choosing a dedicated table: the file is SHARED with the
// library, and `user_version` is a single global slot per database. A future
// library version could start using it, silently colliding with our app version
// and corrupting migration decisions. A dedicated, app-namespaced table is
// collision-proof by construction and additionally yields an append-only audit
// trail (one row per applied migration, with an applied-at timestamp). A guard
// test also asserts the library still does not set user_version, so drift is
// caught either way.

// The concrete better-sqlite3 Database type, sourced from @types/better-sqlite3
// via the type-only import above. Same aliasing pattern as threads.ts.
type Db = BetterSqlite3Database;

// APP-OWNED bookkeeping table. Namespaced (`app_`) so it can never collide with
// the library's `checkpoints` / `writes` tables in the shared file. Append-only:
// one row per successfully-applied migration; the current app schema version is
// MAX(version), and 0 means "no app migrations applied yet" (fresh/legacy file).
export const MIGRATIONS_TABLE = "app_schema_migrations";

// A single forward migration step. `version` is the target app schema version
// this step brings the database TO (1, 2, 3, ...). `up(db)` performs the DDL for
// that step and must be idempotent-safe within its own transaction (it runs only
// when the recorded version is strictly below `version`). There is deliberately
// NO `down()`: this is a FORWARD-ONLY runner (see runMigrations).
export interface Migration {
  version: number;
  up(db: Db): void;
}

// The real, ordered app migration list. APPEND new migrations with the next
// integer version; never renumber or mutate a released migration's `up()`.
//
// Migration 1 creates the `threads` ownership table and its owner index (the DDL
// formerly inlined as threads.ts `setupThreadsSchema`). Behavior is identical to
// the previous CREATE-IF-NOT-EXISTS setup, so all existing thread/ownership/
// retention/atomic-delete behavior is preserved after this step runs.
export const APP_MIGRATIONS: Migration[] = [
  {
    version: 1,
    up(db) {
      db.exec(`
CREATE TABLE IF NOT EXISTS threads (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  title TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);`);
      // Index supports list-by-owner (and ownership-scoped lookups). Named to
      // avoid collisions with any library indexes.
      db.exec(`
CREATE INDEX IF NOT EXISTS idx_threads_owner_id
  ON threads (owner_id, updated_at DESC);`);
    },
  },
];

// The highest version this build of the code knows how to migrate to. A database
// whose recorded version exceeds this was written by a NEWER app and must not be
// operated on (see the forward-only guard in runMigrations).
export const LATEST_APP_SCHEMA_VERSION = APP_MIGRATIONS.reduce(
  (max, m) => Math.max(max, m.version),
  0
);

// Idempotently ensure the bookkeeping table exists. Uses CREATE TABLE IF NOT
// EXISTS so it is safe to call on every startup and never destroys history.
function ensureMigrationsTable(db: Db): void {
  db.exec(`
CREATE TABLE IF NOT EXISTS ${MIGRATIONS_TABLE} (
  version INTEGER PRIMARY KEY,
  applied_at INTEGER NOT NULL
);`);
}

// Read the current app schema version WITHOUT any side effect: if the
// bookkeeping table does not exist yet (brand-new or pre-5e file), the version
// is 0. Otherwise it is MAX(version) over applied migrations (NULL -> 0 when the
// table exists but is empty). Exported so tests (and operators) can assert the
// recorded version directly.
export function getAppSchemaVersion(db: Db): number {
  const tableExists = db
    .prepare(
      `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`
    )
    .get(MIGRATIONS_TABLE);
  if (!tableExists) return 0;
  const row = db
    .prepare(`SELECT MAX(version) AS v FROM ${MIGRATIONS_TABLE}`)
    .get() as { v: number | null };
  return row.v ?? 0;
}

// Validate a migration list is well-formed: positive integer versions, strictly
// increasing after sort (no duplicates). Returns the sorted copy. A malformed
// list is a programming error, so we fail loudly rather than guess an order.
function validateAndSort(migrations: Migration[]): Migration[] {
  const sorted = [...migrations].sort((a, b) => a.version - b.version);
  let prev = 0;
  for (const m of sorted) {
    if (!Number.isInteger(m.version) || m.version <= 0) {
      throw new Error(
        `Invalid migration version ${String(
          m.version
        )}: versions must be positive integers.`
      );
    }
    if (m.version === prev) {
      throw new Error(
        `Duplicate migration version ${m.version}: each version must be unique.`
      );
    }
    prev = m.version;
  }
  return sorted;
}

// Apply all pending forward migrations to `db`.
//
// Contract:
//   - READS the recorded app schema version (absent table => 0).
//   - FORWARD-ONLY SAFETY: if the recorded version is GREATER than the highest
//     version in `migrations` (a DB written by a newer app), throws a loud error
//     and mutates NOTHING — we refuse to operate on an unknown-newer schema.
//   - Applies every migration whose `version` is strictly greater than the
//     current version, in ascending order, EACH inside its own
//     db.transaction(): the step's `up()` DDL and the version-record INSERT
//     commit together, so a throwing `up()` rolls the step back atomically (no
//     partial DDL, recorded version NOT advanced) and the error surfaces.
//   - IDEMPOTENT: on an up-to-date DB, nothing runs.
//   - DATA-PRESERVING: steps use additive DDL (CREATE ... IF NOT EXISTS / ALTER
//     ADD COLUMN); the runner never drops/recreates app tables, so existing rows
//     survive a migration.
//
// The runner is deliberately injectable (it takes `migrations` as a parameter)
// so tests can drive multi-step application, the forward-only guard, and
// rollback with a SYNTHETIC list, without needing a real future migration.
export function runMigrations(db: Db, migrations: Migration[]): void {
  const sorted = validateAndSort(migrations);
  const latest = sorted.reduce((max, m) => Math.max(max, m.version), 0);

  ensureMigrationsTable(db);
  const current = getAppSchemaVersion(db);

  // FORWARD-ONLY guard: a DB recorded as newer than this code understands must
  // not be silently operated on. Fail loudly before any migration runs.
  if (current > latest) {
    throw new Error(
      `App schema version ${current} is newer than this application supports ` +
        `(latest known migration: ${latest}). The database was written by a ` +
        `newer version of the app. Refusing to run forward-only migrations on ` +
        `an unknown schema — upgrade the application or use a compatible database.`
    );
  }

  for (const migration of sorted) {
    if (migration.version <= current) continue;
    // Each step is atomic: SQLite DDL is transactional, so if up() throws, the
    // CREATE/ALTER and the bookkeeping INSERT both roll back together and the
    // recorded version is not advanced.
    db.transaction(() => {
      migration.up(db);
      db.prepare(
        `INSERT INTO ${MIGRATIONS_TABLE} (version, applied_at) VALUES (?, ?)`
      ).run(migration.version, Date.now());
    })();
  }
}
