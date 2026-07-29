import { randomUUID } from "node:crypto";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
// TYPE-ONLY import: brings in better-sqlite3's real type surface (Database,
// Statement, RunResult) without emitting any runtime `require("better-sqlite3")`.
// The `import type` form is erased at compile time, so the "construct the driver
// only via SqliteSaver in memory.ts" convention (and the 4b/4c single-factory
// guards) is unaffected — this module never opens a connection.
import type { Database as BetterSqlite3Database } from "better-sqlite3";
// Phase 5 (5e): app-owned schema versioning + forward-migration runner. The
// `threads` table + owner index are created by APP_MIGRATIONS (migration 1),
// NOT by an inline CREATE-IF-NOT-EXISTS, so the app schema evolves through a
// versioned, transactional, forward-only runner. This manages ONLY app tables;
// the library's checkpoint tables remain the library's responsibility (see
// ensureCheckpointTables below and the version-pin coupling guard).
import { APP_MIGRATIONS, runMigrations } from "./migrations.js";

// Phase 5 (5b): thread identity & ownership MODEL with enforcement.
//
// This module adds an ownership layer ON TOP of the durable checkpointer from
// 5a. It does NOT change how conversations are stored or retrieved — LangGraph's
// checkpoint tables (checkpoints, writes) remain owned by the SqliteSaver. Here
// we add a DEDICATED `threads` table, in the SAME SQLite database file, that
// records who owns each conversation thread.
//
// SECURITY MODEL (the core of this increment):
//   - `thread_id` is a CONVERSATION KEY, never identity or authorization. It is
//     a server-issued UUID (node:crypto randomUUID) and is deliberately NOT
//     derived from the owner, so it cannot be forged or guessed from an owner id.
//   - Every read/delete is scoped by `owner_id`. A non-owner is denied.
//   - NO EXISTENCE LEAK: a thread that exists but is owned by someone else is
//     indistinguishable from one that does not exist (getThread -> null,
//     deleteThread -> false). We never reveal that a thread_id is "taken".
//
// TRANSPORT / AUTH ARE OUT OF SCOPE (5c): this module takes an already-resolved
// `ownerId` from its caller. It does not parse tokens, headers, or HTTP. How the
// caller authenticates a principal and maps it to an `ownerId` is 5c's concern.
//
// STORAGE SWAPPABILITY: consumers depend on the `ThreadStore` interface and the
// `Thread` record type, never on SQLite specifics. A future PostgreSQL backend
// (Phase 10) implements the same interface; only this factory changes.

// A thread-ownership record. `id` is the conversation key (thread_id) used with
// the checkpointer's `sessionConfig(id)`. Timestamps are epoch milliseconds
// (INTEGER) for compact, unambiguous ordering.
export interface Thread {
  id: string;
  ownerId: string;
  title: string | null;
  createdAt: number;
  updatedAt: number;
}

// Storage-agnostic ownership data-access contract. Every method is scoped by the
// caller's `ownerId`; ownership is enforced inside each method, not by callers.
export interface ThreadStore {
  // Create a new thread owned by `ownerId`. Generates a server-issued UUID for
  // the thread_id; the id is NOT derived from ownerId. Returns the new record.
  createThread(ownerId: string, opts?: { title?: string }): Thread;
  // List only `ownerId`'s threads, most-recently-updated first.
  listThreads(ownerId: string): Thread[];
  // Return the thread ONLY if owned by `ownerId`; otherwise null. A thread owned
  // by someone else returns null (indistinguishable from not-found).
  getThread(ownerId: string, threadId: string): Thread | null;
  // Update `updatedAt` for an owned thread. Returns true if a row was touched,
  // false if not found / not owned.
  touchThread(ownerId: string, threadId: string): boolean;
  // HARD delete: remove the ownership row AND the LangGraph checkpoint state for
  // this thread_id ATOMICALLY (one transaction), ONLY if owned by `ownerId`.
  // Returns true if a thread was deleted, false if not found / not owned. A
  // non-owner delete changes NOTHING (neither the ownership row nor checkpoints).
  deleteThread(ownerId: string, threadId: string): boolean;
  // Phase 5 (5d): RETENTION policy HOOK (not a scheduler). Hard-delete every
  // thread whose `updatedAt` is strictly older than `policy.olderThanEpochMs`,
  // reusing the SAME atomic ownership-row + checkpoint-state delete semantics as
  // deleteThread (no orphaned checkpoints/writes). When `policy.ownerId` is set,
  // only that owner's threads are considered (owner-scoped purge); when omitted,
  // all owners' expired threads are pruned (operator/global purge). Returns the
  // number of threads deleted. This is a CALLABLE hook — scheduling and the
  // operational purge cadence are a Phase 10 concern, deliberately out of scope.
  pruneThreads(policy: RetentionPolicy): number;
}

// Input to the retention hook. `olderThanEpochMs` is an absolute epoch-ms cutoff
// (a thread is pruned when updatedAt < cutoff); callers compute it from a
// retention window (e.g. Date.now() - retentionMs). `ownerId`, when present,
// scopes the purge to a single owner.
export interface RetentionPolicy {
  olderThanEpochMs: number;
  ownerId?: string;
}

// The concrete better-sqlite3 Database type, sourced from @types/better-sqlite3
// via the type-only import above. This gives us a GENUINELY typed handle:
// db.prepare(...) returns a typed Statement, and Statement.run(...) returns a
// RunResult whose `changes: number` powers the security-critical ownership gate
// in deleteThread. (SqliteSaver.db is typed as this same Database in the
// checkpoint library, so the store and the saver share one typed connection.)
type Db = BetterSqlite3Database;

// Dedicated ownership table. Name chosen to NOT collide with the library's
// `checkpoints` / `writes` tables (which live in the same file). The table (and
// its owner index) are created by APP_MIGRATIONS migration 1 via runMigrations
// at store construction; this constant is used by the prepared statements below.
const THREADS_TABLE = "threads";

// Shape of a raw row as returned by better-sqlite3 for the threads table.
interface ThreadRow {
  id: string;
  owner_id: string;
  title: string | null;
  created_at: number;
  updated_at: number;
}

function rowToThread(row: ThreadRow): Thread {
  return {
    id: row.id,
    ownerId: row.owner_id,
    title: row.title,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

// Idempotently ensure the library's checkpoint tables exist. SqliteSaver.setup()
// is protected and only runs lazily on the first checkpoint operation, so on a
// brand-new file the `checkpoints` / `writes` tables may not exist yet. Our
// hard-delete replicates the library's two DELETEs, and better-sqlite3 compiles
// (prepares) `DELETE FROM <table>` eagerly — which errors if the table is
// absent. We therefore create these tables up front with CREATE TABLE IF NOT
// EXISTS, mirroring the installed library's DDL EXACTLY so we never conflict with
// its own setup() (only the `thread_id` column, which our DELETEs target, is
// load-bearing). This keeps us off the library's private members.
function ensureCheckpointTables(db: Db): void {
  db.exec(`
CREATE TABLE IF NOT EXISTS checkpoints (
  thread_id TEXT NOT NULL,
  checkpoint_ns TEXT NOT NULL DEFAULT '',
  checkpoint_id TEXT NOT NULL,
  parent_checkpoint_id TEXT,
  type TEXT,
  checkpoint BLOB,
  metadata BLOB,
  PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);`);
  db.exec(`
CREATE TABLE IF NOT EXISTS writes (
  thread_id TEXT NOT NULL,
  checkpoint_ns TEXT NOT NULL DEFAULT '',
  checkpoint_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  channel TEXT NOT NULL,
  type TEXT,
  value BLOB,
  PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);`);
}

// Build a thread-ownership store over the SAME SQLite database the checkpointer
// uses. We take the SqliteSaver (not a path) so the store and the checkpointer
// SHARE ONE better-sqlite3 connection — this is what makes the hard-delete
// atomic (a single transaction spans both the ownership row and the checkpoint
// tables).
//
export function createThreadStore(saver: SqliteSaver): ThreadStore {
  const db = saver.db;
  // Bring the APP-OWNED schema up to date via the versioned, forward-only
  // migration runner (creates the `threads` table + owner index at migration 1,
  // records the app schema version in app_schema_migrations, and fails loudly if
  // the DB was written by a newer app). This manages ONLY app tables. We then
  // ensure the LIBRARY-OWNED checkpoint tables exist (see ensureCheckpointTables
  // — deliberately OUTSIDE the app migration runner) BEFORE preparing any
  // statements, because better-sqlite3 compiles DELETE statements eagerly and
  // would throw on a missing table.
  runMigrations(db, APP_MIGRATIONS);
  ensureCheckpointTables(db);

  // Prepared statements (compiled once). All ownership-sensitive reads/writes
  // carry `owner_id` in the WHERE clause so ownership is enforced in SQL.
  const insertStmt = db.prepare(
    `INSERT INTO ${THREADS_TABLE} (id, owner_id, title, created_at, updated_at)
     VALUES (@id, @owner_id, @title, @created_at, @updated_at)`
  );
  // Read statements are typed at prepare time: the generic <BindParameters,
  // Result> makes .all()/.get() return ThreadRow (no `as` cast needed). The row
  // shape is asserted here, at the single SELECT site, rather than at each call.
  const listStmt = db.prepare<[ownerId: string], ThreadRow>(
    `SELECT id, owner_id, title, created_at, updated_at
       FROM ${THREADS_TABLE}
      WHERE owner_id = ?
      ORDER BY updated_at DESC, created_at DESC, id DESC`
  );
  const getStmt = db.prepare<[ownerId: string, id: string], ThreadRow>(
    `SELECT id, owner_id, title, created_at, updated_at
       FROM ${THREADS_TABLE}
      WHERE owner_id = ? AND id = ?`
  );
  const touchStmt = db.prepare(
    `UPDATE ${THREADS_TABLE}
        SET updated_at = @updated_at
      WHERE owner_id = @owner_id AND id = @id`
  );
  const deleteOwnedThreadStmt = db.prepare(
    `DELETE FROM ${THREADS_TABLE} WHERE owner_id = ? AND id = ?`
  );
  // Replicated LangGraph checkpoint deletes. These mirror SqliteSaver.deleteThread
  // EXACTLY (same two statements, same predicate). See the atomicity note below
  // for why we replicate rather than call the async library method.
  const deleteCheckpointsStmt = db.prepare(
    `DELETE FROM checkpoints WHERE thread_id = ?`
  );
  const deleteWritesStmt = db.prepare(`DELETE FROM writes WHERE thread_id = ?`);

  // Phase 5 (5d) retention: select expired threads (updatedAt strictly older than
  // the cutoff), optionally scoped to one owner. We SELECT the (id, owner_id)
  // pairs first, then delete each via the SAME atomic path as deleteThread so the
  // ownership row and checkpoint state always vanish together. Typed at prepare
  // time so .all(...) returns the id/owner rows without a cast.
  const selectExpiredAllStmt = db.prepare<
    [cutoff: number],
    { id: string; owner_id: string }
  >(`SELECT id, owner_id FROM ${THREADS_TABLE} WHERE updated_at < ?`);
  const selectExpiredByOwnerStmt = db.prepare<
    [cutoff: number, owner: string],
    { id: string; owner_id: string }
  >(
    `SELECT id, owner_id FROM ${THREADS_TABLE}
      WHERE updated_at < ? AND owner_id = ?`
  );

  return {
    createThread(ownerId, opts) {
      const now = Date.now();
      const thread: Thread = {
        id: randomUUID(),
        ownerId,
        title: opts?.title ?? null,
        createdAt: now,
        updatedAt: now,
      };
      insertStmt.run({
        id: thread.id,
        owner_id: thread.ownerId,
        title: thread.title,
        created_at: thread.createdAt,
        updated_at: thread.updatedAt,
      });
      return thread;
    },

    listThreads(ownerId) {
      const rows = listStmt.all(ownerId);
      return rows.map(rowToThread);
    },

    getThread(ownerId, threadId) {
      const row = getStmt.get(ownerId, threadId);
      return row ? rowToThread(row) : null;
    },

    touchThread(ownerId, threadId) {
      const info = touchStmt.run({
        updated_at: Date.now(),
        owner_id: ownerId,
        id: threadId,
      });
      return info.changes > 0;
    },

    deleteThread(ownerId, threadId) {
      // ATOMIC HARD DELETE.
      //
      // Why replicate the library's two DELETEs instead of calling
      // saver.deleteThread(threadId)? SqliteSaver.deleteThread is declared
      // `async` (returns a Promise). better-sqlite3's db.transaction() runs a
      // SYNCHRONOUS function and commits when that function RETURNS; a returned
      // Promise would be treated as an opaque value and the transaction would
      // commit BEFORE the checkpoint deletes ran — breaking atomicity. The
      // library's method body is itself fully synchronous and is exactly these
      // two statements (verified against the installed version), so we run those
      // same statements together with the ownership-row delete inside ONE
      // db.transaction(). Both sides commit together or roll back together.
      const deleted = db.transaction(() => {
        const info = deleteOwnedThreadStmt.run(ownerId, threadId);
        // Ownership gate: if no ownership row matched (not found OR not owned by
        // this caller), do NOT touch the checkpoint state. A non-owner delete
        // must change nothing. Returning here still commits an empty transaction.
        if (info.changes === 0) return false;
        // Owned: remove the conversation state for this thread_id so no orphaned
        // checkpoint/writes rows remain.
        deleteCheckpointsStmt.run(threadId);
        deleteWritesStmt.run(threadId);
        return true;
      })();
      return deleted;
    },

    pruneThreads(policy) {
      const { olderThanEpochMs, ownerId } = policy;
      // Prune the whole matched set in ONE transaction so a batch purge is
      // all-or-nothing and each thread's ownership row + checkpoint state are
      // removed together (no orphaned checkpoints/writes). Reuses the same
      // prepared DELETE statements as deleteThread — the ownership row is deleted
      // with its OWNER in the predicate, so an owner-scoped prune can never touch
      // another owner's thread even if a stale id were somehow reused.
      const deleted = db.transaction(() => {
        const rows =
          ownerId === undefined
            ? selectExpiredAllStmt.all(olderThanEpochMs)
            : selectExpiredByOwnerStmt.all(olderThanEpochMs, ownerId);
        let count = 0;
        for (const { id, owner_id } of rows) {
          const info = deleteOwnedThreadStmt.run(owner_id, id);
          if (info.changes === 0) continue;
          deleteCheckpointsStmt.run(id);
          deleteWritesStmt.run(id);
          count += 1;
        }
        return count;
      })();
      return deleted;
    },
  };
}
