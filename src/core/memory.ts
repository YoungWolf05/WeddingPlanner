import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { type BaseCheckpointSaver } from "@langchain/langgraph";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { config } from "../config.js";

// Phase 5 (5a): durable checkpointer. Conversation state is persisted to a local
// SQLite database via the official LangGraph SqliteSaver, so a thread's history
// survives a process restart (unlike the Phase 2 in-RAM MemorySaver, which reset
// on exit).
//
// Swappability: callers (chain.ts) depend only on the `BaseCheckpointSaver`
// interface exported here — never on SqliteSaver directly — so a future
// PostgreSQL swap (Phase 5 later / Phase 10) is a localized change in this
// factory, not a change to the consuming graph. `sessionConfig` (thread_id
// keying) is unchanged.

// Default database path when CHECKPOINT_DB_PATH is unset. Relative to the
// process working directory; the parent directory is created on first use.
const DEFAULT_CHECKPOINT_DB_PATH = "./data/checkpoints.sqlite";

// Resolve the configured/default database path to an absolute path. Kept pure so
// tests can assert default-vs-provided resolution without touching the disk.
export function resolveCheckpointDbPath(dbPath?: string): string {
  return resolve(dbPath ?? config.checkpointDbPath ?? DEFAULT_CHECKPOINT_DB_PATH);
}

// Build a durable checkpointer backed by SQLite at `dbPath` (or the
// configured/default path when omitted). The parent directory is created if
// missing — better-sqlite3 will not open a database in a nonexistent directory.
//
// Lifecycle: SqliteSaver opens a better-sqlite3 handle immediately. Long-lived
// application processes keep the shared saver (see getCheckpointer) open for
// their lifetime (the OS reclaims the handle on exit). Callers that create
// short-lived instances (e.g. tests) should close the handle via
// `saver.db.close()` to avoid leaking file handles / locking the file on Windows.
export function createCheckpointer(dbPath?: string): BaseCheckpointSaver {
  const absolutePath = resolveCheckpointDbPath(dbPath);
  mkdirSync(dirname(absolutePath), { recursive: true });
  // SqliteSaver.setup() creates the schema lazily on first operation (read or
  // write), so no explicit migration step is required here.
  return SqliteSaver.fromConnString(absolutePath);
}

// The shared, long-lived checkpointer for the application graph, built lazily
// from the configured/default path. Consumers call this and remain agnostic to
// the concrete saver implementation.
//
// LAZINESS IS DELIBERATE: constructing the saver opens (and creates) the SQLite
// file on disk. Deferring construction until first use means merely IMPORTING
// this module has no filesystem side effect, so the offline test suite can
// import it (and inject isolated temp-file savers) without ever creating the
// default ./data database in the repo. Real entrypoints trigger construction on
// their first call and then reuse the single shared instance.
let sharedCheckpointer: BaseCheckpointSaver | undefined;

export function getCheckpointer(): BaseCheckpointSaver {
  sharedCheckpointer ??= createCheckpointer();
  return sharedCheckpointer;
}

// Scopes a conversation to one session; pass as the second arg to invoke/stream.
export function sessionConfig(sessionId: string) {
  return { configurable: { thread_id: sessionId } };
}
