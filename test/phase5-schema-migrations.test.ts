import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { HumanMessage } from "@langchain/core/messages";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import type { Database as BetterSqlite3Database } from "better-sqlite3";
// Static type-only import (erased at compile time; unaffected by vi.mock
// hoisting) for the synthetic migration shape used in the multi-step / rollback
// tests. The runtime values come from the dynamic import below.
import type { Migration } from "../src/core/migrations.js";

// Phase 5 (5e) — APP-OWNED schema versioning + forward-migration runner.
//
// DETERMINISTIC and fully OFFLINE:
//   - The model boundary is mocked (same fake-model approach as 5a/5b) so the
//     "library coexistence" test can seed real checkpoint state via a mocked
//     conversational turn with NO network/credentials.
//   - Every SQLite database lives in a per-test temp directory under os.tmpdir()
//     and is removed in afterEach, so `npm test` NEVER writes DB artifacts into
//     the repo working tree.
//
// These tests exercise BOTH the real APP_MIGRATIONS list (through
// createThreadStore, the production integration point) AND SYNTHETIC migration
// lists passed directly to runMigrations (to prove multi-step application, the
// forward-only guard, atomic rollback, and data preservation without needing a
// real future migration).
vi.mock("../src/core/model.js", async () => {
  const { makeFakeChatModel } = await import("./helpers/fake-model.js");
  return {
    createChatModel: () =>
      makeFakeChatModel({
        responses: ["reply-1", "reply-2", "reply-3", "reply-4"],
      }),
  };
});

const { createConversationalChain } = await import("../src/core/chain.js");
const { createCheckpointer, sessionConfig } = await import(
  "../src/core/memory.js"
);
const { createThreadStore } = await import("../src/core/threads.js");
const {
  runMigrations,
  getAppSchemaVersion,
  APP_MIGRATIONS,
  LATEST_APP_SCHEMA_VERSION,
  MIGRATIONS_TABLE,
} = await import("../src/core/migrations.js");

type Db = BetterSqlite3Database;

const OWNER_A = "owner-alice";
const OWNER_B = "owner-bob";

function asSqlite(saver: ReturnType<typeof createCheckpointer>): SqliteSaver {
  if (!(saver instanceof SqliteSaver)) {
    throw new Error("expected a SqliteSaver");
  }
  return saver;
}

// Introspection helpers over a raw better-sqlite3 handle.
function tableExists(db: Db, name: string): boolean {
  return (
    db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`
      )
      .get(name) !== undefined
  );
}

function indexExists(db: Db, name: string): boolean {
  return (
    db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?`
      )
      .get(name) !== undefined
  );
}

function columnsOf(db: Db, table: string): string[] {
  return (
    db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>
  ).map((c) => c.name);
}

describe("Phase 5 (5e) — app-owned schema versioning + forward migrations", () => {
  let tempDir: string;
  let dbPath: string;
  const openSavers: SqliteSaver[] = [];

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-migrations-"));
    dbPath = path.join(tempDir, "checkpoints.sqlite");
    openSavers.length = 0;
  });

  afterEach(async () => {
    for (const saver of openSavers) {
      try {
        saver.db.close();
      } catch {
        // already closed
      }
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  function makeSaver(p: string = dbPath): SqliteSaver {
    const saver = asSqlite(createCheckpointer(p));
    openSavers.push(saver);
    return saver;
  }

  it("FRESH DB: createThreadStore migrates to latest; threads table + owner index exist", () => {
    const saver = makeSaver();
    const db = saver.db;

    // Before any store: no app schema recorded (fresh file).
    expect(getAppSchemaVersion(db)).toBe(0);

    createThreadStore(saver);

    // After construction: threads table + its owner index exist exactly as before.
    expect(tableExists(db, "threads")).toBe(true);
    expect(indexExists(db, "idx_threads_owner_id")).toBe(true);
    expect(columnsOf(db, "threads").sort()).toEqual(
      ["created_at", "id", "owner_id", "title", "updated_at"].sort()
    );

    // Recorded app schema version equals the latest known migration target,
    // which equals the migration list length (contiguous from 1).
    expect(getAppSchemaVersion(db)).toBe(LATEST_APP_SCHEMA_VERSION);
    expect(LATEST_APP_SCHEMA_VERSION).toBe(APP_MIGRATIONS.length);

    // The bookkeeping table exists and holds one row per applied migration.
    expect(tableExists(db, MIGRATIONS_TABLE)).toBe(true);
    const rows = db
      .prepare(`SELECT version FROM ${MIGRATIONS_TABLE} ORDER BY version`)
      .all() as Array<{ version: number }>;
    expect(rows.map((r) => r.version)).toEqual(
      APP_MIGRATIONS.map((m) => m.version)
    );
  });

  it("IDEMPOTENCY: constructing the store twice / re-running the migrator is a no-op", () => {
    const saver = makeSaver();
    const db = saver.db;

    createThreadStore(saver);
    const versionAfterFirst = getAppSchemaVersion(db);
    const migrationRowsAfterFirst = db
      .prepare(`SELECT COUNT(*) AS n FROM ${MIGRATIONS_TABLE}`)
      .get() as { n: number };

    // Second store over the same connection: must not error, must not re-apply.
    const store2 = createThreadStore(saver);
    // Direct re-run of the migrator is also a no-op.
    runMigrations(db, APP_MIGRATIONS);

    expect(getAppSchemaVersion(db)).toBe(versionAfterFirst);
    const migrationRowsAfterSecond = db
      .prepare(`SELECT COUNT(*) AS n FROM ${MIGRATIONS_TABLE}`)
      .get() as { n: number };
    expect(migrationRowsAfterSecond.n).toBe(migrationRowsAfterFirst.n);

    // Table still intact and usable.
    expect(tableExists(db, "threads")).toBe(true);
    const t = store2.createThread(OWNER_A, { title: "still works" });
    expect(store2.getThread(OWNER_A, t.id)?.title).toBe("still works");
  });

  it("MULTI-STEP application: a synthetic [m1,m2,m3] list reaches version 3, each up() runs once in order", () => {
    const saver = makeSaver();
    const db = saver.db;

    const order: number[] = [];
    const synthetic: Migration[] = [
      {
        version: 1,
        up(d) {
          order.push(1);
          d.exec(`CREATE TABLE IF NOT EXISTS syn_one (id INTEGER PRIMARY KEY);`);
        },
      },
      {
        version: 2,
        up(d) {
          order.push(2);
          d.exec(`CREATE TABLE IF NOT EXISTS syn_two (id INTEGER PRIMARY KEY);`);
        },
      },
      {
        version: 3,
        up(d) {
          order.push(3);
          d.exec(
            `CREATE TABLE IF NOT EXISTS syn_three (id INTEGER PRIMARY KEY);`
          );
        },
      },
    ];

    expect(getAppSchemaVersion(db)).toBe(0);
    runMigrations(db, synthetic);

    expect(getAppSchemaVersion(db)).toBe(3);
    // Each up() ran exactly once, in ascending order.
    expect(order).toEqual([1, 2, 3]);
    expect(tableExists(db, "syn_one")).toBe(true);
    expect(tableExists(db, "syn_two")).toBe(true);
    expect(tableExists(db, "syn_three")).toBe(true);

    // From an intermediate version, only pending steps run.
    order.length = 0;
    const synthetic4: Migration[] = [
      ...synthetic,
      {
        version: 4,
        up(d) {
          order.push(4);
          d.exec(
            `CREATE TABLE IF NOT EXISTS syn_four (id INTEGER PRIMARY KEY);`
          );
        },
      },
    ];
    runMigrations(db, synthetic4);
    expect(getAppSchemaVersion(db)).toBe(4);
    // Only version 4 ran; 1-3 were skipped (already applied).
    expect(order).toEqual([4]);
    expect(tableExists(db, "syn_four")).toBe(true);
  });

  it("DATA PRESERVATION: rows seeded at v1 survive a synthetic additive migration", () => {
    const saver = makeSaver();
    const db = saver.db;

    // Seed real rows via the production store (which migrates to LATEST).
    const store = createThreadStore(saver);
    const t1 = store.createThread(OWNER_A, { title: "keep me" });
    const t2 = store.createThread(OWNER_B, { title: "and me" });

    const baseVersion = getAppSchemaVersion(db);

    // A synthetic FORWARD migration appended after the real ones: add a nullable
    // column to the existing `threads` table (additive, non-destructive).
    const nextVersion = baseVersion + 1;
    const synthetic: Migration[] = [
      ...APP_MIGRATIONS,
      {
        version: nextVersion,
        up(d) {
          d.exec(`ALTER TABLE threads ADD COLUMN archived INTEGER;`);
        },
      },
    ];

    runMigrations(db, synthetic);
    expect(getAppSchemaVersion(db)).toBe(nextVersion);

    // Existing rows are intact and reachable, and the new column is present.
    expect(columnsOf(db, "threads")).toContain("archived");
    expect(store.getThread(OWNER_A, t1.id)?.title).toBe("keep me");
    expect(store.getThread(OWNER_B, t2.id)?.title).toBe("and me");
    const all = db
      .prepare(`SELECT id, title, archived FROM threads ORDER BY title`)
      .all() as Array<{ id: string; title: string; archived: number | null }>;
    expect(all.map((r) => r.title)).toEqual(["and me", "keep me"]);
    // Pre-existing rows have NULL for the newly-added nullable column.
    expect(all.every((r) => r.archived === null)).toBe(true);
  });

  it("LEGACY UPGRADE (5a-5d -> 5e): a pre-5e DB with `threads` but no bookkeeping table upgrades non-destructively", () => {
    // The highest-value real-world case: a database created by 5a-5d, where the
    // `threads` table already exists (from the OLD `CREATE TABLE IF NOT EXISTS`
    // setup) but the `app_schema_migrations` bookkeeping table does NOT. Running
    // the 5e migrator against it must recognize it as version 0, apply migration
    // 1 (whose `up()` is `CREATE TABLE IF NOT EXISTS threads` — a no-op on the
    // existing table), record version 1, and PRESERVE all existing rows.
    const saver = makeSaver();
    const db = saver.db;

    // Simulate the legacy DB by creating the `threads` schema WITHOUT the
    // migrator: replay migration 1's DDL directly, which is byte-identical to
    // the old setupThreadsSchema. This creates `threads` + idx_threads_owner_id
    // but NOT app_schema_migrations (the pre-5e state).
    APP_MIGRATIONS[0]!.up(db);
    expect(tableExists(db, "threads")).toBe(true);
    expect(indexExists(db, "idx_threads_owner_id")).toBe(true);
    expect(tableExists(db, MIGRATIONS_TABLE)).toBe(false);

    // Seed a realistic ownership row directly (the store isn't set up yet).
    const now = Date.now();
    db.prepare(
      `INSERT INTO threads (id, owner_id, title, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?)`
    ).run("legacy-thread-1", OWNER_A, "legacy row", now, now);

    // Capture the exact table/index definitions so we can prove the upgrade did
    // NOT drop-and-recreate them (a destructive recreate would change `sql`).
    const schemaSqlBefore = db
      .prepare(
        `SELECT name, sql FROM sqlite_master
          WHERE name IN ('threads','idx_threads_owner_id') ORDER BY name`
      )
      .all();

    // Pre-5e: with no bookkeeping table, the recorded version is 0.
    expect(getAppSchemaVersion(db)).toBe(0);

    // The upgrade transition itself: must not throw.
    expect(() => runMigrations(db, APP_MIGRATIONS)).not.toThrow();

    // Seeded row survived (data preserved — proof of a non-destructive upgrade).
    const preserved = db
      .prepare(
        `SELECT id, owner_id, title FROM threads WHERE id = ?`
      )
      .get("legacy-thread-1") as
      | { id: string; owner_id: string; title: string }
      | undefined;
    expect(preserved).toEqual({
      id: "legacy-thread-1",
      owner_id: OWNER_A,
      title: "legacy row",
    });

    // Table + index still exist and are UNCHANGED (no destructive recreate).
    expect(tableExists(db, "threads")).toBe(true);
    expect(indexExists(db, "idx_threads_owner_id")).toBe(true);
    const schemaSqlAfter = db
      .prepare(
        `SELECT name, sql FROM sqlite_master
          WHERE name IN ('threads','idx_threads_owner_id') ORDER BY name`
      )
      .all();
    expect(schemaSqlAfter).toEqual(schemaSqlBefore);

    // Recorded version is now LATEST, and the bookkeeping table exists with the
    // expected applied-migration rows.
    expect(getAppSchemaVersion(db)).toBe(LATEST_APP_SCHEMA_VERSION);
    expect(tableExists(db, MIGRATIONS_TABLE)).toBe(true);
    const applied = db
      .prepare(`SELECT version FROM ${MIGRATIONS_TABLE} ORDER BY version`)
      .all() as Array<{ version: number }>;
    expect(applied.map((r) => r.version)).toEqual(
      APP_MIGRATIONS.map((m) => m.version)
    );

    // Coexistence sanity: the app version marker is a dedicated table, so the
    // library's user_version slot is untouched (still 0).
    expect(db.pragma("user_version", { simple: true })).toBe(0);

    // A subsequent store over the same DB treats it as up-to-date (idempotent
    // no-op) and sees the migrated legacy row via the owner-scoped API.
    const store = createThreadStore(saver);
    expect(getAppSchemaVersion(db)).toBe(LATEST_APP_SCHEMA_VERSION);
    expect(store.getThread(OWNER_A, "legacy-thread-1")?.title).toBe(
      "legacy row"
    );
    // Ownership enforcement still holds for the legacy row (no existence leak).
    expect(store.getThread(OWNER_B, "legacy-thread-1")).toBeNull();
  });

  it("FORWARD-ONLY GUARD: a DB recorded newer than the code throws loudly and mutates nothing", () => {
    const saver = makeSaver();
    const db = saver.db;

    // Bring the DB to LATEST via the real store, then FORGE a newer recorded
    // version (as if written by a newer app build).
    createThreadStore(saver);
    const future = LATEST_APP_SCHEMA_VERSION + 5;
    db.prepare(
      `INSERT INTO ${MIGRATIONS_TABLE} (version, applied_at) VALUES (?, ?)`
    ).run(future, Date.now());
    expect(getAppSchemaVersion(db)).toBe(future);

    const schemaBefore = db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY name`
      )
      .all();

    // Re-running the migrator against the real (older) code must fail loudly.
    expect(() => runMigrations(db, APP_MIGRATIONS)).toThrow(
      /newer than this application supports/i
    );

    // Nothing was mutated: version and schema are unchanged.
    expect(getAppSchemaVersion(db)).toBe(future);
    const schemaAfter = db
      .prepare(
        `SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY name`
      )
      .all();
    expect(schemaAfter).toEqual(schemaBefore);
  });

  it("ATOMICITY / ROLLBACK: a throwing up() rolls back the step and does not advance the version", () => {
    const saver = makeSaver();
    const db = saver.db;

    const boom = new Error("synthetic migration failure");
    const synthetic: Migration[] = [
      {
        version: 1,
        up(d) {
          d.exec(`CREATE TABLE IF NOT EXISTS ok_one (id INTEGER PRIMARY KEY);`);
        },
      },
      {
        version: 2,
        up(d) {
          // Perform some DDL BEFORE throwing, to prove the partial DDL of this
          // step is rolled back (not just that the version isn't advanced).
          d.exec(`CREATE TABLE IF NOT EXISTS partial_two (id INTEGER PRIMARY KEY);`);
          throw boom;
        },
      },
    ];

    expect(() => runMigrations(db, synthetic)).toThrow(boom);

    // Step 1 committed; step 2 rolled back entirely.
    expect(getAppSchemaVersion(db)).toBe(1);
    expect(tableExists(db, "ok_one")).toBe(true);
    expect(tableExists(db, "partial_two")).toBe(false);
    // No version-2 bookkeeping row was written.
    const rows = db
      .prepare(`SELECT version FROM ${MIGRATIONS_TABLE} ORDER BY version`)
      .all() as Array<{ version: number }>;
    expect(rows.map((r) => r.version)).toEqual([1]);
  });

  it("LIBRARY COEXISTENCE: app migrations do not disturb checkpoint tables; the checkpointer still works", async () => {
    const saver = makeSaver();
    const db = saver.db;

    // Run app migrations first.
    const store = createThreadStore(saver);
    const created = store.createThread(OWNER_A, { title: "coexist" });

    // The LangGraph checkpointer still works over the SAME shared file: a mocked
    // conversational turn writes checkpoints, and getTuple reads them back.
    const graph = createConversationalChain({}, saver);
    await graph.invoke(
      { messages: [new HumanMessage("hello")] },
      sessionConfig(created.id)
    );
    const tuple = await saver.getTuple(sessionConfig(created.id));
    expect(tuple).toBeDefined();

    // The library owns checkpoints/writes; the app owns threads + the migrations
    // table. They coexist without collision.
    expect(tableExists(db, "checkpoints")).toBe(true);
    expect(tableExists(db, "writes")).toBe(true);
    expect(tableExists(db, "threads")).toBe(true);
    expect(tableExists(db, MIGRATIONS_TABLE)).toBe(true);

    // Version-marker choice is collision-proof: we use a dedicated app table, and
    // the library (verified against 1.0.3 source) does NOT set PRAGMA
    // user_version — so it stays at the default 0 even after checkpoint writes.
    const userVersion = (
      db.pragma("user_version", { simple: true }) as number
    );
    expect(userVersion).toBe(0);
  });

  it("RESTART / DURABILITY: reopening the same file is a migrator no-op; state intact", async () => {
    // Instance A: migrate + seed thread and checkpoint, then close.
    const saverA = makeSaver();
    const storeA = createThreadStore(saverA);
    const created = storeA.createThread(OWNER_A, { title: "durable" });
    const graphA = createConversationalChain({}, saverA);
    await graphA.invoke(
      { messages: [new HumanMessage("first")] },
      sessionConfig(created.id)
    );
    const versionA = getAppSchemaVersion(saverA.db);
    saverA.db.close();

    // Instance B: reopen the SAME file with a fresh store + saver.
    const saverB = makeSaver();
    const migrationRowsBeforeReopen = versionA; // baseline
    const storeB = createThreadStore(saverB);

    // Migrator was a no-op (version unchanged; no extra bookkeeping rows).
    expect(getAppSchemaVersion(saverB.db)).toBe(versionA);
    const rowCount = (
      saverB.db
        .prepare(`SELECT COUNT(*) AS n FROM ${MIGRATIONS_TABLE}`)
        .get() as { n: number }
    ).n;
    expect(rowCount).toBe(migrationRowsBeforeReopen);

    // Ownership + checkpoint state survived the reopen.
    expect(storeB.getThread(OWNER_A, created.id)?.title).toBe("durable");
    expect(await saverB.getTuple(sessionConfig(created.id))).toBeDefined();
  });

  it("MALFORMED lists fail loudly (defensive validation)", () => {
    const saver = makeSaver();
    const db = saver.db;

    expect(() =>
      runMigrations(db, [{ version: 0, up() {} }])
    ).toThrow(/positive integers/i);
    expect(() =>
      runMigrations(db, [
        { version: 1, up() {} },
        { version: 1, up() {} },
      ])
    ).toThrow(/unique/i);
  });
});

describe("Phase 5 (5e) — repo cleanliness guard", () => {
  it("the migrations test does not create a ./data DB in the repo", async () => {
    const testDir = path.dirname(fileURLToPath(import.meta.url));
    const repoRoot = path.resolve(testDir, "..");
    const dataDir = path.join(repoRoot, "data");

    if (existsSync(dataDir)) {
      const entries = await readdir(dataDir);
      const dbArtifacts = entries.filter((e) => e.includes("checkpoints"));
      expect(dbArtifacts).toEqual([]);
    } else {
      expect(existsSync(dataDir)).toBe(false);
    }
  });
});
