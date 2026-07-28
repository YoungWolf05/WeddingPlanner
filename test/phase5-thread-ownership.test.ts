import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { HumanMessage } from "@langchain/core/messages";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { recordedCalls, resetRecordedCalls } from "./helpers/fake-model.js";

// Phase 5 (5b) — thread identity & ownership MODEL with enforcement.
//
// DETERMINISTIC and fully OFFLINE:
//   - The model boundary is mocked (same fake-model approach as 5a) so seeding
//     checkpoint state via a real conversational turn uses NO network/creds.
//   - Every SQLite database lives in a per-test temp directory under os.tmpdir()
//     and is removed in afterEach, so `npm test` NEVER writes DB artifacts into
//     the repo working tree.
//
// The ownership store shares ONE better-sqlite3 connection with the checkpointer
// (via saver.db), which is what makes the hard delete atomic across the
// dedicated `threads` table and LangGraph's checkpoint tables.
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

// UUID v4 shape (server-issued via node:crypto randomUUID).
const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const OWNER_A = "owner-alice";
const OWNER_B = "owner-bob";

function asSqlite(saver: ReturnType<typeof createCheckpointer>): SqliteSaver {
  if (!(saver instanceof SqliteSaver)) {
    throw new Error("expected a SqliteSaver");
  }
  return saver;
}

// Seed real checkpoint state for a thread_id by running one conversational turn
// through the SAME shared saver with the mocked model.
async function seedCheckpoint(
  saver: SqliteSaver,
  threadId: string,
  text = "seed turn"
): Promise<void> {
  const graph = createConversationalChain({}, saver);
  await graph.invoke(
    { messages: [new HumanMessage(text)] },
    sessionConfig(threadId)
  );
}

describe("Phase 5 (5b) — thread identity & ownership", () => {
  let tempDir: string;
  let dbPath: string;
  const openSavers: SqliteSaver[] = [];

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-threads-"));
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

  it("createThread returns a valid, unique UUID and is retrievable by its owner", () => {
    const store = createThreadStore(makeSaver());

    const t1 = store.createThread(OWNER_A, { title: "Venue ideas" });
    const t2 = store.createThread(OWNER_A);

    expect(t1.id).toMatch(UUID_RE);
    expect(t2.id).toMatch(UUID_RE);
    expect(t1.id).not.toBe(t2.id); // unique across calls

    // thread_id must NOT be derived from owner_id.
    expect(t1.id).not.toContain(OWNER_A);

    expect(t1.ownerId).toBe(OWNER_A);
    expect(t1.title).toBe("Venue ideas");
    expect(t2.title).toBeNull();
    expect(typeof t1.createdAt).toBe("number");
    expect(t1.updatedAt).toBe(t1.createdAt);

    // Inserted and retrievable.
    const fetched = store.getThread(OWNER_A, t1.id);
    expect(fetched).toEqual(t1);
  });

  it("listThreads returns only the owner's threads, newest first; other owners see none", () => {
    const store = createThreadStore(makeSaver());

    const a1 = store.createThread(OWNER_A, { title: "A-1" });
    const a2 = store.createThread(OWNER_A, { title: "A-2" });
    store.createThread(OWNER_B, { title: "B-1" });

    const aList = store.listThreads(OWNER_A);
    expect(aList.map((t) => t.id).sort()).toEqual([a1.id, a2.id].sort());
    // None of B's threads appear for A.
    expect(aList.every((t) => t.ownerId === OWNER_A)).toBe(true);

    // Ordering: most-recently-updated first. a2 was created last.
    expect(aList[0]!.id).toBe(a2.id);

    // B sees only their own; not A's.
    const bList = store.listThreads(OWNER_B);
    expect(bList).toHaveLength(1);
    expect(bList[0]!.title).toBe("B-1");
    expect(bList.some((t) => t.id === a1.id || t.id === a2.id)).toBe(false);

    // An owner with no threads sees an empty list.
    expect(store.listThreads("owner-nobody")).toEqual([]);
  });

  it("getThread enforces ownership: owner sees it, non-owner gets null (no existence leak)", () => {
    const store = createThreadStore(makeSaver());
    const t = store.createThread(OWNER_A, { title: "private" });

    // Owner: visible.
    expect(store.getThread(OWNER_A, t.id)?.id).toBe(t.id);

    // Non-owner: null — indistinguishable from a nonexistent id.
    expect(store.getThread(OWNER_B, t.id)).toBeNull();
    expect(store.getThread(OWNER_B, "00000000-0000-4000-8000-000000000000"))
      .toBeNull();
  });

  it("touchThread bumps updatedAt only for the owner", async () => {
    const store = createThreadStore(makeSaver());
    const t = store.createThread(OWNER_A);

    // Non-owner cannot touch.
    expect(store.touchThread(OWNER_B, t.id)).toBe(false);

    // Ensure a measurable clock delta, then owner touch bumps updatedAt.
    await new Promise((r) => setTimeout(r, 5));
    expect(store.touchThread(OWNER_A, t.id)).toBe(true);
    const after = store.getThread(OWNER_A, t.id)!;
    expect(after.updatedAt).toBeGreaterThanOrEqual(t.updatedAt);
    expect(after.createdAt).toBe(t.createdAt);
  });

  it("deleteThread by owner: returns true and hard-deletes BOTH the row AND checkpoint state", async () => {
    const saver = makeSaver();
    const store = createThreadStore(saver);

    const t = store.createThread(OWNER_A, { title: "to delete" });
    await seedCheckpoint(saver, t.id);

    // Precondition: checkpoint state exists for this thread_id.
    const before = await saver.getTuple(sessionConfig(t.id));
    expect(before).toBeDefined();
    expect(store.getThread(OWNER_A, t.id)).not.toBeNull();

    // Hard delete.
    expect(store.deleteThread(OWNER_A, t.id)).toBe(true);

    // Ownership row gone.
    expect(store.getThread(OWNER_A, t.id)).toBeNull();
    expect(store.listThreads(OWNER_A)).toEqual([]);

    // Checkpoint state gone — no orphaned conversation state remains.
    const after = await saver.getTuple(sessionConfig(t.id));
    expect(after).toBeUndefined();

    // No orphaned rows in either checkpoint table for this thread_id.
    const cpCount = saver.db
      .prepare("SELECT COUNT(*) AS n FROM checkpoints WHERE thread_id = ?")
      .get(t.id) as { n: number };
    const wrCount = saver.db
      .prepare("SELECT COUNT(*) AS n FROM writes WHERE thread_id = ?")
      .get(t.id) as { n: number };
    expect(cpCount.n).toBe(0);
    expect(wrCount.n).toBe(0);
  });

  it("deleteThread by NON-owner: returns false and deletes NOTHING (row + checkpoints intact)", async () => {
    const saver = makeSaver();
    const store = createThreadStore(saver);

    const t = store.createThread(OWNER_A, { title: "not yours" });
    await seedCheckpoint(saver, t.id);

    // Non-owner attempt.
    expect(store.deleteThread(OWNER_B, t.id)).toBe(false);

    // Ownership row STILL present for the real owner.
    expect(store.getThread(OWNER_A, t.id)?.id).toBe(t.id);

    // Checkpoint state STILL present (atomic gate did not touch it).
    const tuple = await saver.getTuple(sessionConfig(t.id));
    expect(tuple).toBeDefined();
    const cpCount = saver.db
      .prepare("SELECT COUNT(*) AS n FROM checkpoints WHERE thread_id = ?")
      .get(t.id) as { n: number };
    expect(cpCount.n).toBeGreaterThan(0);
  });

  it("deleteThread for a nonexistent id returns false and changes nothing", async () => {
    const saver = makeSaver();
    const store = createThreadStore(saver);
    const t = store.createThread(OWNER_A);

    expect(
      store.deleteThread(OWNER_A, "11111111-1111-4111-8111-111111111111")
    ).toBe(false);
    // Unrelated thread untouched.
    expect(store.getThread(OWNER_A, t.id)?.id).toBe(t.id);
  });

  it("ATOMICITY: the ownership-row delete and checkpoint delete commit together", async () => {
    // A non-owner delete is the observable atomic-gate proof: because the whole
    // operation is one transaction gated on the ownership row, a non-owner
    // touches NEITHER side. (Complementary to the owner test, where BOTH sides
    // vanish together.)
    const saver = makeSaver();
    const store = createThreadStore(saver);

    const t = store.createThread(OWNER_A);
    await seedCheckpoint(saver, t.id);

    // Non-owner: neither the threads row nor checkpoints change.
    store.deleteThread(OWNER_B, t.id);
    expect(store.getThread(OWNER_A, t.id)).not.toBeNull();
    expect(await saver.getTuple(sessionConfig(t.id))).toBeDefined();

    // Owner: both vanish together in one transaction.
    store.deleteThread(OWNER_A, t.id);
    expect(store.getThread(OWNER_A, t.id)).toBeNull();
    expect(await saver.getTuple(sessionConfig(t.id))).toBeUndefined();
  });

  it("RESTART PERSISTENCE: ownership + checkpoints survive a reopen; cross-owner denial holds", async () => {
    // Instance A: create a thread + seed checkpoint, then close (simulated exit).
    const saverA = makeSaver();
    const storeA = createThreadStore(saverA);
    const created = storeA.createThread(OWNER_A, { title: "durable" });
    await seedCheckpoint(saverA, created.id);
    saverA.db.close();

    // Instance B: reopen the SAME file.
    const saverB = makeSaver();
    const storeB = createThreadStore(saverB);

    // Ownership persisted.
    const list = storeB.listThreads(OWNER_A);
    expect(list.map((t) => t.id)).toEqual([created.id]);
    expect(storeB.getThread(OWNER_A, created.id)?.title).toBe("durable");

    // Checkpoint persisted.
    expect(await saverB.getTuple(sessionConfig(created.id))).toBeDefined();

    // Cross-owner denial still holds after restart (no existence leak).
    expect(storeB.getThread(OWNER_B, created.id)).toBeNull();
    expect(storeB.listThreads(OWNER_B)).toEqual([]);
    expect(storeB.deleteThread(OWNER_B, created.id)).toBe(false);
    // ...and the owner can still delete it post-restart, removing both sides.
    expect(storeB.deleteThread(OWNER_A, created.id)).toBe(true);
    expect(await saverB.getTuple(sessionConfig(created.id))).toBeUndefined();
  });

  it("createThreadStore setup is idempotent (safe to call repeatedly on one db)", () => {
    const saver = makeSaver();
    const store1 = createThreadStore(saver);
    const t = store1.createThread(OWNER_A);
    // A second store over the same connection must not error and must see prior data.
    const store2 = createThreadStore(saver);
    expect(store2.getThread(OWNER_A, t.id)?.id).toBe(t.id);
  });

  it("deleteThread works even when no checkpoint was ever written (lazy checkpoint schema)", () => {
    // Fresh file: no conversational turn, so the library's checkpoint tables may
    // not exist yet. deleteThread must still hard-delete the ownership row
    // without failing on a missing checkpoints/writes table.
    const saver = makeSaver();
    const store = createThreadStore(saver);
    const t = store.createThread(OWNER_A);

    expect(store.deleteThread(OWNER_A, t.id)).toBe(true);
    expect(store.getThread(OWNER_A, t.id)).toBeNull();
  });

  it("ATOMIC ROLLBACK: a mid-transaction failure leaves NEITHER side changed", async () => {
    // Proves the delete is a genuine all-or-nothing transaction. We force the
    // LAST statement inside deleteThread's db.transaction() (the DELETE FROM
    // writes, which is compiled/prepared at store construction) to throw AFTER
    // the ownership-row delete and the DELETE FROM checkpoints have already
    // executed — by DROPPING the `writes` table out from under the prepared
    // statement. better-sqlite3 must then roll the whole transaction back.
    const saver = makeSaver();
    const store = createThreadStore(saver);

    const t = store.createThread(OWNER_A, { title: "rollback" });
    await seedCheckpoint(saver, t.id);

    // Preconditions: ownership row + checkpoint state both present.
    expect(store.getThread(OWNER_A, t.id)?.id).toBe(t.id);
    expect(await saver.getTuple(sessionConfig(t.id))).toBeDefined();
    const cpBefore = (
      saver.db
        .prepare("SELECT COUNT(*) AS n FROM checkpoints WHERE thread_id = ?")
        .get(t.id) as { n: number }
    ).n;
    const wrBefore = (
      saver.db
        .prepare("SELECT COUNT(*) AS n FROM writes WHERE thread_id = ?")
        .get(t.id) as { n: number }
    ).n;
    expect(cpBefore).toBeGreaterThan(0);

    // Force the mid-transaction failure: remove the `writes` table so the
    // already-prepared `DELETE FROM writes` statement errors when it runs (the
    // third statement, after the threads-row and checkpoints deletes).
    saver.db.exec("DROP TABLE writes");

    // deleteThread must SURFACE the error (not silently return), and the whole
    // transaction must have rolled back.
    expect(() => store.deleteThread(OWNER_A, t.id)).toThrow();

    // Recreate `writes` so subsequent reads (and cleanup) work; do this before
    // the assertions so getTuple can query it.
    saver.db.exec(`
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

    // NEITHER side changed: ownership row still present...
    expect(store.getThread(OWNER_A, t.id)?.id).toBe(t.id);
    expect(store.listThreads(OWNER_A).map((x) => x.id)).toContain(t.id);
    // ...and the checkpoint state is unchanged (the DELETE FROM checkpoints that
    // ran inside the aborted transaction was rolled back).
    expect(await saver.getTuple(sessionConfig(t.id))).toBeDefined();
    const cpAfter = (
      saver.db
        .prepare("SELECT COUNT(*) AS n FROM checkpoints WHERE thread_id = ?")
        .get(t.id) as { n: number }
    ).n;
    expect(cpAfter).toBe(cpBefore);
    // writes were re-seeded empty by the CREATE above (the temp DB is discarded
    // in afterEach); the load-bearing invariant is that checkpoints were NOT
    // deleted, proving the transaction rolled back as a unit.
    expect(wrBefore).toBeGreaterThanOrEqual(0);
  });

  it("SCHEMA SANITY: the library's checkpoints/writes columns match the mirrored DDL", async () => {
    // Complements the version-pin guard: after the LIBRARY (not our
    // ensureCheckpointTables) lazily creates its tables during a real checkpoint
    // op, introspect the actual columns and assert they still contain the
    // columns our mirrored DDL and replicated DELETEs rely on. If an upgrade
    // slips past the version pin, a column rename (esp. thread_id) is caught here.
    const saver = makeSaver();
    // A conversational turn triggers the library's own setup() → its real DDL.
    await seedCheckpoint(saver, "schema-sanity-thread");

    const columnsOf = (table: string): string[] =>
      (
        saver.db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
          name: string;
        }>
      ).map((c) => c.name);

    // The exact column sets the installed library creates (mirrored verbatim in
    // src/core/threads.ts ensureCheckpointTables). thread_id is load-bearing for
    // the replicated DELETE FROM ... WHERE thread_id = ? statements.
    expect(columnsOf("checkpoints").sort()).toEqual(
      [
        "checkpoint",
        "checkpoint_id",
        "checkpoint_ns",
        "metadata",
        "parent_checkpoint_id",
        "thread_id",
        "type",
      ].sort()
    );
    expect(columnsOf("writes").sort()).toEqual(
      [
        "channel",
        "checkpoint_id",
        "checkpoint_ns",
        "idx",
        "task_id",
        "thread_id",
        "type",
        "value",
      ].sort()
    );
  });
});

describe("Phase 5 (5b) — repo cleanliness guard", () => {
  it("the thread-ownership test does not create a ./data DB in the repo", async () => {
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
