import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { recordedCalls, resetRecordedCalls } from "./helpers/fake-model.js";

// Phase 5 (5a) — durable SQLite checkpointer.
//
// These tests are DETERMINISTIC and fully OFFLINE:
//   - The model boundary is mocked (same fake-model approach as phase2) so no
//     network/credentials are used.
//   - Every SQLite database lives in a per-test temp directory under os.tmpdir()
//     and is removed in afterEach, so `npm test` NEVER creates the default
//     ./data DB in the repo working tree.
//
// The core proof is RESTART SURVIVAL: a checkpointer instance is created against
// a temp db file, a turn is run, the instance's handle is CLOSED (simulating
// process exit), a NEW instance is opened against the SAME file, and the prior
// turn's history is shown to still be present.
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
const { createCheckpointer, resolveCheckpointDbPath, sessionConfig } =
  await import("../src/core/memory.js");

// Close a checkpointer's underlying SQLite handle. The consuming interface is
// BaseCheckpointSaver, but the concrete durable saver is SqliteSaver, which owns
// a better-sqlite3 Database. Closing it releases the file handle/lock — required
// on Windows before the temp file can be deleted, and the mechanism that
// simulates a process restart.
function closeSaver(saver: BaseCheckpointSaver): void {
  if (saver instanceof SqliteSaver) {
    saver.db.close();
  }
}

async function invokeTurn(
  graph: ReturnType<typeof createConversationalChain>,
  text: string,
  cfg: ReturnType<typeof sessionConfig>
) {
  return graph.invoke({ messages: [new HumanMessage(text)] }, cfg);
}

describe("Phase 5 (5a) — durable SQLite checkpointer", () => {
  let tempDir: string;
  let dbPath: string;
  const openSavers: BaseCheckpointSaver[] = [];

  beforeEach(async () => {
    resetRecordedCalls();
    tempDir = await mkdtemp(path.join(tmpdir(), "wp-checkpoint-"));
    dbPath = path.join(tempDir, "checkpoints.sqlite");
    openSavers.length = 0;
  });

  afterEach(async () => {
    // Defensively close any handles a test forgot to close, then remove the
    // temp directory so no db artifacts survive the suite.
    for (const saver of openSavers) {
      try {
        closeSaver(saver);
      } catch {
        // Already closed — ignore.
      }
    }
    await rm(tempDir, { recursive: true, force: true });
  });

  function makeCheckpointer(p: string = dbPath): BaseCheckpointSaver {
    const saver = createCheckpointer(p);
    openSavers.push(saver);
    return saver;
  }

  it("factory returns a BaseCheckpointSaver and creates the db file (with parent dir)", () => {
    // Parent directory does not exist yet — the factory must create it.
    const nested = path.join(tempDir, "nested", "dir", "checkpoints.sqlite");
    expect(existsSync(path.dirname(nested))).toBe(false);

    const saver = makeCheckpointer(nested);

    expect(saver).toBeInstanceOf(SqliteSaver);
    // BaseCheckpointSaver contract is satisfied (duck-typed methods present).
    expect(typeof saver.getTuple).toBe("function");
    expect(typeof saver.put).toBe("function");
    expect(existsSync(nested)).toBe(true);
  });

  it("resolveCheckpointDbPath: uses a provided path and resolves the default", () => {
    const provided = resolveCheckpointDbPath(dbPath);
    expect(provided).toBe(path.resolve(dbPath));

    // With no argument and no CHECKPOINT_DB_PATH in the (test) env, it resolves
    // the documented default relative to the working directory. Capture and
    // restore any prior env value so this test stays hermetic and never leaks
    // global env state to other tests.
    const priorDbPath = process.env.CHECKPOINT_DB_PATH;
    try {
      delete process.env.CHECKPOINT_DB_PATH;
      const def = resolveCheckpointDbPath();
      expect(def).toBe(path.resolve("./data/checkpoints.sqlite"));
      expect(path.isAbsolute(def)).toBe(true);
    } finally {
      if (priorDbPath === undefined) {
        delete process.env.CHECKPOINT_DB_PATH;
      } else {
        process.env.CHECKPOINT_DB_PATH = priorDbPath;
      }
    }
  });

  it("RESTART SURVIVAL: a new instance against the same file sees prior history", async () => {
    const cfg = sessionConfig("restart-thread");

    // Instance A: run one turn, then drop the instance (close its handle).
    const saverA = makeCheckpointer();
    const graphA = createConversationalChain({}, saverA);
    await invokeTurn(graphA, "We have 120 guests.", cfg);
    closeSaver(saverA);

    // Simulated restart: a brand-new instance B opens the SAME db file.
    const saverB = makeCheckpointer();

    // Prior checkpoint is readable directly from the fresh instance.
    const tuple = await saverB.getTuple(cfg);
    expect(tuple).toBeDefined();

    // ...and a second turn on the same thread SEES the persisted history.
    const graphB = createConversationalChain({}, saverB);
    await invokeTurn(graphB, "What's next?", cfg);

    // Two model calls total; each was recorded with the messages it received.
    expect(recordedCalls).toHaveLength(2);

    // Turn 2 (post-restart) must include turn 1's human + ai messages:
    // [system, human(120 guests), ai(reply-1), human(What's next?)].
    const second: BaseMessage[] = recordedCalls[1]!;
    expect(second).toHaveLength(4);
    expect(second[0]).toBeInstanceOf(SystemMessage);
    expect(second[1]).toBeInstanceOf(HumanMessage);
    expect(second[1]!.content).toBe("We have 120 guests.");
    expect(second[2]).toBeInstanceOf(AIMessage);
    expect(second[2]!.content).toBe("reply-1");
    expect(second[3]).toBeInstanceOf(HumanMessage);
    expect(second[3]!.content).toBe("What's next?");
  });

  it("THREAD ISOLATION persists: a different thread_id does not see another thread's history", async () => {
    // Write thread-A history, then simulate a restart before reading thread-B so
    // isolation is proven against the persisted (not in-RAM) state.
    const saverA = makeCheckpointer();
    const graphA = createConversationalChain({}, saverA);
    await invokeTurn(
      graphA,
      "Secret from thread A.",
      sessionConfig("thread-A")
    );
    closeSaver(saverA);

    const saverB = makeCheckpointer();
    const graphB = createConversationalChain({}, saverB);
    await invokeTurn(graphB, "Fresh start on B.", sessionConfig("thread-B"));

    expect(recordedCalls).toHaveLength(2);

    const threadB: BaseMessage[] = recordedCalls[1]!;
    // thread-B's first turn is fresh: only [system, human(Fresh start on B.)].
    expect(threadB).toHaveLength(2);
    expect(threadB[0]).toBeInstanceOf(SystemMessage);
    expect(threadB[1]!.content).toBe("Fresh start on B.");

    // No thread-A content leaked into thread-B.
    const leaked = threadB.some((m) =>
      String(m.content).includes("Secret from thread A.")
    );
    expect(leaked).toBe(false);

    // And thread-B has no persisted checkpoint under thread-A's key confusion:
    // reading thread-A from the fresh instance still returns its own history.
    const tupleA = await saverB.getTuple(sessionConfig("thread-A"));
    expect(tupleA).toBeDefined();
  });
});

describe("Phase 5 (5a) — repo cleanliness guard", () => {
  it("the durable-checkpointer test does not create a ./data DB in the repo", async () => {
    // The suite must only write to temp dirs. Assert the repo's ./data directory
    // was not created as a side effect of importing/using the module under test.
    const testDir = path.dirname(fileURLToPath(import.meta.url));
    const repoRoot = path.resolve(testDir, "..");
    const dataDir = path.join(repoRoot, "data");

    if (existsSync(dataDir)) {
      // If it exists it must be empty of any sqlite artifacts (belt-and-braces:
      // an empty pre-existing dir is tolerated, but no checkpoints file).
      const entries = await readdir(dataDir);
      const dbArtifacts = entries.filter((e) => e.includes("checkpoints"));
      expect(dbArtifacts).toEqual([]);
    } else {
      expect(existsSync(dataDir)).toBe(false);
    }
  });
});
