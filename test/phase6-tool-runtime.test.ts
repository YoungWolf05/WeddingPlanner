import { describe, it, expect, vi, afterEach } from "vitest";
import {
  withToolTimeout,
  invokeToolWithTimeout,
  ToolTimeoutError,
  DEFAULT_TOOL_TIMEOUT_MS,
} from "../src/core/tool-runtime.js";
import { daysUntilTool, splitBudgetTool } from "../src/core/tools.js";

// Phase 6 (increment 6d) — BOUNDED tool-execution timeout.
//
// Fully OFFLINE and deterministic. The timeout PATH is driven with vitest fake
// timers (no real multi-second waits, no real-clock flakiness): the wrapped work
// is a promise that never settles, so the timer ALWAYS wins once advanced. The
// fast/success path uses genuinely-fast work (a resolved promise / a real sync
// tool), which settles on a microtask and always beats the (generous) timer.

afterEach(() => {
  vi.useRealTimers();
});

describe("Phase 6 — withToolTimeout: timeout path", () => {
  it("rejects with a typed ToolTimeoutError once the bound elapses", async () => {
    vi.useFakeTimers();
    // Work that never settles: only the timer can resolve the race.
    const never = new Promise<number>(() => {});
    const p = withToolTimeout(() => never, 5_000, "slow_tool");
    // Attach a rejection handler BEFORE advancing so there is no unhandled
    // rejection window.
    const assertion = expect(p).rejects.toBeInstanceOf(ToolTimeoutError);
    await vi.advanceTimersByTimeAsync(5_000);
    await assertion;
  });

  it("the timeout error carries the tool name + bound and a redaction-safe message", async () => {
    vi.useFakeTimers();
    const never = new Promise<number>(() => {});
    const p = withToolTimeout(() => never, 1_234, "days_until");
    const captured = p.catch((err) => err);
    await vi.advanceTimersByTimeAsync(1_234);
    const err = (await captured) as ToolTimeoutError;
    expect(err).toBeInstanceOf(ToolTimeoutError);
    expect(err.name).toBe("ToolTimeoutError");
    expect(err.toolName).toBe("days_until");
    expect(err.timeoutMs).toBe(1_234);
    expect(err.message).toMatch(/days_until/);
    expect(err.message).toMatch(/1234ms/);
  });
});

describe("Phase 6 — withToolTimeout: success path (fast work resolves)", () => {
  it("resolves with a fast async value well under a generous bound", async () => {
    const result = await withToolTimeout(
      () => Promise.resolve(42),
      10_000,
      "fast_tool"
    );
    expect(result).toBe(42);
  });

  it("resolves a synchronous return value (normalized to a promise)", async () => {
    const result = await withToolTimeout(() => "sync-ok", 10_000);
    expect(result).toBe("sync-ok");
  });

  it("propagates a genuine tool rejection UNCHANGED (not masked as a timeout)", async () => {
    const boom = new Error("tool blew up for a real reason");
    await expect(
      withToolTimeout(() => Promise.reject(boom), 10_000)
    ).rejects.toBe(boom);
  });

  it("propagates a synchronous throw UNCHANGED", async () => {
    const boom = new Error("sync explosion");
    await expect(
      withToolTimeout(() => {
        throw boom;
      }, 10_000)
    ).rejects.toBe(boom);
  });
});

describe("Phase 6 — withToolTimeout: input validation", () => {
  it("rejects a non-positive timeout", async () => {
    await expect(withToolTimeout(() => 1, 0)).rejects.toThrow(
      /positive, finite/i
    );
    await expect(withToolTimeout(() => 1, -5)).rejects.toThrow(
      /positive, finite/i
    );
  });

  it("rejects a non-finite timeout", async () => {
    await expect(
      withToolTimeout(() => 1, Number.POSITIVE_INFINITY)
    ).rejects.toThrow(/positive, finite/i);
    await expect(withToolTimeout(() => 1, Number.NaN)).rejects.toThrow(
      /positive, finite/i
    );
  });
});

describe("Phase 6 — invokeToolWithTimeout over the REAL 6b tools", () => {
  it("runs days_until under the timeout and returns its content (fast, well within bound)", async () => {
    // The real tool is sync/instantaneous, so it always resolves under a
    // generous bound. content_and_artifact tools return the summary content
    // string when invoked with a plain args object.
    const content = await invokeToolWithTimeout(
      daysUntilTool,
      { date: "2999-01-01" },
      10_000
    );
    expect(String(content)).toMatch(/2999-01-01/);
  });

  it("runs split_budget under the timeout and returns its content", async () => {
    const content = await invokeToolWithTimeout(
      splitBudgetTool,
      { total: 10000 },
      10_000
    );
    expect(String(content)).toMatch(/10000/);
  });

  it("surfaces a real tool INPUT error (invalid date) as a normal rejection, not a timeout", async () => {
    await expect(
      invokeToolWithTimeout(daysUntilTool, { date: "2026-13-01" }, 10_000)
    ).rejects.not.toBeInstanceOf(ToolTimeoutError);
  });

  it("defaults to DEFAULT_TOOL_TIMEOUT_MS when no bound is given", async () => {
    expect(DEFAULT_TOOL_TIMEOUT_MS).toBeGreaterThan(0);
    const content = await invokeToolWithTimeout(splitBudgetTool, {
      total: 5000,
    });
    expect(String(content)).toMatch(/5000/);
  });
});
