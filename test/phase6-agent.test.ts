import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import {
  makeScriptedToolCallingModel,
  type ScriptedTurn,
} from "./helpers/fake-model.js";

// Phase 6 (increment 6c) — the wedding-planning TOOL-LOOP AGENT.
//
// Fully OFFLINE and deterministic. The model boundary (createChatModel) is
// mocked with a SCRIPTED tool-calling fake so the REAL createReactAgent loop +
// REAL ToolNode + REAL 6b tools run without any network/credentials. Date
// determinism for days_until (which defaults `now` to the real clock in the
// tool wrapper) is achieved with vi.useFakeTimers + vi.setSystemTime, restored
// after each test.
//
// A shared, hoisted `control` lets each test install its own script and inspect
// the exact ModelOptions createChatModel was called with (so the opus
// temperature-omit path is asserted without a live call).

const control = vi.hoisted(() => ({
  // The model instance the mocked factory returns (set per test).
  model: undefined as unknown,
  // Every ModelOptions createChatModel was called with, in order.
  captured: [] as Array<Record<string, unknown>>,
}));

vi.mock("../src/core/model.js", () => ({
  createChatModel: (opts?: Record<string, unknown>) => {
    control.captured.push(opts ?? {});
    return control.model;
  },
}));

const {
  createWeddingAgent,
  runWeddingAgent,
  decideAgentModelOptions,
  DEFAULT_AGENT_MODEL,
} = await import("../src/core/agent.js");

const OPUS = "claude-opus-4-8";
const SONNET = "claude-sonnet-4-6";

// Fixed "today" for deterministic days_until. 2026-11-12 UTC -> exactly 30 days
// before the exit-criterion target 2026-12-12.
const FIXED_NOW = new Date("2026-11-12T08:30:00.000Z");
const TARGET_DATE = "2026-12-12";
const EXPECTED_DAYS_UNTIL = 30; // 2026-11-12 -> 2026-12-12, UTC calendar days.

// Helper: install a scripted model as the factory's return value.
function useScript(script: ScriptedTurn[]) {
  const built = makeScriptedToolCallingModel(script);
  control.model = built.model;
  return built.recorder;
}

// Helper: collect all ToolMessages from a final agent state.
function toolMessages(messages: BaseMessage[]): ToolMessage[] {
  return messages.filter((m): m is ToolMessage => m instanceof ToolMessage);
}

// Helper: collect all AIMessages that carried tool_calls, with the calls flat.
function requestedToolCalls(
  messages: BaseMessage[]
): Array<{ name: string; args: Record<string, unknown> }> {
  const calls: Array<{ name: string; args: Record<string, unknown> }> = [];
  for (const m of messages) {
    if (m instanceof AIMessage && m.tool_calls) {
      for (const tc of m.tool_calls) {
        calls.push({ name: tc.name, args: tc.args as Record<string, unknown> });
      }
    }
  }
  return calls;
}

beforeEach(() => {
  control.model = undefined;
  control.captured.length = 0;
  vi.useFakeTimers();
  vi.setSystemTime(FIXED_NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Phase 6 — decideAgentModelOptions (shared opus temperature-omit)", () => {
  it("omits temperature (temperature: null) for claude-opus-4-8", () => {
    const opts = decideAgentModelOptions(OPUS);
    expect(opts.model).toBe(OPUS);
    expect(opts.temperature).toBeNull();
  });

  it("does NOT force temperature to null for sonnet (key absent)", () => {
    const opts = decideAgentModelOptions(SONNET);
    expect(opts.model).toBe(SONNET);
    expect("temperature" in opts).toBe(false);
  });

  it("DEFAULT_AGENT_MODEL is the fully-supported sonnet alias", () => {
    expect(DEFAULT_AGENT_MODEL).toBe(SONNET);
  });
});

describe("Phase 6 — createWeddingAgent model construction (opus temp-omit)", () => {
  it("builds the DEFAULT (sonnet) model with temperature NOT forced to null", () => {
    useScript([{ content: "hi" }]);
    createWeddingAgent();
    expect(control.captured).toHaveLength(1);
    const opts = control.captured[0]!;
    expect(opts.model).toBe(SONNET);
    expect("temperature" in opts).toBe(false);
  });

  it("builds the opus model with temperature omitted (null)", () => {
    useScript([{ content: "hi" }]);
    createWeddingAgent({ model: OPUS });
    expect(control.captured).toHaveLength(1);
    const opts = control.captured[0]!;
    expect(opts.model).toBe(OPUS);
    expect(opts.temperature).toBeNull();
  });

  it("honors an explicit sonnet model id without forcing null", () => {
    useScript([{ content: "hi" }]);
    createWeddingAgent({ model: SONNET });
    const opts = control.captured[0]!;
    expect(opts.model).toBe(SONNET);
    expect("temperature" in opts).toBe(false);
  });
});

describe("Phase 6 — EXIT CRITERION: two-tool end-to-end flow", () => {
  it("runs days_until + split_budget through the real react loop with correct real results", async () => {
    // Turn 1: request BOTH tools. Turn 2 (after ToolNode appends ToolMessages):
    // final answer text summarizing the results.
    const recorder = useScript([
      {
        content: "",
        toolCalls: [
          { name: "days_until", args: { date: TARGET_DATE } },
          { name: "split_budget", args: { total: 30000 } },
        ],
      },
      {
        content: `There are ${EXPECTED_DAYS_UNTIL} days until ${TARGET_DATE}, and your $30000 budget is split across categories.`,
      },
    ]);

    const agent = createWeddingAgent();
    const result = await agent.invoke({
      messages: [
        new HumanMessage(
          `How many days until ${TARGET_DATE} and split a $30k budget`
        ),
      ],
    });
    const messages = result.messages as BaseMessage[];

    // (1) BOTH tools were actually REQUESTED with the right args.
    const requested = requestedToolCalls(messages);
    const daysReq = requested.find((c) => c.name === "days_until");
    const budgetReq = requested.find((c) => c.name === "split_budget");
    expect(daysReq).toBeDefined();
    expect(daysReq!.args.date).toBe(TARGET_DATE);
    expect(budgetReq).toBeDefined();
    expect(budgetReq!.args.total).toBe(30000);

    // (2) BOTH tools actually RAN (ToolNode appended two ToolMessages) and their
    // REAL outputs are correct.
    const tms = toolMessages(messages);
    expect(tms).toHaveLength(2);

    const daysTm = tms.find((t) => t.name === "days_until");
    const budgetTm = tms.find((t) => t.name === "split_budget");
    expect(daysTm).toBeDefined();
    expect(budgetTm).toBeDefined();

    // days_until real artifact: exact day count from the fixed system time.
    const daysArtifact = daysTm!.artifact as {
      targetDate: string;
      daysUntil: number;
      direction: string;
    };
    expect(daysArtifact.targetDate).toBe(TARGET_DATE);
    expect(daysArtifact.daysUntil).toBe(EXPECTED_DAYS_UNTIL);
    expect(daysArtifact.direction).toBe("future");

    // split_budget real artifact: default split, amounts sum EXACTLY to 30000.
    const budgetArtifact = budgetTm!.artifact as {
      total: number;
      categories: Array<{ name: string; amount: number }>;
    };
    expect(budgetArtifact.total).toBe(30000);
    const sum = budgetArtifact.categories.reduce((s, c) => s + c.amount, 0);
    expect(sum).toBe(30000);

    // (3) The loop completed: the LAST message is the final AIMessage (no
    // further tool_calls) and reflects the tool results.
    const last = messages[messages.length - 1];
    expect(last).toBeInstanceOf(AIMessage);
    expect((last as AIMessage).tool_calls ?? []).toHaveLength(0);
    const finalText = (last as AIMessage).content as string;
    expect(finalText).toContain(String(EXPECTED_DAYS_UNTIL));
    expect(finalText).toContain("30000");

    // The scripted model saw the ToolMessages before producing the final turn.
    expect(recorder.calls).toHaveLength(2);
    const secondCallMessages = recorder.calls[1]!;
    expect(
      secondCallMessages.filter((m) => m instanceof ToolMessage)
    ).toHaveLength(2);

    // runWeddingAgent convenience returns the same final text.
    const viaRunner = await (async () => {
      // Fresh script/agent for the runner path (call order is per-instance).
      useScript([
        {
          content: "",
          toolCalls: [{ name: "days_until", args: { date: TARGET_DATE } }],
        },
        { content: `Final: ${EXPECTED_DAYS_UNTIL} days.` },
      ]);
      const a = createWeddingAgent();
      return runWeddingAgent(a, "days?");
    })();
    expect(viaRunner).toBe(`Final: ${EXPECTED_DAYS_UNTIL} days.`);
  });
});

describe("Phase 6 — single-tool prompt", () => {
  it("runs only split_budget when the script requests just that tool", async () => {
    useScript([
      {
        content: "",
        toolCalls: [{ name: "split_budget", args: { total: 12000 } }],
      },
      { content: "Here is your $12000 budget split." },
    ]);

    const agent = createWeddingAgent();
    const result = await agent.invoke({
      messages: [new HumanMessage("split a $12k budget")],
    });
    const messages = result.messages as BaseMessage[];

    const tms = toolMessages(messages);
    expect(tms).toHaveLength(1);
    expect(tms[0]!.name).toBe("split_budget");

    const artifact = tms[0]!.artifact as {
      total: number;
      categories: Array<{ amount: number }>;
    };
    expect(artifact.total).toBe(12000);
    expect(artifact.categories.reduce((s, c) => s + c.amount, 0)).toBe(12000);

    // days_until was NEVER called.
    const requested = requestedToolCalls(messages);
    expect(requested.some((c) => c.name === "days_until")).toBe(false);

    const last = messages[messages.length - 1] as AIMessage;
    expect(last.content).toContain("12000");
  });

  it("runs only days_until when the script requests just that tool", async () => {
    useScript([
      {
        content: "",
        toolCalls: [{ name: "days_until", args: { date: TARGET_DATE } }],
      },
      { content: `It's ${EXPECTED_DAYS_UNTIL} days away.` },
    ]);

    const agent = createWeddingAgent();
    const result = await agent.invoke({
      messages: [new HumanMessage(`days until ${TARGET_DATE}?`)],
    });
    const messages = result.messages as BaseMessage[];

    const tms = toolMessages(messages);
    expect(tms).toHaveLength(1);
    expect(tms[0]!.name).toBe("days_until");
    const artifact = tms[0]!.artifact as { daysUntil: number };
    expect(artifact.daysUntil).toBe(EXPECTED_DAYS_UNTIL);

    const requested = requestedToolCalls(messages);
    expect(requested.some((c) => c.name === "split_budget")).toBe(false);
  });
});

describe("Phase 6 — no-tool prompt (plain-chat-like through the agent)", () => {
  it("returns the final text with NO tool invoked", async () => {
    useScript([
      { content: "Congratulations on your engagement! How can I help?" },
    ]);

    const agent = createWeddingAgent();
    const result = await agent.invoke({
      messages: [new HumanMessage("Hi Aria!")],
    });
    const messages = result.messages as BaseMessage[];

    // No tool ran.
    expect(toolMessages(messages)).toHaveLength(0);
    expect(requestedToolCalls(messages)).toHaveLength(0);

    // Final assistant text is returned.
    const last = messages[messages.length - 1] as AIMessage;
    expect(last).toBeInstanceOf(AIMessage);
    expect(last.content).toContain("Congratulations");
  });
});

describe("Phase 6 — tool-error path (real tool throws on invalid input)", () => {
  it("surfaces the tool error as a ToolMessage and completes the loop gracefully", async () => {
    // First turn: call days_until with an INVALID date (month 13). The real 6b
    // tool throws; the ToolNode surfaces the error as a ToolMessage rather than
    // crashing the run. Second turn: the model produces a final apology.
    useScript([
      {
        content: "",
        toolCalls: [{ name: "days_until", args: { date: "2026-13-01" } }],
      },
      { content: "Sorry, that date is invalid. Please give a real date." },
    ]);

    const agent = createWeddingAgent();

    // The loop must NOT throw / must not produce an unhandled rejection.
    const result = await agent.invoke({
      messages: [new HumanMessage("days until 2026-13-01?")],
    });
    const messages = result.messages as BaseMessage[];

    // ToolNode surfaced the error as a ToolMessage flagged as an error.
    const tms = toolMessages(messages);
    expect(tms).toHaveLength(1);
    expect(tms[0]!.name).toBe("days_until");
    expect(tms[0]!.status).toBe("error");
    // The error content mentions the invalid date (redaction-safe domain text).
    expect(String(tms[0]!.content)).toMatch(/invalid|not a real calendar/i);

    // The loop still completed with a final AIMessage.
    const last = messages[messages.length - 1] as AIMessage;
    expect(last).toBeInstanceOf(AIMessage);
    expect((last.tool_calls ?? [])).toHaveLength(0);
    expect(last.content).toContain("invalid");
  });
});
