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
import { weddingTools } from "../src/core/tools.js";
import { PERMITTED_TOOL_NAMES } from "../src/core/contracts.js";

// Phase 6 (increment 6d) — EXIT CRITERION 3 at the agent MESSAGE-CONTRACT level.
//
// "Only permitted tools execute, and tool state/errors are represented in the
// typed event contract." For Phase 6 the "typed event contract" IS the agent's
// typed LangGraph message stream (tool intentions => AIMessage.tool_calls; tool
// results/errors => ToolMessage with status "error" on failure). Wiring the
// agent into the HTTP SSE contract is DEFERRED to a later phase (see agent.ts).
//
// Fully OFFLINE + deterministic: the model boundary is mocked with a SCRIPTED
// tool-calling fake so the REAL createReactAgent loop + REAL ToolNode + REAL 6b
// tools run without network/credentials. Date determinism via fake timers.

const control = vi.hoisted(() => ({
  model: undefined as unknown,
  captured: [] as Array<Record<string, unknown>>,
}));

vi.mock("../src/core/model.js", () => ({
  createChatModel: (opts?: Record<string, unknown>) => {
    control.captured.push(opts ?? {});
    return control.model;
  },
}));

const { createWeddingAgent } = await import("../src/core/agent.js");

const FIXED_NOW = new Date("2026-11-12T08:30:00.000Z");
const TARGET_DATE = "2026-12-12";
const EXPECTED_DAYS_UNTIL = 30;

function useScript(script: ScriptedTurn[]) {
  const built = makeScriptedToolCallingModel(script);
  control.model = built.model;
  return built.recorder;
}

function toolMessages(messages: BaseMessage[]): ToolMessage[] {
  return messages.filter((m): m is ToolMessage => m instanceof ToolMessage);
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

describe("Phase 6 — ONLY permitted tools are bound/executable", () => {
  it("the agent's tool set is EXACTLY [days_until, split_budget] — no more, no less", () => {
    const boundNames = weddingTools.map((t) => t.name).sort();
    expect(boundNames).toEqual(["days_until", "split_budget"]);
    // The permitted-name registry the probe/classifier uses is the SAME set.
    expect([...PERMITTED_TOOL_NAMES].sort()).toEqual(boundNames);
  });

  it("a request for an UNKNOWN tool is refused by ToolNode and NEVER executes an unpermitted tool", async () => {
    // Turn 1: the model asks for a tool that is NOT in weddingTools. The REAL
    // ToolNode must refuse it with an error ToolMessage rather than executing
    // anything outside the permitted set. Turn 2: the model recovers.
    useScript([
      {
        content: "",
        toolCalls: [
          { name: "delete_everything", args: { confirm: true } },
        ],
      },
      { content: "Sorry, I can only use my safe planning tools." },
    ]);

    const agent = createWeddingAgent();
    const result = await agent.invoke({
      messages: [new HumanMessage("please delete everything")],
    });
    const messages = result.messages as BaseMessage[];

    const tms = toolMessages(messages);
    // Exactly one ToolMessage, for the UNKNOWN tool, flagged as an error.
    expect(tms).toHaveLength(1);
    expect(tms[0]!.name).toBe("delete_everything");
    expect(tms[0]!.status).toBe("error");
    expect(String(tms[0]!.content)).toMatch(/not found|not a valid tool/i);

    // The unpermitted tool produced NO artifact (it never ran) — contrast with
    // the real tools, which always attach a structured artifact.
    expect(tms[0]!.artifact).toBeUndefined();

    // The loop still completed with a final AIMessage (graceful recovery).
    const last = messages[messages.length - 1] as AIMessage;
    expect(last).toBeInstanceOf(AIMessage);
    expect(last.tool_calls ?? []).toHaveLength(0);
  });
});

describe("Phase 6 — tool STATE is represented in the typed message stream", () => {
  it("tool intentions are AIMessage.tool_calls; results are ToolMessages with the real artifact", async () => {
    const recorder = useScript([
      {
        content: "",
        toolCalls: [{ name: "days_until", args: { date: TARGET_DATE } }],
      },
      { content: `It is ${EXPECTED_DAYS_UNTIL} days away.` },
    ]);

    const agent = createWeddingAgent();
    const result = await agent.invoke({
      messages: [new HumanMessage(`days until ${TARGET_DATE}?`)],
    });
    const messages = result.messages as BaseMessage[];

    // (1) STATE: the tool INTENTION is a typed AIMessage carrying tool_calls
    // with the exact name + args.
    const intention = messages.find(
      (m): m is AIMessage =>
        m instanceof AIMessage && (m.tool_calls ?? []).length > 0
    );
    expect(intention).toBeDefined();
    const call = intention!.tool_calls![0]!;
    expect(call.name).toBe("days_until");
    expect(call.args.date).toBe(TARGET_DATE);
    // Each tool_call has a stable id linking it to its ToolMessage.
    expect(typeof call.id).toBe("string");

    // (2) RESULT: a typed ToolMessage carries the real tool result + artifact,
    // linked back to the intention by tool_call_id.
    const tms = toolMessages(messages);
    expect(tms).toHaveLength(1);
    const tm = tms[0]!;
    expect(tm.name).toBe("days_until");
    expect(tm.status).not.toBe("error");
    expect(tm.tool_call_id).toBe(call.id);
    const artifact = tm.artifact as { daysUntil: number; direction: string };
    expect(artifact.daysUntil).toBe(EXPECTED_DAYS_UNTIL);
    expect(artifact.direction).toBe("future");

    // The scripted model actually SAW the ToolMessage before the final turn —
    // i.e. tool state flowed back into the loop, it is not a fabricated summary.
    expect(recorder.calls).toHaveLength(2);
    expect(
      recorder.calls[1]!.some((m) => m instanceof ToolMessage)
    ).toBe(true);
  });
});

describe("Phase 6 — tool ERRORS are represented in the typed message stream", () => {
  it("a real tool failure yields a ToolMessage with status 'error' (not a thrown loop)", async () => {
    // days_until with an invalid calendar date (month 13). The REAL 6b tool
    // throws; the ToolNode surfaces it as an error ToolMessage.
    useScript([
      {
        content: "",
        toolCalls: [{ name: "days_until", args: { date: "2026-13-01" } }],
      },
      { content: "That date is invalid." },
    ]);

    const agent = createWeddingAgent();
    const result = await agent.invoke({
      messages: [new HumanMessage("days until 2026-13-01?")],
    });
    const messages = result.messages as BaseMessage[];

    const tms = toolMessages(messages);
    expect(tms).toHaveLength(1);
    expect(tms[0]!.name).toBe("days_until");
    expect(tms[0]!.status).toBe("error");
    expect(String(tms[0]!.content)).toMatch(/invalid|not a real calendar/i);

    // The error is DATA in the stream, not an exception — the loop completed.
    const last = messages[messages.length - 1] as AIMessage;
    expect(last).toBeInstanceOf(AIMessage);
    expect(last.tool_calls ?? []).toHaveLength(0);
  });
});
