import { describe, it, expect, beforeEach, vi } from "vitest";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import { recordedCalls, resetRecordedCalls } from "./helpers/fake-model.js";

// Mock ONLY the model boundary (createChatModel). Everything else — prompt
// composition, StringOutputParser, RunnableLambda normalization — runs for real
// so these tests exercise the true Phase 1 wiring, just with a deterministic,
// offline fake LLM instead of a live LiteLLM call.
//
// vi.mock is hoisted above imports; the factory imports the shared helper (also
// hoist-safe because it is an async dynamic import) which records the exact
// messages the LLM receives.
vi.mock("../src/core/model.js", async () => {
  const { makeFakeChatModel } = await import("./helpers/fake-model.js");
  return {
    createChatModel: () =>
      makeFakeChatModel({ responses: ["Mocked Aria reply."] }),
  };
});

const { createWeddingPlannerChain } = await import("../src/core/chain.js");
const { WEDDING_PLANNER_SYSTEM_PROMPT } = await import(
  "../src/core/prompts.js"
);

describe("Phase 1 — createWeddingPlannerChain", () => {
  beforeEach(() => {
    resetRecordedCalls();
  });

  it("returns a Runnable that produces the model's text via StringOutputParser", async () => {
    const chain = createWeddingPlannerChain();

    const result = await chain.invoke({ input: "beach wedding ideas" });

    // StringOutputParser unwraps the AIMessage to a plain string.
    expect(typeof result).toBe("string");
    expect(result).toBe("Mocked Aria reply.");
  });

  it("composes the Aria system persona and the human input into the prompt", async () => {
    const chain = createWeddingPlannerChain();

    await chain.invoke({ input: "help me pick a venue" });

    expect(recordedCalls).toHaveLength(1);
    const messages: BaseMessage[] = recordedCalls[0]!;

    // First message must be the Aria system persona verbatim.
    const system = messages[0]!;
    expect(system).toBeInstanceOf(SystemMessage);
    expect(system.content).toBe(WEDDING_PLANNER_SYSTEM_PROMPT);
    expect(String(system.content)).toContain('"Aria"');

    // Last message must be the human input.
    const human = messages[messages.length - 1]!;
    expect(human).toBeInstanceOf(HumanMessage);
    expect(human.content).toBe("help me pick a venue");
  });

  it("passes prior history through the MessagesPlaceholder in order", async () => {
    const chain = createWeddingPlannerChain();

    const history: BaseMessage[] = [
      new HumanMessage("We have 120 guests."),
      new AIMessage("Great, 120 guests noted!"),
    ];

    await chain.invoke({ input: "what's next?", history });

    const messages: BaseMessage[] = recordedCalls[0]!;
    // Expect: [system, ...history, human] = 4 messages.
    expect(messages).toHaveLength(4);
    expect(messages[0]).toBeInstanceOf(SystemMessage);
    expect(messages[1]!.content).toBe("We have 120 guests.");
    expect(messages[2]!.content).toBe("Great, 120 guests noted!");
    expect(messages[3]!.content).toBe("what's next?");
  });

  it("defaults history to [] when omitted (normalizeInput)", async () => {
    const chain = createWeddingPlannerChain();

    await chain.invoke({ input: "no history here" });

    const messages: BaseMessage[] = recordedCalls[0]!;
    // Only [system, human] when no history is supplied — the placeholder
    // contributed nothing.
    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(SystemMessage);
    expect(messages[1]).toBeInstanceOf(HumanMessage);
  });
});
