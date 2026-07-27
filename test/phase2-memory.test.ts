import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { recordedCalls, resetRecordedCalls } from "./helpers/fake-model.js";

// Mock only the model boundary; the real LangGraph StateGraph, MessagesAnnotation
// reducer, and MemorySaver checkpointer all run so we test genuine multi-turn
// memory behavior, not a stub of it.
//
// Each turn's fake response is deterministic; the recorder captures the messages
// the model receives per turn so we can prove history accumulation + isolation.
vi.mock("../src/core/model.js", async () => {
  const { makeFakeChatModel } = await import("./helpers/fake-model.js");
  return {
    createChatModel: () =>
      makeFakeChatModel({
        // Distinct replies so accumulated history is unambiguous across turns.
        responses: ["reply-1", "reply-2", "reply-3", "reply-4"],
      }),
  };
});

const { createConversationalChain } = await import("../src/core/chain.js");
const { sessionConfig } = await import("../src/core/memory.js");
const { WEDDING_PLANNER_SYSTEM_PROMPT } = await import(
  "../src/core/prompts.js"
);

async function invokeTurn(
  graph: Awaited<ReturnType<typeof createConversationalChain>>,
  text: string,
  cfg: ReturnType<typeof sessionConfig>
) {
  return graph.invoke({ messages: [new HumanMessage(text)] }, cfg);
}

describe("Phase 2 — conversational memory", () => {
  beforeEach(() => {
    resetRecordedCalls();
  });

  it("sessionConfig returns { configurable: { thread_id } }", () => {
    expect(sessionConfig("abc-123")).toEqual({
      configurable: { thread_id: "abc-123" },
    });
  });

  it("compiles and injects the Aria persona on every call (callModel)", async () => {
    const graph = createConversationalChain();
    const cfg = sessionConfig("persona-thread");

    await invokeTurn(graph, "hello", cfg);
    await invokeTurn(graph, "again", cfg);

    expect(recordedCalls).toHaveLength(2);
    for (const messages of recordedCalls) {
      const system = messages[0]!;
      expect(system).toBeInstanceOf(SystemMessage);
      expect(system.content).toBe(WEDDING_PLANNER_SYSTEM_PROMPT);
    }
  });

  it("accumulates history across two turns on the SAME thread_id", async () => {
    const graph = createConversationalChain();
    const cfg = sessionConfig("same-thread");

    await invokeTurn(graph, "We have 120 guests.", cfg);
    await invokeTurn(graph, "What's next?", cfg);

    expect(recordedCalls).toHaveLength(2);

    // Turn 1: [system, human(120 guests)].
    const first: BaseMessage[] = recordedCalls[0]!;
    expect(first).toHaveLength(2);
    expect(first[0]).toBeInstanceOf(SystemMessage);
    expect(first[1]).toBeInstanceOf(HumanMessage);
    expect(first[1]!.content).toBe("We have 120 guests.");

    // Turn 2 must SEE the prior turn's messages (history accumulation):
    // [system, human(120), ai(reply-1), human(What's next?)].
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

  it("isolates history between DIFFERENT thread_ids", async () => {
    const graph = createConversationalChain();

    await invokeTurn(graph, "Secret from thread A.", sessionConfig("thread-A"));
    // A different thread must NOT see thread A's history.
    await invokeTurn(graph, "Fresh start on B.", sessionConfig("thread-B"));

    expect(recordedCalls).toHaveLength(2);

    const threadB: BaseMessage[] = recordedCalls[1]!;
    // thread-B's first turn: only [system, human(Fresh start on B.)].
    expect(threadB).toHaveLength(2);
    expect(threadB[0]).toBeInstanceOf(SystemMessage);
    expect(threadB[1]!.content).toBe("Fresh start on B.");

    // Sanity: none of thread-B's messages leak thread-A content.
    const leaked = threadB.some((m) =>
      String(m.content).includes("Secret from thread A.")
    );
    expect(leaked).toBe(false);
  });
});
