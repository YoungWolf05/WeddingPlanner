import { describe, expect, it } from "vitest";
import { newAssistantTurn, turnReducers } from "../src/lib/conversation.js";
import type { SseCitation, SseToolEvent } from "../src/lib/sse-contract.js";

const citation: SseCitation = {
  marker: 1,
  chunkId: "chunk-1",
  documentId: "doc-1",
  sourceUri: "knowledge/corpus/a.md",
  chunkIndex: 0,
  score: 0.87,
  contentHash: "hash-1",
};

describe("conversation reducers", () => {
  it("accumulates token text immutably", () => {
    const t0 = newAssistantTurn("a1", "hi");
    const t1 = turnReducers.appendToken(t0, "Hel");
    const t2 = turnReducers.appendToken(t1, "lo");
    expect(t2.text).toBe("Hello");
    expect(t0.text).toBe(""); // original untouched
  });

  it("sets trusted citations and evidenceStatus from the typed event", () => {
    const t = turnReducers.setCitations(
      newAssistantTurn("a1", "q"),
      [citation],
      "supported"
    );
    expect(t.evidenceStatus).toBe("supported");
    expect(t.citations).toEqual([citation]);
  });

  it("represents an insufficient turn with no citations", () => {
    const t = turnReducers.setCitations(
      newAssistantTurn("a1", "q"),
      [],
      "insufficient"
    );
    expect(t.evidenceStatus).toBe("insufficient");
    expect(t.citations).toEqual([]);
  });

  it("appends tool events in order", () => {
    const call: SseToolEvent = {
      phase: "call",
      name: "days_until",
      toolCallId: "c1",
      args: { date: "2026-12-12" },
    };
    const result: SseToolEvent = {
      phase: "result",
      name: "days_until",
      toolCallId: "c1",
      status: "ok",
      content: "300 days",
    };
    let t = turnReducers.addTool(newAssistantTurn("a1", "q"), call);
    t = turnReducers.addTool(t, result);
    expect(t.toolEvents).toHaveLength(2);
    expect(t.toolEvents[0]?.phase).toBe("call");
    expect(t.toolEvents[1]?.phase).toBe("result");
  });

  it("finishDone falls back to full text only when no tokens streamed", () => {
    const streamed = turnReducers.appendToken(
      newAssistantTurn("a1", "q"),
      "partial"
    );
    expect(turnReducers.finishDone(streamed, "FULL").text).toBe("partial");
    expect(turnReducers.finishDone(newAssistantTurn("a2", "q"), "FULL").text).toBe(
      "FULL"
    );
  });

  it("fail and cancel set the right status", () => {
    expect(turnReducers.fail(newAssistantTurn("a1", "q"), "boom").status).toBe(
      "error"
    );
    expect(turnReducers.cancel(newAssistantTurn("a1", "q")).status).toBe(
      "canceled"
    );
  });
});
