import { describe, expect, it } from "vitest";
import {
  parseFrameLines,
  parseSseFrame,
  splitSseFrames,
} from "../src/lib/sseClient.js";
import { SSE_EVENT } from "../src/lib/sse-contract.js";

describe("splitSseFrames", () => {
  it("splits complete frames and keeps a trailing partial as rest", () => {
    const buffer =
      "event: init\ndata: {\"version\":2}\n\nevent: token\ndata: {\"text\":\"hi\"}\n\nevent: to";
    const { frames, rest } = splitSseFrames(buffer);
    expect(frames).toHaveLength(2);
    expect(frames[0]).toContain("event: init");
    expect(frames[1]).toContain("event: token");
    expect(rest).toBe("event: to");
  });

  it("normalizes CRLF terminators", () => {
    const buffer = "event: done\r\ndata: {\"text\":\"x\"}\r\n\r\n";
    const { frames, rest } = splitSseFrames(buffer);
    expect(frames).toHaveLength(1);
    expect(rest).toBe("");
  });
});

describe("parseFrameLines", () => {
  it("extracts event name and data, stripping one leading space", () => {
    const { eventName, data } = parseFrameLines(
      "event: token\ndata: {\"text\":\"hello\"}"
    );
    expect(eventName).toBe("token");
    expect(data).toBe('{"text":"hello"}');
  });

  it("defaults event name to message and ignores comment lines", () => {
    const { eventName, data } = parseFrameLines(": keep-alive\ndata: {}");
    expect(eventName).toBe("message");
    expect(data).toBe("{}");
  });

  it("joins multiple data lines with newlines", () => {
    const { data } = parseFrameLines("data: a\ndata: b");
    expect(data).toBe("a\nb");
  });
});

describe("parseSseFrame", () => {
  it("parses a typed init event", () => {
    const event = parseSseFrame("init", '{"version":2,"threadId":"t1"}');
    expect(event).not.toBeNull();
    expect(event?.type).toBe(SSE_EVENT.init);
    if (event?.type === SSE_EVENT.init) {
      expect(event.data.version).toBe(2);
      expect(event.data.threadId).toBe("t1");
    }
  });

  it("parses a discriminated tool call event", () => {
    const event = parseSseFrame(
      "tool",
      '{"phase":"call","name":"days_until","toolCallId":"c1","args":{"date":"2026-12-12"}}'
    );
    expect(event?.type).toBe(SSE_EVENT.tool);
    if (event?.type === SSE_EVENT.tool && event.data.phase === "call") {
      expect(event.data.name).toBe("days_until");
      expect(event.data.args["date"]).toBe("2026-12-12");
    }
  });

  it("returns null for an unknown event name (forward-compat ignore)", () => {
    expect(parseSseFrame("mystery", "{}")).toBeNull();
  });

  it("returns null for malformed JSON", () => {
    expect(parseSseFrame("token", "{not json")).toBeNull();
  });
});
