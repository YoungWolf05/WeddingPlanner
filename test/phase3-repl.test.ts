import { describe, it, expect } from "vitest";
import { AIMessageChunk } from "@langchain/core/messages";
import {
  ALLOWED_MODELS,
  DEFAULT_MODEL,
  isAllowedModel,
  selectInitialModel,
  isAbortError,
  parseLine,
  classifyTurnError,
  extractChunkText,
  streamTurn,
  BOT_LABEL,
  type WritableLike,
} from "../src/core/repl.js";
import type { ConversationalChain } from "../src/core/chain.js";
import { sessionConfig } from "../src/core/memory.js";

// Phase 3 REPL logic is tested deterministically as pure units — no readline, no
// TTY, no process.exit. streamTurn is driven with a fake graph + fake writable
// so streaming and abort are exercised without a real terminal or network.

// A writable sink that records every write() for assertions.
function fakeWritable(): WritableLike & { readonly written: string } {
  const buf: string[] = [];
  return {
    write(text: string) {
      buf.push(text);
      return true;
    },
    get written() {
      return buf.join("");
    },
  };
}

// Builds a fake ConversationalChain whose stream() yields the given
// [messageChunk, metadata] tuples. Optionally honors an AbortSignal by throwing
// mid-iteration, mirroring LangGraph's cancellation behavior.
function fakeGraph(
  tuples: Array<[AIMessageChunk, unknown]>,
  opts: { throwOnAbort?: boolean; throwError?: Error } = {}
): ConversationalChain {
  const stream = async (
    _input: unknown,
    config?: { signal?: AbortSignal }
  ): Promise<AsyncIterable<[AIMessageChunk, unknown]>> => {
    if (opts.throwError) throw opts.throwError;
    const signal = config?.signal;
    async function* gen() {
      for (const tuple of tuples) {
        if (opts.throwOnAbort && signal?.aborted) {
          const err = new Error("Aborted");
          err.name = "AbortError";
          throw err;
        }
        yield tuple;
      }
    }
    return gen();
  };
  // Only stream() is exercised by streamTurn; cast through unknown to satisfy the
  // rich ConversationalChain type without pulling in the full graph surface.
  return { stream } as unknown as ConversationalChain;
}

describe("Phase 3 — model allow-list", () => {
  it("accepts every documented allowed model", () => {
    for (const name of ALLOWED_MODELS) {
      expect(isAllowedModel(name)).toBe(true);
    }
  });

  it("rejects an unknown model", () => {
    expect(isAllowedModel("gpt-does-not-exist")).toBe(false);
    expect(isAllowedModel("")).toBe(false);
  });

  it("rejects gpt-5.1-chat — a known-invalid alias the proxy rejects (4c.1)", () => {
    // Locks the removal: the LiteLLM proxy/key rejects gpt-5.1-chat as an
    // invalid model name (see docs/capabilities/2026-07-27.md evidence), so it
    // must no longer be in the allow-list.
    expect(isAllowedModel("gpt-5.1-chat")).toBe(false);
  });

  it("advertises exactly the two valid aliases", () => {
    expect(ALLOWED_MODELS).toEqual(["claude-opus-4-8", "claude-sonnet-4-6"]);
  });

  it("selectInitialModel keeps a supported configured model", () => {
    expect(selectInitialModel("claude-opus-4-8")).toEqual({
      model: "claude-opus-4-8",
      fellBack: false,
    });
  });

  it("selectInitialModel falls back to DEFAULT_MODEL for an unsupported model", () => {
    expect(selectInitialModel("totally-made-up")).toEqual({
      model: DEFAULT_MODEL,
      fellBack: true,
    });
  });
});

describe("Phase 3 — command parsing (parseLine)", () => {
  it("treats whitespace-only input as empty", () => {
    expect(parseLine("   ")).toEqual({ kind: "empty" });
    expect(parseLine("")).toEqual({ kind: "empty" });
  });

  it("parses a plain message as a chat turn (trimmed)", () => {
    expect(parseLine("  beach wedding ideas  ")).toEqual({
      kind: "chat",
      text: "beach wedding ideas",
    });
  });

  it("parses /new with no argument", () => {
    expect(parseLine("/new")).toEqual({
      kind: "command",
      command: "new",
      arg: "",
    });
  });

  it("parses /model with an argument", () => {
    expect(parseLine("/model claude-opus-4-8")).toEqual({
      kind: "command",
      command: "model",
      arg: "claude-opus-4-8",
    });
  });

  it("parses /exit", () => {
    expect(parseLine("/exit")).toEqual({
      kind: "command",
      command: "exit",
      arg: "",
    });
  });

  it("parses an unknown command, preserving its name", () => {
    expect(parseLine("/bogus stuff here")).toEqual({
      kind: "command",
      command: "bogus",
      arg: "stuff here",
    });
  });

  it("parses a bare '/' as an empty command name and no argument", () => {
    expect(parseLine("/")).toEqual({
      kind: "command",
      command: "",
      arg: "",
    });
  });
});

describe("Phase 3 — abort/error classification", () => {
  it("isAbortError recognizes AbortError by name", () => {
    const err = new Error("cancelled");
    err.name = "AbortError";
    expect(isAbortError(err)).toBe(true);
  });

  it("isAbortError recognizes abort-shaped messages", () => {
    expect(isAbortError(new Error("The operation was aborted"))).toBe(true);
  });

  it("isAbortError returns false for ordinary errors and non-errors", () => {
    expect(isAbortError(new Error("network exploded"))).toBe(false);
    expect(isAbortError("not an error")).toBe(false);
    expect(isAbortError(undefined)).toBe(false);
  });

  it("classifyTurnError treats an aborted signal as interrupted", () => {
    expect(classifyTurnError(new Error("boom"), true)).toBe("interrupted");
  });

  it("classifyTurnError treats an abort-shaped error as interrupted", () => {
    const err = new Error("aborted");
    expect(classifyTurnError(err, false)).toBe("interrupted");
  });

  it("classifyTurnError treats a genuine error as failed", () => {
    expect(classifyTurnError(new Error("network exploded"), false)).toBe(
      "failed"
    );
  });
});

describe("Phase 3 — extractChunkText", () => {
  it("returns string content as-is", () => {
    expect(extractChunkText("hello")).toBe("hello");
  });

  it("concatenates text blocks and ignores non-text blocks", () => {
    const content = [
      { type: "text", text: "Hello " },
      { type: "image_url", image_url: "http://x" },
      { type: "text", text: "world" },
      { notText: true },
    ] as unknown as AIMessageChunk["content"];
    expect(extractChunkText(content)).toBe("Hello world");
  });

  it("returns empty string for block arrays with no text", () => {
    const content = [
      { type: "image_url", image_url: "http://x" },
    ] as unknown as AIMessageChunk["content"];
    expect(extractChunkText(content)).toBe("");
  });
});

describe("Phase 3 — streamTurn", () => {
  const cfg = sessionConfig("stream-thread");

  it("writes the bot label then incremental string chunks", async () => {
    const graph = fakeGraph([
      [new AIMessageChunk({ content: "Hello" }), {}],
      [new AIMessageChunk({ content: ", " }), {}],
      [new AIMessageChunk({ content: "Aria!" }), {}],
    ]);
    const out = fakeWritable();
    const controller = new AbortController();

    await streamTurn(graph, "hi", cfg, controller.signal, out);

    expect(out.written).toBe(`${BOT_LABEL}Hello, Aria!\n`);
  });

  it("extracts text from block-array chunk content, ignoring non-text blocks", async () => {
    const blockContent = [
      { type: "text", text: "Block A " },
      { type: "image_url", image_url: "http://x" },
      { type: "text", text: "Block B" },
    ] as unknown as AIMessageChunk["content"];
    const graph = fakeGraph([[new AIMessageChunk({ content: blockContent }), {}]]);
    const out = fakeWritable();
    const controller = new AbortController();

    await streamTurn(graph, "hi", cfg, controller.signal, out);

    expect(out.written).toBe(`${BOT_LABEL}Block A Block B\n`);
  });

  it("stops the turn when the AbortSignal fires and surfaces an abort error", async () => {
    const graph = fakeGraph(
      [
        [new AIMessageChunk({ content: "first" }), {}],
        [new AIMessageChunk({ content: "second" }), {}],
      ],
      { throwOnAbort: true }
    );
    const out = fakeWritable();
    const controller = new AbortController();
    controller.abort(); // pre-aborted: no chunks should be emitted.

    let thrown: unknown;
    try {
      await streamTurn(graph, "hi", cfg, controller.signal, out);
    } catch (err) {
      thrown = err;
    }

    // The turn was interrupted, not completed.
    expect(thrown).toBeInstanceOf(Error);
    expect(classifyTurnError(thrown, controller.signal.aborted)).toBe(
      "interrupted"
    );
    // Only the label was written before the abort; no chunk text leaked.
    expect(out.written).toBe(BOT_LABEL);
  });

  it("propagates a non-abort error, classified as a failure", async () => {
    const graph = fakeGraph([], {
      throwError: new Error("upstream model exploded"),
    });
    const out = fakeWritable();
    const controller = new AbortController();

    let thrown: unknown;
    try {
      await streamTurn(graph, "hi", cfg, controller.signal, out);
    } catch (err) {
      thrown = err;
    }

    expect(thrown).toBeInstanceOf(Error);
    expect(classifyTurnError(thrown, controller.signal.aborted)).toBe("failed");
  });
});
