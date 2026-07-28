import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { config } from "../src/config.js";
import {
  ConsoleSink,
  FileSink,
  MemorySink,
  TracingCallbackHandler,
  buildEvent,
  createTracingHandler,
  extractUsage,
  resolveTraceConfig,
  selectSink,
  serializeEvent,
  type TraceEvent,
} from "../src/core/tracing.js";
import {
  EMAIL_PLACEHOLDER,
  KEY_PLACEHOLDER,
  PHONE_PLACEHOLDER,
  URL_PLACEHOLDER,
} from "../src/core/redaction.js";

// Phase 4 (increment 4d) — OFFLINE tracing tests.
//
// Covers: enablement gating + precedence, pure event construction (schema,
// content gating, redacted error reasons), JSONL serialization, injectable
// sinks (memory / temp-dir file / console), swallowed sink failures, and the
// createChatModel() integration wiring (handler present iff enabled). NO network
// call is ever made; the LLM callback surface is exercised via the handler
// methods directly with synthetic inputs (no real credentials).

// --- Enablement gating & precedence -----------------------------------------

describe("Phase 4 — tracing: resolveTraceConfig", () => {
  it("is OFF by default (no flags)", () => {
    const cfg = resolveTraceConfig({});
    expect(cfg.enabled).toBe(false);
    expect(cfg.captureContent).toBe(false);
  });

  it("enables via LITELLM_TRACE=1", () => {
    expect(resolveTraceConfig({ LITELLM_TRACE: "1" }).enabled).toBe(true);
  });

  it("accepts TRACE=1 as an alias when LITELLM_TRACE is unset", () => {
    expect(resolveTraceConfig({ TRACE: "1" }).enabled).toBe(true);
  });

  it("accepts true/yes/on (case-insensitive) as ON", () => {
    for (const v of ["true", "YES", "On", "TRUE"]) {
      expect(resolveTraceConfig({ LITELLM_TRACE: v }).enabled).toBe(true);
    }
  });

  it("LITELLM_TRACE takes precedence over TRACE (explicit 0 disables)", () => {
    const cfg = resolveTraceConfig({ LITELLM_TRACE: "0", TRACE: "1" });
    expect(cfg.enabled).toBe(false);
  });

  it("falls back to TRACE only when LITELLM_TRACE is unset/empty", () => {
    expect(resolveTraceConfig({ LITELLM_TRACE: "", TRACE: "1" }).enabled).toBe(
      true
    );
  });

  it("content capture is OFF by default even when tracing is enabled", () => {
    expect(resolveTraceConfig({ LITELLM_TRACE: "1" }).captureContent).toBe(
      false
    );
  });

  it("content capture requires tracing enabled", () => {
    // Content flag on but tracing off -> capture stays off.
    expect(
      resolveTraceConfig({ LITELLM_TRACE_CONTENT: "1" }).captureContent
    ).toBe(false);
    expect(
      resolveTraceConfig({ LITELLM_TRACE: "1", LITELLM_TRACE_CONTENT: "1" })
        .captureContent
    ).toBe(true);
  });
});

// --- Event construction (pure) ----------------------------------------------

describe("Phase 4 — tracing: buildEvent", () => {
  const base = {
    timestamp: "2026-07-27T12:00:00.000Z",
    model: "claude-sonnet-4-6",
    operation: "invoke" as const,
    latencyMs: 42,
    streaming: false,
  };

  it("builds a success event with the expected fields", () => {
    const ev = buildEvent({
      ...base,
      usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
    });
    expect(ev.timestamp).toBe(base.timestamp);
    // ISO check.
    expect(new Date(ev.timestamp).toISOString()).toBe(ev.timestamp);
    expect(ev.model).toBe("claude-sonnet-4-6");
    expect(ev.operation).toBe("invoke");
    expect(ev.latencyMs).toBe(42);
    expect(ev.outcome).toBe("success");
    expect(ev.streaming).toBe(false);
    expect(ev.usage).toEqual({ inputTokens: 10, outputTokens: 5, totalTokens: 15 });
    expect(ev.errorReason).toBeUndefined();
    expect(ev.content).toBeUndefined();
  });

  it("marks streaming operation and flag", () => {
    const ev = buildEvent({ ...base, operation: "stream", streaming: true });
    expect(ev.operation).toBe("stream");
    expect(ev.streaming).toBe(true);
  });

  it("omits usage when not provided", () => {
    const ev = buildEvent(base);
    expect(ev.usage).toBeUndefined();
  });

  it("builds an error event with a REDACTED error reason (metadata-only)", () => {
    const err = new Error(
      `boom at ${config.baseURL} key ${config.apiKey} mail a@b.io tel 555-123-4567`
    );
    const ev = buildEvent({ ...base, error: err });
    expect(ev.outcome).toBe("error");
    expect(ev.errorReason).toBeDefined();
    const reason = ev.errorReason!;
    expect(reason).not.toContain(config.apiKey);
    expect(reason).not.toContain(config.baseURL);
    expect(reason).not.toContain("a@b.io");
    expect(reason).toContain(KEY_PLACEHOLDER);
    expect(reason).toContain(URL_PLACEHOLDER);
    expect(reason).toContain(EMAIL_PLACEHOLDER);
    expect(reason).toContain(PHONE_PLACEHOLDER);
    // Metadata-only: no content even on error.
    expect(ev.content).toBeUndefined();
  });

  it("NEVER includes content when rawContent is not supplied (capture off)", () => {
    const ev = buildEvent({ ...base });
    expect(ev.content).toBeUndefined();
    expect(JSON.stringify(ev)).not.toContain("content");
  });

  it("captures content ONLY when supplied, and redacts it", () => {
    const ev = buildEvent({
      ...base,
      rawContent: {
        prompt: `email me at bride@example.com or ${config.apiKey}`,
        response: "here is your plan",
      },
    });
    expect(ev.content).toBeDefined();
    expect(ev.content!.prompt).toContain(EMAIL_PLACEHOLDER);
    expect(ev.content!.prompt).toContain(KEY_PLACEHOLDER);
    expect(ev.content!.prompt).not.toContain("bride@example.com");
    expect(ev.content!.response).toBe("here is your plan");
  });
});

// --- extractUsage -----------------------------------------------------------

describe("Phase 4 — tracing: extractUsage", () => {
  it("reads usage_metadata from the first generation's message", () => {
    const usage = extractUsage({
      generations: [
        [
          {
            text: "hi",
            message: {
              usage_metadata: {
                input_tokens: 7,
                output_tokens: 3,
                total_tokens: 10,
              },
            },
          } as never,
        ],
      ],
    });
    expect(usage).toEqual({ inputTokens: 7, outputTokens: 3, totalTokens: 10 });
  });

  it("falls back to llmOutput.tokenUsage (OpenAI shape)", () => {
    const usage = extractUsage({
      generations: [[{ text: "hi" } as never]],
      llmOutput: {
        tokenUsage: { promptTokens: 4, completionTokens: 6, totalTokens: 10 },
      },
    });
    expect(usage).toEqual({ inputTokens: 4, outputTokens: 6, totalTokens: 10 });
  });

  it("returns undefined when no usage is present", () => {
    expect(extractUsage({ generations: [[{ text: "hi" } as never]] })).toBeUndefined();
  });
});

// --- JSONL serialization ----------------------------------------------------

describe("Phase 4 — tracing: serializeEvent", () => {
  it("produces exactly one valid JSON object per line", () => {
    const ev: TraceEvent = {
      timestamp: "2026-07-27T12:00:00.000Z",
      operation: "invoke",
      latencyMs: 1,
      outcome: "success",
      streaming: false,
    };
    const line = serializeEvent(ev);
    expect(line.endsWith("\n")).toBe(true);
    // Exactly one newline (the terminator).
    expect(line.split("\n").filter((s) => s.length > 0)).toHaveLength(1);
    const parsed = JSON.parse(line.trim());
    expect(parsed).toEqual(ev);
  });

  it("escapes embedded newlines so a multiline reason stays on one line", () => {
    const ev = buildEvent({
      timestamp: "2026-07-27T12:00:00.000Z",
      operation: "invoke",
      latencyMs: 1,
      streaming: false,
      // redactText collapses whitespace, but prove serialization is line-safe.
      rawContent: { response: "a\nb\nc" },
    });
    const line = serializeEvent(ev);
    expect(line.split("\n").filter((s) => s.length > 0)).toHaveLength(1);
    expect(() => JSON.parse(line.trim())).not.toThrow();
  });
});

// --- Sinks ------------------------------------------------------------------

describe("Phase 4 — tracing: MemorySink", () => {
  it("records written events", () => {
    const sink = new MemorySink();
    const ev = buildEvent({
      timestamp: "2026-07-27T12:00:00.000Z",
      operation: "invoke",
      latencyMs: 1,
      streaming: false,
    });
    sink.write(ev);
    expect(sink.events).toHaveLength(1);
    expect(sink.events[0]).toEqual(ev);
  });
});

describe("Phase 4 — tracing: FileSink (temp dir)", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(path.join(tmpdir(), "wp-trace-"));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("creates the directory if missing and appends JSONL lines", async () => {
    const subdir = path.join(dir, "nested", "logs");
    const sink = new FileSink({ dir: subdir, fileName: "trace.jsonl" });
    const ev1 = buildEvent({
      timestamp: "2026-07-27T12:00:00.000Z",
      operation: "invoke",
      latencyMs: 1,
      streaming: false,
    });
    const ev2 = buildEvent({
      timestamp: "2026-07-27T12:00:01.000Z",
      operation: "stream",
      latencyMs: 2,
      streaming: true,
    });
    await sink.write(ev1);
    await sink.write(ev2);
    const content = await readFile(path.join(subdir, "trace.jsonl"), "utf8");
    const lines = content.split("\n").filter((l) => l.length > 0);
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0]!)).toEqual(ev1);
    expect(JSON.parse(lines[1]!)).toEqual(ev2);
  });

  it("does NOT throw when the write fails (path is an existing file)", async () => {
    // Point the sink's dir at a path that cannot be a directory: create a file
    // and use it as the 'dir', so mkdir(recursive) fails.
    const filePath = path.join(dir, "not-a-dir");
    await writeFile(filePath, "x", "utf8");
    const sink = new FileSink({ dir: filePath, fileName: "trace.jsonl" });
    // Must resolve (not reject) despite the underlying error.
    await expect(
      sink.write(
        buildEvent({
          timestamp: "2026-07-27T12:00:00.000Z",
          operation: "invoke",
          latencyMs: 1,
          streaming: false,
        })
      )
    ).resolves.toBeUndefined();
  });
});

describe("Phase 4 — tracing: ConsoleSink", () => {
  it("does not throw and writes a JSONL line to stderr", () => {
    const sink = new ConsoleSink();
    const writes: string[] = [];
    const orig = process.stderr.write.bind(process.stderr);
    process.stderr.write = ((chunk: string) => {
      writes.push(String(chunk));
      return true;
    }) as typeof process.stderr.write;
    try {
      sink.write(
        buildEvent({
          timestamp: "2026-07-27T12:00:00.000Z",
          operation: "invoke",
          latencyMs: 1,
          streaming: false,
        })
      );
    } finally {
      process.stderr.write = orig;
    }
    expect(writes).toHaveLength(1);
    expect(() => JSON.parse(writes[0]!.trim())).not.toThrow();
  });
});

describe("Phase 4 — tracing: selectSink", () => {
  it("defaults to a FileSink", () => {
    expect(selectSink({})).toBeInstanceOf(FileSink);
  });
  it("selects ConsoleSink when LITELLM_TRACE_SINK=console", () => {
    expect(selectSink({ LITELLM_TRACE_SINK: "console" })).toBeInstanceOf(
      ConsoleSink
    );
  });
});

// --- A failing sink does not propagate out of the handler -------------------

describe("Phase 4 — tracing: handler swallows sink failures", () => {
  class ThrowingSink {
    write(): void {
      throw new Error("sink exploded");
    }
  }
  class RejectingSink {
    write(): Promise<void> {
      return Promise.reject(new Error("async sink exploded"));
    }
  }

  it("does not throw when a synchronous sink write throws", () => {
    const handler = new TracingCallbackHandler({
      sink: new ThrowingSink(),
      captureContent: false,
      streaming: false,
    });
    const serialized = { lc: 1, type: "not_implemented", id: ["x"] } as never;
    handler.handleLLMStart(serialized, ["hi"], "run-1");
    expect(() =>
      handler.handleLLMEnd(
        { generations: [[{ text: "ok" } as never]] },
        "run-1"
      )
    ).not.toThrow();
  });

  it("does not reject the process when an async sink rejects", async () => {
    const handler = new TracingCallbackHandler({
      sink: new RejectingSink(),
      captureContent: false,
      streaming: false,
    });
    const serialized = { lc: 1, type: "not_implemented", id: ["x"] } as never;
    handler.handleLLMStart(serialized, ["hi"], "run-2");
    // emit() catches the rejection internally; give the microtask queue a turn.
    handler.handleLLMEnd({ generations: [[{ text: "ok" } as never]] }, "run-2");
    await new Promise((r) => setTimeout(r, 10));
    // Reaching here without an unhandled rejection is the assertion.
    expect(true).toBe(true);
  });
});

// --- Handler emits well-formed events for invoke and stream -----------------

describe("Phase 4 — tracing: handler event emission", () => {
  const serialized = { lc: 1, type: "not_implemented", id: ["x"] } as never;

  it("emits a success invoke event with model/usage", () => {
    const sink = new MemorySink();
    const handler = new TracingCallbackHandler({
      sink,
      captureContent: false,
      model: "claude-opus-4-8",
      streaming: false,
    });
    handler.handleLLMStart(serialized, ["hi"], "r1");
    handler.handleLLMEnd(
      {
        generations: [
          [
            {
              text: "pong",
              message: {
                usage_metadata: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
              },
            } as never,
          ],
        ],
      },
      "r1"
    );
    expect(sink.events).toHaveLength(1);
    const ev = sink.events[0]!;
    expect(ev.operation).toBe("invoke");
    expect(ev.streaming).toBe(false);
    expect(ev.outcome).toBe("success");
    expect(ev.model).toBe("claude-opus-4-8");
    expect(ev.usage).toEqual({ inputTokens: 1, outputTokens: 1, totalTokens: 2 });
    expect(ev.content).toBeUndefined();
  });

  it("emits a stream event when constructed in streaming mode", () => {
    const sink = new MemorySink();
    const handler = new TracingCallbackHandler({
      sink,
      captureContent: false,
      model: "claude-sonnet-4-6",
      streaming: true,
    });
    handler.handleChatModelStart(serialized, [[{ content: "hi" }]], "r2");
    handler.handleLLMEnd({ generations: [[{ text: "1 2 3" } as never]] }, "r2");
    expect(sink.events[0]!.operation).toBe("stream");
    expect(sink.events[0]!.streaming).toBe(true);
  });

  it("emits a redacted error event", () => {
    const sink = new MemorySink();
    const handler = new TracingCallbackHandler({
      sink,
      captureContent: false,
      streaming: false,
    });
    handler.handleLLMStart(serialized, ["hi"], "r3");
    handler.handleLLMError(
      new Error(`fail ${config.apiKey}`),
      "r3"
    );
    const ev = sink.events[0]!;
    expect(ev.outcome).toBe("error");
    expect(ev.errorReason).toContain(KEY_PLACEHOLDER);
    expect(ev.errorReason).not.toContain(config.apiKey);
  });

  it("captures redacted content when capture is on", () => {
    const sink = new MemorySink();
    const handler = new TracingCallbackHandler({
      sink,
      captureContent: true,
      streaming: false,
    });
    handler.handleLLMStart(serialized, ["contact me at guest@example.com"], "r4");
    handler.handleLLMEnd({ generations: [[{ text: "sure" } as never]] }, "r4");
    const ev = sink.events[0]!;
    expect(ev.content).toBeDefined();
    expect(ev.content!.prompt).toContain(EMAIL_PLACEHOLDER);
    expect(ev.content!.response).toBe("sure");
  });
});

// --- createTracingHandler integration decision ------------------------------

describe("Phase 4 — tracing: createTracingHandler gating", () => {
  it("returns undefined (no-op) when tracing is disabled", () => {
    const handler = createTracingHandler({
      model: "claude-sonnet-4-6",
      streaming: false,
      env: {},
    });
    expect(handler).toBeUndefined();
  });

  it("returns a handler wired to the injected sink when enabled", () => {
    const sink = new MemorySink();
    const handler = createTracingHandler({
      model: "claude-sonnet-4-6",
      streaming: false,
      env: { LITELLM_TRACE: "1" },
      sink,
    });
    expect(handler).toBeInstanceOf(TracingCallbackHandler);
    // Drive one event through to prove the sink is wired.
    const serialized = { lc: 1, type: "not_implemented", id: ["x"] } as never;
    handler!.handleLLMStart(serialized, ["hi"], "r");
    handler!.handleLLMEnd({ generations: [[{ text: "ok" } as never]] }, "r");
    expect(sink.events).toHaveLength(1);
  });
});
