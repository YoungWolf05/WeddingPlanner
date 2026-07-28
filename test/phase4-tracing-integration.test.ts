import { describe, it, expect, afterEach } from "vitest";
import { createChatModel } from "../src/core/model.js";
import { TracingCallbackHandler } from "../src/core/tracing.js";

// Phase 4 (increment 4d) — OFFLINE integration test at the createChatModel()
// boundary.
//
// Proves tracing is wired transparently: when the enable flag is OFF, the
// returned ChatOpenAI carries NO tracing handler (true no-op — construction is
// unchanged); when ON, exactly one TracingCallbackHandler is attached via the
// model's `callbacks`. createTracingHandler() reads process.env at call time, so
// toggling the env var per test is sufficient — no module reload, NO network
// call, and no real credentials (the fake dummy env from test/setup/env.ts lets
// config load, but createChatModel is never invoked against the network here).

// Collect any TracingCallbackHandler attached to a model's callbacks, tolerating
// the several shapes `callbacks` can take (undefined | array | manager).
function tracingHandlers(model: {
  callbacks?: unknown;
}): TracingCallbackHandler[] {
  const cb = model.callbacks;
  if (!cb) return [];
  const list = Array.isArray(cb)
    ? cb
    : ((cb as { handlers?: unknown[] }).handlers ?? []);
  return list.filter(
    (h): h is TracingCallbackHandler => h instanceof TracingCallbackHandler
  );
}

const TRACE_KEYS = [
  "LITELLM_TRACE",
  "TRACE",
  "LITELLM_TRACE_CONTENT",
  "LITELLM_TRACE_SINK",
] as const;

function clearTraceEnv(): void {
  for (const k of TRACE_KEYS) delete process.env[k];
}

describe("Phase 4 — tracing: createChatModel() integration", () => {
  afterEach(() => {
    clearTraceEnv();
  });

  it("attaches NO tracing handler when tracing is disabled (default)", () => {
    clearTraceEnv();
    const model = createChatModel({ model: "claude-sonnet-4-6" });
    expect(tracingHandlers(model)).toHaveLength(0);
  });

  it("attaches exactly one tracing handler when LITELLM_TRACE=1", () => {
    clearTraceEnv();
    process.env.LITELLM_TRACE = "1";
    // Route the enabled handler to the console sink so this test never writes a
    // trace file into the repo working tree.
    process.env.LITELLM_TRACE_SINK = "console";
    const model = createChatModel({ model: "claude-opus-4-8" });
    expect(tracingHandlers(model)).toHaveLength(1);
  });

  it("respects LITELLM_TRACE=0 precedence over TRACE=1 (stays a no-op)", () => {
    clearTraceEnv();
    process.env.LITELLM_TRACE = "0";
    process.env.TRACE = "1";
    const model = createChatModel();
    expect(tracingHandlers(model)).toHaveLength(0);
  });

  it("preserves createChatModel's return-type surface (bindTools/withStructuredOutput/pipe/stream)", () => {
    clearTraceEnv();
    const model = createChatModel();
    // Type/shape contract callers depend on must still be present regardless of
    // tracing state.
    expect(typeof model.bindTools).toBe("function");
    expect(typeof model.withStructuredOutput).toBe("function");
    expect(typeof model.pipe).toBe("function");
    expect(typeof model.stream).toBe("function");
    expect(typeof model.invoke).toBe("function");
  });
});
