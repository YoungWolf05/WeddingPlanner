import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import type { LLMResult } from "@langchain/core/outputs";
import { appendFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { redactError, redactText } from "./redaction.js";

// Phase 4 (increment 4d) — local, self-contained tracing with ALWAYS-ON
// secret/PII redaction, integrated at the createChatModel() boundary.
//
// Design summary (see docs/roadmap.md Phase 4 / 4d):
//   - OFF by default; enabled via LITELLM_TRACE=1 (alias: TRACE=1).
//   - METADATA ONLY by default. Explicit opt-in (LITELLM_TRACE_CONTENT=1)
//     captures prompt/response content, which STILL passes through redaction.
//   - Pluggable sinks: default JSONL file sink (gitignored logs/), optional
//     console sink, and an in-memory sink for tests. Sink failures NEVER
//     propagate — tracing must never break a normal run.
//   - Integration is a LangChain callback handler attached only when tracing is
//     enabled, so the returned model behaves identically whether tracing is on
//     or off and no second provider client is constructed (4b guard safe).
//
// The pure logic (config resolution, event construction, JSONL serialization,
// sink selection) is separated from the file I/O so it is unit-testable offline.

// --- Trace event schema -----------------------------------------------------

export type TraceOperation = "invoke" | "stream";
export type TraceOutcome = "success" | "error";

// Token/usage metadata, when the provider/response exposes it. All optional so a
// provider that omits usage still yields a well-formed event.
export interface TraceUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

// One trace event per app LLM call. METADATA ONLY unless content capture is on.
export interface TraceEvent {
  // ISO-8601 timestamp of when the call STARTED (UTC).
  timestamp: string;
  // Model alias the call targeted (e.g. "claude-sonnet-4-6"), when known.
  model?: string;
  // Operation type: a non-streaming invoke or a streaming call.
  operation: TraceOperation;
  // Wall-clock latency in milliseconds from start to end/error.
  latencyMs: number;
  // success | error.
  outcome: TraceOutcome;
  // Whether the model was constructed in streaming mode.
  streaming: boolean;
  // Token usage metadata, present only when the response exposed it.
  usage?: TraceUsage;
  // For errors: a REDACTED, length-capped single-line reason. Never present on
  // success.
  errorReason?: string;
  // Opt-in captured content, ALWAYS redacted. Absent unless content capture is
  // explicitly enabled AND content was available.
  content?: TraceContent;
}

// Opt-in, redacted content capture. Minimal and clearly gated.
export interface TraceContent {
  prompt?: string;
  response?: string;
}

// --- Enablement / config resolution (pure) ----------------------------------

export interface TraceConfig {
  enabled: boolean;
  captureContent: boolean;
}

// Interpret a raw env value as an on/off flag. Accepts "1", "true", "yes", "on"
// (case-insensitive) as ON; everything else (including unset) is OFF.
function isFlagOn(value: string | undefined): boolean {
  if (!value) return false;
  const v = value.trim().toLowerCase();
  return v === "1" || v === "true" || v === "yes" || v === "on";
}

// Resolve tracing config from an environment-like record (pure; testable).
//
// Enablement precedence: LITELLM_TRACE takes precedence over the TRACE alias.
// If LITELLM_TRACE is SET (to any non-empty value) it is authoritative — an
// explicit LITELLM_TRACE=0 disables tracing even if TRACE=1. Only when
// LITELLM_TRACE is unset do we fall back to the TRACE alias.
//
// Content capture (LITELLM_TRACE_CONTENT) is only meaningful when tracing is
// enabled; it is OFF by default and independent of the enable alias.
export function resolveTraceConfig(
  env: Record<string, string | undefined>
): TraceConfig {
  const primary = env["LITELLM_TRACE"];
  const enabled =
    primary !== undefined && primary.trim() !== ""
      ? isFlagOn(primary)
      : isFlagOn(env["TRACE"]);
  const captureContent = enabled && isFlagOn(env["LITELLM_TRACE_CONTENT"]);
  return { enabled, captureContent };
}

// --- JSONL serialization (pure) ---------------------------------------------

// Serialize one event to a single JSONL line (no embedded newlines). JSON.stringify
// escapes any newline inside string values, so the result is always exactly one
// line; we append the trailing "\n" ourselves. Returns "" for an unserializable
// event rather than throwing (defensive; events we build are always plain data).
export function serializeEvent(event: TraceEvent): string {
  try {
    return JSON.stringify(event) + "\n";
  } catch {
    return "";
  }
}

// --- Sinks ------------------------------------------------------------------

// A sink receives fully-built, already-redacted events. Implementations must
// swallow their own errors: a sink failure must never propagate to the app.
export interface TraceSink {
  write(event: TraceEvent): void | Promise<void>;
}

// In-memory sink for tests and programmatic inspection. Records events; never
// touches the filesystem or network.
export class MemorySink implements TraceSink {
  readonly events: TraceEvent[] = [];
  write(event: TraceEvent): void {
    this.events.push(event);
  }
}

// Console sink: writes one JSONL line per event to stderr (so it never mixes
// with normal stdout program output). Swallows serialization/write errors.
export class ConsoleSink implements TraceSink {
  write(event: TraceEvent): void {
    try {
      const line = serializeEvent(event);
      if (line) process.stderr.write(line);
    } catch {
      // Tracing must never break a normal run.
    }
  }
}

// Default repo-local logs directory (gitignored). Resolved relative to this
// module so it is independent of the process working directory.
export function defaultLogDir(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  // src/core/ -> repo root is two levels up.
  return path.resolve(here, "..", "..", "logs");
}

// File sink: appends newline-delimited JSON to a local file. Creates the target
// directory on first write if missing. ALL failures (mkdir, append, serialize)
// are caught and reported to stderr WITHOUT throwing, so a broken log path can
// never crash the application.
export class FileSink implements TraceSink {
  private readonly dir: string;
  private readonly file: string;
  // Serialize writes so concurrent events cannot interleave partial lines.
  private queue: Promise<void> = Promise.resolve();
  private dirEnsured = false;

  constructor(options: { dir?: string; fileName?: string } = {}) {
    this.dir = options.dir ?? defaultLogDir();
    this.file = path.join(this.dir, options.fileName ?? "trace.jsonl");
  }

  write(event: TraceEvent): Promise<void> {
    const line = serializeEvent(event);
    if (!line) return Promise.resolve();
    // Chain onto the queue so appends are ordered and non-overlapping.
    this.queue = this.queue
      .then(async () => {
        if (!this.dirEnsured) {
          await mkdir(this.dir, { recursive: true });
          this.dirEnsured = true;
        }
        await appendFile(this.file, line, "utf8");
      })
      .catch((err: unknown) => {
        // Never propagate — tracing is best-effort. Report redacted reason.
        process.stderr.write(
          `[tracing] sink write failed (swallowed): ${redactError(err)}\n`
        );
      });
    return this.queue;
  }
}

// Select the default sink for a live run. The file sink is the product default;
// LITELLM_TRACE_SINK=console selects the console sink instead.
export function selectSink(
  env: Record<string, string | undefined>
): TraceSink {
  const sink = env["LITELLM_TRACE_SINK"]?.trim().toLowerCase();
  if (sink === "console") return new ConsoleSink();
  return new FileSink();
}

// --- Event construction (pure) ----------------------------------------------

export interface BuildEventInput {
  timestamp: string;
  model?: string;
  operation: TraceOperation;
  latencyMs: number;
  streaming: boolean;
  usage?: TraceUsage;
  // Raw error (only for error outcomes); redacted here.
  error?: unknown;
  // Raw content (only when capture is on); redacted here.
  rawContent?: { prompt?: string; response?: string };
}

// Build a fully-formed, already-redacted TraceEvent from synthetic or live
// inputs. Pure: no I/O, no clock read (timestamp is passed in). This is the
// single place event fields are assembled, so tests can assert the schema.
export function buildEvent(input: BuildEventInput): TraceEvent {
  const outcome: TraceOutcome =
    input.error !== undefined ? "error" : "success";

  const event: TraceEvent = {
    timestamp: input.timestamp,
    operation: input.operation,
    latencyMs: input.latencyMs,
    outcome,
    streaming: input.streaming,
  };
  if (input.model !== undefined) event.model = input.model;
  if (input.usage) event.usage = input.usage;
  if (outcome === "error") {
    // Trace errorReason uses redactError's DEFAULT_MAX_LENGTH (500) cap by
    // design; this intentionally differs from the probe's tighter 200-char
    // evidence cap (traces retain more diagnostic context than dated evidence).
    event.errorReason = redactError(input.error);
  }
  // Content is ONLY attached when explicitly provided (capture on). Always
  // redacted, even though it means metadata-only mode can never leak content.
  if (input.rawContent) {
    const content: TraceContent = {};
    if (input.rawContent.prompt !== undefined) {
      content.prompt = redactText(input.rawContent.prompt);
    }
    if (input.rawContent.response !== undefined) {
      content.response = redactText(input.rawContent.response);
    }
    if (content.prompt !== undefined || content.response !== undefined) {
      event.content = content;
    }
  }
  return event;
}

// Extract token usage from an LLMResult, tolerating provider variation. Looks at
// the AIMessage.usage_metadata on the first generation (LangChain's normalized
// shape) and falls back to llmOutput.tokenUsage (OpenAI-style). Returns
// undefined when no usage is present.
export function extractUsage(result: LLMResult): TraceUsage | undefined {
  const gen = result.generations?.[0]?.[0] as
    | { message?: { usage_metadata?: unknown } }
    | undefined;
  const meta = gen?.message?.usage_metadata as
    | { input_tokens?: number; output_tokens?: number; total_tokens?: number }
    | undefined;
  if (meta && typeof meta === "object") {
    const usage: TraceUsage = {};
    if (typeof meta.input_tokens === "number") usage.inputTokens = meta.input_tokens;
    if (typeof meta.output_tokens === "number") usage.outputTokens = meta.output_tokens;
    if (typeof meta.total_tokens === "number") usage.totalTokens = meta.total_tokens;
    if (Object.keys(usage).length > 0) return usage;
  }
  const tokenUsage = result.llmOutput?.["tokenUsage"] as
    | { promptTokens?: number; completionTokens?: number; totalTokens?: number }
    | undefined;
  if (tokenUsage && typeof tokenUsage === "object") {
    const usage: TraceUsage = {};
    if (typeof tokenUsage.promptTokens === "number") usage.inputTokens = tokenUsage.promptTokens;
    if (typeof tokenUsage.completionTokens === "number") usage.outputTokens = tokenUsage.completionTokens;
    if (typeof tokenUsage.totalTokens === "number") usage.totalTokens = tokenUsage.totalTokens;
    if (Object.keys(usage).length > 0) return usage;
  }
  return undefined;
}

// Extract response text from an LLMResult for opt-in content capture. Returns
// undefined when no text is available. The caller redacts.
function extractResponseText(result: LLMResult): string | undefined {
  const gen = result.generations?.[0]?.[0];
  if (!gen) return undefined;
  if (typeof gen.text === "string" && gen.text.length > 0) return gen.text;
  return undefined;
}

// --- Callback handler (impure edge; delegates to pure builders) -------------

// Per-run bookkeeping so end/error events can compute latency and know whether
// the run was streaming and what the prompt was (for opt-in capture).
interface RunState {
  start: number;
  streaming: boolean;
  prompt?: string;
}

// LangChain callback handler that emits a trace event per LLM call (invoke and
// stream) via the injected sink. Attached to the ChatOpenAI instance only when
// tracing is enabled. Constructs NO provider client, so the 4b guard holds.
export class TracingCallbackHandler extends BaseCallbackHandler {
  name = "wedding_planner_tracing";

  private readonly sink: TraceSink;
  private readonly captureContent: boolean;
  private readonly model?: string;
  private readonly streaming: boolean;
  private readonly runs = new Map<string, RunState>();

  constructor(options: {
    sink: TraceSink;
    captureContent: boolean;
    model?: string;
    streaming: boolean;
  }) {
    super();
    this.sink = options.sink;
    this.captureContent = options.captureContent;
    if (options.model !== undefined) this.model = options.model;
    this.streaming = options.streaming;
  }

  // Determine the operation type. We treat a run as "stream" when the model was
  // constructed in streaming mode; otherwise "invoke".
  private operation(): TraceOperation {
    return this.streaming ? "stream" : "invoke";
  }

  override handleLLMStart(
    _llm: Serialized,
    prompts: string[],
    runId: string
  ): void {
    const state: RunState = { start: Date.now(), streaming: this.streaming };
    if (this.captureContent && prompts.length > 0) {
      state.prompt = prompts.join("\n---\n");
    }
    this.runs.set(runId, state);
  }

  // Chat models call handleChatModelStart instead of handleLLMStart. Reduce the
  // structured messages to a flat prompt string for optional capture and reuse
  // the same start bookkeeping.
  override handleChatModelStart(
    _llm: Serialized,
    messages: unknown[][],
    runId: string
  ): void {
    const state: RunState = { start: Date.now(), streaming: this.streaming };
    if (this.captureContent) {
      const flat = messages
        .flat()
        .map((m) => {
          const content = (m as { content?: unknown }).content;
          return typeof content === "string" ? content : JSON.stringify(content);
        })
        .join("\n---\n");
      if (flat.length > 0) state.prompt = flat;
    }
    this.runs.set(runId, state);
  }

  override handleLLMEnd(output: LLMResult, runId: string): void {
    const state = this.runs.get(runId);
    this.runs.delete(runId);
    const start = state?.start ?? Date.now();
    const usage = extractUsage(output);
    const input: BuildEventInput = {
      timestamp: new Date(start).toISOString(),
      operation: this.operation(),
      latencyMs: Date.now() - start,
      streaming: state?.streaming ?? this.streaming,
    };
    if (this.model !== undefined) input.model = this.model;
    if (usage) input.usage = usage;
    if (this.captureContent) {
      const response = extractResponseText(output);
      const rawContent: { prompt?: string; response?: string } = {};
      if (state?.prompt !== undefined) rawContent.prompt = state.prompt;
      if (response !== undefined) rawContent.response = response;
      if (rawContent.prompt !== undefined || rawContent.response !== undefined) {
        input.rawContent = rawContent;
      }
    }
    this.emit(buildEvent(input));
  }

  override handleLLMError(err: Error, runId: string): void {
    const state = this.runs.get(runId);
    this.runs.delete(runId);
    const start = state?.start ?? Date.now();
    const input: BuildEventInput = {
      timestamp: new Date(start).toISOString(),
      operation: this.operation(),
      latencyMs: Date.now() - start,
      streaming: state?.streaming ?? this.streaming,
      error: err,
    };
    if (this.model !== undefined) input.model = this.model;
    if (this.captureContent && state?.prompt !== undefined) {
      input.rawContent = { prompt: state.prompt };
    }
    this.emit(buildEvent(input));
  }

  // Push an event to the sink, swallowing any synchronous or asynchronous sink
  // error so tracing can never break a normal run.
  private emit(event: TraceEvent): void {
    try {
      const maybe = this.sink.write(event);
      if (maybe instanceof Promise) {
        maybe.catch((err: unknown) => {
          process.stderr.write(
            `[tracing] sink error (swallowed): ${redactError(err)}\n`
          );
        });
      }
    } catch (err) {
      process.stderr.write(
        `[tracing] sink error (swallowed): ${redactError(err)}\n`
      );
    }
  }
}

// --- Factory used at the createChatModel() boundary -------------------------

export interface TracingHandlerOptions {
  model?: string;
  streaming: boolean;
  // Overrides for tests; production reads process.env / selectSink.
  env?: Record<string, string | undefined>;
  sink?: TraceSink;
}

// Build a tracing callback handler IF tracing is enabled, else return
// undefined. This is the single decision point model.ts calls: when it returns
// undefined, no handler is attached and behavior is unchanged (true no-op — no
// events, no files). When enabled, the handler is wired with the resolved sink
// and content-capture setting.
export function createTracingHandler(
  options: TracingHandlerOptions
): TracingCallbackHandler | undefined {
  const env = options.env ?? process.env;
  const cfg = resolveTraceConfig(env);
  if (!cfg.enabled) return undefined;
  const sink = options.sink ?? selectSink(env);
  const handlerOptions: {
    sink: TraceSink;
    captureContent: boolean;
    model?: string;
    streaming: boolean;
  } = {
    sink,
    captureContent: cfg.captureContent,
    streaming: options.streaming,
  };
  if (options.model !== undefined) handlerOptions.model = options.model;
  return new TracingCallbackHandler(handlerOptions);
}
