import { redactText } from "./redaction.js";

// Phase 6 (increment 6d): a small, reusable BOUNDED tool-execution timeout.
//
// WHY THIS EXISTS (and why it is deliberately a defensive bound):
//   The 6b tools (`days_until`, `split_budget`) are PURE, SYNCHRONOUS, and
//   effectively instantaneous — there is no I/O, no unbounded loop, and no
//   blocking call, so in practice they can never actually time out. This wrapper
//   is therefore a DEFENSIVE, forward-looking bound, and it is the concrete
//   mechanism that satisfies the Phase 6 exit-criterion requirement that the
//   budget/date tools have a TIMEOUT (alongside their boundary + failure tests
//   already in `test/phase6-tools.test.ts`). If a future tool ever became async
//   or slow, invoking it through `invokeToolWithTimeout` gives it a hard, typed,
//   redacted upper bound with zero changes at the call site.
//
// DESIGN:
//   - `timeoutMs` is INJECTABLE so tests can drive the timeout path with tiny
//     values (or fake timers) — no real multi-second waits, no real-clock
//     flakiness.
//   - On timeout the wrapper rejects with a typed `ToolTimeoutError` whose
//     message is passed through the shared redaction layer, so nothing that
//     could carry a secret/PII ever reaches a log or client from this path.
//   - A SYNC value or a SYNC throw from the wrapped function is normalized to a
//     resolved / rejected promise, so pure sync tools race correctly and always
//     win against the timer (their result settles on a microtask; the timer
//     fires on a later macrotask).

// Default upper bound for a single tool invocation. Generous because the current
// tools are instantaneous; this is the defensive ceiling, not a tight SLA.
export const DEFAULT_TOOL_TIMEOUT_MS = 5_000;

/**
 * Typed error rejected by {@link withToolTimeout} / {@link invokeToolWithTimeout}
 * when a tool invocation exceeds its bound. The message is redaction-safe (a
 * fixed template over the tool name + bound, run through `redactText`), and the
 * structured `toolName` / `timeoutMs` fields let callers branch without string
 * parsing.
 */
export class ToolTimeoutError extends Error {
  readonly toolName: string;
  readonly timeoutMs: number;

  constructor(toolName: string, timeoutMs: number) {
    super(redactText(`Tool "${toolName}" timed out after ${timeoutMs}ms`));
    this.name = "ToolTimeoutError";
    this.toolName = toolName;
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Race an arbitrary (sync or async) function against a timeout.
 *
 *   - Resolves with `fn()`'s value if it settles before `timeoutMs`.
 *   - Rejects with a {@link ToolTimeoutError} if the bound elapses first.
 *   - Propagates a rejection/throw from `fn` unchanged (a genuine tool failure
 *     is NOT masked as a timeout).
 *
 * The timer is always cleared in `finally`, so a fast success never leaves a
 * dangling handle.
 *
 * @param fn        the work to bound; may be sync or async.
 * @param timeoutMs positive, finite millisecond bound. Injectable for tests.
 * @param toolName  label used only in the timeout error message.
 */
export async function withToolTimeout<T>(
  fn: () => T | Promise<T>,
  timeoutMs: number,
  toolName = "tool"
): Promise<T> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error(
      "withToolTimeout: timeoutMs must be a positive, finite number"
    );
  }

  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new ToolTimeoutError(toolName, timeoutMs)),
      timeoutMs
    );
  });

  try {
    // `Promise.resolve().then(fn)` normalizes a sync value to a resolved promise
    // and a sync throw to a rejected one, so both sync and async tools race
    // correctly against the timer.
    return await Promise.race([Promise.resolve().then(fn), timeout]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/**
 * Minimal structural shape of an invokable LangChain tool: a `name` plus an
 * `invoke(input)` returning a promise. The concrete `tool()` objects in
 * `src/core/tools.ts` satisfy this (their `invoke` also accepts an optional
 * config arg, which is assignment-compatible with this narrower signature).
 */
export interface TimeoutInvokableTool<TInput = unknown, TOutput = unknown> {
  readonly name: string;
  invoke(input: TInput): Promise<TOutput>;
}

/**
 * Invoke a LangChain tool under a hard timeout. Thin convenience over
 * {@link withToolTimeout} that labels the {@link ToolTimeoutError} with the
 * tool's own name.
 *
 * @param tool      any {@link TimeoutInvokableTool} (e.g. `daysUntilTool`).
 * @param input     the tool's validated input object.
 * @param timeoutMs bound in ms; defaults to {@link DEFAULT_TOOL_TIMEOUT_MS}.
 */
export function invokeToolWithTimeout<TInput, TOutput>(
  tool: TimeoutInvokableTool<TInput, TOutput>,
  input: TInput,
  timeoutMs: number = DEFAULT_TOOL_TIMEOUT_MS
): Promise<TOutput> {
  return withToolTimeout(() => tool.invoke(input), timeoutMs, tool.name);
}
