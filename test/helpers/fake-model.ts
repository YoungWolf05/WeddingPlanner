import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import {
  FakeListChatModel,
  type FakeChatInput,
} from "@langchain/core/utils/testing";
import {
  BaseChatModel,
  type BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { ChatResult } from "@langchain/core/outputs";

// Shared, deterministic, OFFLINE fake for the model boundary.
//
// The production code always constructs its LLM via createChatModel() in
// src/core/model.ts. Tests replace that factory (via vi.mock in each test file)
// with a fake that:
//   - returns canned responses (no network, no credentials), and
//   - records the exact messages it is asked to generate, so tests can assert
//     on the real constructed prompt (persona, history, human input) and on
//     multi-turn history accumulation — not just on canned output.
//
// FakeListChatModel from @langchain/core supports both invoke() (Phase 1/2) and
// streaming (Phase 3), so the same fake exercises every phase.

// One entry per model call; each entry is the array of messages passed to the
// LLM for that call. Shared module-level state, reset via resetRecordedCalls().
export const recordedCalls: BaseMessage[][] = [];

export function resetRecordedCalls(): void {
  recordedCalls.length = 0;
}

// Records every generate/stream call's input messages, then delegates to the
// real FakeListChatModel behavior (canned responses + streaming char chunks).
class RecordingFakeChatModel extends FakeListChatModel {
  override async _generate(
    messages: BaseMessage[],
    options?: Parameters<FakeListChatModel["_generate"]>[1],
    runManager?: Parameters<FakeListChatModel["_generate"]>[2]
  ): ReturnType<FakeListChatModel["_generate"]> {
    recordedCalls.push(messages);
    return super._generate(messages, options, runManager);
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: Parameters<FakeListChatModel["_streamResponseChunks"]>[1],
    runManager?: Parameters<FakeListChatModel["_streamResponseChunks"]>[2]
  ): ReturnType<FakeListChatModel["_streamResponseChunks"]> {
    recordedCalls.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

// Factory used by vi.mock factories to build a createChatModel replacement that
// returns a fresh recording fake. Accepts the FakeListChatModel options so each
// test file can choose its canned responses.
export function makeFakeChatModel(
  params: FakeChatInput
): BaseChatModel {
  return new RecordingFakeChatModel(params);
}

// ---------------------------------------------------------------------------
// Phase 6 (6c): scripted TOOL-CALLING fake chat model.
//
// FakeListChatModel returns only string responses and cannot emit tool_calls,
// so it cannot drive a real react (tool-loop) agent. This fake DOES: it plays a
// fixed SCRIPT of AIMessages across successive model calls, so the REAL
// createReactAgent loop + REAL ToolNode + REAL 6b tools run fully offline and
// deterministically.
//
// How it integrates with createReactAgent:
//   - createReactAgent detects a chat model with a bindTools method and calls
//     `llm.bindTools(tools)`. This fake's bindTools returns `this` (the script
//     is keyed by CALL ORDER, not by which tools are bound), so the compiled
//     graph invokes this same model each agent step.
//   - Each agent step calls `_generate`, which returns the next scripted
//     AIMessage. A message with `tool_calls` makes the graph route to the tool
//     node (which executes the real tools and appends ToolMessages); a message
//     with no tool_calls ends the loop as the final answer.
//   - No network, no credentials, no timers — the script is a fixed array.
// ---------------------------------------------------------------------------

// One scripted turn = one AIMessage the fake returns on its next _generate call.
export interface ScriptedTurn {
  // Free text for the message content (the final-answer text, or "" for a
  // tool-call-only step).
  content?: string;
  // Tool calls to emit. Each drives the ToolNode to run the named real tool
  // with the given args. Omit/empty => a terminal (final-answer) message.
  toolCalls?: Array<{ name: string; args: Record<string, unknown> }>;
}

// Records, for a scripted-tool-calling fake, the exact message arrays it was
// asked to generate (so tests can assert the loop actually appended
// ToolMessages before the final step) plus the number of times _generate ran.
export interface ScriptedRecorder {
  calls: BaseMessage[][];
}

// Deterministic, offline, scripted tool-calling chat model. Not exported as a
// class; use makeScriptedToolCallingModel() so tests get a fresh instance with
// its own script + recorder.
class ScriptedToolCallingChatModel extends BaseChatModel {
  private readonly script: ScriptedTurn[];
  private readonly recorder: ScriptedRecorder;
  private step = 0;

  constructor(
    script: ScriptedTurn[],
    recorder: ScriptedRecorder,
    params: BaseChatModelParams = {}
  ) {
    super(params);
    this.script = script;
    this.recorder = recorder;
  }

  _llmType(): string {
    return "scripted-tool-calling-fake";
  }

  // Return `this` so the react loop keeps invoking this same scripted model.
  // The script is keyed by call order, not by the bound tool set, so we can
  // ignore the tools here and still exercise the REAL ToolNode + real tools.
  override bindTools(): this {
    return this;
  }

  override async _generate(
    messages: BaseMessage[],
    _options?: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    this.recorder.calls.push(messages);
    const turn = this.script[this.step];
    if (turn === undefined) {
      throw new Error(
        `ScriptedToolCallingChatModel: no scripted turn for call #${this.step + 1} ` +
          `(script has ${this.script.length} turn(s)). The agent loop ran longer ` +
          `than the script anticipated.`
      );
    }
    this.step += 1;

    const message = new AIMessage({
      content: turn.content ?? "",
      tool_calls: (turn.toolCalls ?? []).map((tc, i) => ({
        // Stable, deterministic ids so nothing depends on randomness.
        id: `call_${this.step}_${i}`,
        name: tc.name,
        args: tc.args,
        type: "tool_call" as const,
      })),
    });

    return {
      generations: [
        {
          message,
          text: typeof message.content === "string" ? message.content : "",
        },
      ],
      llmOutput: {},
    };
  }
}

// Build a fresh scripted tool-calling model + its recorder. Each call to
// _generate consumes the next scripted turn (in order); running past the end of
// the script throws a clear error (a guard that the loop terminated as expected).
export function makeScriptedToolCallingModel(script: ScriptedTurn[]): {
  model: BaseChatModel;
  recorder: ScriptedRecorder;
} {
  const recorder: ScriptedRecorder = { calls: [] };
  const model = new ScriptedToolCallingChatModel(script, recorder);
  return { model, recorder };
}
