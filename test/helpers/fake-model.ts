import type { BaseMessage } from "@langchain/core/messages";
import {
  FakeListChatModel,
  type FakeChatInput,
} from "@langchain/core/utils/testing";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

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
