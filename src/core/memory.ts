import { MemorySaver, type BaseCheckpointSaver } from "@langchain/langgraph";

// Phase 2: in-RAM checkpointer (resets on exit). Phase 8: swap for a persistent
// saver (SqliteSaver/Redis) without changing this signature.
export const checkpointer: BaseCheckpointSaver = new MemorySaver();

// Scopes a conversation to one session; pass as the second arg to invoke/stream.
export function sessionConfig(sessionId: string) {
  return { configurable: { thread_id: sessionId } };
}
