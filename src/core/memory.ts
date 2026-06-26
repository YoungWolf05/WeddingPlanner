import { MemorySaver, type BaseCheckpointSaver } from "@langchain/langgraph";

/**
 * Conversation persistence for the wedding planner.
 *
 * We use LangGraph's checkpointer abstraction (the modern replacement for the
 * deprecated RunnableWithMessageHistory). A checkpointer saves and restores the
 * full graph state — including the message history — keyed by a `thread_id`
 * (our session id).
 *
 * Phase 2: MemorySaver keeps state in-process RAM (resets when the process
 *          exits).
 * Phase 8: swap MemorySaver for a persistent saver (e.g. SqliteSaver or a
 *          Redis/Postgres checkpointer) WITHOUT changing this signature, so the
 *          rest of the app stays untouched.
 */

/** A single shared in-memory checkpointer instance for the app. */
export const checkpointer: BaseCheckpointSaver = new MemorySaver();

/**
 * Builds the LangGraph config that scopes a conversation to one session.
 * Pass the result as the second argument to graph.invoke()/stream().
 */
export function sessionConfig(sessionId: string) {
  return { configurable: { thread_id: sessionId } };
}
