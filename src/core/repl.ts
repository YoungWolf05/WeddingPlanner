import { HumanMessage, type BaseMessage } from "@langchain/core/messages";
import type { ConversationalChain } from "./chain.js";
import type { sessionConfig } from "./memory.js";

// Phase 3 REPL logic, extracted from cli.ts as pure, importable units so the
// control-flow (model allow-list, command parsing, streaming, and abort
// classification) can be tested deterministically without a real TTY, readline,
// or process.exit. cli.ts imports these and keeps its interactive shell + the
// top-level main() side effects. This extraction is behavior-preserving: the
// logic below is byte-for-byte the same as the previous inline implementation.

// Models the LiteLLM proxy exposes (see AGENTS.md).
export const ALLOWED_MODELS = [
  "claude-opus-4-8",
  "claude-sonnet-4-6",
] as const;

export type AllowedModel = (typeof ALLOWED_MODELS)[number];

// Project's documented default (AGENTS.md); used when the configured model is
// missing or not in the allow-list.
export const DEFAULT_MODEL: AllowedModel = "claude-sonnet-4-6";

export const USER_PROMPT = "You> ";
export const BOT_LABEL = "Aria> ";

export function isAllowedModel(name: string): name is AllowedModel {
  return (ALLOWED_MODELS as readonly string[]).includes(name);
}

// Chooses the startup model: the configured one if supported, otherwise the
// documented default. Returns the resolved model plus whether a fallback
// happened, so the caller can decide how to surface the warning.
export function selectInitialModel(configuredModel: string): {
  model: AllowedModel;
  fellBack: boolean;
} {
  if (isAllowedModel(configuredModel)) {
    return { model: configuredModel, fellBack: false };
  }
  return { model: DEFAULT_MODEL, fellBack: true };
}

// True for an intentional user abort (Ctrl-C) rather than a real failure.
export function isAbortError(err: unknown): boolean {
  return (
    err instanceof Error &&
    (err.name === "AbortError" || err.message.toLowerCase().includes("abort"))
  );
}

// Parsed form of a trimmed line the user entered. `kind: "empty"` for
// whitespace-only input (re-prompt), `"chat"` for a normal message, and
// `"command"` for slash commands.
export type ParsedLine =
  | { kind: "empty" }
  | { kind: "chat"; text: string }
  | { kind: "command"; command: string; arg: string };

// Pure parse of one raw input line, mirroring the REPL loop's original logic:
// trim, ignore empty, and split slash commands into command + argument.
export function parseLine(line: string): ParsedLine {
  const trimmed = line.trim();
  if (trimmed === "") return { kind: "empty" };
  if (!trimmed.startsWith("/")) return { kind: "chat", text: trimmed };
  const [command, ...rest] = trimmed.slice(1).split(/\s+/);
  return { kind: "command", command: command ?? "", arg: rest.join(" ").trim() };
}

// Classification of a turn outcome for the REPL's catch block: an intentional
// interrupt (aborted signal or abort-shaped error) vs. a genuine failure.
export type TurnErrorKind = "interrupted" | "failed";

export function classifyTurnError(err: unknown, aborted: boolean): TurnErrorKind {
  return aborted || isAbortError(err) ? "interrupted" : "failed";
}

// Minimal writable sink so streamTurn can be tested against a fake without a
// real stdout / TTY. node's stdout satisfies this shape.
export interface WritableLike {
  write(text: string): unknown;
}

// Extracts the streamable text from a single message chunk's content, which can
// be a plain string or an array of content blocks. Non-text blocks are ignored.
// Returns the concatenated text (may be empty).
export function extractChunkText(content: BaseMessage["content"]): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let out = "";
  for (const block of content) {
    if (
      typeof block === "object" &&
      block !== null &&
      "text" in block &&
      typeof (block as { text: unknown }).text === "string"
    ) {
      out += (block as { text: string }).text;
    }
  }
  return out;
}

// Streams one turn to the provided sink token-by-token. Uses LangGraph's
// streaming API with streamMode "messages", which yields [messageChunk,
// metadata] tuples; we print the incremental string content of each chunk as it
// arrives. Aborting `signal` cancels the in-flight run and rejects iteration.
export async function streamTurn(
  graph: ConversationalChain,
  text: string,
  runConfig: ReturnType<typeof sessionConfig>,
  signal: AbortSignal,
  out: WritableLike
): Promise<void> {
  const stream = await graph.stream(
    { messages: [new HumanMessage(text)] },
    { ...runConfig, streamMode: "messages", signal }
  );

  out.write(BOT_LABEL);
  for await (const [chunk] of stream as AsyncIterable<[BaseMessage, unknown]>) {
    const piece = extractChunkText(chunk.content);
    if (piece.length > 0) out.write(piece);
  }
  out.write("\n");
}
