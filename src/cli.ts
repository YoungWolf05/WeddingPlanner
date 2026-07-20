import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { HumanMessage, type BaseMessage } from "@langchain/core/messages";
import {
  createConversationalChain,
  type ConversationalChain,
} from "./core/chain.js";
import { sessionConfig } from "./core/memory.js";
import { config } from "./config.js";

// Phase 3: interactive terminal REPL with live token streaming. Reuses the
// Phase 2 conversational graph + checkpointer for multi-turn history (keyed by
// thread_id) — no hand-rolled history array. Run: npm run chat

// Models the LiteLLM proxy exposes (see AGENTS.md).
const ALLOWED_MODELS = [
  "claude-opus-4-8",
  "claude-sonnet-4-6",
  "gpt-5.1-chat",
] as const;

type AllowedModel = (typeof ALLOWED_MODELS)[number];

// Project's documented default (AGENTS.md); used when the configured model is
// missing or not in the allow-list.
const DEFAULT_MODEL: AllowedModel = "claude-sonnet-4-6";

const USER_PROMPT = "You> ";
const BOT_LABEL = "Aria> ";

function isAllowedModel(name: string): name is AllowedModel {
  return (ALLOWED_MODELS as readonly string[]).includes(name);
}

// Single place that constructs the conversational graph, so model init options
// (notably streaming: true) stay consistent across startup and /model switches.
function buildGraph(model: string): ConversationalChain {
  return createConversationalChain({ model, streaming: true });
}

// True for an intentional user abort (Ctrl-C) rather than a real failure.
function isAbortError(err: unknown): boolean {
  return (
    err instanceof Error &&
    (err.name === "AbortError" || err.message.toLowerCase().includes("abort"))
  );
}

function printBanner(model: string): void {
  stdout.write(
    [
      "",
      "Aria — your wedding planning assistant.",
      "Type a message and press Enter. Slash commands:",
      "  /new           start a fresh conversation (clears history)",
      "  /model <name>  switch model; valid: " + ALLOWED_MODELS.join(", "),
      "  /exit          quit",
      `Active model: ${model}`,
      "",
    ].join("\n") + "\n"
  );
}

// Streams one turn to stdout token-by-token. Uses LangGraph's streaming API
// with streamMode "messages", which yields [messageChunk, metadata] tuples;
// we print the incremental string content of each chunk as it arrives.
async function streamTurn(
  graph: ConversationalChain,
  text: string,
  runConfig: ReturnType<typeof sessionConfig>,
  signal: AbortSignal
): Promise<void> {
  // `signal` is inherited from RunnableConfig (PregelOptions extends
  // RunnableConfig in @langchain/langgraph); aborting it cancels the in-flight
  // run and rejects the stream iteration.
  const stream = await graph.stream(
    { messages: [new HumanMessage(text)] },
    { ...runConfig, streamMode: "messages", signal }
  );

  stdout.write(BOT_LABEL);
  for await (const [chunk] of stream as AsyncIterable<[BaseMessage, unknown]>) {
    const content = chunk.content;
    // content can be a string or an array of blocks — only stream string parts.
    if (typeof content === "string") {
      if (content.length > 0) stdout.write(content);
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (
          typeof block === "object" &&
          block !== null &&
          "text" in block &&
          typeof (block as { text: unknown }).text === "string"
        ) {
          stdout.write((block as { text: string }).text);
        }
      }
    }
  }
  stdout.write("\n");
}

async function main(): Promise<void> {
  // Validate the configured model against the allow-list; fall back to the
  // documented default if it's unsupported, so the banner never advertises an
  // invalid model.
  let currentModel: AllowedModel;
  if (isAllowedModel(config.model)) {
    currentModel = config.model;
  } else {
    console.error(
      `[warn] Configured model "${config.model}" is not supported; ` +
        `falling back to ${DEFAULT_MODEL}.`
    );
    currentModel = DEFAULT_MODEL;
  }

  let graph = buildGraph(currentModel);
  let threadId = `chat-${Date.now()}`;

  const rl = createInterface({ input: stdin, output: stdout });

  let closing = false;
  const goodbye = (): void => {
    if (closing) return;
    closing = true;
    stdout.write("\nUntil next time — happy planning!\n");
    rl.close();
    process.exit(0);
  };

  // Non-null while a turn is streaming; lets SIGINT abort just that turn.
  let activeTurn: AbortController | null = null;

  // Ctrl-C: if a turn is in flight, abort it and return to the prompt;
  // otherwise (idle at the prompt) exit gracefully.
  rl.on("SIGINT", () => {
    if (activeTurn) {
      activeTurn.abort();
    } else {
      goodbye();
    }
  });
  rl.on("close", () => {
    if (!closing) goodbye();
  });

  printBanner(currentModel);

  // Manual loop (instead of `for await` on the interface) so streaming output
  // and prompts interleave cleanly without racing readline's line events.
  for (;;) {
    let line: string;
    try {
      line = await rl.question(USER_PROMPT);
    } catch {
      // question() rejects when the interface is closed (e.g. Ctrl-C).
      break;
    }

    const trimmed = line.trim();
    if (trimmed === "") continue; // ignore empty input, re-prompt

    if (trimmed.startsWith("/")) {
      const [command, ...rest] = trimmed.slice(1).split(/\s+/);
      const arg = rest.join(" ").trim();

      if (command === "exit") {
        goodbye();
        return;
      }

      if (command === "new") {
        // Fresh thread_id => the checkpointer has no history for it. Same model.
        threadId = `chat-${Date.now()}`;
        stdout.write("Started a new conversation. Previous history cleared.\n");
        continue;
      }

      if (command === "model") {
        if (!arg) {
          stdout.write(
            `Usage: /model <name>. Valid models: ${ALLOWED_MODELS.join(", ")}\n`
          );
          continue;
        }
        if (!isAllowedModel(arg)) {
          stdout.write(
            `Unknown model "${arg}". Valid models: ${ALLOWED_MODELS.join(", ")}\n`
          );
          continue;
        }
        currentModel = arg;
        // Rebuild the graph with the new model. Keep the SAME thread_id so the
        // conversation (and its history in the checkpointer) continues
        // seamlessly across the switch.
        graph = buildGraph(currentModel);
        stdout.write(`Switched model to ${currentModel}.\n`);
        continue;
      }

      stdout.write(
        `Unknown command "/${command}". Available: /new, /model <name>, /exit\n`
      );
      continue;
    }

    // Normal chat turn — stream the reply, but never let one failed turn crash
    // the REPL. A per-turn AbortController lets Ctrl-C interrupt mid-stream.
    const controller = new AbortController();
    activeTurn = controller;
    try {
      await streamTurn(graph, trimmed, sessionConfig(threadId), controller.signal);
    } catch (err) {
      if (controller.signal.aborted || isAbortError(err)) {
        // Intentional user interrupt — not a failure.
        stdout.write("\n[interrupted]\n");
      } else {
        const message = err instanceof Error ? err.message : String(err);
        stdout.write(`\n[Error] Sorry, that turn failed: ${message}\n`);
      }
    } finally {
      activeTurn = null;
    }
  }
}

main().catch((err) => {
  console.error("\nChat session failed to start.");
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
