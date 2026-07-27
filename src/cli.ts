import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import {
  createConversationalChain,
  type ConversationalChain,
} from "./core/chain.js";
import { sessionConfig } from "./core/memory.js";
import { config } from "./config.js";
import {
  ALLOWED_MODELS,
  DEFAULT_MODEL,
  USER_PROMPT,
  type AllowedModel,
  isAllowedModel,
  selectInitialModel,
  parseLine,
  classifyTurnError,
  streamTurn,
} from "./core/repl.js";

// Phase 3: interactive terminal REPL with live token streaming. Reuses the
// Phase 2 conversational graph + checkpointer for multi-turn history (keyed by
// thread_id) — no hand-rolled history array. Run: npm run chat
//
// The pure control-flow (model allow-list, command parsing, streaming, abort
// classification) lives in ./core/repl.js so it can be unit-tested without a
// TTY; this file keeps the interactive shell and top-level main() side effects.

// Single place that constructs the conversational graph, so model init options
// (notably streaming: true) stay consistent across startup and /model switches.
function buildGraph(model: string): ConversationalChain {
  return createConversationalChain({ model, streaming: true });
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

async function main(): Promise<void> {
  // Validate the configured model against the allow-list; fall back to the
  // documented default if it's unsupported, so the banner never advertises an
  // invalid model.
  const selection = selectInitialModel(config.model);
  let currentModel: AllowedModel = selection.model;
  if (selection.fellBack) {
    console.error(
      `[warn] Configured model "${config.model}" is not supported; ` +
        `falling back to ${DEFAULT_MODEL}.`
    );
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

    const parsed = parseLine(line);
    if (parsed.kind === "empty") continue; // ignore empty input, re-prompt

    if (parsed.kind === "command") {
      const { command, arg } = parsed;

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
      await streamTurn(
        graph,
        parsed.text,
        sessionConfig(threadId),
        controller.signal,
        stdout
      );
    } catch (err) {
      if (
        classifyTurnError(err, controller.signal.aborted) === "interrupted"
      ) {
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
