import { HumanMessage } from "@langchain/core/messages";
import { createConversationalChain } from "./core/chain.js";
import { sessionConfig } from "./core/memory.js";

/**
 * Phase 2 demo: proves the graph remembers earlier turns within a session.
 *
 * Sends three messages on the SAME session (thread_id). The facts (budget,
 * date, guest count) are stated up front and then referenced later WITHOUT
 * repeating them, so a correct answer demonstrates working memory.
 *
 * Run with: npm run memory
 */
async function main() {
  const graph = createConversationalChain();
  const config = sessionConfig(`demo-${Date.now()}`);

  const turns = [
    "Hi! We're planning a wedding for 120 guests on 2026-12-12, budget is $40,000.",
    "What's a good rough budget breakdown for us?",
    "Remind me: how many guests are coming and what's our date and total budget?",
  ];

  for (const input of turns) {
    console.log(`\nYou: ${input}`);
    const result = await graph.invoke(
      { messages: [new HumanMessage(input)] },
      config
    );
    const content = result.messages.at(-1)?.content;
    const reply =
      typeof content === "string"
        ? content
        : content
          ? JSON.stringify(content)
          : "(no reply)";
    console.log(`\nAria: ${reply}`);
    console.log("\n" + "-".repeat(60));
  }

  console.log(
    "\nIf Aria correctly recalled 120 guests, 2026-12-12, and $40,000 " +
      "in the final turn, memory is working."
  );
}

main().catch((err) => {
  console.error("\nMemory demo FAILED.");
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
