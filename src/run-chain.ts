import { createWeddingPlannerChain } from "./core/chain.js";

/**
 * Sanity-check script for the Phase 1 LCEL chain.
 *
 * Usage:
 *   npm run chain "ideas for a beach wedding"
 *
 * Invokes the wedding planner chain once with the message passed on the
 * command line and prints the model's reply.
 */
async function main() {
  const input = process.argv.slice(2).join(" ").trim();

  if (!input) {
    console.error('Please provide a message. Example:');
    console.error('  npm run chain "ideas for a beach wedding"');
    process.exit(1);
  }

  const chain = createWeddingPlannerChain();

  console.log(`You: ${input}\n`);
  console.log("Aria:");

  const answer = await chain.invoke({ input });

  console.log(answer);
}

main().catch((err) => {
  console.error("\nChain run FAILED.");
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
