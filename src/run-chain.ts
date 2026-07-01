import { createWeddingPlannerChain } from "./core/chain.js";

// Phase 1 sanity check: npm run chain "ideas for a beach wedding"
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
