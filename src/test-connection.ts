import { createChatModel } from "./core/model.js";
import { config } from "./config.js";

// Connectivity test: verifies LangChain can reach LiteLLM. Run: npm run test:connection
async function main() {
  console.log("Wedding Planner Chatbot - LiteLLM connectivity test");
  console.log("----------------------------------------------------");
  console.log(`Base URL : ${config.baseURL}`);
  console.log(`Model    : ${config.model}`);
  console.log(`API key  : ${config.apiKey.slice(0, 4)}...(hidden)`);
  console.log("");

  // All application model construction goes through the createChatModel()
  // factory (see AGENTS.md). The factory's defaults already match this smoke
  // test's prior client settings: config.model / config.apiKey / config.baseURL,
  // temperature 0.7, non-streaming — so the resulting client is equivalent.
  const llm = createChatModel();

  console.log("Sending a test prompt...\n");

  const response = await llm.invoke([
    {
      role: "system",
      content:
        "You are a friendly wedding planning assistant. Keep answers brief.",
    },
    {
      role: "user",
      content: "Say hello and tell me in one sentence how you can help plan a wedding.",
    },
  ]);

  console.log("Response from model:");
  console.log("--------------------");
  console.log(response.content);
  console.log("");
  console.log("Connection successful. Everything is ready.");
}

main().catch((err) => {
  console.error("\nConnection test FAILED.");
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
