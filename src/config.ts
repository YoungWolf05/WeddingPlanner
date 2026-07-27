import "dotenv/config";

function required(name: string): string {
  const value = process.env[name];
  if (!value || value.trim() === "") {
    throw new Error(
      `Missing required environment variable: ${name}. ` +
        `Copy .env.example to .env and fill it in.`
    );
  }
  return value.trim();
}

export const config = {
  baseURL: required("LITELLM_BASE_URL"),
  apiKey: required("LITELLM_API_KEY"),
  model: process.env.LITELLM_MODEL?.trim() || "claude-sonnet-4-6",
  // Optional embedding alias. Embeddings are a SEPARATE contract from chat
  // aliases (see docs/roadmap.md Phase 4). When unset, the capability probe
  // records embeddings as "N/A — no embedding alias configured" rather than
  // failing; it never assumes a chat alias supports embeddings.
  embedModel: process.env.LITELLM_EMBED_MODEL?.trim() || undefined,
};
