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
};
