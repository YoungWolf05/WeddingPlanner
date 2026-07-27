// Deterministic, offline test environment bootstrap.
//
// `src/config.ts` imports "dotenv/config" and hard-fails at import time if
// LITELLM_BASE_URL or LITELLM_API_KEY is missing. Nearly every module
// transitively imports config.ts, so we must satisfy that contract BEFORE any
// source module loads — hence this file is registered as a Vitest `setupFile`,
// which runs before the test modules are imported.
//
// We set dummy, obviously-fake values via UNCONDITIONAL assignment. This
// overwrites anything already in the environment, so tests can NEVER inherit
// real ambient credentials from the shell/CI — the run stays offline and
// deterministic regardless of the surrounding environment. These are NOT real
// secrets and never leave the test process. Because this setup file runs before
// any source module loads (and thus before config.ts's `import "dotenv/config"`,
// which does not override already-set process.env values), these dummies are
// what config.ts sees.
//
// The whole point of increment 4a is that tests are fully offline: the LLM
// boundary (createChatModel) is mocked per-test, so these values are only used
// to let config.ts load — no network call is ever made with them.
process.env.LITELLM_BASE_URL = "http://localhost:0/test-fake-litellm";
process.env.LITELLM_API_KEY = "sk-test-dummy-not-a-real-key";
// Leave LITELLM_MODEL unset so config.ts exercises its documented default
// (claude-sonnet-4-6). Individual tests override the model where relevant.
delete process.env.LITELLM_MODEL;
