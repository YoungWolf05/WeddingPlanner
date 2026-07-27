import { defineConfig } from "vitest/config";

// Vitest config for deterministic, fully offline tests.
//
// - `environment: "node"` — this is a Node CLI/service project, no DOM needed.
// - `setupFiles` runs BEFORE any test module (and therefore before the
//   transitive import of `src/config.ts`, which hard-fails without the
//   LITELLM_* env vars). The setup file injects dummy, non-secret values so
//   config loads without real credentials. See test/setup/env.ts.
// - Tests live under `test/` (outside `src/`), so the production `tsc` build
//   (`rootDir: src`) never emits them into `dist/`.
//
// Note: Vite/esbuild transpiles the TypeScript sources on the fly and resolves
// the NodeNext-style `.js` import specifiers back to their `.ts` sources, so the
// project's ESM `.js`-extension import convention works unchanged in tests.
export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    setupFiles: ["test/setup/env.ts"],
    // Fail fast if a test accidentally hangs (e.g. an un-driven stream); keeps
    // the suite from silently blocking CI.
    testTimeout: 10_000,
  },
});
