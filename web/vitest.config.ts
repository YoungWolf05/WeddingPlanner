import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Phase 9 (9c): the WEB project's OWN vitest config, fully isolated from the
// backend's root vitest (root include is ["test/**/*.test.ts"], so nothing under
// web/ is ever run by root `npm test`). This config runs ONLY web/test.
export default defineConfig({
  plugins: [react()],
  test: {
    include: ["test/**/*.test.ts", "test/**/*.test.tsx"],
    environment: "jsdom",
    globals: true,
    setupFiles: ["./test/setup.ts"],
  },
});
