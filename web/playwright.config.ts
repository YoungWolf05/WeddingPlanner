import { defineConfig, devices } from "@playwright/test";

// Phase 9 (9d): Playwright BROWSER E2E configuration.
//
// OPT-IN + SEPARATE FROM `npm test`. This config drives Chromium against the
// PRODUCTION SPA bundle talking to a DETERMINISTIC, OFFLINE backend harness. It
// is run ONLY via `npm run e2e` (after `npm run e2e:install`); the web Vitest
// suite (`npm test`) has its OWN config (vitest.config.ts, testDir test/**) and
// never runs anything under e2e/. It is NEVER pulled into the ROOT offline suite.
//
// TWO webServers (started by Playwright, torn down after the run):
//   1. THE DETERMINISTIC BACKEND HARNESS — `npx tsx web/e2e/fixtures/test-server.ts`
//      run from the REPO ROOT (cwd: "..") so it resolves the backend's real
//      node_modules (better-sqlite3, langgraph). It boots the REAL createServer()
//      (real auth/ownership/SSE/redaction) with a FAKE grounded answerTurn seam
//      over a TEMP sqlite db OUTSIDE the repo — NO live model/network. Ready when
//      GET /healthz on E2E_BACKEND_PORT returns 200.
//   2. THE SPA PREVIEW SERVER — `vite build && vite preview` serving web/dist
//      (the PRODUCTION bundle, so the zero-creds bundle is what the browser runs).
//      Its proxy (vite.config.ts `preview.proxy`, target = WEB_PROXY_TARGET) points
//      /threads* and /healthz at the harness backend, keeping the browser
//      same-origin (no CORS). Ready when the preview URL serves the SPA.
//
// The single Chromium project is sufficient for these journeys; firefox/webkit
// would only add browser-download cost without covering a different code path.

// Ports (kept in one place; the harness reads E2E_BACKEND_PORT and the preview
// proxy target is derived from it). Both bind 127.0.0.1 only.
const BACKEND_PORT = 3100;
const PREVIEW_PORT = 4180;

const BASE_URL = `http://127.0.0.1:${String(PREVIEW_PORT)}`;
const BACKEND_URL = `http://127.0.0.1:${String(BACKEND_PORT)}`;

export default defineConfig({
  testDir: "./e2e",
  // Match only *.spec.ts under e2e/ (the harness/fixtures are *.ts, not specs).
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: !!process.env["CI"],
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    // No stored auth state: every spec starts from the token gate.
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      // (1) Deterministic OFFLINE backend harness. Run from the repo ROOT so the
      // backend's own node_modules resolve. NO live model/network.
      command: "npx tsx web/e2e/fixtures/test-server.ts",
      cwd: "..",
      url: `${BACKEND_URL}/healthz`,
      timeout: 60_000,
      reuseExistingServer: !process.env["CI"],
      env: {
        E2E_BACKEND_PORT: String(BACKEND_PORT),
      },
      stdout: "pipe",
      stderr: "pipe",
    },
    {
      // (2) SPA production bundle via vite preview, proxying the API to the
      // harness backend. `vite build` first so web/dist is the current bundle.
      command: `vite build && vite preview --host 127.0.0.1 --port ${String(
        PREVIEW_PORT
      )} --strictPort`,
      url: BASE_URL,
      timeout: 120_000,
      reuseExistingServer: !process.env["CI"],
      env: {
        WEB_PROXY_TARGET: BACKEND_URL,
      },
      stdout: "pipe",
      stderr: "pipe",
    },
  ],
});
