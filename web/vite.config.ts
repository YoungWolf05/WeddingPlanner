import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 9 (9c) Vite config for the Wedding Planner browser SPA.
//
// DEV PROXY (same-origin, NO CORS). During `npm run dev` the SPA is served by
// Vite (default http://localhost:5173) but the backend conversation service
// (`npm run serve`) listens on 127.0.0.1:<SERVICE_PORT> (default 3000). To keep
// the browser SAME-ORIGIN with the API — so no CORS handling is needed on the
// backend and the bearer token / SSE stream are sent to the SAME origin the page
// loaded from — Vite forwards the backend's API routes to the service:
//   - /threads   (POST create, GET list; /threads/:id GET/DELETE; /threads/:id/chat POST SSE)
//   - /healthz   (liveness)
// These are the EXACT routes exposed by src/core/server.ts (matchThreadRoute +
// the /healthz handler). The chat SSE endpoint is a POST that streams
// text/event-stream; `ws: false` and default proxy behavior stream the response
// body through unbuffered, which is what the fetch-based SSE client consumes.
//
// The proxy target is overridable via the WEB_PROXY_TARGET env var for a
// non-default SERVICE_PORT. NOTE: this is a DEV-ONLY convenience read from the
// Node/Vite process env at config time — it is NOT bundled and is NOT a provider
// secret (it is only the backend service origin). No LITELLM_*/provider secret
// is ever referenced here or anywhere in the frontend.
const proxyTarget = process.env.WEB_PROXY_TARGET ?? "http://127.0.0.1:3000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/threads": {
        target: proxyTarget,
        changeOrigin: true,
        // SSE must not be buffered; keep the upstream connection streaming.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            // Defensive: ensure no intermediate buffering hint survives.
            delete proxyRes.headers["content-length"];
          });
        },
      },
      "/healthz": {
        target: proxyTarget,
        changeOrigin: true,
      },
    },
  },
});
