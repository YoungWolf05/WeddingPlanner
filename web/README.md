# Wedding Planner — Web Interface (Phase 9 / 9c)

A minimal, secure React + Vite single-page app (SPA) over the backend's **stable
v2 SSE contract**. This is a **thin client**: it renders streaming answers,
trusted citations, agent tool progress, and structured artifacts from the
backend's *typed* Server-Sent Events, and it holds **no provider credentials**.

> **Isolation.** `web/` is a **separate npm project** with its own
> `package.json`, `tsconfig`s, Vite config, and Vitest. It is deliberately kept
> out of the backend build/test: the root `tsc`/`npm run build` only compile
> `src/**`, and root `npm test` (Vitest) only runs `test/**` — nothing under
> `web/` is compiled or run by the backend tooling. The frontend's deps live in
> `web/node_modules`; run all commands below **from inside `web/`**.

## What this delivers (maps to Phase 9 exit criteria)

- **Exit criterion 1 — no provider creds in the browser + authenticated
  ownership.** The browser only ever talks to the backend service; it holds a
  **bearer token** (the only credential) and derives nothing about identity
  itself. A **tested guardrail** (`npm run test:bundle`) scans the built bundle
  and fails if any provider secret leaks. See *Auth model* and *Zero-credentials
  bundle check* below.
- **Exit criterion 2 — streaming / cancel / retry / reconnect over the versioned
  contract.** The SSE client branches on `init.version` and implements cancel
  (AbortController), retry (re-issue the last message), and reconnect (re-fetch
  thread state / re-issue). See *The v2 SSE contract client*.
- **Exit criterion 3 — citations / tool progress / artifacts from typed trusted
  events.** All of these render **only** from the typed `citation` / `tool` /
  `artifact` events — never fabricated client-side.

Playwright **browser E2E is Phase 9d** (opt-in, separate from `npm test`),
including unauthorized-access and reconnect journeys. See
*E2E (Playwright, opt-in)* below.

## Dev workflow

Two processes: the **backend service** and the **Vite dev server**.

1. **Start the backend conversation service** (from the repo root). It needs
   `AUTH_TOKENS` (it refuses to start with none) and, to exercise the grounded
   RAG path that emits citations/artifacts, a reachable model + embedding + the
   grounded seam enabled. See the root `AGENTS.md` / `.env.example`:

   ```bash
   # repo root
   npm run serve         # binds SERVICE_PORT (default 3000) on 127.0.0.1
   ```

   Configure at least `AUTH_TOKENS` (bearer-token → userId map). The resolved
   userId is the thread **owner**; the browser only sends the bearer token, and
   the backend derives ownership from it.

2. **Start the Vite dev server** (from `web/`):

   ```bash
   # web/
   npm install           # first time only (installs into web/node_modules)
   npm run dev           # serves the SPA (default http://localhost:5173)
   ```

3. Open the SPA, paste one of your `AUTH_TOKENS` bearer tokens, create a
   conversation, and chat.

### Dev proxy (same-origin, no CORS)

`vite.config.ts` proxies the backend API routes so the browser stays
**same-origin** with the service (no CORS handling needed on the backend, and
the bearer token / SSE stream go to the same origin the page loaded from):

- `/threads` and `/threads/:id` and `/threads/:id/chat` → backend
- `/healthz` → backend

The proxy target defaults to `http://127.0.0.1:3000`; override it for a
non-default `SERVICE_PORT` with the **dev-only** `WEB_PROXY_TARGET` env var
(read by Vite at config time — it is the backend *origin*, not a secret, and is
never bundled):

```bash
WEB_PROXY_TARGET=http://127.0.0.1:4000 npm run dev
```

The chat SSE endpoint is a streaming `POST`; the proxy forwards the
`text/event-stream` body unbuffered so tokens arrive incrementally.

## Auth model (bearer → server-derived ownerId; no provider creds)

- The user enters a **bearer token**. It is sent as
  `Authorization: Bearer <token>` on **every** API call (thread create/list/get
  and the chat POST).
- The token is the **only** auth. `ownerId` is derived **server-side** from the
  token (`src/core/auth.ts`); the client **never** sends an owner/user id in any
  body/query/header/path field. `thread_id` is a server-issued conversation key,
  **not** identity or authorization.
- The token is held in memory (React state) and mirrored to **`sessionStorage`**
  so a reload within the tab keeps the session. It is **never baked into the
  bundle**. A `401` from any call signs the user out (the stored token is
  cleared).
- **No provider credentials** (`LITELLM_API_KEY` / `LITELLM_BASE_URL` / any
  `sk-…` key / any `apiKey`/`baseURL` literal) ever appear in the frontend. The
  browser talks only to the backend service.

## The v2 SSE contract client

- `src/lib/sse-contract.ts` — the browser mirror of the wire types in
  **`src/core/sse.ts`** (the canonical source). Because `web/` is a separate
  project, the types are **re-declared** here and **must be kept in lockstep**
  with the backend; 9d E2E is what actually verifies wire compatibility
  end-to-end. `SSE_PROTOCOL_VERSION` is pinned to `2`.
- `src/lib/sseClient.ts` — a **fetch + POST + Bearer** SSE client. The chat
  endpoint is a POST (it carries the message body) and needs an `Authorization`
  header, so `EventSource` (GET-only, no headers) cannot be used. The client
  reads the `text/event-stream` response body via a `ReadableStream` reader and
  parses SSE frames manually (split on blank lines; parse `event:`/`data:`),
  dispatching **typed** events. It **branches on `init.version`**: a mismatch
  surfaces a user-safe error instead of misparsing.
  - **Cancel** — abort the caller's `AbortController`; the fetch/stream stops and
    the turn is marked *canceled* (not an error). The backend also aborts the
    in-flight turn on the client disconnect.
  - **Retry** — re-issue the last message with a fresh `AbortController` (no
    duplicate user bubble).
  - **Reconnect** — on a dropped stream (EOF before `done`, or a transport
    error) the turn is marked *failed* and the user can retry; selecting a thread
    re-fetches its state (`GET /threads/:id`). The backend turn is bounded by the
    SSE idle timeout, so a stalled turn ends deterministically.
- `src/lib/threadsApi.ts` — the thread REST client (create/list/get), mirroring
  `src/core/server.ts` routing, all bearer-authenticated.

## Typed rendering (exit criterion 3)

The conversation view model (`src/lib/conversation.ts`) is a pure reducer over
the typed events. The presentational components render **only** from those typed
payloads:

- `components/Citations.tsx` — the `citation` event's app-owned `SseCitation`
  fields + `evidenceStatus`; the *insufficient-evidence* state renders distinctly
  (no citations). Citations are **never** fabricated client-side.
- `components/ToolProgress.tsx` — the discriminated `tool` events (call/result,
  `status: ok|error`); a tool error renders distinctly.
- `components/Artifacts.tsx` — the `artifact` envelope; the `grounded_answer`
  kind is narrowed and rendered specifically, any other kind renders generically
  as `kind` + pretty JSON (never executed, never treated as HTML).

## Zero-credentials bundle check (exit criterion 1 guardrail)

`scripts/check-bundle-no-secrets.mjs` scans the **built** output under
`web/dist` and **fails (non-zero exit)** if it finds any:

- `LITELLM_*` env var name,
- `sk-…`-style secret key,
- `apiKey` / provider `baseURL` literal,
- any caller-supplied exact needle (CLI arg or `BUNDLE_FORBIDDEN_NEEDLES` CSV) —
  e.g. the actual configured proxy secret, so CI can assert *that* value never
  leaks.

Run it:

```bash
# web/
npm run build:check     # vite build, then scan (recommended in CI)
npm run test:bundle     # scan an already-built web/dist
```

It is deterministic (no network, no randomness) and is the concrete artifact
Phase 9 closeout / 9d cite for "the browser contains no provider credentials".

## E2E (Playwright, opt-in)

Phase 9d adds **browser end-to-end tests** (Playwright/Chromium) that deliver
**Phase 9 exit criterion 4** — *"Browser end-to-end tests cover primary
journeys, recovery paths, and unauthorized access."* They are **opt-in** and
**separate from `npm test`**: the web Vitest suite (`npm test`) has its own
config (`vitest.config.ts`, `test/**`) and never runs anything under `e2e/`; the
backend's root `npm test` is completely unaffected (nothing under `web/` is in
its scope).

### Run it

```bash
# web/
npm run e2e:install   # one-time: download the Chromium browser binary
npm run e2e           # build the SPA, boot the deterministic harness, run the specs
```

`npm run e2e` (via `playwright.config.ts`) starts **two** servers and tears them
down after the run:

1. **A deterministic OFFLINE backend harness** — `web/e2e/fixtures/test-server.ts`,
   run with the repo root's `tsx` (`cwd: ..`) so it resolves the backend's real
   `node_modules`. It boots the **real** `createServer(deps)` from
   `src/core/server.ts` — real **auth**, **ownership** (identical-404 for
   not-owned/nonexistent, generic-401), the real **v2 SSE wire**, and real
   **redaction** — with a **real `ThreadStore`** over a temp sqlite db under
   `os.tmpdir()` (**outside the repo**). **Only the model boundary is faked**: an
   injected `answerTurn` returns *scripted* grounded results keyed off the message
   text (a supported answer with one trusted citation, and — when the message
   contains "insufficient" — the insufficient-evidence case). **No live LiteLLM /
   no network / no `./data`.** The plain-chat `createChat` is wired to throw, so
   the grounded path is provably always used. (Benign placeholder `LITELLM_*` env
   values are set only so the backend `config` module loads; they are never used
   to make a call and never reach the browser bundle.)
2. **The production SPA bundle** — `vite build && vite preview` serving
   `web/dist` (so the browser runs the same **zero-credentials** bundle the 9c
   scan guards). Its proxy (`preview.proxy`, `WEB_PROXY_TARGET`) forwards
   `/threads*` to the harness, keeping the browser **same-origin** (no CORS).

Because the specs drive the **real** server over the **real** v2 wire, the
primary-journey citation assertions (rendered fields **equal** the scripted
app-owned values) are the **end-to-end wire-compat proof** that the 9c
re-declared v2 types match the live backend — the drift risk `sse-contract.ts`
flags is what this verifies.

### What each spec covers (→ exit criterion 4)

- **`primary.spec.ts`** — *primary journey*: token gate → valid token → create
  thread (appears in the list) → send a grounded message → the answer streams in
  → **citations** render from the typed `citation` event (rendered `sourceUri` /
  `documentId` / `marker` / `chunkIndex` / `score` **equal** the scripted
  app-owned values; `ownerId` is dropped from the wire and absent in the DOM) →
  the `grounded_answer` **artifact** renders. Also the **insufficient-evidence**
  case: the distinct insufficient state renders, with **no** citations (no
  fabricated sources).
- **`recovery.spec.ts`** — *recovery paths*: **cancel** (start a turn, click
  cancel → `turn-canceled`, not `turn-error`); **retry** (after cancel, retry
  re-issues and succeeds with **no duplicate user bubble**); **reconnect** (a
  dropped stream → `turn-error`, then retry recovers). The dropped-stream and
  mid-stream-cancel timing are made **deterministic** via Playwright **route
  interception** on the chat SSE POST (hold-pending for cancel; a truncated
  SSE body with no `done` for the drop) — chosen over a bespoke harness "drop"
  mode because it is self-contained and deterministic.
- **`unauthorized.spec.ts`** — *unauthorized access*: an **invalid bearer token**
  yields a generic auth failure (back to the gate, no thread list, and **no**
  provider/secret string in the UI); and **cross-owner isolation** — user2
  cannot reach user1's thread (identical **404**, no existence leak, no data
  leak), driven with a direct authenticated `fetch` in the browser context using
  the **two-owner** auth map.
- **`resume.spec.ts`** — *thread resume*: select a thread, send, switch to a new
  one and back. The backend exposes **no full history replay**; per 9c, selecting
  a thread **re-fetches its metadata** (`GET /threads/:id`) and **resets the
  transcript**. The spec asserts that **documented** behavior (the thread is
  selectable/active, the transcript is reset, a new turn works) — **not** a false
  history replay.

### Notes

- **Deterministic + offline** — no live LiteLLM / no network model call; the
  harness fakes only the model boundary and uses the **real** server code paths.
- **Not part of `npm test` / CI-fast** — separate dir + config; the web Vitest
  `include` stays `test/**`, never `e2e/**`. The root offline suite is untouched.
- **Browser download** — `npm run e2e:install` downloads Chromium. In an
  environment without network for the browser binary, the specs/config/harness
  are still complete; run them where the browser is available.
- **Hygiene** — Playwright's `playwright-report/` and `test-results/`, and the
  harness's temp sqlite dir (under `os.tmpdir()`), are gitignored / outside the
  repo. The E2E never writes `./data` or repo docs.
- **Type-checking** — the `e2e/` dir is covered by `tsconfig.e2e.json` (part of
  `npm run typecheck`), strict, no `any`. The harness re-declares only the narrow
  backend surface it uses (`e2e/fixtures/backend-types.ts`) so the web project
  never type-checks the backend `src/**` — same isolation pattern as the 9c SSE
  mirror.

## Commands (all from `web/`)

```bash
npm run dev          # Vite dev server (proxies API to the backend)
npm run build        # tsc -b && vite build -> web/dist
npm run build:check  # build, then run the zero-creds bundle scan
npm run preview      # preview the production build locally
npm run typecheck    # tsc -b (strict; no emit)
npm test             # web-only Vitest unit tests (jsdom)
npm run test:bundle  # zero-creds bundle scan over an existing web/dist
npm run e2e:install  # one-time: download the Chromium browser for E2E
npm run e2e          # opt-in Playwright browser E2E (deterministic offline harness)
```
