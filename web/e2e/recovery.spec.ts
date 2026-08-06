// Phase 9 (9d) E2E — RECOVERY PATHS (exit criterion 4: "recovery paths").
//
// Covers CANCEL, RETRY, and RECONNECT. The deterministic harness resolves a
// grounded turn near-instantly, so to exercise the mid-stream cancel and the
// dropped-stream reconnect DETERMINISTICALLY we use Playwright ROUTE INTERCEPTION
// on the chat SSE POST (**/threads/*/chat):
//   - CANCEL: hold the intercepted request pending (never fulfill) so the turn
//     stays "streaming"; click cancel; the client's AbortController aborts the
//     fetch and the turn is marked CANCELED (not error).
//   - RECONNECT: fulfill with a TRUNCATED SSE body (init + token, but NO `done`)
//     so the client hits EOF before completion and marks the turn FAILED
//     (turn-error) — the dropped-stream path. Retry then recovers.
//   - RETRY: after a canceled/failed turn, click retry; with interception removed
//     the request reaches the REAL harness backend and succeeds — and NO
//     duplicate user bubble is created.
// Route interception is chosen over a bespoke harness "drop" message because it
// is deterministic, self-contained, and does not require a special backend mode.

import { expect, test, type Route } from "@playwright/test";
import { TOKEN_USER } from "./fixtures/auth.js";
import { SUPPORTED_ANSWER } from "./fixtures/scripted.js";
import { createThread, sendMessage, signInWith } from "./helpers.js";

const CHAT_GLOB = "**/threads/*/chat";

// A one-shot gate: a promise the test resolves later. Avoids the closure
// control-flow-narrowing pitfall of assigning a resolver to an outer `let`.
interface Gate {
  promise: Promise<void>;
  release: () => void;
}
function makeGate(): Gate {
  let release = (): void => {};
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

// A truncated SSE stream: a valid v2 init + one token, then EOF with NO `done`.
// The client treats EOF-before-done as a dropped stream (turn failed).
const TRUNCATED_SSE =
  `event: init\ndata: {"version":2,"threadId":"intercepted"}\n\n` +
  `event: token\ndata: {"text":"partial answer that never completes"}\n\n`;

test.describe("recovery paths", () => {
  test("CANCEL mid-stream marks the turn canceled (not error)", async ({
    page,
  }) => {
    await signInWith(page, TOKEN_USER);
    await createThread(page);

    // Hold the chat request pending so the turn stays streaming until we cancel.
    // We never fulfill; the client abort tears the fetch down.
    const gate = makeGate();
    await page.route(CHAT_GLOB, async (route: Route) => {
      await gate.promise; // hold until the test releases (after cancel).
      await route.abort("aborted").catch(() => {});
    });

    await sendMessage(page, "recommend a venue");

    // While streaming, the cancel affordance is shown; click it.
    const cancel = page.getByTestId("cancel-button");
    await expect(cancel).toBeVisible();
    await cancel.click();

    // The turn is marked CANCELED (a user cancel is not an error).
    await expect(page.getByTestId("turn-canceled")).toBeVisible();
    await expect(page.getByTestId("turn-error")).toHaveCount(0);

    // Release the held route so nothing dangles.
    gate.release();
  });

  test("RETRY after cancel re-issues and succeeds with NO duplicate user bubble", async ({
    page,
  }) => {
    await signInWith(page, TOKEN_USER);
    await createThread(page);

    // First turn: hold pending, then cancel.
    const gate = makeGate();
    const holdRoute = async (route: Route): Promise<void> => {
      await gate.promise;
      await route.abort("aborted").catch(() => {});
    };
    await page.route(CHAT_GLOB, holdRoute);

    await sendMessage(page, "recommend a venue");
    await page.getByTestId("cancel-button").click();
    await expect(page.getByTestId("turn-canceled")).toBeVisible();

    // Exactly one user bubble so far.
    await expect(page.getByTestId("message-user")).toHaveCount(1);

    // Remove interception so retry reaches the REAL backend, and release the
    // held request.
    await page.unroute(CHAT_GLOB, holdRoute);
    gate.release();

    // Retry re-issues the SAME message; the real grounded answer arrives.
    await page.getByTestId("retry-button").click();
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toBeVisible();

    // RETRY must NOT append a duplicate user bubble.
    await expect(page.getByTestId("message-user")).toHaveCount(1);
  });

  test("RECONNECT: a dropped stream (EOF before done) fails, and retry recovers", async ({
    page,
  }) => {
    await signInWith(page, TOKEN_USER);
    await createThread(page);

    // Intercept the FIRST chat request and fulfill with a truncated SSE body
    // (no `done`) to simulate a dropped stream; subsequent requests pass through.
    let dropped = false;
    const dropOnce = async (route: Route): Promise<void> => {
      if (!dropped) {
        dropped = true;
        await route.fulfill({
          status: 200,
          headers: { "content-type": "text/event-stream" },
          body: TRUNCATED_SSE,
        });
        return;
      }
      await route.continue();
    };
    await page.route(CHAT_GLOB, dropOnce);

    await sendMessage(page, "recommend a venue");

    // The dropped stream surfaces as a turn error with a retry affordance.
    await expect(page.getByTestId("turn-error")).toBeVisible();
    await expect(page.getByTestId("retry-button")).toBeVisible();

    // Retry: the second request passes through to the real backend and succeeds.
    // Retry appends a NEW assistant turn (the original failed turn remains in the
    // transcript by design), so we assert the recovery turn completed rather than
    // asserting zero historical errors.
    await page.getByTestId("retry-button").click();
    const recovered = page
      .getByTestId("message-assistant")
      .filter({ hasText: SUPPORTED_ANSWER });
    await expect(recovered).toBeVisible();
    await expect(recovered).toHaveAttribute("data-status", "done");
    // RETRY must NOT append a duplicate user bubble.
    await expect(page.getByTestId("message-user")).toHaveCount(1);
  });
});
