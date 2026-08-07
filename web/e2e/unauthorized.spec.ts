// Phase 9 (9d) E2E — UNAUTHORIZED ACCESS (exit criterion 4, explicit).
//
// Two independent unauthorized journeys, both against the REAL server (real auth
// + real ownership) through the harness:
//   (a) INVALID BEARER TOKEN — the app surfaces a generic auth failure, does not
//       show a thread list, and leaks NO provider/secret detail in the UI.
//   (b) CROSS-OWNER ACCESS — a valid user (user2) CANNOT reach another owner's
//       (user1's) thread: an identical 404 with no data leak. The SPA only lists
//       the caller's own threads, so this is driven at the API/journey level with
//       a direct authenticated fetch IN THE BROWSER CONTEXT (same bearer the app
//       uses), proving ownership isolation end-to-end through the real server.

import { expect, test } from "@playwright/test";
import { TOKEN_INVALID, TOKEN_USER, TOKEN_USER2 } from "./fixtures/auth.js";
import { createThread, sendMessage, signInWith } from "./helpers.js";
import { SUPPORTED_ANSWER } from "./fixtures/scripted.js";

test.describe("unauthorized access", () => {
  test("invalid bearer token: generic auth failure, no thread list, no secret leak", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.getByTestId("token-gate")).toBeVisible();

    // Enter an invalid token and continue.
    await page.getByTestId("token-input").fill(TOKEN_INVALID);
    await page.getByTestId("token-submit").click();

    // The app handles the 401 by signing the user out — it returns to the token
    // gate and never shows the conversation view / thread list.
    await expect(page.getByTestId("token-gate")).toBeVisible();
    await expect(page.getByTestId("thread-list")).toHaveCount(0);
    await expect(page.getByTestId("sign-out")).toHaveCount(0);

    // No provider/secret detail leaks anywhere in the rendered page. (The 9c
    // bundle scan is the primary guardrail; this asserts the runtime UI too.)
    // NOTE: the gate's benign help copy legitimately mentions "bearer token" —
    // that is UI guidance, not a leaked credential — so we assert on the actual
    // secret shapes (provider name / key prefixes), never on that benign word.
    const bodyText = (await page.locator("body").innerText()).toLowerCase();
    expect(bodyText).not.toContain("litellm");
    expect(bodyText).not.toContain("sk-");
    expect(bodyText).not.toContain("api_key");
    expect(bodyText).not.toContain("apikey=");
  });

  test("cross-owner: user2 cannot access user1's thread (identical 404, no leak)", async ({
    page,
  }) => {
    // User1 creates a conversation. LAZY-CREATE (BUG 1 fix): the server thread is
    // created on the FIRST send, so we send a message to persist a REAL thread,
    // then capture its server-issued id from the active DOM item (the pre-send
    // draft carries data-thread-id="draft", not a real id).
    await signInWith(page, TOKEN_USER);
    await createThread(page);
    await sendMessage(page, "recommend a venue");
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toBeVisible();
    const user1ThreadId = await page
      .locator('[data-testid="thread-item"][aria-current="true"]')
      .getAttribute("data-thread-id");
    expect(user1ThreadId).not.toBeNull();
    expect(user1ThreadId).not.toBe("draft");

    // Sign out and sign in as user2.
    await page.getByTestId("sign-out").click();
    await signInWith(page, TOKEN_USER2);

    // User2's own list does NOT contain user1's thread (owner-scoped listing).
    await expect(
      page.locator(`[data-testid="thread-item"][data-thread-id="${user1ThreadId!}"]`)
    ).toHaveCount(0);

    // Drive the cross-owner attempt at the API level with user2's REAL bearer
    // token, from the browser context (same-origin, through the app's proxy).
    // GET a random (nonexistent) id and GET user1's id: they must be IDENTICAL
    // 404s (no existence leak), and neither returns user1's content.
    const probe = await page.evaluate(
      async (args: { ownedId: string; token: string }) => {
        const headers = {
          Authorization: `Bearer ${args.token}`,
          Accept: "application/json",
        };
        const randomId = "00000000-0000-4000-8000-000000000000";
        const ownedRes = await fetch(`/threads/${args.ownedId}`, { headers });
        const randomRes = await fetch(`/threads/${randomId}`, { headers });
        return {
          ownedStatus: ownedRes.status,
          randomStatus: randomRes.status,
          ownedBody: await ownedRes.text(),
          randomBody: await randomRes.text(),
        };
      },
      { ownedId: user1ThreadId!, token: TOKEN_USER2 }
    );

    // Identical 404 for not-owned and nonexistent — no existence leak.
    expect(probe.ownedStatus).toBe(404);
    expect(probe.randomStatus).toBe(404);
    expect(probe.ownedBody).toBe(probe.randomBody);
    // The response never contains user1's owner id / thread content.
    expect(probe.ownedBody).not.toContain("e2e-user");
    expect(probe.ownedBody).not.toContain(user1ThreadId!);
  });
});
