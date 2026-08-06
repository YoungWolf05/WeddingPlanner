// Phase 9 (9d) E2E — THREAD RESUME (exit criterion 4: primary journeys / resume).
//
// DOCUMENTED BEHAVIOR (matches 9c). The backend does NOT yet expose full history
// replay; per 9c, selecting a thread RE-FETCHES the thread's metadata (GET
// /threads/:id — confirming ownership/existence) and RESETS the transcript. So
// this spec asserts that documented behavior — the resumed thread is selectable,
// becomes the active conversation, and the app is in a coherent state (a fresh,
// empty transcript ready for a new turn) — NOT a false claim of replayed history.

import { expect, test } from "@playwright/test";
import { TOKEN_USER } from "./fixtures/auth.js";
import { SUPPORTED_ANSWER } from "./fixtures/scripted.js";
import { createThread, sendMessage, signInWith } from "./helpers.js";

test.describe("thread resume", () => {
  test("select a thread, send, create/switch to another, and back — resume re-fetches metadata + resets transcript", async ({
    page,
  }) => {
    await signInWith(page, TOKEN_USER);

    // Create thread A (the newly created thread is selected + first in the list)
    // and send a message; the assistant answer renders. NOTE: the deterministic
    // harness keeps ONE backend process (and one store) across specs, so we
    // reason about the SPECIFIC threads this spec creates by id rather than about
    // absolute list counts.
    await createThread(page);
    const threadAId = await page
      .getByTestId("thread-item")
      .first()
      .getAttribute("data-thread-id");
    expect(threadAId).not.toBeNull();

    await sendMessage(page, "recommend a venue");
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toBeVisible();

    // Create thread B (a second conversation): it becomes active with a fresh
    // (empty) transcript, and both A and B are present in the list. Read B's id
    // from the ACTIVE item (aria-current) rather than list position — a
    // background thread-list refresh from A's completed turn can reorder the list
    // by recency, so "first" is not reliably the just-created thread.
    await createThread(page);
    const activeItem = page.locator(
      '[data-testid="thread-item"][aria-current="true"]'
    );
    await expect(activeItem).toHaveCount(1);
    const threadBId = await activeItem.getAttribute("data-thread-id");
    expect(threadBId).not.toBeNull();
    expect(threadBId).not.toBe(threadAId);
    await expect(
      page.locator(`[data-testid="thread-item"][data-thread-id="${threadAId!}"]`)
    ).toHaveCount(1);
    await expect(
      page.locator(`[data-testid="thread-item"][data-thread-id="${threadBId!}"]`)
    ).toHaveCount(1);
    await expect(page.getByTestId("message-assistant")).toHaveCount(0);

    // Resume thread A by selecting it. It re-fetches (GET /threads/:id) and
    // becomes the active conversation.
    const threadA = page.locator(
      `[data-testid="thread-item"][data-thread-id="${threadAId!}"]`
    );
    await threadA.click();
    await expect(threadA).toHaveAttribute("aria-current", "true");

    // DOCUMENTED resume behavior: the transcript is RESET (no full history replay
    // is claimed) and the app is coherent — the composer is enabled, ready for a
    // new turn. We assert the reset, NOT a false history replay.
    await expect(page.getByTestId("message-user")).toHaveCount(0);
    await expect(page.getByTestId("message-assistant")).toHaveCount(0);
    await expect(page.getByTestId("composer-input")).toBeEnabled();

    // A new turn on the resumed thread still works end-to-end.
    await sendMessage(page, "another venue idea");
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toBeVisible();
  });
});
