// Phase 9 (9d) E2E — THREAD RESUME (exit criterion 4: primary journeys / resume).
//
// CONVERSATION-HISTORY REPLAY (this increment). Selecting a thread now HYDRATES
// its transcript from the backend history-replay route (GET /threads/:id/messages
// — real auth + real ownership gate + real redaction) instead of resetting to an
// empty transcript. So this spec asserts that switching AWAY from a thread and
// then BACK to it shows that thread's real prior messages (the user's sent
// message + the assistant reply), not a blank slate.
//
// DETERMINISTIC HISTORY SOURCE. The E2E harness runs the GROUNDED path
// (answerTurn), which does NOT persist to the conversational checkpointer, so the
// harness records each completed turn per-thread and serves it through the
// injected readHistory seam (see fixtures/test-server.ts). The route, auth,
// ownership gate, and wire shape exercised are the REAL production code.
//
// LAZY-CREATE (BUG 1 fix). "+ New conversation" prepares a UI DRAFT and the
// server thread is created on the FIRST send (title derived from that message,
// which PERSISTS). While a conversation is a draft it renders as an active
// thread-item with data-thread-id="draft"; the REAL server id only exists after
// the first send. So this spec reads each thread's real id AFTER its first
// message is sent.

import { expect, test } from "@playwright/test";
import { TOKEN_USER } from "./fixtures/auth.js";
import { SUPPORTED_ANSWER } from "./fixtures/scripted.js";
import { createThread, sendMessage, signInWith } from "./helpers.js";

test.describe("thread resume", () => {
  test("switching away from a thread and back replays its real history (user + assistant messages)", async ({
    page,
  }) => {
    await signInWith(page, TOKEN_USER);

    // Create thread A (a UI draft) and send its first message; the real server
    // thread is created on that send with a derived title, and the assistant
    // answer renders. NOTE: the deterministic harness keeps ONE backend process
    // (and one store) across specs, so we reason about the SPECIFIC threads this
    // spec creates by id rather than about absolute list counts. Read A's REAL id
    // from the active item AFTER the first send (the pre-send draft has no id).
    await createThread(page);
    await sendMessage(page, "recommend a venue");
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toBeVisible();
    const threadAId = await page
      .locator('[data-testid="thread-item"][aria-current="true"]')
      .getAttribute("data-thread-id");
    expect(threadAId).not.toBeNull();
    expect(threadAId).not.toBe("draft");

    // Create thread B (a second conversation): it becomes active as a draft with
    // a fresh (empty) transcript. Send its first message so it too becomes a real
    // server thread, then read B's REAL id from the ACTIVE item (aria-current)
    // rather than list position — a background thread-list refresh can reorder
    // the list by recency, so "first" is not reliably the just-created thread.
    await createThread(page);
    await sendMessage(page, "another venue for the shortlist");
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toBeVisible();
    const activeItem = page.locator(
      '[data-testid="thread-item"][aria-current="true"]'
    );
    await expect(activeItem).toHaveCount(1);
    const threadBId = await activeItem.getAttribute("data-thread-id");
    expect(threadBId).not.toBeNull();
    expect(threadBId).not.toBe("draft");
    expect(threadBId).not.toBe(threadAId);
    await expect(
      page.locator(`[data-testid="thread-item"][data-thread-id="${threadAId!}"]`)
    ).toHaveCount(1);
    await expect(
      page.locator(`[data-testid="thread-item"][data-thread-id="${threadBId!}"]`)
    ).toHaveCount(1);

    // Resume thread A by selecting it. It re-fetches (GET /threads/:id) then
    // HYDRATES the transcript from GET /threads/:id/messages, and becomes active.
    const threadA = page.locator(
      `[data-testid="thread-item"][data-thread-id="${threadAId!}"]`
    );
    await threadA.click();
    await expect(threadA).toHaveAttribute("aria-current", "true");

    // HISTORY REPLAY: thread A's real prior messages appear — the user's first
    // message AND the assistant reply — NOT a blank transcript. B's message must
    // NOT appear on A.
    await expect(
      page.getByTestId("message-user").filter({ hasText: "recommend a venue" })
    ).toBeVisible();
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toBeVisible();
    await expect(
      page
        .getByTestId("message-user")
        .filter({ hasText: "another venue for the shortlist" })
    ).toHaveCount(0);
    // The replayed assistant turn is a COMPLETED turn (not streaming).
    await expect(
      page.locator('[data-testid="message-assistant"][data-status="done"]')
    ).toHaveCount(1);
    // Composer remains enabled — ready for a new turn on the resumed thread.
    await expect(page.getByTestId("composer-input")).toBeEnabled();

    // A new turn on the resumed thread still works end-to-end, appended AFTER the
    // replayed history.
    await sendMessage(page, "another venue idea");
    await expect(
      page.getByTestId("message-user").filter({ hasText: "another venue idea" })
    ).toBeVisible();
    await expect(
      page.getByTestId("message-assistant").filter({ hasText: SUPPORTED_ANSWER })
    ).toHaveCount(2);
  });
});
