// Phase 9 (9d) E2E — PRIMARY JOURNEY + INSUFFICIENT EVIDENCE (exit criterion 4:
// "primary journeys"; also the WIRE-COMPAT proof for the 9c re-declared v2 types).
//
// These run against the PRODUCTION SPA bundle (vite preview) talking to the REAL
// backend server through the deterministic harness (fixtures/test-server.ts):
// real auth + ownership + SSE v2 wire + redaction; only the model boundary is a
// scripted answerTurn. So the assertions below are a genuine end-to-end proof
// that the browser + the real v2 SSE wire interoperate.

import { expect, test } from "@playwright/test";
import { TOKEN_USER } from "./fixtures/auth.js";
import {
  SCRIPTED_CITATION,
  SUPPORTED_ANSWER,
} from "./fixtures/scripted.js";
import { createThread, sendMessage, signInWith } from "./helpers.js";

test.describe("primary journey", () => {
  test("token gate -> create thread -> grounded answer streams with trusted citations + artifact (v2 wire-compat proof)", async ({
    page,
  }) => {
    // Load the SPA and sign in with a valid bearer token.
    await signInWith(page, TOKEN_USER);

    // Create a conversation; it appears in the thread list and becomes active.
    // (The deterministic harness keeps one store across specs, so we assert the
    // NEWLY created thread is present + active rather than an absolute list size.)
    await createThread(page);
    await expect(page.getByTestId("thread-list")).toBeVisible();
    const newThread = page.getByTestId("thread-item").first();
    await expect(newThread).toHaveAttribute("aria-current", "true");

    // Send a grounded message; the assistant turn streams in.
    await sendMessage(page, "recommend a venue");
    await expect(page.getByTestId("message-user")).toHaveText(
      "recommend a venue"
    );

    const assistant = page.getByTestId("message-assistant");
    await expect(assistant).toBeVisible();
    // The full grounded answer arrives (emitted as one token by the grounded
    // path) and the turn finishes ("done" => data-status "done").
    await expect(assistant).toContainText(SUPPORTED_ANSWER);
    await expect(assistant).toHaveAttribute("data-status", "done");

    // CITATIONS render from the TYPED citation event — supported evidence state.
    await expect(page.getByTestId("citations")).toBeVisible();
    await expect(page.getByTestId("evidence-supported")).toBeVisible();
    await expect(page.getByTestId("evidence-insufficient")).toHaveCount(0);

    // Exactly one citation, and its rendered app-owned fields EQUAL the scripted
    // values — the end-to-end wire-compat proof (the 9c v2 types match the live
    // backend projection). sourceUri is the rendered source text; documentId is
    // carried on the title attribute; marker + chunkIndex + score are shown.
    const citation = page.getByTestId("citation-item");
    await expect(citation).toHaveCount(1);
    await expect(page.getByTestId("citation-marker")).toHaveText(
      `[${String(SCRIPTED_CITATION.marker)}]`
    );
    const source = page.getByTestId("citation-source");
    await expect(source).toHaveText(SCRIPTED_CITATION.sourceUri);
    await expect(source).toHaveAttribute("title", SCRIPTED_CITATION.documentId);
    await expect(citation).toContainText(
      `chunk #${String(SCRIPTED_CITATION.chunkIndex)}`
    );
    await expect(citation).toContainText(SCRIPTED_CITATION.score.toFixed(3));

    // The internal authorization field (ownerId) is DROPPED from the wire, so it
    // must never appear in the rendered citation DOM.
    await expect(citation).not.toContainText("ownerId");

    // The grounded_answer ARTIFACT renders from the typed artifact envelope.
    await expect(page.getByTestId("artifacts")).toBeVisible();
    await expect(page.getByTestId("artifact-grounded")).toBeVisible();
    await expect(page.getByTestId("artifact-evidence-status")).toHaveText(
      "supported"
    );

    // No turn error / cancel state on the happy path.
    await expect(page.getByTestId("turn-error")).toHaveCount(0);
    await expect(page.getByTestId("turn-canceled")).toHaveCount(0);
  });

  test("insufficient evidence: distinct state, NO citations, no fabricated sources", async ({
    page,
  }) => {
    await signInWith(page, TOKEN_USER);
    await createThread(page);

    // The harness keys the insufficient case off the word "insufficient".
    await sendMessage(page, "insufficient — an obscure unanswerable question");

    const assistant = page.getByTestId("message-assistant");
    await expect(assistant).toBeVisible();
    await expect(assistant).toHaveAttribute("data-status", "done");

    // The distinct insufficient-evidence state renders...
    await expect(page.getByTestId("evidence-insufficient")).toBeVisible();
    // ...the supported citations block does NOT, and there are zero citations
    // (no fabricated sources).
    await expect(page.getByTestId("citations")).toHaveCount(0);
    await expect(page.getByTestId("citation-item")).toHaveCount(0);

    // The artifact still renders, reporting insufficient evidence.
    await expect(page.getByTestId("artifact-evidence-status")).toHaveText(
      "insufficient"
    );
  });
});
