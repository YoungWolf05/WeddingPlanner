// Phase (history) hydration test: selecting a thread HYDRATES the transcript
// from GET /threads/:id/messages instead of leaving it blank.
//
// OFFLINE + deterministic: a FAKE fetch routes by URL/method to canned JSON — no
// network, no real backend, no SSE. We render the real <App/>, sign in through
// the token gate, then click a listed thread and assert the transcript shows the
// replayed user + assistant messages (the assistant as a COMPLETED turn,
// data-status="done"). A separate case proves a thread with no history hydrates
// to an empty transcript (and shows the empty-state), and another proves a
// history-fetch error falls back to an empty transcript + a notice (no crash).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { App } from "../src/App.js";

const TOKEN = "test-token";

interface Route {
  match: (url: string, method: string) => boolean;
  respond: () => { status: number; body: unknown };
}

function ok(body: unknown): { status: number; body: unknown } {
  return { status: 200, body };
}

// Install a URL/method-routed fake fetch. The first matching route wins; an
// unmatched request resolves to 404 so a stray call is visible, not silent.
function installRoutedFetch(routes: Route[]): void {
  vi.stubGlobal(
    "fetch",
    (async (url: string, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const route = routes.find((r) => r.match(url, method));
      const result = route
        ? route.respond()
        : { status: 404, body: { error: "Not found" } };
      return {
        ok: result.status >= 200 && result.status < 300,
        status: result.status,
        json: async () => result.body,
      } as unknown as Response;
    }) as typeof fetch
  );
}

function thread(id: string, title: string): Record<string, unknown> {
  return { id, ownerId: "owner", title, createdAt: 1, updatedAt: 2 };
}

async function signIn(): Promise<void> {
  render(<App />);
  fireEvent.change(screen.getByTestId("token-input"), {
    target: { value: TOKEN },
  });
  fireEvent.click(screen.getByTestId("token-submit"));
  await waitFor(() => expect(screen.getByTestId("sign-out")).toBeTruthy());
}

beforeEach(() => {
  sessionStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});

describe("App history hydration on thread select", () => {
  it("hydrates the transcript with replayed user + assistant messages", async () => {
    installRoutedFetch([
      {
        match: (u, m) => u.endsWith("/threads") && m === "GET",
        respond: () => ok({ threads: [thread("t-a", "Venue ideas")] }),
      },
      {
        match: (u, m) => u.endsWith("/threads/t-a/messages") && m === "GET",
        respond: () =>
          ok({
            messages: [
              { role: "user", text: "We have 120 guests." },
              { role: "assistant", text: "Here are venue ideas." },
            ],
          }),
      },
      {
        match: (u, m) => u.endsWith("/threads/t-a") && m === "GET",
        respond: () => ok({ thread: thread("t-a", "Venue ideas") }),
      },
    ]);

    await signIn();

    // The listed thread appears; click it to resume.
    const item = await screen.findByTestId("thread-item");
    fireEvent.click(item);

    // The transcript hydrates: the user message + a COMPLETED assistant turn.
    await waitFor(() =>
      expect(screen.getByTestId("message-user").textContent).toContain(
        "We have 120 guests."
      )
    );
    const assistant = screen.getByTestId("message-assistant");
    expect(assistant.textContent).toContain("Here are venue ideas.");
    expect(assistant.getAttribute("data-status")).toBe("done");
    // No notice on the happy path.
    expect(screen.queryByTestId("notice")).toBeNull();
  });

  it("hydrates to an EMPTY transcript for a thread with no history", async () => {
    installRoutedFetch([
      {
        match: (u, m) => u.endsWith("/threads") && m === "GET",
        respond: () => ok({ threads: [thread("t-empty", "Fresh")] }),
      },
      {
        match: (u, m) => u.endsWith("/threads/t-empty/messages") && m === "GET",
        respond: () => ok({ messages: [] }),
      },
      {
        match: (u, m) => u.endsWith("/threads/t-empty") && m === "GET",
        respond: () => ok({ thread: thread("t-empty", "Fresh") }),
      },
    ]);

    await signIn();
    fireEvent.click(await screen.findByTestId("thread-item"));

    // The thread becomes active with an empty transcript (empty-state visible).
    await waitFor(() =>
      expect(
        screen.getByTestId("thread-item").getAttribute("aria-current")
      ).toBe("true")
    );
    expect(screen.queryByTestId("message-user")).toBeNull();
    expect(screen.queryByTestId("message-assistant")).toBeNull();
    // Composer is enabled, ready for a new turn.
    expect(
      (screen.getByTestId("composer-input") as HTMLTextAreaElement).disabled
    ).toBe(false);
  });

  it("on a history-fetch error, falls back to an empty transcript + a notice (no crash)", async () => {
    installRoutedFetch([
      {
        match: (u, m) => u.endsWith("/threads") && m === "GET",
        respond: () => ok({ threads: [thread("t-a", "Venue ideas")] }),
      },
      {
        match: (u, m) => u.endsWith("/threads/t-a") && m === "GET",
        respond: () => ok({ thread: thread("t-a", "Venue ideas") }),
      },
      {
        // The history read fails with a server error.
        match: (u, m) => u.endsWith("/threads/t-a/messages") && m === "GET",
        respond: () => ({ status: 500, body: { error: "Internal error" } }),
      },
    ]);

    await signIn();
    fireEvent.click(await screen.findByTestId("thread-item"));

    // Thread still selected; transcript empty; a user-safe notice shown.
    await waitFor(() => expect(screen.getByTestId("notice")).toBeTruthy());
    expect(screen.queryByTestId("message-user")).toBeNull();
    expect(screen.queryByTestId("message-assistant")).toBeNull();
  });
});
