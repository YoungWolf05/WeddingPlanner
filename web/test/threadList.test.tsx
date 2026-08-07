import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThreadList } from "../src/components/ThreadList.js";
import type { Thread } from "../src/lib/threadsApi.js";

// BUG 1: the sidebar must never render "(untitled)". A titled thread shows its
// title; a null-title thread shows the "New conversation" placeholder; and a
// pending draft shows as an active, non-selectable thread item.

function makeThread(overrides: Partial<Thread>): Thread {
  return {
    id: "t-1",
    ownerId: "owner-1",
    title: "Venue ideas",
    createdAt: 1,
    updatedAt: 2,
    ...overrides,
  };
}

const noop = (): void => {};

describe("ThreadList", () => {
  it("renders a thread's title and never '(untitled)'", () => {
    render(
      <ThreadList
        threads={[makeThread({ title: "Venue ideas" })]}
        currentThreadId={null}
        onSelect={noop}
        onCreate={noop}
        onRefresh={noop}
        busy={false}
        draft={false}
      />
    );
    const item = screen.getByTestId("thread-item");
    expect(item.textContent).toContain("Venue ideas");
    expect(item.textContent).not.toContain("untitled");
  });

  it("shows 'New conversation' for a null-title thread (not '(untitled)')", () => {
    render(
      <ThreadList
        threads={[makeThread({ title: null })]}
        currentThreadId={null}
        onSelect={noop}
        onCreate={noop}
        onRefresh={noop}
        busy={false}
        draft={false}
      />
    );
    const item = screen.getByTestId("thread-item");
    expect(item.textContent).toContain("New conversation");
    expect(item.textContent).not.toContain("untitled");
  });

  it("renders a pending draft as the active thread item", () => {
    render(
      <ThreadList
        threads={[]}
        currentThreadId={null}
        onSelect={noop}
        onCreate={noop}
        onRefresh={noop}
        busy={false}
        draft={true}
      />
    );
    const item = screen.getByTestId("thread-item");
    expect(item.getAttribute("aria-current")).toBe("true");
    expect(item.getAttribute("data-thread-id")).toBe("draft");
    expect(item.textContent).toContain("New conversation");
  });

  it("shows the empty state when there are no threads and no draft", () => {
    render(
      <ThreadList
        threads={[]}
        currentThreadId={null}
        onSelect={noop}
        onCreate={noop}
        onRefresh={noop}
        busy={false}
        draft={false}
      />
    );
    expect(screen.queryByTestId("thread-item")).toBeNull();
    expect(screen.getByText(/No conversations yet/i)).toBeTruthy();
  });
});
