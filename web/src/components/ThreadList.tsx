// Phase 9 (9c): the thread sidebar — list, select (resume), and create.
//
// Threads come from GET /threads (owner-scoped server-side). Selecting a thread
// resumes it. `thread_id` is a server-issued conversation key, never identity.

import { threadTitleLabel, UNTITLED_PLACEHOLDER } from "../lib/title.js";
import type { Thread } from "../lib/threadsApi.js";

interface ThreadListProps {
  threads: Thread[];
  currentThreadId: string | null;
  onSelect: (threadId: string) => void;
  onCreate: () => void;
  onRefresh: () => void;
  busy: boolean;
  // A brand-new, not-yet-persisted conversation (BUG 1: lazy-create). When
  // present it is rendered as the active item at the top of the list with a
  // placeholder title; it becomes a real server thread on the first send.
  draft: boolean;
}

export function ThreadList(props: ThreadListProps): React.ReactElement {
  const { threads, currentThreadId, onSelect, onCreate, onRefresh, busy, draft } =
    props;

  const hasItems = draft || threads.length > 0;

  return (
    <aside className="threads" data-testid="thread-list">
      <div className="threads__actions">
        <button
          type="button"
          onClick={onCreate}
          disabled={busy}
          data-testid="new-thread"
        >
          + New conversation
        </button>
        <button
          type="button"
          onClick={onRefresh}
          disabled={busy}
          data-testid="refresh-threads"
        >
          Refresh
        </button>
      </div>
      {!hasItems ? (
        <div className="threads__empty">
          No conversations yet. Start one to see it here.
        </div>
      ) : (
        <ul className="threads__list">
          {draft ? (
            // The unsaved draft: an active, non-selectable item (there is no
            // server id yet). It keeps `new-thread + first send yields a listed
            // thread` intact and shows the placeholder until the title is set.
            <li key="__draft__">
              <span
                className="threads__item threads__item--active threads__item--draft"
                data-testid="thread-item"
                data-thread-id="draft"
                aria-current={true}
              >
                <span className="threads__title">{UNTITLED_PLACEHOLDER}</span>
                <span className="threads__badge">Draft</span>
              </span>
            </li>
          ) : null}
          {threads.map((thread) => {
            const active = !draft && thread.id === currentThreadId;
            return (
              <li key={thread.id}>
                <button
                  type="button"
                  className={
                    active
                      ? "threads__item threads__item--active"
                      : "threads__item"
                  }
                  onClick={() => onSelect(thread.id)}
                  data-testid="thread-item"
                  data-thread-id={thread.id}
                  aria-current={active}
                >
                  <span className="threads__title">
                    {threadTitleLabel(thread.title)}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
