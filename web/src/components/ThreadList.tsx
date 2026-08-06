// Phase 9 (9c): the thread sidebar — list, select (resume), and create.
//
// Threads come from GET /threads (owner-scoped server-side). Selecting a thread
// resumes it. `thread_id` is a server-issued conversation key, never identity.

import type { Thread } from "../lib/threadsApi.js";

interface ThreadListProps {
  threads: Thread[];
  currentThreadId: string | null;
  onSelect: (threadId: string) => void;
  onCreate: () => void;
  onRefresh: () => void;
  busy: boolean;
}

export function ThreadList(props: ThreadListProps): React.ReactElement {
  const { threads, currentThreadId, onSelect, onCreate, onRefresh, busy } =
    props;

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
      {threads.length === 0 ? (
        <div className="threads__empty">No conversations yet.</div>
      ) : (
        <ul className="threads__list">
          {threads.map((thread) => (
            <li key={thread.id}>
              <button
                type="button"
                className={
                  thread.id === currentThreadId
                    ? "threads__item threads__item--active"
                    : "threads__item"
                }
                onClick={() => onSelect(thread.id)}
                data-testid="thread-item"
                data-thread-id={thread.id}
                aria-current={thread.id === currentThreadId}
              >
                {thread.title ?? "(untitled)"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
