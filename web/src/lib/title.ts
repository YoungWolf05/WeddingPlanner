// Phase 9 (post-9d fix): derive a human, persistable conversation title from the
// first user message — standard chat-app behavior.
//
// The backend supports a `title` ONLY at CREATE time (POST /threads { title? });
// there is no update route (see src/core/server.ts). So the App creates a thread
// LAZILY on the first send using this derived title, which is why the title then
// persists across reload (it is stored at creation). This helper is a PURE
// function so it is unit-tested directly and used identically in the UI.

// The maximum length of a derived title (characters, before the ellipsis). Kept
// in the ~40–60 range so the sidebar shows a meaningful, single-line label.
export const MAX_TITLE_LENGTH = 48;

// The single-character ellipsis appended when a title is truncated.
const ELLIPSIS = "…";

// Derive a title from a raw user message:
//   - trim leading/trailing whitespace,
//   - collapse any internal whitespace run (incl. newlines) to a single space,
//   - truncate to MAX_TITLE_LENGTH, cutting on a word boundary when reasonable
//     and appending a single-character ellipsis.
// Returns null when the message has no usable (non-whitespace) content, so the
// caller can fall back to a placeholder ("New conversation") instead.
export function deriveThreadTitle(message: string): string | null {
  const collapsed = message.replace(/\s+/g, " ").trim();
  if (collapsed === "") return null;
  if (collapsed.length <= MAX_TITLE_LENGTH) return collapsed;

  // Truncate to the budget, then prefer to break at the last word boundary so we
  // don't slice a word in half — unless that would drop too much (a single very
  // long token), in which case a hard cut is fine.
  const hardCut = collapsed.slice(0, MAX_TITLE_LENGTH);
  const lastSpace = hardCut.lastIndexOf(" ");
  const body =
    lastSpace >= Math.floor(MAX_TITLE_LENGTH * 0.6)
      ? hardCut.slice(0, lastSpace)
      : hardCut;
  return `${body.trimEnd()}${ELLIPSIS}`;
}

// The placeholder shown for a conversation that has no title yet (a brand-new
// draft, or — defensively — any thread whose stored title is null/empty). This
// replaces the old "(untitled)" fallback so a created thread never reads as
// untitled.
export const UNTITLED_PLACEHOLDER = "New conversation";

// Resolve the label to show for a thread title value (possibly null/empty),
// falling back to the placeholder. Used by the sidebar.
export function threadTitleLabel(title: string | null | undefined): string {
  if (title === null || title === undefined) return UNTITLED_PLACEHOLDER;
  const trimmed = title.trim();
  return trimmed === "" ? UNTITLED_PLACEHOLDER : trimmed;
}
