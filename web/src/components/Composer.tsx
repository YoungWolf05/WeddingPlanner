// Phase 9 (9c): the message composer with SEND and CANCEL.
//
// SEND posts the message to the chat SSE endpoint (App wires the stream). CANCEL
// aborts the in-flight stream via App's AbortController. The input is disabled
// while no thread is selected.
//
// BUG 1 (lazy-create): the FIRST send in a new conversation lazily creates the
// server thread; if that create fails, `onSend` resolves false and the composer
// RESTORES the typed message so the user does not lose it.

import { useState } from "react";

interface ComposerProps {
  // Returns whether the send was accepted. May be async (the first send in a
  // draft creates the thread first). A false result restores the input text.
  onSend: (message: string) => boolean | Promise<boolean>;
  onCancel: () => void;
  streaming: boolean;
  disabled: boolean;
}

export function Composer(props: ComposerProps): React.ReactElement {
  const { onSend, onCancel, streaming, disabled } = props;
  const [value, setValue] = useState("");

  const submit = (): void => {
    const trimmed = value.trim();
    if (trimmed === "" || streaming || disabled) return;
    // Clear optimistically for a responsive feel; restore if the send is
    // rejected (e.g. the lazy thread-create failed) so the message isn't lost.
    setValue("");
    void Promise.resolve(onSend(trimmed)).then((accepted) => {
      if (!accepted) setValue((current) => (current === "" ? trimmed : current));
    });
  };

  return (
    <form
      className="composer"
      data-testid="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        className="composer__input"
        data-testid="composer-input"
        value={value}
        disabled={disabled}
        placeholder={
          disabled
            ? "Start a new conversation to begin"
            : "Ask Aria anything about your wedding…"
        }
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          // Enter sends; Shift+Enter inserts a newline.
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      {streaming ? (
        <button
          type="button"
          onClick={onCancel}
          data-testid="cancel-button"
          className="composer__cancel"
        >
          Cancel
        </button>
      ) : (
        <button
          type="submit"
          data-testid="send-button"
          disabled={disabled || value.trim() === ""}
          className="composer__send"
        >
          Send
        </button>
      )}
    </form>
  );
}
