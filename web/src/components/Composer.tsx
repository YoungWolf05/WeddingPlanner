// Phase 9 (9c): the message composer with SEND and CANCEL.
//
// SEND posts the message to the chat SSE endpoint (App wires the stream). CANCEL
// aborts the in-flight stream via App's AbortController. The input is disabled
// while no thread is selected.

import { useState } from "react";

interface ComposerProps {
  onSend: (message: string) => void;
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
    onSend(trimmed);
    setValue("");
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
          disabled ? "Select or create a conversation first" : "Type a message…"
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
