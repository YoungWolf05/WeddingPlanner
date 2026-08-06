// Phase 9 (9c): render the conversation transcript — user messages and assistant
// turns (streaming text + citations + tool progress + artifacts + status).

import type { ChatMessage } from "../lib/conversation.js";
import { Artifacts } from "./Artifacts.js";
import { Citations } from "./Citations.js";
import { ToolProgress } from "./ToolProgress.js";

interface TranscriptProps {
  messages: ChatMessage[];
  // Retry the last failed/canceled assistant turn (re-issue its source message).
  onRetry: (sourceMessage: string) => void;
  // Whether a stream is currently in flight (disables retry buttons).
  streaming: boolean;
}

export function Transcript(props: TranscriptProps): React.ReactElement {
  const { messages, onRetry, streaming } = props;
  return (
    <div className="transcript" data-testid="transcript">
      {messages.length === 0 ? (
        <div className="transcript__empty">
          Start the conversation by sending a message.
        </div>
      ) : (
        messages.map((message) =>
          message.role === "user" ? (
            <div
              key={message.id}
              className="message message--user"
              data-testid="message-user"
            >
              {message.text}
            </div>
          ) : (
            <div
              key={message.id}
              className="message message--assistant"
              data-testid="message-assistant"
              data-status={message.status}
            >
              <div className="message__text">
                {message.text}
                {message.status === "streaming" ? (
                  <span className="message__cursor" aria-hidden="true">
                    ▋
                  </span>
                ) : null}
              </div>

              <ToolProgress toolEvents={message.toolEvents} />
              <Citations
                citations={message.citations}
                evidenceStatus={message.evidenceStatus}
              />
              <Artifacts artifacts={message.artifacts} />

              {message.status === "error" ? (
                <div className="message__error" data-testid="turn-error">
                  {message.error ?? "The turn failed."}{" "}
                  <button
                    type="button"
                    disabled={streaming}
                    onClick={() => onRetry(message.sourceMessage)}
                    data-testid="retry-button"
                  >
                    Retry
                  </button>
                </div>
              ) : null}

              {message.status === "canceled" ? (
                <div className="message__canceled" data-testid="turn-canceled">
                  Canceled.{" "}
                  <button
                    type="button"
                    disabled={streaming}
                    onClick={() => onRetry(message.sourceMessage)}
                    data-testid="retry-button"
                  >
                    Retry
                  </button>
                </div>
              ) : null}
            </div>
          )
        )
      )}
    </div>
  );
}
