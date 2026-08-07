// Phase 9 (9c): render the conversation transcript — user messages and assistant
// turns (streaming text + citations + tool progress + artifacts + status).
//
// BUG 2 (auto-scroll): the transcript is the ONLY scroll region. It auto-scrolls
// to the newest content as messages are appended AND while tokens stream — but
// only when the user is already near the bottom, so reading history is not
// interrupted. Smooth scrolling is suppressed under prefers-reduced-motion.

import { useLayoutEffect, useRef } from "react";
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

// How close to the bottom (px) still counts as "at the bottom" for the
// auto-follow guard. Generous enough to survive a single streamed line.
const NEAR_BOTTOM_PX = 120;

// The accumulated streamed text of the last assistant turn (if any). Used as an
// effect dependency so auto-scroll follows tokens, not just whole messages.
function lastTurnText(messages: ChatMessage[]): string {
  const last = messages[messages.length - 1];
  return last !== undefined && last.role === "assistant" ? last.text : "";
}

export function Transcript(props: TranscriptProps): React.ReactElement {
  const { messages, onRetry, streaming } = props;

  const scrollRef = useRef<HTMLDivElement | null>(null);

  const streamedText = lastTurnText(messages);

  // Measure-then-scroll in a single layout effect (runs after the DOM is
  // committed but before paint). The layout read + scroll happen here — never in
  // the render body — so this is safe under StrictMode/concurrent rendering.
  //
  // Semantics preserved exactly: we auto-follow only when the user is near the
  // bottom (within NEAR_BOTTOM_PX), so a user who scrolled up to read history is
  // not yanked; and smooth scrolling is suppressed under prefers-reduced-motion.
  //
  // "Near the bottom" is evaluated against the pre-scroll position: the effect
  // observes the container's CURRENT scroll metrics before it performs any
  // scroll. Because this update only grows the transcript (an appended message or
  // additional streamed tokens), the newest content sits at/after the previous
  // bottom, so a user pinned to the bottom stays within NEAR_BOTTOM_PX and keeps
  // following, while a user scrolled up remains outside the threshold.
  useLayoutEffect(() => {
    const node = scrollRef.current;
    if (node === null) return;
    // Read layout to decide whether to follow. In jsdom these metrics are 0, so
    // the distance is 0 (<= NEAR_BOTTOM_PX) and the effect degrades safely.
    const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
    const nearBottom = distance <= NEAR_BOTTOM_PX;
    if (!nearBottom) return; // user is reading history — don't interrupt.
    const prefersReduced =
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    node.scrollTo({
      top: node.scrollHeight,
      behavior: prefersReduced ? "auto" : "smooth",
    });
    // Re-run when the message count changes or the streaming text grows.
  }, [messages.length, streamedText, streaming]);

  return (
    <div className="transcript" data-testid="transcript" ref={scrollRef}>
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
