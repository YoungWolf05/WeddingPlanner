// Phase 9 (9c): render agent tool progress from the typed `tool` events.
//
// EXIT CRITERION 3. Each entry comes ONLY from an SseToolEvent (discriminated
// call/result). `args` (on a call) and `content` (on a result) were ALREADY
// redacted by the backend before the wire; the client only displays them.
// A "result" with status "error" is rendered DISTINCTLY (this also covers the
// backend ToolNode refusing an unknown/unpermitted tool).

import type { SseToolEvent } from "../lib/sse-contract.js";

interface ToolProgressProps {
  toolEvents: SseToolEvent[];
}

export function ToolProgress(
  props: ToolProgressProps
): React.ReactElement | null {
  const { toolEvents } = props;
  if (toolEvents.length === 0) return null;

  return (
    <div className="tools" data-testid="tool-progress">
      <div className="tools__header">Tool activity</div>
      <ul className="tools__list">
        {toolEvents.map((evt, idx) => (
          <li
            // toolCallId groups a call/result pair; index disambiguates the two
            // phases (and any repeat) so keys stay stable + unique.
            key={`${evt.toolCallId}:${evt.phase}:${idx}`}
            className={toolItemClass(evt)}
            data-testid={`tool-${evt.phase}`}
          >
            <ToolEntry evt={evt} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function toolItemClass(evt: SseToolEvent): string {
  if (evt.phase === "result" && evt.status === "error") {
    return "tools__item tools__item--error";
  }
  return `tools__item tools__item--${evt.phase}`;
}

function ToolEntry(props: { evt: SseToolEvent }): React.ReactElement {
  const { evt } = props;
  if (evt.phase === "call") {
    return (
      <>
        <span className="tools__phase">calling</span>{" "}
        <code className="tools__name">{evt.name}</code>{" "}
        <span className="tools__args" data-testid="tool-args">
          {JSON.stringify(evt.args)}
        </span>
      </>
    );
  }
  // result
  const statusLabel = evt.status === "error" ? "error" : "result";
  return (
    <>
      <span className="tools__phase">{statusLabel}</span>{" "}
      <code className="tools__name">{evt.name}</code>
      {evt.content !== undefined ? (
        <>
          {" "}
          <span className="tools__content" data-testid="tool-content">
            {evt.content}
          </span>
        </>
      ) : null}
    </>
  );
}
