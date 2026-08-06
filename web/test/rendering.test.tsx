import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Citations } from "../src/components/Citations.js";
import { ToolProgress } from "../src/components/ToolProgress.js";
import { Artifacts } from "../src/components/Artifacts.js";
import {
  GROUNDED_ANSWER_ARTIFACT_KIND,
  type SseArtifactEvent,
  type SseCitation,
  type SseToolEvent,
} from "../src/lib/sse-contract.js";

// EXIT CRITERION 3: citations / tool progress / artifacts / evidenceStatus
// render ONLY from typed trusted events. These tests assert the presentational
// components project the typed payloads faithfully and distinguish the
// insufficient-evidence + tool-error states.

const citation: SseCitation = {
  marker: 2,
  chunkId: "chunk-x",
  documentId: "doc-x",
  sourceUri: "knowledge/corpus/venues.md",
  chunkIndex: 3,
  score: 0.912,
  contentHash: "hash-x",
};

describe("Citations", () => {
  it("renders trusted citation fields for a supported turn", () => {
    render(<Citations citations={[citation]} evidenceStatus="supported" />);
    expect(screen.getByTestId("citations")).toBeTruthy();
    expect(screen.getByTestId("citation-marker").textContent).toContain("[2]");
    expect(screen.getByTestId("citation-source").textContent).toContain(
      "knowledge/corpus/venues.md"
    );
  });

  it("renders the insufficient-evidence state distinctly with no citations", () => {
    render(<Citations citations={[]} evidenceStatus="insufficient" />);
    expect(screen.getByTestId("evidence-insufficient")).toBeTruthy();
    expect(screen.queryByTestId("citation-item")).toBeNull();
  });

  it("renders nothing before a citation event arrives", () => {
    const { container } = render(
      <Citations citations={[]} evidenceStatus={null} />
    );
    expect(container.firstChild).toBeNull();
  });
});

describe("ToolProgress", () => {
  it("renders call args and a distinct error result", () => {
    const events: SseToolEvent[] = [
      {
        phase: "call",
        name: "split_budget",
        toolCallId: "c1",
        args: { total: 30000 },
      },
      {
        phase: "result",
        name: "split_budget",
        toolCallId: "c1",
        status: "error",
        content: "invalid input",
      },
    ];
    render(<ToolProgress toolEvents={events} />);
    expect(screen.getByTestId("tool-args").textContent).toContain("30000");
    const errorItem = screen.getByTestId("tool-result");
    expect(errorItem.className).toContain("tools__item--error");
  });
});

describe("Artifacts", () => {
  it("renders the grounded-answer envelope with its evidenceStatus", () => {
    const artifact: SseArtifactEvent = {
      kind: GROUNDED_ANSWER_ARTIFACT_KIND,
      data: { answer: "Book early.", evidenceStatus: "supported" },
    };
    render(<Artifacts artifacts={[artifact]} />);
    expect(screen.getByTestId("artifact-grounded")).toBeTruthy();
    expect(
      screen.getByTestId("artifact-evidence-status").textContent
    ).toBe("supported");
  });

  it("renders an unknown artifact kind generically as JSON", () => {
    const artifact: SseArtifactEvent = {
      kind: "budget_plan",
      data: { total: 1000 },
    };
    render(<Artifacts artifacts={[artifact]} />);
    const generic = screen.getByTestId("artifact-generic");
    expect(generic.textContent).toContain("budget_plan");
    expect(generic.textContent).toContain("1000");
  });
});
