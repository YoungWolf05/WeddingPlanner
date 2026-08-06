import { describe, expect, it } from "vitest";
import {
  GROUNDED_ANSWER_ARTIFACT_KIND,
  isGroundedAnswerArtifact,
  SSE_PROTOCOL_VERSION,
} from "../src/lib/sse-contract.js";

describe("sse-contract", () => {
  it("pins the protocol version to 2 (must match src/core/sse.ts)", () => {
    expect(SSE_PROTOCOL_VERSION).toBe(2);
  });

  it("narrows a valid grounded-answer artifact", () => {
    expect(
      isGroundedAnswerArtifact({
        kind: GROUNDED_ANSWER_ARTIFACT_KIND,
        data: { answer: "hi", evidenceStatus: "supported" },
      })
    ).toBe(true);
  });

  it("rejects a wrong kind", () => {
    expect(
      isGroundedAnswerArtifact({
        kind: "something_else",
        data: { answer: "hi", evidenceStatus: "supported" },
      })
    ).toBe(false);
  });

  it("rejects malformed data", () => {
    expect(
      isGroundedAnswerArtifact({
        kind: GROUNDED_ANSWER_ARTIFACT_KIND,
        data: { answer: 5, evidenceStatus: "supported" },
      })
    ).toBe(false);
    expect(
      isGroundedAnswerArtifact({
        kind: GROUNDED_ANSWER_ARTIFACT_KIND,
        data: { answer: "x", evidenceStatus: "maybe" },
      })
    ).toBe(false);
    expect(
      isGroundedAnswerArtifact({
        kind: GROUNDED_ANSWER_ARTIFACT_KIND,
        data: null,
      })
    ).toBe(false);
  });
});
