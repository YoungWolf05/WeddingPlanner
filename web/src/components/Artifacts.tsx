// Phase 9 (9c): render structured artifacts from the typed `artifact` events.
//
// EXIT CRITERION 3. Each artifact comes ONLY from an SseArtifactEvent envelope
// ({ kind, data }). We switch on the trusted `kind` discriminator to pick a
// renderer. The grounded-answer envelope (GROUNDED_ANSWER_ARTIFACT_KIND) is
// narrowed via the shared isGroundedAnswerArtifact predicate; any other/unknown
// kind renders generically as kind + pretty-printed JSON (never executed, never
// treated as HTML). All free-text inside `data` was already redacted server-side.

import {
  isGroundedAnswerArtifact,
  type SseArtifactEvent,
} from "../lib/sse-contract.js";

interface ArtifactsProps {
  artifacts: SseArtifactEvent[];
}

export function Artifacts(props: ArtifactsProps): React.ReactElement | null {
  const { artifacts } = props;
  if (artifacts.length === 0) return null;

  return (
    <div className="artifacts" data-testid="artifacts">
      <div className="artifacts__header">Structured output</div>
      {artifacts.map((artifact, idx) => (
        <ArtifactCard key={`${artifact.kind}:${idx}`} artifact={artifact} />
      ))}
    </div>
  );
}

function ArtifactCard(props: {
  artifact: SseArtifactEvent;
}): React.ReactElement {
  const { artifact } = props;

  if (isGroundedAnswerArtifact(artifact)) {
    return (
      <div
        className="artifact artifact--grounded"
        data-testid="artifact-grounded"
      >
        <div className="artifact__kind">grounded answer</div>
        <div className="artifact__evidence">
          evidence:{" "}
          <span data-testid="artifact-evidence-status">
            {artifact.data.evidenceStatus}
          </span>
        </div>
        <p className="artifact__answer">{artifact.data.answer}</p>
      </div>
    );
  }

  // Unknown kind: render self-describing envelope generically (safe text only).
  return (
    <div className="artifact artifact--generic" data-testid="artifact-generic">
      <div className="artifact__kind">{artifact.kind}</div>
      <pre className="artifact__json">
        {JSON.stringify(artifact.data, null, 2)}
      </pre>
    </div>
  );
}
