// Phase 9 (9c): render the turn's TRUSTED citations and evidenceStatus.
//
// EXIT CRITERION 3. Everything here comes ONLY from the typed `citation` event
// (SseCitationEvent): the app-owned citation fields and the app-authoritative
// evidenceStatus. The component NEVER fabricates a citation and never derives
// citation identity from model/answer text — it renders the trusted array as-is.
// An "insufficient" turn is rendered DISTINCTLY (with no citations).

import type { EvidenceStatus, SseCitation } from "../lib/sse-contract.js";

interface CitationsProps {
  citations: SseCitation[];
  evidenceStatus: EvidenceStatus | null;
}

export function Citations(props: CitationsProps): React.ReactElement | null {
  const { citations, evidenceStatus } = props;

  // Nothing to render until a citation event has arrived.
  if (evidenceStatus === null) return null;

  if (evidenceStatus === "insufficient") {
    return (
      <div
        className="citations citations--insufficient"
        data-testid="evidence-insufficient"
      >
        <span className="citations__seal citations__seal--muted" aria-hidden="true" />
        <div className="citations__insufficient-body">
          <strong>Insufficient evidence.</strong> Aria couldn&rsquo;t find trusted
          sources to ground this answer, so no sources are shown.
        </div>
      </div>
    );
  }

  // supported
  return (
    <div className="citations citations--supported" data-testid="citations">
      <div className="citations__header" data-testid="evidence-supported">
        <span className="citations__eyebrow">Sources</span>
        <span className="citations__count">{citations.length}</span>
      </div>
      {citations.length === 0 ? (
        <div className="citations__empty">No citations for this answer.</div>
      ) : (
        <ol className="citations__list">
          {citations.map((c) => (
            <li
              key={c.chunkId}
              className="citations__item source-card"
              data-testid="citation-item"
            >
              <span
                className="citations__marker source-card__seal"
                data-testid="citation-marker"
              >
                [{c.marker}]
              </span>
              <span className="source-card__body">
                <span
                  className="citations__source source-card__source"
                  data-testid="citation-source"
                  title={c.documentId}
                >
                  {c.sourceUri ?? "(unknown source)"}
                </span>
                <span className="citations__meta source-card__meta">
                  chunk #{c.chunkIndex} · score {c.score.toFixed(3)}
                </span>
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
