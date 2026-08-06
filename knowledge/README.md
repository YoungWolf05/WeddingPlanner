# Curated wedding-planning knowledge corpus

Small, curated, benign knowledge base authored **for this repository** as the
ingestion input for Phase 7 retrieval and as the basis for the retrieval-only
evaluation set (`evals/retrieval.jsonl`).

## Provenance and licensing

- **Authored for this repo.** Every document under `corpus/` is original,
  factual-ish general wedding-planning guidance written specifically for this
  project. It is NOT scraped, copied, or excerpted from any third-party source.
- **License.** Distributed under the repository's MIT license, same as the code.
- **Benign + PII-free.** The content contains **no real names, emails, phone
  numbers, postal addresses, secrets, or credentials**. Any figures (budget
  percentages, timelines) are illustrative general guidance, not personal data.

## Identity: `source_uri` is the stable key

Each document is ingested with a **`source_uri`** equal to its repo-relative
path, e.g. `knowledge/corpus/budget-basics.md`. The knowledge store derives the
app-owned `document_id` deterministically as
`sha256(normalizeSourceUri(source_uri))` (see
`src/core/knowledge-store.ts` `computeDocumentId`), so:

- the `source_uri` is the **stable, human-readable identity** of a document, and
- the retrieval-eval dataset references relevance by `source_uri`
  (`relevantSourceUris`) rather than by content or by a chunk id, so it stays
  stable when chunking parameters are tuned.

Do not rename a corpus file without updating any `evals/retrieval.jsonl` entries
that reference it — the path IS the identity.

## Categories

The corpus loosely spans the evaluation categories (budget, timeline/checklist,
venue/theme, guest logistics, cultural/ceremony, vendor selection):

| File | Category |
| --- | --- |
| `corpus/budget-basics.md` | budget |
| `corpus/budget-saving-tips.md` | budget |
| `corpus/timeline-12-month.md` | timeline |
| `corpus/day-of-timeline.md` | timeline |
| `corpus/venue-selection.md` | venue/theme |
| `corpus/theme-and-decor.md` | venue/theme |
| `corpus/guest-list-and-rsvps.md` | guest logistics |
| `corpus/seating-and-accessibility.md` | guest logistics |
| `corpus/ceremony-traditions.md` | cultural/ceremony |
| `corpus/vendor-selection.md` | vendor selection |
| `corpus/catering-and-menu.md` | vendor selection |
| `corpus/photography-and-video.md` | vendor selection |

Each document is intentionally short (a few paragraphs) so ingestion and the
retrieval eval run quickly.
