# ADR 0001 — Defer agentic retrieval; keep deterministic two-step RAG

- **Status:** Accepted
- **Date:** 2026-08-06
- **Phase:** 8 (Grounded Answers and Trusted Citations), increment 8d
- **Satisfies:** Phase 8 exit criterion 4 — "Any move to agentic retrieval is
  supported by comparative measurements and a separate ADR."

## Context

Phase 8 delivers a **deterministic, two-step retrieve-then-generate** RAG
pipeline (`src/core/rag.ts` `answerQuestion`): embed the query, retrieve the
top-k chunks with trusted app-owned metadata (`src/core/retriever.ts`), build a
numbered/delimited context block, generate a structured `GroundedAnswer`, resolve
each app-assigned citation marker back to a trusted, authorized store ID
(`src/core/citations.ts`), and reconcile a supported-vs-insufficient verdict
(`src/core/evidence.ts`). This shape was the roadmap's stated initial design and
was user-approved at Phase 8 activation.

An alternative is **agentic / iterative retrieval**: an LLM-driven loop that
issues multiple, self-directed retrieval queries (query rewriting, follow-up
searches, tool-calling retrieval) before answering. It can improve recall on
multi-hop or under-specified questions, but it adds token cost, latency, and
control-flow complexity, and it widens the prompt-injection surface (each
model-directed retrieval step is another place untrusted context can influence
behavior).

The Phase 8 (8d) grounded-answer evaluation (`src/core/rag-eval.ts`,
`evals/rag.jsonl`, live runner `src/run-rag-eval.ts` → `docs/rag-eval/<date>.md`)
measures the two-step pipeline on groundedness, citation precision/recall,
prompt-injection / malicious-source resistance, and missing-evidence handling
against a PROPOSED baseline (pending user approval at closeout).

## Decision

**Defer agentic / iterative retrieval.** Phase 8 ships the deterministic two-step
pipeline only. No agentic retrieval is implemented in this phase.

Adopting agentic retrieval later is **gated**: it requires (1) **comparative
measurements** demonstrating a clear advantage over the two-step baseline on the
same 8d eval axes (groundedness, citation precision/recall, injection resistance,
missing-evidence accuracy) at acceptable cost/latency, and (2) a **separate,
future ADR** recording that decision and its evidence. This ADR records only the
DEFERRAL; it does not pre-approve adoption.

## Consequences / Rationale

- The two-step design meets the Phase 8 grounded-answer + trusted-citation goals
  at acceptable quality, as measured by the 8d eval (see
  `docs/rag-eval/<date>.md` once the live run is recorded; the PROPOSED baseline
  and the 8c `DEFAULT_MIN_EVIDENCE_SCORE` are ratified with that evidence at
  closeout).
- Determinism keeps the pipeline auditable and offline-testable end-to-end (a
  mocked model + fake embedder cover it), and keeps the prompt-injection surface
  small: retrieved content is treated as untrusted DATA behind a single guardrail,
  with no model-directed retrieval loop.
- The cost of deferral is lower recall on multi-hop / under-specified questions
  that iterative retrieval could improve. This is an accepted trade-off for this
  phase; the curated corpus and eval questions are single-source-answerable, so
  the ceiling lost is small.
- If a future need (multi-hop questions, larger/noisier corpus) justifies
  agentic retrieval, the gate above ensures the change is evidence-driven: a
  measured comparison against this two-step baseline plus its own ADR, rather than
  an unmeasured architecture change.

## References

- Pipeline: `src/core/rag.ts`, `src/core/retriever.ts`, `src/core/citations.ts`,
  `src/core/evidence.ts`, `src/core/prompts.ts`.
- Eval: `src/core/rag-eval.ts`, `evals/rag.jsonl`, `src/run-rag-eval.ts`,
  `docs/rag-eval/<date>.md`.
- Roadmap: `docs/roadmap.md` — Phase 8 design decisions + exit criterion 4.
