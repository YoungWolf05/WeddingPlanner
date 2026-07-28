# Capability matrices

Dated LiteLLM capability matrices produced by the opt-in `npm run test:capabilities`
command. Each file is an **immutable historical snapshot** of one live probe run
against the proxy: it records, per chat alias, whether invoke, streaming, abort,
usage metadata, tool calling, and structured output are `Supported`,
`Unsupported`, `Degraded`, `Error`, or `N/A`, plus a separate embeddings
assessment. Secrets are redacted (host-only base URL, masked key).

The pure rendering/classification logic lives in
[`src/core/capabilities.ts`](../../src/core/capabilities.ts) and the live probe
in [`src/probe-capabilities.ts`](../../src/probe-capabilities.ts).

## Latest matrix

- [2026-07-28](2026-07-28.md) — `claude-opus-4-8` and `claude-sonnet-4-6` fully
  `Supported`; embeddings (`gemini-embedding-001`) `Supported`, 768 dims.

Older dated files are retained unchanged as historical snapshots; do not edit
their recorded results. For example, [2026-07-27](2026-07-27.md) predates the
`gpt-5.1-chat` alias removal and the `claude-opus-4-8` structured-output re-probe,
so it still records those earlier results. To refresh, run
`npm run test:capabilities`, which writes a new dated file.
