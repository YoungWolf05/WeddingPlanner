# Embedding & dimension compatibility

Dated embedding compatibility reports produced by the opt-in `npm run test:embedding`
command (LIVE — makes real, credentialed LiteLLM calls; NOT part of `npm test`
or CI). Each file is an **immutable historical snapshot** of one live probe run
against the proxy, targeting Phase 7 exit criterion 4: "the embedding alias and
vector dimensions are verified through the proxy and covered by compatibility
checks".

For the configured embedding alias (`LITELLM_EMBED_MODEL`) the probe embeds a
benign, PII-free wedding-domain sentence and records:

- **State** — one of `Compatible`, `DimensionMismatch`, `Unverified`, or `Error`.
- **Expected dim** — the single source of truth `config.embedDim`
  (`LITELLM_EMBED_DIM`, default 768) — the dimension the knowledge store is built
  with.
- **Observed dim** — the vector dimension the alias actually produced.

`Compatible` means the alias produced a non-empty vector whose dimension EQUALS
the expected dimension. Secrets are redacted (host-only base URL, masked key; any
error reason passes through the shared redaction).

The pure classification/rendering logic lives in
[`src/core/embedding-compat.ts`](../../src/core/embedding-compat.ts) and the live
probe in [`src/probe-embedding.ts`](../../src/probe-embedding.ts). The pure logic
is unit-tested offline in `test/phase7-embedding-compat.test.ts`.

## Latest report

- [2026-08-06](2026-08-06.md) — `gemini-embedding-001` → `Compatible` (expected
  768, observed 768), verified live through the proxy.

The independent capability matrix
[`docs/capabilities/2026-07-28.md`](../capabilities/2026-07-28.md) also records
`gemini-embedding-001` as Supported at 768 dimensions.

To refresh, run `npm run test:embedding`, which writes a new dated file. Older
dated files are retained unchanged as historical snapshots; do not edit their
recorded results.
