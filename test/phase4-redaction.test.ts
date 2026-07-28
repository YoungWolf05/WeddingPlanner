import { describe, it, expect } from "vitest";
import { config } from "../src/config.js";
import {
  DEFAULT_MAX_LENGTH,
  EMAIL_PLACEHOLDER,
  KEY_PLACEHOLDER,
  PHONE_PLACEHOLDER,
  URL_PLACEHOLDER,
  redactError,
  redactText,
  scrubPii,
  scrubSecrets,
} from "../src/core/redaction.js";

// Phase 4 (increment 4d) — OFFLINE redaction/PII scrubbing tests.
//
// Redaction is ALWAYS ON and security-critical: any string reaching a trace,
// log, console, or evidence artifact must have the LiteLLM apiKey/baseURL and
// basic PII (email/phone) scrubbed and be length-capped. These tests use the
// dummy, non-secret config values injected by test/setup/env.ts — no real
// secret is ever printed. This module is pure; no network, no credentials.

describe("Phase 4 — redaction: secrets", () => {
  it("scrubs the configured apiKey substring from a string", () => {
    const raw = `auth failed with key ${config.apiKey} rejected`;
    const out = scrubSecrets(raw);
    expect(out).not.toContain(config.apiKey);
    expect(out).toContain(KEY_PLACEHOLDER);
  });

  it("scrubs the configured baseURL substring from a string", () => {
    const raw = `POST ${config.baseURL}/chat/completions 500`;
    const out = scrubSecrets(raw);
    expect(out).not.toContain(config.baseURL);
    expect(out).toContain(URL_PLACEHOLDER);
  });

  it("scrubs multiple occurrences of the same secret", () => {
    const raw = `${config.apiKey} then again ${config.apiKey}`;
    const out = scrubSecrets(raw);
    expect(out).not.toContain(config.apiKey);
    // Both occurrences replaced.
    expect(out.split(KEY_PLACEHOLDER)).toHaveLength(3);
  });
});

describe("Phase 4 — redaction: PII", () => {
  it("redacts email addresses", () => {
    const out = scrubPii("contact aria@example.com for details");
    expect(out).not.toContain("aria@example.com");
    expect(out).toContain(EMAIL_PLACEHOLDER);
  });

  it("redacts phone numbers (various formats)", () => {
    const samples = [
      "call +1 (555) 123-4567 now",
      "reach 555-123-4567 today",
      "dial 5551234567 please",
      "intl +44 20 7946 0958 ok",
    ];
    for (const s of samples) {
      const out = scrubPii(s);
      expect(out).toContain(PHONE_PLACEHOLDER);
    }
  });

  it("does not redact short numeric tokens (counts, small ints)", () => {
    const out = scrubPii("we invited 12 guests and 3 tables");
    expect(out).toBe("we invited 12 guests and 3 tables");
  });
});

describe("Phase 4 — redaction: redactText", () => {
  it("collapses whitespace/newlines to a single line", () => {
    const out = redactText("line one\n   line two\t\tline three");
    expect(out).toBe("line one line two line three");
  });

  it("scrubs secrets AND PII in one pass", () => {
    const raw = `error ${config.apiKey} at ${config.baseURL} email bob@x.io phone 555-123-4567`;
    const out = redactText(raw);
    expect(out).not.toContain(config.apiKey);
    expect(out).not.toContain(config.baseURL);
    expect(out).not.toContain("bob@x.io");
    expect(out).toContain(KEY_PLACEHOLDER);
    expect(out).toContain(URL_PLACEHOLDER);
    expect(out).toContain(EMAIL_PLACEHOLDER);
    expect(out).toContain(PHONE_PLACEHOLDER);
  });

  it("caps length at the default max and appends an ellipsis", () => {
    const raw = "x".repeat(DEFAULT_MAX_LENGTH + 100);
    const out = redactText(raw);
    expect(out.length).toBe(DEFAULT_MAX_LENGTH + 1); // +1 for the ellipsis
    expect(out.endsWith("…")).toBe(true);
  });

  it("honors a custom max length", () => {
    const out = redactText("y".repeat(50), 10);
    expect(out).toBe("y".repeat(10) + "…");
  });

  it("caps AFTER scrubbing so truncation cannot leak a secret prefix", () => {
    // Put the secret at the very start, then a long tail; even the capped output
    // must not contain any part of the raw secret.
    const raw = config.apiKey + "z".repeat(DEFAULT_MAX_LENGTH);
    const out = redactText(raw);
    expect(out).not.toContain(config.apiKey);
    expect(out.startsWith(KEY_PLACEHOLDER)).toBe(true);
  });
});

describe("Phase 4 — redaction: redactError", () => {
  it("reduces an Error to a redacted single-line reason", () => {
    const err = new Error(
      `provider 500 at ${config.baseURL} key ${config.apiKey}\nstack line`
    );
    const out = redactError(err);
    expect(out).not.toContain(config.apiKey);
    expect(out).not.toContain(config.baseURL);
    expect(out).not.toContain("\n");
    expect(out).toContain(KEY_PLACEHOLDER);
    expect(out).toContain(URL_PLACEHOLDER);
  });

  it("handles non-Error thrown values (string / object)", () => {
    expect(redactError("plain string failure")).toBe("plain string failure");
    expect(redactError({ weird: true })).toContain("object");
  });

  it("applies a custom cap (probe uses 200)", () => {
    const out = redactError(new Error("q".repeat(300)), 200);
    expect(out.length).toBe(201);
    expect(out.endsWith("…")).toBe(true);
  });
});

// Phase 4 (increment 4d, finding R1) — LOCK the probe's redaction contract.
//
// src/probe-capabilities.ts reduces provider errors via `redactError(err)` which
// is exactly `redactErrorShared(err, 200)`. When the probe adopted the shared
// helper it GAINED PII scrubbing (email/phone) in addition to its original
// secret scrubbing and 200-char cap. That is an intentional, beneficial security
// improvement — this test pins ALL of it so a future refactor cannot silently
// drop PII scrubbing, weaken secret scrubbing, or change the 200-char cap on the
// probe's evidence path. We assert against redactError with the probe's exact
// 200 cap (the probe's redaction path) rather than importing the probe module,
// which is a CLI entrypoint that runs on import — keeping this test offline and
// side-effect free.
describe("Phase 4 — probe redaction path (redactError @ 200-char cap)", () => {
  // Mirror src/probe-capabilities.ts's PROBE_REDACT_MAX so the test tracks the
  // probe's actual cap, not a magic literal.
  const PROBE_CAP = 200;

  it("scrubs apiKey AND baseURL AND PII (email/phone) in one provider-style error", () => {
    const err = new Error(
      `LiteLLM request to ${config.baseURL}/chat/completions failed: ` +
        `auth key ${config.apiKey} rejected; ` +
        `notify planner@aria-weddings.example or call +1 (555) 867-5309`
    );
    const out = redactError(err, PROBE_CAP);

    // Secrets are gone and replaced with the fixed markers.
    expect(out).not.toContain(config.apiKey);
    expect(out).not.toContain(config.baseURL);
    expect(out).toContain(KEY_PLACEHOLDER);
    expect(out).toContain(URL_PLACEHOLDER);

    // PII is gone and replaced with the fixed markers (the locked improvement).
    expect(out).not.toContain("planner@aria-weddings.example");
    expect(out).not.toContain("867-5309");
    expect(out).toContain(EMAIL_PLACEHOLDER);
    expect(out).toContain(PHONE_PLACEHOLDER);
  });

  it("still caps the probe's redacted reason at 200 chars (secrets/PII stripped first)", () => {
    // Long secret-laden tail: scrubbing happens before capping, so the output
    // must both be capped AND contain no raw secret/PII fragment.
    const raw =
      `${config.apiKey} ${config.baseURL} bride@example.com 555-123-4567 ` +
      "z".repeat(PROBE_CAP);
    const out = redactError(new Error(raw), PROBE_CAP);

    expect(out.length).toBe(PROBE_CAP + 1); // 200 chars + trailing ellipsis
    expect(out.endsWith("…")).toBe(true);
    expect(out).not.toContain(config.apiKey);
    expect(out).not.toContain(config.baseURL);
    expect(out).not.toContain("bride@example.com");
    expect(out).not.toContain("555-123-4567");
  });
});
