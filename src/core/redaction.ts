import { config } from "../config.js";

// Shared, PURE, offline redaction helpers.
//
// Phase 4 (increment 4d) extracted these from src/probe-capabilities.ts so the
// capability probe AND the tracing layer scrub secrets/PII through ONE
// implementation. Redaction is ALWAYS ON and not disableable: any string that
// might reach a log, trace file, console, or evidence artifact must pass through
// here first.
//
// This module is pure and I/O-free (it only reads the loaded config values), so
// it is fully unit-testable offline.

// Default cap for a single redacted string (error reason or captured content).
// Keeps trace lines bounded and prevents a runaway provider error from bloating
// the JSONL sink. Callers may override per call site.
export const DEFAULT_MAX_LENGTH = 500;

// Placeholders emitted in place of scrubbed secrets/PII. Fixed, non-secret
// markers so a human can see that redaction happened without leaking any byte.
export const KEY_PLACEHOLDER = "[redacted-key]";
export const URL_PLACEHOLDER = "[redacted-url]";
export const EMAIL_PLACEHOLDER = "[redacted-email]";
export const PHONE_PLACEHOLDER = "[redacted-phone]";

// Email addresses: conservative RFC-ish local@domain.tld shape. Matches common
// addresses without trying to be a full RFC 5322 validator (which would be
// error-prone and could over-match). Global + case-insensitive.
const EMAIL_PATTERN =
  /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;

// Phone numbers: sequences of 7+ digits with optional country code, separators
// (space, dash, dot), and parentheses. Deliberately requires enough digits that
// short numeric tokens (e.g. small counts, years) are not scrubbed. Anchored on
// a digit-group shape to avoid eating arbitrary long integers embedded in words.
const PHONE_PATTERN =
  /(?<!\w)\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?(?:[\s.-]?\d{2,4}){2,}(?!\w)/g;

// Scrub the configured LiteLLM apiKey and baseURL substrings anywhere they
// appear in `text`. Uses split/join (not regex) so special characters in the
// secret cannot corrupt the pattern. This mirrors the original probe behavior
// exactly and is the security-critical core: even in metadata-only tracing, the
// error reason passes through here.
export function scrubSecrets(text: string): string {
  let out = text;
  if (config.apiKey) out = out.split(config.apiKey).join(KEY_PLACEHOLDER);
  if (config.baseURL) out = out.split(config.baseURL).join(URL_PLACEHOLDER);
  return out;
}

// Scrub basic PII patterns (emails, phone numbers) from `text`. Applied to any
// captured content or error text so opt-in content capture (and error reasons)
// cannot leak personal data.
export function scrubPii(text: string): string {
  return text
    .replace(EMAIL_PATTERN, EMAIL_PLACEHOLDER)
    .replace(PHONE_PATTERN, PHONE_PLACEHOLDER);
}

// Full redaction for an arbitrary string destined for a trace/log/evidence
// artifact: collapse whitespace to a single line, scrub secrets, scrub PII, and
// cap the length. Order matters — secrets first, then PII, then cap — so a
// truncation can never split a secret into a leakable prefix.
export function redactText(
  text: string,
  maxLength: number = DEFAULT_MAX_LENGTH
): string {
  let out = text.replace(/\s+/g, " ").trim();
  out = scrubSecrets(out);
  out = scrubPii(out);
  if (out.length > maxLength) {
    out = out.slice(0, maxLength) + "…";
  }
  return out;
}

// Reduce any thrown value to a concise, fully-redacted single-line reason.
// Reused by the capability probe (previously its private redactError) and the
// tracing layer's error events. Never throws.
export function redactError(
  err: unknown,
  maxLength: number = DEFAULT_MAX_LENGTH
): string {
  const message =
    err instanceof Error
      ? err.message
      : typeof err === "string"
        ? err
        : String(err);
  return redactText(message, maxLength);
}
