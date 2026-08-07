// Phase 9 (9c) web test setup: wire @testing-library matchers cleanup.
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
});

// jsdom does not implement Element.prototype.scrollTo, which the Transcript's
// auto-scroll layout effect calls once the transcript actually contains messages
// (e.g. hydrated history). Provide a no-op so rendering a populated transcript
// under jsdom does not throw. This mirrors the component's own graceful jsdom
// degradation (its scroll-metric reads are all 0 there).
if (typeof Element !== "undefined" && typeof Element.prototype.scrollTo !== "function") {
  Element.prototype.scrollTo = function scrollTo(): void {
    /* no-op in jsdom */
  };
}
