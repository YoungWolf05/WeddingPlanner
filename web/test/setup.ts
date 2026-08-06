// Phase 9 (9c) web test setup: wire @testing-library matchers cleanup.
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
});
