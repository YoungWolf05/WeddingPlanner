import { describe, it, expect } from "vitest";
import {
  buildChatModelParams,
  createChatModel,
  type ModelOptions,
} from "../src/core/model.js";

// Phase 4 (increment 4e.0) — the "omit temperature" factory enabler.
//
// DETERMINISTIC, fully OFFLINE. The heart of this step is the pure
// buildChatModelParams() helper: it decides whether a `temperature` key is sent
// to ChatOpenAI at all. Testing the helper directly (approach b) lets us assert
// the THREE cases unambiguously — including that the key is ABSENT when omission
// is requested — without depending on ChatOpenAI's internal field naming or its
// default-coalescing behavior, and without constructing any provider client or
// making a network call.
//
// A small integration check against createChatModel() confirms the resolved
// temperature is reflected on the real instance and, crucially, that the omit
// mode leaves the constructed model with no explicit temperature.

// Fixed, obviously-fake deps so the helper assertions are independent of ambient
// config. The helper is pure, so these never leave the test process.
const DEPS = {
  apiKey: "sk-test-dummy-not-a-real-key",
  baseURL: "http://localhost:0/test-fake-litellm",
  model: "claude-sonnet-4-6",
  streaming: false,
} as const;

describe("Phase 4 (4e.0) — buildChatModelParams temperature handling", () => {
  it("defaults temperature to 0.7 when the option is omitted (undefined)", () => {
    const params = buildChatModelParams({}, DEPS);
    expect(params).toHaveProperty("temperature", 0.7);
  });

  it("uses the provided number verbatim (e.g. 0.2)", () => {
    const params = buildChatModelParams({ temperature: 0.2 }, DEPS);
    expect(params).toHaveProperty("temperature", 0.2);
  });

  it("preserves temperature: 0 (does not fall back to the 0.7 default)", () => {
    // Guards the `?? 0.7` fallback against the classic falsy-zero bug: 0 is a
    // valid explicit temperature and must be kept.
    const params = buildChatModelParams({ temperature: 0 }, DEPS);
    expect(params).toHaveProperty("temperature", 0);
  });

  it("OMITS the temperature key entirely when temperature is null", () => {
    const params = buildChatModelParams({ temperature: null }, DEPS);
    // The key must be ABSENT, not present-and-undefined.
    expect("temperature" in params).toBe(false);
    expect(Object.keys(params)).not.toContain("temperature");
  });

  it("sets model/apiKey/baseURL/streaming correctly regardless of temperature", () => {
    for (const t of [undefined, 0.2, null] as ModelOptions["temperature"][]) {
      const params = buildChatModelParams(
        { temperature: t },
        { ...DEPS, streaming: true }
      );
      expect(params.model).toBe(DEPS.model);
      expect(params.apiKey).toBe(DEPS.apiKey);
      expect(params.configuration).toEqual({ baseURL: DEPS.baseURL });
      expect(params.streaming).toBe(true);
    }
  });

  it("is pure: returns a plain object and constructs no client", () => {
    const params = buildChatModelParams({ temperature: null }, DEPS);
    // A plain data object — no methods a provider client would expose.
    expect(typeof (params as { invoke?: unknown }).invoke).toBe("undefined");
    expect(typeof (params as { bindTools?: unknown }).bindTools).toBe(
      "undefined"
    );
    expect(params.constructor).toBe(Object);
  });
});

describe("Phase 4 (4e.0) — createChatModel reflects the resolved temperature", () => {
  it("defaults the instance temperature to 0.7 when omitted", () => {
    const model = createChatModel({ model: "claude-sonnet-4-6" });
    expect(model.temperature).toBe(0.7);
  });

  it("reflects an explicit numeric temperature (0.2)", () => {
    const model = createChatModel({
      model: "claude-sonnet-4-6",
      temperature: 0.2,
    });
    expect(model.temperature).toBe(0.2);
  });

  it("carries NO explicit temperature when temperature: null (omit mode)", () => {
    const model = createChatModel({
      model: "claude-opus-4-8",
      temperature: null,
    });
    // With the key omitted, ChatOpenAI keeps its own class default of
    // `undefined` for temperature, so nothing is sent to the provider.
    expect(model.temperature).toBeUndefined();
  });
});
