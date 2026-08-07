// Phase (history) web unit tests: getThreadMessages — the conversation-history
// replay REST client fn. OFFLINE + deterministic: a FAKE fetch stands in for the
// backend (no network, no real server). Covers the happy path, defensive
// wire-shape validation (rejecting malformed roles/text/non-array bodies), the
// auth/error mapping (401/404), and that the request carries the bearer token +
// hits the /threads/:id/messages route (owner-scoped, GET-only, read-only).

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getThreadMessages,
  ThreadApiError,
  type HistoryMessage,
} from "../src/lib/threadsApi.js";

// A minimal Response-like value carrying only the fields requestJson reads.
interface ResponseLike {
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
}

function jsonResponse(status: number, body: unknown): ResponseLike {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

// Install a fake global fetch returning `response`, recording the URL + init.
function installFetch(response: ResponseLike): {
  calls: Array<{ url: string; init: RequestInit | undefined }>;
} {
  const calls: Array<{ url: string; init: RequestInit | undefined }> = [];
  vi.stubGlobal(
    "fetch",
    (async (url: string, init?: RequestInit): Promise<Response> => {
      calls.push({ url, init });
      return response as unknown as Response;
    }) as typeof fetch
  );
  return { calls };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

const CONFIG = { baseUrl: "", token: "tok-123" };

describe("getThreadMessages", () => {
  it("returns the validated history in order on a 200 { messages }", async () => {
    const wire: HistoryMessage[] = [
      { role: "user", text: "We have 120 guests." },
      { role: "assistant", text: "Here are venue ideas." },
    ];
    const { calls } = installFetch(jsonResponse(200, { messages: wire }));

    const result = await getThreadMessages(CONFIG, "thread-a");
    expect(result).toEqual(wire);

    // Hits the messages sub-route with GET + bearer auth; thread id is encoded.
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe("/threads/thread-a/messages");
    expect(calls[0]!.init?.method).toBe("GET");
    const headers = calls[0]!.init?.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer tok-123");
  });

  it("returns [] for an owned thread with no history", async () => {
    installFetch(jsonResponse(200, { messages: [] }));
    expect(await getThreadMessages(CONFIG, "empty")).toEqual([]);
  });

  it("url-encodes the thread id", async () => {
    const { calls } = installFetch(jsonResponse(200, { messages: [] }));
    await getThreadMessages(CONFIG, "a/b c");
    expect(calls[0]!.url).toBe("/threads/a%2Fb%20c/messages");
  });

  it("throws ThreadApiError(404) on a not-owned/not-found thread", async () => {
    installFetch(jsonResponse(404, { error: "Not found" }));
    await expect(getThreadMessages(CONFIG, "nope")).rejects.toMatchObject({
      name: "ThreadApiError",
      status: 404,
    });
  });

  it("throws ThreadApiError(401) on an unauthorized response", async () => {
    installFetch(jsonResponse(401, { error: "Unauthorized" }));
    await expect(getThreadMessages(CONFIG, "x")).rejects.toBeInstanceOf(
      ThreadApiError
    );
    await expect(getThreadMessages(CONFIG, "x")).rejects.toMatchObject({
      status: 401,
    });
  });

  it("rejects a non-array messages body", async () => {
    installFetch(jsonResponse(200, { messages: "not-an-array" }));
    await expect(getThreadMessages(CONFIG, "x")).rejects.toBeInstanceOf(
      ThreadApiError
    );
  });

  it("rejects an entry with an unknown role", async () => {
    installFetch(
      jsonResponse(200, { messages: [{ role: "system", text: "hi" }] })
    );
    await expect(getThreadMessages(CONFIG, "x")).rejects.toBeInstanceOf(
      ThreadApiError
    );
  });

  it("rejects an entry whose text is not a string", async () => {
    installFetch(
      jsonResponse(200, { messages: [{ role: "user", text: 42 }] })
    );
    await expect(getThreadMessages(CONFIG, "x")).rejects.toBeInstanceOf(
      ThreadApiError
    );
  });

  it("rejects a non-object entry", async () => {
    installFetch(jsonResponse(200, { messages: [null] }));
    await expect(getThreadMessages(CONFIG, "x")).rejects.toBeInstanceOf(
      ThreadApiError
    );
  });
});
