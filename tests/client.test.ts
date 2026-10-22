/** The browser chat-completions client: retries, provider quirks and error messages. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { type CompletionRequest, LlmError, chatCompletion } from "../src/llm/client";
import type { ProviderId } from "../src/llm/providers";
import type { ToolInfo } from "../src/tau/env";

type Reply = { status: number; body: unknown; headers?: Record<string, string> };

function provider(replies: Reply[]) {
  const fetch = vi.fn(async (_url: string, _init: RequestInit) => {
    const r = replies.shift();
    if (!r) throw new Error("no more replies");
    const text = typeof r.body === "string" ? r.body : JSON.stringify(r.body);
    return new Response(text, { status: r.status, headers: r.headers });
  });
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

const ok = (message: Record<string, unknown>, extra: Record<string, unknown> = {}): Reply => ({
  status: 200,
  body: { choices: [{ message: { role: "assistant", ...message } }], ...extra },
});

function request(provider: ProviderId = "custom", extra: Partial<CompletionRequest> = {}): CompletionRequest {
  return {
    endpoint: { provider, baseUrl: "https://example.test/v1/", apiKey: "key" },
    model: "m",
    messages: [{ role: "user", content: "hi" }],
    ...extra,
  };
}

/** Run a request while letting fake timers advance through its retry waits. */
async function run(req: CompletionRequest) {
  const p = chatCompletion(req);
  p.catch(() => {});
  await vi.runAllTimersAsync();
  return p;
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("chatCompletion", () => {
  it("posts to <base>/chat/completions with the key", async () => {
    const fetch = provider([ok({ content: "hello" }, { usage: { prompt_tokens: 3, completion_tokens: 1 } })]);
    const res = await run(request("custom", { temperature: 0, reasoningEffort: "low" }));
    expect(res.message).toEqual({ role: "assistant", content: "hello" });
    expect(res.usage).toEqual({ prompt_tokens: 3, completion_tokens: 1 });
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("https://example.test/v1/chat/completions");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer key");
    expect(JSON.parse(String(init.body))).toMatchObject({ model: "m", temperature: 0, reasoning_effort: "low" });
  });

  it("waits as long as Retry-After says on 429, then retries", async () => {
    provider([
      { status: 429, body: { error: { message: "slow down" } }, headers: { "Retry-After": "3" } },
      ok({ content: "ok" }),
    ]);
    const onRetry = vi.fn();
    const res = await run(request("custom", { onRetry }));
    expect(res.message.content).toBe("ok");
    expect(onRetry).toHaveBeenCalledWith(expect.objectContaining({ attempt: 1, waitMs: 3000 }));
  });

  it("reads Gemini's retryDelay from the error body", async () => {
    const body = [{ error: { code: 429, message: "Resource has been exhausted", details: [{ retryDelay: "32s" }] } }];
    provider([{ status: 429, body }, ok({ content: "ok" })]);
    const onRetry = vi.fn();
    await run(request("google", { onRetry }));
    expect(onRetry.mock.calls[0][0].waitMs).toBe(32_000);
  });

  it("does not retry a daily quota and reports it as not retryable", async () => {
    const fetch = provider([
      {
        status: 429,
        body: [{ error: { message: "Quota exceeded for quota metric 'GenerateRequestsPerDayPerProjectPerModel'" } }],
      },
    ]);
    const err = await run(request("google")).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LlmError);
    expect((err as LlmError).retryable).toBe(false);
    expect((err as LlmError).message).toContain("HTTP 429: Quota exceeded");
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("backs off on server errors", async () => {
    provider([{ status: 503, body: "overloaded" }, { status: 502, body: "bad gateway" }, ok({ content: "ok" })]);
    const onRetry = vi.fn();
    await run(request("custom", { onRetry }));
    expect(onRetry.mock.calls.map((c) => c[0].waitMs)).toEqual([2000, 4000]);
  });

  it("gives up after repeated server errors with a retryable error", async () => {
    provider(Array.from({ length: 5 }, () => ({ status: 500, body: { error: { message: "internal" } } })));
    const err = await run(request()).catch((e: unknown) => e);
    expect(err).toMatchObject({ status: 500, retryable: true, message: "HTTP 500: internal" });
  });

  it("explains a rejected key", async () => {
    provider([{ status: 400, body: [{ error: { code: 400, message: "API key not valid.", status: "INVALID_ARGUMENT" } }] }]);
    const err = await run(request("google")).catch((e: unknown) => e);
    expect(err).toMatchObject({ message: "HTTP 400: API key not valid.", retryable: false });
  });

  it("drops empty parameter schemas for Gemini and keeps thought signatures", async () => {
    const tools: ToolInfo[] = [
      {
        type: "function",
        function: {
          name: "list_all_airports",
          description: "List airports.",
          parameters: { type: "object", properties: {}, required: [] },
        },
      },
      {
        type: "function",
        function: {
          name: "think",
          description: "Think.",
          parameters: { type: "object", properties: { thought: { type: "string" } } },
        },
      },
    ];
    const call = {
      id: "c1",
      type: "function",
      function: { name: "think", arguments: { thought: "x" } },
      extra_content: { google: { thought_signature: "sig" } },
    };
    const fetch = provider([ok({ content: null, tool_calls: [call] })]);
    const res = await run(request("google", { tools }));
    const sent = JSON.parse(String(fetch.mock.calls[0][1].body)).tools;
    expect(sent[0]).toEqual({ type: "function", function: { name: "list_all_airports", description: "List airports." } });
    expect(sent[1]).toEqual(tools[1]);
    // object arguments are serialized, provider extras are kept for the next turn
    expect(res.message.tool_calls?.[0]).toEqual({
      id: "c1",
      type: "function",
      function: { name: "think", arguments: '{"thought":"x"}' },
      extra_content: { google: { thought_signature: "sig" } },
    });
  });

  it("gives Gemini a placeholder thought signature for tool calls it did not make", async () => {
    const human = { id: "h1", type: "function" as const, function: { name: "think", arguments: "{}" } };
    const gemini = { ...human, id: "g1", extra_content: { google: { thought_signature: "real" } } };
    const messages = [
      { role: "user" as const, content: "hi" },
      { role: "assistant" as const, content: null, tool_calls: [human] },
      { role: "tool" as const, tool_call_id: "h1", name: "think", content: "" },
      { role: "assistant" as const, content: null, tool_calls: [gemini] },
    ];
    const fetch = provider([ok({ content: "x" }), ok({ content: "x" })]);
    await run(request("google", { messages }));
    const sent = JSON.parse(String(fetch.mock.calls[0][1].body)).messages;
    expect(sent[1].tool_calls[0].extra_content).toEqual({ google: { thought_signature: "skip_thought_signature_validator" } });
    expect(sent[3].tool_calls[0].extra_content).toEqual({ google: { thought_signature: "real" } });
    expect(messages[1].tool_calls?.[0]).not.toHaveProperty("extra_content"); // the history itself is untouched

    await run(request("openrouter", { messages }));
    expect(JSON.parse(String(fetch.mock.calls[1][1].body)).messages[1].tool_calls[0]).not.toHaveProperty("extra_content");
  });

  it("keeps OpenRouter reasoning details and shows reasoning text", async () => {
    provider([ok({ content: "hi", reasoning: "because", reasoning_details: [{ type: "reasoning.text", text: "because" }] })]);
    const res = await run(request("openrouter"));
    expect(res.reasoning).toBe("because");
    expect(res.message.reasoning_details).toEqual([{ type: "reasoning.text", text: "because" }]);
  });

  it("stops waiting when aborted", async () => {
    provider([{ status: 429, body: "busy", headers: { "Retry-After": "60" } }]);
    const controller = new AbortController();
    const p = chatCompletion(request("custom", { signal: controller.signal }));
    const settled = p.then(
      () => "resolved",
      (e: unknown) => (e as Error).name,
    );
    await vi.advanceTimersByTimeAsync(1000);
    controller.abort();
    expect(await settled).toBe("AbortError");
  });
});
