/**
 * OpenAI-compatible chat completions client that runs in the browser.
 * Retries rate limits (honouring the provider's suggested delay) and
 * transient server errors, and keeps provider-specific fields that must be
 * sent back on the next turn (Gemini thought signatures, OpenRouter
 * reasoning_details).
 */
import type { ToolInfo } from "../tau/env";
import type { Endpoint, ProviderId, ReasoningEffort } from "./providers";

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
  [extra: string]: unknown;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
  [extra: string]: unknown;
}

export interface CompletionRequest {
  endpoint: Endpoint;
  model: string;
  messages: ChatMessage[];
  tools?: ToolInfo[];
  temperature?: number | null;
  reasoningEffort?: ReasoningEffort;
  signal?: AbortSignal;
  /** Called before waiting to retry, so the UI can show a countdown. */
  onRetry?: (info: { attempt: number; waitMs: number; reason: string }) => void;
}

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface CompletionResult {
  /** The assistant message, cleaned up so it can be appended to the history. */
  message: ChatMessage;
  /** Reasoning text if the provider returned any (display only). */
  reasoning?: string;
  usage?: Usage;
  latencyMs: number;
}

/** No API key is set for the provider a role uses (thrown before any request). */
export class MissingKeyError extends Error {
  constructor(
    readonly provider: ProviderId,
    message: string,
  ) {
    super(message);
    this.name = "MissingKeyError";
  }
}

export class LlmError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly retryable = false,
  ) {
    super(message);
    this.name = "LlmError";
  }
}

const MAX_RATE_LIMIT_RETRIES = 6;
const MAX_SERVER_RETRIES = 4;
const MAX_WAIT_MS = 90_000;
const REQUEST_TIMEOUT_MS = 180_000;

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Seconds to wait, from a Retry-After header or a provider message ("retry in 32.5s", "retryDelay": "32s"). */
function suggestedDelayMs(res: Response, bodyText: string): number | null {
  const header = res.headers.get("retry-after");
  if (header) {
    const secs = Number(header);
    if (Number.isFinite(secs)) return secs * 1000;
    const date = Date.parse(header);
    if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
  }
  const m =
    /retry(?:Delay)?"?\s*[:=]?\s*"?(?:in\s+)?(\d+(?:\.\d+)?)\s*(ms|s)\b/i.exec(bodyText) ??
    /try again in\s+(\d+(?:\.\d+)?)\s*(ms|s)\b/i.exec(bodyText);
  if (m) return Number(m[1]) * (m[2].toLowerCase() === "ms" ? 1 : 1000);
  return null;
}

function errorMessage(bodyText: string): string {
  try {
    const body = JSON.parse(bodyText) as unknown;
    const pick = (b: unknown): string | undefined => {
      if (!b || typeof b !== "object") return undefined;
      const o = b as Record<string, unknown>;
      if (typeof o.message === "string") {
        const raw = (o.metadata as Record<string, unknown> | undefined)?.raw;
        return typeof raw === "string" && raw.length < 600 ? `${o.message} (${raw})` : o.message;
      }
      if (o.error) return pick(o.error);
      return undefined;
    };
    const fromArray = Array.isArray(body) ? pick(body[0]) : undefined;
    return fromArray ?? pick(body) ?? bodyText.slice(0, 500);
  } catch {
    return bodyText.slice(0, 500) || "empty response";
  }
}

function isDailyQuota(text: string): boolean {
  return /per[\s_-]?day|daily|PerDay/i.test(text);
}

/** Text of a message's `content` (a string, or a list of text parts). */
export function textContent(content: unknown): string | null {
  if (content === null || content === undefined) return null;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => (typeof p === "string" ? p : typeof p?.text === "string" ? p.text : ""))
      .join("");
  }
  return String(content);
}

let callCounter = 0;

/** Keep only what the provider accepts back in the conversation history. */
function cleanAssistantMessage(raw: Record<string, unknown>, provider: ProviderId): ChatMessage {
  const msg: ChatMessage = { role: "assistant", content: textContent(raw.content) };
  const calls = Array.isArray(raw.tool_calls) ? (raw.tool_calls as Record<string, unknown>[]) : [];
  const toolCalls: ToolCall[] = [];
  for (const c of calls) {
    const fn = (c.function ?? {}) as Record<string, unknown>;
    if (typeof fn.name !== "string") continue;
    const args = fn.arguments;
    const call: ToolCall = {
      id: typeof c.id === "string" && c.id ? c.id : `call_${Date.now().toString(36)}_${callCounter++}`,
      type: "function",
      function: { name: fn.name, arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}) },
    };
    if (provider === "google" && c.extra_content) call.extra_content = c.extra_content;
    toolCalls.push(call);
  }
  if (toolCalls.length > 0) msg.tool_calls = toolCalls;
  if (provider === "google" && raw.extra_content) msg.extra_content = raw.extra_content;
  if (provider === "openrouter" && Array.isArray(raw.reasoning_details) && raw.reasoning_details.length > 0) {
    msg.reasoning_details = raw.reasoning_details;
  }
  return msg;
}

/** Gemini rejects object schemas with no properties; a function without parameters is equivalent. */
function toolsForProvider(tools: ToolInfo[], provider: ProviderId): unknown[] {
  if (provider !== "google") return tools;
  return tools.map((t) => {
    const params = t.function.parameters;
    if (params.properties && Object.keys(params.properties).length > 0) return t;
    return { type: "function", function: { name: t.function.name, description: t.function.description } };
  });
}

/**
 * Gemini 3 rejects a function call in the history that carries no thought signature, which is the case for
 * calls a human made (or another model, after a model switch). Google documents this placeholder for them.
 */
const SKIP_SIGNATURE = { google: { thought_signature: "skip_thought_signature_validator" } };

function messagesForProvider(messages: ChatMessage[], provider: ProviderId): ChatMessage[] {
  if (provider !== "google") return messages;
  return messages.map((m) =>
    m.tool_calls?.some((c) => !c.extra_content)
      ? { ...m, tool_calls: m.tool_calls.map((c) => (c.extra_content ? c : { ...c, extra_content: SKIP_SIGNATURE })) }
      : m,
  );
}

export async function chatCompletion(req: CompletionRequest): Promise<CompletionResult> {
  const { endpoint } = req;
  const url = `${endpoint.baseUrl.replace(/\/+$/, "")}/chat/completions`;
  const body: Record<string, unknown> = { model: req.model, messages: messagesForProvider(req.messages, endpoint.provider) };
  if (req.tools && req.tools.length > 0) body.tools = toolsForProvider(req.tools, endpoint.provider);
  if (req.temperature !== null && req.temperature !== undefined) body.temperature = req.temperature;
  if (req.reasoningEffort) body.reasoning_effort = req.reasoningEffort;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (endpoint.apiKey) headers.Authorization = `Bearer ${endpoint.apiKey}`;
  if (endpoint.provider === "openrouter") {
    headers["X-Title"] = "tau-bench playground";
    if (typeof location !== "undefined" && location.protocol.startsWith("http")) {
      headers["HTTP-Referer"] = location.origin;
    }
  }

  let rateLimitRetries = 0;
  let serverRetries = 0;
  for (;;) {
    const started = performance.now();
    let res: Response;
    try {
      const signals = [AbortSignal.timeout(REQUEST_TIMEOUT_MS)];
      if (req.signal) signals.push(req.signal);
      res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.any(signals),
      });
    } catch (e) {
      if (req.signal?.aborted) throw e;
      if (e instanceof DOMException && e.name === "TimeoutError") {
        throw new LlmError(`No response from ${endpoint.baseUrl} within ${REQUEST_TIMEOUT_MS / 1000} s.`, undefined, true);
      }
      if (serverRetries < 2) {
        serverRetries++;
        const waitMs = 2000 * serverRetries;
        req.onRetry?.({ attempt: serverRetries, waitMs, reason: "network error" });
        await sleep(waitMs, req.signal);
        continue;
      }
      throw new LlmError(
        `Network error calling ${endpoint.baseUrl} (${e instanceof Error ? e.message : String(e)}). ` +
          "Check the base URL, your connection, and that the endpoint allows browser (CORS) requests.",
        undefined,
        true,
      );
    }

    const text = await res.text();
    if (res.ok) {
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(text) as Record<string, unknown>;
      } catch {
        throw new LlmError(`Invalid JSON from provider: ${text.slice(0, 300)}`, res.status, true);
      }
      if (parsed.error) throw new LlmError(errorMessage(text), res.status, true);
      const choices = parsed.choices as { message?: Record<string, unknown>; finish_reason?: string }[] | undefined;
      const raw = choices?.[0]?.message;
      if (!raw) throw new LlmError(`No choices in response: ${text.slice(0, 300)}`, res.status, true);
      const reasoning =
        typeof raw.reasoning === "string"
          ? raw.reasoning
          : typeof raw.reasoning_content === "string"
            ? raw.reasoning_content
            : undefined;
      return {
        message: cleanAssistantMessage(raw, endpoint.provider),
        reasoning: reasoning || undefined,
        usage: parsed.usage as Usage | undefined,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const msg = errorMessage(text);
    if (res.status === 429 && !isDailyQuota(text) && rateLimitRetries < MAX_RATE_LIMIT_RETRIES) {
      rateLimitRetries++;
      const suggested = suggestedDelayMs(res, text);
      const waitMs = Math.min(MAX_WAIT_MS, Math.max(1000, suggested ?? 5000 * 2 ** (rateLimitRetries - 1)));
      req.onRetry?.({ attempt: rateLimitRetries, waitMs, reason: `rate limited (429): ${msg}` });
      await sleep(waitMs, req.signal);
      continue;
    }
    if ([500, 502, 503, 504, 529].includes(res.status) && serverRetries < MAX_SERVER_RETRIES) {
      serverRetries++;
      const waitMs = 2000 * 2 ** (serverRetries - 1);
      req.onRetry?.({ attempt: serverRetries, waitMs, reason: `server error ${res.status}: ${msg}` });
      await sleep(waitMs, req.signal);
      continue;
    }
    const retryable = (res.status === 429 && !isDailyQuota(text)) || res.status >= 500;
    throw new LlmError(`HTTP ${res.status}: ${msg}`, res.status, retryable);
  }
}
