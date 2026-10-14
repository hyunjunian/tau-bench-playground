/**
 * Model providers reachable straight from the browser (all send CORS headers
 * and speak the OpenAI chat-completions format).
 *
 * Default: Google AI Studio's free tier with gemini-3.5-flash-lite, the free
 * model with the highest score on OpenRouter's τ²-Bench airline leaderboard
 * (run 2026-10-13). Scores below are from that leaderboard and are shown in
 * the model picker for reference.
 */

export type ProviderId = "google" | "openrouter" | "groq" | "cerebras" | "custom";

export interface ModelPreset {
  id: string;
  label: string;
  /** OpenRouter τ²-Bench airline accuracy (%) on 2026-10-13, when listed. */
  tau2Airline?: number;
}

export interface ProviderDef {
  id: ProviderId;
  label: string;
  baseUrl: string;
  keyUrl?: string;
  keyPlaceholder: string;
  models: ModelPreset[];
}

export const PROVIDERS: Record<ProviderId, ProviderDef> = {
  google: {
    id: "google",
    label: "Google AI Studio (Gemini API)",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
    keyUrl: "https://aistudio.google.com/apikey",
    keyPlaceholder: "AIza…",
    models: [
      { id: "gemini-3.5-flash-lite", label: "Gemini 3.5 Flash-Lite", tau2Airline: 76.9 },
      { id: "gemma-4-31b-it", label: "Gemma 4 31B", tau2Airline: 76.8 },
      { id: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview", tau2Airline: 75.3 },
      { id: "gemini-3.6-flash", label: "Gemini 3.6 Flash", tau2Airline: 74.7 },
      { id: "gemini-3.1-flash-lite", label: "Gemini 3.1 Flash-Lite", tau2Airline: 72.0 },
      { id: "gemini-3.8-flash", label: "Gemini 3.8 Flash", tau2Airline: 71.3 },
      { id: "gemma-4-26b-a4b-it", label: "Gemma 4 26B A4B", tau2Airline: 69.2 },
    ],
  },
  openrouter: {
    id: "openrouter",
    label: "OpenRouter (free models)",
    baseUrl: "https://openrouter.ai/api/v1",
    keyUrl: "https://openrouter.ai/keys",
    keyPlaceholder: "sk-or-…",
    models: [
      { id: "google/gemma-4-31b-it:free", label: "Gemma 4 31B (free)", tau2Airline: 76.8 },
      { id: "nvidia/nemotron-3-ultra-550b-a55b:free", label: "Nemotron 3 Ultra (free)", tau2Airline: 72.0 },
      { id: "google/gemma-4-26b-a4b-it:free", label: "Gemma 4 26B A4B (free)", tau2Airline: 69.2 },
      { id: "thinkingmachines/inkling:free", label: "Inkling (free)", tau2Airline: 67.2 },
      { id: "nvidia/nemotron-3.5-lightning:free", label: "Nemotron 3.5 Lightning (free)", tau2Airline: 65.0 },
      { id: "thinkingmachines/inkling-small:free", label: "Inkling Small (free)", tau2Airline: 60.7 },
      { id: "nvidia/nemotron-3-super-120b-a12b:free", label: "Nemotron 3 Super (free)" },
    ],
  },
  groq: {
    id: "groq",
    label: "Groq",
    baseUrl: "https://api.groq.com/openai/v1",
    keyUrl: "https://console.groq.com/keys",
    keyPlaceholder: "gsk_…",
    models: [],
  },
  cerebras: {
    id: "cerebras",
    label: "Cerebras",
    baseUrl: "https://api.cerebras.ai/v1",
    keyUrl: "https://cloud.cerebras.ai/",
    keyPlaceholder: "csk-…",
    models: [],
  },
  custom: {
    id: "custom",
    label: "Custom (OpenAI-compatible)",
    baseUrl: "http://localhost:11434/v1",
    keyPlaceholder: "optional",
    models: [],
  },
};

export const PROVIDER_ORDER: ProviderId[] = ["google", "openrouter", "groq", "cerebras", "custom"];

export type ReasoningEffort = "" | "none" | "minimal" | "low" | "medium" | "high";

export interface ModelConfig {
  provider: ProviderId;
  model: string;
  /** null keeps the provider's default (recommended for Gemini 3.x). */
  temperature: number | null;
  reasoningEffort: ReasoningEffort;
}

export const DEFAULT_MODEL: ModelConfig = {
  provider: "google",
  model: "gemini-3.5-flash-lite",
  temperature: null,
  reasoningEffort: "",
};

export interface Endpoint {
  provider: ProviderId;
  baseUrl: string;
  apiKey: string;
}

export function findPreset(provider: ProviderId, model: string): ModelPreset | undefined {
  return PROVIDERS[provider].models.find((m) => m.id === model);
}

/** Free models currently listed by OpenRouter (public endpoint, no key needed). */
export async function fetchOpenRouterFreeModels(signal?: AbortSignal): Promise<ModelPreset[]> {
  const res = await fetch("https://openrouter.ai/api/v1/models", { signal });
  if (!res.ok) throw new Error(`OpenRouter models: HTTP ${res.status}`);
  const body = (await res.json()) as {
    data: { id: string; name: string; supported_parameters?: string[]; pricing?: { prompt?: string; completion?: string } }[];
  };
  const known = new Map(PROVIDERS.openrouter.models.map((m) => [m.id, m]));
  return body.data
    .filter(
      (m) =>
        (m.id.endsWith(":free") || (m.pricing?.prompt === "0" && m.pricing?.completion === "0")) &&
        (m.supported_parameters ?? []).includes("tools"),
    )
    .map((m) => known.get(m.id) ?? { id: m.id, label: m.name })
    .sort((a, b) => (b.tau2Airline ?? -1) - (a.tau2Airline ?? -1) || a.label.localeCompare(b.label));
}

/** `GET /models` on any OpenAI-compatible endpoint. */
export async function fetchModelIds(endpoint: Endpoint, signal?: AbortSignal): Promise<string[]> {
  const headers: Record<string, string> = {};
  if (endpoint.apiKey) headers.Authorization = `Bearer ${endpoint.apiKey}`;
  const res = await fetch(`${endpoint.baseUrl.replace(/\/+$/, "")}/models`, { headers, signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const body = (await res.json()) as { data?: { id: string }[] };
  return (body.data ?? []).map((m) => m.id.replace(/^models\//, "")).sort();
}
