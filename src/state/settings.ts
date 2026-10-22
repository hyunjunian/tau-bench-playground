import { useCallback, useEffect, useState } from "react";
import { type Lang, detectLang } from "../i18n";
import { DEFAULT_MODEL, type ModelConfig, PROVIDERS, type ProviderId } from "../llm/providers";
import { DOMAIN_SPLITS, type DomainName } from "../tau/env";
import type { Controller } from "../tau/episode";

export interface RoleSettings {
  controller: Controller;
  model: ModelConfig;
}

export interface BatchSettings {
  /** First and last task index (inclusive). */
  from: number;
  to: number;
  trials: number;
}

export type Theme = "system" | "light" | "dark";

export interface Settings {
  lang: Lang;
  theme: Theme;
  domain: DomainName;
  split: string;
  taskIndex: number;
  agent: RoleSettings;
  user: RoleSettings;
  keys: Partial<Record<ProviderId, string>>;
  customBaseUrl: string;
  maxAgentSteps: number;
  stepMode: boolean;
  batch: BatchSettings;
}

const STORAGE_KEY = "tau-playground.settings.v1";

export function defaultSettings(): Settings {
  return {
    lang: detectLang(),
    theme: "system",
    domain: "retail",
    split: "test",
    taskIndex: 0,
    agent: { controller: "model", model: { ...DEFAULT_MODEL } },
    user: { controller: "model", model: { ...DEFAULT_MODEL } },
    keys: {},
    customBaseUrl: PROVIDERS.custom.baseUrl,
    maxAgentSteps: 30,
    stepMode: false,
    batch: { from: 0, to: 9, trials: 1 },
  };
}

const isCount = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 0;

function sanitizeBatch(raw: unknown, d: BatchSettings): BatchSettings {
  const b = raw && typeof raw === "object" ? (raw as Partial<BatchSettings>) : {};
  const from = isCount(b.from) ? b.from : d.from;
  const to = isCount(b.to) && b.to >= from ? b.to : Math.max(from, d.to);
  const trials = isCount(b.trials) && b.trials >= 1 ? Math.min(b.trials, 100) : d.trials;
  return { from, to, trials };
}

export function sanitize(raw: unknown): Settings {
  const d = defaultSettings();
  if (!raw || typeof raw !== "object") return d;
  const s = { ...d, ...(raw as Partial<Settings>) };
  if (!(s.domain in DOMAIN_SPLITS)) s.domain = d.domain;
  if (!DOMAIN_SPLITS[s.domain].includes(s.split)) s.split = "test";
  for (const role of ["agent", "user"] as const) {
    const r = s[role];
    if (!r || (r.controller !== "human" && r.controller !== "model") || !r.model || !(r.model.provider in PROVIDERS)) {
      s[role] = d[role];
    } else {
      s[role] = { controller: r.controller, model: { ...DEFAULT_MODEL, ...r.model } };
    }
  }
  if (!Number.isInteger(s.taskIndex) || s.taskIndex < 0) s.taskIndex = 0;
  if (!Number.isInteger(s.maxAgentSteps) || s.maxAgentSteps < 0) s.maxAgentSteps = 30;
  if (s.lang !== "ko" && s.lang !== "en") s.lang = d.lang;
  if (s.theme !== "light" && s.theme !== "dark") s.theme = "system";
  const keys = typeof s.keys === "object" && s.keys ? s.keys : {};
  s.keys = Object.fromEntries(Object.entries(keys).filter(([p, v]) => p in PROVIDERS && typeof v === "string"));
  if (typeof s.customBaseUrl !== "string") s.customBaseUrl = d.customBaseUrl;
  s.stepMode = s.stepMode === true;
  s.batch = sanitizeBatch(s.batch, d.batch);
  return s;
}

export type TaskLink = Pick<Settings, "domain" | "split" | "taskIndex">;

/** `#retail/test/12` -> that task, so a task can be linked to. */
export function taskFromHash(hash: string): TaskLink | null {
  const m = /^#\/?(retail|airline)\/([a-z]+)\/(\d+)$/.exec(hash);
  if (!m) return null;
  const domain = m[1] as DomainName;
  if (!DOMAIN_SPLITS[domain].includes(m[2])) return null;
  return { domain, split: m[2], taskIndex: Number(m[3]) };
}

export function taskHash(s: TaskLink): string {
  return `#${s.domain}/${s.split}/${s.taskIndex}`;
}

function load(): Settings {
  let settings: Settings;
  try {
    const text = localStorage.getItem(STORAGE_KEY);
    settings = sanitize(text ? JSON.parse(text) : null);
  } catch {
    settings = defaultSettings();
  }
  const link = typeof location === "undefined" ? null : taskFromHash(location.hash);
  return link ? { ...settings, ...link } : settings;
}

export function useSettings(): [Settings, (patch: Partial<Settings> | ((s: Settings) => Settings)) => void] {
  const [settings, setSettings] = useState<Settings>(load);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // storage unavailable (private mode, blocked): settings just won't persist
    }
  }, [settings]);
  const update = useCallback((patch: Partial<Settings> | ((s: Settings) => Settings)) => {
    setSettings((s) => (typeof patch === "function" ? patch(s) : { ...s, ...patch }));
  }, []);
  return [settings, update];
}
