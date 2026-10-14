import { useCallback, useEffect, useState } from "react";
import { type Lang, detectLang } from "../i18n";
import { DEFAULT_MODEL, type ModelConfig, PROVIDERS, type ProviderId } from "../llm/providers";
import { DOMAIN_SPLITS, type DomainName } from "../tau/env";
import type { Controller } from "../tau/episode";

export interface RoleSettings {
  controller: Controller;
  model: ModelConfig;
}

export interface Settings {
  lang: Lang;
  domain: DomainName;
  split: string;
  taskIndex: number;
  agent: RoleSettings;
  user: RoleSettings;
  keys: Partial<Record<ProviderId, string>>;
  customBaseUrl: string;
  maxAgentSteps: number;
  stepMode: boolean;
}

const STORAGE_KEY = "tau-playground.settings.v1";

export function defaultSettings(): Settings {
  return {
    lang: detectLang(),
    domain: "retail",
    split: "test",
    taskIndex: 0,
    agent: { controller: "model", model: { ...DEFAULT_MODEL } },
    user: { controller: "model", model: { ...DEFAULT_MODEL } },
    keys: {},
    customBaseUrl: PROVIDERS.custom.baseUrl,
    maxAgentSteps: 30,
    stepMode: false,
  };
}

function sanitize(raw: unknown): Settings {
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
  s.keys = typeof s.keys === "object" && s.keys ? s.keys : {};
  return s;
}

function load(): Settings {
  try {
    const text = localStorage.getItem(STORAGE_KEY);
    return sanitize(text ? JSON.parse(text) : null);
  } catch {
    return defaultSettings();
  }
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
