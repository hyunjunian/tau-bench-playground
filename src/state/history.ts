import { useCallback, useState } from "react";
import type { DomainName } from "../tau/env";
import type { EndReason } from "../tau/episode";

export interface HistoryEntry {
  id: string;
  at: string;
  domain: DomainName;
  split: string;
  index: number;
  agent: string;
  user: string;
  reward: number;
  reason: EndReason;
  steps: number;
}

const STORAGE_KEY = "tau-playground.history.v1";
const MAX_ENTRIES = 300;

function load(): HistoryEntry[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as unknown;
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

function save(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // ignore: history is a convenience
  }
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(load);
  const add = useCallback((entry: HistoryEntry) => {
    setEntries((prev) => {
      const next = [entry, ...prev].slice(0, MAX_ENTRIES);
      save(next);
      return next;
    });
  }, []);
  const clear = useCallback(() => {
    save([]);
    setEntries([]);
  }, []);
  return { entries, add, clear };
}
