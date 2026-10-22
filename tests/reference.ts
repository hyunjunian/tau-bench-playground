/**
 * Reference traces recorded from the original Python tau-bench
 * (scripts/make_reference.py), replayed through the TypeScript port by the
 * port-*.test.ts files (one per trace file, so they run in parallel).
 *
 * REF_DIR overrides the reference directory (default: tests/reference).
 * Files may be plain .json or gzipped .json.gz.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import type { Domain, DomainName } from "../src/tau/env";
import { type PyDict, deepCopy } from "../src/tau/py";
import { ROOT } from "./helpers";

const REF_DIR = process.env.REF_DIR ?? join(ROOT, "tests", "reference");
export const TIMEOUT = 600_000;

export const sha = (s: string) => createHash("sha256").update(s, "utf8").digest("hex");

export function findReference(name: string): string | null {
  for (const file of [`${name}.json`, `${name}.json.gz`]) {
    const p = join(REF_DIR, file);
    if (existsSync(p)) return p;
  }
  return null;
}

export function loadReference<T>(path: string | null): T[] {
  if (!path) return [];
  const raw = readFileSync(path);
  return JSON.parse((path.endsWith(".gz") ? gunzipSync(raw) : raw).toString("utf8")) as T[];
}

export function step(domain: Domain, data: PyDict, name: string, kwargs: PyDict): string | null {
  if (!domain.hasTool(name)) return `Unknown action ${name}`;
  return domain.invokeTool(data, name, deepCopy(kwargs));
}

/** Reference observations are either raw strings (--full) or sha256 hashes. */
export function matches(expected: string | null, actual: string | null): boolean {
  if (expected === null || actual === null) return expected === actual;
  return expected === actual || expected === sha(actual);
}

export interface GtTrace {
  domain: DomainName;
  split: string;
  index: number;
  observations: string[];
  final: string;
}

export interface FuzzTrace {
  domain: DomainName;
  steps: { name: string; kwargs: string; observation: string | null }[];
  final: string;
}

export interface RewardCase {
  domain: DomainName;
  split: string;
  index: number;
  variant: string;
  actions: { name: string; kwargs: string }[];
  responds: string[];
  expected: { reward: number; r_actions: boolean; r_outputs: number | null };
}
