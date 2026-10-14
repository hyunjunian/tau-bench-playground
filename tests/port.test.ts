/**
 * Replays reference traces recorded from the original Python tau-bench
 * (scripts/make_reference.py) through the TypeScript port and checks that
 * every observation, every final database state and every reward match.
 *
 * REF_DIR overrides the reference directory (default: tests/reference).
 * Files may be plain .json or gzipped .json.gz.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { Domain, type DomainName, type Task, TauEnv, type ToolSpec, parseTasks } from "../src/tau/env";
import { type PyDict, deepCopy, deltaCanonical, parseJson } from "../src/tau/py";

const ROOT = join(__dirname, "..");
const REF_DIR = process.env.REF_DIR ?? join(ROOT, "tests", "reference");
const TIMEOUT = 600_000;

const read = (p: string) => readFileSync(join(ROOT, p), "utf8");
const sha = (s: string) => createHash("sha256").update(s, "utf8").digest("hex");

function findReference(name: string): string | null {
  for (const file of [`${name}.json`, `${name}.json.gz`]) {
    const p = join(REF_DIR, file);
    if (existsSync(p)) return p;
  }
  return null;
}

function loadReference<T>(path: string | null): T[] {
  if (!path) return [];
  const raw = readFileSync(path);
  return JSON.parse((path.endsWith(".gz") ? gunzipSync(raw) : raw).toString("utf8")) as T[];
}

function loadDomain(name: DomainName): Domain {
  const tools = JSON.parse(read(`public/data/${name}/tools.json`)) as ToolSpec[];
  return new Domain(name, read(`public/data/${name}/db.json`), tools, read(`public/data/${name}/wiki.md`));
}

const domains = { retail: loadDomain("retail"), airline: loadDomain("airline") };

const taskCache = new Map<string, Task[]>();
function tasksFor(domain: DomainName, split: string): Task[] {
  const key = `${domain}/${split}`;
  if (!taskCache.has(key)) taskCache.set(key, parseTasks(read(`public/data/${domain}/tasks_${split}.json`)));
  return taskCache.get(key)!;
}

function step(domain: Domain, data: PyDict, name: string, kwargs: PyDict): string | null {
  if (!domain.hasTool(name)) return `Unknown action ${name}`;
  return domain.invokeTool(data, name, deepCopy(kwargs));
}

/** Reference observations are either raw strings (--full) or sha256 hashes. */
function matches(expected: string | null, actual: string | null): boolean {
  if (expected === null || actual === null) return expected === actual;
  return expected === actual || expected === sha(actual);
}

interface GtTrace {
  domain: DomainName;
  split: string;
  index: number;
  observations: string[];
  final: string;
}

interface FuzzTrace {
  domain: DomainName;
  steps: { name: string; kwargs: string; observation: string | null }[];
  final: string;
}

interface RewardCase {
  domain: DomainName;
  split: string;
  index: number;
  variant: string;
  actions: { name: string; kwargs: string }[];
  responds: string[];
  expected: { reward: number; r_actions: boolean; r_outputs: number | null };
}

const gtPath = findReference("gt_traces");
const fuzzPath = findReference("fuzz_traces");
const rewardPath = findReference("reward_cases");

describe.skipIf(!gtPath)("ground-truth task replays match Python tau-bench", () => {
  const traces = loadReference<GtTrace>(gtPath);

  it(
    `replays ${traces.length} tasks`,
    () => {
      const failures: string[] = [];
      for (const trace of traces) {
        const domain = domains[trace.domain];
        const task = tasksFor(trace.domain, trace.split)[trace.index];
        const data = domain.freshData();
        task.actions.forEach((action, i) => {
          const obs = step(domain, data, action.name, action.kwargs);
          if (!matches(trace.observations[i], obs)) {
            failures.push(
              `${trace.domain}/${trace.split}#${trace.index} action ${i} (${action.name}):\n  expected ${trace.observations[i]?.slice(0, 300)}\n  actual   ${obs?.slice(0, 300)}`,
            );
          }
        });
        if (sha(deltaCanonical(domain.pristine, data)) !== trace.final) {
          failures.push(`${trace.domain}/${trace.split}#${trace.index}: final database differs`);
        }
      }
      expect(failures.slice(0, 10).join("\n")).toBe("");
    },
    TIMEOUT,
  );
});

describe.skipIf(!fuzzPath)("randomized tool calls match Python tau-bench", () => {
  const traces = loadReference<FuzzTrace>(fuzzPath);

  it(
    `replays ${traces.length} fuzz episodes`,
    () => {
      const failures: string[] = [];
      traces.forEach((trace, e) => {
        const domain = domains[trace.domain];
        const data = domain.freshData();
        trace.steps.forEach((s, i) => {
          const obs = step(domain, data, s.name, parseJson(s.kwargs) as PyDict);
          if (!matches(s.observation, obs)) {
            failures.push(
              `episode ${e} (${trace.domain}) step ${i} ${s.name}(${s.kwargs.slice(0, 200)}):\n  expected ${s.observation?.slice(0, 400)}\n  actual   ${obs?.slice(0, 400)}`,
            );
          }
        });
        if (sha(deltaCanonical(domain.pristine, data)) !== trace.final) {
          failures.push(`episode ${e} (${trace.domain}): final database differs`);
        }
      });
      expect(failures.slice(0, 15).join("\n")).toBe("");
    },
    TIMEOUT,
  );
});

describe.skipIf(!rewardPath)("rewards match Env.calculate_reward", () => {
  const cases = loadReference<RewardCase>(rewardPath);

  it(
    `scores ${cases.length} trajectories`,
    () => {
      const failures: string[] = [];
      for (const c of cases) {
        const env = new TauEnv(domains[c.domain], tasksFor(c.domain, c.split)[c.index]);
        for (const a of c.actions) env.stepTool({ name: a.name, kwargs: parseJson(a.kwargs) as PyDict });
        for (const content of c.responds) env.recordRespond(content);
        const r = env.calculateReward();
        const actual = { reward: r.reward, r_actions: r.rActions, r_outputs: r.rOutputs };
        if (JSON.stringify(actual) !== JSON.stringify(c.expected)) {
          failures.push(
            `${c.domain}/${c.split}#${c.index} ${c.variant}: expected ${JSON.stringify(c.expected)}, got ${JSON.stringify(actual)}`,
          );
        }
      }
      expect(failures.slice(0, 15).join("\n")).toBe("");
    },
    TIMEOUT,
  );
});
