/**
 * Reading saved episodes: the playground's own JSON export (one episode, or a
 * batch results file with many) and the results files written by tau-bench's
 * run.py (a list of EnvRunResult: task_id, reward, info, traj, trial).
 */
import type { ChatMessage } from "../llm/client";
import { DEFAULT_MODEL } from "../llm/providers";
import { DOMAIN_SPLITS, type Domain, type DomainName, type Task, parseTasks } from "./env";
import { type Controller, type EndReason, Episode, type ReplaySource, type Role } from "./episode";
import { PyDict, PyFloat, type PyValue, jsonDumps, parseJson, strictEqual, toPlain } from "./py";

export interface EpisodeRecord {
  /** Position in the file (0-based). */
  position: number;
  taskId: number | null;
  trial: number | null;
  reward: number | null;
  /** Known from the file itself (playground exports) or guessed from the policy / tool names. */
  domain: DomainName | null;
  split: string | null;
  /** The task embedded in the file (info.task), if any. */
  task: Task | null;
  traj: ChatMessage[];
  controllers: Partial<Record<Role, Controller>>;
  models: Partial<Record<Role, string>>;
  endReason: EndReason | null;
  /** null when the file does not say (tau-bench's default is 30). */
  maxAgentSteps: number | null;
  /** tau-bench stores the exception here when a run crashed (and then has no trajectory). */
  error: string | null;
}

const END_REASONS: EndReason[] = ["user_stop", "transfer", "max_steps", "manual", "crash", "error", "incomplete"];
const RETAIL_HINTS = ["get_order_details", "find_user_id_by_email", "find_user_id_by_name_zip", "get_product_details"];
const AIRLINE_HINTS = ["get_reservation_details", "search_direct_flight", "book_reservation", "get_user_details"];

function asDict(v: PyValue | undefined): PyDict | null {
  return v instanceof PyDict ? v : null;
}

function asNumber(v: PyValue | undefined): number | null {
  if (typeof v === "number") return v;
  return v instanceof PyFloat ? v.value : null;
}

function asString(v: PyValue | undefined): string | null {
  return typeof v === "string" ? v : null;
}

function guessDomain(traj: ChatMessage[]): DomainName | null {
  const system = traj.find((m) => m.role === "system");
  const head = typeof system?.content === "string" ? system.content.trimStart().slice(0, 80).toLowerCase() : "";
  if (head.startsWith("# retail")) return "retail";
  if (head.startsWith("# airline")) return "airline";
  const names = traj.flatMap((m) => (m.tool_calls ?? []).map((c) => c.function.name));
  const text = traj.map((m) => (typeof m.content === "string" ? m.content : "")).join("\n");
  if (RETAIL_HINTS.some((n) => names.includes(n) || text.includes(`"${n}"`))) return "retail";
  if (AIRLINE_HINTS.some((n) => names.includes(n) || text.includes(`"${n}"`))) return "airline";
  return null;
}

function readTask(v: PyValue | undefined): Task | null {
  const t = asDict(v);
  if (!t) return null;
  const actions = t.get("actions");
  if (typeof t.get("instruction") !== "string" || !Array.isArray(actions)) return null;
  try {
    // Round-trip through the Python-compatible parser so int/float kwargs survive.
    return parseTasks(jsonDumps([t]))[0];
  } catch {
    return null;
  }
}

function readTraj(v: PyValue | undefined): ChatMessage[] {
  if (!Array.isArray(v)) return [];
  return v
    .filter((m): m is PyDict => m instanceof PyDict && typeof m.get("role") === "string")
    .map((m) => {
      // Some files (the Claude trajectories in tau-bench's historical_trajectories) store tool-call
      // arguments as an object rather than a JSON string; serialize them like Python so ints and
      // floats stay apart.
      const calls = m.get("tool_calls");
      if (Array.isArray(calls)) {
        for (const c of calls) {
          const fn = c instanceof PyDict ? asDict(c.get("function")) : null;
          const args = fn?.get("arguments");
          if (fn && args instanceof PyDict) fn.set("arguments", jsonDumps(args));
        }
      }
      return toPlain(m) as ChatMessage;
    });
}

/**
 * Real runs that the tau-bench authors published (historical_trajectories), pinned to the commit the
 * playground's data comes from. raw.githubusercontent.com allows cross-origin reads.
 */
export const SAMPLE_RUNS = [
  { label: "GPT-4o airline", file: "gpt-4o-airline.json", mb: 4, agent: "gpt-4o" },
  { label: "GPT-4o retail", file: "gpt-4o-retail.json", mb: 11, agent: "gpt-4o" },
  { label: "Claude 3.5 Sonnet airline", file: "sonnet-35-new-airline.json", mb: 11, agent: "claude-3.5-sonnet" },
  { label: "Claude 3.5 Sonnet retail", file: "sonnet-35-new-retail.json", mb: 27, agent: "claude-3.5-sonnet" },
].map((s) => ({
  ...s,
  url: `https://raw.githubusercontent.com/sierra-research/tau-bench/59a200c6d575d595120f1cb70fea53cef0632f6b/historical_trajectories/${s.file}`,
}));

/** Model names from a tau-bench results file name (run.py's checkpoint path). */
export function modelsFromFileName(name: string): Partial<Record<Role, string>> {
  const base = name.split(/[\\/]/).pop() ?? name;
  const sample = SAMPLE_RUNS.find((s) => s.file === base);
  if (sample) return { agent: sample.agent }; // the files do not say which model simulated the user
  const m =
    /^(?:tool-calling|act|react|few-shot)-(.+)-\d+(?:\.\d+)?_range_-?\d+--?\d+_user-(.+)-(?:llm|react|verify|reflection)_/.exec(
      base,
    );
  return m ? { agent: m[1], user: m[2] } : {};
}

function readRecord(raw: PyValue, position: number, fileModels: Partial<Record<Role, string>>): EpisodeRecord | null {
  const r = asDict(raw);
  if (!r || !(r.has("traj") || r.has("info"))) return null;
  const info = asDict(r.get("info"));
  const pg = asDict(r.get("playground"));
  const traj = readTraj(r.get("traj"));
  const controllers: Partial<Record<Role, Controller>> = {};
  const models: Partial<Record<Role, string>> = { ...fileModels };
  for (const role of ["agent", "user"] as const) {
    const d = asDict(pg?.get(role));
    const c = asString(d?.get("controller"));
    if (c === "human" || c === "model") controllers[role] = c;
    const used = d?.get("models");
    if (Array.isArray(used) && used.length > 0) models[role] = used.filter((x) => typeof x === "string").join(", ");
  }
  const env = asString(pg?.get("env"));
  const split = asString(pg?.get("task_split"));
  const reason = asString(info?.get("end_reason"));
  const error = asString(info?.get("error"));
  return {
    position,
    taskId: asNumber(r.get("task_id")),
    trial: asNumber(r.get("trial")),
    reward: asNumber(r.get("reward")),
    domain: env === "retail" || env === "airline" ? env : guessDomain(traj),
    split,
    task: readTask(info?.get("task")),
    traj,
    controllers,
    models,
    endReason: END_REASONS.includes(reason as EndReason) ? (reason as EndReason) : null,
    maxAgentSteps: asNumber(pg?.get("max_agent_steps")),
    error,
  };
}

/** Parse a saved file: a single episode object or a list of them. */
export function parseRecordsFile(text: string, fileName = ""): EpisodeRecord[] {
  const value = parseJson(text.replace(/^\uFEFF/, ""));
  if (value instanceof PyDict && value.has("simulations")) {
    throw new Error("This looks like a τ²-bench results file. Only τ-bench (v1) trajectories can be replayed here.");
  }
  const list = Array.isArray(value) ? value : [value];
  const fileModels = modelsFromFileName(fileName);
  const records = list.map((v, i) => readRecord(v, i, fileModels)).filter((r): r is EpisodeRecord => r !== null);
  if (records.length === 0) throw new Error("No episodes found (expected a playground export or a tau-bench results file).");
  return records;
}

export interface ResolvedRecord {
  domain: Domain;
  task: Task;
  taskRef: { domain: DomainName; split: string; index: number };
  source: ReplaySource;
  maxAgentSteps: number;
}

export interface Loaders {
  loadDomain: (name: DomainName) => Promise<Domain>;
  loadTasks: (domain: DomainName, split: string) => Promise<Task[]>;
}

function sameTask(a: Task, b: Task): boolean {
  return a.instruction === b.instruction && a.user_id === b.user_id;
}

/** Same user and ground truth: the task, with its instruction reworded since the run (as tau-bench did for airline). */
function sameGroundTruth(a: Task, b: Task): boolean {
  return (
    a.user_id === b.user_id &&
    a.actions.length === b.actions.length &&
    a.actions.every((x, i) => x.name === b.actions[i].name && strictEqual(x.kwargs, b.actions[i].kwargs)) &&
    a.outputs.length === b.outputs.length &&
    a.outputs.every((o, i) => o === b.outputs[i])
  );
}

/** Count of agent actions (tool calls and replies) in a trajectory. */
function agentActions(traj: ChatMessage[]): number {
  return traj.filter((m) => m.role === "assistant").length;
}

/**
 * Find the dataset task a record belongs to (so the reward uses the exact
 * ground truth) and turn the record into something Episode.replay accepts.
 */
export async function resolveRecord(record: EpisodeRecord, loaders: Loaders): Promise<ResolvedRecord> {
  if (record.traj.length === 0) {
    throw new Error(
      record.error ? `This run has no trajectory; it failed with: ${record.error}` : "This record has no trajectory.",
    );
  }
  const domainName = record.domain;
  if (!domainName) throw new Error("Could not tell whether this trajectory is from the retail or the airline domain.");
  const domain = await loaders.loadDomain(domainName);

  let found: { split: string; index: number; task: Task } | null = null;
  const splits = DOMAIN_SPLITS[domainName];
  const preferred = record.split && splits.includes(record.split) ? record.split : "test";
  if (record.taskId !== null) {
    const tasks = await loaders.loadTasks(domainName, preferred);
    const t = tasks[record.taskId];
    if (t && (!record.task || sameTask(t, record.task))) found = { split: preferred, index: record.taskId, task: t };
  }
  if (!found && record.task) {
    for (const split of [preferred, ...splits.filter((s) => s !== preferred)]) {
      const tasks = await loaders.loadTasks(domainName, split);
      const index = tasks.findIndex((t) => sameTask(t, record.task!));
      if (index >= 0) {
        found = { split, index, task: tasks[index] };
        break;
      }
    }
  }
  if (!found && record.task && record.taskId !== null) {
    // The same task with its instruction reworded since the run: keep the wording the user simulator was
    // given (the reward is the same). Only after an exact match failed, since some tasks differ only in wording.
    const t = (await loaders.loadTasks(domainName, preferred))[record.taskId];
    if (t && sameGroundTruth(t, record.task)) found = { split: preferred, index: record.taskId, task: record.task };
  }
  if (!found) {
    if (!record.task) throw new Error("The task of this record is not in the dataset and the file does not include it.");
    found = { split: record.split ?? "file", index: record.taskId ?? record.position, task: record.task };
  }

  const actions = agentActions(record.traj);
  const maxAgentSteps = record.maxAgentSteps ?? (actions > 30 ? 0 : 30);
  return {
    domain,
    task: found.task,
    taskRef: { domain: domainName, split: found.split, index: found.index },
    source: {
      traj: record.traj,
      controllers: record.controllers,
      models: record.models,
      endReason: record.endReason,
      recordedReward: record.reward,
    },
    maxAgentSteps,
  };
}

/** Rebuild a record as a finished episode (see Episode.replay). No model is ever called for it. */
export async function replayRecord(record: EpisodeRecord, loaders: Loaders, label = ""): Promise<Episode> {
  const r = await resolveRecord(record, loaders);
  return Episode.replay(
    {
      domain: r.domain,
      task: r.task,
      taskRef: r.taskRef,
      controllers: { agent: record.controllers.agent ?? "model", user: record.controllers.user ?? "model" },
      modelFor: () => DEFAULT_MODEL,
      maxAgentSteps: r.maxAgentSteps,
      resolveEndpoint: () => {
        throw new Error("A replayed episode does not call models.");
      },
      replayOf: label,
    },
    r.source,
  );
}
