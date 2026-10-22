/**
 * Reading saved episodes: the playground's own JSON export (one episode, or a
 * batch results file with many) and the results files written by tau-bench's
 * run.py (a list of EnvRunResult: task_id, reward, info, traj, trial).
 */
import type { ChatMessage } from "../llm/client";
import { DOMAIN_SPLITS, type Domain, type DomainName, type Task, parseTasks } from "./env";
import type { Controller, EndReason, ReplaySource, Role } from "./episode";
import { PyDict, PyFloat, type PyValue, jsonDumps, parseJson, toPlain } from "./py";

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

const END_REASONS: EndReason[] = ["user_stop", "transfer", "max_steps", "manual", "crash", "incomplete"];
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
    .map((m) => toPlain(m) as ChatMessage);
}

/** Model names from a tau-bench results file name (run.py's checkpoint path). */
export function modelsFromFileName(name: string): Partial<Record<Role, string>> {
  const base = name.split(/[\\/]/).pop() ?? name;
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
