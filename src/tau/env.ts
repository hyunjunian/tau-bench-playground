/**
 * Port of tau_bench/envs/base.py: tool dispatch, terminate tools and the
 * reward (database state must equal the ground-truth replay, and every
 * required output must appear in one of the agent's messages to the user).
 */
import { PyDict, PyException, type PyValue, TypeError_, deepCopy, parseJson, strictEqual } from "./py";
import { AIRLINE_TOOLS, RETAIL_TOOLS, type ToolImpl } from "./tools";

export type DomainName = "retail" | "airline";

export const DOMAIN_SPLITS: Record<DomainName, string[]> = {
  retail: ["test", "train", "dev"],
  airline: ["test"],
};

export const RESPOND_ACTION_NAME = "respond";
export const STOP_TOKEN = "###STOP###";
/** Same for both domains (MockRetailDomainEnv / MockAirlineDomainEnv). */
export const TERMINATE_TOOLS = ["transfer_to_human_agents"];

export interface JsonSchema {
  type?: string;
  description?: string;
  enum?: string[];
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  required?: string[];
}

export interface ToolInfo {
  type: "function";
  function: { name: string; description: string; parameters: JsonSchema };
}

export interface ToolSpec {
  class_name: string;
  qualified_name: string;
  params: string[];
  info: ToolInfo;
}

export interface Action {
  name: string;
  kwargs: PyDict;
}

export interface Task {
  user_id: string;
  instruction: string;
  actions: Action[];
  outputs: string[];
}

/** Everything static about a domain: initial database, tools, policy. */
export class Domain {
  readonly pristine: PyDict;
  readonly toolSpecs: Map<string, ToolSpec>;
  readonly toolsInfo: ToolInfo[];
  private readonly impls: Record<string, ToolImpl>;
  /** Only the last task is kept: scoring the same task again is common, and each database copy is large. */
  private lastGroundTruth: { task: Task; data: PyDict } | null = null;

  constructor(
    readonly name: DomainName,
    dbText: string,
    tools: ToolSpec[],
    readonly wiki: string,
  ) {
    this.pristine = parseJson(dbText) as PyDict;
    this.toolSpecs = new Map(tools.map((t) => [t.info.function.name, t]));
    this.toolsInfo = tools.map((t) => t.info);
    this.impls = name === "retail" ? RETAIL_TOOLS : AIRLINE_TOOLS;
    for (const t of tools) {
      if (!this.impls[t.info.function.name]) throw new Error(`No implementation for tool ${t.info.function.name}`);
    }
  }

  freshData(): PyDict {
    return deepCopy(this.pristine);
  }

  hasTool(name: string): boolean {
    return this.toolSpecs.has(name);
  }

  /** The database after replaying a task's ground-truth actions. Shared between calls: treat it as read-only. */
  groundTruthData(task: Task): PyDict {
    if (this.lastGroundTruth?.task === task) return this.lastGroundTruth.data;
    const data = this.freshData();
    for (const action of task.actions) {
      if (TERMINATE_TOOLS.includes(action.name) || action.name === RESPOND_ACTION_NAME) continue;
      if (this.hasTool(action.name)) this.invokeTool(data, action.name, deepCopy(action.kwargs));
    }
    this.lastGroundTruth = { task, data };
    return data;
  }

  /**
   * `tool.invoke(data=data, **kwargs)` wrapped in the env's try/except.
   * Returns null where the Python tool returns None (tau-bench then crashes).
   */
  invokeTool(data: PyDict, name: string, kwargs: PyDict): string | null {
    const spec = this.toolSpecs.get(name)!;
    try {
      const args = bindArguments(spec, kwargs);
      return this.impls[name](data, ...args);
    } catch (e) {
      if (e instanceof PyException) return `Error: ${e.message}`;
      // A JS error here means the port diverged from Python; surface it like an exception.
      console.error(`tool ${name} raised a non-Python error`, e);
      return `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
}

function quoteList(names: string[]): string {
  const q = names.map((n) => `'${n}'`);
  if (q.length === 1) return q[0];
  if (q.length === 2) return `${q[0]} and ${q[1]}`;
  return `${q.slice(0, -1).join(", ")}, and ${q[q.length - 1]}`;
}

/** Python's call-time argument binding for `invoke(data, <params>)` with **kwargs. */
function bindArguments(spec: ToolSpec, kwargs: PyDict): PyValue[] {
  if (kwargs.has("data")) {
    throw TypeError_(`${spec.qualified_name}() got multiple values for keyword argument 'data'`);
  }
  for (const key of kwargs.keys()) {
    if (!spec.params.includes(key)) {
      throw TypeError_(`${spec.class_name}.invoke() got an unexpected keyword argument '${key}'`);
    }
  }
  const missing = spec.params.filter((p) => !kwargs.has(p));
  if (missing.length > 0) {
    const plural = missing.length > 1 ? "s" : "";
    throw TypeError_(
      `${spec.class_name}.invoke() missing ${missing.length} required positional argument${plural}: ${quoteList(missing)}`,
    );
  }
  return spec.params.map((p) => kwargs.get(p)!);
}

/** Tasks are parsed with the Python-compatible JSON parser so int/float kwargs survive. */
export function parseTasks(text: string): Task[] {
  const list = parseJson(text) as PyDict[];
  return list.map((t) => ({
    user_id: t.get("user_id") as string,
    instruction: t.get("instruction") as string,
    outputs: (t.get("outputs") as string[]).slice(),
    actions: (t.get("actions") as PyDict[]).map((a) => ({
      name: a.get("name") as string,
      kwargs: a.get("kwargs") as PyDict,
    })),
  }));
}

export class EnvCrash extends Error {}

export interface StepResult {
  observation: string;
  done: boolean;
}

export interface RewardInfo {
  reward: number;
  /** Database matches the ground-truth end state. */
  rActions: boolean;
  /** null when the task has no required outputs. */
  rOutputs: number | null;
  outputs: { output: string; found: boolean }[];
  /** Database after replaying the ground-truth actions. */
  gtData: PyDict;
}

/** One episode's environment state (tau_bench.envs.base.Env without the user simulator). */
export class TauEnv {
  data: PyDict;
  readonly actions: Action[] = [];

  constructor(
    readonly domain: Domain,
    readonly task: Task,
  ) {
    this.data = domain.freshData();
  }

  /** Run a tool action. Respond actions are handled by the episode (they need the user). */
  stepTool(action: Action): StepResult {
    this.actions.push(action);
    let observation: string | null;
    if (this.domain.hasTool(action.name)) {
      observation = this.domain.invokeTool(this.data, action.name, deepCopy(action.kwargs));
      if (observation === null) {
        throw new EnvCrash(
          `${action.name} returned None. The original tau-bench run would crash here (EnvResponse requires a string observation).`,
        );
      }
      return { observation, done: TERMINATE_TOOLS.includes(action.name) };
    }
    observation = `Unknown action ${action.name}`;
    return { observation, done: false };
  }

  recordRespond(content: string): void {
    this.actions.push({ name: RESPOND_ACTION_NAME, kwargs: new PyDict([["content", content]]) });
  }

  calculateReward(): RewardInfo {
    const gtData = this.domain.groundTruthData(this.task);
    // tau-bench compares consistent_hash(to_hashable(data)); strictEqual distinguishes exactly the same things.
    const rActions = strictEqual(this.data, gtData);
    let reward = rActions ? 1 : 0;

    let rOutputs: number | null = null;
    const outputs: { output: string; found: boolean }[] = [];
    if (this.task.outputs.length > 0) {
      rOutputs = 1;
      for (const output of this.task.outputs) {
        const found = this.actions.some(
          (a) =>
            a.name === RESPOND_ACTION_NAME &&
            String(a.kwargs.get("content") ?? "")
              .toLowerCase()
              .replaceAll(",", "")
              .includes(output.toLowerCase()),
        );
        outputs.push({ output, found });
        if (!found) {
          rOutputs = 0;
          reward = 0;
        }
      }
    }
    return { reward, rActions, rOutputs, outputs, gtData };
  }
}
