/**
 * One playground episode: the tau-bench conversation loop where the agent and
 * the user can each be a human (via the UI) or a model.
 *
 * Model-controlled turns reproduce tau-bench exactly:
 *  - agent: ToolCallingAgent.solve (system prompt = policy wiki, tools = the
 *    domain's tool schemas, only the first tool call of a message is used,
 *    at most `maxAgentSteps` actions)
 *  - user: LLMUserSimulationEnv (system prompt with the task instruction, the
 *    conversation opens with "Hi! How can I help you today?", the episode
 *    ends when the user says ###STOP###)
 */
import { type ChatMessage, LlmError, type Usage, chatCompletion } from "../llm/client";
import type { Endpoint, ModelConfig } from "../llm/providers";
import {
  type Domain,
  type DomainName,
  EnvCrash,
  type RewardInfo,
  STOP_TOKEN,
  type Task,
  TauEnv,
} from "./env";
import { USER_OPENING, userSystemPrompt } from "./prompts";
import { PyDict, jsonDumps, parseJson, toPlain } from "./py";

export type Controller = "human" | "model";
export type Role = "agent" | "user";

export interface EpisodeSetup {
  domain: Domain;
  task: Task;
  taskRef: { domain: DomainName; split: string; index: number };
  /** Who controls each role; fixed for the episode. */
  controllers: Record<Role, Controller>;
  /** Model settings are read at every call, so a model can be swapped after an error. */
  modelFor: (role: Role) => ModelConfig;
  /** tau-bench's max_num_steps (30). 0 = unlimited. */
  maxAgentSteps: number;
  /** Start with model turns paused (step through manually). */
  startPaused?: boolean;
  resolveEndpoint: (cfg: ModelConfig) => Endpoint;
  onFinish?: (episode: Episode) => void;
}

export interface GenMeta {
  model: string;
  latencyMs: number;
  reasoning?: string;
  usage?: Usage;
}

export type Item =
  | { id: number; kind: "greeting"; text: string }
  | { id: number; kind: "user"; text: string; by: Controller; meta?: GenMeta }
  | { id: number; kind: "agent"; text: string; by: Controller; step: number; meta?: GenMeta }
  | {
      id: number;
      kind: "tool";
      by: Controller;
      step: number;
      name: string;
      args: string;
      observation: string;
      error: boolean;
      note: string | null;
      meta?: GenMeta;
    }
  | { id: number; kind: "notice"; level: "info" | "warn" | "error"; text: string };

type NewItem = Item extends infer T ? (T extends Item ? Omit<T, "id"> : never) : never;

export type EndReason = "user_stop" | "transfer" | "max_steps" | "manual" | "crash";

export interface EpisodeResult {
  reason: EndReason;
  /** The reward tau-bench would record (0 when the run hits max steps or crashes). */
  reward: number;
  /** Full reward breakdown (also computed for max_steps, for inspection). */
  info: RewardInfo | null;
  message?: string;
}

export interface EpisodeState {
  items: Item[];
  turn: Role | null;
  busy: boolean;
  retry: { until: number; reason: string } | null;
  error: { role: Role; message: string } | null;
  paused: boolean;
  agentSteps: number;
  done: EpisodeResult | null;
  usage: Record<Role, { calls: number; prompt: number; completion: number }>;
}

class StepError extends Error {}

function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

/** Tool-call arguments as tau-bench parses them (json.loads); "" is accepted as {}. */
export function parseToolArguments(text: string): PyDict {
  if (text.trim() === "") return new PyDict();
  const value = parseJson(text);
  if (!(value instanceof PyDict)) throw new Error("tool arguments must be a JSON object");
  return value;
}

export class Episode {
  readonly env: TauEnv;
  readonly startedAt = new Date();
  finishedAt: Date | null = null;
  /** The agent's conversation in OpenAI format (tau-bench's `messages`). */
  readonly agentMessages: ChatMessage[] = [];
  /** The user simulator's conversation (tau-bench's LLMUserSimulationEnv.messages). */
  readonly userMessages: ChatMessage[];

  private state: EpisodeState;
  private listeners = new Set<() => void>();
  private nextId = 1;
  private started = false;
  private pumping = false;
  private stepBudget = 0;
  private abortController: AbortController | null = null;
  private humanCalls = 0;

  constructor(readonly setup: EpisodeSetup) {
    this.env = new TauEnv(setup.domain, setup.task);
    this.userMessages = [
      { role: "system", content: userSystemPrompt(setup.task.instruction) },
      { role: "user", content: USER_OPENING },
    ];
    this.state = {
      items: [{ id: this.nextId++, kind: "greeting", text: USER_OPENING }],
      turn: null,
      busy: false,
      retry: null,
      error: null,
      paused: !!setup.startPaused,
      agentSteps: 0,
      done: null,
      usage: {
        agent: { calls: 0, prompt: 0, completion: 0 },
        user: { calls: 0, prompt: 0, completion: 0 },
      },
    };
  }

  // --- subscription (useSyncExternalStore) ------------------------------

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getState = (): EpisodeState => this.state;

  private update(patch: Partial<EpisodeState>) {
    this.state = { ...this.state, ...patch };
    for (const l of this.listeners) l();
  }

  private push(item: NewItem) {
    this.update({ items: [...this.state.items, { ...item, id: this.nextId++ } as Item] });
  }

  controllerOf(role: Role): Controller {
    return this.setup.controllers[role];
  }

  // --- lifecycle -----------------------------------------------------------

  start() {
    if (this.state.turn !== null || this.state.done) return;
    this.update({ turn: "user" });
    void this.pump();
  }

  pause() {
    this.update({ paused: true });
  }

  resume() {
    this.update({ paused: false });
    void this.pump();
  }

  /** Run exactly one model action while paused. */
  stepOnce() {
    if (this.state.busy || this.state.done) return;
    this.stepBudget = 1;
    void this.pump();
  }

  /** Cancel the request in flight; model turns stay paused until resumed. */
  abort() {
    this.abortController?.abort();
    this.update({ paused: true });
  }

  retry() {
    if (!this.state.error) return;
    this.update({ error: null });
    void this.pump();
  }

  /** End the conversation now and score the current database state. */
  endNow() {
    if (this.state.done) return;
    this.abortController?.abort();
    this.finish("manual");
  }

  // --- human actions -------------------------------------------------------

  sendUserMessage(text: string) {
    if (this.state.turn !== "user" || this.controllerOf("user") !== "human" || this.state.done) return;
    this.userMessages.push({ role: "assistant", content: text });
    this.applyUserMessage(text, "human");
    void this.pump();
  }

  sendAgentReply(text: string) {
    if (!this.canHumanAgentAct()) return;
    if (this.stepLimitReached()) return this.finish("max_steps");
    this.applyAgentRespond(text, { role: "assistant", content: text }, "human");
    void this.pump();
  }

  /** Returns the observation (or null if the episode ended). */
  callAgentTool(name: string, kwargs: PyDict): string | null {
    if (!this.canHumanAgentAct()) return null;
    if (this.stepLimitReached()) {
      this.finish("max_steps");
      return null;
    }
    const args = jsonDumps(kwargs);
    const callId = `call_human_${++this.humanCalls}`;
    const message: ChatMessage = {
      role: "assistant",
      content: null,
      tool_calls: [{ id: callId, type: "function", function: { name, arguments: args } }],
    };
    const observation = this.applyAgentTool(name, kwargs, args, message, callId, "human", null);
    void this.pump();
    return observation;
  }

  private canHumanAgentAct(): boolean {
    return this.state.turn === "agent" && this.controllerOf("agent") === "human" && !this.state.done;
  }

  private stepLimitReached(): boolean {
    const max = this.setup.maxAgentSteps;
    return max > 0 && this.state.agentSteps >= max;
  }

  // --- model turns ------------------------------------------------------

  private async pump() {
    if (this.pumping) return;
    this.pumping = true;
    try {
      while (!this.state.done && !this.state.error && this.state.turn) {
        const role = this.state.turn;
        if (this.controllerOf(role) === "human") break;
        if (this.state.paused) {
          if (this.stepBudget <= 0) break;
          this.stepBudget--;
        }
        const ok = role === "agent" ? await this.runModelAgent() : await this.runModelUser();
        if (!ok) break;
      }
    } finally {
      this.pumping = false;
      this.stepBudget = 0;
    }
  }

  private async generate(role: Role, cfg: ModelConfig, messages: ChatMessage[], withTools: boolean) {
    const endpoint = this.setup.resolveEndpoint(cfg);
    this.abortController = new AbortController();
    this.update({ busy: true, retry: null });
    try {
      const res = await chatCompletion({
        endpoint,
        model: cfg.model,
        messages,
        tools: withTools ? this.setup.domain.toolsInfo : undefined,
        temperature: cfg.temperature,
        reasoningEffort: cfg.reasoningEffort,
        signal: this.abortController.signal,
        onRetry: ({ waitMs, reason }) => this.update({ retry: { until: Date.now() + waitMs, reason } }),
      });
      const u = this.state.usage[role];
      this.update({
        usage: {
          ...this.state.usage,
          [role]: {
            calls: u.calls + 1,
            prompt: u.prompt + (res.usage?.prompt_tokens ?? 0),
            completion: u.completion + (res.usage?.completion_tokens ?? 0),
          },
        },
      });
      const meta: GenMeta = { model: cfg.model, latencyMs: res.latencyMs, reasoning: res.reasoning, usage: res.usage };
      return { message: res.message, meta };
    } finally {
      this.abortController = null;
      this.update({ busy: false, retry: null });
    }
  }

  private fail(role: Role, e: unknown): false {
    if (isAbort(e)) {
      if (!this.state.done) this.update({ paused: true });
      return false;
    }
    const message = e instanceof LlmError || e instanceof StepError || e instanceof Error ? e.message : String(e);
    this.update({ error: { role, message } });
    return false;
  }

  private async runModelAgent(): Promise<boolean> {
    if (this.stepLimitReached()) {
      this.finish("max_steps");
      return false;
    }
    try {
      const { message, meta } = await this.generate("agent", this.setup.modelFor("agent"), this.agentMessages, true);
      if (this.state.done) return false;
      const call = message.tool_calls?.[0];
      if (call) {
        let kwargs: PyDict;
        try {
          kwargs = parseToolArguments(call.function.arguments);
        } catch (e) {
          throw new StepError(
            `The agent model called ${call.function.name} with arguments that are not a JSON object ` +
              `(${e instanceof Error ? e.message : e}): ${call.function.arguments.slice(0, 300)}. ` +
              "tau-bench would count this run as a failure; retry to sample again or end the episode.",
          );
        }
        const stored: ChatMessage = { ...message, tool_calls: [call] };
        this.applyAgentTool(call.function.name, kwargs, call.function.arguments, stored, call.id, "model", message.content, meta);
        return true;
      }
      const content = message.content ?? "";
      if (!content.trim()) {
        throw new StepError("The agent model returned an empty message (no text and no tool call).");
      }
      this.applyAgentRespond(content, message, "model", meta);
      return true;
    } catch (e) {
      return this.fail("agent", e);
    }
  }

  private async runModelUser(): Promise<boolean> {
    try {
      const { message, meta } = await this.generate("user", this.setup.modelFor("user"), this.userMessages, false);
      if (this.state.done) return false;
      const content = message.content ?? "";
      if (!content.trim()) throw new StepError("The user model returned an empty message.");
      this.userMessages.push({ role: "assistant", content });
      this.applyUserMessage(content, "model", meta);
      return true;
    } catch (e) {
      return this.fail("user", e);
    }
  }

  // --- transitions shared by human and model turns ----------------------

  private applyUserMessage(text: string, by: Controller, meta?: GenMeta) {
    this.push({ kind: "user", text, by, meta });
    if (!this.started) {
      // env.reset(): the first user message starts the agent's conversation.
      this.started = true;
      this.agentMessages.push({ role: "system", content: this.setup.domain.wiki }, { role: "user", content: text });
      this.update({ turn: "agent" });
      return;
    }
    this.agentMessages.push({ role: "user", content: text });
    if (text.includes(STOP_TOKEN)) {
      this.finish("user_stop");
      return;
    }
    this.update({ turn: "agent" });
  }

  private applyAgentRespond(text: string, message: ChatMessage, by: Controller, meta?: GenMeta) {
    this.env.recordRespond(text);
    const step = this.state.agentSteps + 1;
    this.agentMessages.push(message);
    this.userMessages.push({ role: "user", content: text });
    this.update({ agentSteps: step });
    this.push({ kind: "agent", text, by, step, meta });
    this.update({ turn: "user" });
  }

  private applyAgentTool(
    name: string,
    kwargs: PyDict,
    args: string,
    message: ChatMessage,
    callId: string,
    by: Controller,
    note: string | null,
    meta?: GenMeta,
  ): string | null {
    const step = this.state.agentSteps + 1;
    let observation: string;
    let done: boolean;
    try {
      ({ observation, done } = this.env.stepTool({ name, kwargs }));
    } catch (e) {
      if (e instanceof EnvCrash) {
        this.update({ agentSteps: step });
        this.push({ kind: "tool", by, step, name, args, observation: "None", error: true, note, meta });
        this.finish("crash", e.message);
        return null;
      }
      throw e;
    }
    this.agentMessages.push(message, { role: "tool", tool_call_id: callId, name, content: observation });
    this.update({ agentSteps: step });
    this.push({
      kind: "tool",
      by,
      step,
      name,
      args,
      observation,
      error: observation.startsWith("Error") || observation.startsWith("Unknown action"),
      note: note && note.trim() ? note : null,
      meta,
    });
    if (done) this.finish("transfer");
    return observation;
  }

  private finish(reason: EndReason, message?: string) {
    if (this.state.done) return;
    const info = reason === "crash" ? null : this.env.calculateReward();
    const reward = reason === "crash" || reason === "max_steps" ? 0 : (info?.reward ?? 0);
    this.finishedAt = new Date();
    this.update({ done: { reason, reward, info, message }, turn: null, busy: false, retry: null });
    this.setup.onFinish?.(this);
  }

  // --- export -----------------------------------------------------------

  /** Roughly tau-bench's EnvRunResult, plus playground metadata. */
  /** Models that actually produced messages for a role. */
  modelsUsed(role: Role): string[] {
    const kinds = role === "agent" ? ["agent", "tool"] : ["user"];
    const models = new Set<string>();
    for (const item of this.state.items) {
      if (kinds.includes(item.kind) && "meta" in item && item.meta) models.add(item.meta.model);
    }
    return Array.from(models);
  }

  toJSON() {
    const { task, taskRef } = this.setup;
    const done = this.state.done;
    const describe = (role: Role) =>
      this.controllerOf(role) === "human"
        ? { controller: "human" }
        : { controller: "model", models: this.modelsUsed(role), config: this.setup.modelFor(role) };
    return {
      task_id: taskRef.index,
      reward: done?.reward ?? null,
      info: {
        task: {
          user_id: task.user_id,
          instruction: task.instruction,
          actions: task.actions.map((a) => ({ name: a.name, kwargs: toPlain(a.kwargs) })),
          outputs: task.outputs,
        },
        reward_info: done?.info
          ? {
              reward: done.info.reward,
              r_actions: done.info.rActions,
              r_outputs: done.info.rOutputs,
              outputs: Object.fromEntries(done.info.outputs.map((o) => [o.output, o.found])),
            }
          : null,
        end_reason: done?.reason ?? null,
      },
      traj: this.agentMessages,
      user_traj: this.userMessages,
      actions: this.env.actions.map((a) => ({ name: a.name, kwargs: toPlain(a.kwargs) })),
      playground: {
        env: taskRef.domain,
        task_split: taskRef.split,
        agent: describe("agent"),
        user: describe("user"),
        max_agent_steps: this.setup.maxAgentSteps,
        agent_steps: this.state.agentSteps,
        usage: this.state.usage,
        started_at: this.startedAt.toISOString(),
        finished_at: this.finishedAt?.toISOString() ?? null,
      },
    };
  }
}
