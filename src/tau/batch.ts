/**
 * Runs model-vs-model episodes over a range of tasks, one after another,
 * repeating the whole range for each trial like tau-bench's run.py
 * (--num-trials). Each finished episode is kept as its JSON export so the
 * results can be downloaded or reopened without holding every database copy.
 */
import type { DomainName } from "./env";
import type { EndReason, Episode } from "./episode";

export interface BatchPlan {
  domain: DomainName;
  split: string;
  indices: number[];
  trials: number;
}

export interface BatchRunRef {
  index: number;
  /** 0-based, like tau-bench's `trial`. */
  trial: number;
}

export interface BatchResult extends BatchRunRef {
  reward: number;
  reason: EndReason;
  steps: number;
  /** Episode.toJSON() plus `trial`. */
  record: Record<string, unknown>;
}

export interface BatchState {
  plan: BatchPlan;
  status: "running" | "stopped" | "done";
  /** Runs not started yet, in order. */
  pending: BatchRunRef[];
  current: BatchRunRef | null;
  results: BatchResult[];
  startedAt: string;
}

/** Starts the episode for one run and calls `onFinish` when it ends. */
export type LaunchRun = (run: BatchRunRef, onFinish: (episode: Episode) => void) => Episode;

/** Pause between episodes so the finished one stays on screen for a moment. */
const GAP_MS = 800;

export class BatchRun {
  private state: BatchState;
  private listeners = new Set<() => void>();
  private episode: Episode | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    plan: BatchPlan,
    private readonly launch: LaunchRun,
    private readonly gapMs = GAP_MS,
  ) {
    const pending: BatchRunRef[] = [];
    for (let trial = 0; trial < plan.trials; trial++) {
      for (const index of plan.indices) pending.push({ index, trial });
    }
    this.state = { plan, status: "running", pending, current: null, results: [], startedAt: new Date().toISOString() };
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getState = (): BatchState => this.state;

  private update(patch: Partial<BatchState>) {
    this.state = { ...this.state, ...patch };
    for (const l of this.listeners) l();
  }

  get total(): number {
    return this.state.plan.indices.length * this.state.plan.trials;
  }

  start() {
    if (this.state.current || this.state.results.length > 0) return;
    this.next();
  }

  /** Stop after aborting the request in flight; the current episode stays open and paused. */
  stop() {
    if (this.state.status !== "running") return;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.update({ status: "stopped" });
    this.episode?.abort();
  }

  resume() {
    if (this.state.status !== "stopped") return;
    this.update({ status: "running" });
    if (this.episode && !this.episode.getState().done) this.episode.resume();
    else this.next();
  }

  /** Forget the episode in flight without recording it (the user replaced it). */
  detach() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.episode = null;
    const current = this.state.current;
    this.update({
      status: this.state.status === "done" ? "done" : "stopped",
      current: null,
      pending: current ? [current, ...this.state.pending] : this.state.pending,
    });
  }

  private next() {
    this.timer = null;
    if (this.state.status !== "running") return;
    const [head, ...rest] = this.state.pending;
    if (!head) {
      this.update({ status: "done", current: null });
      return;
    }
    this.update({ pending: rest, current: head });
    this.episode = this.launch(head, (ep) => this.finished(ep, head));
  }

  private finished(episode: Episode, run: BatchRunRef) {
    // Runs are matched by identity, so a detached or replaced run is never recorded.
    if (this.state.current !== run) return;
    const s = episode.getState();
    if (!s.done) return;
    const record = { ...episode.toJSON(), trial: run.trial };
    const result: BatchResult = { ...run, reward: s.done.reward, reason: s.done.reason, steps: s.agentSteps, record };
    this.episode = null;
    this.update({ results: [...this.state.results, result], current: null });
    if (this.state.pending.length === 0) {
      this.update({ status: "done" });
      return;
    }
    if (this.state.status === "running") this.timer = setTimeout(() => this.next(), this.gapMs);
  }

  /** tau-bench style results file: one EnvRunResult-like record per episode. */
  exportResults(): Record<string, unknown>[] {
    return this.state.results.map((r) => r.record);
  }
}
