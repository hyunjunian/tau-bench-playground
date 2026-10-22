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
  /** Set while waiting to retry the current episode after a transient error. */
  autoRetry: { at: number; attempt: number } | null;
  /** Runs not started yet, in order. */
  pending: BatchRunRef[];
  current: BatchRunRef | null;
  results: BatchResult[];
  startedAt: string;
}

/** Starts the episode for one run and calls `onFinish` when it ends. */
export type LaunchRun = (run: BatchRunRef, onFinish: (episode: Episode) => void) => Episode;

export interface BatchOptions {
  /** Pause between episodes so the finished one stays on screen for a moment. */
  gapMs?: number;
  /** Wait before retrying after a transient error; grows with each attempt. */
  retryMs?: number;
}

/** Transient errors (network, 5xx, rate limits) are retried this many times per episode before waiting for the user. */
export const MAX_AUTO_RETRIES = 3;

export class BatchRun {
  private state: BatchState;
  private listeners = new Set<() => void>();
  private episode: Episode | null = null;
  /** Bumped to cancel a scheduled start of the next run. */
  private scheduled = 0;
  private unwatch: (() => void) | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private attempts = 0;
  private readonly gapMs: number;
  private readonly retryMs: number;

  constructor(
    plan: BatchPlan,
    private readonly launch: LaunchRun,
    options: BatchOptions = {},
  ) {
    this.gapMs = options.gapMs ?? 800;
    this.retryMs = options.retryMs ?? 30_000;
    const pending: BatchRunRef[] = [];
    for (let trial = 0; trial < plan.trials; trial++) {
      for (const index of plan.indices) pending.push({ index, trial });
    }
    this.state = {
      plan,
      status: "running",
      autoRetry: null,
      pending,
      current: null,
      results: [],
      startedAt: new Date().toISOString(),
    };
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
    this.scheduled++;
    this.cancelRetry();
    this.update({ status: "stopped" });
    this.episode?.abort();
  }

  resume() {
    if (this.state.status !== "stopped") return;
    this.scheduled++;
    this.update({ status: "running" });
    const ep = this.episode;
    if (ep && !ep.getState().done) {
      this.attempts = 0;
      ep.resume(); // stop() paused it
      if (ep.getState().error) ep.retry();
    } else this.next();
  }

  /** Forget the episode in flight without recording it (the user replaced it). */
  detach() {
    this.scheduled++;
    this.cancelRetry();
    this.unwatch?.();
    this.unwatch = null;
    this.episode = null;
    const current = this.state.current;
    this.update({
      status: this.state.status === "done" ? "done" : "stopped",
      current: null,
      pending: current ? [current, ...this.state.pending] : this.state.pending,
    });
  }

  private schedule() {
    const token = ++this.scheduled;
    const go = () => this.scheduled === token && this.next();
    // Background tabs throttle timers (down to once a minute), so don't pause there.
    if (typeof document !== "undefined" && document.hidden) void Promise.resolve().then(go);
    else setTimeout(go, this.gapMs);
  }

  private next() {
    if (this.state.status !== "running") return;
    const [head, ...rest] = this.state.pending;
    if (!head) {
      this.update({ status: "done", current: null });
      return;
    }
    this.update({ pending: rest, current: head });
    const episode = this.launch(head, (ep) => this.finished(ep, head));
    this.episode = episode;
    this.attempts = 0;
    this.unwatch?.();
    this.unwatch = episode.subscribe(() => this.watch(episode));
    this.watch(episode);
  }

  /** Retry transient errors on our own, so an unattended batch survives a flaky provider. */
  private watch(episode: Episode) {
    if (episode !== this.episode) return;
    const error = episode.getState().error;
    if (!error || !error.retryable || this.retryTimer || this.state.status !== "running") return;
    if (this.attempts >= MAX_AUTO_RETRIES) return;
    this.attempts++;
    const delay = this.retryMs * this.attempts;
    this.update({ autoRetry: { at: Date.now() + delay, attempt: this.attempts } });
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.update({ autoRetry: null });
      if (episode === this.episode && this.state.status === "running") episode.retry();
    }, delay);
  }

  private cancelRetry() {
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    if (this.state.autoRetry) this.update({ autoRetry: null });
  }

  private finished(episode: Episode, run: BatchRunRef) {
    // Runs are matched by identity, so a detached or replaced run is never recorded.
    if (this.state.current !== run) return;
    const s = episode.getState();
    if (!s.done) return;
    const record = { ...episode.toJSON(), trial: run.trial };
    const result: BatchResult = { ...run, reward: s.done.reward, reason: s.done.reason, steps: s.agentSteps, record };
    this.cancelRetry();
    this.unwatch?.();
    this.unwatch = null;
    this.episode = null;
    this.update({ results: [...this.state.results, result], current: null });
    if (this.state.pending.length === 0) {
      this.update({ status: "done" });
      return;
    }
    if (this.state.status === "running") this.schedule();
  }

  /** tau-bench style results file: one EnvRunResult-like record per episode. */
  exportResults(): Record<string, unknown>[] {
    return this.state.results.map((r) => r.record);
  }
}
