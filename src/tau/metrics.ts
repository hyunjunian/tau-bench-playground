/**
 * Aggregate metrics from tau-bench's run.py (display_metrics): average reward
 * and pass^k, the chance that k independent trials of a task all succeed.
 */

export interface TrialOutcome {
  /** Anything that identifies the task, e.g. "retail/test/12". */
  task: string;
  reward: number;
}

export function isSuccess(reward: number): boolean {
  return 1 - 1e-6 <= reward && reward <= 1 + 1e-6;
}

/** C(n, k) as a float (n is a handful of trials). */
export function comb(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}

export function averageReward(outcomes: TrialOutcome[]): number | null {
  if (outcomes.length === 0) return null;
  return outcomes.reduce((s, o) => s + o.reward, 0) / outcomes.length;
}

/**
 * pass^k = mean over tasks of C(c, k) / C(n, k), where a task was tried n
 * times and succeeded c times. tau-bench assumes every task has the same n;
 * here each task uses its own n and k only goes up to the smallest n, so a
 * partly finished run still gives an unbiased estimate.
 */
export function passHatK(outcomes: TrialOutcome[]): { k: number; value: number }[] {
  const perTask = new Map<string, { n: number; c: number }>();
  for (const o of outcomes) {
    const t = perTask.get(o.task) ?? { n: 0, c: 0 };
    t.n += 1;
    if (isSuccess(o.reward)) t.c += 1;
    perTask.set(o.task, t);
  }
  if (perTask.size === 0) return [];
  const tasks = Array.from(perTask.values());
  const maxK = Math.min(...tasks.map((t) => t.n));
  const out: { k: number; value: number }[] = [];
  for (let k = 1; k <= maxK; k++) {
    const sum = tasks.reduce((s, t) => s + comb(t.c, k) / comb(t.n, k), 0);
    out.push({ k, value: sum / tasks.length });
  }
  return out;
}
