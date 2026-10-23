/** Rewards of recorded trajectories match tau-bench's Env.calculate_reward. */
import { describe, expect, it } from "vitest";
import { TauEnv } from "../src/tau/env";
import { type PyDict, parseJson } from "../src/tau/py";
import { domainFor, tasksFor } from "./helpers";
import { type RewardCase, TIMEOUT, findReference, loadReference } from "./reference";

const rewardPath = findReference("reward_cases");

describe.skipIf(!rewardPath)("rewards match Env.calculate_reward", () => {
  const cases = loadReference<RewardCase>(rewardPath);

  it(
    `scores ${cases.length} trajectories`,
    () => {
      const failures: string[] = [];
      for (const c of cases) {
        const env = new TauEnv(domainFor(c.domain), tasksFor(c.domain, c.split)[c.index]);
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
