/** Every task's ground-truth actions give the same observations and final database as Python tau-bench. */
import { describe, expect, it } from "vitest";
import { deltaCanonical } from "../src/tau/py";
import { domainFor, tasksFor } from "./helpers";
import { type GtTrace, TIMEOUT, findReference, loadReference, matches, sha, step } from "./reference";

const gtPath = findReference("gt_traces");

describe.skipIf(!gtPath)("ground-truth task replays match Python tau-bench", () => {
  const traces = loadReference<GtTrace>(gtPath);

  it(
    `replays ${traces.length} tasks`,
    () => {
      const failures: string[] = [];
      for (const trace of traces) {
        const domain = domainFor(trace.domain);
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
