/** Random tool calls (valid and malformed) give the same observations and final database as Python tau-bench. */
import { describe, expect, it } from "vitest";
import { type PyDict, deltaCanonical, parseJson } from "../src/tau/py";
import { domainFor } from "./helpers";
import { type FuzzTrace, TIMEOUT, findReference, loadReference, matches, sha, step } from "./reference";

const fuzzPath = findReference("fuzz_traces");

describe.skipIf(!fuzzPath)("randomized tool calls match Python tau-bench", () => {
  const traces = loadReference<FuzzTrace>(fuzzPath);

  it(
    `replays ${traces.length} fuzz episodes`,
    () => {
      const failures: string[] = [];
      traces.forEach((trace, e) => {
        const domain = domainFor(trace.domain);
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
