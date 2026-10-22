/**
 * The playground's conversation loop, batch runner and replay of saved
 * trajectories, driven by a fake OpenAI-compatible endpoint whose agent
 * replays each task's ground-truth actions.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ChatMessage } from "../src/llm/client";
import { BatchRun, MAX_AUTO_RETRIES } from "../src/tau/batch";
import { compareWrites } from "../src/tau/diff";
import { type Domain, type DomainName, type Task, TauEnv } from "../src/tau/env";
import { Episode, type EpisodeSetup, parseToolArguments } from "../src/tau/episode";
import { averageReward, comb, passHatK } from "../src/tau/metrics";
import { jsonDumps } from "../src/tau/py";
import { modelsFromFileName, parseRecordsFile, replayRecord, resolveRecord } from "../src/tau/records";
import { domainFor, tasksFor } from "./helpers";

const domains = { retail: domainFor("retail"), airline: domainFor("airline") };
const loaders = {
  loadDomain: async (name: DomainName) => domains[name],
  loadTasks: async (domain: DomainName, split: string) => tasksFor(domain, split),
};

/** What the agent says once every ground-truth action is done (contains the required outputs). */
const finalReply = (task: Task) => `All done. ${task.outputs.join(", ")}`.trim();

/** Fake chat completions: the agent replays the ground truth, the user stops after the final reply. */
function fakeProvider(taskFor: (messages: ChatMessage[]) => Task) {
  return vi.fn(async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as { messages: ChatMessage[]; tools?: unknown[] };
    const msgs = body.messages;
    let message: Record<string, unknown>;
    if (body.tools) {
      const task = taskFor(msgs);
      const done = msgs.filter((m) => m.role === "tool").length;
      const a = task.actions[done];
      message = a
        ? {
            role: "assistant",
            content: null,
            tool_calls: [{ id: `call_${done}`, type: "function", function: { name: a.name, arguments: jsonDumps(a.kwargs) } }],
          }
        : { role: "assistant", content: finalReply(task) };
    } else {
      const turns = msgs.filter((m) => m.role === "assistant").length;
      message = { role: "assistant", content: turns === 0 ? "Hi, I need some help." : "Thanks! ###STOP###" };
    }
    const payload = { choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 3 } };
    return new Response(JSON.stringify(payload), { status: 200, headers: { "Content-Type": "application/json" } });
  });
}

/** The agent always calls a tool with arguments that are not JSON. */
function brokenAgentProvider() {
  return vi.fn(async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as { tools?: unknown };
    const message = body.tools
      ? {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "x", type: "function", function: { name: "get_order_details", arguments: "{oops" } }],
        }
      : { role: "assistant", content: "Hello" };
    return new Response(JSON.stringify({ choices: [{ message }] }), { status: 200 });
  });
}

function setupFor(domain: Domain, task: Task, index: number, extra: Partial<EpisodeSetup> = {}): EpisodeSetup {
  return {
    domain,
    task,
    taskRef: { domain: domain.name, split: "test", index },
    controllers: { agent: "model", user: "model" },
    modelFor: () => ({ provider: "custom", model: "fake-model", temperature: null, reasoningEffort: "" }),
    maxAgentSteps: 30,
    resolveEndpoint: () => ({ provider: "custom", baseUrl: "http://fake.test/v1", apiKey: "" }),
    ...extra,
  };
}

function untilSettled(ep: Episode): Promise<void> {
  return new Promise((resolve) => {
    const check = () => {
      const s = ep.getState();
      if (s.done || s.error) {
        unsubscribe();
        resolve();
      }
    };
    const unsubscribe = ep.subscribe(check);
    check();
  });
}

/** A tau-bench results record (tool-calling agent) that performs the ground truth. */
function toolCallingRecord(domain: Domain, task: Task, index: number): Record<string, unknown> {
  const env = new TauEnv(domain, task);
  const traj: Record<string, unknown>[] = [
    { role: "system", content: domain.wiki },
    { role: "user", content: "Hi, I need some help." },
  ];
  let transferred = false;
  task.actions.forEach((a, i) => {
    const { observation, done } = env.stepTool({ name: a.name, kwargs: a.kwargs });
    const call = { id: `call_${i}`, type: "function", function: { name: a.name, arguments: jsonDumps(a.kwargs) } };
    traj.push({ role: "assistant", content: null, tool_calls: [call], function_call: null });
    traj.push({ role: "tool", tool_call_id: call.id, name: a.name, content: observation });
    transferred ||= done;
  });
  if (!transferred) {
    traj.push({ role: "assistant", content: finalReply(task), tool_calls: null });
    traj.push({ role: "user", content: "###STOP###" });
  }
  return {
    task_id: index,
    reward: 1.0,
    info: { task: JSON.parse(taskJson(task)), source: "user" },
    traj,
    trial: 0,
  };
}

function taskJson(task: Task): string {
  const actions = task.actions.map((a) => `{"name": ${JSON.stringify(a.name)}, "kwargs": ${jsonDumps(a.kwargs)}}`);
  return `{"user_id": ${JSON.stringify(task.user_id)}, "actions": [${actions.join(", ")}], "instruction": ${JSON.stringify(
    task.instruction,
  )}, "outputs": ${JSON.stringify(task.outputs)}}`;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("compareWrites", () => {
  it("explains a missed write by the closest call, and lists writes the ground truth does not have", () => {
    const task = tasksFor("retail", "test")[0];
    const isWrite = (name: string) => !/^(get_|find_|list_|calculate|think)/.test(name);
    const exchange = task.actions.find((a) => a.name === "exchange_delivered_order_items")!;
    const wrongItems = parseToolArguments(jsonDumps(exchange.kwargs).replace("7747408585", "1111111111"));
    const agent = [
      ...task.actions.filter((a) => !isWrite(a.name)),
      { name: "exchange_delivered_order_items", kwargs: wrongItems },
      { name: "cancel_pending_order", kwargs: parseToolArguments('{"order_id": "#W1", "reason": "no longer needed"}') },
      { name: "respond", kwargs: parseToolArguments('{"content": "done"}') },
    ];
    const { checks, extra } = compareWrites(task.actions, agent, isWrite);
    expect(checks).toHaveLength(1);
    expect(checks[0].matched).toBe(false);
    expect(checks[0].closest?.diffs).toEqual([{ path: "new_item_ids[1]", expected: "7747408585", actual: "1111111111" }]);
    expect(extra.map((a) => a.name)).toEqual(["exchange_delivered_order_items", "cancel_pending_order"]);

    const perfect = compareWrites(task.actions, task.actions, isWrite);
    expect(perfect.checks.every((c) => c.matched)).toBe(true);
    expect(perfect.extra).toEqual([]);
    expect(compareWrites(task.actions, [], isWrite).checks[0]).toMatchObject({ matched: false, closest: null });
  });
});

describe("metrics", () => {
  it("computes pass^k like tau-bench", () => {
    expect(comb(4, 2)).toBe(6);
    expect(comb(1, 2)).toBe(0);
    const outcomes = [
      { task: "a", reward: 1 },
      { task: "a", reward: 1 },
      { task: "b", reward: 1 },
      { task: "b", reward: 0 },
    ];
    expect(averageReward(outcomes)).toBe(0.75);
    expect(passHatK(outcomes)).toEqual([
      { k: 1, value: 0.75 },
      { k: 2, value: 0.5 },
    ]);
    // k stops at the smallest number of trials of any task
    expect(passHatK([...outcomes, { task: "c", reward: 1 }]).map((p) => p.k)).toEqual([1]);
    expect(passHatK([])).toEqual([]);
  });
});

describe("Episode with model players", () => {
  it.each([
    ["retail", 0],
    ["retail", 5],
    ["airline", 0],
    ["airline", 7],
  ] as const)("replaying %s test #%i's ground truth through the loop scores 1", async (name, index) => {
    const domain = domains[name];
    const task = tasksFor(name, "test")[index];
    const fetch = fakeProvider(() => task);
    vi.stubGlobal("fetch", fetch);
    const ep = new Episode(setupFor(domain, task, index));
    ep.start();
    await untilSettled(ep);
    const s = ep.getState();
    expect(s.error).toBeNull();
    expect(s.done?.reward).toBe(1);
    expect(s.agentSteps).toBe(task.actions.length + (s.done?.reason === "transfer" ? 0 : 1));
    // tau-bench's message layout: system, first user message, then assistant/tool pairs
    expect(ep.agentMessages[0]).toEqual({ role: "system", content: domain.wiki });
    expect(ep.agentMessages[1]).toEqual({ role: "user", content: "Hi, I need some help." });
    expect(s.usage.agent.calls + s.usage.user.calls).toBe(fetch.mock.calls.length);
  });

  it("drafts the next move for a human player without acting on it", async () => {
    const domain = domains.retail;
    const task = tasksFor("retail", "test")[0];
    vi.stubGlobal(
      "fetch",
      fakeProvider(() => task),
    );
    const ep = new Episode(setupFor(domain, task, 0, { controllers: { agent: "human", user: "model" } }));
    ep.start();
    await new Promise<void>((resolve) => {
      const check = () => ep.getState().turn === "agent" && !ep.getState().busy && resolve();
      ep.subscribe(check);
      check();
    });
    const before = { items: ep.getState().items.length, messages: ep.agentMessages.length, actions: ep.env.actions.length };
    const draft = await ep.draft("agent");
    expect(draft.tool).toEqual({ name: task.actions[0].name, args: jsonDumps(task.actions[0].kwargs) });
    expect({ items: ep.getState().items.length, messages: ep.agentMessages.length, actions: ep.env.actions.length }).toEqual(
      before,
    );
    expect(ep.getState().usage.agent.calls).toBe(1);

    // the human runs it, and the next draft follows on from there
    ep.callAgentTool(draft.tool!.name, parseToolArguments(draft.tool!.args));
    expect((await ep.draft("agent")).tool?.name).toBe(task.actions[1].name);
    expect(ep.toJSON().playground).toMatchObject({ model_drafts: { agent: 2, user: 0 } });
  });

  it("drafts the next customer message for a human user", async () => {
    const task = tasksFor("retail", "test")[0];
    vi.stubGlobal(
      "fetch",
      fakeProvider(() => task),
    );
    const ep = new Episode(setupFor(domains.retail, task, 0, { controllers: { agent: "model", user: "human" } }));
    ep.start();
    expect(ep.getState().turn).toBe("user");
    expect(await ep.draft("user")).toEqual({ model: "fake-model", text: "Hi, I need some help." });
    expect(ep.getState().items).toHaveLength(1); // only the greeting
  });

  it("stops at the step limit with reward 0", async () => {
    const domain = domains.retail;
    const task = tasksFor("retail", "test")[0];
    vi.stubGlobal(
      "fetch",
      fakeProvider(() => task),
    );
    const ep = new Episode(setupFor(domain, task, 0, { maxAgentSteps: 2 }));
    ep.start();
    await untilSettled(ep);
    expect(ep.getState().done).toMatchObject({ reason: "max_steps", reward: 0 });
    expect(ep.getState().agentSteps).toBe(2);
  });

  it("pauses on an unparsable tool call instead of guessing", async () => {
    const domain = domains.retail;
    const task = tasksFor("retail", "test")[0];
    vi.stubGlobal("fetch", brokenAgentProvider());
    const ep = new Episode(setupFor(domain, task, 0));
    ep.start();
    await untilSettled(ep);
    expect(ep.getState().error).toMatchObject({ role: "agent", modelError: true, retryable: false });
    expect(ep.getState().done).toBeNull();
    ep.endNow();
    expect(ep.getState().done?.reason).toBe("manual");
  });

  it("can give up on an error with reward 0, like a tau-bench run that raises", async () => {
    vi.stubGlobal("fetch", brokenAgentProvider());
    const ep = new Episode(setupFor(domains.retail, tasksFor("retail", "test")[0], 0));
    ep.start();
    await untilSettled(ep);
    ep.failOnError();
    expect(ep.getState().error).toBeNull();
    expect(ep.getState().done).toMatchObject({ reason: "error", reward: 0 });
    expect(ep.getState().done?.message).toContain("not a JSON object");
  });
});

describe("replaying saved episodes", () => {
  it("rebuilds a playground export exactly", async () => {
    const domain = domains.retail;
    const index = tasksFor("retail", "test").findIndex((t) => t.outputs.length > 0);
    const task = tasksFor("retail", "test")[index];
    vi.stubGlobal(
      "fetch",
      fakeProvider(() => task),
    );
    const live = new Episode(setupFor(domain, task, index));
    live.start();
    await untilSettled(live);
    expect(live.getState().done?.reward).toBe(1);

    const [record] = parseRecordsFile(JSON.stringify(live.toJSON()), "export.json");
    expect(record).toMatchObject({ domain: "retail", split: "test", taskId: index, reward: 1, endReason: "user_stop" });
    expect(record.models).toEqual({ agent: "fake-model", user: "fake-model" });
    const resolved = await resolveRecord(record, loaders);
    expect(resolved.task).toBe(task);
    const replay = Episode.replay(setupFor(resolved.domain, resolved.task, index), resolved.source);
    const a = live.getState();
    const b = replay.getState();
    expect(b.done).toMatchObject({ reason: "user_stop", reward: 1 });
    expect(b.agentSteps).toBe(a.agentSteps);
    expect(b.items.map((i) => i.kind)).toEqual(a.items.map((i) => i.kind));
    expect(replay.agentMessages.map((m) => [m.role, m.content])).toEqual(live.agentMessages.map((m) => [m.role, m.content]));
    expect(replay.userMessages).toEqual(live.userMessages);
    expect(b.items.some((i) => i.kind === "notice")).toBe(false);
    // a replay's export names the models from the file, not the current settings
    const exported = Episode.replay(
      setupFor(resolved.domain, resolved.task, index, { replayOf: "export.json" }),
      resolved.source,
    ).toJSON();
    expect(exported.playground.agent).toEqual({ controller: "model", models: ["fake-model"] });
    expect(exported.playground).toMatchObject({ replay_of: "export.json" });
  });

  it("scores tau-bench tool-calling results like the original", async () => {
    const records: Record<string, unknown>[] = [];
    for (const name of ["retail", "airline"] as const) {
      tasksFor(name, "test").forEach((task, index) => {
        if (index % 4 === 0) records.push(toolCallingRecord(domains[name], task, index));
      });
    }
    const parsed = parseRecordsFile(JSON.stringify(records), "tool-calling-some-model-0.0_range_0--1_user-gpt-4o-llm_0101.json");
    expect(parsed).toHaveLength(records.length);
    for (const record of parsed) {
      expect(record.domain).not.toBeNull();
      expect(record.models).toEqual({ agent: "some-model", user: "gpt-4o" });
      const resolved = await resolveRecord(record, loaders);
      expect(resolved.taskRef.split).toBe("test");
      const ep = Episode.replay(setupFor(resolved.domain, resolved.task, resolved.taskRef.index), resolved.source);
      const done = ep.getState().done;
      expect([resolved.taskRef.domain, resolved.taskRef.index, done?.reward]).toEqual([
        resolved.taskRef.domain,
        record.taskId,
        1,
      ]);
      expect(ep.getState().items.filter((i) => i.kind === "notice")).toEqual([]);
    }
  });

  it("flags a recorded reward or observation that the replay does not reproduce", async () => {
    const domain = domains.retail;
    const task = tasksFor("retail", "test")[0];
    const record = toolCallingRecord(domain, task, 0);
    const traj = record.traj as Record<string, unknown>[];
    // drop the final write: the database no longer matches, so the reward becomes 0
    const writeAt = traj.map((m) => m.role === "assistant" && Array.isArray(m.tool_calls)).lastIndexOf(true);
    traj.splice(writeAt, 2);
    (traj[3] as { content: string }).content = "tampered";
    const [parsed] = parseRecordsFile(JSON.stringify([record]));
    const resolved = await resolveRecord(parsed, loaders);
    const ep = Episode.replay(setupFor(resolved.domain, resolved.task, 0), resolved.source);
    expect(ep.getState().done?.reward).toBe(0);
    const notices = ep
      .getState()
      .items.filter((i) => i.kind === "notice")
      .map((i) => (i as { text: string }).text);
    expect(notices.some((n) => n.includes("differs"))).toBe(true);
    expect(notices.some((n) => n.includes("records reward 1"))).toBe(true);
  });

  it("replays ReAct trajectories", async () => {
    const domain = domains.airline;
    const index = 0;
    const task = tasksFor("airline", "test")[index];
    const env = new TauEnv(domain, task);
    const traj: Record<string, unknown>[] = [
      { role: "system", content: `${domain.wiki}\n#Available tools\n...` },
      { role: "user", content: "Hi, I need some help." },
    ];
    for (const a of task.actions) {
      const { observation } = env.stepTool({ name: a.name, kwargs: a.kwargs });
      traj.push({
        role: "assistant",
        content: `Thought:\nLet me do it.\nAction:\n{"name": "${a.name}", "arguments": ${jsonDumps(a.kwargs)}}`,
      });
      traj.push({ role: "user", content: `API output: ${observation}` });
    }
    traj.push({
      role: "assistant",
      content: `Thought:\nDone.\nAction:\n{"name": "respond", "arguments": {"content": "${finalReply(task)}"}}`,
    });
    traj.push({ role: "user", content: "###STOP###" });
    const [parsed] = parseRecordsFile(
      JSON.stringify([{ task_id: index, reward: 1.0, info: {}, traj, trial: 0 }]),
      "react-x-0.0_range_0--1_user-y-llm_1.json",
    );
    expect(parsed.domain).toBe("airline");
    const resolved = await resolveRecord(parsed, loaders);
    const ep = Episode.replay(setupFor(resolved.domain, resolved.task, index), resolved.source);
    expect(ep.getState().done).toMatchObject({ reason: "user_stop", reward: 1 });
    expect(ep.getState().items.filter((i) => i.kind === "tool")).toHaveLength(task.actions.length);
    expect(ep.getState().items.some((i) => i.kind === "notice")).toBe(false);
  });

  it("replays a record on its own (for recomputing a whole file)", async () => {
    const task = tasksFor("airline", "test")[5];
    const [record] = parseRecordsFile(JSON.stringify([toolCallingRecord(domains.airline, task, 5)]));
    const ep = await replayRecord(record, loaders, "file.json");
    expect(ep.getState().done?.reward).toBe(1);
    expect(ep.setup.replayOf).toBe("file.json");
    expect(() => ep.setup.resolveEndpoint(ep.setup.modelFor("agent"))).toThrow();
  });

  it("marks a trajectory that just stops as incomplete", async () => {
    const record = toolCallingRecord(domains.retail, tasksFor("retail", "test")[0], 0);
    record.traj = (record.traj as unknown[]).slice(0, 6);
    const [parsed] = parseRecordsFile(JSON.stringify([record]));
    const resolved = await resolveRecord(parsed, loaders);
    const ep = Episode.replay(setupFor(resolved.domain, resolved.task, 0), resolved.source);
    expect(ep.getState().done).toMatchObject({ reason: "incomplete", reward: 0 });
    expect(ep.getState().agentSteps).toBe(2);
  });

  it("treats a tool-calling reply that mentions Action: as a reply, not a ReAct tool call", async () => {
    const action = `Action: {"name": "cancel_pending_order", "arguments": {"order_id": "#W2378156", "reason": "no longer needed"}}`;
    const traj = [
      { role: "system", content: domains.retail.wiki },
      { role: "user", content: "Hi" },
      { role: "assistant", content: `I would do this:\n${action}` },
      { role: "user", content: "###STOP###" },
    ];
    const [parsed] = parseRecordsFile(JSON.stringify([{ task_id: 0, reward: 0.0, info: {}, traj, trial: 0 }]));
    const ep = await replayRecord(parsed, loaders);
    expect(ep.getState().done?.reason).toBe("user_stop");
    expect(ep.env.actions.map((a) => a.name)).toEqual(["respond"]);
    expect(ep.getState().items.some((i) => i.kind === "tool")).toBe(false);
  });

  it("replays an exported crash as a crash", async () => {
    const domain = domains.airline;
    const task = tasksFor("airline", "test")[0];
    // the agent keeps sending certificates; the 4th has no free slot and the Python tool returns None
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        const body = JSON.parse(String(init.body)) as { tools?: unknown };
        const args = JSON.stringify({ user_id: task.user_id, amount: 100 });
        const message = body.tools
          ? {
              role: "assistant",
              content: null,
              tool_calls: [{ id: "c", type: "function", function: { name: "send_certificate", arguments: args } }],
            }
          : { role: "assistant", content: "Hi, I need some help." };
        return new Response(JSON.stringify({ choices: [{ message }] }), { status: 200 });
      }),
    );
    const live = new Episode(setupFor(domain, task, 0));
    live.start();
    await untilSettled(live);
    expect(live.getState().done).toMatchObject({ reason: "crash", reward: 0 });
    const [record] = parseRecordsFile(JSON.stringify(live.toJSON()));
    const replay = await replayRecord(record, loaders);
    expect(replay.getState().done).toMatchObject({ reason: "crash", reward: 0 });
    expect(replay.getState().agentSteps).toBe(4);
    expect(replay.getState().items.some((i) => i.kind === "notice")).toBe(false);

    // an export that lacks the crashing call (made before it was kept) still scores 0, with a notice
    record.traj = record.traj.slice(0, -1);
    const old = await replayRecord(record, loaders);
    expect(old.getState().done).toMatchObject({ reason: "crash", reward: 0 });
    expect(
      old
        .getState()
        .items.filter((i) => i.kind === "notice")
        .map((i) => (i as { code?: string }).code),
    ).toEqual(["replayCrashRecorded"]);
  });

  it("reports runs that crashed in tau-bench", async () => {
    const [parsed] = parseRecordsFile(JSON.stringify([{ task_id: 3, reward: 0.0, info: { error: "boom" }, traj: [], trial: 0 }]));
    await expect(resolveRecord(parsed, loaders)).rejects.toThrow("boom");
    expect(() => parseRecordsFile('{"hello": 1}')).toThrow("No episodes found");
    expect(() => parseRecordsFile('{"info": {}, "tasks": [], "simulations": []}')).toThrow("τ²-bench");
    // files saved by some editors start with a byte order mark
    expect(parseRecordsFile('\uFEFF[{"task_id": 1, "reward": 0.0, "info": {}, "traj": [], "trial": 0}]')).toHaveLength(1);
  });

  it("reads model names from tau-bench result file names", () => {
    expect(modelsFromFileName("results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_0920102016.json")).toEqual({
      agent: "gpt-4o",
      user: "gpt-4o",
    });
    expect(modelsFromFileName("random.json")).toEqual({});
  });
});

describe("BatchRun", () => {
  it("runs every task once per trial, trial by trial", async () => {
    const domain = domains.retail;
    const list = tasksFor("retail", "test");
    // Episodes run one at a time, so the fake agent can follow whichever task was launched last.
    let launching = list[0];
    vi.stubGlobal(
      "fetch",
      fakeProvider(() => launching),
    );
    const order: string[] = [];
    const batch = new BatchRun(
      { domain: "retail", split: "test", indices: [3, 4, 5], trials: 2 },
      (run, onFinish) => {
        order.push(`${run.index}/${run.trial}`);
        launching = list[run.index];
        const ep = new Episode(setupFor(domain, list[run.index], run.index, { onFinish }));
        ep.start();
        return ep;
      },
      { gapMs: 0 },
    );
    const finished = new Promise<void>((resolve) => batch.subscribe(() => batch.getState().status === "done" && resolve()));
    batch.start();
    await finished;
    const s = batch.getState();
    expect(order).toEqual(["3/0", "4/0", "5/0", "3/1", "4/1", "5/1"]);
    expect(s.results.map((r) => r.reward)).toEqual([1, 1, 1, 1, 1, 1]);
    expect(s.results[3].record).toMatchObject({ task_id: 3, trial: 1, reward: 1 });
    expect(batch.exportResults()).toHaveLength(6);
  });

  it("retries transient errors on its own, a few times", async () => {
    const listeners = new Set<() => void>();
    let state: { done: null; error: { role: "agent"; message: string; retryable: boolean } | null } = { done: null, error: null };
    let retries = 0;
    const fake = {
      subscribe: (l: () => void) => (listeners.add(l), () => listeners.delete(l)),
      getState: () => state,
      retry: () => {
        retries++;
        state = { ...state, error: null };
        listeners.forEach((l) => l());
      },
      abort() {},
      resume() {},
    } as unknown as Episode;
    const fail = (retryable: boolean) => {
      state = { ...state, error: { role: "agent", message: "HTTP 503", retryable } };
      listeners.forEach((l) => l());
    };
    const batch = new BatchRun({ domain: "retail", split: "test", indices: [0], trials: 1 }, () => fake, { retryMs: 1 });
    batch.start();
    for (let i = 0; i < MAX_AUTO_RETRIES + 2; i++) {
      fail(true);
      await new Promise((r) => setTimeout(r, 15));
    }
    expect(retries).toBe(MAX_AUTO_RETRIES);
    expect(batch.getState().autoRetry).toBeNull();

    // a bad model output or a missing key waits for the user
    const other = new BatchRun({ domain: "retail", split: "test", indices: [0], trials: 1 }, () => fake, { retryMs: 1 });
    retries = 0;
    state = { done: null, error: null };
    other.start();
    fail(false);
    await new Promise((r) => setTimeout(r, 15));
    expect(retries).toBe(0);
  });

  it("scores an unusable model output as a failed run and moves on", async () => {
    vi.stubGlobal("fetch", brokenAgentProvider());
    const list = tasksFor("retail", "test");
    const batch = new BatchRun(
      { domain: "retail", split: "test", indices: [0, 1], trials: 1 },
      (run, onFinish) => {
        const ep = new Episode(setupFor(domains.retail, list[run.index], run.index, { onFinish }));
        ep.start();
        return ep;
      },
      { gapMs: 0 },
    );
    const finished = new Promise<void>((resolve) => batch.subscribe(() => batch.getState().status === "done" && resolve()));
    batch.start();
    await finished;
    expect(batch.getState().results.map((r) => [r.reason, r.reward])).toEqual([
      ["error", 0],
      ["error", 0],
    ]);
  });

  it("never records an episode it let go of", () => {
    const launched: { index: number; end: () => void }[] = [];
    const batch = new BatchRun(
      { domain: "retail", split: "test", indices: [0, 1], trials: 1 },
      (run, onFinish) => {
        let done: { reward: number; reason: string } | null = null;
        const ep = {
          abort() {},
          resume() {},
          subscribe: () => () => {},
          getState: () => ({ done, error: null, agentSteps: 1 }),
          toJSON: () => ({ task_id: run.index }),
        } as unknown as Episode;
        const end = () => {
          done = { reward: 1, reason: "user_stop" };
          onFinish(ep);
        };
        launched.push({ index: run.index, end });
        return ep;
      },
      { gapMs: 0 },
    );
    batch.start();
    batch.detach(); // e.g. the user opened something else
    batch.resume(); // the same run starts again
    expect(launched.map((l) => l.index)).toEqual([0, 0]);
    launched[0].end(); // the abandoned episode finishes late
    expect(batch.getState().results).toEqual([]);
    expect(batch.getState().current).toEqual({ index: 0, trial: 0 });
    launched[1].end();
    expect(batch.getState().results.map((r) => r.index)).toEqual([0]);
  });

  it("does not stack an automatic retry on Resume or on a manual Retry", async () => {
    const listeners = new Set<() => void>();
    let state: { done: null; error: { message: string; retryable: boolean; modelError: boolean } | null; paused: boolean } = {
      done: null,
      error: null,
      paused: false,
    };
    const set = (patch: Partial<typeof state>) => {
      state = { ...state, ...patch };
      listeners.forEach((l) => l());
    };
    let retries = 0;
    const fake = {
      subscribe: (l: () => void) => (listeners.add(l), () => listeners.delete(l)),
      getState: () => state,
      retry: () => {
        retries++;
        set({ error: null });
      },
      resume: () => set({ paused: false }),
      abort: () => set({ paused: true }),
    } as unknown as Episode;
    const batch = new BatchRun({ domain: "retail", split: "test", indices: [0], trials: 1 }, () => fake, { retryMs: 20 });
    batch.start();
    batch.stop();
    set({ error: { message: "HTTP 503", retryable: true, modelError: false } });
    batch.resume();
    expect(retries).toBe(1);
    expect(batch.getState().autoRetry).toBeNull();

    // a pending automatic retry is dropped when the user retries first
    set({ error: { message: "HTTP 503", retryable: true, modelError: false } });
    expect(batch.getState().autoRetry).not.toBeNull();
    fake.retry();
    expect(batch.getState().autoRetry).toBeNull();
    await new Promise((r) => setTimeout(r, 40));
    expect(retries).toBe(2);
  });

  it("scores a model error raised while stopped when resumed, and can detach without stopping the episode", () => {
    let state: { done: null | { reward: number; reason: string }; error: { modelError: boolean; retryable: boolean } | null } = {
      done: null,
      error: null,
    };
    const calls: string[] = [];
    const fake = {
      subscribe: () => () => {},
      getState: () => state,
      retry: () => calls.push("retry"),
      resume: () => calls.push("resume"),
      abort: () => calls.push("abort"),
      failOnError: () => calls.push("failOnError"),
    } as unknown as Episode;
    const batch = new BatchRun({ domain: "retail", split: "test", indices: [0, 1], trials: 1 }, () => fake, { gapMs: 0 });
    batch.start();
    batch.stop();
    state = { ...state, error: { modelError: true, retryable: false } };
    batch.resume();
    expect(calls).toEqual(["abort", "failOnError"]);

    calls.length = 0;
    batch.stop();
    batch.detach({ abort: false });
    expect(calls).toEqual(["abort"]); // from stop(); detaching itself left the episode alone
  });

  it("puts a detached run back in the queue", () => {
    const launched: number[] = [];
    const batch = new BatchRun({ domain: "retail", split: "test", indices: [0, 1], trials: 1 }, (run) => {
      launched.push(run.index);
      return {
        abort() {},
        resume() {},
        subscribe: () => () => {},
        getState: () => ({ done: null, error: null }),
      } as unknown as Episode;
    });
    batch.start();
    expect(batch.getState().current).toEqual({ index: 0, trial: 0 });
    batch.detach();
    expect(batch.getState()).toMatchObject({ status: "stopped", current: null });
    expect(batch.getState().pending.map((r) => r.index)).toEqual([0, 1]);
    batch.resume();
    expect(launched).toEqual([0, 0]);
  });
});
