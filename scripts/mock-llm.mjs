#!/usr/bin/env node
/**
 * A scripted OpenAI-compatible endpoint for trying the playground without an
 * API key (development and screenshots). It is not a model: the "agent"
 * replays each task's ground-truth actions and the "user" opens the chat and
 * stops after the agent's reply.
 *
 *   node scripts/mock-llm.mjs [port]      # default 8787
 *
 * In the app, set a role's provider to "Custom (OpenAI-compatible)", the base
 * URL to http://localhost:8787/v1, and the model to one of:
 *
 *   gt             replay the ground truth (reward 1)
 *   gt-slow        the same, 1.2 s per call
 *   scripted-demo  like gt, but now and then skips the last write (reward 0)
 *   flaky          like gt, but every other call is rate limited (429)
 *   badargs        tool calls with arguments that are not JSON
 *   empty          the user model answers with an empty message
 *   quota          always fails with a daily-quota 429
 *
 * Use it for both roles: the agent finds the task from a marker that the user
 * side hides (zero-width characters) in its first message.
 */
import { readFileSync } from "node:fs";
import http from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const DATA = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "data");
const PORT = Number(process.argv[2] ?? process.env.PORT ?? 8787);
const MODELS = ["gt", "gt-slow", "scripted-demo", "flaky", "badargs", "empty", "quota"];

const tasks = [];
for (const [domain, splits] of [
  ["retail", ["test", "train", "dev"]],
  ["airline", ["test"]],
]) {
  for (const split of splits) tasks.push(...JSON.parse(readFileSync(join(DATA, domain, `tasks_${split}.json`), "utf8")));
}
const byInstruction = new Map();
for (const [i, t] of tasks.entries()) if (!byInstruction.has(t.instruction)) byInstruction.set(t.instruction, i);

// Task index <-> invisible marker (word joiners around zero-width space/non-joiner bits).
const hide = (n) => `⁠${n.toString(2).replace(/0/g, "​").replace(/1/g, "‌")}⁠`;
const unhide = (text) => {
  const m = /⁠([​‌]+)⁠/.exec(text);
  return m ? parseInt(m[1].replace(/​/g, "0").replace(/‌/g, "1"), 2) : -1;
};

let calls = 0;
const attempts = new Map();
const failing = new Set();

function send(res, status, body) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
  });
  res.end(JSON.stringify(body));
}

const reply = (message) => ({
  id: `mock-${calls}`,
  object: "chat.completion",
  choices: [{ index: 0, message: { role: "assistant", ...message }, finish_reason: "stop" }],
  usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
});

function userTurn(model, messages) {
  const instruction = String(messages[0]?.content ?? "").split("\n\nInstruction: ")[1]?.split("\nRules:")[0]?.trim();
  const index = byInstruction.get(instruction) ?? -1;
  const turns = messages.filter((m) => m.role === "assistant").length;
  if (turns > 0) return { content: model === "empty" ? "" : "That's everything, thank you! ###STOP###" };
  const n = (attempts.get(index) ?? 0) + 1;
  attempts.set(index, n);
  if (model === "scripted-demo" && (index * 7 + n) % 4 === 0) failing.add(index);
  else failing.delete(index);
  return { content: `Hi! I need some help with my account.${hide(index)}` };
}

function agentTurn(model, messages) {
  const task = tasks[unhide(String(messages.find((m) => m.role === "user")?.content ?? ""))];
  if (!task) return { content: "This scripted agent only knows tasks whose first customer message came from the mock user." };
  const actions = failing.has(tasks.indexOf(task)) ? task.actions.slice(0, -1) : task.actions;
  const done = messages.filter((m) => m.role === "tool").length;
  const action = actions[done];
  if (!action) {
    const outputs = task.outputs.length ? ` (${task.outputs.join(", ")})` : "";
    return { content: `Your request is complete${outputs}. Is there anything else I can help you with?` };
  }
  const args = model === "badargs" ? "{not json" : JSON.stringify(action.kwargs);
  return { content: null, tool_calls: [{ id: `call_${done}`, type: "function", function: { name: action.name, arguments: args } }] };
}

http
  .createServer(async (req, res) => {
    if (req.method === "OPTIONS") return send(res, 204, {});
    if (req.url?.endsWith("/models")) return send(res, 200, { data: MODELS.map((id) => ({ id, object: "model" })) });
    if (req.method !== "POST" || !req.url?.endsWith("/chat/completions")) return send(res, 404, { error: { message: "not found" } });
    let body = "";
    for await (const chunk of req) body += chunk;
    const { model, messages, tools } = JSON.parse(body);
    calls++;
    if (model === "gt-slow") await new Promise((resolve) => setTimeout(resolve, 1200));
    if (model === "quota") {
      return send(res, 429, [{ error: { code: 429, message: "Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel" } }]);
    }
    if (model === "flaky" && calls % 2 === 0) return send(res, 429, { error: { message: "Rate limited. Please retry in 2s." } });
    send(res, 200, reply(tools ? agentTurn(model, messages) : userTurn(model, messages)));
  })
  .listen(PORT, () => console.log(`mock LLM on http://localhost:${PORT}/v1 (models: ${MODELS.join(", ")})`));
