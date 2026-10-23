/** Test data shared by the test files: the exported tau-bench domains and tasks. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Domain, type DomainName, type Task, type ToolSpec, parseTasks } from "../src/tau/env";

export const ROOT = join(__dirname, "..");

export const read = (p: string) => readFileSync(join(ROOT, p), "utf8");

export function loadDomain(name: DomainName): Domain {
  const tools = JSON.parse(read(`public/data/${name}/tools.json`)) as ToolSpec[];
  return new Domain(name, read(`public/data/${name}/db.json`), tools, read(`public/data/${name}/wiki.md`));
}

const domainCache = new Map<DomainName, Domain>();
export function domainFor(name: DomainName): Domain {
  if (!domainCache.has(name)) domainCache.set(name, loadDomain(name));
  return domainCache.get(name)!;
}

const taskCache = new Map<string, Task[]>();
export function tasksFor(domain: DomainName, split: string): Task[] {
  const key = `${domain}/${split}`;
  if (!taskCache.has(key)) taskCache.set(key, parseTasks(read(`public/data/${domain}/tasks_${split}.json`)));
  return taskCache.get(key)!;
}
