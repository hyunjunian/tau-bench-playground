import { Domain, type DomainName, type Task, type ToolSpec, parseTasks } from "../tau/env";

const BASE = `${import.meta.env.BASE_URL}data`;

async function fetchText(path: string): Promise<string> {
  const res = await fetch(`${BASE}/${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.text();
}

const domains = new Map<DomainName, Promise<Domain>>();
const tasks = new Map<string, Promise<Task[]>>();

export function loadDomain(name: DomainName): Promise<Domain> {
  let p = domains.get(name);
  if (!p) {
    p = Promise.all([fetchText(`${name}/db.json`), fetchText(`${name}/tools.json`), fetchText(`${name}/wiki.md`)]).then(
      ([db, tools, wiki]) => new Domain(name, db, JSON.parse(tools) as ToolSpec[], wiki),
    );
    p.catch(() => domains.delete(name));
    domains.set(name, p);
  }
  return p;
}

export function loadTasks(domain: DomainName, split: string): Promise<Task[]> {
  const key = `${domain}/${split}`;
  let p = tasks.get(key);
  if (!p) {
    p = fetchText(`${domain}/tasks_${split}.json`).then(parseTasks);
    p.catch(() => tasks.delete(key));
    tasks.set(key, p);
  }
  return p;
}
