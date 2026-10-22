import { type Action, RESPOND_ACTION_NAME } from "./env";
import { type PyDict, type PyValue, changedRecords, isDict, strictEqual } from "./py";

export interface FieldDiff {
  path: string;
  expected: PyValue | undefined;
  actual: PyValue | undefined;
}

/** Leaf-level differences between two values, as dotted/indexed paths. */
export function diffValues(expected: PyValue | undefined, actual: PyValue | undefined, path = ""): FieldDiff[] {
  if (expected !== undefined && actual !== undefined && strictEqual(expected, actual)) return [];
  if (expected !== undefined && actual !== undefined && isDict(expected) && isDict(actual)) {
    const keys = [...expected.keys(), ...Array.from(actual.keys()).filter((k) => !expected.has(k))];
    return keys.flatMap((k) => diffValues(expected.get(k), actual.get(k), path ? `${path}.${k}` : k));
  }
  if (Array.isArray(expected) && Array.isArray(actual)) {
    const n = Math.max(expected.length, actual.length);
    const out: FieldDiff[] = [];
    for (let i = 0; i < n; i++) out.push(...diffValues(expected[i], actual[i], `${path}[${i}]`));
    return out;
  }
  return [{ path: path || "(record)", expected, actual }];
}

export interface RecordComparison {
  collection: string;
  key: string;
  expected: PyValue | undefined;
  actual: PyValue | undefined;
  ok: boolean;
  diffs: FieldDiff[];
}

/** Every record changed by the agent or by the ground truth, compared with the expected end state. */
export function compareDatabases(initial: PyDict, actual: PyDict, expected: PyDict): RecordComparison[] {
  const seen = new Set<string>();
  const out: RecordComparison[] = [];
  for (const c of [...changedRecords(initial, expected), ...changedRecords(initial, actual)]) {
    const id = `${c.collection}\u0000${c.key}`;
    if (seen.has(id)) continue;
    seen.add(id);
    const e = (expected.get(c.collection) as PyDict | undefined)?.get(c.key);
    const a = (actual.get(c.collection) as PyDict | undefined)?.get(c.key);
    const ok = e !== undefined && a !== undefined ? strictEqual(e, a) : e === a;
    out.push({ collection: c.collection, key: c.key, expected: e, actual: a, ok, diffs: ok ? [] : diffValues(e, a) });
  }
  return out;
}

export interface WriteCheck {
  /** A ground-truth action that changes the database. */
  action: Action;
  matched: boolean;
  /** When nothing matched exactly: the agent's call to the same tool with the fewest differing arguments. */
  closest: { call: Action; diffs: FieldDiff[] } | null;
}

/**
 * The ground truth's writes checked against the agent's calls (exact match, else the nearest call to the
 * same tool and how its arguments differ), plus the agent's writes that match no ground-truth action.
 */
export function compareWrites(
  groundTruth: Action[],
  agent: Action[],
  isWrite: (name: string) => boolean,
): { checks: WriteCheck[]; extra: Action[] } {
  const gtWrites = groundTruth.filter((a) => isWrite(a.name));
  const agentWrites = agent.filter((a) => a.name !== RESPOND_ACTION_NAME && isWrite(a.name));
  const same = (a: Action, b: Action) => a.name === b.name && strictEqual(a.kwargs, b.kwargs);
  const checks = gtWrites.map((action): WriteCheck => {
    if (agentWrites.some((c) => same(c, action))) return { action, matched: true, closest: null };
    let closest: WriteCheck["closest"] = null;
    for (const call of agentWrites.filter((c) => c.name === action.name)) {
      const diffs = diffValues(action.kwargs, call.kwargs);
      if (!closest || diffs.length < closest.diffs.length) closest = { call, diffs };
    }
    return { action, matched: false, closest };
  });
  const extra = agentWrites.filter((c) => !gtWrites.some((g) => same(g, c)));
  return { checks, extra };
}
