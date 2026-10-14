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
