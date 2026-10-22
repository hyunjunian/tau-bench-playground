/**
 * Values a human agent is likely to type into a tool form (order ids, item
 * ids, payment methods, flight numbers...), gathered from earlier tool
 * results and the customer's messages.
 */
import { PyDict, type PyValue, parseJson } from "../tau/py";

/** Field name -> values, most recently seen first. */
export type Suggestions = Map<string, string[]>;

export type SuggestionSource =
  | { kind: "tool"; name: string; observation: string; error: boolean }
  | { kind: "user"; text: string };

const ALIASES: Record<string, string[]> = {
  payment_id: ["payment_method_id"],
  payment_method_id: ["payment_id"],
};

/**
 * Parameters whose best candidates come from a particular place: items to
 * return or exchange are in an order's `items`, replacements are available
 * product `variants`. Keys are "<nearest collection>.<field>".
 */
const CONTEXT: Record<string, string[]> = {
  item_ids: ["items.item_id"],
  new_item_ids: ["variants.item_id"],
};

const isIdLike = (key: string) => /\d/.test(key);

const singular = (key: string) => (key.endsWith("s") ? key.slice(0, -1) : key);

export function collectSuggestions(sources: SuggestionSource[]): Suggestions {
  const out: Suggestions = new Map();
  const add = (key: string, value: string) => {
    const v = value.trim();
    if (!v || v.length > 100) return;
    const list = out.get(key) ?? [];
    const at = list.indexOf(v);
    if (at >= 0) list.splice(at, 1);
    list.unshift(v);
    out.set(key, list);
  };
  /** `key` is the field holding `v`; `collection` the nearest enclosing field that is not an id. */
  const walk = (v: PyValue, key: string | null, collection: string | null) => {
    if (typeof v === "string") {
      if (key) add(key, v);
    } else if (Array.isArray(v)) {
      for (const x of v) {
        // e.g. "orders": ["#W..."] in user details -> order_id
        if (typeof x === "string" && key) {
          add(key, x);
          add(`${singular(key)}_id`, x);
        } else walk(x, key, key);
      }
    } else if (v instanceof PyDict) {
      const unavailable = v.get("available") === false;
      for (const [k, x] of v) {
        // e.g. "payment_methods": {"credit_card_123": {...}} -> payment_method_id
        if (key && x instanceof PyDict && isIdLike(k)) add(`${singular(key)}_id`, k);
        if (typeof x === "string" && collection && !unavailable) add(`${collection}.${k}`, x);
        walk(x, k, key && isIdLike(key) ? collection : key);
      }
    }
  };
  for (const s of sources) {
    if (s.kind === "user") {
      for (const m of s.text.matchAll(/#W\d{7}\b/g)) add("order_id", m[0]);
      for (const m of s.text.matchAll(/[\w.+-]+@[\w-]+(?:\.[\w-]+)+/g)) add("email", m[0]);
      continue;
    }
    if (s.error) continue;
    const text = s.observation.trim();
    if (text.startsWith("{") || text.startsWith("[")) {
      try {
        walk(parseJson(text), null, null);
      } catch {
        // not JSON
      }
    } else if (s.name.startsWith("find_user_id") && text) {
      add("user_id", text);
    }
  }
  return out;
}

/** Candidates for a tool parameter: same-named fields, singular forms for lists (item_ids -> item_id), and aliases. */
export function suggestionsFor(all: Suggestions, param: string): string[] {
  const base = param.replace(/^new_/, "");
  const keys = [...(CONTEXT[param] ?? []), param, base, singular(base), ...(ALIASES[base] ?? [])];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const k of keys) {
    for (const v of all.get(k) ?? []) {
      if (!seen.has(v)) {
        seen.add(v);
        out.push(v);
      }
    }
  }
  return out;
}
