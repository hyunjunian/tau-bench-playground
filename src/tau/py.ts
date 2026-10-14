/**
 * Minimal Python value model so the tau-bench tools behave exactly like the
 * original Python implementation (dict key order, int/float distinction,
 * json.dumps formatting, round(), sum(), and the exception messages that end
 * up in tool observations).
 *
 * Python value  -> JS representation
 *   None        -> null
 *   bool        -> boolean
 *   int         -> number
 *   float       -> PyFloat
 *   str         -> string
 *   list        -> PyValue[]
 *   dict        -> PyDict (an insertion-ordered Map; plain objects would reorder
 *                  numeric keys such as product ids)
 */

export class PyFloat {
  constructor(readonly value: number) {}
}

export class PyDict extends Map<string, PyValue> {}

export type PyValue = null | boolean | number | string | PyFloat | PyValue[] | PyDict;

export type PyNumber = boolean | number | PyFloat;

/** A Python exception. `message` is what `str(e)` returns in Python. */
export class PyException extends Error {
  constructor(
    readonly pyType: string,
    message: string,
  ) {
    super(message);
    this.name = pyType;
  }
}

export const TypeError_ = (msg: string) => new PyException("TypeError", msg);
export const ValueError_ = (msg: string) => new PyException("ValueError", msg);
export const KeyError_ = (key: PyValue) => new PyException("KeyError", pyRepr(key));
export const IndexError_ = (msg: string) => new PyException("IndexError", msg);
export const AttributeError_ = (msg: string) => new PyException("AttributeError", msg);
export const ZeroDivisionError_ = (msg: string) => new PyException("ZeroDivisionError", msg);
export const OverflowError_ = (msg: string) => new PyException("OverflowError", msg);
export const SyntaxError_ = (msg: string) => new PyException("SyntaxError", msg);

// ---------------------------------------------------------------------------
// Type inspection

export function isDict(v: PyValue): v is PyDict {
  return v instanceof PyDict;
}

export function isNumberLike(v: PyValue): v is PyNumber {
  return typeof v === "number" || typeof v === "boolean" || v instanceof PyFloat;
}

export function typeName(v: PyValue): string {
  if (v === null) return "NoneType";
  if (typeof v === "boolean") return "bool";
  if (typeof v === "number") return "int";
  if (typeof v === "string") return "str";
  if (v instanceof PyFloat) return "float";
  if (Array.isArray(v)) return "list";
  return "dict";
}

function isFloatish(v: PyNumber): v is PyFloat {
  return v instanceof PyFloat;
}

/** Numeric value of an int/float/bool. */
export function num(v: PyNumber): number {
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "number") return v;
  return v.value;
}

function hashable(v: PyValue): void {
  if (Array.isArray(v)) throw TypeError_("unhashable type: 'list'");
  if (v instanceof PyDict) throw TypeError_("unhashable type: 'dict'");
}

// ---------------------------------------------------------------------------
// Equality / ordering

export function pyEq(a: PyValue, b: PyValue): boolean {
  if (isNumberLike(a) && isNumberLike(b)) return num(a) === num(b);
  if (a === null || b === null) return a === b;
  if (typeof a === "string" || typeof b === "string") return a === b;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((x, i) => pyEq(x, b[i]));
  }
  if (a instanceof PyDict && b instanceof PyDict) {
    if (a.size !== b.size) return false;
    for (const [k, v] of a) {
      if (!b.has(k) || !pyEq(v, b.get(k)!)) return false;
    }
    return true;
  }
  return false;
}

function cmpCodePoints(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const ca = a.codePointAt(i)!;
    const cb = b.codePointAt(i)!;
    if (ca !== cb) return ca < cb ? -1 : 1;
    if (ca > 0xffff) i++;
  }
  return a.length === b.length ? 0 : a.length < b.length ? -1 : 1;
}

/** Three-way comparison with Python's rules for `<`; throws TypeError on mixed types. */
export function pyCompare(a: PyValue, b: PyValue, op: string): number {
  if (isNumberLike(a) && isNumberLike(b)) {
    const x = num(a);
    const y = num(b);
    return x < y ? -1 : x > y ? 1 : 0;
  }
  if (typeof a === "string" && typeof b === "string") return cmpCodePoints(a, b);
  if (Array.isArray(a) && Array.isArray(b)) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      if (!pyEq(a[i], b[i])) return pyCompare(a[i], b[i], op);
    }
    return a.length - b.length;
  }
  throw TypeError_(`'${op}' not supported between instances of '${typeName(a)}' and '${typeName(b)}'`);
}

export const pyLt = (a: PyValue, b: PyValue) => pyCompare(a, b, "<") < 0;
export const pyGt = (a: PyValue, b: PyValue) => pyCompare(a, b, ">") > 0;

// ---------------------------------------------------------------------------
// Containers

/** `key in container` */
export function pyIn(key: PyValue, container: PyValue): boolean {
  if (container instanceof PyDict) {
    hashable(key);
    return typeof key === "string" && container.has(key);
  }
  if (Array.isArray(container)) return container.some((x) => pyEq(x, key));
  if (typeof container === "string") {
    if (typeof key !== "string") {
      throw TypeError_(`'in <string>' requires string as left operand, not ${typeName(key)}`);
    }
    return container.includes(key);
  }
  throw TypeError_(`argument of type '${typeName(container)}' is not iterable`);
}

/** `container[key]` */
export function getItem(container: PyValue, key: PyValue): PyValue {
  if (container instanceof PyDict) {
    hashable(key);
    if (typeof key === "string" && container.has(key)) return container.get(key)!;
    throw KeyError_(key);
  }
  if (Array.isArray(container) || typeof container === "string") {
    const kind = Array.isArray(container) ? "list" : "string";
    if (typeof key === "number" || typeof key === "boolean") {
      const seq = Array.isArray(container) ? container : Array.from(container);
      let i = num(key);
      if (i < 0) i += seq.length;
      if (i < 0 || i >= seq.length) throw IndexError_(`${kind} index out of range`);
      return seq[i];
    }
    if (kind === "list") {
      throw TypeError_(`list indices must be integers or slices, not ${typeName(key)}`);
    }
    throw TypeError_(`string indices must be integers, not '${typeName(key)}'`);
  }
  throw TypeError_(`'${typeName(container)}' object is not subscriptable`);
}

/** `container[key] = value` for dicts (the only kind the tools assign into). */
export function setItem(container: PyValue, key: string, value: PyValue): void {
  if (!(container instanceof PyDict)) {
    throw TypeError_(`'${typeName(container)}' object does not support item assignment`);
  }
  container.set(key, value);
}

export function pyLen(v: PyValue): number {
  if (typeof v === "string") return Array.from(v).length;
  if (Array.isArray(v)) return v.length;
  if (v instanceof PyDict) return v.size;
  throw TypeError_(`object of type '${typeName(v)}' has no len()`);
}

/** Materialize `iter(v)`. */
export function pyIter(v: PyValue): PyValue[] {
  if (typeof v === "string") return Array.from(v);
  if (Array.isArray(v)) return v.slice();
  if (v instanceof PyDict) return Array.from(v.keys());
  throw TypeError_(`'${typeName(v)}' object is not iterable`);
}

/** `seq.count(x)` for list/str receivers. */
export function pyCount(seq: PyValue, x: PyValue): number {
  if (Array.isArray(seq)) return seq.filter((e) => pyEq(e, x)).length;
  if (typeof seq === "string") {
    if (typeof x !== "string") throw TypeError_(`must be str, not ${typeName(x)}`);
    if (x === "") return Array.from(seq).length + 1;
    return seq.split(x).length - 1;
  }
  throw AttributeError_(`'${typeName(seq)}' object has no attribute 'count'`);
}

export function pyLower(v: PyValue): string {
  if (typeof v !== "string") throw AttributeError_(`'${typeName(v)}' object has no attribute 'lower'`);
  return v.toLowerCase();
}

export function pySorted(v: PyValue): PyValue[] {
  const items = pyIter(v);
  // Python's sort only uses `<`; a stable merge with the same comparator gives the same result.
  return items.sort((a, b) => pyCompare(a, b, "<"));
}

/** `zip(a, b)` */
export function pyZip(a: PyValue, b: PyValue): [PyValue, PyValue][] {
  const xs = pyIter(a);
  const ys = pyIter(b);
  const n = Math.min(xs.length, ys.length);
  const out: [PyValue, PyValue][] = [];
  for (let i = 0; i < n; i++) out.push([xs[i], ys[i]]);
  return out;
}

function copyValue(v: PyValue): PyValue {
  if (Array.isArray(v)) return v.map(copyValue);
  if (v instanceof PyDict) {
    const d = new PyDict();
    for (const [k, x] of v) d.set(k, copyValue(x));
    return d;
  }
  return v; // primitives and PyFloat are immutable
}

export function deepCopy<T extends PyValue>(v: T): T {
  return copyValue(v) as T;
}

export function dict(entries: [string, PyValue][]): PyDict {
  return new PyDict(entries);
}

// ---------------------------------------------------------------------------
// Arithmetic (int/float/bool, plus the str/list cases Python allows)

function binopError(op: string, a: PyValue, b: PyValue): PyException {
  return TypeError_(`unsupported operand type(s) for ${op}: '${typeName(a)}' and '${typeName(b)}'`);
}

function makeNumber(x: number, isFloat: boolean): number | PyFloat {
  return isFloat ? new PyFloat(x) : x;
}

/** `a + b` (pass op "+=" for augmented assignment so error messages match). */
export function pyAdd(a: PyValue, b: PyValue, op = "+"): PyValue {
  if (isNumberLike(a) && isNumberLike(b)) {
    return makeNumber(num(a) + num(b), isFloatish(a) || isFloatish(b));
  }
  if (typeof a === "string" && typeof b === "string") return a + b;
  if (Array.isArray(a) && Array.isArray(b)) return [...a, ...b];
  if (typeof a === "string") {
    throw TypeError_(`can only concatenate str (not "${typeName(b)}") to str`);
  }
  if (Array.isArray(a)) {
    // list.__iadd__ extends with any iterable; list.__add__ only accepts lists.
    if (op === "+=") return [...a, ...pyIter(b)];
    throw TypeError_(`can only concatenate list (not "${typeName(b)}") to list`);
  }
  throw binopError(op, a, b);
}

export function pySub(a: PyValue, b: PyValue, op = "-"): PyValue {
  if (isNumberLike(a) && isNumberLike(b)) {
    return makeNumber(num(a) - num(b), isFloatish(a) || isFloatish(b));
  }
  throw binopError(op, a, b);
}

function repeat(seq: string | PyValue[], times: PyValue): PyValue {
  if (!(typeof times === "number" || typeof times === "boolean")) {
    throw TypeError_(`can't multiply sequence by non-int of type '${typeName(times)}'`);
  }
  const n = Math.max(0, num(times));
  if (n * (typeof seq === "string" ? seq.length : seq.length) > 10_000_000) {
    throw new PyException("MemoryError", "");
  }
  if (typeof seq === "string") return seq.repeat(n);
  const out: PyValue[] = [];
  for (let i = 0; i < n; i++) out.push(...seq);
  return out;
}

export function pyMul(a: PyValue, b: PyValue, op = "*"): PyValue {
  if (isNumberLike(a) && isNumberLike(b)) {
    return makeNumber(num(a) * num(b), isFloatish(a) || isFloatish(b));
  }
  if (typeof a === "string" || Array.isArray(a)) {
    if (typeof b === "number" || typeof b === "boolean") return repeat(a, b);
    if (b instanceof PyFloat) return repeat(a, b);
  }
  if (typeof b === "string" || Array.isArray(b)) {
    if (typeof a === "number" || typeof a === "boolean") return repeat(b, a);
    if (a instanceof PyFloat) return repeat(b, a);
  }
  throw binopError(op, a, b);
}

export function pyNeg(a: PyValue): PyValue {
  if (isNumberLike(a)) return makeNumber(-num(a), isFloatish(a));
  throw TypeError_(`bad operand type for unary -: '${typeName(a)}'`);
}

export function pyAbs(a: PyValue): PyValue {
  if (isNumberLike(a)) return makeNumber(Math.abs(num(a)), isFloatish(a));
  throw TypeError_(`bad operand type for abs(): '${typeName(a)}'`);
}

/** `max(a, b)`: returns the first maximal argument, like Python. */
export function pyMax(a: PyValue, b: PyValue): PyValue {
  return pyGt(b, a) ? b : a;
}

/** `round(x, ndigits)` */
export function pyRound(x: PyValue, ndigits: number): PyValue {
  if (typeof x === "number" || typeof x === "boolean") return num(x); // round(int, n) -> int
  if (x instanceof PyFloat) return new PyFloat(roundFloat(x.value, ndigits));
  throw TypeError_(`type ${typeName(x)} doesn't define __round__ method`);
}

/**
 * Python's float.__round__: the decimal value closest to the exact binary
 * value, ties to even, converted back to the nearest double.
 */
export function roundFloat(x: number, ndigits: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const hi = view.getUint32(0);
  const lo = view.getUint32(4);
  const negative = hi >>> 31 === 1;
  const biasedExp = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let exp2: number;
  if (biasedExp === 0) {
    exp2 = -1074;
  } else {
    mant |= 1n << 52n;
    exp2 = biasedExp - 1075;
  }
  // |x| * 10^ndigits = mant * 10^ndigits * 2^exp2; round that to an integer q.
  const scale = 10n ** BigInt(ndigits);
  let q: bigint;
  if (exp2 >= 0) {
    q = (mant * scale) << BigInt(exp2);
  } else {
    const numer = mant * scale;
    const denom = 1n << BigInt(-exp2);
    q = numer / denom;
    const twiceRem = 2n * (numer - q * denom);
    if (twiceRem > denom || (twiceRem === denom && (q & 1n) === 1n)) q += 1n;
  }
  const result = Number(`${q}e-${ndigits}`);
  return negative ? -result : result;
}

/**
 * Python 3.12+ `sum(iterable)` (integer fast path, Neumaier-compensated
 * floats). Consumes the iterable lazily, like a generator expression.
 */
export function pySum(iterable: Iterable<PyValue>): PyValue {
  const it = iterable[Symbol.iterator]();
  let next = it.next();
  let intResult = 0;
  while (!next.done && (typeof next.value === "number" || typeof next.value === "boolean")) {
    intResult += num(next.value);
    next = it.next();
  }
  if (next.done) return intResult;
  let result: PyValue = pyAdd(intResult, next.value);
  next = it.next();
  if (result instanceof PyFloat) {
    let f = result.value;
    let c = 0;
    for (; !next.done; next = it.next()) {
      const item = next.value;
      if (item instanceof PyFloat) {
        const x = item.value;
        const t = f + x;
        if (Math.abs(f) >= Math.abs(x)) c += f - t + x;
        else c += x - t + f;
        f = t;
        continue;
      }
      if (typeof item === "number" || typeof item === "boolean") {
        f += num(item);
        continue;
      }
      break;
    }
    if (c && Number.isFinite(c)) f += c;
    if (next.done) return new PyFloat(f);
    result = pyAdd(new PyFloat(f), next.value);
    next = it.next();
  }
  for (; !next.done; next = it.next()) result = pyAdd(result, next.value);
  return result;
}

/** Python truthiness. */
export function truthy(v: PyValue): boolean {
  if (v === null || v === false || v === "") return false;
  if (typeof v === "number") return v !== 0;
  if (v instanceof PyFloat) return v.value !== 0;
  if (Array.isArray(v)) return v.length > 0;
  if (v instanceof PyDict) return v.size > 0;
  return true;
}

/** `target.update(source)` */
export function dictUpdate(target: PyValue, source: PyValue): void {
  if (!(target instanceof PyDict)) throw AttributeError_(`'${typeName(target)}' object has no attribute 'update'`);
  if (!(source instanceof PyDict)) throw TypeError_(`'${typeName(source)}' object is not iterable`);
  for (const [k, v] of source) target.set(k, v);
}

/** `seq.append(x)` / `seq.extend(xs)` on a list that lives in the database. */
export function listAppend(seq: PyValue, x: PyValue): void {
  if (!Array.isArray(seq)) throw AttributeError_(`'${typeName(seq)}' object has no attribute 'append'`);
  seq.push(x);
}

export function listExtend(seq: PyValue, xs: PyValue[]): void {
  if (!Array.isArray(seq)) throw AttributeError_(`'${typeName(seq)}' object has no attribute 'extend'`);
  seq.push(...xs);
}

/** `del container[key]` */
export function delItem(container: PyValue, key: PyValue): void {
  if (!(container instanceof PyDict)) {
    throw TypeError_(`'${typeName(container)}' object does not support item deletion`);
  }
  hashable(key);
  if (typeof key !== "string" || !container.has(key)) throw KeyError_(key);
  container.delete(key);
}

/** `int(s)` for the string slice used by search_onestop_flight. */
export function pyInt(v: PyValue): number {
  if (typeof v === "number" || typeof v === "boolean") return num(v);
  if (v instanceof PyFloat) {
    if (!Number.isFinite(v.value)) {
      throw Number.isNaN(v.value)
        ? ValueError_("cannot convert float NaN to integer")
        : OverflowError_("cannot convert float infinity to integer");
    }
    return Math.trunc(v.value);
  }
  if (typeof v === "string") {
    const s = v.trim();
    if (/^[+-]?\d+(?:_\d+)*$/.test(s)) {
      const n = Number(s.replace(/_/g, ""));
      return Object.is(n, -0) ? 0 : n;
    }
    throw ValueError_(`invalid literal for int() with base 10: ${strRepr(v)}`);
  }
  throw TypeError_(
    `int() argument must be a string, a bytes-like object or a real number, not '${typeName(v)}'`,
  );
}

/** `v[-2:]` */
export function lastTwo(v: PyValue): PyValue {
  if (typeof v === "string") return Array.from(v).slice(-2).join("");
  if (Array.isArray(v)) return v.slice(-2);
  if (v instanceof PyDict) throw new PyException("KeyError", "slice(-2, None, None)");
  throw TypeError_(`'${typeName(v)}' object is not subscriptable`);
}

// ---------------------------------------------------------------------------
// String conversion

/** Shortest round-trip digits of a positive finite double, as digits + decimal exponent. */
function shortestDigits(ax: number): { digits: string; exp: number } {
  const s = String(ax);
  let mant = s;
  let e = 0;
  const ei = s.indexOf("e");
  if (ei >= 0) {
    mant = s.slice(0, ei);
    e = parseInt(s.slice(ei + 1), 10);
  }
  let intPart = mant;
  let frac = "";
  const di = mant.indexOf(".");
  if (di >= 0) {
    intPart = mant.slice(0, di);
    frac = mant.slice(di + 1);
  }
  let digits = intPart + frac;
  let exp = e + intPart.length - 1;
  const lead = digits.length - digits.replace(/^0+/, "").length;
  digits = digits.slice(lead);
  exp -= lead;
  digits = digits.replace(/0+$/, "");
  return { digits, exp };
}

/** `repr(float)` */
export function floatRepr(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const sign = x < 0 ? "-" : "";
  const { digits, exp } = shortestDigits(Math.abs(x));
  if (exp < -4 || exp >= 16) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    const ae = Math.abs(exp);
    return `${sign}${m}e${exp < 0 ? "-" : "+"}${ae < 10 ? "0" : ""}${ae}`;
  }
  if (exp >= 0) {
    if (digits.length <= exp + 1) return `${sign}${digits}${"0".repeat(exp + 1 - digits.length)}.0`;
    return `${sign}${digits.slice(0, exp + 1)}.${digits.slice(exp + 1)}`;
  }
  return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
}

function intStr(x: number): string {
  if (Object.is(x, -0)) return "0";
  if (Number.isSafeInteger(x)) return String(x);
  return BigInt(Math.trunc(x)).toString();
}

function isPrintable(cp: number): boolean {
  if (cp === 0x20) return true;
  return !/[\p{C}\p{Z}]/u.test(String.fromCodePoint(cp));
}

/** `repr(str)` */
export function strRepr(s: string): string {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    if (ch === quote || ch === "\\") out += "\\" + ch;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (cp < 0x7f && cp >= 0x20) out += ch;
    else if (isPrintable(cp)) out += ch;
    else if (cp <= 0xff) out += "\\x" + cp.toString(16).padStart(2, "0");
    else if (cp <= 0xffff) out += "\\u" + cp.toString(16).padStart(4, "0");
    else out += "\\U" + cp.toString(16).padStart(8, "0");
  }
  return out + quote;
}

export function pyRepr(v: PyValue): string {
  if (v === null) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return intStr(v);
  if (v instanceof PyFloat) return floatRepr(v.value);
  if (typeof v === "string") return strRepr(v);
  if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
  return `{${Array.from(v, ([k, x]) => `${strRepr(k)}: ${pyRepr(x)}`).join(", ")}}`;
}

/** `str(v)` (what f-strings interpolate). */
export function pyStr(v: PyValue): string {
  return typeof v === "string" ? v : pyRepr(v);
}

// ---------------------------------------------------------------------------
// JSON

function jsonString(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    const ch = s[i];
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (c < 0x20 || c > 0x7e) out += "\\u" + c.toString(16).padStart(4, "0");
    else out += ch;
  }
  return out + '"';
}

function jsonFloat(x: number): string {
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return x > 0 ? "Infinity" : "-Infinity";
  return floatRepr(x);
}

/** `json.dumps(v)` with default arguments (", "/": " separators, ensure_ascii). */
export function jsonDumps(v: PyValue): string {
  if (v === null) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return intStr(v);
  if (v instanceof PyFloat) return jsonFloat(v.value);
  if (typeof v === "string") return jsonString(v);
  if (Array.isArray(v)) return `[${v.map(jsonDumps).join(", ")}]`;
  return `{${Array.from(v, ([k, x]) => `${jsonString(k)}: ${jsonDumps(x)}`).join(", ")}}`;
}

/** Pretty JSON for display (2-space indent, Python number formatting). */
export function jsonPretty(v: PyValue, indent = ""): string {
  const next = indent + "  ";
  if (Array.isArray(v)) {
    if (v.length === 0) return "[]";
    return `[\n${v.map((x) => next + jsonPretty(x, next)).join(",\n")}\n${indent}]`;
  }
  if (v instanceof PyDict) {
    if (v.size === 0) return "{}";
    const rows = Array.from(v, ([k, x]) => `${next}${jsonString(k)}: ${jsonPretty(x, next)}`);
    return `{\n${rows.join(",\n")}\n${indent}}`;
  }
  if (typeof v === "string") return JSON.stringify(v);
  return jsonDumps(v);
}

/**
 * Canonical, order-independent serialization used to compare database states.
 * Equivalent to tau-bench's `consistent_hash(to_hashable(data))` comparison:
 * dict keys are sorted, list order matters, and ints differ from floats.
 */
export function canonical(v: PyValue): string {
  const parts: string[] = [];
  const walk = (x: PyValue) => {
    if (x === null) parts.push("null");
    else if (typeof x === "boolean") parts.push(x ? "true" : "false");
    else if (typeof x === "number") parts.push(intStr(x));
    else if (x instanceof PyFloat) parts.push(floatRepr(x.value));
    else if (typeof x === "string") parts.push(JSON.stringify(x));
    else if (Array.isArray(x)) {
      parts.push("[");
      x.forEach((y, i) => {
        if (i) parts.push(",");
        walk(y);
      });
      parts.push("]");
    } else {
      parts.push("{");
      const keys = Array.from(x.keys()).sort(cmpCodePoints);
      keys.forEach((k, i) => {
        if (i) parts.push(",");
        parts.push(JSON.stringify(k), ":");
        walk(x.get(k)!);
      });
      parts.push("}");
    }
  };
  walk(v);
  return parts.join("");
}

/**
 * Type-strict structural equality: what tau-bench's hash comparison
 * (`str(to_hashable(data))`) distinguishes. Dict key order is ignored, list
 * order is not, and 1, 1.0 and True are all different.
 */
export function strictEqual(a: PyValue, b: PyValue): boolean {
  if (a === b) return true;
  if (a instanceof PyFloat) return b instanceof PyFloat && Object.is(a.value, b.value);
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!strictEqual(a[i], b[i])) return false;
    return true;
  }
  if (a instanceof PyDict) {
    if (!(b instanceof PyDict) || a.size !== b.size) return false;
    for (const [k, v] of a) {
      const w = b.get(k);
      if (w === undefined || !strictEqual(v, w)) return false;
    }
    return true;
  }
  return false;
}

export interface RecordChange {
  collection: string;
  key: string;
  before: PyValue | undefined;
  after: PyValue | undefined;
}

/** Records (second-level entries such as users/orders/reservations) that differ between two databases. */
export function changedRecords(before: PyDict, after: PyDict): RecordChange[] {
  const changes: RecordChange[] = [];
  const collections = new Set([...before.keys(), ...after.keys()]);
  for (const collection of Array.from(collections).sort(cmpCodePoints)) {
    const b = before.get(collection);
    const a = after.get(collection);
    if (!(b instanceof PyDict) || !(a instanceof PyDict)) {
      if (b === undefined || a === undefined || !strictEqual(b, a)) {
        changes.push({ collection, key: "", before: b, after: a });
      }
      continue;
    }
    const keys = new Set([...b.keys(), ...a.keys()]);
    for (const key of Array.from(keys).sort(cmpCodePoints)) {
      const x = b.get(key);
      const y = a.get(key);
      if (x === undefined || y === undefined || !strictEqual(x, y)) {
        changes.push({ collection, key, before: x, after: y });
      }
    }
  }
  return changes;
}

/** Canonical text of the changed records only (used to compare against Python traces cheaply). */
export function deltaCanonical(before: PyDict, after: PyDict): string {
  return changedRecords(before, after)
    .map((c) => `${c.collection}/${c.key}=${c.after === undefined ? "<deleted>" : canonical(c.after)}`)
    .join("\n");
}

export class JsonParseError extends Error {}

/**
 * JSON parser that keeps Python's distinction between ints and floats (a
 * number token with a fraction or exponent becomes a PyFloat) and keeps dict
 * key order. Accepts NaN/Infinity like Python's json.loads.
 */
export function parseJson(text: string): PyValue {
  let i = 0;
  const n = text.length;
  const fail = (msg: string): never => {
    throw new JsonParseError(`${msg} at position ${i}`);
  };
  const ws = () => {
    while (i < n) {
      const c = text.charCodeAt(i);
      if (c === 0x20 || c === 0x0a || c === 0x0d || c === 0x09) i++;
      else break;
    }
  };
  const parseString = (): string => {
    i++; // opening quote
    let out = "";
    let start = i;
    while (i < n) {
      const c = text.charCodeAt(i);
      if (c === 0x22) {
        out += text.slice(start, i);
        i++;
        return out;
      }
      if (c === 0x5c) {
        out += text.slice(start, i);
        const e = text[i + 1];
        i += 2;
        switch (e) {
          case '"':
            out += '"';
            break;
          case "\\":
            out += "\\";
            break;
          case "/":
            out += "/";
            break;
          case "b":
            out += "\b";
            break;
          case "f":
            out += "\f";
            break;
          case "n":
            out += "\n";
            break;
          case "r":
            out += "\r";
            break;
          case "t":
            out += "\t";
            break;
          case "u": {
            const hex = text.slice(i, i + 4);
            if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail("Invalid \\uXXXX escape");
            out += String.fromCharCode(parseInt(hex, 16));
            i += 4;
            break;
          }
          default:
            fail("Invalid \\escape");
        }
        start = i;
        continue;
      }
      if (c < 0x20) fail("Invalid control character");
      i++;
    }
    return fail("Unterminated string");
  };
  const parseNumber = (): number | PyFloat => {
    const m = /^-?(?:0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?/.exec(text.slice(i, i + 400));
    if (!m) return fail("Expecting value");
    i += m[0].length;
    if (m[1] !== undefined || m[2] !== undefined) return new PyFloat(Number(m[0]));
    const v = Number(m[0]);
    return Object.is(v, -0) ? 0 : v;
  };
  const parseValue = (): PyValue => {
    ws();
    if (i >= n) fail("Expecting value");
    const c = text[i];
    if (c === "{") {
      i++;
      const d = new PyDict();
      ws();
      if (text[i] === "}") {
        i++;
        return d;
      }
      for (;;) {
        ws();
        if (text[i] !== '"') fail("Expecting property name enclosed in double quotes");
        const k = parseString();
        ws();
        if (text[i] !== ":") fail("Expecting ':' delimiter");
        i++;
        d.set(k, parseValue());
        ws();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "}") {
          i++;
          return d;
        }
        fail("Expecting ',' delimiter");
      }
    }
    if (c === "[") {
      i++;
      const arr: PyValue[] = [];
      ws();
      if (text[i] === "]") {
        i++;
        return arr;
      }
      for (;;) {
        arr.push(parseValue());
        ws();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "]") {
          i++;
          return arr;
        }
        fail("Expecting ',' delimiter");
      }
    }
    if (c === '"') return parseString();
    if (text.startsWith("true", i)) {
      i += 4;
      return true;
    }
    if (text.startsWith("false", i)) {
      i += 5;
      return false;
    }
    if (text.startsWith("null", i)) {
      i += 4;
      return null;
    }
    if (text.startsWith("NaN", i)) {
      i += 3;
      return new PyFloat(NaN);
    }
    if (text.startsWith("Infinity", i)) {
      i += 8;
      return new PyFloat(Infinity);
    }
    if (text.startsWith("-Infinity", i)) {
      i += 9;
      return new PyFloat(-Infinity);
    }
    return parseNumber();
  };
  const v = parseValue();
  ws();
  if (i < n) fail("Extra data");
  return v;
}

/** Convert a PyValue to plain JSON-compatible JS (for display or export). */
export function toPlain(v: PyValue): unknown {
  if (v instanceof PyFloat) return v.value;
  if (Array.isArray(v)) return v.map(toPlain);
  if (v instanceof PyDict) {
    const o: Record<string, unknown> = {};
    for (const [k, x] of v) o[k] = toPlain(x);
    return o;
  }
  return v;
}
