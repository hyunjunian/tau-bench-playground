/**
 * The `calculate` tool evaluates `str(round(float(eval(expression)), 2))` in
 * Python after checking the characters are in "0123456789+-*\/(). ". This is a
 * small Python-compatible evaluator for that subset (arbitrary precision ints,
 * true/floor division, right-associative `**`, Python's error messages).
 */
import {
  OverflowError_,
  PyException,
  type PyValue,
  SyntaxError_,
  TypeError_,
  ZeroDivisionError_,
  floatRepr,
  pyIn,
  pyIter,
  roundFloat,
} from "./py";

const ALLOWED = "0123456789+-*/(). ";

type Val =
  | { t: "int"; v: bigint }
  | { t: "float"; v: number }
  | { t: "tuple" }
  | { t: "complex" };

type Token =
  | { k: "num"; text: string; isFloat: boolean }
  | { k: "op"; text: "+" | "-" | "*" | "**" | "/" | "//" | "." }
  | { k: "lpar" }
  | { k: "rpar" };

function syntaxError(msg: string, line = 1): PyException {
  return SyntaxError_(`${msg} (<string>, line ${line})`);
}

function tokenize(src: string): Token[] {
  const tokens: Token[] = [];
  let depth = 0;
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === " ") {
      i++;
      continue;
    }
    const m = /^(?:\d+(?:\.\d*)?|\.\d+)/.exec(src.slice(i));
    if (m) {
      const text = m[0];
      const isFloat = text.includes(".");
      if (!isFloat && text.length > 1 && text[0] === "0" && /[1-9]/.test(text)) {
        throw syntaxError(
          "leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers",
        );
      }
      tokens.push({ k: "num", text, isFloat });
      i += text.length;
      continue;
    }
    if (c === "(") {
      depth++;
      tokens.push({ k: "lpar" });
      i++;
      continue;
    }
    if (c === ")") {
      if (depth === 0) throw syntaxError("unmatched ')'");
      depth--;
      tokens.push({ k: "rpar" });
      i++;
      continue;
    }
    if (c === "*" || c === "/") {
      if (src[i + 1] === c) {
        tokens.push({ k: "op", text: c === "*" ? "**" : "//" });
        i += 2;
      } else {
        tokens.push({ k: "op", text: c });
        i++;
      }
      continue;
    }
    if (c === "+" || c === "-" || c === ".") {
      tokens.push({ k: "op", text: c });
      i++;
      continue;
    }
    throw syntaxError("invalid syntax");
  }
  if (depth > 0) throw syntaxError("'(' was never closed");
  return tokens;
}

// --- arithmetic --------------------------------------------------------------

function typeOf(v: Val): string {
  return v.t;
}

function bitLength(x: bigint): number {
  return x === 0n ? 0 : x.toString(2).length;
}

function intToFloat(x: bigint): number {
  const f = Number(x);
  if (!Number.isFinite(f)) throw OverflowError_("int too large to convert to float");
  return f;
}

function asFloat(v: Val): number {
  if (v.t === "int") return intToFloat(v.v);
  if (v.t === "float") return v.v;
  throw new Error("unreachable");
}

function isNumeric(v: Val): v is { t: "int"; v: bigint } | { t: "float"; v: number } {
  return v.t === "int" || v.t === "float";
}

function unsupported(op: string, a: Val, b: Val): PyException {
  return TypeError_(`unsupported operand type(s) for ${op}: '${typeOf(a)}' and '${typeOf(b)}'`);
}

/** Correctly rounded int / int, like CPython's long_true_divide. */
function intTrueDiv(a: bigint, b: bigint): number {
  if (b === 0n) throw ZeroDivisionError_("division by zero");
  const limit = 1n << 53n;
  const absA = a < 0n ? -a : a;
  const absB = b < 0n ? -b : b;
  if (absA <= limit && absB <= limit) return Number(a) / Number(b);
  const negative = a < 0n !== b < 0n;
  if (absA === 0n) return negative ? -0 : 0;
  const shift = 60 - (bitLength(absA) - bitLength(absB));
  let q: bigint;
  let r: bigint;
  if (shift >= 0) {
    const n = absA << BigInt(shift);
    q = n / absB;
    r = n % absB;
  } else {
    const d = absB << BigInt(-shift);
    q = absA / d;
    r = absA % d;
  }
  if (r !== 0n) q |= 1n; // sticky bit below the rounding position
  const value = Number(q) * 2 ** -shift;
  if (!Number.isFinite(value)) throw OverflowError_("integer division result too large for a float");
  return negative ? -value : value;
}

function floatFloorDiv(vx: number, wx: number): number {
  if (wx === 0) throw ZeroDivisionError_("float floor division by zero");
  let mod = vx % wx;
  let div = (vx - mod) / wx;
  if (mod) {
    if (wx < 0 !== mod < 0) {
      mod += wx;
      div -= 1.0;
    }
  }
  if (div) {
    let floordiv = Math.floor(div);
    if (div - floordiv > 0.5) floordiv += 1.0;
    return floordiv;
  }
  const s = vx / wx;
  return s < 0 || Object.is(s, -0) ? -0 : 0;
}

function floatPow(iv: number, iw: number): Val {
  if (iw === 0) return { t: "float", v: 1.0 };
  if (Number.isNaN(iv)) return { t: "float", v: iv };
  if (Number.isNaN(iw)) return { t: "float", v: iv === 1.0 ? 1.0 : iw };
  if (iv === 0) {
    if (iw < 0) throw ZeroDivisionError_("0.0 cannot be raised to a negative power");
  }
  if (iv < 0 && Number.isFinite(iw) && iw !== Math.floor(iw)) return { t: "complex" };
  const r = Math.pow(iv, iw);
  if (!Number.isFinite(r) && Number.isFinite(iv) && Number.isFinite(iw)) {
    throw OverflowError_("(34, 'Numerical result out of range')");
  }
  return { t: "float", v: r };
}

function binop(op: string, a: Val, b: Val): Val {
  if (a.t === "tuple" || b.t === "tuple") {
    if (op === "+" && a.t === "tuple" && b.t === "tuple") return { t: "tuple" };
    if (op === "+" && a.t === "tuple") {
      throw TypeError_(`can only concatenate tuple (not "${typeOf(b)}") to tuple`);
    }
    if (op === "*" && (a.t === "int" || b.t === "int")) return { t: "tuple" };
    if (op === "*" && (a.t === "float" || b.t === "float")) {
      throw TypeError_("can't multiply sequence by non-int of type 'float'");
    }
    throw unsupported(op, a, b);
  }
  if (a.t === "complex" || b.t === "complex") {
    if (op === "//") throw unsupported(op, a, b);
    if (op === "/" && isNumeric(b) && asFloat(b) === 0) throw ZeroDivisionError_("division by zero");
    return { t: "complex" };
  }
  if (a.t === "int" && b.t === "int") {
    switch (op) {
      case "+":
        return { t: "int", v: a.v + b.v };
      case "-":
        return { t: "int", v: a.v - b.v };
      case "*":
        return { t: "int", v: a.v * b.v };
      case "/":
        return { t: "float", v: intTrueDiv(a.v, b.v) };
      case "//": {
        if (b.v === 0n) throw ZeroDivisionError_("integer division or modulo by zero");
        let q = a.v / b.v;
        if (a.v % b.v !== 0n && a.v < 0n !== b.v < 0n) q -= 1n;
        return { t: "int", v: q };
      }
      case "**": {
        if (b.v < 0n) return floatPow(intToFloat(a.v), intToFloat(b.v));
        const absBase = a.v < 0n ? -a.v : a.v;
        if (absBase > 1n && Number(b.v) * bitLength(absBase) > 10_000_000) {
          throw OverflowError_("int too large to convert to float");
        }
        return { t: "int", v: a.v ** b.v };
      }
    }
  }
  const x = asFloat(a);
  const y = asFloat(b);
  switch (op) {
    case "+":
      return { t: "float", v: x + y };
    case "-":
      return { t: "float", v: x - y };
    case "*":
      return { t: "float", v: x * y };
    case "/":
      if (y === 0) throw ZeroDivisionError_("float division by zero");
      return { t: "float", v: x / y };
    case "//":
      return { t: "float", v: floatFloorDiv(x, y) };
    case "**":
      return floatPow(x, y);
  }
  throw new Error(`unknown operator ${op}`);
}

function unary(op: "+" | "-", v: Val): Val {
  if (v.t === "int") return { t: "int", v: op === "-" ? -v.v : v.v };
  if (v.t === "float") return { t: "float", v: op === "-" ? -v.v : v.v };
  if (v.t === "complex") return v;
  throw TypeError_(`bad operand type for unary ${op}: '${typeOf(v)}'`);
}

// --- parser (evaluates while parsing; Python evaluates left to right) -------

function evaluate(src: string): Val {
  const stripped = src.replace(/^[ \t]+/, "");
  if (stripped.trim() === "") throw syntaxError("invalid syntax", 0);
  const tokens = tokenize(stripped);
  let pos = 0;
  const peek = () => tokens[pos];
  const isOp = (text: string) => {
    const t = tokens[pos];
    return t !== undefined && t.k === "op" && t.text === text;
  };
  const invalid = () => syntaxError("invalid syntax");

  // Parse first, evaluate afterwards: a syntax error anywhere wins over runtime errors.
  type Node =
    | { n: "num"; tok: Extract<Token, { k: "num" }> }
    | { n: "tuple" }
    | { n: "unary"; op: "+" | "-"; e: Node }
    | { n: "bin"; op: string; l: Node; r: Node }
    | { n: "call"; f: Node; arg: Node | null };

  const parseSum = (): Node => {
    let left = parseTerm();
    while (isOp("+") || isOp("-")) {
      const op = (tokens[pos++] as { text: string }).text;
      left = { n: "bin", op, l: left, r: parseTerm() };
    }
    return left;
  };
  const parseTerm = (): Node => {
    let left = parseFactor();
    while (isOp("*") || isOp("/") || isOp("//")) {
      const op = (tokens[pos++] as { text: string }).text;
      left = { n: "bin", op, l: left, r: parseFactor() };
    }
    return left;
  };
  const parseFactor = (): Node => {
    if (isOp("+") || isOp("-")) {
      const op = (tokens[pos++] as { text: "+" | "-" }).text;
      return { n: "unary", op, e: parseFactor() };
    }
    return parsePower();
  };
  const parsePower = (): Node => {
    const base = parsePrimary();
    if (isOp("**")) {
      pos++;
      return { n: "bin", op: "**", l: base, r: parseFactor() };
    }
    return base;
  };
  const parsePrimary = (): Node => {
    let node = parseAtom();
    while (peek()?.k === "lpar") {
      pos++;
      let arg: Node | null = null;
      if (peek()?.k !== "rpar") arg = parseSum();
      if (peek()?.k !== "rpar") throw invalid();
      pos++;
      node = { n: "call", f: node, arg };
    }
    return node;
  };
  const parseAtom = (): Node => {
    const t = peek();
    if (t === undefined) throw invalid();
    if (t.k === "num") {
      pos++;
      return { n: "num", tok: t };
    }
    if (t.k === "lpar") {
      pos++;
      if (peek()?.k === "rpar") {
        pos++;
        return { n: "tuple" };
      }
      const inner = parseSum();
      if (peek()?.k !== "rpar") throw invalid();
      pos++;
      return inner;
    }
    throw invalid();
  };

  const tree = parseSum();
  if (pos !== tokens.length) throw invalid();

  const run = (node: Node): Val => {
    switch (node.n) {
      case "num":
        return node.tok.isFloat
          ? { t: "float", v: Number(node.tok.text) }
          : { t: "int", v: BigInt(node.tok.text) };
      case "tuple":
        return { t: "tuple" };
      case "unary":
        return unary(node.op, run(node.e));
      case "bin": {
        const l = run(node.l);
        const r = run(node.r);
        return binop(node.op, l, r);
      }
      case "call": {
        const f = run(node.f);
        if (node.arg) run(node.arg);
        throw TypeError_(`'${typeOf(f)}' object is not callable`);
      }
    }
  };
  return run(tree);
}

/** Python's `float(x)` for the evaluator's values. */
function toFloat(v: Val): number {
  if (v.t === "int" || v.t === "float") return asFloat(v);
  throw TypeError_(`float() argument must be a string or a real number, not '${v.t}'`);
}

export function calculate(expression: PyValue): string {
  // `all(char in "0123456789+-*/(). " for char in expression)` runs outside the try.
  for (const ch of pyIter(expression)) {
    if (!pyIn(ch, ALLOWED)) return "Error: invalid characters in expression";
  }
  try {
    if (typeof expression !== "string") {
      throw TypeError_("eval() arg 1 must be a string, bytes or code object");
    }
    return floatRepr(roundFloat(toFloat(evaluate(expression)), 2));
  } catch (e) {
    if (e instanceof PyException) return `Error: ${e.message}`;
    throw e;
  }
}
