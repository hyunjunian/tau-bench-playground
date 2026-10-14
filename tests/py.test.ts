/**
 * Python semantics the tools rely on. Expected strings were recorded from
 * CPython 3.13 running the original tau-bench tools.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Domain, type DomainName, type ToolSpec } from "../src/tau/env";
import { PyDict, PyFloat, type PyValue, floatRepr, jsonDumps, parseJson, pySum, roundFloat } from "../src/tau/py";

const ROOT = join(__dirname, "..");
const read = (p: string) => readFileSync(join(ROOT, p), "utf8");
const domain = (name: DomainName) =>
  new Domain(name, read(`public/data/${name}/db.json`), JSON.parse(read(`public/data/${name}/tools.json`)) as ToolSpec[], "");
const retail = domain("retail");
const airline = domain("airline");

const call = (d: Domain, name: string, kwargs: Record<string, PyValue>) =>
  d.invokeTool(d.freshData(), name, new PyDict(Object.entries(kwargs)));

describe("calculate", () => {
  const cases: [string, string][] = [
    ["2+2", "4.0"],
    ["1/3", "0.33"],
    ["10/2", "5.0"],
    ["2**10", "1024.0"],
    ["2 ** -1", "0.5"],
    ["7//2", "3.0"],
    ["-7//2", "-4.0"],
    ["7.5//2", "3.0"],
    ["1/0", "Error: division by zero"],
    ["1.0/0", "Error: float division by zero"],
    ["1//0", "Error: integer division or modulo by zero"],
    ["1.5//0", "Error: float floor division by zero"],
    ["0**-1", "Error: 0.0 cannot be raised to a negative power"],
    ["", "Error: invalid syntax (<string>, line 0)"],
    [" ", "Error: invalid syntax (<string>, line 0)"],
    ["2 3", "Error: invalid syntax (<string>, line 1)"],
    ["(1+2", "Error: '(' was never closed (<string>, line 1)"],
    ["1+", "Error: invalid syntax (<string>, line 1)"],
    [")", "Error: unmatched ')' (<string>, line 1)"],
    [
      "01",
      "Error: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers (<string>, line 1)",
    ],
    ["00", "0.0"],
    ["1..2", "Error: invalid syntax (<string>, line 1)"],
    ["1.", "1.0"],
    [".5", "0.5"],
    [" 2+2", "4.0"],
    ["2+2 ", "4.0"],
    ["(-8)**0.5", "Error: float() argument must be a string or a real number, not 'complex'"],
    ["10.0**400", "Error: (34, 'Numerical result out of range')"],
    ["10**400", "Error: int too large to convert to float"],
    ["-2**2", "-4.0"],
    ["2**3**2", "512.0"],
    ["--3", "3.0"],
    ["+-3", "-3.0"],
    ["2.675*1", "2.67"],
    ["0.125", "0.12"],
    ["0.375", "0.38"],
    ["1e5", "Error: invalid characters in expression"],
    ["12345678901234567890*98765432109876543210", "1.219326311370218e+39"],
    ["3 * (4 + 5) / 2", "13.5"],
    ["1/3*3", "1.0"],
    ["100 - 33.33", "66.67"],
    ["54.04 - 41.64", "12.4"],
    ["1/7", "0.14"],
  ];
  it.each(cases)("%j -> %j", (expression, expected) => {
    expect(call(retail, "calculate", { expression })).toBe(expected);
  });

  it("rejects non-string expressions like Python", () => {
    expect(call(retail, "calculate", { expression: 5 })).toBe("Error: 'int' object is not iterable");
    expect(call(retail, "calculate", { expression: null })).toBe("Error: 'NoneType' object is not iterable");
    expect(call(retail, "calculate", { expression: ["1"] })).toBe(
      "Error: eval() arg 1 must be a string, bytes or code object",
    );
  });
});

describe("argument binding errors", () => {
  it("matches CPython's TypeError messages", () => {
    expect(call(retail, "cancel_pending_order", { order_id: "x" })).toBe(
      "Error: CancelPendingOrder.invoke() missing 1 required positional argument: 'reason'",
    );
    expect(call(retail, "cancel_pending_order", {})).toBe(
      "Error: CancelPendingOrder.invoke() missing 2 required positional arguments: 'order_id' and 'reason'",
    );
    expect(call(retail, "cancel_pending_order", { order_id: "x", reason: "y", foo: 1 })).toBe(
      "Error: CancelPendingOrder.invoke() got an unexpected keyword argument 'foo'",
    );
    expect(call(airline, "book_reservation", { user_id: "u" })).toBe(
      "Error: BookReservation.invoke() missing 10 required positional arguments: 'origin', 'destination', 'flight_type', 'cabin', 'flights', 'passengers', 'payment_methods', 'total_baggages', 'nonfree_baggages', and 'insurance'",
    );
    expect(call(retail, "cancel_pending_order", { data: 1, order_id: "#W2378156", reason: "no longer needed" })).toBe(
      "Error: tau_bench.envs.retail.tools.cancel_pending_order.CancelPendingOrder.invoke() got multiple values for keyword argument 'data'",
    );
    expect(call(retail, "cancel_pending_order", { order_id: ["#W2378156"], reason: "x" })).toBe(
      "Error: unhashable type: 'list'",
    );
    expect(call(retail, "get_order_details", { order_id: new PyDict([["a", 1]]) })).toBe("Error: unhashable type: 'dict'");
  });
});

describe("number formatting and arithmetic", () => {
  it("json.dumps keeps float formatting", () => {
    const v = parseJson(
      '{"a": 46.0, "b": 1e16, "c": 1e-5, "d": 0.0001, "e": -0.0, "f": 1e15, "g": 123456789.123, "h": "\\u00e9\\u2028\\"\\\\/\\n\\t\\u0001", "i": true, "j": null}',
    );
    expect(jsonDumps(v)).toBe(
      '{"a": 46.0, "b": 1e+16, "c": 1e-05, "d": 0.0001, "e": -0.0, "f": 1000000000000000.0, "g": 123456789.123, "h": "\\u00e9\\u2028\\"\\\\/\\n\\t\\u0001", "i": true, "j": null}',
    );
  });

  it("round() is correctly rounded, ties to even", () => {
    expect([2.675, 0.125, 0.375, -0.125, 1.005].map((x) => roundFloat(x, 2))).toEqual([2.67, 0.12, 0.38, -0.12, 1.0]);
  });

  it("sum() uses Neumaier summation for floats (Python 3.12+)", () => {
    const f = (x: number) => new PyFloat(x);
    expect(floatRepr((pySum([f(0.1), f(0.2), f(0.3)]) as PyFloat).value)).toBe("0.6");
    expect(floatRepr((pySum([1, f(2.5), f(0.1), f(0.2)]) as PyFloat).value)).toBe("3.8");
    expect(pySum([1, 2, true])).toBe(4);
  });

  it("str(float) matches repr", () => {
    expect([1e16, 1e-5, 100.0, -0.0].map(floatRepr)).toEqual(["1e+16", "1e-05", "100.0", "-0.0"]);
  });
});
