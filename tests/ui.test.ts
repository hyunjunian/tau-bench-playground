/** Helpers behind the UI: tool-form suggestions, task links and stored-settings cleanup. */
import { describe, expect, it } from "vitest";
import { makeT } from "../src/i18n";
import { sanitize, taskFromHash, taskHash } from "../src/state/settings";
import { parseTasks } from "../src/tau/env";
import { PyDict } from "../src/tau/py";
import { type SuggestionSource, collectSuggestions, suggestionsFor } from "../src/ui/suggest";
import { domainFor, read } from "./helpers";

const retail = domainFor("retail");

function observe(name: string, kwargs: Record<string, string>): SuggestionSource {
  const observation = retail.invokeTool(retail.freshData(), name, new PyDict(Object.entries(kwargs)))!;
  return { kind: "tool", name, observation, error: observation.startsWith("Error") };
}

describe("tool-form suggestions", () => {
  // retail test #0: exchange a keyboard and a thermostat in order #W2378156
  const task = parseTasks(read("public/data/retail/tasks_test.json"))[0];
  const exchange = task.actions.find((a) => a.name === "exchange_delivered_order_items")!;
  const sources: SuggestionSource[] = [
    { kind: "user", text: "Hi, I'm Yusuf Rossi from 19122, about order #W2378156." },
    observe("find_user_id_by_name_zip", { first_name: "Yusuf", last_name: "Rossi", zip: "19122" }),
    observe("get_user_details", { user_id: "yusuf_rossi_9620" }),
    observe("get_order_details", { order_id: "#W2378156" }),
    observe("get_product_details", { product_id: "1656367028" }),
    observe("get_product_details", { product_id: "4896585277" }),
    observe("get_order_details", { order_id: "#nope" }),
  ];
  const all = collectSuggestions(sources);

  it("offers the ids the ground truth needs", () => {
    expect(suggestionsFor(all, "user_id")).toContain("yusuf_rossi_9620");
    expect(suggestionsFor(all, "order_id")[0]).toBe("#W2378156");
    expect(suggestionsFor(all, "payment_method_id")).toContain(exchange.kwargs.get("payment_method_id"));
    for (const id of exchange.kwargs.get("new_item_ids") as string[]) expect(suggestionsFor(all, "new_item_ids")).toContain(id);
  });

  it("ranks an order's items first for item_ids and available variants first for new_item_ids", () => {
    const orderItems = suggestionsFor(all, "item_ids").slice(0, 5);
    for (const id of exchange.kwargs.get("item_ids") as string[]) expect(orderItems).toContain(id);
    const products = retail.pristine.get("products") as PyDict;
    const variant = (id: string) => {
      for (const p of products.values()) {
        const v = ((p as PyDict).get("variants") as PyDict).get(id);
        if (v) return v as PyDict;
      }
      return null;
    };
    const ranked = suggestionsFor(all, "new_item_ids");
    const firstUnavailable = ranked.findIndex((id) => variant(id)?.get("available") === false);
    const lastAvailable = ranked.map((id) => variant(id)?.get("available") === true).lastIndexOf(true);
    expect(firstUnavailable === -1 || firstUnavailable > lastAvailable).toBe(true);
  });

  it("ignores error observations and maps payment aliases", () => {
    expect(suggestionsFor(all, "order_id")).not.toContain("#nope");
    expect(suggestionsFor(all, "payment_id")).toEqual(suggestionsFor(all, "payment_method_id"));
  });
});

describe("translations", () => {
  it("inserts values literally, even ones that look like replacement patterns", () => {
    const text = makeT("en")("replayObsDiffers", { name: "get_x", recorded: "costs $$5 and $' and $& and {name}" });
    expect(text).toContain("costs $$5 and $' and $& and");
  });
});

describe("task links", () => {
  it("round-trips #domain/split/index", () => {
    expect(taskFromHash("#airline/test/7")).toEqual({ domain: "airline", split: "test", taskIndex: 7 });
    expect(taskFromHash("#/retail/train/42")).toEqual({ domain: "retail", split: "train", taskIndex: 42 });
    expect(taskHash({ domain: "retail", split: "dev", taskIndex: 3 })).toBe("#retail/dev/3");
  });

  it("rejects unknown domains and splits", () => {
    expect(taskFromHash("#airline/train/1")).toBeNull();
    expect(taskFromHash("#shop/test/1")).toBeNull();
    expect(taskFromHash("#retail/test/-1")).toBeNull();
    expect(taskFromHash("")).toBeNull();
  });
});

describe("stored settings", () => {
  it("drops values that would break requests", () => {
    const s = sanitize({
      domain: "airline",
      split: "train",
      keys: { google: "k", openrouter: 5, nope: "x" },
      customBaseUrl: null,
      batch: { from: 5, to: 2, trials: 0 },
      agent: { controller: "robot" },
    });
    expect(s.split).toBe("test");
    expect(s.keys).toEqual({ google: "k" });
    expect(typeof s.customBaseUrl).toBe("string");
    expect(s.batch).toEqual({ from: 5, to: 9, trials: 1 });
    expect(s.agent.controller).toBe("model");
    expect(sanitize({ theme: "neon" }).theme).toBe("system");
    expect(sanitize({ theme: "dark" }).theme).toBe("dark");
  });
});
