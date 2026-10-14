"""Record reference traces from the original Python tau-bench to test the TypeScript port.

Usage:
    python scripts/make_reference.py /path/to/tau-bench tests/reference [--episodes 300] [--seed 0] [--full]

Writes:
    gt_traces.json    every task's ground-truth actions replayed: observation hashes + final DB hash
    fuzz_traces.json  randomized action sequences (valid, invalid and malformed arguments)
    reward_cases.json trajectories scored by the real Env.calculate_reward (exact, dropped
                      write, int->float arguments, extra actions; outputs present/missing)

Hashes are sha256 of the observation string / of the canonical serialization
(sorted keys, Python float repr) of the database records that changed, which
tests/port.test.ts recomputes with the TypeScript tools. With --full the raw observations are
stored instead of hashes (useful for debugging a mismatch).
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import types


def canonical(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, list):
        return "[" + ",".join(canonical(v) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canonical(v) for k, v in sorted(x.items())) + "}"
    raise TypeError(type(x))


def sha(s):
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def strict_eq(a, b):
    """Equality that, like tau-bench's hash, tells 1, 1.0 and True apart."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_eq(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return repr(a) == repr(b)
    return a == b


def delta_canonical(before, after):
    """Canonical text of the records that differ (same format as deltaCanonical in src/tau/py.ts)."""
    lines = []
    for collection in sorted(set(before) | set(after)):
        b, a = before.get(collection), after.get(collection)
        if not isinstance(b, dict) or not isinstance(a, dict):
            if b is None or a is None or not strict_eq(b, a):
                lines.append(f"{collection}/=" + ("<deleted>" if a is None else canonical(a)))
            continue
        for key in sorted(set(b) | set(a)):
            if key not in b or key not in a or not strict_eq(b[key], a[key]):
                lines.append(f"{collection}/{key}=" + ("<deleted>" if key not in a else canonical(a[key])))
    return "\n".join(lines)


class Runner:
    """Invokes tools exactly like Env.step does (minus the user simulator)."""

    def __init__(self, domain):
        tools = __import__(f"tau_bench.envs.{domain}.tools", fromlist=["ALL_TOOLS"])
        data_mod = __import__(f"tau_bench.envs.{domain}.data", fromlist=["load_data"])
        self.tools_map = {t.get_info()["function"]["name"]: t for t in tools.ALL_TOOLS}
        self._pristine = data_mod.load_data()

    def fresh(self):
        return copy.deepcopy(self._pristine)

    def final_hash(self, data):
        return sha(delta_canonical(self._pristine, data))

    def step(self, data, name, kwargs):
        if name not in self.tools_map:
            return f"Unknown action {name}"
        try:
            observation = self.tools_map[name].invoke(data=data, **kwargs)
        except Exception as e:
            observation = f"Error: {e}"
        if observation is None:
            return None
        return observation


# ---------------------------------------------------------------------------
# random action generation

JUNK = [None, True, False, 0, 1, -1, 2.5, 3.0, "", "abc", "#W0000000", [], ["a"], [1, 2], {}, {"a": 1}, 1e20, "１２", "1x"]


def maybe_junk(rng, value, p=0.08):
    return copy.deepcopy(rng.choice(JUNK)) if rng.random() < p else value


def random_expression(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.35:
        kind = rng.random()
        if kind < 0.45:
            return str(rng.randint(0, 2000))
        if kind < 0.85:
            return f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}"
        return rng.choice(["0", "00", "007", ".5", "5.", "1e3", "10**400", "0.125", "2.675"])
    if r < 0.45:
        return "(" + random_expression(rng, depth + 1) + ")"
    if r < 0.55:
        return rng.choice(["-", "+", "- "]) + random_expression(rng, depth + 1)
    op = rng.choice(["+", "-", "*", "/", "//", "**", " + ", " * ", " / "])
    if op.strip() == "**":
        return random_expression(rng, depth + 1) + op + str(rng.randint(-3, 4))
    return random_expression(rng, depth + 1) + op + random_expression(rng, depth + 1)


def broken_expression(rng):
    return rng.choice(
        ["", " ", "2 3", "(1+2", "1+", ")", "1..2", "2(3)", "()", "(1)(2)", "1/0", "1.0/0", "1//0", "1.5//0",
         "0**-1", "(-8)**0.5", "10.0**400", "abc", "1 % 2", "2**3**2", "-2**2", "--3", "1 . 2", "(())",
         "12345678901234567890*98765432109876543210", "10**400//10**399", "7.5//-2", "-7//2", "1/3", "0.1+0.2"]
    )


def mutate_kwargs(rng, kwargs, params):
    r = rng.random()
    if r < 0.03 and kwargs:
        del kwargs[rng.choice(list(kwargs))]
    elif r < 0.05:
        kwargs["extra_arg"] = "x"
    elif r < 0.06:
        kwargs["data"] = {}
    return kwargs


class RetailFuzzer:
    def __init__(self, rng, data):
        self.rng = rng
        self.users = list(data["users"].items())
        self.orders = data["orders"]
        self.products = data["products"]
        self.by_status = {}
        for oid, o in self.orders.items():
            self.by_status.setdefault(o["status"], []).append(oid)

    def order_id(self, statuses=None):
        rng = self.rng
        if rng.random() < 0.08:
            return rng.choice(["#W9999999", "W2378156", "#w2378156", 2378156])
        pool = []
        for s in statuses or self.by_status:
            pool += self.by_status.get(s, [])
        return rng.choice(pool or list(self.orders))

    def payment_method(self, data, order_id):
        rng = self.rng
        order = data["orders"].get(order_id) if isinstance(order_id, str) else None
        if order and rng.random() < 0.85:
            pms = list(data["users"][order["user_id"]]["payment_methods"])
            return rng.choice(pms)
        uid, profile = rng.choice(self.users)
        return rng.choice(list(profile["payment_methods"]) + ["gift_card_0000000", "credit_card_1"])

    def item_lists(self, data, order_id, same_product=True):
        rng = self.rng
        order = data["orders"].get(order_id) if isinstance(order_id, str) else None
        if not order:
            return ["1151293680"], ["7706410293"]
        items = order["items"]
        k = rng.randint(0, min(3, len(items)))
        chosen = rng.sample(items, k) if k else []
        if chosen and rng.random() < 0.1:
            chosen.append(chosen[0])  # duplicate
        item_ids = [it["item_id"] for it in chosen]
        new_ids = []
        for it in chosen:
            variants = list(self.products[it["product_id"]]["variants"].values())
            if rng.random() < 0.08:
                other = rng.choice(list(self.products.values()))
                variants = list(other["variants"].values())
            v = rng.choice(variants)
            new_ids.append(v["item_id"])
        if rng.random() < 0.05:
            new_ids = new_ids[:-1]
        if rng.random() < 0.05:
            item_ids = item_ids + ["0000000000"]
        return item_ids, new_ids

    def address_kwargs(self):
        rng = self.rng
        return {
            "address1": f"{rng.randint(1, 999)} {rng.choice(['Main St', 'Oak Ave', 'Elm Street'])}",
            "address2": rng.choice(["", f"Suite {rng.randint(1, 999)}", "Apt 1"]),
            "city": rng.choice(["Austin", "Denver", "Fort Worth"]),
            "state": rng.choice(["TX", "CO", "NY"]),
            "country": "USA",
            "zip": maybe_junk(rng, str(rng.randint(10000, 99999))),
        }

    def action(self, data):
        rng = self.rng
        name = rng.choice(
            ["calculate", "cancel_pending_order", "exchange_delivered_order_items", "find_user_id_by_email",
             "find_user_id_by_name_zip", "get_order_details", "get_product_details", "get_user_details",
             "list_all_product_types", "modify_pending_order_address", "modify_pending_order_items",
             "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items", "think",
             "transfer_to_human_agents"]
            + ["cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_items",
               "modify_pending_order_payment", "return_delivered_order_items"] * 2
        )
        if rng.random() < 0.01:
            name = "nonexistent_tool"
        uid, profile = rng.choice(self.users)
        if name == "calculate":
            kw = {"expression": random_expression(rng) if rng.random() < 0.8 else broken_expression(rng)}
            kw["expression"] = maybe_junk(rng, kw["expression"], 0.05)
        elif name == "cancel_pending_order":
            kw = {"order_id": maybe_junk(rng, self.order_id(["pending"])),
                  "reason": rng.choice(["no longer needed", "ordered by mistake", "other", None])}
        elif name in ("exchange_delivered_order_items", "modify_pending_order_items"):
            oid = maybe_junk(rng, self.order_id(["delivered"] if name.startswith("exchange") else ["pending"]))
            item_ids, new_ids = self.item_lists(data, oid)
            kw = {"order_id": oid, "item_ids": maybe_junk(rng, item_ids, 0.05),
                  "new_item_ids": maybe_junk(rng, new_ids, 0.05),
                  "payment_method_id": maybe_junk(rng, self.payment_method(data, oid), 0.05)}
        elif name == "find_user_id_by_email":
            email = profile["email"]
            kw = {"email": maybe_junk(rng, rng.choice([email, email.upper(), "nobody@example.com"]))}
        elif name == "find_user_id_by_name_zip":
            kw = {"first_name": maybe_junk(rng, rng.choice([profile["name"]["first_name"], profile["name"]["first_name"].lower()])),
                  "last_name": maybe_junk(rng, profile["name"]["last_name"]),
                  "zip": maybe_junk(rng, rng.choice([profile["address"]["zip"], "00000"]))}
        elif name == "get_order_details":
            kw = {"order_id": maybe_junk(rng, self.order_id())}
        elif name == "get_product_details":
            pid = rng.choice(list(self.products))
            kw = {"product_id": maybe_junk(rng, rng.choice([pid, "1151293680", "0"]))}
        elif name == "get_user_details":
            kw = {"user_id": maybe_junk(rng, rng.choice([uid, "nobody_123"]))}
        elif name == "list_all_product_types":
            kw = {}
        elif name == "modify_pending_order_address":
            kw = {"order_id": maybe_junk(rng, self.order_id(["pending", "processed"])), **self.address_kwargs()}
        elif name == "modify_pending_order_payment":
            oid = maybe_junk(rng, self.order_id(["pending"]))
            kw = {"order_id": oid, "payment_method_id": maybe_junk(rng, self.payment_method(data, oid))}
        elif name == "modify_user_address":
            kw = {"user_id": maybe_junk(rng, uid), **self.address_kwargs()}
        elif name == "return_delivered_order_items":
            oid = maybe_junk(rng, self.order_id(["delivered"]))
            item_ids, _ = self.item_lists(data, oid)
            kw = {"order_id": oid, "item_ids": maybe_junk(rng, item_ids, 0.05),
                  "payment_method_id": maybe_junk(rng, self.payment_method(data, oid))}
        elif name == "think":
            kw = {"thought": "Let me think."}
        elif name == "transfer_to_human_agents":
            kw = {"summary": "User needs help."}
        else:
            kw = {"foo": 1}
        return name, mutate_kwargs(rng, kw, None)


class AirlineFuzzer:
    CABINS = ["basic_economy", "economy", "business"]

    def __init__(self, rng, data):
        self.rng = rng
        self.users = list(data["users"])
        self.flights = data["flights"]
        self.routes = sorted({(f["origin"], f["destination"]) for f in self.flights.values()})
        self.airports = sorted({o for o, _ in self.routes})

    def date(self):
        rng = self.rng
        if rng.random() < 0.06:
            return rng.choice(["2024-05-1x", "2024-05-3", 20240515, None, "2024-5-15", "2024-05-0 "])
        return f"2024-05-{rng.randint(1, 30):02d}"

    def user(self, data):
        return self.rng.choice(self.users) if self.rng.random() > 0.05 else "nobody_123"

    def reservation_id(self, data, user_id=None):
        rng = self.rng
        if rng.random() < 0.05:
            return rng.choice(["ZZZZZZ", "HATHAT", 123456])
        if user_id in data["users"] and data["users"][user_id]["reservations"] and rng.random() < 0.7:
            return rng.choice(data["users"][user_id]["reservations"])
        return rng.choice(list(data["reservations"]))

    def available_flights(self, data, origin, destination, date):
        out = []
        for f in data["flights"].values():
            if f["origin"] == origin and f["destination"] == destination and date in f["dates"]:
                if f["dates"][date]["status"] == "available":
                    out.append(f)
        return out

    def passengers(self, data, user_id, n):
        rng = self.rng
        user = data["users"].get(user_id)
        base = [{"first_name": user["name"]["first_name"], "last_name": user["name"]["last_name"], "dob": user["dob"]}] if user else []
        pool = base + (user.get("saved_passengers", []) if user else [])
        out = [copy.deepcopy(rng.choice(pool)) if pool and rng.random() < 0.6 else
               {"first_name": rng.choice(["Ava", "Noah"]), "last_name": rng.choice(["Kim", "Lee"]), "dob": "1990-01-01"}
               for _ in range(n)]
        if rng.random() < 0.05:
            out = maybe_junk(rng, out, 1.0)
        return out

    def pay(self, data, user_id, total):
        rng = self.rng
        user = data["users"].get(user_id)
        if not user or rng.random() < 0.1:
            return [{"payment_id": rng.choice(["credit_card_0", "certificate_1"]), "amount": total}]
        methods = list(user["payment_methods"].values())
        rng.shuffle(methods)
        out = []
        remaining = total
        for m in methods[: rng.randint(1, 3)]:
            if m["source"] in ("gift_card", "certificate"):
                amt = min(m["amount"], remaining) if rng.random() < 0.8 else m["amount"] + 1
            else:
                amt = remaining
            if amt <= 0:
                continue
            out.append({"payment_id": m["id"], "amount": float(amt) if rng.random() < 0.1 else amt})
            remaining -= amt
        if remaining > 0 and rng.random() < 0.7:
            cards = [m for m in methods if m["source"] == "credit_card"]
            if cards:
                out.append({"payment_id": cards[0]["id"], "amount": remaining})
        if rng.random() < 0.05 and out:
            out.append(copy.deepcopy(out[0]))
        return out

    def book(self, data):
        rng = self.rng
        user_id = self.user(data)
        origin, destination = rng.choice(self.routes)
        date = f"2024-05-{rng.randint(10, 30):02d}"
        cabin = rng.choice(self.CABINS + (["first"] if rng.random() < 0.05 else []))
        flights = []
        options = self.available_flights(data, origin, destination, date)
        if options and rng.random() < 0.85:
            f = rng.choice(options)
            flights.append({"flight_number": f["flight_number"], "date": date})
        else:
            f = rng.choice(list(data["flights"].values()))
            flights.append({"flight_number": f["flight_number"], "date": self.date()})
        flight_type = "one_way"
        if rng.random() < 0.4:
            back = self.available_flights(data, destination, origin, f"2024-05-{min(30, int(date[-2:]) + rng.randint(0, 5)):02d}")
            if back:
                b = rng.choice(back)
                flights.append({"flight_number": b["flight_number"], "date": f"2024-05-{min(30, int(date[-2:]) + rng.randint(0, 5)):02d}"})
                flight_type = "round_trip"
        if rng.random() < 0.05:
            flights[0]["note"] = "window seat"
        n = rng.randint(1, 4)
        passengers = self.passengers(data, user_id, n)
        insurance = rng.choice(["yes", "no"])
        nonfree = rng.randint(0, 3)
        total = 0
        for fl in flights:
            try:
                total += data["flights"][fl["flight_number"]]["dates"][fl["date"]]["prices"][cabin] * n
            except Exception:
                total += 100
        if insurance == "yes":
            total += 30 * n
        total += 50 * nonfree
        if rng.random() < 0.15:
            total += rng.choice([1, -1, 0.5])
        kw = {
            "user_id": user_id, "origin": origin, "destination": destination, "flight_type": flight_type,
            "cabin": cabin, "flights": maybe_junk(rng, flights, 0.04), "passengers": passengers,
            "payment_methods": maybe_junk(rng, self.pay(data, user_id, total), 0.04),
            "total_baggages": nonfree + rng.randint(0, 2), "nonfree_baggages": maybe_junk(rng, nonfree, 0.05),
            "insurance": maybe_junk(rng, insurance, 0.03),
        }
        return kw

    def update_flights(self, data):
        rng = self.rng
        user_id = rng.choice(self.users)
        rid = self.reservation_id(data, user_id)
        res = data["reservations"].get(rid) if isinstance(rid, str) else None
        cabin = rng.choice(self.CABINS)
        flights = []
        if res:
            if rng.random() < 0.5:
                cabin = res["cabin"]
            for seg in res["flights"]:
                if rng.random() < 0.5:
                    flights.append({"flight_number": seg["flight_number"], "date": seg["date"]})
                else:
                    date = f"2024-05-{rng.randint(14, 30):02d}"
                    opts = self.available_flights(data, seg["origin"], seg["destination"], date)
                    if opts:
                        flights.append({"flight_number": rng.choice(opts)["flight_number"], "date": date})
                    else:
                        flights.append({"flight_number": seg["flight_number"], "date": date})
            user = data["users"][res["user_id"]]
            payment_id = rng.choice(list(user["payment_methods"]))
        else:
            flights = [{"flight_number": "HAT001", "date": "2024-05-20"}]
            payment_id = "credit_card_0"
        return {"reservation_id": rid, "cabin": cabin, "flights": maybe_junk(rng, flights, 0.04),
                "payment_id": maybe_junk(rng, payment_id, 0.04)}

    def action(self, data):
        rng = self.rng
        name = rng.choice(
            ["book_reservation", "calculate", "cancel_reservation", "get_reservation_details", "get_user_details",
             "list_all_airports", "search_direct_flight", "search_onestop_flight", "send_certificate", "think",
             "transfer_to_human_agents", "update_reservation_baggages", "update_reservation_flights",
             "update_reservation_passengers"]
            + ["book_reservation", "update_reservation_flights", "update_reservation_baggages", "cancel_reservation",
               "search_onestop_flight"] * 2
        )
        user_id = self.user(data)
        if name == "book_reservation":
            kw = self.book(data)
        elif name == "calculate":
            kw = {"expression": random_expression(rng) if rng.random() < 0.8 else broken_expression(rng)}
        elif name == "cancel_reservation":
            kw = {"reservation_id": self.reservation_id(data, user_id)}
        elif name == "get_reservation_details":
            kw = {"reservation_id": self.reservation_id(data, user_id)}
        elif name == "get_user_details":
            kw = {"user_id": maybe_junk(rng, user_id)}
        elif name == "list_all_airports":
            kw = {}
        elif name in ("search_direct_flight", "search_onestop_flight"):
            o, d = rng.choice(self.routes) if rng.random() < 0.6 else (rng.choice(self.airports), rng.choice(self.airports))
            kw = {"origin": maybe_junk(rng, o, 0.03), "destination": d, "date": self.date()}
        elif name == "send_certificate":
            kw = {"user_id": user_id, "amount": maybe_junk(rng, rng.choice([50, 100, 150.0, 200]), 0.1)}
        elif name == "think":
            kw = {"thought": "Considering options."}
        elif name == "transfer_to_human_agents":
            kw = {"summary": "Needs a human."}
        elif name == "update_reservation_baggages":
            rid = self.reservation_id(data, user_id)
            res = data["reservations"].get(rid) if isinstance(rid, str) else None
            pms = list(data["users"][res["user_id"]]["payment_methods"]) if res else ["credit_card_0"]
            cur = res["nonfree_baggages"] if res else 0
            nonfree = maybe_junk(rng, cur + rng.randint(-1, 3), 0.06)
            kw = {"reservation_id": rid, "total_baggages": rng.randint(0, 6), "nonfree_baggages": nonfree,
                  "payment_id": maybe_junk(rng, rng.choice(pms), 0.04)}
        elif name == "update_reservation_flights":
            kw = self.update_flights(data)
        else:  # update_reservation_passengers
            rid = self.reservation_id(data, user_id)
            res = data["reservations"].get(rid) if isinstance(rid, str) else None
            n = len(res["passengers"]) if res else 1
            if rng.random() < 0.15:
                n += 1
            kw = {"reservation_id": rid, "passengers": self.passengers(data, user_id, n)}
        if rng.random() < 0.01:
            name = "nonexistent_tool"
        return name, mutate_kwargs(rng, kw, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tau_bench")
    ap.add_argument("out_dir")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--train-reward-tasks", type=int, default=40)
    args = ap.parse_args()

    sys.modules["litellm"] = types.SimpleNamespace(completion=None, provider_list=[])
    sys.path.insert(0, os.path.abspath(args.tau_bench))
    os.makedirs(args.out_dir, exist_ok=True)
    enc = (lambda s: s) if args.full else sha

    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
    from tau_bench.envs.retail.tasks_dev import TASKS_DEV
    from tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TEST

    runners = {"retail": Runner("retail"), "airline": Runner("airline")}

    gt = []
    for domain, split, tasks in [("retail", "test", TASKS_TEST), ("retail", "train", TASKS_TRAIN),
                                 ("retail", "dev", TASKS_DEV), ("airline", "test", AIRLINE_TEST)]:
        runner = runners[domain]
        for index, task in enumerate(tasks):
            data = runner.fresh()
            obs = [enc(runner.step(data, a.name, copy.deepcopy(a.kwargs))) for a in task.actions]
            gt.append({"domain": domain, "split": split, "index": index, "observations": obs,
                       "final": runner.final_hash(data)})
        print(f"gt {domain}/{split}: {len(tasks)} tasks")
    with open(os.path.join(args.out_dir, "gt_traces.json"), "w") as f:
        json.dump(gt, f, separators=(",", ":"))

    fuzz = []
    rng = random.Random(args.seed)
    for ep in range(args.episodes):
        domain = "retail" if ep % 2 == 0 else "airline"
        runner = runners[domain]
        data = runner.fresh()
        fuzzer = (RetailFuzzer if domain == "retail" else AirlineFuzzer)(rng, data)
        steps = []
        for _ in range(args.steps):
            name, kwargs = fuzzer.action(data)
            kwargs_json = json.dumps(kwargs)
            observation = runner.step(data, name, json.loads(kwargs_json))
            steps.append({"name": name, "kwargs": kwargs_json,
                          "observation": None if observation is None else enc(observation)})
            if observation is None:
                break
        fuzz.append({"domain": domain, "steps": steps, "final": runner.final_hash(data)})
    with open(os.path.join(args.out_dir, "fuzz_traces.json"), "w") as f:
        json.dump(fuzz, f, separators=(",", ":"))
    n_steps = sum(len(e["steps"]) for e in fuzz)
    print(f"fuzz: {len(fuzz)} episodes, {n_steps} actions")

    cases = reward_cases(rng, runners, {
        ("retail", "test"): TASKS_TEST, ("retail", "train"): TASKS_TRAIN[: args.train_reward_tasks],
        ("retail", "dev"): TASKS_DEV, ("airline", "test"): AIRLINE_TEST,
    })
    with open(os.path.join(args.out_dir, "reward_cases.json"), "w") as f:
        json.dump(cases, f, separators=(",", ":"))
    print(f"reward: {len(cases)} cases, {sum(c['expected']['reward'] for c in cases):.0f} with reward 1")


def floats_for_ints(x):
    """Turn every int into a float (e.g. amount 250 -> 250.0); tau-bench's hash tells them apart."""
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return float(x)
    if isinstance(x, list):
        return [floats_for_ints(v) for v in x]
    if isinstance(x, dict):
        return {k: floats_for_ints(v) for k, v in x.items()}
    return x


def reward_cases(rng, runners, splits):
    """Score agent trajectories with the real Env.calculate_reward."""
    from tau_bench.envs.airline import MockAirlineDomainEnv
    from tau_bench.envs.retail import MockRetailDomainEnv
    from tau_bench.types import Action

    envs = {}
    cases = []
    for (domain, split), tasks in splits.items():
        if (domain, split) not in envs:
            cls = MockRetailDomainEnv if domain == "retail" else MockAirlineDomainEnv
            envs[(domain, split)] = cls(user_strategy="human", task_split=split, task_index=0)
        env = envs[(domain, split)]
        for index, task in enumerate(tasks):
            gt = [(a.name, a.kwargs) for a in task.actions if a.name not in env.terminate_tools]
            writes = [i for i, (n, _) in enumerate(gt) if not n.startswith(("get_", "find_", "list_", "search_", "calculate", "think"))]
            variants = [("exact", gt)]
            if writes:
                drop = rng.choice(writes)
                variants.append(("drop_write", [a for i, a in enumerate(gt) if i != drop]))
                variants.append(("float_args", [(n, floats_for_ints(k)) if i in writes else (n, k) for i, (n, k) in enumerate(gt)]))
            variants.append(("reordered_reads_extra", [("think", {"thought": "checking"})] + gt))
            for label, actions in variants:
                outputs = list(task.outputs)
                responds = []
                if outputs:
                    style = rng.choice(["all", "all_commas", "missing_one"])
                    shown = outputs if style != "missing_one" else outputs[:-1]
                    if style == "all_commas":
                        shown = [o[:-6] + "," + o[-6:] if len(o) > 6 and o.replace(".", "").isdigit() else o for o in shown]
                    responds = ["The values are: " + "; ".join(s.upper() for s in shown) + "."]
                env.data = env.data_load_func()
                env.task = tasks[index]
                env.actions = []
                for name, kwargs in actions:
                    env.step(Action(name=name, kwargs=copy.deepcopy(kwargs)))
                for c in responds:
                    env.actions.append(Action(name="respond", kwargs={"content": c}))
                data_hash = env.get_data_hash()
                res = env.calculate_reward()
                r_actions = data_hash == env.get_data_hash()
                cases.append({
                    "domain": domain, "split": split, "index": index, "variant": label,
                    "actions": [{"name": n, "kwargs": json.dumps(k)} for n, k in actions],
                    "responds": responds,
                    "expected": {"reward": res.reward, "r_actions": r_actions,
                                 "r_outputs": getattr(res.info, "r_outputs", None)},
                })
        print(f"reward cases {domain}/{split}: {len(tasks)} tasks")
    return cases


if __name__ == "__main__":
    main()
