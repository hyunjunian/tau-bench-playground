/**
 * Line-by-line ports of the tau-bench tools (tau_bench/envs/{retail,airline}/tools).
 * Statement order matches the Python code, because a tool can mutate the
 * database before it fails and those partial writes count toward the reward.
 */
import { calculate } from "./calc";
import {
  PyDict,
  type PyValue,
  deepCopy,
  delItem,
  dict,
  dictUpdate,
  getItem as get,
  jsonDumps,
  lastTwo,
  listAppend,
  listExtend,
  pyAbs,
  pyAdd,
  pyCount,
  pyEq,
  pyGt,
  pyIn,
  pyInt,
  pyIter,
  pyLen,
  pyLower,
  pyLt,
  pyMax,
  pyMul,
  pyNeg,
  pyRound,
  pySorted,
  pyStr,
  pySub,
  pySum,
  pyZip,
  setItem,
  truthy,
} from "./py";

/** A tool returns the observation string, or null where Python returns None. */
export type ToolImpl = (data: PyDict, ...args: PyValue[]) => string | null;

const think: ToolImpl = () => "";
const transferToHumanAgents: ToolImpl = () => "Transfer successful";

function getDetails(collection: string, notFound: string): ToolImpl {
  return (data, id) => {
    const items = get(data, collection);
    if (pyIn(id, items)) return jsonDumps(get(items, id));
    return notFound;
  };
}

function firstItemWithId(order: PyValue, itemId: PyValue): PyValue {
  const matches = pyIter(get(order, "items")).filter((item) => pyEq(get(item, "item_id"), itemId));
  return get(matches, 0);
}

function variantAvailable(products: PyValue, productId: PyValue, newItemId: PyValue): boolean {
  return (
    pyIn(newItemId, get(get(products, productId), "variants")) &&
    truthy(get(get(get(get(products, productId), "variants"), newItemId), "available"))
  );
}

function address(
  address1: PyValue,
  address2: PyValue,
  city: PyValue,
  state: PyValue,
  country: PyValue,
  zip: PyValue,
): PyDict {
  return dict([
    ["address1", address1],
    ["address2", address2],
    ["city", city],
    ["state", state],
    ["country", country],
    ["zip", zip],
  ]);
}

// ---------------------------------------------------------------------------
// retail

export const RETAIL_TOOLS: Record<string, ToolImpl> = {
  calculate: (_data, expression) => calculate(expression),

  cancel_pending_order: (data, order_id, reason) => {
    const orders = get(data, "orders");
    if (!pyIn(order_id, orders)) return "Error: order not found";
    const order = get(orders, order_id);
    if (!pyEq(get(order, "status"), "pending")) return "Error: non-pending order cannot be cancelled";

    if (!pyIn(reason, ["no longer needed", "ordered by mistake"])) return "Error: invalid reason";

    const refunds: PyValue[] = [];
    for (const payment of pyIter(get(order, "payment_history"))) {
      const payment_id = get(payment, "payment_method_id");
      const refund = dict([
        ["transaction_type", "refund"],
        ["amount", get(payment, "amount")],
        ["payment_method_id", payment_id],
      ]);
      refunds.push(refund);
      if (pyIn("gift_card", payment_id)) {
        const payment_method = get(get(get(get(data, "users"), get(order, "user_id")), "payment_methods"), payment_id);
        setItem(payment_method, "balance", pyAdd(get(payment_method, "balance"), get(payment, "amount"), "+="));
        setItem(payment_method, "balance", pyRound(get(payment_method, "balance"), 2));
      }
    }

    setItem(order, "status", "cancelled");
    setItem(order, "cancel_reason", reason);
    listExtend(get(order, "payment_history"), refunds);
    return jsonDumps(order);
  },

  exchange_delivered_order_items: (data, order_id, item_ids, new_item_ids, payment_method_id) => {
    const products = get(data, "products");
    const orders = get(data, "orders");
    const users = get(data, "users");

    if (!pyIn(order_id, orders)) return "Error: order not found";
    const order = get(orders, order_id);
    if (!pyEq(get(order, "status"), "delivered")) return "Error: non-delivered order cannot be exchanged";

    const all_item_ids = pyIter(get(order, "items")).map((item) => get(item, "item_id"));
    for (const item_id of pyIter(item_ids)) {
      if (pyCount(item_ids, item_id) > pyCount(all_item_ids, item_id)) return `Error: ${pyStr(item_id)} not found`;
    }

    if (pyLen(item_ids) !== pyLen(new_item_ids)) return "Error: the number of items to be exchanged should match";

    let diff_price: PyValue = 0;
    for (const [item_id, new_item_id] of pyZip(item_ids, new_item_ids)) {
      const item = firstItemWithId(order, item_id);
      const product_id = get(item, "product_id");
      if (!variantAvailable(products, product_id, new_item_id)) {
        return `Error: new item ${pyStr(new_item_id)} not found or available`;
      }
      const old_price = get(item, "price");
      const new_price = get(get(get(get(products, product_id), "variants"), new_item_id), "price");
      diff_price = pyAdd(diff_price, pySub(new_price, old_price), "+=");
    }

    diff_price = pyRound(diff_price, 2);

    if (!pyIn(payment_method_id, get(get(users, get(order, "user_id")), "payment_methods"))) {
      return "Error: payment method not found";
    }

    const payment_method = get(get(get(users, get(order, "user_id")), "payment_methods"), payment_method_id);
    if (pyEq(get(payment_method, "source"), "gift_card") && pyLt(get(payment_method, "balance"), diff_price)) {
      return "Error: insufficient gift card balance to pay for the price difference";
    }

    setItem(order, "status", "exchange requested");
    setItem(order, "exchange_items", pySorted(item_ids));
    setItem(order, "exchange_new_items", pySorted(new_item_ids));
    setItem(order, "exchange_payment_method_id", payment_method_id);
    setItem(order, "exchange_price_difference", diff_price);
    return jsonDumps(order);
  },

  find_user_id_by_email: (data, email) => {
    const users = get(data, "users") as PyDict;
    for (const [user_id, profile] of users) {
      if (pyLower(get(profile, "email")) === pyLower(email)) return user_id;
    }
    return "Error: user not found";
  },

  find_user_id_by_name_zip: (data, first_name, last_name, zip) => {
    const users = get(data, "users") as PyDict;
    for (const [user_id, profile] of users) {
      if (
        pyLower(get(get(profile, "name"), "first_name")) === pyLower(first_name) &&
        pyLower(get(get(profile, "name"), "last_name")) === pyLower(last_name) &&
        pyEq(get(get(profile, "address"), "zip"), zip)
      ) {
        return user_id;
      }
    }
    return "Error: user not found";
  },

  get_order_details: getDetails("orders", "Error: order not found"),
  get_product_details: getDetails("products", "Error: product not found"),
  get_user_details: getDetails("users", "Error: user not found"),

  list_all_product_types: (data) => {
    const products = get(data, "products") as PyDict;
    const productDict = new PyDict();
    for (const product of products.values()) {
      const name = get(product, "name");
      if (typeof name !== "string") throw new Error("product names are strings");
      productDict.set(name, get(product, "product_id"));
    }
    const sorted = pySorted(Array.from(productDict, ([name, id]) => [name, id] as PyValue[])) as PyValue[][];
    return jsonDumps(new PyDict(sorted.map(([name, id]) => [name as string, id])));
  },

  modify_pending_order_address: (data, order_id, address1, address2, city, state, country, zip) => {
    const orders = get(data, "orders");
    if (!pyIn(order_id, orders)) return "Error: order not found";
    const order = get(orders, order_id);
    if (!pyEq(get(order, "status"), "pending")) return "Error: non-pending order cannot be modified";

    setItem(order, "address", address(address1, address2, city, state, country, zip));
    return jsonDumps(order);
  },

  modify_pending_order_items: (data, order_id, item_ids, new_item_ids, payment_method_id) => {
    const products = get(data, "products");
    const orders = get(data, "orders");
    const users = get(data, "users");

    if (!pyIn(order_id, orders)) return "Error: order not found";
    const order = get(orders, order_id);
    if (!pyEq(get(order, "status"), "pending")) return "Error: non-pending order cannot be modified";

    const all_item_ids = pyIter(get(order, "items")).map((item) => get(item, "item_id"));
    for (const item_id of pyIter(item_ids)) {
      if (pyCount(item_ids, item_id) > pyCount(all_item_ids, item_id)) return `Error: ${pyStr(item_id)} not found`;
    }

    if (pyLen(item_ids) !== pyLen(new_item_ids)) return "Error: the number of items to be exchanged should match";

    let diff_price: PyValue = 0;
    for (const [item_id, new_item_id] of pyZip(item_ids, new_item_ids)) {
      const item = firstItemWithId(order, item_id);
      const product_id = get(item, "product_id");
      if (!variantAvailable(products, product_id, new_item_id)) {
        return `Error: new item ${pyStr(new_item_id)} not found or available`;
      }
      const old_price = get(item, "price");
      const new_price = get(get(get(get(products, product_id), "variants"), new_item_id), "price");
      diff_price = pyAdd(diff_price, pySub(new_price, old_price), "+=");
    }

    if (!pyIn(payment_method_id, get(get(users, get(order, "user_id")), "payment_methods"))) {
      return "Error: payment method not found";
    }

    const payment_method = get(get(get(users, get(order, "user_id")), "payment_methods"), payment_method_id);
    if (pyEq(get(payment_method, "source"), "gift_card") && pyLt(get(payment_method, "balance"), diff_price)) {
      return "Error: insufficient gift card balance to pay for the new item";
    }

    listAppend(
      get(order, "payment_history"),
      dict([
        ["transaction_type", pyGt(diff_price, 0) ? "payment" : "refund"],
        ["amount", pyAbs(diff_price)],
        ["payment_method_id", payment_method_id],
      ]),
    );
    if (pyEq(get(payment_method, "source"), "gift_card")) {
      setItem(payment_method, "balance", pySub(get(payment_method, "balance"), diff_price, "-="));
      setItem(payment_method, "balance", pyRound(get(payment_method, "balance"), 2));
    }

    for (const [item_id, new_item_id] of pyZip(item_ids, new_item_ids)) {
      const item = firstItemWithId(order, item_id);
      setItem(item, "item_id", new_item_id);
      setItem(item, "price", get(get(get(get(products, get(item, "product_id")), "variants"), new_item_id), "price"));
      setItem(
        item,
        "options",
        get(get(get(get(products, get(item, "product_id")), "variants"), new_item_id), "options"),
      );
    }
    setItem(order, "status", "pending (item modified)");
    return jsonDumps(order);
  },

  modify_pending_order_payment: (data, order_id, payment_method_id) => {
    const orders = get(data, "orders");
    if (!pyIn(order_id, orders)) return "Error: order not found";
    const order = get(orders, order_id);
    if (!pyEq(get(order, "status"), "pending")) return "Error: non-pending order cannot be modified";

    const userPaymentMethods = () => get(get(get(data, "users"), get(order, "user_id")), "payment_methods");
    if (!pyIn(payment_method_id, userPaymentMethods())) return "Error: payment method not found";

    const history = () => get(order, "payment_history");
    if (pyLen(history()) > 1 || !pyEq(get(get(history(), 0), "transaction_type"), "payment")) {
      return "Error: there should be exactly one payment for a pending order";
    }

    if (pyEq(get(get(history(), 0), "payment_method_id"), payment_method_id)) {
      return "Error: the new payment method should be different from the current one";
    }

    const amount = get(get(history(), 0), "amount");
    const payment_method = get(userPaymentMethods(), payment_method_id);

    if (pyEq(get(payment_method, "source"), "gift_card") && pyLt(get(payment_method, "balance"), amount)) {
      return "Error: insufficient gift card balance to pay for the order";
    }

    listExtend(history(), [
      dict([
        ["transaction_type", "payment"],
        ["amount", amount],
        ["payment_method_id", payment_method_id],
      ]),
      dict([
        ["transaction_type", "refund"],
        ["amount", amount],
        ["payment_method_id", get(get(history(), 0), "payment_method_id")],
      ]),
    ]);

    if (pyEq(get(payment_method, "source"), "gift_card")) {
      setItem(payment_method, "balance", pySub(get(payment_method, "balance"), amount, "-="));
      setItem(payment_method, "balance", pyRound(get(payment_method, "balance"), 2));
    }

    if (pyIn("gift_card", get(get(history(), 0), "payment_method_id"))) {
      const old_payment_method = get(userPaymentMethods(), get(get(history(), 0), "payment_method_id"));
      setItem(old_payment_method, "balance", pyAdd(get(old_payment_method, "balance"), amount, "+="));
      setItem(old_payment_method, "balance", pyRound(get(old_payment_method, "balance"), 2));
    }

    return jsonDumps(order);
  },

  modify_user_address: (data, user_id, address1, address2, city, state, country, zip) => {
    const users = get(data, "users");
    if (!pyIn(user_id, users)) return "Error: user not found";
    const user = get(users, user_id);
    setItem(user, "address", address(address1, address2, city, state, country, zip));
    return jsonDumps(user);
  },

  return_delivered_order_items: (data, order_id, item_ids, payment_method_id) => {
    const orders = get(data, "orders");
    if (!pyIn(order_id, orders)) return "Error: order not found";
    const order = get(orders, order_id);
    if (!pyEq(get(order, "status"), "delivered")) return "Error: non-delivered order cannot be returned";

    if (!pyIn(payment_method_id, get(get(get(data, "users"), get(order, "user_id")), "payment_methods"))) {
      return "Error: payment method not found";
    }
    if (
      !pyIn("gift_card", payment_method_id) &&
      !pyEq(payment_method_id, get(get(get(order, "payment_history"), 0), "payment_method_id"))
    ) {
      return "Error: payment method should be either the original payment method or a gift card";
    }

    const all_item_ids = pyIter(get(order, "items")).map((item) => get(item, "item_id"));
    for (const item_id of pyIter(item_ids)) {
      if (pyCount(item_ids, item_id) > pyCount(all_item_ids, item_id)) return "Error: some item not found";
    }

    setItem(order, "status", "return requested");
    setItem(order, "return_items", pySorted(item_ids));
    setItem(order, "return_payment_method_id", payment_method_id);
    return jsonDumps(order);
  },

  think,
  transfer_to_human_agents: transferToHumanAgents,
};

// ---------------------------------------------------------------------------
// airline

const AIRPORTS: [string, string][] = [
  ["SFO", "San Francisco"],
  ["JFK", "New York"],
  ["LAX", "Los Angeles"],
  ["ORD", "Chicago"],
  ["DFW", "Dallas"],
  ["DEN", "Denver"],
  ["SEA", "Seattle"],
  ["ATL", "Atlanta"],
  ["MIA", "Miami"],
  ["BOS", "Boston"],
  ["PHX", "Phoenix"],
  ["IAH", "Houston"],
  ["LAS", "Las Vegas"],
  ["MCO", "Orlando"],
  ["EWR", "Newark"],
  ["CLT", "Charlotte"],
  ["MSP", "Minneapolis"],
  ["DTW", "Detroit"],
  ["PHL", "Philadelphia"],
  ["LGA", "LaGuardia"],
];

function withoutDates(flight: PyValue): PyDict {
  return new PyDict(Array.from(flight as PyDict).filter(([k]) => k !== "dates"));
}

function* amounts(payment_methods: PyValue): Generator<PyValue> {
  for (const payment of pyIter(payment_methods)) yield get(payment, "amount");
}

export const AIRLINE_TOOLS: Record<string, ToolImpl> = {
  book_reservation: (
    data,
    user_id,
    origin,
    destination,
    flight_type,
    cabin,
    flights,
    passengers,
    payment_methods,
    total_baggages,
    nonfree_baggages,
    insurance,
  ) => {
    const reservations = get(data, "reservations");
    const users = get(data, "users");
    if (!pyIn(user_id, users)) return "Error: user not found";
    const user = get(users, user_id);

    // assume each task makes at most 3 reservations
    let reservation_id = "HATHAT";
    if (pyIn(reservation_id, reservations)) {
      reservation_id = "HATHAU";
      if (pyIn(reservation_id, reservations)) reservation_id = "HATHAV";
    }

    const reservation = dict([
      ["reservation_id", reservation_id],
      ["user_id", user_id],
      ["origin", origin],
      ["destination", destination],
      ["flight_type", flight_type],
      ["cabin", cabin],
      ["flights", deepCopy(flights)],
      ["passengers", passengers],
      ["payment_history", payment_methods],
      ["created_at", "2024-05-15T15:00:00"],
      ["total_baggages", total_baggages],
      ["nonfree_baggages", nonfree_baggages],
      ["insurance", insurance],
    ]);

    // update flights and calculate price
    let total_price: PyValue = 0;
    for (const flight of pyIter(get(reservation, "flights"))) {
      const flight_number = get(flight, "flight_number");
      if (!pyIn(flight_number, get(data, "flights"))) return `Error: flight ${pyStr(flight_number)} not found`;
      const flight_data = get(get(data, "flights"), flight_number);
      if (!pyIn(get(flight, "date"), get(flight_data, "dates"))) {
        return `Error: flight ${pyStr(flight_number)} not found on date ${pyStr(get(flight, "date"))}`;
      }
      const flight_date_data = get(get(flight_data, "dates"), get(flight, "date"));
      if (!pyEq(get(flight_date_data, "status"), "available")) {
        return `Error: flight ${pyStr(flight_number)} not available on date ${pyStr(get(flight, "date"))}`;
      }
      if (pyLt(get(get(flight_date_data, "available_seats"), cabin), pyLen(passengers))) {
        return `Error: not enough seats on flight ${pyStr(flight_number)}`;
      }
      setItem(flight, "price", get(get(flight_date_data, "prices"), cabin));
      setItem(flight, "origin", get(flight_data, "origin"));
      setItem(flight, "destination", get(flight_data, "destination"));
      total_price = pyAdd(total_price, pyMul(get(flight, "price"), pyLen(passengers)), "+=");
    }

    if (pyEq(insurance, "yes")) total_price = pyAdd(total_price, pyMul(30, pyLen(passengers)), "+=");

    total_price = pyAdd(total_price, pyMul(50, nonfree_baggages), "+=");

    for (const payment_method of pyIter(payment_methods)) {
      const payment_id = get(payment_method, "payment_id");
      const amount = get(payment_method, "amount");
      if (!pyIn(payment_id, get(user, "payment_methods"))) return `Error: payment method ${pyStr(payment_id)} not found`;
      if (pyIn(get(get(get(user, "payment_methods"), payment_id), "source"), ["gift_card", "certificate"])) {
        if (pyLt(get(get(get(user, "payment_methods"), payment_id), "amount"), amount)) {
          return `Error: not enough balance in payment method ${pyStr(payment_id)}`;
        }
      }
    }
    if (!pyEq(pySum(amounts(payment_methods)), total_price)) {
      return `Error: payment amount does not add up, total price is ${pyStr(total_price)}, but paid ${pyStr(pySum(amounts(payment_methods)))}`;
    }

    // if checks pass, deduct payment and update seats
    for (const payment_method of pyIter(payment_methods)) {
      const payment_id = get(payment_method, "payment_id");
      const amount = get(payment_method, "amount");
      const method = get(get(user, "payment_methods"), payment_id);
      if (pyEq(get(method, "source"), "gift_card")) {
        setItem(method, "amount", pySub(get(method, "amount"), amount, "-="));
      } else if (pyEq(get(get(get(user, "payment_methods"), payment_id), "source"), "certificate")) {
        delItem(get(user, "payment_methods"), payment_id);
      }
    }

    setItem(reservations, reservation_id, reservation);
    listAppend(get(user, "reservations"), reservation_id);
    return jsonDumps(reservation);
  },

  calculate: (_data, expression) => calculate(expression),

  cancel_reservation: (data, reservation_id) => {
    const reservations = get(data, "reservations");
    if (!pyIn(reservation_id, reservations)) return "Error: reservation not found";
    const reservation = get(reservations, reservation_id);

    // reverse the payment
    const refunds: PyValue[] = [];
    for (const payment of pyIter(get(reservation, "payment_history"))) {
      refunds.push(
        dict([
          ["payment_id", get(payment, "payment_id")],
          ["amount", pyNeg(get(payment, "amount"))],
        ]),
      );
    }
    listExtend(get(reservation, "payment_history"), refunds);
    setItem(reservation, "status", "cancelled");
    return jsonDumps(reservation);
  },

  get_reservation_details: getDetails("reservations", "Error: user not found"),
  get_user_details: getDetails("users", "Error: user not found"),

  list_all_airports: () => jsonDumps(new PyDict(AIRPORTS)),

  search_direct_flight: (data, origin, destination, date) => {
    const flights = get(data, "flights") as PyDict;
    const results: PyValue[] = [];
    for (const flight of flights.values()) {
      if (pyEq(get(flight, "origin"), origin) && pyEq(get(flight, "destination"), destination)) {
        if (
          pyIn(date, get(flight, "dates")) &&
          pyEq(get(get(get(flight, "dates"), date), "status"), "available")
        ) {
          // results add flight except dates, but add flight["datas"][date]
          results.push(withoutDates(flight));
          dictUpdate(results[results.length - 1], get(get(flight, "dates"), date));
        }
      }
    }
    return jsonDumps(results);
  },

  search_onestop_flight: (data, origin, destination, date) => {
    const flights = get(data, "flights") as PyDict;
    const results: PyValue[] = [];
    for (const flight1 of flights.values()) {
      if (!pyEq(get(flight1, "origin"), origin)) continue;
      for (const flight2 of flights.values()) {
        if (
          pyEq(get(flight2, "destination"), destination) &&
          pyEq(get(flight1, "destination"), get(flight2, "origin"))
        ) {
          const date2 = pyIn("+1", get(flight1, "scheduled_arrival_time_est"))
            ? `2024-05-${pyInt(lastTwo(date)) + 1}`
            : date;
          if (pyGt(get(flight1, "scheduled_arrival_time_est"), get(flight2, "scheduled_departure_time_est"))) {
            continue;
          }
          if (pyIn(date, get(flight1, "dates")) && pyIn(date2, get(flight2, "dates"))) {
            if (
              pyEq(get(get(get(flight1, "dates"), date), "status"), "available") &&
              pyEq(get(get(get(flight2, "dates"), date2), "status"), "available")
            ) {
              const result1 = withoutDates(flight1);
              dictUpdate(result1, get(get(flight1, "dates"), date));
              setItem(result1, "date", date);
              const result2 = withoutDates(flight2);
              dictUpdate(result2, get(get(flight2, "dates"), date));
              setItem(result2, "date", date2);
              results.push([result1, result2]);
            }
          }
        }
      }
    }
    return jsonDumps(results);
  },

  send_certificate: (data, user_id, amount) => {
    const users = get(data, "users");
    if (!pyIn(user_id, users)) return "Error: user not found";
    const user = get(users, user_id);

    // add a certificate, assume at most 3 cases per task
    for (const id of [3221322, 3221323, 3221324]) {
      const payment_id = `certificate_${id}`;
      if (!pyIn(payment_id, get(user, "payment_methods"))) {
        setItem(
          get(user, "payment_methods"),
          payment_id,
          dict([
            ["source", "certificate"],
            ["amount", amount],
            ["id", payment_id],
          ]),
        );
        return `Certificate ${payment_id} added to user ${pyStr(user_id)} with amount ${pyStr(amount)}.`;
      }
    }
    return null;
  },

  think,
  transfer_to_human_agents: transferToHumanAgents,

  update_reservation_baggages: (data, reservation_id, total_baggages, nonfree_baggages, payment_id) => {
    const users = get(data, "users");
    const reservations = get(data, "reservations");
    if (!pyIn(reservation_id, reservations)) return "Error: reservation not found";
    const reservation = get(reservations, reservation_id);

    const total_price = pyMul(50, pyMax(0, pySub(nonfree_baggages, get(reservation, "nonfree_baggages"))));
    if (!pyIn(payment_id, get(get(users, get(reservation, "user_id")), "payment_methods"))) {
      return "Error: payment method not found";
    }
    const payment_method = get(get(get(users, get(reservation, "user_id")), "payment_methods"), payment_id);
    if (pyEq(get(payment_method, "source"), "certificate")) {
      return "Error: certificate cannot be used to update reservation";
    } else if (pyEq(get(payment_method, "source"), "gift_card") && pyLt(get(payment_method, "amount"), total_price)) {
      return "Error: gift card balance is not enough";
    }

    setItem(reservation, "total_baggages", total_baggages);
    setItem(reservation, "nonfree_baggages", nonfree_baggages);
    if (pyEq(get(payment_method, "source"), "gift_card")) {
      setItem(payment_method, "amount", pySub(get(payment_method, "amount"), total_price, "-="));
    }

    if (!pyEq(total_price, 0)) {
      listAppend(
        get(reservation, "payment_history"),
        dict([
          ["payment_id", payment_id],
          ["amount", total_price],
        ]),
      );
    }

    return jsonDumps(reservation);
  },

  update_reservation_flights: (data, reservation_id, cabin, flights, payment_id) => {
    const users = get(data, "users");
    const reservations = get(data, "reservations");
    if (!pyIn(reservation_id, reservations)) return "Error: reservation not found";
    const reservation = get(reservations, reservation_id);

    // update flights and calculate price
    let total_price: PyValue = 0;
    const newFlights = deepCopy(flights);
    for (const flight of pyIter(newFlights)) {
      // if existing flight, ignore
      const existing = pyIter(get(reservation, "flights")).filter(
        (f) =>
          pyEq(get(f, "flight_number"), get(flight, "flight_number")) &&
          pyEq(get(f, "date"), get(flight, "date")) &&
          pyEq(cabin, get(reservation, "cabin")),
      );
      if (existing.length > 0) {
        total_price = pyAdd(
          total_price,
          pyMul(get(existing[0], "price"), pyLen(get(reservation, "passengers"))),
          "+=",
        );
        setItem(flight, "price", get(existing[0], "price"));
        setItem(flight, "origin", get(existing[0], "origin"));
        setItem(flight, "destination", get(existing[0], "destination"));
        continue;
      }
      const flight_number = get(flight, "flight_number");
      if (!pyIn(flight_number, get(data, "flights"))) return `Error: flight ${pyStr(flight_number)} not found`;
      const flight_data = get(get(data, "flights"), flight_number);
      if (!pyIn(get(flight, "date"), get(flight_data, "dates"))) {
        return `Error: flight ${pyStr(flight_number)} not found on date ${pyStr(get(flight, "date"))}`;
      }
      const flight_date_data = get(get(flight_data, "dates"), get(flight, "date"));
      if (!pyEq(get(flight_date_data, "status"), "available")) {
        return `Error: flight ${pyStr(flight_number)} not available on date ${pyStr(get(flight, "date"))}`;
      }
      if (pyLt(get(get(flight_date_data, "available_seats"), cabin), pyLen(get(reservation, "passengers")))) {
        return `Error: not enough seats on flight ${pyStr(flight_number)}`;
      }
      setItem(flight, "price", get(get(flight_date_data, "prices"), cabin));
      setItem(flight, "origin", get(flight_data, "origin"));
      setItem(flight, "destination", get(flight_data, "destination"));
      total_price = pyAdd(total_price, pyMul(get(flight, "price"), pyLen(get(reservation, "passengers"))), "+=");
    }

    const oldPrices = function* () {
      for (const f of pyIter(get(reservation, "flights"))) yield get(f, "price");
    };
    total_price = pySub(total_price, pyMul(pySum(oldPrices()), pyLen(get(reservation, "passengers"))), "-=");

    // check payment
    if (!pyIn(payment_id, get(get(users, get(reservation, "user_id")), "payment_methods"))) {
      return "Error: payment method not found";
    }
    const payment_method = get(get(get(users, get(reservation, "user_id")), "payment_methods"), payment_id);
    if (pyEq(get(payment_method, "source"), "certificate")) {
      return "Error: certificate cannot be used to update reservation";
    } else if (pyEq(get(payment_method, "source"), "gift_card") && pyLt(get(payment_method, "amount"), total_price)) {
      return "Error: gift card balance is not enough";
    }

    // if checks pass, deduct payment and update seats
    if (pyEq(get(payment_method, "source"), "gift_card")) {
      setItem(payment_method, "amount", pySub(get(payment_method, "amount"), total_price, "-="));
    }
    setItem(reservation, "flights", newFlights);
    if (!pyEq(total_price, 0)) {
      listAppend(
        get(reservation, "payment_history"),
        dict([
          ["payment_id", payment_id],
          ["amount", total_price],
        ]),
      );
    }
    // do not make flight database update here, assume it takes time to be updated
    return jsonDumps(reservation);
  },

  update_reservation_passengers: (data, reservation_id, passengers) => {
    const reservations = get(data, "reservations");
    if (!pyIn(reservation_id, reservations)) return "Error: reservation not found";
    const reservation = get(reservations, reservation_id);
    if (pyLen(passengers) !== pyLen(get(reservation, "passengers"))) {
      return "Error: number of passengers does not match";
    }
    setItem(reservation, "passengers", passengers);
    return jsonDumps(reservation);
  },
};
