import { type PyValue, jsonDumps, jsonPretty, parseJson } from "../tau/py";

/** Pretty-print a tool observation if it is JSON (keeping Python's number formatting). */
export function prettyObservation(obs: string): { text: string; isJson: boolean } {
  const t = obs.trim();
  if (t.startsWith("{") || t.startsWith("[")) {
    try {
      return { text: jsonPretty(parseJson(t)), isJson: true };
    } catch {
      // not JSON after all
    }
  }
  return { text: obs, isJson: false };
}

export function prettyArgs(args: string): string {
  try {
    return jsonPretty(parseJson(args));
  } catch {
    return args;
  }
}

export function oneLine(text: string, max = 160): string {
  const s = text.replace(/\s+/g, " ").trim();
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

export function valueText(v: PyValue | undefined, absent: string): string {
  return v === undefined ? absent : jsonDumps(v);
}

const READ_PREFIXES = ["get_", "find_", "list_", "search_"];

export function isReadTool(name: string): boolean {
  return READ_PREFIXES.some((p) => name.startsWith(p)) || name === "calculate" || name === "think";
}

export function formatBytes(n: number): string {
  return n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`;
}

export function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
