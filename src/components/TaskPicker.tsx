import { useEffect, useMemo, useRef, useState } from "react";
import { useT } from "../i18n";
import type { HistoryEntry } from "../state/history";
import type { DomainName, Task } from "../tau/env";
import { isSuccess } from "../tau/metrics";

type Filter = "all" | "untried" | "failed" | "passed";

interface Row {
  index: number;
  task: Task;
  tries: number;
  passes: number;
}

/** A searchable list of the split's tasks, with how earlier attempts went (from History). */
export function TaskPicker({
  domain,
  split,
  tasks,
  selected,
  history,
  hideInstructions,
  onPick,
  onClose,
}: {
  domain: DomainName;
  split: string;
  tasks: Task[];
  selected: number;
  history: HistoryEntry[];
  /** A human agent should not read the customers' goals. */
  hideInstructions: boolean;
  onPick: (index: number) => void;
  onClose: () => void;
}) {
  const t = useT();
  const dialog = useRef<HTMLDialogElement>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [showText, setShowText] = useState(!hideInstructions);

  // Open once, on mount. (Picking changes `selected` right before the dialog closes; that must not reopen it.)
  const initiallySelected = useRef(selected);
  useEffect(() => {
    const d = dialog.current;
    if (!d || d.open) return;
    d.showModal();
    d.querySelector<HTMLElement>(`[data-index="${initiallySelected.current}"]`)?.scrollIntoView({ block: "center" });
    d.querySelector<HTMLInputElement>("input[type=search]")?.focus();
  }, []);

  const rows = useMemo<Row[]>(() => {
    const stats = new Map<number, { tries: number; passes: number }>();
    for (const e of history) {
      if (e.domain !== domain || e.split !== split) continue;
      const s = stats.get(e.index) ?? { tries: 0, passes: 0 };
      s.tries += 1;
      if (isSuccess(e.reward)) s.passes += 1;
      stats.set(e.index, s);
    }
    return tasks.map((task, index) => ({ index, task, tries: 0, passes: 0, ...stats.get(index) }));
  }, [tasks, history, domain, split]);

  const q = query.trim().toLowerCase().replace(/^#/, "");
  const shown = rows.filter((r) => {
    if (filter === "untried" && r.tries > 0) return false;
    if (filter === "failed" && !(r.tries > 0 && r.passes < r.tries)) return false;
    if (filter === "passed" && r.passes === 0) return false;
    if (!q) return true;
    if (/^\d+$/.test(q)) return String(r.index).startsWith(q);
    return r.task.user_id.toLowerCase().includes(q) || (showText && r.task.instruction.toLowerCase().includes(q));
  });

  const filters: [Filter, string][] = [
    ["all", t("pickerAll")],
    ["untried", t("pickerUntried")],
    ["failed", t("pickerFailed")],
    ["passed", t("pickerPassed")],
  ];

  return (
    <dialog ref={dialog} className="task-picker" aria-label={t("pickerTitle")} onClose={onClose}>
      <div className="picker-head">
        <strong>
          {t("pickerTitle")} <span className="muted">{`${domain} · ${split}`}</span>
        </strong>
        <span className="spacer" />
        <button type="button" className="btn btn-small btn-ghost" onClick={() => dialog.current?.close()}>
          {t("batchClose")}
        </button>
      </div>
      <div className="picker-controls">
        <input
          type="search"
          placeholder={showText ? t("pickerSearch") : t("pickerSearchIds")}
          value={query}
          spellCheck={false}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="chips">
          {filters.map(([id, label]) => (
            <button key={id} type="button" className={`chip ${filter === id ? "active" : ""}`} onClick={() => setFilter(id)}>
              {label}
            </button>
          ))}
        </div>
        <label className="check">
          <input type="checkbox" checked={showText} onChange={(e) => setShowText(e.target.checked)} />
          <span>{t("pickerShowInstructions")}</span>
        </label>
      </div>
      <p className="muted small">{t("pickerCount", { n: shown.length, total: rows.length })}</p>
      <ul className="picker-list">
        {shown.map((r) => (
          <li key={r.index}>
            <button
              type="button"
              data-index={r.index}
              className={r.index === selected ? "active" : ""}
              onClick={() => {
                onPick(r.index);
                dialog.current?.close();
              }}
            >
              <span className="picker-index">#{r.index}</span>
              <span className="picker-main">
                <code>{r.task.user_id}</code>
                {showText && <span className="picker-text">{r.task.instruction}</span>}
              </span>
              {r.tries > 0 && (
                <span className={`badge ${r.passes > 0 ? "badge-ok" : "badge-stop"}`} title={t("pickerTries")}>
                  {r.passes}/{r.tries}
                </span>
              )}
            </button>
          </li>
        ))}
      </ul>
    </dialog>
  );
}
