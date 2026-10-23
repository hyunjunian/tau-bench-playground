import { Fragment, useMemo, useRef, useState } from "react";
import { useT } from "../i18n";
import type { HistoryEntry } from "../state/history";
import { compareDatabases, compareWrites } from "../tau/diff";
import type { Domain, Task } from "../tau/env";
import type { Episode, EpisodeState } from "../tau/episode";
import { averageReward, isSuccess, passHatK } from "../tau/metrics";
import { userSystemPrompt } from "../tau/prompts";
import { PyDict, type PyValue, jsonPretty } from "../tau/py";
import { type EpisodeRecord, SAMPLE_RUNS } from "../tau/records";
import { isReadTool, valueText } from "../ui/format";
import { endReasonText } from "../ui/reasons";
import { JsonBlock } from "./JsonBlock";
import { Markdown } from "./Markdown";

export type InspectorTab = "task" | "policy" | "tools" | "result" | "data" | "history";

/** A saved-episodes file the user opened (shown in the History tab). */
export interface OpenedFile {
  name: string;
  records: EpisodeRecord[];
  /** Rewards recomputed by replaying the records, by record position. */
  recomputed?: Map<number, Recomputed>;
  recomputing?: boolean;
}

function Spoiler({ hint, children, initiallyOpen }: { hint: string; children: React.ReactNode; initiallyOpen: boolean }) {
  const t = useT();
  const [open, setOpen] = useState(initiallyOpen);
  const [prev, setPrev] = useState(initiallyOpen);
  if (prev !== initiallyOpen) {
    setPrev(initiallyOpen);
    setOpen(initiallyOpen);
  }
  if (open) {
    return (
      <div className="spoiler open">
        {children}
        {!initiallyOpen && (
          <button type="button" className="btn btn-small btn-ghost" onClick={() => setOpen(false)}>
            {t("conceal")}
          </button>
        )}
      </div>
    );
  }
  return (
    <div className="spoiler">
      <span className="muted">{hint}</span>
      <button type="button" className="btn btn-small" onClick={() => setOpen(true)}>
        {t("reveal")}
      </button>
    </div>
  );
}

function ActionList({ actions }: { actions: { name: string; kwargs: PyDict }[] }) {
  const t = useT();
  if (actions.length === 0) return <p className="muted">{t("noToolCalls")}</p>;
  return (
    <ol className="action-list">
      {actions.map((a, i) => (
        <li key={i} className={isReadTool(a.name) ? "read" : "write"}>
          <code className="tool-name">{a.name}</code>
          <span className={`badge ${isReadTool(a.name) ? "" : "badge-write"}`}>{isReadTool(a.name) ? t("readTool") : t("writeTool")}</span>
          <JsonBlock text={jsonPretty(a.kwargs)} className="compact" />
        </li>
      ))}
    </ol>
  );
}

function TaskTab({ task, agentIsHuman, done }: { task: Task; agentIsHuman: boolean; done: boolean }) {
  const t = useT();
  // A human agent should only learn the customer's goal from the conversation.
  const hideInstruction = agentIsHuman && !done;
  return (
    <div className="tab-body">
      <div className="kv">
        <span>{t("userId")}</span>
        <code>{task.user_id}</code>
      </div>
      <h3>{t("instruction")}</h3>
      {hideInstruction ? (
        <Spoiler hint={t("instructionHidden")} initiallyOpen={false}>
          <p className="instruction">{task.instruction}</p>
        </Spoiler>
      ) : (
        <>
          <p className="instruction">{task.instruction}</p>
          <details className="user-prompt">
            <summary>{t("userPromptTitle")}</summary>
            <pre>{userSystemPrompt(task.instruction)}</pre>
          </details>
        </>
      )}
      <h3>{t("requiredOutputs")}</h3>
      {task.outputs.length === 0 ? (
        <p className="muted">{t("noOutputs")}</p>
      ) : (
        <Spoiler hint={t("gtHidden")} initiallyOpen={done}>
          <ul>
            {task.outputs.map((o) => (
              <li key={o}>
                <code>{o}</code>
              </li>
            ))}
          </ul>
        </Spoiler>
      )}
      <h3>{t("gtActions")}</h3>
      <Spoiler hint={t("gtHidden")} initiallyOpen={done}>
        <ActionList actions={task.actions} />
      </Spoiler>
    </div>
  );
}

function ToolsTab({ domain, onUse, canUse }: { domain: Domain; onUse: (name: string) => void; canUse: boolean }) {
  const t = useT();
  return (
    <div className="tab-body">
      <p className="note">{t("toolsNote")}</p>
      {domain.toolsInfo.map((info) => {
        const props = Object.entries(info.function.parameters.properties ?? {});
        const read = isReadTool(info.function.name);
        return (
          <div key={info.function.name} className="tool-doc">
            <div className="row space">
              <span>
                <code className="tool-name">{info.function.name}</code>{" "}
                <span className={`badge ${read ? "" : "badge-write"}`}>{read ? t("readTool") : t("writeTool")}</span>
              </span>
              {canUse && (
                <button type="button" className="btn btn-small" onClick={() => onUse(info.function.name)}>
                  {t("useTool")}
                </button>
              )}
            </div>
            <p>{info.function.description}</p>
            {props.length > 0 && (
              <ul className="params">
                {props.map(([name, s]) => (
                  <li key={name}>
                    <code>{name}</code> <span className="muted">{s.type}</span>
                    {s.enum && <span className="muted"> ∈ {s.enum.join(" | ")}</span>}
                    {s.description && <div className="muted">{s.description}</div>}
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ResultTab({ episode, state }: { episode: Episode | null; state: EpisodeState | null }) {
  const t = useT();
  const done = state?.done;
  const comparisons = useMemo(() => {
    if (!episode || !done?.info) return null;
    return compareDatabases(episode.setup.domain.pristine, episode.env.data, done.info.gtData);
  }, [episode, done]);
  if (!episode || !done) return <div className="tab-body muted">{t("resultPending")}</div>;
  const info = done.info;
  const agentTools = episode.env.actions.filter((a) => a.name !== "respond");
  const writes = compareWrites(episode.setup.task.actions, episode.env.actions, (name) => !isReadTool(name));
  return (
    <div className="tab-body">
      <div className={`result-summary ${done.reward === 1 ? "ok" : "bad"}`}>
        <span className="reward-big">{done.reward.toFixed(1)}</span>
        {info && (
          <div>
            <div className={info.rActions ? "check-ok" : "check-bad"}>
              {info.rActions ? "✓" : "✗"} {t("rActions")}
            </div>
            {info.rOutputs !== null && (
              <div className={info.rOutputs === 1 ? "check-ok" : "check-bad"}>
                {info.rOutputs === 1 ? "✓" : "✗"} {t("rOutputs")}
              </div>
            )}
          </div>
        )}
      </div>

      {info && info.outputs.length > 0 && (
        <>
          <h3>{t("requiredOutputs")}</h3>
          <ul className="outputs">
            {info.outputs.map((o) => (
              <li key={o.output} className={o.found ? "check-ok" : "check-bad"}>
                <code>{o.output}</code> — {o.found ? t("outputFound") : t("outputMissing")}
              </li>
            ))}
          </ul>
        </>
      )}

      {comparisons && (
        <>
          <h3>{t("dbChanges")}</h3>
          {comparisons.length === 0 && <p className="muted">{t("dbNoChanges")}</p>}
          {comparisons.map((c) => (
            <details key={`${c.collection}/${c.key}`} className={`record ${c.ok ? "ok" : "bad"}`} open={!c.ok}>
              <summary>
                <span className={c.ok ? "check-ok" : "check-bad"}>{c.ok ? "✓" : "✗"}</span>{" "}
                <code>
                  {c.collection}/{c.key}
                </code>{" "}
                <span className="muted">{c.ok ? t("recordOk") : t("recordBad")}</span>
              </summary>
              {c.ok ? (
                c.actual !== undefined && <JsonBlock text={jsonPretty(c.actual)} className="compact" />
              ) : (
                <table className="diff">
                  <thead>
                    <tr>
                      <th>{t("pathCol")}</th>
                      <th>{t("expected")}</th>
                      <th>{t("actual")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {c.diffs.map((d) => (
                      <tr key={d.path}>
                        <td>
                          <code>{d.path}</code>
                        </td>
                        <td>
                          <code>{valueText(d.expected, t("absent"))}</code>
                        </td>
                        <td>
                          <code>{valueText(d.actual, t("absent"))}</code>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </details>
          ))}
        </>
      )}

      <h3>{t("gtActions")}</h3>
      <ol className="action-list">
        {writes.checks.map(({ action: a, matched, closest }, i) => (
          <li key={i} className="write">
            <span className={matched ? "check-ok" : "check-bad"} title={matched ? t("matchedGt") : undefined}>
              {matched ? "✓" : "✗"}
            </span>{" "}
            <code className="tool-name">{a.name}</code>
            {!matched && <span className="muted small"> {closest ? t("gtClosest") : t("gtNeverCalled")}</span>}
            <JsonBlock text={jsonPretty(a.kwargs)} className="compact" />
            {closest && (
              <table className="diff">
                <thead>
                  <tr>
                    <th>{t("argCol")}</th>
                    <th>{t("expected")}</th>
                    <th>{t("actual")}</th>
                  </tr>
                </thead>
                <tbody>
                  {closest.diffs.map((d) => (
                    <tr key={d.path}>
                      <td>
                        <code>{d.path}</code>
                      </td>
                      <td>
                        <code>{valueText(d.expected, t("absent"))}</code>
                      </td>
                      <td>
                        <code>{valueText(d.actual, t("absent"))}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </li>
        ))}
      </ol>
      {writes.extra.length > 0 && (
        <>
          <h3>{t("extraWrites")}</h3>
          <ActionList actions={writes.extra} />
        </>
      )}
      <h3>{t("agentActionsTitle")}</h3>
      <ActionList actions={agentTools} />
    </div>
  );
}

const LOOKUPS: [string, (id: string) => string[]][] = [
  ["users", (id) => [id]],
  ["orders", (id) => [id, id.startsWith("#") ? id : `#${id}`]],
  ["products", (id) => [id]],
  ["reservations", (id) => [id, id.toUpperCase()]],
  ["flights", (id) => [id, id.toUpperCase()]],
];

function lookup(db: PyDict, raw: string): { path: string; value: PyValue }[] {
  const id = raw.trim();
  if (!id) return [];
  const out: { path: string; value: PyValue }[] = [];
  for (const [collection, candidates] of LOOKUPS) {
    const table = db.get(collection);
    if (!(table instanceof PyDict)) continue;
    for (const key of new Set(candidates(id))) {
      const v = table.get(key);
      if (v !== undefined) out.push({ path: `${collection}/${key}`, value: v });
    }
  }
  // item ids live inside product variants
  const products = db.get("products");
  if (products instanceof PyDict && /^\d+$/.test(id)) {
    for (const [pid, p] of products) {
      const variants = (p as PyDict).get("variants");
      if (variants instanceof PyDict && variants.has(id)) {
        out.push({ path: `products/${pid}/variants/${id}`, value: variants.get(id)! });
      }
    }
  }
  return out;
}

function DataTab({ domain, episode }: { domain: Domain; episode: Episode | null }) {
  const t = useT();
  const [query, setQuery] = useState("");
  const db = episode?.setup.domain === domain ? episode.env.data : domain.pristine;
  const results = lookup(db, query);
  return (
    <div className="tab-body">
      <p className="note">{t("dataNote")}</p>
      <input className="mono" placeholder={t("dataSearch")} value={query} spellCheck={false} onChange={(e) => setQuery(e.target.value)} />
      <small className="muted">{db === domain.pristine ? t("dataInitial") : t("dataLive")}</small>
      {query.trim() && results.length === 0 && <p className="muted">{t("dataNotFound")}</p>}
      {results.map((r) => (
        <div key={r.path}>
          <h3>
            <code>{r.path}</code>
          </h3>
          <JsonBlock text={jsonPretty(r.value)} className="compact" />
        </div>
      ))}
    </div>
  );
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleString(undefined, { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function OpenFileButton({ onOpen }: { onOpen: (file: File) => void }) {
  const t = useT();
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <button type="button" className="btn btn-small" onClick={() => input.current?.click()}>
        {t("openFile")}
      </button>
      <input
        ref={input}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) onOpen(file);
        }}
      />
    </>
  );
}

/** Result of replaying one record of an opened file: its recomputed reward, or why it could not be replayed. */
export type Recomputed = { reward: number; reason: string } | { error: string };

function OpenedFileList({
  file,
  onOpen,
  onClose,
  onRecompute,
}: {
  file: OpenedFile;
  onOpen: (r: EpisodeRecord) => void;
  onClose: () => void;
  onRecompute: () => void;
}) {
  const t = useT();
  const [failedOnly, setFailedOnly] = useState(false);
  const results = file.recomputed ?? null;
  const running = !!file.recomputing;
  const recorded = file.records.filter((r) => r.reward !== null).map((r) => ({ task: String(r.taskId), reward: r.reward! }));
  const avg = averageReward(recorded);
  const passK = passHatK(recorded);

  let same = 0;
  let differ = 0;
  let failed = 0;
  let unrecorded = 0;
  for (const r of file.records) {
    const c = results?.get(r.position);
    if (!c) continue;
    if ("error" in c) failed++;
    else if (r.reward === null) unrecorded++;
    else if (Math.abs(c.reward - r.reward) > 1e-6) differ++;
    else same++;
  }

  return (
    <section className="opened-file">
      <div className="row space">
        <h3>
          {t("openedFile")}: <code>{file.name}</code>
        </h3>
        <button type="button" className="btn btn-small btn-ghost" onClick={onClose}>
          {t("batchClose")}
        </button>
      </div>
      <p className="muted small">
        {t("openedCount", { n: file.records.length })}
        {avg !== null && ` · ${t("batchAvg", { avg: avg.toFixed(3) })}`}
        {passK.length > 1 && passK.map(({ k, value }) => ` · pass^${k} ${value.toFixed(3)}`).join("")}
      </p>
      <div className="row space verify-row">
        <span className="small">
          {running
            ? t("verifyRunning", { done: results?.size ?? 0, total: file.records.length })
            : results
              ? t("verifySummary", { same, differ, failed }) + (unrecorded ? ` ${t("verifyUnrecorded", { n: unrecorded })}` : "")
              : t("verifyHint")}
        </span>
        <button type="button" className="btn btn-small" disabled={running} onClick={onRecompute}>
          {t("verifyRun")}
        </button>
      </div>
      <label className="check small">
        <input type="checkbox" checked={failedOnly} onChange={(e) => setFailedOnly(e.target.checked)} />
        <span>{t("openedFailedOnly")}</span>
      </label>
      <table className="history">
        <thead>
          <tr>
            <th>#</th>
            <th>{t("colTask")}</th>
            <th>{t("colTrial")}</th>
            <th>{t("colReward")}</th>
            {results && <th>{t("colRecomputed")}</th>}
          </tr>
        </thead>
        <tbody>
          {file.records.map((r) => {
            if (failedOnly && r.reward !== null && isSuccess(r.reward)) return null;
            const c = results?.get(r.position);
            const differs = c && !("error" in c) && r.reward !== null && Math.abs(c.reward - r.reward) > 1e-6;
            return (
              <tr key={r.position} className={differs ? "differs" : undefined}>
                <td>{r.position + 1}</td>
                <td>
                  <button type="button" className="link-btn" title={t("historyOpen")} onClick={() => onOpen(r)}>
                    {r.domain ?? "?"}
                    {r.split ? `/${r.split}` : ""}#{r.taskId ?? "?"}
                  </button>
                </td>
                <td>{r.trial !== null ? r.trial + 1 : ""}</td>
                <td className={r.reward === null ? "" : isSuccess(r.reward) ? "check-ok" : "check-bad"}>
                  {r.reward?.toFixed(1) ?? "–"}
                </td>
                {results && (
                  <td
                    className={!c || "error" in c ? "muted" : isSuccess(c.reward) ? "check-ok" : "check-bad"}
                    title={c ? ("error" in c ? c.error : endReasonText(t, c.reason)) : undefined}
                  >
                    {!c ? "" : "error" in c ? "!" : `${differs ? "≠ " : ""}${c.reward.toFixed(1)}`}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function ModelSummary({ entries }: { entries: HistoryEntry[] }) {
  const t = useT();
  const groups = new Map<string, HistoryEntry[]>();
  for (const e of entries) {
    const key = `${e.agent}\u0000${e.user}`;
    groups.set(key, [...(groups.get(key) ?? []), e]);
  }
  if (groups.size < 2) return null;
  return (
    <>
      <h3>{t("historySummary")}</h3>
      <table className="history">
        <thead>
          <tr>
            <th>{t("colAgent")}</th>
            <th>{t("colUser")}</th>
            <th>{t("colEpisodes")}</th>
            <th>{t("colTasks")}</th>
            <th>{t("colAvg")}</th>
          </tr>
        </thead>
        <tbody>
          {Array.from(groups.values(), (g) => {
            const avg = averageReward(g.map((e) => ({ task: "", reward: e.reward }))) ?? 0;
            const tasks = new Set(g.map((e) => `${e.domain}/${e.split}/${e.index}`)).size;
            return (
              <tr key={`${g[0].agent}/${g[0].user}`}>
                <td>{g[0].agent}</td>
                <td>{g[0].user}</td>
                <td>{g.length}</td>
                <td>{tasks}</td>
                <td>{avg.toFixed(3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

/** Links to the runs the tau-bench authors published; a link also works as ?file= in a new tab. */
export function SampleRunLinks({ onOpenUrl, downloading }: { onOpenUrl: (url: string) => void; downloading: boolean }) {
  const t = useT();
  return (
    <>
      {SAMPLE_RUNS.map((s, i) => (
        <Fragment key={s.file}>
          {i > 0 && " · "}
          <a
            href={`?file=${encodeURIComponent(s.url)}`}
            title={`${s.file} (${s.mb} MB)`}
            onClick={(e) => {
              if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return; // new tab or window
              e.preventDefault();
              onOpenUrl(s.url);
            }}
          >
            {s.label}
          </a>
        </Fragment>
      ))}
      {downloading && (
        <span className="welcome-downloading" role="status">
          <span className="spinner" aria-hidden /> {t("downloading")}
        </span>
      )}
    </>
  );
}

function HistoryTab({
  entries,
  onClear,
  onDownloadAll,
  onOpenEntry,
  opened,
  onOpenFile,
  onOpenUrl,
  downloading,
  onOpenRecord,
  onCloseFile,
  onRecompute,
}: {
  entries: HistoryEntry[];
  onClear: () => void;
  onDownloadAll: () => void;
  onOpenEntry: (e: HistoryEntry) => void;
  opened: OpenedFile | null;
  onOpenFile: (file: File) => void;
  onOpenUrl: (url: string) => void;
  /** A file is being downloaded from a URL. */
  downloading: boolean;
  onOpenRecord: (r: EpisodeRecord) => void;
  onCloseFile: () => void;
  onRecompute: () => void;
}) {
  const t = useT();
  const avg = averageReward(entries.map((e) => ({ task: "", reward: e.reward })));
  const askUrl = () => {
    const url = window.prompt(t("openUrlPrompt"))?.trim();
    if (url) onOpenUrl(url);
  };
  return (
    <div className="tab-body">
      <div className="row space">
        <span className="muted small">{t("openFileHint")}</span>
        <span className="row">
          <OpenFileButton onOpen={onOpenFile} />
          <button type="button" className="btn btn-small" onClick={askUrl}>
            {t("openUrl")}
          </button>
        </span>
      </div>
      <p className="welcome-samples muted small">
        {t("samplesLabel")} <SampleRunLinks onOpenUrl={onOpenUrl} downloading={downloading} />
      </p>
      {opened && <OpenedFileList file={opened} onOpen={onOpenRecord} onClose={onCloseFile} onRecompute={onRecompute} />}
      <h3>{t("historyLocal")}</h3>
      {entries.length === 0 ? (
        <p className="muted">{t("historyEmpty")}</p>
      ) : (
        <>
          <div className="row space">
            <span>{t("historyAverage", { avg: (avg ?? 0).toFixed(3), n: entries.length })}</span>
            <span className="row">
              <button type="button" className="btn btn-small" title={t("historyDownloadHint")} onClick={onDownloadAll}>
                {t("historyDownload")}
              </button>
              <button type="button" className="btn btn-small btn-ghost" onClick={onClear}>
                {t("historyClear")}
              </button>
            </span>
          </div>
          <ModelSummary entries={entries} />
          <table className="history">
            <thead>
              <tr>
                <th>{t("colWhen")}</th>
                <th>{t("colTask")}</th>
                <th>{t("colAgent")}</th>
                <th>{t("colUser")}</th>
                <th>{t("colReward")}</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.id} title={endReasonText(t, e.reason)}>
                  <td>{formatWhen(e.at)}</td>
                  <td>
                    <button type="button" className="link-btn" title={t("historyOpen")} onClick={() => onOpenEntry(e)}>
                      {e.domain}/{e.split}#{e.index}
                    </button>
                    {e.trial !== undefined && <span className="muted"> · {e.trial + 1}</span>}
                  </td>
                  <td className="clip" title={e.agent}>
                    {e.agent}
                  </td>
                  <td className="clip" title={e.user}>
                    {e.user}
                  </td>
                  <td className={e.reward === 1 ? "check-ok" : "check-bad"}>{e.reward.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

export function Inspector({
  tab,
  setTab,
  domain,
  task,
  episode,
  state,
  agentIsHuman,
  taskDone,
  history,
  onClearHistory,
  onDownloadHistory,
  onOpenHistory,
  opened,
  onOpenFile,
  onOpenUrl,
  downloading,
  onOpenRecord,
  onCloseFile,
  onRecompute,
  onUseTool,
}: {
  tab: InspectorTab;
  setTab: (t: InspectorTab) => void;
  domain: Domain | null;
  task: Task | null;
  episode: Episode | null;
  state: EpisodeState | null;
  /** Whether the agent of the shown task is (or will be) played by a human. */
  agentIsHuman: boolean;
  /** The shown task belongs to a finished episode (spoilers can open). */
  taskDone: boolean;
  history: HistoryEntry[];
  onClearHistory: () => void;
  onDownloadHistory: () => void;
  onOpenHistory: (e: HistoryEntry) => void;
  opened: OpenedFile | null;
  onOpenFile: (file: File) => void;
  onOpenUrl: (url: string) => void;
  downloading: boolean;
  onOpenRecord: (r: EpisodeRecord) => void;
  onCloseFile: () => void;
  onRecompute: () => void;
  onUseTool: (name: string) => void;
}) {
  const t = useT();
  const tabs: [InspectorTab, string][] = [
    ["task", t("tabTask")],
    ["policy", t("tabPolicy")],
    ["tools", t("tabTools")],
    ["result", t("tabResult")],
    ["data", t("tabData")],
    ["history", t("tabHistory")],
  ];
  const canUseTool = !!state && state.turn === "agent" && !!episode && episode.controllerOf("agent") === "human";
  // Arrow keys move between tabs (WAI-ARIA tabs pattern).
  const onTabKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const buttons = Array.from(e.currentTarget.querySelectorAll<HTMLButtonElement>("[role=tab]"));
    const focused = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const i = focused >= 0 ? focused : tabs.findIndex(([id]) => id === tab);
    const j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (j === undefined) return;
    e.preventDefault();
    const next = (j + tabs.length) % tabs.length;
    setTab(tabs[next][0]);
    buttons[next]?.focus();
  };
  return (
    <div className="inspector">
      <div className="tabs" role="tablist" onKeyDown={onTabKey}>
        {tabs.map(([id, label]) => (
          <button
            key={id}
            id={`inspector-tab-${id}`}
            type="button"
            role="tab"
            aria-selected={tab === id}
            aria-controls="inspector-panel"
            tabIndex={tab === id ? 0 : -1}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
          >
            {label}
            {id === "result" && state?.done && <span className={`dot ${state.done.reward === 1 ? "ok" : "bad"}`} />}
          </button>
        ))}
      </div>
      <div className="tab-panel" role="tabpanel" id="inspector-panel" aria-labelledby={`inspector-tab-${tab}`}>
        {tab === "history" ? (
          <HistoryTab
            entries={history}
            onClear={onClearHistory}
            onDownloadAll={onDownloadHistory}
            onOpenEntry={onOpenHistory}
            opened={opened}
            onOpenFile={onOpenFile}
            onOpenUrl={onOpenUrl}
            downloading={downloading}
            onOpenRecord={onOpenRecord}
            onCloseFile={onCloseFile}
            onRecompute={onRecompute}
          />
        ) : !domain || !task ? (
          <div className="tab-body muted">{t("loading")}</div>
        ) : tab === "task" ? (
          <TaskTab task={task} agentIsHuman={agentIsHuman} done={taskDone} />
        ) : tab === "policy" ? (
          <div className="tab-body">
            <p className="note">{t("policyNote")}</p>
            <Markdown text={domain.wiki} className="policy" />
          </div>
        ) : tab === "tools" ? (
          <ToolsTab domain={domain} onUse={onUseTool} canUse={canUseTool} />
        ) : tab === "result" ? (
          <ResultTab episode={episode} state={state} />
        ) : (
          <DataTab domain={domain} episode={episode} />
        )}
      </div>
    </div>
  );
}
