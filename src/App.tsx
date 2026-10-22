import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { BatchCard } from "./components/BatchCard";
import { EpisodeView } from "./components/EpisodeView";
import { Inspector, type InspectorTab, OpenFileButton, type OpenedFile, type Recomputed } from "./components/Inspector";
import { SetupPanel } from "./components/SetupPanel";
import type { PreferredTool } from "./components/ToolForm";
import { loadDomain, loadTasks } from "./data/loader";
import { LangContext, type TKey, makeT } from "./i18n";
import { MissingKeyError } from "./llm/client";
import { type Endpoint, type ModelConfig, PROVIDERS, type ProviderId } from "./llm/providers";
import { archiveEpisode, loadArchivedEpisode } from "./state/archive";
import { type HistoryEntry, useHistory } from "./state/history";
import { type Theme, taskFromHash, taskHash, useSettings } from "./state/settings";
import { type BatchPlan, BatchRun, type BatchResult, type BatchState } from "./tau/batch";
import type { Domain, Task } from "./tau/env";
import { type Controller, Episode, type EpisodeState, type Role } from "./tau/episode";
import { isSuccess } from "./tau/metrics";
import { type EpisodeRecord, parseRecordsFile, replayRecord } from "./tau/records";
import { downloadJson } from "./ui/format";

const noopSubscribe = () => () => {};
const nullState = () => null;

function useEpisodeState(episode: Episode | null): EpisodeState | null {
  return useSyncExternalStore(episode ? episode.subscribe : noopSubscribe, episode ? episode.getState : nullState);
}

function useBatchState(batch: BatchRun | null): BatchState | null {
  return useSyncExternalStore(batch ? batch.subscribe : noopSubscribe, batch ? batch.getState : nullState);
}

/** What to run: a task and where it lives in the dataset. */
interface RunSpec {
  domain: Domain;
  task: Task;
  split: string;
  index: number;
}

const loaders = { loadDomain, loadTasks };

const THEMES: Record<Theme, { icon: string; label: TKey; next: Theme }> = {
  system: { icon: "◐", label: "themeSystem", next: "light" },
  light: { icon: "☀", label: "themeLight", next: "dark" },
  dark: { icon: "☾", label: "themeDark", next: "system" },
};

export default function App() {
  const [settings, update] = useSettings();
  const t = useMemo(() => makeT(settings.lang), [settings.lang]);
  const history = useHistory();

  const [domain, setDomain] = useState<Domain | null>(null);
  const [tasks, setTasks] = useState<Task[] | null>(null);
  /** "<domain>/<split>" of `tasks` (they lag behind the settings while a new split loads). */
  const [tasksKey, setTasksKey] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [episode, setEpisode] = useState<Episode | null>(null);
  const state = useEpisodeState(episode);
  const [batch, setBatch] = useState<BatchRun | null>(null);
  const batchState = useBatchState(batch);
  const [opened, setOpened] = useState<OpenedFile | null>(null);
  const [tab, setTab] = useState<InspectorTab>("task");
  // A fresh object per click, so picking the same tool again still switches the form to it.
  const [preferredTool, setPreferredTool] = useState<PreferredTool | null>(null);

  // Episodes read keys and model settings at call time, so fixes apply on retry.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const tRef = useRef(t);
  tRef.current = t;

  useEffect(() => {
    document.documentElement.lang = settings.lang;
  }, [settings.lang]);

  // index.html applies a stored theme before the first paint; this keeps it in sync afterwards.
  useEffect(() => {
    const root = document.documentElement;
    if (settings.theme === "system") delete root.dataset.theme;
    else root.dataset.theme = settings.theme;
  }, [settings.theme]);

  // Batch progress in the tab title, for runs left in a background tab.
  const batchProgress =
    batchState?.status === "running" ? `(${batchState.results.length}/${batchState.plan.indices.length * batchState.plan.trials}) ` : "";
  useEffect(() => {
    document.title = `${batchProgress}${t("appTitle")}`;
  }, [batchProgress, t]);

  // The address bar names the selected task (#retail/test/12), so a task can be shared or bookmarked.
  const hash = taskHash(settings);
  useEffect(() => {
    if (location.hash !== hash) window.history.replaceState(null, "", hash);
  }, [hash]);

  useEffect(() => {
    const onHash = () => {
      const link = taskFromHash(location.hash);
      if (link) update(link);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [update]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setLoadError(null);
    Promise.all([loadDomain(settings.domain), loadTasks(settings.domain, settings.split)])
      .then(([d, ts]) => {
        if (!alive) return;
        setDomain(d);
        setTasks(ts);
        setTasksKey(`${d.name}/${settings.split}`);
      })
      .catch((e: unknown) => alive && setLoadError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [settings.domain, settings.split]);

  const endpointFor = useCallback((provider: ProviderId): Endpoint => {
    const s = settingsRef.current;
    const baseUrl = provider === "custom" ? s.customBaseUrl : PROVIDERS[provider].baseUrl;
    return { provider, baseUrl, apiKey: s.keys[provider] ?? "" };
  }, []);

  const resolveEndpoint = useCallback(
    (cfg: ModelConfig): Endpoint => {
      const endpoint = endpointFor(cfg.provider);
      if (!endpoint.apiKey && cfg.provider !== "custom") {
        throw new MissingKeyError(cfg.provider, tRef.current("keyMissing", { provider: PROVIDERS[cfg.provider].label }));
      }
      if (!cfg.model) throw new Error("No model selected.");
      return endpoint;
    },
    [endpointFor],
  );

  const tasksReady = !!tasks && tasksKey === `${settings.domain}/${settings.split}`;
  const taskCount = tasksReady ? tasks.length : null;
  const task = tasksReady ? (tasks[Math.min(settings.taskIndex, tasks.length - 1)] ?? null) : null;

  // Keep the selection inside the split, e.g. after following a link to #retail/test/999.
  useEffect(() => {
    if (taskCount !== null && settings.taskIndex >= taskCount) update({ taskIndex: Math.max(0, taskCount - 1) });
  }, [taskCount, settings.taskIndex, update]);

  const describeRole = (ep: Episode, role: Role) =>
    ep.controllerOf(role) === "human" ? "human" : ep.modelsUsed(role).join(", ") || settingsRef.current[role].model.model;

  const record = (finished: Episode, trial?: number) => {
    const s = finished.getState();
    if (!s.done || s.items.length <= 1) return;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const ref = finished.setup.taskRef;
    const entry: HistoryEntry = {
      id,
      at: new Date().toISOString(),
      domain: ref.domain,
      split: ref.split,
      index: ref.index,
      agent: describeRole(finished, "agent"),
      user: describeRole(finished, "user"),
      reward: s.done.reward,
      reason: s.done.reason,
      steps: s.agentSteps,
      trial,
    };
    history.add(entry);
    archiveEpisode(id, JSON.stringify(trial === undefined ? finished.toJSON() : { ...finished.toJSON(), trial })).catch(() => {});
  };

  /** Create and start an episode; `controllers` defaults to the current settings. */
  const launch = (
    spec: RunSpec,
    opts: {
      controllers?: Record<Role, Controller>;
      trial?: number;
      /** Batch episodes run unattended: no step-by-step pausing, no tab switching. */
      unattended?: boolean;
      onFinish?: (ep: Episode) => void;
    } = {},
  ): Episode => {
    const s = settingsRef.current;
    const ep = new Episode({
      domain: spec.domain,
      task: spec.task,
      taskRef: { domain: spec.domain.name, split: spec.split, index: spec.index },
      controllers: opts.controllers ?? { agent: s.agent.controller, user: s.user.controller },
      modelFor: (role) => settingsRef.current[role].model,
      maxAgentSteps: s.maxAgentSteps,
      startPaused: s.stepMode && !opts.unattended,
      resolveEndpoint,
      onFinish: (finished) => {
        record(finished, opts.trial);
        opts.onFinish?.(finished);
      },
    });
    setPreferredTool(null);
    setEpisode(ep);
    if (!opts.unattended) setTab("task");
    ep.start();
    return ep;
  };

  // Handlers that run after an await read these, not the values captured when they started.
  const episodeRef = useRef(episode);
  episodeRef.current = episode;
  const batchRef = useRef(batch);
  batchRef.current = batch;

  const inProgress = (ep: Episode | null) => {
    const s = ep?.getState();
    return !!s && !s.done && s.items.length > 1;
  };

  /**
   * Ask before throwing away an episode in progress or a running batch. On yes the batch lets go of
   * its episode and the episode on screen is stopped.
   */
  const confirmReplace = (): boolean => {
    const b = batchRef.current;
    if (b?.getState().status === "running") {
      if (!window.confirm(t("confirmStopBatch"))) return false;
    } else if (inProgress(episodeRef.current) && !window.confirm(t("confirmNew"))) {
      return false;
    }
    b?.detach();
    episodeRef.current?.abort();
    return true;
  };

  const replaceWith = (spec: RunSpec): boolean => {
    if (!confirmReplace()) return false;
    launch(spec);
    return true;
  };

  /** Resuming a batch shows its episode again, so whatever is on screen gets the same question. */
  const resumeBatch = () => {
    const b = batchRef.current;
    if (!b) return;
    const shown = episodeRef.current;
    if (shown !== b.currentEpisode) {
      if (inProgress(shown) && !window.confirm(t("confirmNew"))) return;
      shown?.abort();
    }
    b.resume();
  };

  const startFromSettings = () => {
    if (!domain || !tasks || !task) return;
    const index = Math.max(0, Math.min(settings.taskIndex, tasks.length - 1));
    replaceWith({ domain, task: tasks[index], split: settings.split, index });
  };

  /** Run the finished episode's own task again (not whatever the setup panel shows now). */
  const runAgain = (ep: Episode) => {
    const ref = ep.setup.taskRef;
    replaceWith({ domain: ep.setup.domain, task: ep.setup.task, split: ref.split, index: ref.index });
  };

  /** The task after the episode's task, in the episode's domain and split. */
  const runNext = async (ep: Episode) => {
    const ref = ep.setup.taskRef;
    let list: Task[];
    try {
      list = await loadTasks(ref.domain, ref.split);
    } catch {
      return;
    }
    const index = list.length ? (ref.index + 1) % list.length : 0;
    if (!list[index]) return;
    if (replaceWith({ domain: ep.setup.domain, task: list[index], split: ref.split, index })) {
      update({ domain: ref.domain, split: ref.split, taskIndex: index });
    }
  };

  const runBatch = (plan: BatchPlan, batchDomain: Domain, batchTasks: Task[]) => {
    if (!confirmReplace()) return;
    const run = new BatchRun(plan, ({ index, trial }, onFinish) =>
      launch(
        { domain: batchDomain, task: batchTasks[index], split: plan.split, index },
        { controllers: { agent: "model", user: "model" }, trial, unattended: true, onFinish },
      ),
    );
    setBatch(run);
    run.start();
  };

  const startBatch = () => {
    if (!domain || !tasks || !tasksReady || tasks.length === 0) return;
    const b = settings.batch;
    const from = Math.min(b.from, tasks.length - 1);
    const to = Math.min(Math.max(b.to, from), tasks.length - 1);
    const indices = Array.from({ length: to - from + 1 }, (_, i) => from + i);
    runBatch({ domain: domain.name, split: settings.split, indices, trials: b.trials }, domain, tasks);
  };

  /** A new batch over the tasks of the shown batch that failed at least once, with the same number of trials. */
  const rerunFailed = async () => {
    const shown = batchRef.current?.getState();
    if (!shown) return;
    const failed = [...new Set(shown.results.filter((r) => !isSuccess(r.reward)).map((r) => r.index))].sort((a, b) => a - b);
    if (failed.length === 0) return;
    const { plan } = shown;
    const [batchDomain, batchTasks] = await Promise.all([loadDomain(plan.domain), loadTasks(plan.domain, plan.split)]);
    runBatch({ ...plan, indices: failed }, batchDomain, batchTasks);
  };

  const closeBatch = () => {
    batch?.detach();
    setBatch(null);
  };

  const downloadBatch = () => {
    if (!batch) return;
    const { plan } = batch.getState();
    downloadJson(`tau-bench-${plan.domain}-${plan.split}-batch-${Date.now()}.json`, batch.exportResults());
  };

  const download = () => {
    if (!episode) return;
    const { domain: d, split, index } = episode.setup.taskRef;
    downloadJson(`tau-bench-${d}-${split}-${index}-${Date.now()}.json`, episode.toJSON());
  };

  /** Rebuild a saved episode and show it (read-only, already scored). */
  const openRecord = async (rec: EpisodeRecord, label: string) => {
    let ep: Episode;
    try {
      ep = await replayRecord(rec, loaders, label);
    } catch (e) {
      window.alert(t("openFailed", { error: e instanceof Error ? e.message : String(e) }));
      return false;
    }
    // Only now that the replay worked, ask about what is on screen.
    if (!confirmReplace()) return false;
    const ref = ep.setup.taskRef;
    if (ref.split !== "file") update({ domain: ref.domain, split: ref.split, taskIndex: ref.index });
    setPreferredTool(null);
    setEpisode(ep);
    setTab("result");
    return true;
  };

  const openedRef = useRef(opened);
  openedRef.current = opened;

  /** Replay every record of the opened file and compare its reward with the recorded one. */
  const recomputeOpened = async () => {
    const file = openedRef.current;
    if (!file || file.recomputing) return;
    const results = new Map<number, Recomputed>();
    const publish = (recomputing: boolean) =>
      setOpened((o) => (o?.records === file.records ? { ...o, recomputed: new Map(results), recomputing } : o));
    publish(true);
    for (const r of file.records) {
      if (openedRef.current?.records !== file.records) return; // closed or replaced
      try {
        const done = (await replayRecord(r, loaders)).getState().done!;
        results.set(r.position, { reward: done.reward, reason: done.reason });
      } catch (e) {
        results.set(r.position, { error: e instanceof Error ? e.message : String(e) });
      }
      publish(true);
      // Let the page paint between replays (each one copies the database).
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    publish(false);
  };

  const openFile = async (file: File) => {
    try {
      const records = parseRecordsFile(await file.text(), file.name);
      setOpened({ name: file.name, records });
      if (records.length === 1) void openRecord(records[0], file.name);
      else setTab("history");
    } catch (e) {
      window.alert(t("openFailed", { error: e instanceof Error ? e.message : String(e) }));
    }
  };

  // A JSON file dropped anywhere on the page is opened like "Open JSON…".
  const openFileRef = useRef(openFile);
  openFileRef.current = openFile;
  useEffect(() => {
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const over = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const drop = (e: DragEvent) => {
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      e.preventDefault();
      void openFileRef.current(file);
    };
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []);

  /** Every stored episode, oldest first, in the same list shape as a batch results file. */
  const downloadHistory = async () => {
    const records: unknown[] = [];
    for (const entry of [...history.entries].reverse()) {
      const text = await loadArchivedEpisode(entry.id).catch(() => null);
      try {
        if (text) records.push(JSON.parse(text));
      } catch {
        // a damaged entry is skipped
      }
    }
    if (records.length === 0) {
      window.alert(t("historyNotStored"));
      return;
    }
    downloadJson(`tau-bench-history-${Date.now()}.json`, records);
  };

  const openHistoryEntry = async (entry: HistoryEntry) => {
    let text: string | null = null;
    try {
      text = await loadArchivedEpisode(entry.id);
    } catch {
      text = null;
    }
    let rec: EpisodeRecord | undefined;
    try {
      rec = text ? parseRecordsFile(text)[0] : undefined;
    } catch {
      rec = undefined;
    }
    if (!rec) {
      window.alert(t("historyNotStored"));
      return;
    }
    await openRecord(rec, `${t("tabHistory")} · ${new Date(entry.at).toLocaleString()}`);
  };

  const openBatchResult = (r: BatchResult) => {
    const [rec] = parseRecordsFile(JSON.stringify(r.record));
    void openRecord(rec, `${t("batchTitle")} · #${r.index} · ${t("colTrial")} ${r.trial + 1}`);
  };

  const running = !!state && !state.done;
  // The inspector follows the episode, except that a finished episode gives way to the task picked next.
  const ref = episode?.setup.taskRef;
  const showEpisodeTask =
    !!episode && (running || (ref?.domain === settings.domain && ref?.split === settings.split && ref?.index === settings.taskIndex));

  return (
    <LangContext.Provider value={{ lang: settings.lang, t }}>
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <span className="logo" aria-hidden>
              τ
            </span>
            <div>
              <h1>{t("appTitle")}</h1>
              <p>{t("appTagline")}</p>
            </div>
          </div>
          <nav className="top-links">
            <a href="https://arxiv.org/abs/2406.12045" target="_blank" rel="noopener noreferrer">
              {t("paper")} ↗
            </a>
            <a href="https://github.com/sierra-research/tau-bench" target="_blank" rel="noopener noreferrer">
              tau-bench ↗
            </a>
            <a href="https://github.com/hyunjunian/tau-bench-playground" target="_blank" rel="noopener noreferrer">
              {t("source")} ↗
            </a>
            <div className="segmented lang">
              <button type="button" className={settings.lang === "ko" ? "active" : ""} onClick={() => update({ lang: "ko" })}>
                한국어
              </button>
              <button type="button" className={settings.lang === "en" ? "active" : ""} onClick={() => update({ lang: "en" })}>
                EN
              </button>
            </div>
            <button
              type="button"
              className="btn btn-small theme-toggle"
              title={t("themeTitle", { theme: t(THEMES[settings.theme].label) })}
              aria-label={t("themeTitle", { theme: t(THEMES[settings.theme].label) })}
              onClick={() => update({ theme: THEMES[settings.theme].next })}
            >
              {THEMES[settings.theme].icon}
            </button>
          </nav>
        </header>

        <main className="layout">
          <aside className="col col-setup">
            <SetupPanel
              settings={settings}
              update={update}
              taskCount={taskCount}
              tasks={taskCount !== null ? tasks : null}
              history={history.entries}
              loading={loading}
              loadError={loadError}
              episodeRunning={running}
              batchRunning={batchState?.status === "running"}
              endpointFor={endpointFor}
              onStart={startFromSettings}
              onBatch={startBatch}
            />
          </aside>

          <section className="col col-main">
            {batchState && (
              <BatchCard
                state={batchState}
                waitingOnError={batchState.status === "running" && !!state?.error && episode === batch?.currentEpisode}
                onStop={() => batch?.stop()}
                onResume={resumeBatch}
                onClose={closeBatch}
                onDownload={downloadBatch}
                onRerunFailed={() => void rerunFailed()}
                onOpen={openBatchResult}
              />
            )}
            {episode && state ? (
              <EpisodeView
                episode={episode}
                state={state}
                preferredTool={preferredTool}
                onDetails={() => setTab("result")}
                onDownload={download}
                onAgain={() => runAgain(episode)}
                onNext={() => void runNext(episode)}
              />
            ) : (
              <div className="welcome">
                <h2>{t("welcomeTitle")}</h2>
                <p>{t("welcomeBody")}</p>
                <ol>
                  <li>{t("welcomeStep1")}</li>
                  <li>{t("welcomeStep2")}</li>
                  <li>{t("welcomeStep3")}</li>
                </ol>
                <button type="button" className="btn btn-primary" disabled={loading || !task} onClick={startFromSettings}>
                  {t("start")}
                </button>
                <p className="welcome-open muted">
                  {t("welcomeBatch")}
                  <br />
                  {t("welcomeOpen")} <OpenFileButton onOpen={(f) => void openFile(f)} />
                </p>
              </div>
            )}
          </section>

          <aside className="col col-inspector">
            <Inspector
              tab={tab}
              setTab={setTab}
              domain={showEpisodeTask ? episode.setup.domain : domain}
              task={showEpisodeTask ? episode.setup.task : task}
              episode={episode}
              state={state}
              agentIsHuman={showEpisodeTask && running ? episode.controllerOf("agent") === "human" : settings.agent.controller === "human"}
              taskDone={showEpisodeTask && !!state?.done}
              history={history.entries}
              onClearHistory={history.clear}
              onDownloadHistory={() => void downloadHistory()}
              onOpenHistory={(e) => void openHistoryEntry(e)}
              opened={opened}
              onOpenFile={(f) => void openFile(f)}
              onOpenRecord={(r) => void openRecord(r, `${opened?.name ?? ""} · ${r.position + 1}`)}
              onCloseFile={() => setOpened(null)}
              onRecompute={() => void recomputeOpened()}
              onUseTool={(name) => setPreferredTool({ name })}
            />
          </aside>
        </main>
      </div>
    </LangContext.Provider>
  );
}
