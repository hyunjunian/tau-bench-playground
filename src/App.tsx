import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { BatchCard } from "./components/BatchCard";
import type { PreferredTool } from "./components/ToolForm";
import { EpisodeView } from "./components/EpisodeView";
import { Inspector, type InspectorTab, OpenFileButton, type OpenedFile } from "./components/Inspector";
import { SetupPanel } from "./components/SetupPanel";
import { loadDomain, loadTasks } from "./data/loader";
import { LangContext, makeT } from "./i18n";
import { MissingKeyError } from "./llm/client";
import { type Endpoint, type ModelConfig, PROVIDERS, type ProviderId } from "./llm/providers";
import { archiveEpisode, loadArchivedEpisode } from "./state/archive";
import { type HistoryEntry, useHistory } from "./state/history";
import { taskFromHash, taskHash, useSettings } from "./state/settings";
import { BatchRun, type BatchResult, type BatchState } from "./tau/batch";
import type { Domain, Task } from "./tau/env";
import { type Controller, Episode, type EpisodeState, type Role } from "./tau/episode";
import { type EpisodeRecord, parseRecordsFile, resolveRecord } from "./tau/records";
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

export default function App() {
  const [settings, update] = useSettings();
  const t = useMemo(() => makeT(settings.lang), [settings.lang]);
  const history = useHistory();

  const [domain, setDomain] = useState<Domain | null>(null);
  const [tasks, setTasks] = useState<Task[] | null>(null);
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
        // e.g. a link to #retail/test/999
        update((s) => (s.taskIndex >= ts.length ? { ...s, taskIndex: Math.max(0, ts.length - 1) } : s));
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

  const taskCount = tasks && domain?.name === settings.domain ? tasks.length : null;
  const task = tasks && domain?.name === settings.domain ? (tasks[Math.min(settings.taskIndex, tasks.length - 1)] ?? null) : null;

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
    opts: { controllers?: Record<Role, Controller>; trial?: number; keepTab?: boolean; onFinish?: (ep: Episode) => void } = {},
  ): Episode => {
    const s = settingsRef.current;
    const ep = new Episode({
      domain: spec.domain,
      task: spec.task,
      taskRef: { domain: spec.domain.name, split: spec.split, index: spec.index },
      controllers: opts.controllers ?? { agent: s.agent.controller, user: s.user.controller },
      modelFor: (role) => settingsRef.current[role].model,
      maxAgentSteps: s.maxAgentSteps,
      startPaused: s.stepMode,
      resolveEndpoint,
      onFinish: (finished) => {
        record(finished, opts.trial);
        opts.onFinish?.(finished);
      },
    });
    setPreferredTool(null);
    setEpisode(ep);
    if (!opts.keepTab) setTab("task");
    ep.start();
    return ep;
  };

  /** Ask before throwing away an episode in progress or a running batch; the batch lets go of its episode. */
  const confirmReplace = (): boolean => {
    if (batchState?.status === "running") {
      if (!window.confirm(t("confirmStopBatch"))) return false;
    } else {
      const current = episode?.getState();
      if (current && !current.done && current.items.length > 1 && !window.confirm(t("confirmNew"))) return false;
    }
    batch?.detach();
    return true;
  };

  const replaceWith = (spec: RunSpec): boolean => {
    if (!confirmReplace()) return false;
    episode?.abort();
    launch(spec);
    return true;
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

  const startBatch = () => {
    if (!domain || !tasks || tasks.length === 0) return;
    if (!confirmReplace()) return;
    episode?.abort();
    const b = settings.batch;
    const from = Math.min(b.from, tasks.length - 1);
    const to = Math.min(Math.max(b.to, from), tasks.length - 1);
    const indices = Array.from({ length: to - from + 1 }, (_, i) => from + i);
    const batchDomain = domain;
    const batchTasks = tasks;
    const split = settings.split;
    const run = new BatchRun({ domain: batchDomain.name, split, indices, trials: b.trials }, ({ index, trial }, onFinish) =>
      launch(
        { domain: batchDomain, task: batchTasks[index], split, index },
        { controllers: { agent: "model", user: "model" }, trial, keepTab: true, onFinish },
      ),
    );
    setBatch(run);
    run.start();
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
    if (!confirmReplace()) return false;
    try {
      const resolved = await resolveRecord(rec, loaders);
      const ep = Episode.replay(
        {
          domain: resolved.domain,
          task: resolved.task,
          taskRef: resolved.taskRef,
          controllers: { agent: rec.controllers.agent ?? "model", user: rec.controllers.user ?? "model" },
          modelFor: (role) => settingsRef.current[role].model,
          maxAgentSteps: resolved.maxAgentSteps,
          resolveEndpoint,
          replayOf: label,
        },
        resolved.source,
      );
      episode?.abort();
      const ref = resolved.taskRef;
      if (ref.split !== "file") update({ domain: ref.domain, split: ref.split, taskIndex: ref.index });
      setPreferredTool(null);
      setEpisode(ep);
      setTab("result");
      return true;
    } catch (e) {
      window.alert(t("openFailed", { error: e instanceof Error ? e.message : String(e) }));
      return false;
    }
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
                onStop={() => batch?.stop()}
                onResume={() => batch?.resume()}
                onClose={closeBatch}
                onDownload={downloadBatch}
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
              onOpenHistory={(e) => void openHistoryEntry(e)}
              opened={opened}
              onOpenFile={(f) => void openFile(f)}
              onOpenRecord={(r) => void openRecord(r, `${opened?.name ?? ""} · ${r.position + 1}`)}
              onCloseFile={() => setOpened(null)}
              onUseTool={(name) => setPreferredTool({ name })}
            />
          </aside>
        </main>
      </div>
    </LangContext.Provider>
  );
}
