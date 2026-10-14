import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { EpisodeView } from "./components/EpisodeView";
import { Inspector, type InspectorTab } from "./components/Inspector";
import { SetupPanel } from "./components/SetupPanel";
import { loadDomain, loadTasks } from "./data/loader";
import { LangContext, makeT } from "./i18n";
import { type Endpoint, type ModelConfig, PROVIDERS, type ProviderId } from "./llm/providers";
import { useHistory } from "./state/history";
import { useSettings } from "./state/settings";
import type { Domain, Task } from "./tau/env";
import { Episode, type EpisodeState, type Role } from "./tau/episode";
import { downloadJson } from "./ui/format";

const noopSubscribe = () => () => {};
const nullState = () => null;

function useEpisodeState(episode: Episode | null): EpisodeState | null {
  return useSyncExternalStore(episode ? episode.subscribe : noopSubscribe, episode ? episode.getState : nullState);
}

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
  const [tab, setTab] = useState<InspectorTab>("task");
  const [preferredTool, setPreferredTool] = useState<string | null>(null);

  // Episodes read keys and model settings at call time, so fixes apply on retry.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  useEffect(() => {
    document.documentElement.lang = settings.lang;
  }, [settings.lang]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setLoadError(null);
    Promise.all([loadDomain(settings.domain), loadTasks(settings.domain, settings.split)])
      .then(([d, ts]) => {
        if (!alive) return;
        setDomain(d);
        setTasks(ts);
      })
      .catch((e: unknown) => alive && setLoadError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [settings.domain, settings.split]);

  const endpointFor = useCallback(
    (provider: ProviderId): Endpoint => {
      const s = settingsRef.current;
      const baseUrl = provider === "custom" ? s.customBaseUrl : PROVIDERS[provider].baseUrl;
      return { provider, baseUrl, apiKey: s.keys[provider] ?? "" };
    },
    [],
  );

  const resolveEndpoint = useCallback(
    (cfg: ModelConfig): Endpoint => {
      const endpoint = endpointFor(cfg.provider);
      if (!endpoint.apiKey && cfg.provider !== "custom") {
        throw new Error(makeT(settingsRef.current.lang)("keyMissing", { provider: PROVIDERS[cfg.provider].label }));
      }
      if (!cfg.model) throw new Error("No model selected.");
      return endpoint;
    },
    [endpointFor],
  );

  const task = tasks && domain?.name === settings.domain ? (tasks[Math.min(settings.taskIndex, tasks.length - 1)] ?? null) : null;

  const describeRole = (ep: Episode, role: Role) =>
    ep.controllerOf(role) === "human" ? "human" : ep.modelsUsed(role).join(", ") || settingsRef.current[role].model.model;

  const startEpisode = (taskIndex = settings.taskIndex) => {
    if (!domain || !tasks) return;
    const current = episode?.getState();
    if (current && !current.done && current.items.length > 1 && !window.confirm(t("confirmNew"))) return;
    episode?.abort();
    const index = Math.max(0, Math.min(taskIndex, tasks.length - 1));
    const ep = new Episode({
      domain,
      task: tasks[index],
      taskRef: { domain: domain.name, split: settings.split, index },
      controllers: { agent: settings.agent.controller, user: settings.user.controller },
      modelFor: (role) => settingsRef.current[role].model,
      maxAgentSteps: settings.maxAgentSteps,
      startPaused: settings.stepMode,
      resolveEndpoint,
      onFinish: (finished) => {
        const done = finished.getState().done;
        if (!done || finished.getState().items.length <= 1) return;
        history.add({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          at: new Date().toISOString(),
          domain: finished.setup.taskRef.domain,
          split: finished.setup.taskRef.split,
          index: finished.setup.taskRef.index,
          agent: describeRole(finished, "agent"),
          user: describeRole(finished, "user"),
          reward: done.reward,
          reason: done.reason,
          steps: finished.getState().agentSteps,
        });
      },
    });
    setPreferredTool(null);
    setEpisode(ep);
    setTab("task");
    ep.start();
  };

  const nextTask = () => {
    if (!tasks) return;
    const next = (settings.taskIndex + 1) % tasks.length;
    update({ taskIndex: next });
    startEpisode(next);
  };

  const download = () => {
    if (!episode) return;
    const { domain: d, split, index } = episode.setup.taskRef;
    downloadJson(`tau-bench-${d}-${split}-${index}-${Date.now()}.json`, episode.toJSON());
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
              taskCount={tasks && domain?.name === settings.domain ? tasks.length : null}
              loading={loading}
              loadError={loadError}
              episodeRunning={running}
              endpointFor={endpointFor}
              onStart={() => startEpisode()}
            />
          </aside>

          <section className="col col-main">
            {episode && state ? (
              <EpisodeView
                episode={episode}
                state={state}
                preferredTool={preferredTool}
                onDetails={() => setTab("result")}
                onDownload={download}
                onAgain={() => startEpisode(episode.setup.taskRef.index)}
                onNext={nextTask}
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
                <button type="button" className="btn btn-primary" disabled={loading || !task} onClick={() => startEpisode()}>
                  {t("start")}
                </button>
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
              onUseTool={(name) => setPreferredTool(name)}
            />
          </aside>
        </main>
      </div>
    </LangContext.Provider>
  );
}
