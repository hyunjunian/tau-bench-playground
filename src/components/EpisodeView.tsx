import { useEffect, useState } from "react";
import { useT } from "../i18n";
import type { Episode, EpisodeState } from "../tau/episode";
import { AgentComposer, UserComposer } from "./Composers";
import { Transcript } from "./Transcript";

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

function StatusBar({ episode, state }: { episode: Episode; state: EpisodeState }) {
  const t = useT();
  const now = useNow(!!state.retry);
  const role = state.turn;
  const label = role === "agent" ? t("thinkingAgent") : t("thinkingUser");
  return (
    <div className="status-bar">
      {state.busy ? (
        <span className="status">
          <span className="spinner" aria-hidden />
          {state.retry ? t("waitingRetry", { s: Math.max(0, Math.ceil((state.retry.until - now) / 1000)) }) : label}
        </span>
      ) : state.paused ? (
        <span className="status">⏸ {t("paused")}</span>
      ) : (
        <span className="status">
          <span className="spinner" aria-hidden />
          {label}
        </span>
      )}
      <span className="spacer" />
      {state.busy && (
        <button type="button" className="btn btn-small" onClick={() => episode.abort()}>
          {t("stop")}
        </button>
      )}
      {!state.busy && state.paused && (
        <>
          <button type="button" className="btn btn-small" onClick={() => episode.stepOnce()}>
            {t("stepOnce")}
          </button>
          <button type="button" className="btn btn-small btn-primary" onClick={() => episode.resume()}>
            {t("resume")}
          </button>
        </>
      )}
      {state.busy && !state.paused && (
        <button type="button" className="btn btn-small" onClick={() => episode.pause()}>
          {t("pause")}
        </button>
      )}
    </div>
  );
}

function ResultBanner({
  state,
  onDetails,
  onDownload,
  onAgain,
  onNext,
}: {
  state: EpisodeState;
  onDetails: () => void;
  onDownload: () => void;
  onAgain: () => void;
  onNext: () => void;
}) {
  const t = useT();
  const done = state.done!;
  const reasonText = {
    user_stop: t("doneUserStop"),
    transfer: t("doneTransfer"),
    max_steps: t("doneMaxSteps"),
    manual: t("doneManual"),
    crash: t("doneCrash"),
  }[done.reason];
  const ok = done.reward === 1;
  return (
    <div className={`result-banner ${ok ? "ok" : "bad"}`}>
      <div className="reward-big" aria-label={t("rewardLabel")}>
        <small>{t("rewardLabel")}</small>
        {done.reward.toFixed(1)}
      </div>
      <div className="result-text">
        <strong>{reasonText}</strong>
        {done.message && <div className="muted">{done.message}</div>}
        {done.info && (
          <div className="checks">
            <span className={done.info.rActions ? "check-ok" : "check-bad"}>
              {done.info.rActions ? "✓" : "✗"} {t("rActions")}
            </span>
            {done.info.rOutputs !== null && (
              <span className={done.info.rOutputs === 1 ? "check-ok" : "check-bad"}>
                {done.info.rOutputs === 1 ? "✓" : "✗"} {t("rOutputs")}
              </span>
            )}
          </div>
        )}
      </div>
      <div className="result-actions">
        <button type="button" className="btn btn-small" onClick={onDetails}>
          {t("viewDetails")}
        </button>
        <button type="button" className="btn btn-small" onClick={onDownload}>
          {t("download")}
        </button>
        <button type="button" className="btn btn-small" onClick={onAgain}>
          {t("sameTaskAgain")}
        </button>
        <button type="button" className="btn btn-small btn-primary" onClick={onNext}>
          {t("nextTask")} →
        </button>
      </div>
    </div>
  );
}

export function EpisodeView({
  episode,
  state,
  preferredTool,
  onDetails,
  onDownload,
  onAgain,
  onNext,
}: {
  episode: Episode;
  state: EpisodeState;
  preferredTool: string | null;
  onDetails: () => void;
  onDownload: () => void;
  onAgain: () => void;
  onNext: () => void;
}) {
  const t = useT();
  const { taskRef, domain, task } = episode.setup;
  const humanTurn = state.turn && episode.controllerOf(state.turn) === "human";
  const totalTokens = state.usage.agent.prompt + state.usage.agent.completion + state.usage.user.prompt + state.usage.user.completion;
  const userStarted = state.items.some((i) => i.kind === "user");

  let footer: React.ReactNode;
  if (state.done) {
    footer = <ResultBanner state={state} onDetails={onDetails} onDownload={onDownload} onAgain={onAgain} onNext={onNext} />;
  } else if (state.error) {
    footer = (
      <div className="error-box" role="alert">
        <strong>{t("errorTitle")}</strong>
        <p>{state.error.message}</p>
        <div className="row end">
          <button type="button" className="btn btn-small" onClick={() => episode.endNow()}>
            {t("endNow")}
          </button>
          <button type="button" className="btn btn-small btn-primary" onClick={() => episode.retry()}>
            {t("retry")}
          </button>
        </div>
      </div>
    );
  } else if (humanTurn && state.turn === "agent") {
    footer = (
      <AgentComposer
        tools={domain.toolsInfo}
        preferredTool={preferredTool}
        onReply={(text) => episode.sendAgentReply(text)}
        onTool={(name, kwargs) => episode.callAgentTool(name, kwargs)}
      />
    );
  } else if (humanTurn && state.turn === "user") {
    footer = <UserComposer instruction={task.instruction} started={userStarted} onSend={(text) => episode.sendUserMessage(text)} />;
  } else if (state.turn) {
    footer = <StatusBar episode={episode} state={state} />;
  }

  return (
    <div className="episode">
      <div className="episode-head">
        <span className="pill">
          {taskRef.domain} · {taskRef.split} · #{taskRef.index}
        </span>
        <span className="muted">{t("agentActions", { n: `${state.agentSteps}${episode.setup.maxAgentSteps ? `/${episode.setup.maxAgentSteps}` : ""}` })}</span>
        {totalTokens > 0 && <span className="muted">{t("tokens", { n: totalTokens.toLocaleString() })}</span>}
        <span className="spacer" />
        {!state.done && (
          <button type="button" className="btn btn-small btn-ghost" onClick={() => episode.endNow()}>
            {t("endNow")}
          </button>
        )}
      </div>
      <Transcript items={state.items} />
      {footer && <div className={`episode-footer ${state.done ? "is-done" : ""}`}>{footer}</div>}
    </div>
  );
}
