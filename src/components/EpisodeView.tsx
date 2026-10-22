import { useEffect, useMemo, useRef } from "react";
import { useT } from "../i18n";
import type { ProviderId } from "../llm/providers";
import type { Episode, EpisodeState } from "../tau/episode";
import { type SuggestionSource, collectSuggestions } from "../ui/suggest";
import { useNow } from "../ui/useNow";
import { AgentComposer, UserComposer } from "./Composers";
import { keyInputId } from "./SetupPanel";
import type { PreferredTool } from "./ToolForm";
import { Transcript } from "./Transcript";

function focusKeyInput(provider: ProviderId) {
  const input = document.getElementById(keyInputId(provider));
  input?.scrollIntoView({ block: "center", behavior: "smooth" });
  input?.focus({ preventScroll: true });
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
    incomplete: t("doneIncomplete"),
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
  preferredTool: PreferredTool | null;
  onDetails: () => void;
  onDownload: () => void;
  onAgain: () => void;
  onNext: () => void;
}) {
  const t = useT();
  const { taskRef, domain, task } = episode.setup;
  const root = useRef<HTMLDivElement>(null);
  // On a phone the columns stack, so bring a new episode into view.
  useEffect(() => {
    if (window.matchMedia("(max-width: 760px)").matches) root.current?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [episode]);
  const humanTurn = state.turn && episode.controllerOf(state.turn) === "human";
  const totalTokens = state.usage.agent.prompt + state.usage.agent.completion + state.usage.user.prompt + state.usage.user.completion;
  const userStarted = state.items.some((i) => i.kind === "user");
  const humanAgent = episode.controllerOf("agent") === "human";
  const suggestions = useMemo(() => {
    if (!humanAgent) return new Map<string, string[]>();
    const sources: SuggestionSource[] = [];
    for (const i of state.items) {
      if (i.kind === "tool") sources.push({ kind: "tool", name: i.name, observation: i.observation, error: i.error });
      else if (i.kind === "user") sources.push({ kind: "user", text: i.text });
    }
    return collectSuggestions(sources);
  }, [humanAgent, state.items]);

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
          {state.error.missingKey && (
            <button type="button" className="btn btn-small" onClick={() => focusKeyInput(state.error!.missingKey!)}>
              {t("enterKey")}
            </button>
          )}
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
        suggestions={suggestions}
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
    <div className="episode" ref={root}>
      <div className="episode-head">
        <span className="pill">
          {taskRef.domain} · {taskRef.split} · #{taskRef.index}
        </span>
        {episode.setup.replayOf && (
          <span className="badge badge-accent replay-badge" title={t("replayNote")}>
            {t("replayBadge")} · {episode.setup.replayOf}
          </span>
        )}
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
