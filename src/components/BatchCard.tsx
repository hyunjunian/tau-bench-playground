import { useT } from "../i18n";
import { type BatchResult, type BatchRunRef, type BatchState, MAX_AUTO_RETRIES } from "../tau/batch";
import { averageReward, isSuccess, passHatK } from "../tau/metrics";
import { endReasonText } from "../ui/reasons";
import { useNow } from "../ui/useNow";

function cellClass(result: BatchResult | undefined, running: boolean): string {
  if (result) return isSuccess(result.reward) ? "cell ok" : "cell bad";
  return running ? "cell running" : "cell";
}

export function BatchCard({
  state,
  waitingOnError,
  onStop,
  onResume,
  onClose,
  onDownload,
  onRerunFailed,
  onOpen,
}: {
  state: BatchState;
  /** The current episode stopped on an error that needs the user (shown below the batch card). */
  waitingOnError: boolean;
  onStop: () => void;
  onResume: () => void;
  onClose: () => void;
  onDownload: () => void;
  onRerunFailed: () => void;
  onOpen: (result: BatchResult) => void;
}) {
  const t = useT();
  const { plan, results, current, status } = state;
  const total = plan.indices.length * plan.trials;
  const outcomes = results.map((r) => ({ task: String(r.index), reward: r.reward }));
  const avg = averageReward(outcomes);
  const passK = passHatK(outcomes);
  const tokens = results.reduce((s, r) => s + r.tokens, 0);
  const timed = results.filter((r) => r.ms > 0);
  const left = total - results.length;
  const etaMin =
    status === "running" && timed.length > 0 && left > 0
      ? Math.ceil((left * timed.reduce((s, r) => s + r.ms, 0)) / timed.length / 60_000)
      : null;
  const byRun = new Map(results.map((r) => [`${r.index}/${r.trial}`, r]));
  const isCurrent = (run: BatchRunRef) => current?.index === run.index && current.trial === run.trial;
  const first = plan.indices[0];
  const last = plan.indices[plan.indices.length - 1];
  const contiguous = plan.indices.every((index, i) => index === first + i);
  const range = contiguous ? `#${first}–${last}` : t("batchTaskCount", { n: plan.indices.length });
  const statusText = { running: t("batchRunning"), stopped: t("batchStopped"), done: t("batchDone") }[status];
  const now = useNow(!!state.autoRetry);

  return (
    <div className={`batch-card is-${status}`}>
      <div className="batch-head">
        <div className="batch-stats">
          <strong>{t("batchTitle")}</strong>
          <span className="pill">
            {plan.domain} · {plan.split} · {range} × {plan.trials}
          </span>
          <span className={`badge ${status === "done" ? "badge-accent" : ""}`}>
            {status === "running" && <span className="spinner" aria-hidden />}
            {statusText}
          </span>
          <span className="muted">{t("batchProgress", { done: results.length, total })}</span>
          {etaMin !== null && <span className="muted">{t("batchEta", { m: etaMin })}</span>}
          {tokens > 0 && <span className="muted">{t("tokens", { n: tokens.toLocaleString() })}</span>}
          {waitingOnError && !state.autoRetry && <span className="batch-retry">{t("batchWaiting")}</span>}
          {state.autoRetry && (
            <span className="batch-retry">
              {t("batchAutoRetry", {
                s: Math.max(0, Math.ceil((state.autoRetry.at - now) / 1000)),
                n: state.autoRetry.attempt,
                max: MAX_AUTO_RETRIES,
              })}
            </span>
          )}
          {avg !== null && <strong className="batch-avg">{t("batchAvg", { avg: avg.toFixed(3) })}</strong>}
          {passK.length > 1 &&
            passK.map(({ k, value }) => (
              <span key={k} className="muted">
                pass^{k} {value.toFixed(3)}
              </span>
            ))}
        </div>
        <div className="batch-actions">
          {status === "running" && (
            <button type="button" className="btn btn-small" onClick={onStop}>
              {t("batchStop")}
            </button>
          )}
          {status === "stopped" && (
            <button type="button" className="btn btn-small btn-primary" onClick={onResume}>
              {t("batchResume")}
            </button>
          )}
          {status !== "running" && results.some((r) => !isSuccess(r.reward)) && (
            <button type="button" className="btn btn-small" title={t("batchRerunHint")} onClick={onRerunFailed}>
              {t("batchRerun")}
            </button>
          )}
          {results.length > 0 && (
            <button type="button" className="btn btn-small" onClick={onDownload}>
              {t("batchDownload")}
            </button>
          )}
          {status !== "running" && (
            <button type="button" className="btn btn-small btn-ghost" onClick={onClose}>
              {t("batchClose")}
            </button>
          )}
        </div>
      </div>
      <div className="progress" role="progressbar" aria-label={t("batchTitle")} aria-valuemin={0} aria-valuemax={total} aria-valuenow={results.length}>
        <div style={{ width: `${total ? (100 * results.length) / total : 0}%` }} />
      </div>
      <div className="batch-grid" title={t("batchOpenHint")}>
        {plan.indices.map((index) => (
          <span key={index} className="batch-task">
            <span className="batch-index">{index}</span>
            {Array.from({ length: plan.trials }, (_, trial) => {
              const r = byRun.get(`${index}/${trial}`);
              const running = isCurrent({ index, trial });
              const label = r
                ? t("batchCell", { index, trial: trial + 1, reward: r.reward.toFixed(1), reason: endReasonText(t, r.reason) })
                : running
                  ? t("batchCellRunning", { index, trial: trial + 1 })
                  : t("batchCellPending", { index, trial: trial + 1 });
              return (
                <button
                  key={trial}
                  type="button"
                  className={cellClass(r, running)}
                  title={label}
                  aria-label={label}
                  disabled={!r}
                  onClick={() => r && onOpen(r)}
                />
              );
            })}
          </span>
        ))}
      </div>
    </div>
  );
}
