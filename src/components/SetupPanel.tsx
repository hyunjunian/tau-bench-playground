import { useState } from "react";
import { useT } from "../i18n";
import { type Endpoint, PROVIDERS, type ProviderId } from "../llm/providers";
import type { HistoryEntry } from "../state/history";
import type { RoleSettings, Settings } from "../state/settings";
import { DOMAIN_SPLITS, type DomainName, type Task } from "../tau/env";
import type { Controller, Role } from "../tau/episode";
import { ModelPicker } from "./ModelPicker";
import { TaskPicker } from "./TaskPicker";

interface Props {
  settings: Settings;
  update: (patch: Partial<Settings> | ((s: Settings) => Settings)) => void;
  taskCount: number | null;
  /** Tasks of the selected domain and split, once loaded. */
  tasks: Task[] | null;
  history: HistoryEntry[];
  loading: boolean;
  loadError: string | null;
  episodeRunning: boolean;
  batchRunning: boolean;
  endpointFor: (p: ProviderId) => Endpoint;
  onStart: () => void;
  onBatch: () => void;
}

const QUICK: { key: "quickWatch" | "quickPlayAgent" | "quickPlayUser" | "quickHumans"; agent: Controller; user: Controller }[] = [
  { key: "quickWatch", agent: "model", user: "model" },
  { key: "quickPlayAgent", agent: "human", user: "model" },
  { key: "quickPlayUser", agent: "model", user: "human" },
  { key: "quickHumans", agent: "human", user: "human" },
];

function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={value === o.value ? "active" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function RoleCard({
  role,
  value,
  onChange,
  endpointFor,
}: {
  role: Role;
  value: RoleSettings;
  onChange: (v: RoleSettings) => void;
  endpointFor: (p: ProviderId) => Endpoint;
}) {
  const t = useT();
  return (
    <div className={`role-card role-${role}`}>
      <div className="role-head">
        <span className="role-dot" />
        <strong>{t(role === "agent" ? "roleAgent" : "roleUser")}</strong>
        <small>{t(role === "agent" ? "roleAgentHint" : "roleUserHint")}</small>
      </div>
      <Segmented<Controller>
        value={value.controller}
        options={[
          { value: "human", label: t("human") },
          { value: "model", label: t("model") },
        ]}
        onChange={(controller) => onChange({ ...value, controller })}
      />
      {value.controller === "model" ? (
        <ModelPicker value={value.model} onChange={(model) => onChange({ ...value, model })} endpointFor={endpointFor} />
      ) : (
        // A human can still ask this model for a draft of the next move.
        <details className="draft-model">
          <summary>{t("draftModelTitle", { model: value.model.model || "?" })}</summary>
          <ModelPicker value={value.model} onChange={(model) => onChange({ ...value, model })} endpointFor={endpointFor} />
        </details>
      )}
    </div>
  );
}

/** DOM id of a provider's API key field (the episode's error box links to it). */
export const keyInputId = (provider: ProviderId) => `api-key-${provider}`;

export function focusKeyInput(provider: ProviderId) {
  const input = document.getElementById(keyInputId(provider));
  input?.scrollIntoView({ block: "center", behavior: "smooth" });
  input?.focus({ preventScroll: true });
}

function KeyInput({ provider, value, onChange }: { provider: ProviderId; value: string; onChange: (v: string) => void }) {
  const t = useT();
  const [visible, setVisible] = useState(false);
  const def = PROVIDERS[provider];
  return (
    <label className="field">
      <span className="row space">
        <span>{def.label}</span>
        {def.keyUrl && (
          <a href={def.keyUrl} target="_blank" rel="noopener noreferrer">
            {t("getFreeKey")} ↗
          </a>
        )}
      </span>
      <div className="row">
        <input
          id={keyInputId(provider)}
          className="mono"
          type={visible ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          placeholder={def.keyPlaceholder}
          value={value}
          onChange={(e) => onChange(e.target.value.trim())}
        />
        <button type="button" className="btn btn-small" onClick={() => setVisible((v) => !v)}>
          {visible ? t("hide") : t("show")}
        </button>
      </div>
    </label>
  );
}

function CountInput({ value, min, max, onChange }: { value: number; min: number; max?: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      min={min}
      max={max}
      value={value}
      onChange={(e) => {
        const v = Math.floor(Number(e.target.value) || 0);
        onChange(Math.max(min, max === undefined ? v : Math.min(max, v)));
      }}
    />
  );
}

function BatchSetup({
  settings,
  update,
  taskCount,
  disabled,
  onBatch,
}: {
  settings: Settings;
  update: Props["update"];
  taskCount: number | null;
  disabled: boolean;
  onBatch: () => void;
}) {
  const t = useT();
  const b = settings.batch;
  const last = Math.max(0, (taskCount ?? 1) - 1);
  const from = Math.min(b.from, last);
  const to = Math.min(Math.max(b.to, from), last);
  const episodes = (to - from + 1) * b.trials;
  const bothModels = settings.agent.controller === "model" && settings.user.controller === "model";
  const set = (patch: Partial<Settings["batch"]>) => update((s) => ({ ...s, batch: { ...s.batch, ...patch } }));
  return (
    <details className="batch-setup">
      <summary>{t("batchSection")}</summary>
      <p className="note">{t("batchHint", { k: b.trials })}</p>
      <div className="row three">
        <label className="field">
          <span>{t("batchFrom")}</span>
          <CountInput value={from} min={0} max={last} onChange={(v) => set({ from: v, to: Math.max(v, b.to) })} />
        </label>
        <label className="field">
          <span>{t("batchTo")}</span>
          <CountInput value={to} min={0} max={last} onChange={(v) => set({ to: v, from: Math.min(v, b.from) })} />
        </label>
        <label className="field">
          <span>{t("batchTrials")}</span>
          <CountInput value={b.trials} min={1} max={100} onChange={(v) => set({ trials: v })} />
        </label>
      </div>
      {!bothModels && <p className="note">{t("batchNeedsModels")}</p>}
      <button type="button" className="btn btn-block" disabled={disabled || !bothModels || !taskCount} onClick={onBatch}>
        {t("batchRun", { n: episodes })}
      </button>
    </details>
  );
}

export function SetupPanel({
  settings,
  update,
  taskCount,
  tasks,
  history,
  loading,
  loadError,
  episodeRunning,
  batchRunning,
  endpointFor,
  onStart,
  onBatch,
}: Props) {
  const t = useT();
  const [picking, setPicking] = useState(false);
  const splits = DOMAIN_SPLITS[settings.domain];
  const max = taskCount ?? 0;
  const setTask = (i: number) => update({ taskIndex: Math.max(0, Math.min(max - 1, i)) });
  // Human roles count too: their model writes drafts on request.
  const providersInUse = Array.from(new Set((["agent", "user"] as const).map((r) => settings[r].model.provider)));
  const anyModel = settings.agent.controller === "model" || settings.user.controller === "model";

  return (
    <div className="setup">
      <section className="panel-section">
        <h2>{t("setupTask")}</h2>
        <Segmented<DomainName>
          value={settings.domain}
          options={[
            { value: "retail", label: t("domainRetail") },
            { value: "airline", label: t("domainAirline") },
          ]}
          onChange={(domain) => update({ domain, split: "test", taskIndex: 0 })}
        />
        <div className="row task-row">
          <label className="field split-field">
            <span>{t("split")}</span>
            <select value={settings.split} onChange={(e) => update({ split: e.target.value, taskIndex: 0 })}>
              {splits.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="field grow">
            <span>
              {t("taskNumber")} <small className="muted">{taskCount !== null ? t("taskOf", { n: taskCount - 1 }) : ""}</small>
            </span>
            <div className="row task-index">
              <button type="button" className="btn btn-small" aria-label={t("prevTask")} onClick={() => setTask(settings.taskIndex - 1)}>
                ‹
              </button>
              <input
                type="number"
                min={0}
                max={Math.max(0, max - 1)}
                value={settings.taskIndex}
                onChange={(e) => setTask(Number(e.target.value) || 0)}
              />
              <button type="button" className="btn btn-small" aria-label={t("nextTask")} onClick={() => setTask(settings.taskIndex + 1)}>
                ›
              </button>
              <button
                type="button"
                className="btn btn-small"
                title={t("randomTask")}
                aria-label={t("randomTask")}
                onClick={() => setTask(Math.floor(Math.random() * max))}
              >
                🎲
              </button>
              <button
                type="button"
                className="btn btn-small"
                title={t("pickerTitle")}
                aria-label={t("pickerTitle")}
                disabled={!tasks}
                onClick={() => setPicking(true)}
              >
                ☰
              </button>
            </div>
          </label>
        </div>
        {picking && tasks && (
          <TaskPicker
            domain={settings.domain}
            split={settings.split}
            tasks={tasks}
            selected={settings.taskIndex}
            history={history}
            hideInstructions={settings.agent.controller === "human"}
            onPick={(i) => setTask(i)}
            onClose={() => setPicking(false)}
          />
        )}
        {loading && <small className="muted">{t("loading")}</small>}
        {loadError && <small className="error-text">{t("loadFailed", { error: loadError })}</small>}
      </section>

      <section className="panel-section">
        <h2>{t("setupRoles")}</h2>
        <div className="chips">
          {QUICK.map((q) => (
            <button
              key={q.key}
              type="button"
              className={`chip ${settings.agent.controller === q.agent && settings.user.controller === q.user ? "active" : ""}`}
              onClick={() =>
                update((s) => ({
                  ...s,
                  agent: { ...s.agent, controller: q.agent },
                  user: { ...s.user, controller: q.user },
                }))
              }
            >
              {t(q.key)}
            </button>
          ))}
        </div>
        <RoleCard role="agent" value={settings.agent} onChange={(agent) => update({ agent })} endpointFor={endpointFor} />
        <RoleCard role="user" value={settings.user} onChange={(user) => update({ user })} endpointFor={endpointFor} />
        {anyModel && <p className="note">{t("modelNote")}</p>}
      </section>

      {providersInUse.length > 0 && (
        <section className="panel-section">
          <h2>{t("apiKeys")}</h2>
          {providersInUse.map((p) => (
            <div key={p}>
              {p === "custom" && (
                <label className="field">
                  <span>{t("baseUrl")}</span>
                  <input
                    className="mono"
                    value={settings.customBaseUrl}
                    spellCheck={false}
                    onChange={(e) => update({ customBaseUrl: e.target.value.trim() })}
                  />
                </label>
              )}
              <KeyInput
                provider={p}
                value={settings.keys[p] ?? ""}
                onChange={(v) => update((s) => ({ ...s, keys: { ...s.keys, [p]: v } }))}
              />
            </div>
          ))}
          <p className="note">{t("apiKeysNote")}</p>
        </section>
      )}

      <section className="panel-section">
        <h2>{t("options")}</h2>
        <label className="field">
          <span>
            {t("maxSteps")} <small className="muted">{t("maxStepsHint")}</small>
          </span>
          <input
            type="number"
            min={0}
            value={settings.maxAgentSteps}
            onChange={(e) => update({ maxAgentSteps: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
          />
        </label>
        <label className="check">
          <input type="checkbox" checked={settings.stepMode} onChange={(e) => update({ stepMode: e.target.checked })} />
          <span>{t("stepMode")}</span>
        </label>
        <BatchSetup settings={settings} update={update} taskCount={taskCount} disabled={loading || batchRunning} onBatch={onBatch} />
      </section>

      <div className="start-bar">
        <button type="button" className="btn btn-primary btn-block" disabled={loading || !taskCount} onClick={onStart}>
          {episodeRunning ? t("newEpisode") : t("start")}
        </button>
      </div>
    </div>
  );
}
