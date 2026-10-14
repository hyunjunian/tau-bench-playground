import { useEffect, useState } from "react";
import { useT } from "../i18n";
import {
  DEFAULT_MODEL,
  type Endpoint,
  type ModelConfig,
  type ModelPreset,
  PROVIDERS,
  PROVIDER_ORDER,
  type ProviderId,
  type ReasoningEffort,
  fetchModelIds,
  fetchOpenRouterFreeModels,
  findPreset,
} from "../llm/providers";

const EFFORTS: ReasoningEffort[] = ["", "none", "minimal", "low", "medium", "high"];
const OTHER = "__other__";

let openRouterFree: Promise<ModelPreset[]> | null = null;

function useOpenRouterFree(enabled: boolean): ModelPreset[] | null {
  const [models, setModels] = useState<ModelPreset[] | null>(null);
  useEffect(() => {
    if (!enabled) return;
    openRouterFree ??= fetchOpenRouterFreeModels().catch(() => {
      openRouterFree = null;
      return PROVIDERS.openrouter.models;
    });
    let alive = true;
    void openRouterFree.then((m) => alive && setModels(m));
    return () => {
      alive = false;
    };
  }, [enabled]);
  return models;
}

export function ModelBadge({ provider, model }: { provider: ProviderId; model: string }) {
  const t = useT();
  const preset = findPreset(provider, model);
  return (
    <span className="badges">
      {preset?.tau2Airline !== undefined && <span className="badge">{t("tauScore", { score: preset.tau2Airline })}</span>}
      {provider === DEFAULT_MODEL.provider && model === DEFAULT_MODEL.model && (
        <span className="badge badge-accent">{t("bestFree")}</span>
      )}
    </span>
  );
}

export function ModelPicker({
  value,
  onChange,
  endpointFor,
}: {
  value: ModelConfig;
  onChange: (m: ModelConfig) => void;
  endpointFor: (p: ProviderId) => Endpoint;
}) {
  const t = useT();
  const provider = PROVIDERS[value.provider];
  const liveFree = useOpenRouterFree(value.provider === "openrouter");
  const [loaded, setLoaded] = useState<{ provider: ProviderId; ids: string[] } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [forceOther, setForceOther] = useState(false);

  const presets: ModelPreset[] =
    value.provider === "openrouter" && liveFree ? liveFree : provider.models;
  const extra = loaded?.provider === value.provider ? loaded.ids.filter((id) => !presets.some((p) => p.id === id)) : [];
  const known = presets.some((p) => p.id === value.model) || extra.includes(value.model);
  const showOther = forceOther || !known || presets.length + extra.length === 0;

  const loadList = async () => {
    setLoadError(null);
    try {
      const ids = await fetchModelIds(endpointFor(value.provider));
      setLoaded({ provider: value.provider, ids });
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="model-picker">
      <label className="field">
        <span>{t("provider")}</span>
        <select
          value={value.provider}
          onChange={(e) => {
            const p = e.target.value as ProviderId;
            setForceOther(false);
            onChange({ ...value, provider: p, model: PROVIDERS[p].models[0]?.id ?? "" });
          }}
        >
          {PROVIDER_ORDER.map((id) => (
            <option key={id} value={id}>
              {PROVIDERS[id].label}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span className="row space">
          <span>{t("modelId")}</span>
          <button type="button" className="link-btn" onClick={loadList}>
            {t("loadModels")}
          </button>
        </span>
        {presets.length + extra.length > 0 && (
            <select
              value={showOther ? OTHER : value.model}
              onChange={(e) => {
                if (e.target.value === OTHER) {
                  setForceOther(true);
                  return;
                }
                setForceOther(false);
                onChange({ ...value, model: e.target.value });
              }}
            >
              {presets.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                  {m.tau2Airline !== undefined ? ` · τ² ${m.tau2Airline}%` : ""}
                </option>
              ))}
              {extra.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
              <option value={OTHER}>{t("customModel")}</option>
            </select>
        )}
        {showOther && (
          <input
            className="mono"
            value={value.model}
            placeholder="model-id"
            spellCheck={false}
            onChange={(e) => onChange({ ...value, model: e.target.value.trim() })}
          />
        )}
        {loadError && <small className="error-text">{loadError}</small>}
      </label>

      <ModelBadge provider={value.provider} model={value.model} />

      <div className="row two">
        <label className="field">
          <span>{t("temperature")}</span>
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            placeholder={t("temperatureDefault")}
            value={value.temperature ?? ""}
            onChange={(e) =>
              onChange({ ...value, temperature: e.target.value === "" ? null : Number(e.target.value) })
            }
          />
        </label>
        <label className="field">
          <span>{t("reasoning")}</span>
          <select
            value={value.reasoningEffort}
            onChange={(e) => onChange({ ...value, reasoningEffort: e.target.value as ReasoningEffort })}
          >
            {EFFORTS.map((e) => (
              <option key={e} value={e}>
                {e || t("reasoningDefault")}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
