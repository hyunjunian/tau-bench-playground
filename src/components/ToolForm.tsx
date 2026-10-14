import { useEffect, useMemo, useState } from "react";
import { useT } from "../i18n";
import type { JsonSchema, ToolInfo } from "../tau/env";
import { parseToolArguments } from "../tau/episode";
import { PyDict, PyFloat, type PyValue, jsonPretty } from "../tau/py";
import { isReadTool } from "../ui/format";

type Row = Record<string, string>;
type Values = Record<string, string | Row[]>;

function isObjectArray(s: JsonSchema): boolean {
  return s.type === "array" && s.items?.type === "object";
}

function emptyValues(info: ToolInfo): Values {
  const out: Values = {};
  for (const [name, schema] of Object.entries(info.function.parameters.properties ?? {})) {
    out[name] = isObjectArray(schema) ? [{}] : "";
  }
  return out;
}

/** Number fields keep Python's int/float distinction: "250" -> int, "250.0" -> float. */
function parseNumber(raw: string): PyValue | undefined {
  const s = raw.trim();
  if (s === "") return undefined;
  if (/^-?(0|[1-9]\d*)$/.test(s)) return Number(s);
  if (/^-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?$/.test(s)) return new PyFloat(Number(s));
  return s;
}

function fieldValue(schema: JsonSchema, raw: string): PyValue | undefined {
  if (schema.type === "number" || schema.type === "integer") return parseNumber(raw);
  if (schema.type === "array") return raw.split(/[\s,]+/).filter(Boolean);
  return raw;
}

function buildKwargs(info: ToolInfo, values: Values): PyDict {
  const kwargs = new PyDict();
  for (const [name, schema] of Object.entries(info.function.parameters.properties ?? {})) {
    const v = values[name];
    if (Array.isArray(v)) {
      const itemProps = schema.items?.properties ?? {};
      const rows = v
        .filter((row) => Object.values(row).some((x) => x.trim() !== ""))
        .map((row) => {
          const d = new PyDict();
          for (const [k, s] of Object.entries(itemProps)) {
            const fv = fieldValue(s, row[k] ?? "");
            if (fv !== undefined) d.set(k, fv);
          }
          return d;
        });
      kwargs.set(name, rows);
    } else {
      const fv = fieldValue(schema, v ?? "");
      if (fv !== undefined) kwargs.set(name, fv);
    }
  }
  return kwargs;
}

function Input({ schema, value, onChange }: { schema: JsonSchema; value: string; onChange: (v: string) => void }) {
  if (schema.enum) {
    return (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="" />
        {schema.enum.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  }
  return (
    <input
      className="mono"
      value={value}
      spellCheck={false}
      inputMode={schema.type === "number" || schema.type === "integer" ? "decimal" : undefined}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

export function ToolForm({
  tools,
  preferred,
  onRun,
}: {
  tools: ToolInfo[];
  preferred: string | null;
  onRun: (name: string, kwargs: PyDict) => void;
}) {
  const t = useT();
  const [name, setName] = useState(preferred ?? tools[0]?.function.name ?? "");
  const info = useMemo(() => tools.find((x) => x.function.name === name) ?? tools[0], [tools, name]);
  const [values, setValues] = useState<Values>(() => emptyValues(info));
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState("{}");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (preferred) setName(preferred);
  }, [preferred]);

  useEffect(() => {
    setValues(emptyValues(info));
    setJsonText("{}");
    setError(null);
  }, [info]);

  const props = Object.entries(info.function.parameters.properties ?? {});
  const required = new Set(info.function.parameters.required ?? []);

  const switchMode = () => {
    if (!jsonMode) setJsonText(jsonPretty(buildKwargs(info, values)));
    setJsonMode((m) => !m);
    setError(null);
  };

  const submit = () => {
    let kwargs: PyDict;
    if (jsonMode) {
      try {
        kwargs = parseToolArguments(jsonText);
      } catch (e) {
        setError(t("invalidJson", { error: e instanceof Error ? e.message : String(e) }));
        return;
      }
    } else {
      kwargs = buildKwargs(info, values);
    }
    setError(null);
    onRun(info.function.name, kwargs);
  };

  return (
    <div className="tool-form">
      <div className="row">
        <label className="field grow">
          <span>{t("chooseTool")}</span>
          <select value={info.function.name} onChange={(e) => setName(e.target.value)}>
            {tools.map((x) => (
              <option key={x.function.name} value={x.function.name}>
                {x.function.name} ({isReadTool(x.function.name) ? t("readTool") : t("writeTool")})
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn btn-small align-end" onClick={switchMode}>
          {jsonMode ? t("formMode") : t("jsonMode")}
        </button>
      </div>
      <p className="tool-desc">{info.function.description}</p>

      {jsonMode ? (
        <label className="field">
          <span>{t("argsJson")}</span>
          <textarea className="mono" rows={6} spellCheck={false} value={jsonText} onChange={(e) => setJsonText(e.target.value)} />
        </label>
      ) : (
        <div className="tool-fields">
          {props.map(([pname, schema]) => {
            const v = values[pname];
            if (isObjectArray(schema)) {
              const rows = (Array.isArray(v) ? v : [{}]) as Row[];
              const itemProps = Object.entries(schema.items?.properties ?? {});
              return (
                <fieldset key={pname} className="field">
                  <legend>
                    <code>{pname}</code>
                    {required.has(pname) && <span className="req">*</span>} <small className="muted">{schema.description}</small>
                  </legend>
                  {rows.map((row, i) => (
                    <div key={i} className="obj-row">
                      {itemProps.map(([k, s]) => (
                        <label key={k} className="field">
                          <span>
                            <code>{k}</code>
                          </span>
                          <Input
                            schema={s}
                            value={row[k] ?? ""}
                            onChange={(nv) =>
                              setValues((old) => {
                                const list = [...((old[pname] as Row[]) ?? [])];
                                list[i] = { ...list[i], [k]: nv };
                                return { ...old, [pname]: list };
                              })
                            }
                          />
                        </label>
                      ))}
                      <button
                        type="button"
                        className="btn btn-small btn-ghost"
                        aria-label={t("removeRow")}
                        onClick={() =>
                          setValues((old) => ({ ...old, [pname]: (old[pname] as Row[]).filter((_, j) => j !== i) }))
                        }
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    className="btn btn-small"
                    onClick={() => setValues((old) => ({ ...old, [pname]: [...((old[pname] as Row[]) ?? []), {}] }))}
                  >
                    + {t("addRow")}
                  </button>
                </fieldset>
              );
            }
            return (
              <label key={pname} className="field">
                <span>
                  <code>{pname}</code>
                  {required.has(pname) && <span className="req">*</span>}{" "}
                  <small className="muted">
                    {schema.type === "array" ? `${t("listHint")} · ` : ""}
                    {schema.description}
                  </small>
                </span>
                <Input schema={schema} value={(v as string) ?? ""} onChange={(nv) => setValues((old) => ({ ...old, [pname]: nv }))} />
              </label>
            );
          })}
        </div>
      )}
      {error && <div className="error-text">{error}</div>}
      <div className="row end">
        <button type="button" className="btn btn-primary" onClick={submit}>
          {t("runTool")}
        </button>
      </div>
    </div>
  );
}
