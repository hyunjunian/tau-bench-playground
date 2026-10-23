import { memo, useEffect, useRef, useState } from "react";
import { type TKey, useT } from "../i18n";
import { STOP_TOKEN } from "../tau/env";
import type { Controller, GenMeta, Item } from "../tau/episode";
import { formatBytes, oneLine, prettyArgs, prettyObservation } from "../ui/format";
import { JsonBlock } from "./JsonBlock";
import { Markdown } from "./Markdown";

function Who({ by, meta }: { by: Controller; meta?: GenMeta }) {
  const t = useT();
  if (by === "human") return <span className="who human">{t("you")}</span>;
  return (
    <span className="who model" title={meta?.latencyMs !== undefined ? t("latency", { ms: meta.latencyMs }) : undefined}>
      {meta?.model ?? t("model")}
    </span>
  );
}

function Reasoning({ text }: { text?: string }) {
  const t = useT();
  if (!text) return null;
  return (
    <details className="reasoning">
      <summary>{t("thoughts")}</summary>
      <pre>{text}</pre>
    </details>
  );
}

const ToolCard = memo(function ToolCard({ item }: { item: Extract<Item, { kind: "tool" }> }) {
  const t = useT();
  const short = item.observation.length <= 160 && !item.observation.includes("\n");
  // A human agent needs to read what their own tool call returned.
  const [open, setOpen] = useState(item.error || item.by === "human");
  const obs = prettyObservation(item.observation);
  return (
    <div className={`tool-card ${item.error ? "is-error" : ""}`}>
      <div className="tool-head">
        <span className="tool-icon" aria-hidden>
          ⚙
        </span>
        <code className="tool-name">{item.name}</code>
        <span className="spacer" />
        <span className="meta">
          {t("step", { n: item.step })} · <Who by={item.by} meta={item.meta} />
        </span>
      </div>
      {item.args.trim() !== "{}" && item.args.trim() !== "" && (
        <code className="tool-args" title={prettyArgs(item.args)}>
          {oneLine(item.args, 400)}
        </code>
      )}
      {item.note && (
        <div className="tool-note">
          <small>{t("notSentToUser")}</small> {item.note}
        </div>
      )}
      <Reasoning text={item.meta?.reasoning} />
      {short ? (
        <div className="tool-obs-inline">
          <span className="muted">→</span> <code>{item.observation === "" ? '""' : item.observation}</code>
        </div>
      ) : (
        <details className="tool-obs" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
          <summary>
            {t("observation")} <span className="muted">({formatBytes(item.observation.length)})</span>{" "}
            {!open && <code className="muted">{oneLine(item.observation, 120)}</code>}
          </summary>
          {open && (obs.isJson ? <JsonBlock text={obs.text} focusable /> : <pre tabIndex={0}>{obs.text}</pre>)}
        </details>
      )}
    </div>
  );
});

const ItemView = memo(function ItemView({ item }: { item: Item }) {
  const t = useT();
  switch (item.kind) {
    case "greeting":
      return (
        <div className="msg msg-agent greeting">
          <div className="msg-head">
            <span className="role-name">{t("agentLabel")}</span>
            <span className="meta">{t("greetingLabel")}</span>
          </div>
          <div className="bubble">{item.text}</div>
        </div>
      );
    case "user": {
      const stop = item.text.includes(STOP_TOKEN);
      return (
        <div className="msg msg-user">
          <div className="msg-head">
            <span className="role-name">{t("userLabel")}</span>
            <Who by={item.by} meta={item.meta} />
            {stop && <span className="badge badge-stop">{t("stopBadge")}</span>}
          </div>
          <Reasoning text={item.meta?.reasoning} />
          <div className="bubble">
            <Markdown text={item.text} />
          </div>
        </div>
      );
    }
    case "agent":
      return (
        <div className="msg msg-agent">
          <div className="msg-head">
            <span className="role-name">{t("agentLabel")}</span>
            <span className="meta">
              {t("step", { n: item.step })} · <Who by={item.by} meta={item.meta} />
            </span>
          </div>
          <Reasoning text={item.meta?.reasoning} />
          <div className="bubble">
            <Markdown text={item.text} />
          </div>
        </div>
      );
    case "tool":
      return <ToolCard item={item} />;
    case "notice":
      return <div className={`notice notice-${item.level}`}>{item.code ? t(item.code as TKey, item.vars) : item.text}</div>;
  }
});

export function Transcript({ items }: { items: Item[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [items.length]);
  return (
    <div
      className="transcript"
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
    >
      {items.map((item) => (
        <ItemView key={item.id} item={item} />
      ))}
    </div>
  );
}
