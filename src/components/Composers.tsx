import { useEffect, useRef, useState } from "react";
import { useT } from "../i18n";
import { MissingKeyError } from "../llm/client";
import type { ProviderId } from "../llm/providers";
import { STOP_TOKEN, type ToolInfo } from "../tau/env";
import type { Draft } from "../tau/episode";
import type { PyDict } from "../tau/py";
import type { Suggestions } from "../ui/suggest";
import { focusKeyInput } from "./SetupPanel";
import { type PreferredTool, type ToolDraft, ToolForm } from "./ToolForm";

/** Asks the human's own role model for a draft; cancelling aborts the request. */
function DraftButton({
  model,
  version,
  onDraft,
  onResult,
}: {
  model: string;
  /** Changes whenever the conversation does: a draft still on its way would no longer fit, so it is dropped. */
  version: number;
  onDraft: (signal: AbortSignal) => Promise<Draft>;
  onResult: (draft: Draft) => void;
}) {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ message: string; missingKey?: ProviderId } | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), [version]);
  const ask = async () => {
    const c = new AbortController();
    controller.current = c;
    setBusy(true);
    setError(null);
    try {
      onResult(await onDraft(c.signal));
    } catch (e) {
      if (!c.signal.aborted) {
        const message = t("draftFailed", { error: e instanceof Error ? e.message : String(e) });
        setError({ message, missingKey: e instanceof MissingKeyError ? e.provider : undefined });
      }
    } finally {
      if (controller.current === c) controller.current = null;
      setBusy(false);
    }
  };
  return (
    <span className="draft">
      {busy ? (
        <>
          <span className="spinner" aria-hidden />
          <span className="muted">{t("drafting")}</span>
          <button type="button" className="btn btn-small btn-ghost" onClick={() => controller.current?.abort()}>
            {t("stop")}
          </button>
        </>
      ) : (
        <button type="button" className="btn btn-small" title={t("draftHint", { model })} onClick={() => void ask()}>
          ✦ {t("draftAsk")}
        </button>
      )}
      {error && <span className="error-text">{error.message}</span>}
      {error?.missingKey && (
        <button type="button" className="btn btn-small" onClick={() => focusKeyInput(error.missingKey!)}>
          {t("enterKey")}
        </button>
      )}
    </span>
  );
}

function MessageBox({
  placeholder,
  onSend,
  extra,
  draft,
}: {
  placeholder: string;
  onSend: (text: string) => void;
  extra?: React.ReactNode;
  /** Text to load into the box (a new object per draft). */
  draft?: { text: string } | null;
}) {
  const t = useT();
  const [text, setText] = useState("");
  useEffect(() => {
    if (draft) setText(draft.text);
  }, [draft]);
  const send = () => {
    if (!text.trim()) return;
    onSend(text);
    setText("");
  };
  return (
    <div className="message-box">
      <textarea
        rows={3}
        value={text}
        placeholder={placeholder}
        autoFocus
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            send();
          }
        }}
      />
      <div className="row end">
        {extra}
        <button type="button" className="btn btn-primary" disabled={!text.trim()} onClick={send}>
          {t("send")}
        </button>
      </div>
    </div>
  );
}

export function AgentComposer({
  tools,
  preferredTool,
  suggestions,
  draftModel,
  version,
  onDraft,
  onReply,
  onTool,
}: {
  tools: ToolInfo[];
  preferredTool: PreferredTool | null;
  suggestions: Suggestions;
  draftModel: string;
  /** Conversation length, see DraftButton. */
  version: number;
  onDraft: (signal: AbortSignal) => Promise<Draft>;
  onReply: (text: string) => void;
  onTool: (name: string, kwargs: PyDict) => void;
}) {
  const t = useT();
  const [tab, setTab] = useState<"reply" | "tool">(preferredTool ? "tool" : "reply");
  const [replyDraft, setReplyDraft] = useState<{ text: string } | null>(null);
  const [toolDraft, setToolDraft] = useState<ToolDraft | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const useDraft = (d: Draft) => {
    if (d.tool) {
      setToolDraft({ ...d.tool });
      setNote(d.text.trim() || null);
      setTab("tool");
    } else {
      setReplyDraft({ text: d.text });
      setNote(null);
      setTab("reply");
    }
  };
  const [prevPreferred, setPrevPreferred] = useState(preferredTool);
  if (preferredTool !== prevPreferred) {
    setPrevPreferred(preferredTool);
    if (preferredTool) {
      // A newer pick replaces a draft that has not been used yet.
      setToolDraft(null);
      setNote(null);
      setTab("tool");
    }
  }
  return (
    <div className="composer composer-agent">
      <div className="composer-title">{t("waitingHumanAgent")}</div>
      <div className="composer-tabs">
        <div className="tabs small">
          <button type="button" className={tab === "reply" ? "active" : ""} onClick={() => setTab("reply")}>
            {t("replyTab")}
          </button>
          <button type="button" className={tab === "tool" ? "active" : ""} onClick={() => setTab("tool")}>
            {t("toolTab")}
          </button>
        </div>
        <DraftButton model={draftModel} version={version} onDraft={onDraft} onResult={useDraft} />
      </div>
      {note && (
        <p className="draft-note">
          <small>{t("draftNote")}</small> {note}
        </p>
      )}
      {tab === "reply" ? (
        <MessageBox
          placeholder={t("replyPlaceholder")}
          draft={replyDraft}
          onSend={(text) => {
            setReplyDraft(null); // used up: switching tabs must not bring it back
            onReply(text);
          }}
        />
      ) : (
        <ToolForm
          tools={tools}
          preferred={preferredTool}
          draft={toolDraft}
          suggestions={suggestions}
          onRun={(name, kwargs) => {
            setToolDraft(null);
            setNote(null);
            onTool(name, kwargs);
          }}
        />
      )}
    </div>
  );
}

export function UserComposer({
  instruction,
  started,
  draftModel,
  version,
  onDraft,
  onSend,
}: {
  instruction: string;
  started: boolean;
  draftModel: string;
  /** Conversation length, see DraftButton. */
  version: number;
  onDraft: (signal: AbortSignal) => Promise<Draft>;
  onSend: (text: string) => void;
}) {
  const t = useT();
  const [draft, setDraft] = useState<{ text: string } | null>(null);
  return (
    <div className="composer composer-user">
      <div className="row space composer-head">
        <div className="composer-title">{t("waitingHumanUser")}</div>
        <DraftButton model={draftModel} version={version} onDraft={onDraft} onResult={(d) => setDraft({ text: d.text })} />
      </div>
      <details className="instruction-card" open>
        <summary>{t("userInstructionTitle")}</summary>
        <p>{instruction}</p>
        <small className="muted">{t("userRules")}</small>
      </details>
      <MessageBox
        placeholder={t("userPlaceholder")}
        onSend={onSend}
        draft={draft}
        extra={
          started ? (
            <button type="button" className="btn btn-danger-outline" onClick={() => onSend(STOP_TOKEN)}>
              {t("endConversation")}
            </button>
          ) : null
        }
      />
    </div>
  );
}
