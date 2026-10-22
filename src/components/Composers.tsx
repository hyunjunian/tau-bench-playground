import { useState } from "react";
import { useT } from "../i18n";
import { STOP_TOKEN, type ToolInfo } from "../tau/env";
import type { PyDict } from "../tau/py";
import type { Suggestions } from "../ui/suggest";
import { type PreferredTool, ToolForm } from "./ToolForm";

function MessageBox({ placeholder, onSend, extra }: { placeholder: string; onSend: (text: string) => void; extra?: React.ReactNode }) {
  const t = useT();
  const [text, setText] = useState("");
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
  onReply,
  onTool,
}: {
  tools: ToolInfo[];
  preferredTool: PreferredTool | null;
  suggestions: Suggestions;
  onReply: (text: string) => void;
  onTool: (name: string, kwargs: PyDict) => void;
}) {
  const t = useT();
  const [tab, setTab] = useState<"reply" | "tool">(preferredTool ? "tool" : "reply");
  const [prevPreferred, setPrevPreferred] = useState(preferredTool);
  if (preferredTool !== prevPreferred) {
    setPrevPreferred(preferredTool);
    if (preferredTool) setTab("tool");
  }
  return (
    <div className="composer composer-agent">
      <div className="composer-title">{t("waitingHumanAgent")}</div>
      <div className="tabs small">
        <button type="button" className={tab === "reply" ? "active" : ""} onClick={() => setTab("reply")}>
          {t("replyTab")}
        </button>
        <button type="button" className={tab === "tool" ? "active" : ""} onClick={() => setTab("tool")}>
          {t("toolTab")}
        </button>
      </div>
      {tab === "reply" ? (
        <MessageBox placeholder={t("replyPlaceholder")} onSend={onReply} />
      ) : (
        <ToolForm tools={tools} preferred={preferredTool} suggestions={suggestions} onRun={onTool} />
      )}
    </div>
  );
}

export function UserComposer({
  instruction,
  started,
  onSend,
}: {
  instruction: string;
  started: boolean;
  onSend: (text: string) => void;
}) {
  const t = useT();
  return (
    <div className="composer composer-user">
      <div className="composer-title">{t("waitingHumanUser")}</div>
      <details className="instruction-card" open>
        <summary>{t("userInstructionTitle")}</summary>
        <p>{instruction}</p>
        <small className="muted">{t("userRules")}</small>
      </details>
      <MessageBox
        placeholder={t("userPlaceholder")}
        onSend={onSend}
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
