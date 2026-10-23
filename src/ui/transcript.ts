/** A readable Markdown version of an episode, for pasting into an issue or a chat. */
import type { Item } from "../tau/episode";
import { oneLine } from "./format";

export interface TranscriptLabels {
  title: string;
  agent: string;
  user: string;
  /** e.g. "Reward 1.0: The user ended the conversation." */
  outcome: string | null;
}

const RESULT_CHARS = 300;

/** Text for an inline code span (a backtick would end it early). */
const code = (s: string) => s.replaceAll("`", "'");

function speaker(label: string, item: Extract<Item, { kind: "user" | "agent" }>): string {
  return item.by === "human" ? `${label} (human)` : item.meta?.model ? `${label} (${item.meta.model})` : label;
}

export function transcriptMarkdown(items: Item[], labels: TranscriptLabels): string {
  const lines = [`## ${labels.title}`, ""];
  for (const item of items) {
    switch (item.kind) {
      case "greeting":
        lines.push(`**${labels.agent}:** ${item.text}`, "");
        break;
      case "user":
        lines.push(`**${speaker(labels.user, item)}:** ${item.text}`, "");
        break;
      case "agent":
        lines.push(`**${speaker(labels.agent, item)}:** ${item.text}`, "");
        break;
      case "tool": {
        const result = item.observation === "" ? '""' : oneLine(item.observation, RESULT_CHARS);
        lines.push(`- \`${item.name}(${code(oneLine(item.args, RESULT_CHARS))})\` → \`${code(result)}\``, "");
        break;
      }
      case "notice":
        lines.push(`> ${item.text}`, "");
        break;
    }
  }
  if (labels.outcome) lines.push(`**${labels.outcome}**`);
  return lines.join("\n").trimEnd() + "\n";
}
