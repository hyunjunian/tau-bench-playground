import { memo, type ReactNode } from "react";

const TOKEN = /("(?:[^"\\]|\\.)*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|NaN|-?Infinity)/g;

/** Pretty JSON with light syntax colouring (React nodes only, no HTML injection). */
export const JsonBlock = memo(function JsonBlock({ text, className }: { text: string; className?: string }) {
  const nodes: ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const m of text.matchAll(TOKEN)) {
    const start = m.index ?? 0;
    if (start > last) nodes.push(text.slice(last, start));
    if (m[1] !== undefined) {
      nodes.push(
        <span key={i++} className={m[2] ? "j-key" : "j-str"}>
          {m[1]}
        </span>,
      );
      if (m[2]) nodes.push(m[2]);
    } else if (m[3] !== undefined) {
      nodes.push(
        <span key={i++} className="j-lit">
          {m[3]}
        </span>,
      );
    } else {
      nodes.push(
        <span key={i++} className="j-num">
          {m[4]}
        </span>,
      );
    }
    last = start + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return <pre className={`json ${className ?? ""}`}>{nodes}</pre>;
});
