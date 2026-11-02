import { useState, type ReactNode } from "react";
import { useLang } from "../i18n";

// A small shell highlighter (bash and PowerShell): the command word, flags, quoted strings, URLs, $variables and
// the separators between commands. Enough for copy-paste install lines; not a full parser.
const TOKEN = /("[^"]*"|'[^']*'|https?:\/\/\S+|\$[A-Za-z_]\w*|&&|\|\||[|;]|\s+|[^\s"'|;&]+|&)/g;

function highlight(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  let expectCommand = true;
  for (const [i, tok] of (line.match(TOKEN) ?? []).entries()) {
    let cls = "";
    if (/^\s+$/.test(tok)) {
      out.push(tok);
      continue;
    }
    if (/^(&&|\|\||[|;])$/.test(tok)) {
      cls = "text-gray-400";
      expectCommand = true;
    } else if (/^["']/.test(tok)) cls = "text-amber-300";
    else if (/^https?:/.test(tok)) cls = "text-sky-300 underline decoration-sky-300/40";
    else if (/^\$/.test(tok)) cls = "text-pink-300";
    else if (expectCommand) {
      cls = "font-semibold text-emerald-300";
      expectCommand = false;
    } else if (/^--?[A-Za-z]/.test(tok)) cls = "text-violet-300";
    out.push(
      <span key={i} className={cls}>
        {tok}
      </span>,
    );
  }
  return out;
}

/** A dark copy-paste box for one or more shell commands: wrapped (no sideways scrolling) and highlighted. */
export default function ShellCommand({ text, size = "text-xs" }: { text: string; size?: string }) {
  const { t } = useLang();
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-start gap-2 rounded-md bg-gray-900 px-3 py-2">
      <code
        className={`min-w-0 flex-1 font-mono ${size} leading-relaxed break-all whitespace-pre-wrap text-gray-100`}
        data-testid="command"
      >
        {text.split("\n").map((line, i) => (
          <span key={i}>
            {i > 0 && "\n"}
            {highlight(line)}
          </span>
        ))}
      </code>
      <button
        onClick={() =>
          void navigator.clipboard?.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          })
        }
        className="shrink-0 text-xs text-gray-300 hover:text-white"
      >
        {copied ? t("be.copied") : t("be.copy")}
      </button>
    </div>
  );
}
