import { useEffect, useRef, useState } from "react";
import type { Problem } from "../types";
import { useLang } from "../i18n";
import { PanelBar } from "./PlotPanel";

interface Props {
  problem: Problem;
  console: string;
  error: string | null;
  running: boolean;
  className?: string;
}

export default function TerminalPanel({ problem, console, error, running, className = "" }: Props) {
  const { t } = useLang();
  const ref = useRef<HTMLPreElement>(null);
  const command = `python -m problems.${problem.category}.${problem.slug}.main`;
  // "Clear" hides what is shown now; the next run (new output) appears again.
  const [cleared, setCleared] = useState<{ console: string; error: string | null } | null>(null);
  const isCleared = !running && cleared !== null && cleared.console === console && cleared.error === error;
  useEffect(() => {
    if (running) setCleared(null);
  }, [running]);

  // Keep the newest output in view.
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [console, error, running]);

  return (
    <section data-ai-region="terminal" className={`flex min-w-0 flex-col bg-white ${className}`}>
      <PanelBar title={t("terminal")}>
        <button
          onClick={() => setCleared({ console, error })}
          disabled={running || isCleared || (!console && !error)}
          title={t("terminal.clear")}
          className="text-gray-500 hover:text-gray-800 disabled:opacity-40"
        >
          ⌫ {t("terminal.clear")}
        </button>
      </PanelBar>
      <pre ref={ref} className="min-h-0 flex-1 overflow-auto px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-gray-800">
        <span className="text-gray-400">$ </span>
        <span className="text-blue-700">{command}</span>
        {"\n"}
        {!isCleared && console}
        {running && <span className="animate-pulse text-gray-400">{console ? "\n" : ""}▍</span>}
        {error && !running && !isCleared && <span className="text-red-600">{`${console ? "\n" : ""}error: ${error}`}</span>}
      </pre>
    </section>
  );
}
