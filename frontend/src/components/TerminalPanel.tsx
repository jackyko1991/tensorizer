import { useEffect, useRef } from "react";
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

  // Keep the newest output in view.
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [console, error, running]);

  return (
    <section className={`flex min-w-0 flex-col bg-white ${className}`}>
      <PanelBar title={t("terminal")} />
      <pre ref={ref} className="min-h-0 flex-1 overflow-auto px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-gray-800">
        <span className="text-gray-400">$ </span>
        <span className="text-blue-700">{command}</span>
        {"\n"}
        {console}
        {running && <span className="animate-pulse text-gray-400">{console ? "\n" : ""}▍</span>}
        {error && !running && <span className="text-red-600">{`${console ? "\n" : ""}error: ${error}`}</span>}
      </pre>
    </section>
  );
}
