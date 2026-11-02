import { useEffect, useRef, useState } from "react";
import type { TestTally } from "../api";
import type { Problem } from "../types";
import { useLang } from "../i18n";
import { PanelBar } from "./PlotPanel";
import QueueModal from "./QueueModal";

interface Props {
  problem: Problem;
  console: string;
  error: string | null;
  running: boolean;
  /** test.py's verdict on the last run (null: not run here yet, or the problem has no tests). */
  tests?: TestTally | null;
  className?: string;
}

// Colour by what a line says: test results, errors, warnings, section rules, traces and numbers.
const RULES: [RegExp, string][] = [
  [/\bFAILED\b|\bERROR\b|\bfailed\b.* =+$|^E\s|Traceback|Error:|^\s*\^+\s*$/, "text-red-600"],
  [/\bPASSED\b|^=+ .*\bpassed\b.* =+$/, "text-emerald-600"],
  [/warning|Warning|\bskipped\b|⚠/, "text-amber-600"],
  [/^(──|==|--)/, "font-semibold text-indigo-600"],
  [/^\s*→ /, "text-sky-700"],
  [/^\s*← /, "text-gray-400"],
];

function lineClass(line: string): string {
  for (const [re, cls] of RULES) if (re.test(line)) return cls;
  return "";
}

/** One console line, with numbers (and units after "=") picked out. */
function Line({ text }: { text: string }) {
  const cls = lineClass(text);
  if (cls) return <span className={cls}>{text}</span>;
  const parts = text.split(/(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)/i);
  return (
    <span>
      {parts.map((p, i) =>
        i % 2 ? (
          <span key={i} className="text-violet-700">
            {p}
          </span>
        ) : (
          p
        ),
      )}
    </span>
  );
}

export default function TerminalPanel({ problem, console, error, running, tests, className = "" }: Props) {
  const { t } = useLang();
  const ref = useRef<HTMLPreElement>(null);
  const command = `python -m problems.${problem.category}.${problem.slug}.main`;
  // "Clear" hides what is shown now; the next run (new output) appears again.
  const [cleared, setCleared] = useState<{ console: string; error: string | null } | null>(null);
  const [queueOpen, setQueueOpen] = useState(false);
  const isCleared = !running && cleared !== null && cleared.console === console && cleared.error === error;
  useEffect(() => {
    if (running) setCleared(null);
  }, [running]);

  // Keep the newest output in view.
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [console, error, running]);

  const lines = isCleared || !console ? [] : console.split("\n");
  return (
    <section data-ai-region="terminal" className={`flex min-w-0 flex-col bg-white ${className}`}>
      <PanelBar title={t("terminal")}>
        {tests && !running && (
          <span
            data-testid="tdd-light"
            title={t("tdd.hint")}
            className={`flex items-center gap-1.5 rounded-full px-2 py-0.5 font-medium ${
              tests.ok ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${tests.ok ? "bg-emerald-500" : "bg-red-500"}`} />
            TDD {tests.passed}/{tests.passed + tests.failed} {t("tdd.pass")}
            {tests.failed > 0 && ` · ${tests.failed} ${t("tdd.fail")}`}
          </span>
        )}
        <button
          onClick={() => setQueueOpen(true)}
          title={t("q.title")}
          data-testid="queue-button"
          className="text-gray-500 hover:text-gray-800"
        >
          ☰ {t("q.button")}
        </button>
        <button
          onClick={() => setCleared({ console, error })}
          disabled={running || isCleared || (!console && !error)}
          title={t("terminal.clear")}
          className="text-gray-500 hover:text-gray-800 disabled:opacity-40"
        >
          ⌫ {t("terminal.clear")}
        </button>
      </PanelBar>
      <pre
        ref={ref}
        className="min-h-0 flex-1 overflow-auto px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-gray-800"
      >
        <span className="text-gray-400">$ </span>
        <span className="text-blue-700">{command}</span>
        {lines.map((line, i) => (
          <span key={i}>
            {"\n"}
            <Line text={line} />
          </span>
        ))}
        {running && <span className="animate-pulse text-gray-400">{"\n"}▍</span>}
        {error && !running && !isCleared && <span className="text-red-600">{`\nerror: ${error}`}</span>}
      </pre>
      {queueOpen && <QueueModal onClose={() => setQueueOpen(false)} />}
    </section>
  );
}
