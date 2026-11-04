import { python } from "@codemirror/lang-python";
import CodeMirror, { type ViewUpdate } from "@uiw/react-codemirror";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { askAI, emitSelection, type CodeSelection } from "../agentBus";
import { useLang, type Key } from "../i18n";
import type { CodeFile, Dataset } from "../types";
import DataView from "./DataView";

const ROLE: Record<string, Key> = {
  "test.py": "role.test",
  "main.py": "role.main",
  "equation.py": "role.equation",
  "tensors.py": "role.tensors",
  "solve.py": "role.solve",
  "plot.py": "role.plot",
};

interface Props {
  /** Problem id, so a code selection can be sent to the Ask AI panel. */
  problem: string;
  files: CodeFile[];
  /** Library problems: bundled data files, shown as read-only tabs after the code. */
  datasets?: Dataset[];
  edits: Record<string, string>;
  editable: boolean;
  onEdit: (name: string, code: string) => void;
  onReset: () => void;
  /** Library problems: right-click offers "Fork to Workspace" (their code is read-only). */
  onFork?: () => void;
  /** Rendered to the right of the editor (the parameters panel). */
  side?: ReactNode;
  /** Rendered at the right of the info row (notebook export). */
  toolbar?: ReactNode;
  className?: string;
}

export default function CodePanel({
  problem,
  files,
  datasets = [],
  edits,
  editable,
  onEdit,
  onReset,
  onFork,
  side,
  toolbar,
  className = "",
}: Props) {
  const { t } = useLang();
  const [active, setActive] = useState(files[0]?.name);
  const dataset = datasets.find((d) => d.name === active);
  const file = files.find((f) => f.name === active) ?? files[0];
  const edited = Object.keys(edits).length > 0;
  const selection = useRef<CodeSelection | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);

  // Tell the Ask AI panel what is highlighted (nothing, when the selection is empty).
  function onUpdate(update: ViewUpdate) {
    if (!update.selectionSet && !update.docChanged) return;
    const { from, to } = update.state.selection.main;
    const doc = update.state.doc;
    const next =
      from === to
        ? null
        : {
            problem,
            file: file.name,
            fromLine: doc.lineAt(from).number,
            toLine: doc.lineAt(to).number,
            text: doc.sliceString(from, to),
          };
    if (JSON.stringify(next) !== JSON.stringify(selection.current)) {
      selection.current = next;
      emitSelection(next);
    }
  }
  useEffect(() => () => emitSelection(null), [problem, file.name]);

  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    window.addEventListener("click", close);
    window.addEventListener("keydown", close);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", close);
      window.removeEventListener("scroll", close, true);
    };
  }, [menu]);

  return (
    <section className={`flex flex-col overflow-hidden bg-white xl:flex-row ${className}`}>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex shrink-0 items-end gap-1 overflow-x-auto border-b border-gray-200 bg-gray-50 px-2 pt-2">
          {files.map((f, i) => (
            <button
              key={f.name}
              onClick={() => setActive(f.name)}
              title={t(ROLE[f.name])}
              className={`shrink-0 rounded-t-md border border-b-0 px-3 py-1.5 font-mono text-xs ${
                f.name === file.name
                  ? "border-blue-200 bg-blue-50 font-medium text-blue-700"
                  : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              {/* test.py is step 0: written first (TDD) */}
              <span className="mr-1 text-gray-400">{files[0]?.name === "test.py" ? i : i + 1}.</span>
              {f.name}
              {edits[f.name] !== undefined && <span className="ml-1 text-amber-600">●</span>}
            </button>
          ))}
          {datasets.map((d) => (
            <button
              key={d.name}
              onClick={() => setActive(d.name)}
              title={t("dataset.role")}
              data-testid="data-tab"
              className={`shrink-0 rounded-t-md border border-b-0 px-3 py-1.5 font-mono text-xs ${
                d.name === dataset?.name
                  ? "border-emerald-200 bg-emerald-50 font-medium text-emerald-700"
                  : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              📊 {d.name.replace(/^data\//, "")}
            </button>
          ))}
        </div>

        <div className="flex h-7 shrink-0 items-center justify-between gap-3 border-b border-gray-100 px-3 text-xs text-gray-500">
          <span className="truncate">
            {dataset ? t("dataset.role") : t(ROLE[file.name])} (
            {t(editable && !dataset ? "code.editable" : "code.readonly")})
            {edited && (
              <button onClick={onReset} className="ml-3 text-blue-600 hover:underline">
                {t("code.reset")}
              </button>
            )}
          </span>
          {toolbar && <span className="flex shrink-0 items-center gap-3">{toolbar}</span>}
        </div>

        <div
          className="max-h-[60vh] min-h-0 flex-1 overflow-auto xl:max-h-none"
          onContextMenu={(e) => {
            // Right-click: Explain / Fix with AI on highlighted code, and Fork for read-only library code.
            if (!selection.current && !onFork) return;
            e.preventDefault();
            setMenu({ x: e.clientX, y: e.clientY });
          }}
        >
          {dataset ? (
            <div className="flex h-full flex-col p-2">
              <DataView dataset={dataset} tall />
            </div>
          ) : (
            <CodeMirror
              value={edits[file.name] ?? file.code}
              extensions={[python()]}
              editable={editable}
              readOnly={!editable}
              theme="light"
              basicSetup={{ foldGutter: false, highlightActiveLine: editable }}
              onChange={(code) => onEdit(file.name, code)}
              onUpdate={onUpdate}
            />
          )}
        </div>
      </div>
      {side}

      {menu && (selection.current || onFork) && (
        <div
          role="menu"
          style={{ left: menu.x, top: menu.y }}
          className="fixed z-50 w-48 rounded-lg border border-gray-200 bg-white p-1 text-sm shadow-lg"
        >
          {onFork && (
            <button
              role="menuitem"
              onClick={onFork}
              className="flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-left text-gray-700 hover:bg-blue-50"
            >
              <span>⑂</span>
              {t("fork")}
            </button>
          )}
          {selection.current &&
            (["explain", "fix"] as const).map((action) => (
              <button
                key={action}
                role="menuitem"
                onClick={() => selection.current && askAI(action, selection.current)}
                className="flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-left text-gray-700 hover:bg-blue-50"
              >
                <span>{action === "explain" ? "💡" : "🔧"}</span>
                {t(action === "explain" ? "agent.explainSel" : "agent.fixSel")}
              </button>
            ))}
        </div>
      )}
    </section>
  );
}
