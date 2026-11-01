import { python } from "@codemirror/lang-python";
import CodeMirror from "@uiw/react-codemirror";
import { useCallback, useEffect, useMemo, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { emitContext, emitSelection, outputText, selectionReporter, textareaSelection } from "../agentBus";
import type { NotebookCellResult } from "../api";
import { useLang } from "../i18n";
import type { Cell, CellOutput, Notebook, Params } from "../types";
import { newCell } from "../workspaceStore";
import Tex, { RichText } from "./Tex";

interface Props {
  notebook: Notebook;
  /** The Workspace problem's id and default parameters, for the AI panel's view of this notebook. */
  problemId: string;
  problemTitle: string;
  params: Params;
  /** Functional update, so edits made while a cell runs are kept. */
  onChange: (fn: (nb: Notebook) => Notebook) => void;
  onDelete: () => void;
  /** Run these code cells in a fresh process; the last cell's output, or every cell's with all. */
  onRun: (code: string[], all: boolean) => Promise<NotebookCellResult[]>;
  canRun: boolean;
}

/**
 * A notebook of ordered Markdown (with LaTeX) and Python cells. Execution is stateless: running cell k
 * re-runs code cells 1..k in a fresh process on the backend and shows cell k's printed text and figures.
 */
export default function NotebookView({
  notebook,
  problemId,
  problemTitle,
  params,
  onChange,
  onDelete,
  onRun,
  canRun,
}: Props) {
  const { t } = useLang();
  const [running, setRunning] = useState<Set<string>>(new Set());
  const [editing, setEditing] = useState<Set<string>>(() => new Set(notebook.cells.filter((c) => !c.source).map((c) => c.id)));
  const cells = notebook.cells;
  const busy = running.size > 0;
  // The cell the user is in (by id, so moving cells keeps it), for the AI panel.
  const [focused, setFocused] = useState<string | null>(null);
  const focusedIndex = cells.findIndex((c) => c.id === focused);
  const aiCells = useMemo(
    () => cells.map((c) => ({ id: c.id, type: c.type, source: c.source, output: outputText(c.output) })),
    [cells],
  );
  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    emitContext({
      problem: problemId,
      title: problemTitle.replace(/\$/g, ""),
      edits: {},
      params: JSON.parse(paramsKey),
      workspace: {
        kind: "notebook",
        notebookId: notebook.id,
        notebookTitle: notebook.title,
        cells: aiCells,
        focusedCell: focusedIndex >= 0 ? focusedIndex + 1 : undefined,
      },
    });
  }, [problemId, problemTitle, paramsKey, notebook.id, notebook.title, aiCells, focusedIndex]);
  useEffect(() => () => emitSelection(null), [notebook.id]);
  const cellLabel = (i: number) => `notebook: ${notebook.title || t("ws.notebook")}, cell ${i + 1}`;

  const setCells = (fn: (cells: Cell[]) => Cell[]) => onChange((nb) => ({ ...nb, cells: fn(nb.cells) }));
  const patch = (id: string, fields: Partial<Cell>) =>
    setCells((cs) => cs.map((c) => (c.id === id ? { ...c, ...fields } : c)));
  const toggle = (set: Set<string>, id: string, on: boolean) => {
    const next = new Set(set);
    if (on) next.add(id);
    else next.delete(id);
    return next;
  };

  function add(type: Cell["type"], after?: number) {
    const cell = newCell(type);
    setCells((cs) => {
      const at = after === undefined ? cs.length : after + 1;
      return [...cs.slice(0, at), cell, ...cs.slice(at)];
    });
    setEditing((s) => toggle(s, cell.id, true));
  }
  function move(i: number, by: -1 | 1) {
    setCells((cs) => {
      const j = i + by;
      if (j < 0 || j >= cs.length) return cs;
      const next = [...cs];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });
  }

  /** Run code cells up to `upTo` (inclusive); all=true captures each one's output (Run all). */
  async function run(upTo: number, all: boolean) {
    const code = cells.slice(0, upTo + 1).filter((c) => c.type === "code");
    if (!code.length) return;
    const ids = all ? code.map((c) => c.id) : [cells[upTo].id];
    setRunning(new Set(ids));
    let outputs: Record<string, CellOutput | null>;
    try {
      const results = await onRun(
        code.map((c) => c.source),
        all,
      );
      outputs = Object.fromEntries(ids.map((id) => [id, null]));
      for (const r of results) outputs[code[r.index].id] = { stdout: r.stdout, figures: r.figures, error: r.error };
      const failed = results.find((r) => r.error);
      const target = cells[upTo].id;
      if (!all && failed && code[failed.index].id !== target)
        outputs[target] = { stdout: "", figures: [], error: t("notebook.earlierFailed") };
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      const console = (e as { console?: string }).console;
      outputs = { [ids[ids.length - 1]]: { stdout: "", figures: [], error: console ? `${message}\n${console}` : message } };
    } finally {
      setRunning(new Set());
    }
    setCells((cs) => cs.map((c) => (c.id in outputs ? { ...c, output: outputs[c.id] } : c)));
  }

  const lastCode = cells.map((c) => c.type).lastIndexOf("code");

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-gray-50/60 lg:min-h-0">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-gray-200 bg-white px-4 py-2">
        <span className="text-lg">📓</span>
        <input
          value={notebook.title}
          onChange={(e) => {
            const title = e.target.value;
            onChange((nb) => ({ ...nb, title }));
          }}
          placeholder={t("notebook.titlePlaceholder")}
          aria-label={t("notebook.titlePlaceholder")}
          className="min-w-0 flex-1 rounded-md border border-transparent px-2 py-1 text-base font-semibold text-gray-900 hover:border-gray-200 focus:border-blue-300 focus:outline-none"
        />
        <span className="hidden truncate text-xs text-gray-400 sm:inline">
          <RichText text={problemTitle} />
        </span>
        <button
          onClick={() => run(lastCode, true)}
          disabled={!canRun || busy || lastCode < 0}
          title={canRun ? t("notebook.runAllHint") : t("workspace.unavailable")}
          className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? t("notebook.running") : `▶▶ ${t("notebook.runAll")}`}
        </button>
        <button
          onClick={() => confirm(t("notebook.deleteConfirm")) && onDelete()}
          title={t("notebook.delete")}
          className="rounded-md px-2 py-1.5 text-sm text-gray-400 hover:bg-red-50 hover:text-red-600"
        >
          🗑
        </button>
      </div>

      <div className="space-y-3 p-3 sm:p-5 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
        <p className="text-xs text-gray-500">
          <RichText text={t("notebook.hint")} />
        </p>
        {cells.length === 0 && <p className="py-6 text-center text-sm text-gray-400">{t("notebook.empty")}</p>}
        {cells.map((cell, i) => {
          const isRunning = running.has(cell.id);
          const isEditing = cell.type === "code" || editing.has(cell.id);
          return (
            <div
              key={cell.id}
              data-cell={cell.type}
              onFocusCapture={() => setFocused(cell.id)}
              className={`group rounded-xl border bg-white ${isRunning ? "border-blue-300 ring-2 ring-blue-100" : "border-gray-200"}`}
            >
              <div className="flex items-center gap-1 border-b border-gray-100 px-2 py-1 text-xs text-gray-500">
                <span className="w-7 font-mono text-gray-400">[{i + 1}]</span>
                <span className="rounded bg-gray-100 px-1.5 py-0.5 font-medium">
                  {cell.type === "code" ? "Python" : "Markdown"}
                </span>
                <span className="ml-auto flex items-center gap-0.5">
                  {cell.type === "code" ? (
                    <CellButton
                      onClick={() => run(i, false)}
                      disabled={!canRun || busy}
                      title={canRun ? t("notebook.runCellHint") : t("workspace.unavailable")}
                      className="font-medium text-blue-700"
                    >
                      ▶ {t("notebook.runCell")}
                    </CellButton>
                  ) : (
                    <CellButton onClick={() => setEditing((s) => toggle(s, cell.id, !isEditing))}>
                      {isEditing ? `👁 ${t("notebook.preview")}` : `✎ ${t("notebook.edit")}`}
                    </CellButton>
                  )}
                  <CellButton onClick={() => move(i, -1)} disabled={i === 0} title={t("notebook.moveUp")}>
                    ↑
                  </CellButton>
                  <CellButton onClick={() => move(i, 1)} disabled={i === cells.length - 1} title={t("notebook.moveDown")}>
                    ↓
                  </CellButton>
                  <CellButton
                    onClick={() => setCells((cs) => cs.filter((c) => c.id !== cell.id))}
                    title={t("notebook.deleteCell")}
                    className="hover:text-red-600"
                  >
                    ✕
                  </CellButton>
                </span>
              </div>

              {cell.type === "code" ? (
                <CellEditor
                  source={cell.source}
                  onChange={(source) => patch(cell.id, { source })}
                  problem={problemId}
                  region={cellLabel(i)}
                />
              ) : isEditing ? (
                <textarea
                  value={cell.source}
                  onChange={(e) => patch(cell.id, { source: e.target.value })}
                  onSelect={(e) =>
                    emitSelection(
                      textareaSelection(e.currentTarget, {
                        problem: problemId,
                        file: "notebook",
                        region: cellLabel(i),
                        lang: "markdown",
                      }),
                    )
                  }
                  placeholder={t("notebook.mdPlaceholder")}
                  rows={Math.max(3, cell.source.split("\n").length + 1)}
                  className="block w-full resize-y rounded-b-xl px-3 py-2 font-mono text-sm text-gray-800 focus:outline-none"
                />
              ) : (
                <div
                  className="cursor-text px-4 py-3 text-sm leading-relaxed text-gray-800"
                  onDoubleClick={() => setEditing((s) => toggle(s, cell.id, true))}
                >
                  {cell.source.trim() ? <Markdown source={cell.source} /> : <span className="text-gray-400">{t("notebook.mdPlaceholder")}</span>}
                </div>
              )}

              {cell.type === "code" && (isRunning || cell.output) && (
                <div data-output className="space-y-2 border-t border-gray-100 bg-gray-50/70 px-3 py-2">
                  {isRunning ? (
                    <p className="animate-pulse text-xs text-blue-700">{t("notebook.running")}</p>
                  ) : (
                    cell.output && <Output output={cell.output} />
                  )}
                </div>
              )}
            </div>
          );
        })}

        <div className="flex justify-center gap-2 pt-1">
          <button
            onClick={() => add("markdown")}
            className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
          >
            + {t("notebook.addText")}
          </button>
          <button
            onClick={() => add("code")}
            className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
          >
            + {t("notebook.addCode")}
          </button>
        </div>
      </div>
    </main>
  );
}

/** A code cell's editor; highlighted text goes to the AI panel labelled with the notebook and cell. */
function CellEditor({ source, onChange, problem, region }: { source: string; onChange: (s: string) => void; problem: string; region: string }) {
  const onUpdate = useMemo(
    () => selectionReporter({ problem, file: "notebook", region, lang: "python" }),
    [problem, region],
  );
  // Stable callbacks: a new onChange each render makes react-codemirror reconfigure the editor in an effect, which
  // while typing fast re-entered onChange → store update → render in a loop ("Maximum update depth exceeded").
  const latest = useRef(onChange);
  latest.current = onChange;
  const change = useCallback((s: string) => latest.current(s), []);
  return (
    <CodeMirror
      value={source}
      extensions={PY}
      theme="light"
      basicSetup={CELL_SETUP}
      onChange={change}
      onUpdate={onUpdate}
      className="text-sm"
    />
  );
}
const PY = [python()];
const CELL_SETUP = { foldGutter: false, highlightActiveLine: false };

function CellButton({
  children,
  className = "",
  ...props
}: { children: ReactNode; className?: string } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className={`rounded px-1.5 py-0.5 hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-40 ${className}`}
    >
      {children}
    </button>
  );
}

function Output({ output }: { output: CellOutput }) {
  return (
    <>
      {output.stdout && (
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap font-mono text-xs text-gray-800">{output.stdout}</pre>
      )}
      {output.error && (
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-md bg-red-50 p-2 font-mono text-xs text-red-700">
          {output.error}
        </pre>
      )}
      {output.figures.map((src, k) => (
        <img key={k} src={src} alt={`Figure ${k + 1}`} className="max-w-full rounded-md border border-gray-200 bg-white" />
      ))}
    </>
  );
}

/** A small Markdown subset: # headings, - lists, paragraphs, `code`, **bold**, $inline$ and $$display$$ math. */
export function Markdown({ source }: { source: string }) {
  const parts = source.split(/\$\$([\s\S]+?)\$\$/);
  return (
    <div className="space-y-2">
      {parts.map((part, i) =>
        i % 2 ? (
          <div key={i} className="overflow-x-auto">
            <Tex tex={part.trim()} display />
          </div>
        ) : (
          blocks(part).map((b, j) => {
            const key = `${i}-${j}`;
            if (b.kind === "h") {
              const size = ["text-xl", "text-lg", "text-base"][b.level - 1];
              return (
                <p key={key} className={`${size} font-semibold text-gray-900`}>
                  <RichText text={b.text} />
                </p>
              );
            }
            if (b.kind === "ul")
              return (
                <ul key={key} className="list-disc space-y-0.5 pl-5">
                  {b.items.map((item, k) => (
                    <li key={k}>
                      <RichText text={item} />
                    </li>
                  ))}
                </ul>
              );
            return (
              <p key={key}>
                <RichText text={b.text} />
              </p>
            );
          })
        ),
      )}
    </div>
  );
}

type Block = { kind: "h"; level: number; text: string } | { kind: "ul"; items: string[] } | { kind: "p"; text: string };

function blocks(text: string): Block[] {
  const out: Block[] = [];
  for (const line of text.split("\n")) {
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    const item = line.match(/^\s*[-*]\s+(.*)$/);
    const last = out[out.length - 1];
    if (heading) out.push({ kind: "h", level: heading[1].length, text: heading[2] });
    else if (item) {
      if (last?.kind === "ul") last.items.push(item[1]);
      else out.push({ kind: "ul", items: [item[1]] });
    } else if (!line.trim()) out.push({ kind: "p", text: "" });
    else if (last?.kind === "p" && last.text) last.text += " " + line.trim();
    else out.push({ kind: "p", text: line.trim() });
  }
  return out.filter((b) => b.kind !== "p" || b.text);
}
