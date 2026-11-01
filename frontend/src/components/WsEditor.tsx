/** In-place editing of a workspace problem: inline fields, or the whole thing as one Markdown document. */
import { markdown } from "@codemirror/lang-markdown";
import { EditorView } from "@codemirror/view";
import CodeMirror from "@uiw/react-codemirror";
import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { emitSelection, selectionReporter } from "../agentBus";
import { useLang } from "../i18n";
import { fromMarkdown, pickMd, toMarkdown, type MetaFields } from "../metaDoc";
import type { ParamSpec, Problem, TranslationRow } from "../types";
import EquationPanel from "./EquationPanel";
import Tex, { RichText } from "./Tex";

export type EditMode = "inline" | "markdown";
const MODE_KEY = "tensorizer.wsEditMode";

/** The chosen edit mode, remembered per browser. */
export function useEditMode(): [EditMode, (m: EditMode) => void] {
  const [mode, setMode] = useState<EditMode>(() => {
    try {
      return localStorage.getItem(MODE_KEY) === "markdown" ? "markdown" : "inline";
    } catch {
      return "inline";
    }
  });
  const choose = (m: EditMode) => {
    setMode(m);
    try {
      localStorage.setItem(MODE_KEY, m);
    } catch {
      // private window: the choice lasts for this page only
    }
  };
  return [mode, choose];
}

export function ModeToggle({ mode, onMode }: { mode: EditMode; onMode: (m: EditMode) => void }) {
  const { t } = useLang();
  return (
    <div className="inline-flex rounded-md border border-gray-200 bg-gray-50 p-0.5 text-xs" role="tablist">
      {(["inline", "markdown"] as const).map((m) => (
        <button
          key={m}
          role="tab"
          aria-selected={mode === m}
          onClick={() => onMode(m)}
          className={`rounded px-2.5 py-1 font-medium ${mode === m ? "bg-white text-blue-700 shadow-sm" : "text-gray-500 hover:text-gray-800"}`}
        >
          {t(m === "inline" ? "edit.inline" : "edit.markdown")}
        </button>
      ))}
    </div>
  );
}

/** A value shown rendered; click to edit, Enter (single line) or blur saves, Esc cancels. */
export function Field({
  value,
  onCommit,
  render,
  multiline = false,
  placeholder,
  className = "",
  inputClass = "",
  preview,
  label,
}: {
  value: string;
  onCommit: (v: string) => void;
  render?: (v: string) => ReactNode;
  multiline?: boolean;
  placeholder: string;
  className?: string;
  inputClass?: string;
  /** Shown live under the input while editing (e.g. KaTeX). */
  preview?: (v: string) => ReactNode;
  label?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const done = (save: boolean) => {
    if (save && draft !== null && draft !== value) onCommit(draft);
    setDraft(null);
  };
  if (draft !== null) {
    const common = {
      autoFocus: true,
      value: draft,
      "aria-label": label ?? placeholder,
      placeholder,
      onChange: (e: { target: { value: string } }) => setDraft(e.target.value),
      onBlur: () => done(true),
      className: `w-full rounded-md border border-blue-300 bg-white px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-100 ${inputClass}`,
    };
    return (
      <div className={className}>
        {multiline ? (
          <textarea
            {...common}
            rows={Math.max(2, draft.split("\n").length)}
            onKeyDown={(e) => e.key === "Escape" && done(false)}
          />
        ) : (
          <input
            {...common}
            onKeyDown={(e) => {
              if (e.key === "Enter") done(true);
              else if (e.key === "Escape") done(false);
            }}
          />
        )}
        {preview && draft.trim() && <div className="mt-1 rounded-md bg-gray-50 px-2 py-1">{preview(draft)}</div>}
      </div>
    );
  }
  return (
    <div
      role="button"
      tabIndex={0}
      title={label ?? placeholder}
      aria-label={label ?? placeholder}
      onClick={() => setDraft(value)}
      onKeyDown={(e) => e.key === "Enter" && setDraft(value)}
      className={`group/field relative cursor-text rounded-md ${
        value.trim() ? "hover:bg-blue-50/60" : "border border-dashed border-gray-300 px-2 py-1 text-gray-400 hover:border-blue-300 hover:text-blue-600"
      } ${className}`}
    >
      {value.trim() ? (render ? render(value) : value) : `+ ${placeholder}`}
      {value.trim() && (
        <span className="pointer-events-none absolute -top-1 right-0 text-xs text-blue-500 opacity-0 group-hover/field:opacity-100">✎</span>
      )}
    </div>
  );
}

const rich = (v: string) => <RichText text={v} />;

/** Vector form, index form, translation table, notes and parameters, each editable in place. */
export function InlineEquations({ problem, onMeta }: { problem: Problem; onMeta: (f: Partial<MetaFields>) => void }) {
  const { t } = useLang();
  const eq = (key: "vector_form" | "index_form", title: string) => (
    <>
      <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-gray-500 first:mt-0">{title}</h3>
      <Field
        className="mt-1 text-lg"
        inputClass="font-mono text-sm"
        value={problem[key]}
        placeholder={t("edit.addEquation")}
        label={title}
        render={(v) => <Tex tex={v} display />}
        preview={(v) => <Tex tex={v} display />}
        onCommit={(v) => onMeta({ [key]: v })}
      />
    </>
  );
  return (
    <section data-ai-region="equations" className="rounded-xl border border-gray-200 bg-white p-5" data-testid="ws-equations">
      {eq("vector_form", t("eq.vector"))}
      {eq("index_form", t("eq.index"))}
      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.translation")}</h3>
      <TranslationEditor rows={problem.translation} onChange={(translation) => onMeta({ translation })} />
      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("edit.notes")}</h3>
      <div data-ai-region="notes">
      <Field
        multiline
        className="mt-1 text-sm leading-relaxed text-gray-600"
        inputClass="text-sm"
        value={problem.notes ?? ""}
        placeholder={t("edit.addNotes")}
        render={rich}
        onCommit={(notes) => onMeta({ notes })}
      />
      </div>
      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("edit.params")}</h3>
      <div data-ai-region="params">
        <ParamsEditor params={problem.params} onChange={(params) => onMeta({ params })} />
      </div>
    </section>
  );
}

function TranslationEditor({ rows, onChange }: { rows: TranslationRow[]; onChange: (rows: TranslationRow[]) => void }) {
  const { t } = useLang();
  const set = (i: number, patch: Partial<TranslationRow>) => onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  return (
    <div data-ai-region="translation" className="mt-2 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-xs text-gray-500">
            <th className="py-1.5 pr-2 font-medium">{t("col.symbol")}</th>
            <th className="py-1.5 pr-2 font-medium">{t("col.code")}</th>
            <th className="py-1.5 pr-2 font-medium">{t("col.shape")}</th>
            <th className="py-1.5 pr-2 font-medium">{t("col.meaning")}</th>
            <th className="w-6" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-gray-100 align-top" data-testid="ws-translation-row">
              <td className="py-1.5 pr-2 whitespace-nowrap">
                <Field
                  value={row.symbol}
                  placeholder="x_i"
                  label={t("col.symbol")}
                  inputClass="font-mono text-xs"
                  render={(v) => <Tex tex={v} />}
                  preview={(v) => <Tex tex={v} />}
                  onCommit={(symbol) => set(i, { symbol })}
                />
              </td>
              <td className="py-1.5 pr-2">
                <Field
                  value={row.code}
                  placeholder="x[i]"
                  label={t("col.code")}
                  inputClass="font-mono text-xs"
                  render={(v) => <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs text-gray-800">{v}</code>}
                  onCommit={(code) => set(i, { code })}
                />
              </td>
              <td className="py-1.5 pr-2 font-mono text-xs text-gray-600">
                <Field value={row.shape} placeholder="(N,)" label={t("col.shape")} inputClass="font-mono text-xs" onCommit={(shape) => set(i, { shape })} />
              </td>
              <td className="py-1.5 pr-2 text-[13px] text-gray-600">
                <Field value={row.meaning} placeholder={t("col.meaning")} render={rich} inputClass="text-xs" onCommit={(meaning) => set(i, { meaning })} />
              </td>
              <td className="py-1.5">
                <button
                  onClick={() => onChange(rows.filter((_, j) => j !== i))}
                  title={t("edit.deleteRow")}
                  aria-label={t("edit.deleteRow")}
                  className="rounded px-1 text-xs text-gray-400 hover:bg-red-50 hover:text-red-600"
                >
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button
        onClick={() => onChange([...rows, { symbol: "", code: "", shape: "", meaning: "" }])}
        className="mt-1.5 rounded-md border border-dashed border-gray-300 px-2 py-0.5 text-xs text-gray-500 hover:border-blue-300 hover:text-blue-600"
      >
        + {t("edit.addRow")}
      </button>
    </div>
  );
}

const num = (s: string) => (s.trim() === "" || isNaN(Number(s)) ? undefined : Number(s));

function ParamsEditor({ params, onChange }: { params: Record<string, ParamSpec>; onChange: (p: Record<string, ParamSpec>) => void }) {
  const { t } = useLang();
  const entries = Object.entries(params);
  const write = (list: [string, ParamSpec][]) => onChange(Object.fromEntries(list));
  const set = (i: number, name: string, spec: ParamSpec) => {
    // drop empty optional fields so meta.yaml stays tidy
    const clean = Object.fromEntries(Object.entries(spec).filter(([, v]) => v !== undefined && v !== "")) as unknown as ParamSpec;
    write(entries.map((e, j) => (j === i ? [name, clean] : e)));
  };
  const cellInput = "w-full min-w-[3.5rem] rounded border border-gray-200 px-1.5 py-0.5 font-mono text-xs focus:border-blue-300 focus:outline-none";
  return (
    <div className="mt-2 overflow-x-auto">
      {entries.length > 0 && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-gray-500">
              {(["name", "default", "min", "max", "step", "label", "unit", "type"] as const).map((c) => (
                <th key={c} className="py-1 pr-1 font-medium">
                  {t(`param.${c}`)}
                </th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, spec], i) => {
              const choice = spec.type === "choice";
              const text = (v: unknown) => (v === undefined ? "" : String(v));
              const cell = (value: string, commit: (v: string) => void, label: string, span = 1) => (
                <td className="py-0.5 pr-1" colSpan={span}>
                  <input
                    key={value}
                    defaultValue={value}
                    aria-label={label}
                    onBlur={(e) => e.target.value !== value && commit(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
                    className={cellInput}
                  />
                </td>
              );
              return (
                <tr key={i} data-testid="ws-param-row">
                  {cell(name, (v) => v.trim() && !params[v.trim()] && set(i, v.trim().replace(/\W/g, "_"), spec), t("param.name"))}
                  {cell(text(spec.default), (v) => set(i, name, { ...spec, default: choice ? v : (num(v) ?? 0) }), t("param.default"))}
                  {choice
                    ? cell(
                        (spec.options ?? []).join(", "),
                        (v) => set(i, name, { ...spec, options: v.split(",").map((o) => o.trim()).filter(Boolean) }),
                        t("param.options"),
                        3,
                      )
                    : (["min", "max", "step"] as const).map((k) => (
                        <Fragment key={k}>{cell(text(spec[k]), (v) => set(i, name, { ...spec, [k]: num(v) }), t(`param.${k}`))}</Fragment>
                      ))}
                  {cell(text(spec.label), (v) => set(i, name, { ...spec, label: v || undefined }), t("param.label"))}
                  {cell(text(spec.unit), (v) => set(i, name, { ...spec, unit: v || undefined }), t("param.unit"))}
                  <td className="py-0.5 pr-1">
                    <select
                      value={spec.type ?? "float"}
                      aria-label={t("param.type")}
                      onChange={(e) => {
                        const type = e.target.value as ParamSpec["type"];
                        const next: ParamSpec = { ...spec, type: type === "float" ? undefined : type };
                        if (type === "choice") {
                          next.options = spec.options ?? [String(spec.default)];
                          next.default = String(next.options[0]);
                          delete next.min;
                          delete next.max;
                          delete next.step;
                        } else {
                          delete next.options;
                          next.default = num(String(spec.default)) ?? 0;
                          if (type === "int") next.default = Math.round(next.default as number);
                        }
                        set(i, name, next);
                      }}
                      className="rounded border border-gray-200 bg-white px-1 py-0.5 text-xs"
                    >
                      <option value="float">float</option>
                      <option value="int">int</option>
                      <option value="choice">choice</option>
                    </select>
                  </td>
                  <td>
                    <button
                      onClick={() => write(entries.filter((_, j) => j !== i))}
                      title={t("edit.deleteParam")}
                      aria-label={t("edit.deleteParam")}
                      className="rounded px-1 text-gray-400 hover:bg-red-50 hover:text-red-600"
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <button
        onClick={() => {
          let n = entries.length + 1;
          while (params[`p${n}`]) n++;
          write([...entries, [`p${n}`, { default: 1, min: 0, max: 10, step: 0.1, label: `p${n}` }]]);
        }}
        className="mt-1.5 rounded-md border border-dashed border-gray-300 px-2 py-0.5 text-xs text-gray-500 hover:border-blue-300 hover:text-blue-600"
      >
        + {t("edit.addParam")}
      </button>
    </div>
  );
}

/**
 * The problem as one Markdown document, with a live rendered preview. Parsed back (debounced) into the meta
 * fields; while it can't be parsed a warning shows and the last good state stays. The text is never replaced
 * while it differs only because the user is typing.
 */
const MD_EXTENSIONS = [markdown(), EditorView.lineWrapping];

export function MarkdownEditor({ problem, onMeta }: { problem: Problem; onMeta: (f: Partial<MetaFields>) => void }) {
  const { t } = useLang();
  const fields = pickMd({ ...problem, notes: problem.notes ?? "" });
  const fieldsKey = JSON.stringify(fields);
  const [text, setText] = useState(() => toMarkdown(fields));
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState(fields);
  const synced = useRef(fieldsKey); // the fields the text last agreed with
  const timer = useRef<number | undefined>(undefined);
  // Text highlighted in the document → the AI panel's selection, labelled "markdown".
  const onUpdate = useMemo(
    () => selectionReporter({ problem: problem.id, file: "markdown", region: "markdown", lang: "markdown" }),
    [problem.id],
  );
  useEffect(() => () => emitSelection(null), [problem.id]); // leaving the document: its highlight goes too

  // Changed elsewhere (inline, meta.yaml): show it, unless it is the change this editor just made.
  useEffect(() => {
    if (fieldsKey === synced.current) return;
    synced.current = fieldsKey;
    const next = JSON.parse(fieldsKey);
    setText(toMarkdown(next));
    setPreview(next);
    setError(null);
  }, [fieldsKey]);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  function edit(value: string) {
    setText(value);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      try {
        const parsed = fromMarkdown(value);
        setError(null);
        setPreview(parsed);
        const key = JSON.stringify(parsed);
        if (key !== synced.current) {
          synced.current = key;
          onMeta(parsed);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }, 500);
  }

  return (
    <div className="space-y-3" data-testid="ws-markdown">
      <div className="min-w-0 overflow-hidden rounded-xl border border-gray-200 bg-white">
        <CodeMirror value={text} extensions={MD_EXTENSIONS} theme="light" basicSetup={{ foldGutter: false }} onChange={edit} onUpdate={onUpdate} className="text-sm" />
        {error && (
          <p className="border-t border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-800">
            ⚠ {t("edit.mdError")}: {error}
          </p>
        )}
      </div>
      <div className="min-w-0 space-y-3">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <h2 className="text-lg font-semibold">
            <RichText text={preview.title} />
          </h2>
          {preview.summary && (
            <p className="mt-1 whitespace-pre-line text-sm text-gray-600">
              <RichText text={preview.summary} />
            </p>
          )}
        </div>
        <EquationPanel problem={{ ...problem, ...preview }} />
      </div>
    </div>
  );
}
