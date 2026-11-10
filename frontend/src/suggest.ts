/**
 * Suggested prompts for the AI panel's "This problem" tab, built from what the problem has (meta.yaml fields, code,
 * results) and what is open. Self-contained (type imports only) so `node --test frontend/tests` can run it.
 * Returns string keys (in strings.agent.ts) with their {placeholders}; the panel fills them in its language.
 */
import type { ViewContext } from "./agentBus";
import type { Problem } from "./types";

export interface Suggestion {
  key: string;
  vars: Record<string, string>;
}

/** A library problem without enough metadata for its own prompts falls back to these. */
export const LIBRARY_PROMPTS = ["agent.ex1", "agent.ex3"];
const MAX = 4;

/** A new problem's boilerplate (workspace.py BLANK_MAIN), an old empty canvas, or one with no real code. */
export function isSkeleton(problem: Pick<Problem, "files">): boolean {
  const code = (name: string) => problem.files.find((f) => f.name === name)?.code ?? "";
  const main = code("main.py");
  const eq = code("equation.py");
  const hasDef = /^\s*def\s+\w+/m.test(eq.replace(/"""[\s\S]*?"""/g, ""));
  return main.includes("Boilerplate example") || main.includes("Empty canvas: write your problem in main.py") || !hasDef;
}

const short = (text: string, n: number) => {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > n ? `${flat.slice(0, n - 1)}…` : flat;
};

/** A number as a person would write it in a prompt: 0.1, 25, 1.5e-3. */
const num = (x: number) =>
  Math.abs(x) >= 1e4 || (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(1) : String(+x.toPrecision(3));

/** One parameter worth experimenting with, and a value to try: the far end of its range (or a doubling). */
export function paramExperiment(problem: Problem, lang?: string) {
  const labels = (lang && problem.i18n?.[lang]?.labels) || {};
  for (const [name, spec] of Object.entries(problem.params)) {
    const label = short(labels[name] ?? spec.label ?? name, 40);
    if (spec.type === "choice" || Array.isArray(spec.options)) {
      const other = spec.options?.find((o) => o !== String(spec.default));
      if (other) return { param: label, from: String(spec.default), to: other, unit: "" };
      continue;
    }
    const d = Number(spec.default);
    if (!Number.isFinite(d)) continue;
    const to =
      spec.max != null && spec.max !== d ? spec.max : spec.min != null && spec.min !== d ? spec.min : d === 0 ? 1 : d * 2;
    return { param: label, from: num(d), to: num(to), unit: spec.unit ? ` ${spec.unit}` : "" };
  }
  return null;
}

/** Prompts made from a library problem's own content: its index form, a real parameter, its tests, its result. */
function libraryPrompts(problem: Problem, title: string, lang?: string): Suggestion[] {
  const out: Suggestion[] = [];
  const add = (key: string, extra: Record<string, string> = {}) => out.push({ key, vars: { title, ...extra } });
  if (problem.index_form.trim()) add("agent.sg.libIndex");
  const experiment = paramExperiment(problem, lang);
  const meanings = (lang && problem.i18n?.[lang]?.meanings) || [];
  const rowAt = problem.translation.findIndex((r) => r.symbol.trim() && r.shape.trim());
  if (experiment) add("agent.sg.libParam", experiment);
  else if (rowAt >= 0) {
    const row = problem.translation[rowAt];
    const symbol = `$${row.symbol.replace(/\$/g, "")}$`;
    add("agent.sg.libSymbol", { symbol, meaning: short(meanings[rowAt] ?? row.meaning, 50) });
  }
  if (problem.files.some((f) => f.name === "test.py")) add("agent.sg.libTests");
  add(problem.animated ? "agent.sg.libAnim" : "agent.sg.libPlot");
  if (out.length < 2) return LIBRARY_PROMPTS.map((key) => ({ key, vars: {} }));
  return out.slice(0, MAX);
}

export function suggestPrompts(problem: Problem, view: ViewContext | null, lang?: string): Suggestion[] {
  const localTitle = (lang && problem.i18n?.[lang]?.title) || problem.title;
  if (problem.category !== "workspace")
    return libraryPrompts(problem, short(localTitle.replace(/\$/g, ""), 60) || "this problem", lang);
  const title = short(problem.title.replace(/\$/g, ""), 60) || "this problem";
  const summary = short(problem.summary || problem.title, 160);
  const vars = { title, summary };
  const out: Suggestion[] = [];
  const add = (key: string, extra: Record<string, string> = {}) => out.push({ key, vars: { ...vars, ...extra } });
  const ws = view?.problem === problem.id ? view.workspace : undefined;

  if (ws?.kind === "notebook") {
    const cells = ws.cells ?? [];
    const lastCode = cells.map((c) => c.type).lastIndexOf("code") + 1;
    const n = ws.focusedCell ?? lastCode;
    if (n > 0) add("agent.sg.explainCell", { n: String(n) });
    add("agent.sg.plotCell");
  }
  const data = problem.data?.find((d) => /\.(csv|tsv|txt|dat|json|npy|npz)$/i.test(d.name)) ?? problem.data?.[0];
  if (data) add("agent.sg.fitData", { file: data.name });
  const skeleton = isSkeleton(problem);
  const run = view?.problem === problem.id ? view.run : undefined;
  const rendered = !!run?.output && !run.failed;
  if (!problem.vector_form.trim() || !problem.index_form.trim()) add("agent.sg.equations");
  if (rendered) add("agent.sg.explainResult"); // a result on screen: explaining it is the next most useful step
  if (skeleton) add("agent.sg.implement");
  else if (!rendered) add("agent.sg.runFix"); // code but no result yet, or the last run failed
  if (!problem.translation.length) add("agent.sg.translation");
  if (!Object.keys(problem.params).length) add("agent.sg.params");
  return out.slice(0, MAX);
}
