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

/** The library's read-only problems keep these. */
export const LIBRARY_PROMPTS = ["agent.ex1", "agent.ex2", "agent.ex3", "agent.ex4"];
const MAX = 4;

/** A blank problem's main.py / equation.py (workspace.py BLANK_MAIN, BLANK_DOCS) or one with no real code. */
export function isSkeleton(problem: Pick<Problem, "files">): boolean {
  const code = (name: string) => problem.files.find((f) => f.name === name)?.code ?? "";
  const main = code("main.py");
  const eq = code("equation.py");
  const hasDef = /^\s*def\s+\w+/m.test(eq.replace(/"""[\s\S]*?"""/g, ""));
  return main.includes("Empty canvas: write your problem in main.py") || !hasDef;
}

const short = (text: string, n: number) => {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > n ? `${flat.slice(0, n - 1)}…` : flat;
};

export function suggestPrompts(problem: Problem, view: ViewContext | null): Suggestion[] {
  if (problem.category !== "workspace") return LIBRARY_PROMPTS.map((key) => ({ key, vars: {} }));
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
