/** Small window-event bus between the workspace (editor, parameters) and the Ask AI panel, which live in
 *  different parts of the page. */
import type { ViewUpdate } from "@codemirror/view";
import { ExternalChange } from "@uiw/react-codemirror";
import type { Params, Problem } from "./types";

export interface CodeSelection {
  problem: string;
  /** A code file or meta.yaml; for other editors a short name ("markdown", "notebook"). */
  file: string;
  fromLine: number;
  toLine: number;
  text: string;
  /** Where it was highlighted when not a problem file, e.g. "markdown" or "notebook: Notebook 1, cell 3". */
  region?: string;
  /** Language of the highlighted text: python | markdown | yaml. */
  lang?: string;
}

/** A notebook cell as the AI sees it: its source and a short text of its last output. */
export interface ContextCell {
  id: string;
  type: "markdown" | "code";
  source: string;
  output?: string;
}

/** Which part of a Workspace problem is open: the Source (in the Inline / Markdown view or the meta.yaml editor)
 *  or one of its notebooks. */
export interface WorkspaceView {
  kind: "source" | "notebook" | "data";
  mode?: "inline" | "markdown" | "yaml";
  /** The Markdown document, in Markdown mode. */
  markdown?: string;
  notebookId?: string;
  notebookTitle?: string;
  cells?: ContextCell[];
  /** 1-based index of the cell the user is in. */
  focusedCell?: number;
}

/** What the user currently has on screen: the problem, their edited files and parameters. */
export interface ViewContext {
  problem: string;
  title: string;
  edits: Record<string, string>;
  params: Params;
  workspace?: WorkspaceView;
  /** Whether the problem has a rendered result on screen, and whether the last run failed. */
  run?: { output: boolean; failed: boolean };
}

export type AskAction = "explain" | "fix";

const SELECTION = "tensorizer:selection";
const CONTEXT = "tensorizer:context";
const ASK = "tensorizer:ask";
const APPLY = "tensorizer:agent-files";
const NOTEBOOK = "tensorizer:agent-notebook";
const OPEN = "tensorizer:open-agent";

function on<T>(name: string, handler: (detail: T) => void): () => void {
  const listener = (e: Event) => handler((e as CustomEvent<T>).detail);
  window.addEventListener(name, listener);
  return () => window.removeEventListener(name, listener);
}
function emit<T>(name: string, detail: T): void {
  window.dispatchEvent(new CustomEvent(name, { detail }));
}

export const emitSelection = (s: CodeSelection | null) => emit(SELECTION, s);
export const onSelection = (h: (s: CodeSelection | null) => void) => on(SELECTION, h);
// The view changes on every keystroke (notebook cells, Markdown); the latest one is delivered after a short pause,
// outside React's commit, so typing fast never chains synchronous updates into the AI panel.
let pendingContext: ViewContext | null = null;
let contextTimer: ReturnType<typeof setTimeout> | undefined;
export const emitContext = (c: ViewContext) => {
  pendingContext = c;
  clearTimeout(contextTimer);
  contextTimer = setTimeout(() => {
    if (pendingContext) emit(CONTEXT, pendingContext);
    pendingContext = null;
  }, 120);
};
export const onContext = (h: (c: ViewContext) => void) => on(CONTEXT, h);
export const askAI = (action: AskAction, selection: CodeSelection) => emit(ASK, { action, selection });
export const onAsk = (h: (a: { action: AskAction; selection: CodeSelection }) => void) => on(ASK, h);
/** Claude's edited files → the code panel of that problem (a Workspace problem: also its re-parsed meta.yaml). */
export interface AppliedFiles {
  problem: string;
  files: Record<string, string>;
  entry?: Problem;
}
export const applyFiles = (problem: string, files: Record<string, string>, entry?: Problem) =>
  emit<AppliedFiles>(APPLY, { problem, files, entry });
export const onApplyFiles = (h: (a: AppliedFiles) => void) => on(APPLY, h);
/** The agent's edited notebook cells → that notebook in the Workspace. */
export interface AppliedNotebook {
  problem: string;
  notebook: string;
  cells: { id: string; type: "markdown" | "code"; source: string }[];
}
export const applyNotebook = (a: AppliedNotebook) => emit(NOTEBOOK, a);
export const onApplyNotebook = (h: (a: AppliedNotebook) => void) => on(NOTEBOOK, h);
/** Open the AI panel on one of its tabs. */
// The Ask AI panel lives on problem pages only: a request made elsewhere (e.g. the Workspace page's "Ask AI", which
// first opens a problem page) waits here until the panel is mounted and subscribes.
let pendingOpen: "current" | "new" | null = null;
export const openAgent = (tab: "current" | "new") => {
  pendingOpen = tab;
  emit(OPEN, tab);
};
export const onOpenAgent = (h: (tab: "current" | "new") => void) => {
  if (pendingOpen) {
    const tab = pendingOpen;
    pendingOpen = null;
    queueMicrotask(() => h(tab));
  }
  return on(OPEN, (tab: "current" | "new") => {
    pendingOpen = null;
    h(tab);
  });
};

/** A CodeMirror onUpdate handler that reports the highlighted text (null when nothing is highlighted). */
export function selectionReporter(base: Omit<CodeSelection, "fromLine" | "toLine" | "text">) {
  let last = "null";
  return (update: ViewUpdate) => {
    if (!update.selectionSet && !update.docChanged) return;
    // the page replaced the text (e.g. the AI's edit arrived): not something the user highlighted
    if (update.transactions.some((tr) => tr.annotation(ExternalChange))) return;
    const { from, to } = update.state.selection.main;
    const doc = update.state.doc;
    const next: CodeSelection | null =
      from === to
        ? null
        : { ...base, fromLine: doc.lineAt(from).number, toLine: doc.lineAt(to).number, text: doc.sliceString(from, to) };
    const key = JSON.stringify(next);
    if (key !== last) {
      last = key;
      emitSelection(next);
    }
  };
}

/** A textarea's highlighted text as a selection (line numbers counted in its value). */
export function textareaSelection(
  el: HTMLTextAreaElement,
  base: Omit<CodeSelection, "fromLine" | "toLine" | "text">,
): CodeSelection | null {
  const { selectionStart: a, selectionEnd: b, value } = el;
  if (a === b) return null;
  const line = (i: number) => value.slice(0, i).split("\n").length;
  return { ...base, fromLine: line(a), toLine: line(b), text: value.slice(a, b) };
}

/** A notebook cell's last output in a few lines for the AI: printed text, "[figure]" per plot, the error. */
export function outputText(output: { stdout: string; figures: string[]; error?: string | null } | null | undefined) {
  if (!output) return undefined;
  const parts = [
    output.stdout.trim() && output.stdout.trim().slice(-600),
    ...output.figures.map(() => "[figure]"),
    output.error && `Error: ${output.error.trim().slice(-400)}`,
  ].filter(Boolean);
  return parts.length ? parts.join("\n") : undefined;
}
