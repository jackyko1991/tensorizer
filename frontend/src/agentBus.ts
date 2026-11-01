/** Small window-event bus between the workspace (editor, parameters) and the Ask AI panel, which live in
 *  different parts of the page. */
import type { Params } from "./types";

export interface CodeSelection {
  problem: string;
  file: string;
  fromLine: number;
  toLine: number;
  text: string;
}

/** What the user currently has on screen: the problem, their edited files and parameters. */
export interface ViewContext {
  problem: string;
  title: string;
  edits: Record<string, string>;
  params: Params;
}

export type AskAction = "explain" | "fix";

const SELECTION = "tensorizer:selection";
const CONTEXT = "tensorizer:context";
const ASK = "tensorizer:ask";
const APPLY = "tensorizer:agent-files";

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
export const emitContext = (c: ViewContext) => emit(CONTEXT, c);
export const onContext = (h: (c: ViewContext) => void) => on(CONTEXT, h);
export const askAI = (action: AskAction, selection: CodeSelection) => emit(ASK, { action, selection });
export const onAsk = (h: (a: { action: AskAction; selection: CodeSelection }) => void) => on(ASK, h);
/** Claude's edited files → the code panel of that problem. */
export const applyFiles = (problem: string, files: Record<string, string>) => emit(APPLY, { problem, files });
export const onApplyFiles = (h: (a: { problem: string; files: Record<string, string> }) => void) => on(APPLY, h);
