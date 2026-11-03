/** Shape of generated/manifest.json, written by scripts/build_assets.py. */

export interface ParamSpec {
  default: number | string;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  unit?: string;
  type?: "float" | "int" | "choice";
  options?: string[];
}

export interface TranslationRow {
  symbol: string;
  code: string;
  shape: string;
  meaning: string;
}

export interface CodeFile {
  name: string;
  code: string;
}

export interface ProblemTranslation {
  title?: string;
  summary?: string;
  notes?: string;
  /** one entry per translation row, same order */
  meanings?: string[];
  labels?: Record<string, string>;
}

export interface Problem {
  id: string;
  category: string;
  level: Level;
  slug: string;
  title: string;
  order: number;
  summary: string;
  animated: boolean;
  vector_form: string;
  index_form: string;
  translation: TranslationRow[];
  notes?: string;
  params: Record<string, ParamSpec>;
  /** Labels from meta.yaml `tags`: library tags such as `millennium`, or a Workspace problem's own labels. */
  tags?: string[];
  files: CodeFile[];
  output: string | null;
  console: string;
  diagram: string | null;
  /** Workspace problems only: the server sandbox (or agent session) holding the files. */
  session?: string;
  /** Workspace problems only: the raw meta.yaml, and its keys the page doesn't edit (category, order, …). */
  meta?: string;
  meta_extra?: Record<string, unknown>;
  /** Workspace problems only: uploaded data files (the bytes are in the browser's IndexedDB, synced to data/). */
  data?: DataFile[];
  /** Wikipedia article titles by wiki language code (en, zh, fr, …), from content/wiki.json. */
  wiki?: { wikidata: string; titles: Record<string, string> } | null;
  /** Key papers (label, title, DOI/arXiv url), from content/papers.json. */
  papers?: { label: string; title: string; url: string | null }[];
  notebook?: string;
  pixi_zip?: string;
  /** Translations by language code; any missing field falls back to English. */
  i18n?: Record<string, ProblemTranslation>;
}

export interface DataFile {
  name: string;
  size: number;
  /** MIME type as the browser reported it. */
  type: string;
}

export type Level = "beginner" | "intermediate" | "advanced";

export interface Manifest {
  categories: { id: string; title: string }[];
  levels: Level[];
  problems: Problem[];
}

export type Params = Record<string, number | string>;

// --- Workspace (kept in the browser's IndexedDB) ------------------------------------------------------

export interface CellOutput {
  stdout: string;
  /** PNG data URLs */
  figures: string[];
  error?: string | null;
}

export interface Cell {
  id: string;
  type: "markdown" | "code";
  source: string;
  output?: CellOutput | null;
}

export interface Notebook {
  id: string;
  title: string;
  cells: Cell[];
}

/** A problem in a workspace: `entry` is its library-shaped view (files are the truth), `meta` its meta.yaml. */
export interface WsProblem {
  /** Stable local id; the page shows it at #/workspace/<id>. */
  id: string;
  entry: Problem;
  meta: string;
  notebooks: Notebook[];
  /** When it was last changed (ms since 1970), for the Workspace home. */
  updated?: number;
}

export interface Workspace {
  id: string;
  name: string;
  problems: WsProblem[];
}
