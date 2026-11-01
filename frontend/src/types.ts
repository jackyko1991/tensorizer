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
  files: CodeFile[];
  output: string | null;
  console: string;
  diagram: string | null;
  /** Workspace problems only: the agent session whose sandbox holds the files. */
  session?: string;
  /** Wikipedia article titles by wiki language code (en, zh, fr, …), from content/wiki.json. */
  wiki?: { wikidata: string; titles: Record<string, string> } | null;
  notebook?: string;
  pixi_zip?: string;
  /** Translations by language code; any missing field falls back to English. */
  i18n?: Record<string, ProblemTranslation>;
}

export type Level = "beginner" | "intermediate" | "advanced";

export interface Manifest {
  categories: { id: string; title: string }[];
  levels: Level[];
  problems: Problem[];
}

export type Params = Record<string, number | string>;
