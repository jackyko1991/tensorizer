/**
 * A workspace problem's meta fields in three interchangeable forms: the edited object, meta.yaml text, and one
 * Markdown document (title, summary, equations, translation table, notes). Self-contained (type imports only) so
 * `node --test frontend/tests` can run it without a bundler.
 */
import type { Level, ParamSpec, Problem, TranslationRow } from "./types";

export interface MetaFields {
  title: string;
  level: Level;
  summary: string;
  animated: boolean;
  vector_form: string;
  index_form: string;
  translation: TranslationRow[];
  notes: string;
  params: Record<string, ParamSpec>;
  /** the user's own labels (Workspace), e.g. "homework 3" */
  tags?: string[];
}

/** The fields the Markdown document carries (level, animated and params stay inline / in meta.yaml). */
export const MD_KEYS = ["title", "summary", "vector_form", "index_form", "translation", "notes"] as const;
export type MdFields = Pick<MetaFields, (typeof MD_KEYS)[number]>;

export const pickMd = (f: MdFields): MdFields => ({
  title: f.title,
  summary: f.summary,
  vector_form: f.vector_form,
  index_form: f.index_form,
  translation: f.translation.map((r) => ({ symbol: r.symbol, code: r.code, shape: r.shape, meaning: r.meaning })),
  notes: f.notes,
});

/** meta.yaml as an object, in the conventional key order, keeping keys the page doesn't edit. */
export function metaObject(p: Problem): Record<string, unknown> {
  return {
    title: p.title,
    ...(p.meta_extra ?? {}),
    level: p.level,
    summary: p.summary,
    animated: p.animated,
    ...(p.tags?.length ? { tags: p.tags } : {}),
    vector_form: p.vector_form,
    index_form: p.index_form,
    translation: p.translation,
    notes: p.notes ?? "",
    params: p.params,
  };
}

// --- YAML (a small emitter for meta.yaml's shapes: mappings, lists of mappings, scalars) -------------------

const PLAIN = /^[A-Za-z][A-Za-z0-9 _.,()/-]*$/;
const RESERVED = /^(true|false|yes|no|on|off|null|y|n)$/i;

function scalar(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  const s = String(v);
  if (s === "") return '""';
  if (PLAIN.test(s) && !RESERVED.test(s) && !s.endsWith(" ")) return s;
  return `'${s.replace(/'/g, "''")}'`; // single quotes: backslashes (LaTeX) stay literal
}

const key = (k: string) => (/^[A-Za-z_][A-Za-z0-9_]*$/.test(k) ? k : scalar(k));
const isMap = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

function value(v: unknown, pad: string): string {
  if (typeof v === "string" && v.includes("\n")) {
    if (/^\s/.test(v)) return ` ${JSON.stringify(v)}`; // a block can't start with indentation
    return ` |-\n${v
      .split("\n")
      .map((l) => (l ? pad + "  " + l : ""))
      .join("\n")}`;
  }
  if (Array.isArray(v)) {
    if (!v.length) return " []";
    if (v.every((x) => !isMap(x) && !Array.isArray(x))) return ` [${v.map(scalar).join(", ")}]`;
    return (
      "\n" +
      v.map((item) => `${pad}  - ${mapping(isMap(item) ? item : { value: item }, pad + "    ").trimStart()}`).join("\n")
    );
  }
  if (isMap(v)) return Object.keys(v).length ? "\n" + mapping(v, pad + "  ") : " {}";
  return ` ${scalar(v)}`;
}

function mapping(obj: Record<string, unknown>, pad: string): string {
  return Object.entries(obj)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${pad}${key(k)}:${value(v, pad)}`)
    .join("\n");
}

export const toYaml = (obj: Record<string, unknown>) => mapping(obj, "") + "\n";

// --- Markdown ------------------------------------------------------------------------------------------

const SECTIONS = {
  "vector form": "vector_form",
  "index form": "index_form",
  translation: "translation",
  notes: "notes",
} as const;
const cell = (s: string) => s.replace(/\|/g, "\\|").replace(/\n/g, " ");

export function toMarkdown(f: MdFields): string {
  const eq = (tex: string) => (tex.trim() ? `$$\n${tex.trim()}\n$$` : "");
  const rows = f.translation.map(
    (r) =>
      `| ${r.symbol ? `$${cell(r.symbol)}$` : ""} | ${r.code ? `\`${cell(r.code)}\`` : ""} | ${cell(r.shape)} | ${cell(r.meaning)} |`,
  );
  return [
    `# ${f.title}`,
    f.summary.trim(),
    "## Vector form",
    eq(f.vector_form),
    "## Index form",
    eq(f.index_form),
    "## Translation",
    ["| symbol | code | shape | meaning |", "|---|---|---|---|", ...rows].join("\n"),
    "## Notes",
    f.notes.trim(),
  ]
    .filter(Boolean)
    .join("\n\n")
    .concat("\n");
}

export class MarkdownError extends Error {}

const unwrapMath = (s: string) =>
  s
    .trim()
    .replace(/^\$\$([\s\S]*)\$\$$/, "$1")
    .trim();

function splitRow(line: string): string[] {
  const body = line
    .trim()
    .replace(/^\|/, "")
    .replace(/(?<!\\)\|$/, "");
  return body.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
}

/** The Markdown document back into fields; throws MarkdownError (with a line number) when it can't. */
export function fromMarkdown(md: string): MdFields {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const first = lines.findIndex((l) => l.trim());
  const title = lines[first]?.match(/^#\s+(.*)$/);
  if (!title) throw new MarkdownError(`line ${first + 1}: start with "# title"`);
  const out: MdFields = {
    title: title[1].trim(),
    summary: "",
    vector_form: "",
    index_form: "",
    translation: [],
    notes: "",
  };
  let section: string = "summary";
  let start = first + 1;
  const flush = (end: number) => {
    const text = lines.slice(start, end).join("\n");
    if (section === "summary" || section === "notes") out[section] = text.trim();
    else if (section === "vector_form" || section === "index_form") out[section] = unwrapMath(text);
    else if (section === "translation") out.translation = table(lines, start, end);
  };
  for (let i = first + 1; i <= lines.length; i++) {
    const heading = i < lines.length ? lines[i].match(/^##\s+(.*)$/) : null;
    if (i < lines.length && !heading) {
      if (/^#\s/.test(lines[i])) throw new MarkdownError(`line ${i + 1}: only one "# title"; use "## " for sections`);
      continue;
    }
    flush(i);
    if (!heading) break;
    const name = heading[1].trim().toLowerCase() as keyof typeof SECTIONS;
    if (!(name in SECTIONS))
      throw new MarkdownError(
        `line ${i + 1}: unknown section "${heading[1].trim()}" (use Vector form, Index form, Translation, Notes)`,
      );
    section = SECTIONS[name];
    start = i + 1;
  }
  return out;
}

function table(lines: string[], start: number, end: number): TranslationRow[] {
  const rows: TranslationRow[] = [];
  let header = true;
  for (let i = start; i < end; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    if (!line.startsWith("|")) throw new MarkdownError(`line ${i + 1}: the Translation section holds only a table`);
    const cells = splitRow(line);
    if (header) {
      header = false;
      continue;
    }
    if (cells.every((c) => /^:?-+:?$/.test(c))) continue; // |---|---|
    if (cells.length !== 4)
      throw new MarkdownError(`line ${i + 1}: a row needs 4 cells (symbol | code | shape | meaning)`);
    const [symbol, code, shape, meaning] = cells;
    rows.push({ symbol: symbol.replace(/^\$(.*)\$$/, "$1"), code: code.replace(/^`(.*)`$/, "$1"), shape, meaning });
  }
  return rows;
}

// --- LaTeX (export only) -------------------------------------------------------------------------------

const TEX_SPECIAL: Record<string, string> = {
  "\\": "\\textbackslash{}",
  "{": "\\{",
  "}": "\\}",
  $: "\\$",
  "&": "\\&",
  "%": "\\%",
  "#": "\\#",
  _: "\\_",
  "~": "\\textasciitilde{}",
  "^": "\\textasciicircum{}",
};
const texEscape = (s: string) => s.replace(/[\\{}$&%#_~^]/g, (c) => TEX_SPECIAL[c]);

/** Prose with $math$, `code` and **bold** → LaTeX: math kept as written, everything else escaped. */
function texText(s: string): string {
  return s
    .split(/(?<!\\)\$(.+?)(?<!\\)\$/s)
    .map((part, i) =>
      i % 2
        ? `$${part}$`
        : part
            .replace(/\\\$/g, "$")
            .split(/(`[^`]+`|\*\*[^*]+\*\*)/)
            .map((bit) =>
              bit.startsWith("`") && bit.endsWith("`") && bit.length > 1
                ? `\\texttt{${texEscape(bit.slice(1, -1))}}`
                : bit.startsWith("**") && bit.endsWith("**") && bit.length > 4
                  ? `\\textbf{${texEscape(bit.slice(2, -2))}}`
                  : texEscape(bit),
            )
            .join(""),
    )
    .join("");
}

/** The formulation as a standalone LaTeX document (compile with xelatex or lualatex: the text is Unicode). */
export function toLatex(f: MdFields): string {
  const all = [f.title, f.summary, f.notes, ...f.translation.map((r) => r.meaning)].join(" ");
  const cjk = /[぀-ヿ㐀-鿿가-힯]/.test(all);
  const eq = (tex: string) => (tex.trim() ? `\\[\n${tex.trim()}\n\\]` : "");
  const rows = f.translation
    .filter((r) => r.symbol || r.code || r.shape || r.meaning)
    .map(
      (r) =>
        `${r.symbol ? `$${r.symbol}$` : ""} & ${r.code ? `\\texttt{${texEscape(r.code)}}` : ""} & ${texText(r.shape)} & ${texText(r.meaning)} \\\\`,
    );
  const col = (w: number, last = false) => `${last ? "" : ">{\\raggedright}"}p{${w}\\linewidth}`;
  return [
    "% Exported from Tensorizer. Compile with xelatex or lualatex (Unicode text).",
    "\\documentclass[11pt]{article}",
    "\\usepackage{amsmath,amssymb}",
    "\\usepackage{fontspec}",
    ...(cjk ? ["\\usepackage{xeCJK}"] : []),
    "\\usepackage{longtable,booktabs,array}",
    "\\usepackage[margin=2.2cm]{geometry}",
    "\\begin{document}",
    "",
    `\\section*{${texText(f.title)}}`,
    texText(f.summary.trim()),
    "",
    "\\subsection*{Vector form}",
    eq(f.vector_form),
    "",
    "\\subsection*{Index form}",
    eq(f.index_form),
    "",
    "\\subsection*{Translation}",
    `\\begin{longtable}{${col(0.17)}${col(0.3)}${col(0.13)}${col(0.32, true)}}`,
    "\\toprule",
    "symbol & code & shape & meaning \\tabularnewline",
    "\\midrule",
    ...rows.map((r) => r.replace(/ \\\\$/, " \\tabularnewline")),
    "\\bottomrule",
    "\\end{longtable}",
    ...(f.notes.trim() ? ["", "\\subsection*{Notes}", texText(f.notes.trim())] : []),
    "",
    "\\end{document}",
    "",
  ].join("\n");
}
