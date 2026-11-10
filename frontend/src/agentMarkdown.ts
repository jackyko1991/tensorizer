/** How the AI Assistant's answers are split for display (no React here, so it is unit-tested with node --test).
 *
 *  Agents write Markdown with code fences and maths in several notations: Claude mostly `$…$` / `$$…$$`, Codex
 *  and others `\(…\)` / `\[…\]`. An answer is cut into code blocks, display maths and Markdown text (whose inline
 *  maths is normalised to `$…$` for RichText). A fence still open at the end (an answer that is streaming in) is
 *  shown as code so far. */

export type Segment =
  | { kind: "code"; lang: string; code: string }
  | { kind: "math"; tex: string }
  | { kind: "md"; text: string };

const FENCE = /```([\w+#.-]*)[^\n]*\n?([\s\S]*?)(?:```|$(?![\s\S]))/g;
const DISPLAY = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g;
const INLINE = /\\\(([\s\S]+?)\\\)/g;

export function parseAnswer(text: string): Segment[] {
  const out: Segment[] = [];
  let at = 0;
  for (const m of text.matchAll(FENCE)) {
    if (m.index! > at) prose(text.slice(at, m.index), out);
    out.push({ kind: "code", lang: m[1].toLowerCase(), code: m[2].replace(/\n$/, "") });
    at = m.index! + m[0].length;
  }
  if (at < text.length) prose(text.slice(at), out);
  return out;
}

function prose(text: string, out: Segment[]) {
  let at = 0;
  for (const m of text.matchAll(DISPLAY)) {
    if (m.index! > at) md(text.slice(at, m.index), out);
    out.push({ kind: "math", tex: (m[1] ?? m[2]).trim() });
    at = m.index! + m[0].length;
  }
  if (at < text.length) md(text.slice(at), out);
}

function md(text: string, out: Segment[]) {
  const clean = text.replace(INLINE, (_, tex: string) => `$${tex.trim()}$`);
  if (clean.trim()) out.push({ kind: "md", text: clean });
}

/** The fence languages highlighted as Python (an unlabelled fence too: the answers are about Python code). */
export const isPython = (lang: string) => ["", "python", "py", "python3", "pycon"].includes(lang);

/** The text of an answer for the clipboard: the agent's own Markdown, as written. */
export function copyText(blocks: { kind: string; text?: string }[]): string {
  return blocks
    .filter((b) => b.kind === "text" && b.text)
    .map((b) => b.text!.trim())
    .join("\n\n");
}
