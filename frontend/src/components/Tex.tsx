import katex from "katex";
import { ALIAS_INDEX, ALIAS_RE } from "../scientists";
import { useLayoutEffect, useMemo, useRef } from "react";

const MIN_SCALE = 0.6; // below this, a piece scrolls sideways instead of shrinking further

/**
 * Split a display equation into separate relations at top-level gaps — "\qquad", or "\quad"
 * right after a comma ("a = b,\quad c = d") — so they can wrap onto separate lines in a
 * narrow column. Braces are respected; "a \quad\Longrightarrow\quad b" stays together.
 */
function splitRelations(tex: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < tex.length; i++) {
    const ch = tex[i];
    const gap = ch === "\\" && depth === 0 ? gapAt(tex, i, tex.slice(start, i)) : 0;
    if (gap) {
      parts.push(tex.slice(start, i));
      start = i + gap;
      i += gap - 1;
    } else if (ch === "\\") {
      i++; // skip escaped character such as \{ or \}
    } else if (ch === "{") {
      depth++;
    } else if (ch === "}") {
      depth--;
    }
  }
  parts.push(tex.slice(start));
  return parts
    .map((p) =>
      p
        .trim()
        .replace(/^,\s*/, "")
        .replace(/(\\[,;:!]|\\quad|,|\s)+$/, ""),
    )
    .filter(Boolean);
}

/** Length of a relation-separating gap command at position i, or 0. */
function gapAt(tex: string, i: number, before: string): number {
  for (const cmd of ["\\qquad", "\\quad"]) {
    if (!tex.startsWith(cmd, i) || /[a-zA-Z]/.test(tex[i + cmd.length] ?? "")) continue;
    if (cmd === "\\qquad" || /,(\s|\\[,;:!])*$/.test(before)) return cmd.length;
  }
  return 0;
}

const render = (tex: string, display: boolean) =>
  katex.renderToString(tex, { displayMode: display, throwOnError: false });

export default function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  const pieces = useMemo(() => (display ? splitRelations(tex).map((t) => render(t, true)) : []), [tex, display]);
  const inline = useMemo(() => (display ? "" : render(tex, false)), [tex, display]);
  const ref = useRef<HTMLDivElement>(null);

  // Shrink any piece that is still wider than the container.
  useLayoutEffect(() => {
    const container = ref.current;
    if (!display || !container) return;
    const fit = () => {
      const width = container.clientWidth - 16; // room for the .katex-display padding
      container.querySelectorAll<HTMLElement>(":scope > .tex-piece").forEach((piece) => {
        piece.style.fontSize = "";
        const inner = piece.querySelector<HTMLElement>(".katex-display > .katex");
        if (inner && inner.scrollWidth > width)
          piece.style.fontSize = `${Math.max(MIN_SCALE, width / inner.scrollWidth)}em`;
      });
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(container);
    return () => observer.disconnect();
  }, [pieces, display]);

  if (!display) return <span dangerouslySetInnerHTML={{ __html: inline }} />;
  return (
    <div ref={ref} className="flex flex-wrap items-center justify-center gap-x-10">
      {pieces.map((html, i) => (
        <div key={i} className="tex-piece min-w-0 max-w-full" dangerouslySetInnerHTML={{ __html: html }} />
      ))}
    </div>
  );
}

/** Inline math reduced to plain text for <option> labels and alt text: "$k_{on}$" → "k_on". */
export function plainText(text: string): string {
  return text.replace(/(?<!\\)\$(.+?)(?<!\\)\$/gs, (_, m: string) =>
    m
      .replace(/\\(mathrm|text|mathbf|boldsymbol)\{([^}]*)\}/g, "$2")
      .replace(/\\([a-zA-Z]+)/g, "$1")
      .replace(/[{}]/g, ""),
  );
}

/** Plain text with inline math: every `$...$` span is typeset with KaTeX (`\$` is a literal dollar), `code` is monospaced. */
/** Wrap the first mention of each historical scientist (outside `code`) in a dotted-underline span that the
 *  SciCard easter egg reacts to. Plain text in, HTML out (the text is already escaped). */
function markPeople(html: string): string {
  const seen = new Set<string>();
  return html
    .split(/(`[^`]+`)/)
    .map((part, i) =>
      i % 2
        ? part
        : part.replace(ALIAS_RE, (name) => {
            const s = ALIAS_INDEX.get(name);
            if (!s || seen.has(s.id)) return name;
            seen.add(s.id);
            return `<span data-sci="${s.id}" class="cursor-help border-b border-dotted border-slate-400">${name}</span>`;
          }),
    )
    .join("");
}

export function RichText({ text, people = false }: { text: string; people?: boolean }) {
  const html = useMemo(() => {
    const escape = (s: string) => {
      const e = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      return people ? markPeople(e) : e;
    };
    return text
      .split(/(?<!\\)\$(.+?)(?<!\\)\$/s)
      .map((part, i) =>
        i % 2
          ? render(part, false)
          : escape(part.replace(/\\\$/g, "$"))
              .replace(
                /`([^`]+)`/g,
                '<code class="rounded bg-gray-100 px-1 py-0.5 font-mono text-[0.85em] text-gray-800">$1</code>',
              )
              .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>"),
      )
      .join("");
  }, [text, people]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
