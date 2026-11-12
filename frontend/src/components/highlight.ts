/**
 * Syntax colours for short inline `code` spans in RichText (a copy of learnkit/web/src/highlight.ts until Tensorizer
 * adopts learnkit). A small, language-agnostic tokenizer that suits the
 * sites' languages (Python, Lean, shell, tinyNotation): strings, numbers, keywords, function names before "(".
 * Input and output are HTML-escaped text (only &, <, > are escaped, so quotes are literal).
 */

const KEYWORDS = new Set(
  (
    "def return import from as for in if elif else while class lambda None True False and or not with yield try " +
    "except finally raise pass break continue global async await is del " +
    "theorem lemma example def by fun let have show calc intro exact apply rw simp sorry where match do open namespace end " +
    "const let var function new typeof"
  ).split(" "),
);

const TOKEN =
  // strings: "…", or '…' when the quote does not follow a letter (so tinyNotation c' d' stays plain)
  /("[^"\n]*"|(?<![\w'])'[^'\n]*'(?![\w]))|(&(?:amp|lt|gt);)|(\b\d+(?:\.\d+)?\b)|\b([A-Za-z_][\w.]*)(?=\()|\b([A-Za-z_]\w*)\b/g;

const span = (cls: string, s: string) => `<span class="${cls}">${s}</span>`;

export function highlightCode(escaped: string): string {
  return escaped.replace(TOKEN, (m, str: string, ent: string, num: string, call: string, word: string) => {
    if (str) return span("text-emerald-700", str);
    if (ent) return m;
    if (num) return span("text-amber-700", num);
    if (call) return span("text-blue-700", call);
    if (word && KEYWORDS.has(word)) return span("text-violet-700", word);
    return m;
  });
}
