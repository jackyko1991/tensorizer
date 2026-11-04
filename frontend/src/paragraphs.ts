/** Split a problem's notes into readable paragraphs, in any language, without editing the library text.
 *
 *  1. A blank line always starts a paragraph.
 *  2. A sentence that opens with a numbered step ((1), （1）, 1.), a postulate label (P0 …) or a short "Label:" starts
 *     one too.
 *  3. Otherwise sentences are grouped until a paragraph reaches about TARGET characters.
 *  Sentence ends are ". ! ?" followed by a space (Latin scripts), "。！？" (CJK) or "।" (Devanagari), never inside
 *  $math$ or `code`, and not after common abbreviations (e.g., et al., Phys. Rev.). */

const TARGET = 420;
const ABBREVIATIONS =
  /(?:\b(?:e\.g|i\.e|et al|cf|vs|approx|Fig|Figs|Eq|Eqs|Ref|Refs|No|Vol|pp|Ch|Sec|Phys|Rev|Lett|Astrophys|Astron|Proc|Nat|Sci|Math|Ann|Commun|Mon|Not|Soc|Acad|Natl|J|Dr|Prof|St|Jr|Sr|al|ca|c)\.|\b[A-Z]\.)$/;
const OPENS_SECTION = /^(?:\(\d+\)|（\d+）|\d+\.\s|P\d+\s*[(（]|[A-Z][A-Za-z -]{1,28}:\s|[^\s$`]{1,10}[:：])/;

/** Positions just after each sentence end in `text` (outside $…$ and `…`). */
function sentenceEnds(text: string): number[] {
  const ends: number[] = [];
  let inMath = false;
  let inCode = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "$" && text[i - 1] !== "\\") inMath = !inMath;
    else if (c === "`") inCode = !inCode;
    if (inMath || inCode) continue;
    if ("。！？।".includes(c)) {
      ends.push(i + 1);
    } else if (".!?".includes(c) && text[i + 1] === " ") {
      const before = text.slice(Math.max(0, i - 12), i + 1);
      if (c === "." && ABBREVIATIONS.test(before)) continue;
      if (c === "." && /\d$/.test(text.slice(0, i)) && /^ \d/.test(text.slice(i + 1))) continue; // 3. 5 (numbers)
      ends.push(i + 2);
    }
  }
  return ends;
}

function splitLong(block: string): string[] {
  const cuts = sentenceEnds(block);
  const sentences: string[] = [];
  let start = 0;
  for (const end of cuts) {
    sentences.push(block.slice(start, end).trim());
    start = end;
  }
  if (start < block.length) sentences.push(block.slice(start).trim());
  const out: string[] = [];
  let current = "";
  for (const s of sentences.filter(Boolean)) {
    const section = OPENS_SECTION.test(s);
    if (current && (section || current.length >= TARGET)) {
      out.push(current);
      current = "";
    }
    current = current ? `${current}${/[。！？]$/.test(current) ? "" : " "}${s}` : s;
  }
  if (current) out.push(current);
  return out;
}

export function paragraphs(text: string): string[] {
  return text
    .replace(/\r\n/g, "\n")
    .split(/\n\s*\n/)
    .map((b) => b.replace(/\s*\n\s*/g, " ").trim())
    .filter(Boolean)
    .flatMap(splitLong); // splits only at numbered steps / labels, or once a paragraph is long
}
