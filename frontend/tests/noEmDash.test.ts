// The Concept, Coding Skills and Agent Skills pages use no em dashes (U+2014), horizontal bars (U+2015), Chinese
// "——" or spaced en dashes " – " in any language: punctuation is a colon, comma, parentheses or a new sentence.
// En dashes in ranges ("1884–1894", "2–3") are fine. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../src/", import.meta.url);
const DASH = /[—―]| – /;

// Every key prefix in pages.<lang>.json belongs to the Concept or Coding Skills page
const PREFIXES = ["concept.", "flow.", "lab.", "maths.", "history.", "skills.", "loops.", "blas.", "nvlink.", "cpuml."];

const SOURCES = [
  "pages/ConceptPage.tsx",
  "pages/DataflowConcept.tsx",
  "pages/MathsConcept.tsx",
  "pages/HistoryConcept.tsx",
  "pages/SkillsPage.tsx",
  "pages/LoopsTab.tsx",
  "pages/BlasTab.tsx",
  "pages/NvlinkTab.tsx",
  "pages/CpumlTab.tsx",
  "pages/AgentsPage.tsx",
  "components/TechniqueAnim.tsx",
  "components/LoopVsTensorAnim.tsx",
  "components/animKit.tsx",
  "components/HwChip.tsx",
  "components/TensorLab.tsx",
  "components/TranslationTable.tsx",
  "historyCards.ts",
  "nvlinkModel.ts",
  "cpumlChart.ts",
  "strings.skills.ts", // the Agent Skills page text (ag.*) in every language
];

const contentDir = new URL("content/", root);

test("pages.<lang>.json text has no em dashes", () => {
  const files = readdirSync(contentDir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));
  assert.ok(files.length >= 10, `languages: ${files}`);
  const bad: string[] = [];
  for (const f of files) {
    const text = JSON.parse(readFileSync(new URL(f, contentDir), "utf-8")) as Record<string, string>;
    for (const [k, v] of Object.entries(text)) {
      assert.ok(PREFIXES.some((p) => k.startsWith(p)), `${f}: unexpected key ${k} (add its prefix if it is on these pages)`);
      if (DASH.test(v)) bad.push(`${f} ${k}: ${v}`);
    }
  }
  assert.deepEqual(bad, []);
});

test("the benchmark notes shown on Coding Skills have no em dashes", () => {
  const files = readdirSync(contentDir).filter((f) => /_bench\.json$/.test(f));
  assert.ok(files.length >= 3, `bench files: ${files}`);
  for (const f of files) assert.ok(!DASH.test(readFileSync(new URL(f, contentDir), "utf-8")), f);
});

test("the pages' source files have no em dashes outside comments", () => {
  const bad: string[] = [];
  for (const f of SOURCES) {
    const code = readFileSync(new URL(f, root), "utf-8")
      .replace(/\/\*[\s\S]*?\*\//g, "") // block and JSX comments
      .replace(/(^|[\s;,{}()])\/\/.*$/gm, "$1"); // line comments (not "https://")
    code.split("\n").forEach((line, i) => {
      if (DASH.test(line)) bad.push(`${f}:${i + 1}: ${line.trim()}`);
    });
  }
  assert.deepEqual(bad, []);
});
