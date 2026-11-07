// Every `feynman.*` page-text key the Feynman technique tab uses exists in English and in every translation.
// Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
// mirrors the constants in pages/FeynmanConcept.tsx (keys built from template strings)
const SECTIONS = ["steps", "roots", "life", "site", "example", "pitfalls"];
const FEYNMAN = ["plain", "analogy", "precise", "example"];
const ROOTS = ["notebook", "create", "freshman", "lectures"];
const LIFE = ["losalamos", "diagrams", "nobel", "challenger"];
const SITE = ["maths", "ladder", "tdd"];

function usedKeys(): Set<string> {
  const keys = new Set<string>();
  for (const f of ["pages/FeynmanConcept.tsx", "pages/ConceptPage.tsx"])
    for (const m of read(f).matchAll(/["'`](feynman\.[A-Za-z0-9_.]+)["'`]/g))
      if (!m[1].endsWith(".webp")) keys.add(m[1]); // not the photo file name
  for (const s of SECTIONS) keys.add(`feynman.${s}.title`);
  for (const f of FEYNMAN) keys.add(`feynman.steps.${f}`);
  for (let n = 1; n <= 4; n++) {
    keys.add(`feynman.step${n}.title`);
    keys.add(`feynman.step${n}.text`);
    keys.add(`feynman.example.s${n}.title`);
    keys.add(`feynman.example.s${n}.text`);
  }
  for (const r of ROOTS) for (const p of ["title", "text", "how"]) keys.add(`feynman.roots.${r}.${p}`);
  for (const e of LIFE) for (const p of ["title", "text"]) keys.add(`feynman.life.${e}.${p}`);
  for (const c of SITE) for (const p of ["title", "text"]) keys.add(`feynman.site.${c}.${p}`);
  for (let n = 1; n <= 3; n++) keys.add(`feynman.pitfalls.p${n}`);
  for (let n = 1; n <= 6; n++) keys.add(`feynman.check.c${n}`);
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the tab's feynman.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 70, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
});

test("the template-string lists match the component", () => {
  const src = read("pages/FeynmanConcept.tsx");
  const list = (name: string) => JSON.parse(src.match(new RegExp(`${name} = (\\[[^\\]]*\\])`))![1]);
  assert.deepEqual(list("FEYNMAN_SECTIONS"), SECTIONS);
  assert.deepEqual(list("ROOTS"), ROOTS);
  assert.deepEqual(list("LIFE"), LIFE);
  assert.deepEqual(list("SITE"), SITE);
});

test("the photo credited on the page is in public/feynman/CREDITS.md", () => {
  const credits = readFileSync(new URL("../public/feynman/CREDITS.md", import.meta.url), "utf-8");
  assert.ok(credits.includes("feynman.webp") && credits.includes("Richard_Feynman_1959.png"));
  assert.ok(read("pages/FeynmanConcept.tsx").includes("Richard_Feynman_1959.png"));
});
