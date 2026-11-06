// Every `maths.*` page-text key the Tensor maths tab uses exists in English and in every translation.
// Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const SECTIONS = ["tensor", "einsum", "penrose", "topo", "roof"];
const FEYNMAN = ["plain", "analogy", "precise", "example"];

function usedKeys(): Set<string> {
  const keys = new Set<string>();
  for (const f of ["pages/MathsConcept.tsx", "pages/ConceptPage.tsx"])
    for (const m of read(f).matchAll(/["'`](maths\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  // keys built from template strings
  for (const s of [...SECTIONS, "recap"]) keys.add(`maths.${s}.title`);
  for (const s of SECTIONS) for (const f of FEYNMAN) keys.add(`maths.${s}.${f}`);
  for (const f of FEYNMAN) keys.add(`maths.label.${f}`);
  for (let i = 1; i <= 5; i++) keys.add(`maths.recap.mis${i}`);
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the tab's maths.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 60, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
});

test("the machine sentence keeps its {cpu} and {gpu} placeholders in every language", () => {
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const s = text["maths.roof.machine"];
    assert.ok(s.includes("{cpu}") && s.includes("{gpu}"), f);
  }
});
