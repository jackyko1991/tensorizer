// Every `loops.*` page-text key the For loop vs tensor tab uses exists in English and in every translation.
// Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const SOURCES = ["pages/LoopsTab.tsx", "components/LoopVsTensorAnim.tsx", "components/HwChip.tsx"];
const CASES = ["dot", "pairwise", "matmul", "pendulum", "diffusion", "mc_pi"];

function usedKeys(): Set<string> {
  const keys = new Set<string>();
  for (const f of SOURCES)
    for (const m of read(f).matchAll(/["'`](loops\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  // keys built from template strings
  for (const c of CASES) {
    keys.add(`loops.${c}.title`);
    keys.add(`loops.${c}.text`);
    keys.add(`loops.anim.${c}.loop`);
    keys.add(`loops.anim.${c}.tensor`);
  }
  for (const c of ["pairwise", "pendulum", "diffusion"]) keys.add(`loops.eq.${c}.note`);
  for (const h of ["vendor", "unit", "since", "prec", "via"]) keys.add(`loops.hw.table.${h}`);
  for (const k of ["interp", "memory", "batch", "gpu"]) keys.add(`loops.why.${k}`);
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the tab's loops.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 40, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
});
