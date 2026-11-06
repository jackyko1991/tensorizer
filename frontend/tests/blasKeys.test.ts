// Every `blas.*` page-text key the BLAS tab uses exists in English and in every translation.
// Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const SOURCES = ["pages/BlasTab.tsx"];
const SECTIONS = ["what", "libs", "which", "repro", "bench", "method"];
const LIBS = ["netlib", "openblas", "mkl", "aocl", "blis", "accelerate", "cublas", "rocblas", "onednn"];
const SERIES = ["cpu_float32_all", "cpu_float32_1t", "cpu_float64_all", "cpu_float64_1t", "cuda_float32", "cuda_float64"];

function usedKeys(): Set<string> {
  const keys = new Set<string>(["skills.tab.blas"]);
  for (const f of SOURCES)
    for (const m of read(f).matchAll(/["'`](blas\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  // keys built from template strings
  for (const s of SECTIONS) keys.add(`blas.${s}.title`);
  for (const l of LIBS) for (const c of ["hw", "notes", "via"]) keys.add(`blas.lib.${l}.${c}`);
  for (const s of SERIES) keys.add(`blas.series.${s}`);
  for (const h of ["lib", "hw", "licence", "notes", "via"]) keys.add(`blas.libs.table.${h}`);
  for (const h of ["level", "op", "routines", "flops", "data", "ai"]) keys.add(`blas.levels.${h}`);
  for (const k of ["blocking", "kernel", "threads"]) keys.add(`blas.what.${k}`);
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the BLAS tab lists every library and section it builds keys for", () => {
  const src = read("pages/BlasTab.tsx");
  for (const l of LIBS) assert.match(src, new RegExp(`id: "${l}"`), l);
  for (const s of SERIES) assert.match(src, new RegExp(`id: "${s}"`), s);
  assert.match(src, new RegExp(`SECTIONS = \\[${SECTIONS.map((s) => `"${s}"`).join(", ")}\\]`));
});

test("the tab's blas.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 80, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
});

test("library links open the official docs in a new tab", () => {
  const src = read("pages/BlasTab.tsx");
  assert.match(src, /target="_blank" rel="noopener noreferrer"/);
  for (const m of src.matchAll(/url: "([^"]+)"/g)) assert.match(m[1], /^https:\/\//, m[1]);
});
