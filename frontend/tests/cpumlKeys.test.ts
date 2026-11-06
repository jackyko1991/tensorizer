// Every `cpuml.*` page-text key the "Migrating CPU-only ML to tensors" tab uses exists in English and in every translation.
// Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const SRC = "pages/CpumlTab.tsx";
const CASES = ["pca", "linsvm", "ksvm", "logreg", "knn", "kmeans"];
const SECTIONS = ["why", ...CASES, "umap", "cuml", "bench"];
const METRICS = ["evr_sum", "evr1", "evr_maxdiff", "min_cos10", "seconds", "test_acc", "objective", "pred_match", "coef_rel_diff", "dual", "n_sv", "inertia", "label_match"];
const LINKS = ["docs", "accel", "compat", "api", "install", "github", "sklearn"];

function usedKeys(): Set<string> {
  const keys = new Set<string>(["skills.tab.cpuml"]);
  for (const m of read(SRC).matchAll(/["'`](cpuml\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  // keys built from template strings
  for (const s of SECTIONS) keys.add(`cpuml.${s}.title`);
  for (const c of CASES) for (const k of ["text", "changes", "result", "short"]) keys.add(`cpuml.${c}.${k}`);
  for (const k of METRICS) keys.add(`cpuml.metric.${k}`);
  for (const k of ["sklearn", "cpu", "cuda"]) keys.add(`cpuml.agree.col.${k}`);
  for (const k of ["sklearn", "gain", "recipe", "honest"]) keys.add(`cpuml.why.${k}`);
  for (const k of ["knn", "sigma", "union", "forces"]) keys.add(`cpuml.umap.step.${k}`);
  for (const k of ["trust", "cont", "recall", "acc", "seconds"]) keys.add(`cpuml.umap.col.${k}`);
  for (const k of ["forces", "autograd", "umap_learn", "pca"]) keys.add(`cpuml.umap.row.${k}`);
  for (const k of LINKS) keys.add(`cpuml.cuml.link.${k}`);
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the tab lists the sections, cases, metrics and links it builds keys for", () => {
  const src = read(SRC);
  assert.match(src, new RegExp(`SECTIONS = \\[${SECTIONS.map((s) => `"${s}"`).join(", ")}\\]`));
  assert.match(src, new RegExp(`CASES = \\[${CASES.map((s) => `"${s}"`).join(", ")}\\]`));
  for (const k of METRICS) assert.match(src, new RegExp(`key: "${k}"`), k);
  for (const k of LINKS) assert.match(src, new RegExp(`key: "${k}"`), k);
});

test("the tab's cpuml.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 100, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
});

test("no em dashes in the tab's text, in any language", () => {
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const bad = Object.entries(text).filter(([k, v]) => (k.startsWith("cpuml.") || k === "skills.tab.cpuml") && /—|――/.test(v));
    assert.deepEqual(bad.map(([k]) => k), [], `${f}: em dash in ${bad.map(([k]) => k).join(", ")}`);
  }
});

test("documentation links open in a new tab, over https", () => {
  const src = read(SRC);
  assert.match(src, /target="_blank" rel="noopener noreferrer"/);
  for (const m of src.matchAll(/url: "([^"]+)"/g)) assert.match(m[1], /^https:\/\//, m[1]);
});

test("the benchmark JSON has every case, with agreement and timings", () => {
  const b = JSON.parse(read("content/cpuml_bench.json"));
  for (const c of CASES) {
    assert.ok(b.code[c]?.sklearn && b.code[c]?.tensor, `code of ${c}`);
    assert.ok(b.agreement[c]?.cpu, `agreement of ${c}`);
    assert.ok(Array.isArray(b.cases[c]) && b.cases[c].length > 0, `timings of ${c}`);
  }
  for (const r of ["forces", "autograd", "umap_learn", "pca"]) assert.ok(b.umap.rows[r], r);
});
