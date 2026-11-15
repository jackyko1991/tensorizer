// Coding Skills ▸ Mixed precision: every mixed.* key it uses exists in English and every translation, no em dashes,
// and its sourced points are https-linked. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import { MIXED_PLAN } from "../src/chipsAnim.ts";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const SRC = read("pages/MixedTab.tsx") + read("components/MixedFigures.tsx");
const SECTIONS = ["why", "formats", "accumulate", "recipe", "pytorch", "chips", "fp8", "science", "checklist"];

function usedKeys(): Set<string> {
  const keys = new Set<string>(["skills.tab.mixed"]);
  for (const m of SRC.matchAll(/["'`](mixed\.[A-Za-z0-9_.-]+)["'`]/g)) keys.add(m[1]);
  for (const s of SECTIONS) keys.add(`mixed.${s}.title`);
  for (const k of ["format", "bits", "layout", "max", "minNormal", "eps", "digits", "notes"]) keys.add(`mixed.formats.${k}`);
  for (const k of ["family", "low", "autocast", "scaling", "fp8", "notes"]) keys.add(`mixed.chips.${k}`);
  for (const k of ["measure", "bf16", "fp16", "keep", "check", "physics", "fp8"]) keys.add(`mixed.checklist.${k}`);
  for (const k of ["cuda", "tensor"]) keys.add(`mixed.cores.${k}`);
  for (const k of ["default", "deprecated", "preview"]) keys.add(`mixed.flags.${k}`);
  for (const k of ["in", "mul", "add", "out"]) keys.add(`mixed.fig.mac.${k}`);
  for (const k of ["fp32", "s1", "s128"]) keys.add(`mixed.fig.loss.${k}`);
  for (const [id, plan] of Object.entries(MIXED_PLAN)) {
    for (const k of ["title", "caption", ...plan.labels]) keys.add(`mixed.anim.${id}.${k}`);
    plan.steps.forEach((_, i) => keys.add(`mixed.anim.${id}.s${i}`));
  }
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the tab lists its sections in this order", () => {
  assert.match(SRC, new RegExp(`SECTIONS = \\[${SECTIONS.map((s) => `"${s}"`).join(", ")}\\]`));
});

test("the tab's mixed.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 80, `found only ${keys.size} keys`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.slice(0, 20).join(", ")}${missing.length > 20 ? ` … (${missing.length})` : ""}`);
  }
});

test("no em dashes in the tab's text and code", () => {
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const bad = Object.keys(text).filter((k) => k.startsWith("mixed.") && /[—―]/.test(text[k]));
    assert.deepEqual(bad, [], `${f}: ${bad.join(", ")}`);
  }
  assert.doesNotMatch(SRC, /—/);
});

test("mixed-precision points and table rows are sourced over https", () => {
  const m = JSON.parse(read("content/chips/mixed.json")) as { points: { id: string; source: string; topic: string }[]; table: { family: string; source: string }[] };
  assert.ok(m.points.length >= 30);
  const topics = new Set(m.points.map((p) => p.topic));
  for (const t of ["method", "pytorch", "bf16", "tpu", "amd", "intel", "apple", "ascend", "fp8", "science"]) assert.ok(topics.has(t), t);
  for (const p of m.points) assert.match(p.source, /^https:\/\//, p.id);
  for (const r of m.table) assert.match(r.source, /^https:\/\//, r.family);
});

test("data overlays: every key is an English data key and keeps its code and math spans", () => {
  const mixed = JSON.parse(read("content/chips/mixed.json")) as { points: { id: string; text: string }[]; table: Record<string, string>[] };
  const hist = JSON.parse(read("content/chips/history.json")) as { facts: { id: string; text: string }[] };
  const libs = JSON.parse(read("content/chips/libraries.json")) as { precisionDefaults: { text: string }[] };
  const formats = JSON.parse(read("content/chips/formats.json")) as { id: string; notes: string }[];
  const en: Record<string, string> = {};
  for (const p of mixed.points) en[`point.${p.id}`] = p.text;
  mixed.table.forEach((r, i) => ["lossScaling", "fp8", "notes"].forEach((f) => (en[`mixtable.${i}.${f}`] = r[f])));
  for (const f of hist.facts) en[`fact.${f.id}`] = f.text;
  libs.precisionDefaults.forEach((d, i) => (en[`default.${i}`] = d.text));
  for (const f of formats) en[`format.${f.id}.notes`] = f.notes;
  const spans = (s: string) => [...(s.match(/`[^`]+`/g) ?? []), ...(s.match(/\$[^$]+\$/g) ?? [])].sort();
  const dir = new URL("content/chips/i18n/", root);
  const files = readdirSync(dir).filter((f) => f.endsWith(".json"));
  assert.ok(files.length >= 9, `${files.length} overlays`);
  for (const f of files) {
    const tr = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    for (const [k, v] of Object.entries(tr)) {
      if (!(k in en)) continue; // chips-tab keys are checked by chipsKeys
      assert.deepEqual(spans(v), spans(en[k]), `${f} ${k}`);
      assert.doesNotMatch(v, /[—―]/, `${f} ${k}`);
    }
    const mixedKeys = Object.keys(en).filter((k) => !k.startsWith("fact.") || k.match(/^fact\.(tc-accumulate|fp8-accumulate|kahan|summation)/));
    const missing = mixedKeys.filter((k) => !tr[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.slice(0, 10).join(", ")}`);
  }
});
