// Coding Skills ▸ Chips: every chips.* page-text key it uses exists in English and every translation, and its data
// (content/chips/*.json, merged by scripts/chips_data.py) is sourced and consistent. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import { CHIPS_PLAN } from "../src/chipsAnim.ts";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const SRC = read("pages/ChipsTab.tsx");
const json = (p: string) => JSON.parse(read(p));
const SECTIONS = ["why", "history", "table", "precision", "under", "libs", "china", "arm", "desktop", "measure", "choose"];
const FAMILIES = ["nvidia-gpu", "amd-gpu", "intel-gpu", "x86-cpu", "arm-cpu", "apple-gpu", "apple-ane", "qualcomm", "google-tpu", "aws-trainium", "huawei-ascend", "other-china", "edge-npu"];

function usedKeys(): Set<string> {
  const keys = new Set<string>(["skills.tab.chips"]);
  for (const m of SRC.matchAll(/["'`](chips\.[A-Za-z0-9_.-]+)["'`]/g)) keys.add(m[1]);
  for (const s of SECTIONS) keys.add(`chips.${s}.title`);
  for (const f of FAMILIES) keys.add(`chips.fam.${f}`);
  for (const c of ["igpu", "dgpu", "soc", "accelerator", "cpu", "board", "npu"]) keys.add(`chips.class.${c}`);
  for (const l of ["official", "preview", "community", "none"]) keys.add(`chips.level.${l}`);
  for (const k of ["library", "chip", "standard", "policy"]) keys.add(`chips.history.${k}`);
  for (const k of ["power", "bandwidth", "units", "ecosystem"]) keys.add(`chips.why.${k}`);
  for (const k of ["laptop", "mac", "board", "nvidia", "cloud", "china", "physics"]) keys.add(`chips.choose.${k}`);
  for (const k of ["memoryType", "matrixUnit", "formats", "blas", "kernelLangs", "interconnect", "fp64", "fp32", "bf16", "fp8", "int8", "npu"]) keys.add(`chips.detail.${k}`);
  for (const k of ["family", "blas", "langs", "best", "avoid"]) keys.add(`chips.under.${k}`);
  for (const k of ["a", "b"]) keys.add(`chips.decode.${k}`);
  for (const k of ["title", "caption", "chip0", "chip1", "chip2", "note0", "note1", "note2", "cpu", "gpu", "npu", "io", "cache", "schematic"]) keys.add(`chips.budget.${k}`);
  for (const k of ["mac", "amd", "dgx", "rtx", "chip", "memory", "bandwidth", "fits", "decode", "power"]) keys.add(`chips.desktop.${k}`);
  for (const [id, plan] of Object.entries(CHIPS_PLAN)) {
    for (const k of ["title", "caption", ...plan.labels]) keys.add(`chips.anim.${id}.${k}`);
    plan.steps.forEach((_, i) => keys.add(`chips.anim.${id}.s${i}`));
  }
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the tab lists its sections in this order", () => {
  assert.match(SRC, new RegExp(`SECTIONS = \\[${SECTIONS.map((s) => `"${s}"`).join(", ")}\\]`));
});

test("the tab's chips.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 120, `found only ${keys.size} keys`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.slice(0, 20).join(", ")}${missing.length > 20 ? ` … (${missing.length})` : ""}`);
  }
});

test("no em dashes in the tab's text and code", () => {
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const bad = Object.keys(text).filter((k) => k.startsWith("chips.") && /[—―]/.test(text[k]));
    assert.deepEqual(bad, [], `${f}: ${bad.join(", ")}`);
  }
  assert.doesNotMatch(SRC, /—/);
});

interface ChipRec {
  id: string;
  class: string;
  sources: Record<string, string>;
  derived: string[];
  [k: string]: unknown;
}
const NUMERIC = ["bandwidthGBs", "fp64TF", "fp64TensorTF", "fp32TF", "tf32TF", "bf16TF", "fp16TF", "fp8TF", "fp6TF", "fp4TF", "int8TOPS", "npuTOPS", "powerW", "memoryGB"];

test("chips: unique ids, every number sourced (or derived and explained), https links", () => {
  const { chips } = json("content/chips/chips.json") as { chips: ChipRec[] };
  assert.ok(chips.length >= 40, `${chips.length} chips`);
  const ids = chips.map((c) => c.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const c of chips) {
    const urls = Object.values(c.sources);
    for (const u of urls) assert.match(u, /^https:\/\//, `${c.id}: ${u}`);
    const hasNumber = NUMERIC.some((k) => c[k] != null);
    if (hasNumber) assert.ok(urls.length > 0, `${c.id} has numbers but no source`);
    if (c.derived.length) assert.ok(String(c.notes).length > 20, `${c.id}: derived figures need notes`);
  }
});

test("libraries: every library has a cell for every family, levels are known, sources https", () => {
  const { libraries, under } = json("content/chips/libraries.json") as {
    libraries: { id: string; cells: Record<string, { level: string; source?: string }> }[];
    under: { family: string }[];
  };
  for (const lib of libraries)
    for (const f of FAMILIES) {
      const c = lib.cells[f];
      assert.ok(c, `${lib.id} × ${f}`);
      assert.ok(["official", "preview", "community", "none"].includes(c.level), `${lib.id} × ${f}: ${c.level}`);
      if (c.source) assert.match(c.source, /^https:\/\//, `${lib.id} × ${f}`);
    }
  assert.deepEqual(under.map((u) => u.family).sort(), [...FAMILIES].sort());
});

test("timeline: dated, sourced, in order of years", () => {
  const { timeline } = json("content/chips/history.json") as { timeline: { id: string; year: number; source: string; kind: string }[] };
  assert.ok(timeline.length >= 30);
  for (const e of timeline) {
    assert.match(e.source, /^https:\/\//, e.id);
    assert.ok(e.year >= 1970 && e.year <= 2026, e.id);
    assert.ok(["library", "chip", "standard", "policy"].includes(e.kind), e.id);
  }
});
