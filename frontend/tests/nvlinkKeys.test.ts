// Every `nvlink.*` page-text key the NVLink tab uses exists in English and in every translation.
// Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const SRC = read("pages/NvlinkTab.tsx");
const SECTIONS = ["why", "gens", "topo", "coll", "code", "practice"];
const SYSTEMS = ["dgx1", "dgx2", "nvl72", "geforce"];
const COLLECTIVES = ["allreduce", "allgather", "reducescatter", "broadcast"];

function usedKeys(): Set<string> {
  const keys = new Set<string>(["skills.tab.nvlink"]);
  for (const m of SRC.matchAll(/["'`](nvlink\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  // keys built from template strings
  for (const s of SECTIONS) keys.add(`nvlink.${s}.title`);
  for (const s of SYSTEMS) for (const c of ["tip", "text"]) keys.add(`nvlink.topo.${s}.${c}`);
  for (const c of COLLECTIVES) for (const f of ["name", "what"]) keys.add(`nvlink.coll.${c}.${f}`);
  for (const h of ["gen", "gpu", "links", "perDir", "total", "mem", "src"]) keys.add(`nvlink.gens.table.${h}`);
  for (const h of ["op", "what", "shape", "factor"]) keys.add(`nvlink.coll.table.${h}`);
  for (const k of ["dp", "tp", "pp"]) keys.add(`nvlink.why.${k}`);
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the NVLink tab lists every section, system and collective it builds keys for", () => {
  assert.match(SRC, new RegExp(`SECTIONS = \\[${SECTIONS.map((s) => `"${s}"`).join(", ")}\\]`));
  for (const s of SYSTEMS) assert.match(SRC, new RegExp(`id: "${s}"`), s);
  for (const c of COLLECTIVES) assert.match(SRC, new RegExp(`id: "${c}"`), c);
});

test("the tab's nvlink.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 80, `found only ${keys.size} keys`);
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
    const bad = Object.keys(text).filter((k) => (k.startsWith("nvlink.") || k === "skills.tab.nvlink") && /—/.test(text[k]));
    assert.deepEqual(bad, [], `${f}: ${bad.join(", ")}`);
  }
  assert.doesNotMatch(SRC, /—/);
});

test("sources open in a new tab over https", () => {
  assert.match(SRC, /target="_blank" rel="noopener noreferrer"/);
  const urls = [...SRC.matchAll(/url: "([^"]+)"/g)].map((m) => m[1]);
  assert.ok(urls.length >= 20);
  for (const u of urls) assert.match(u, /^https:\/\//, u);
});
