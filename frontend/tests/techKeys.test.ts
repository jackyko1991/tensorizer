// Every `skills.anim.*` page-text key the Techniques animations use exists in English and in every translation,
// and every Techniques card has an animation. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import { TECH_CASES, stepCount } from "../src/techAnim.ts";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");

function usedKeys(): Set<string> {
  const keys = new Set<string>();
  for (const m of read("components/TechniqueAnim.tsx").matchAll(/["'`](skills\.anim\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  for (const c of TECH_CASES) {
    keys.add(`skills.anim.${c}.caption`);
    for (let k = 0; k < stepCount(c); k++) keys.add(`skills.anim.${c}.s${k}`);
  }
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the Techniques animations' skills.anim.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size >= 50, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
});

test("every Techniques card has an animation case", () => {
  const ids = [...read("pages/SkillsPage.tsx").matchAll(/^\s{4}id: "(\w+)",$/gm)].map((m) => m[1]);
  assert.ok(ids.length >= 11, `cards: ${ids}`);
  assert.deepEqual(ids, TECH_CASES);
});
