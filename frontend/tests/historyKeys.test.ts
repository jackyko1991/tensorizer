// Every `history.*` page-text key the Tensor history tab uses exists in English and in every translation, and every
// card's picture is in public/history/ with a credit in CREDITS.md. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import { HISTORY_CARDS } from "../src/historyCards.ts";

const root = new URL("../src/", import.meta.url);
const pub = new URL("../public/history/", import.meta.url);

function usedKeys(): Set<string> {
  const keys = new Set<string>();
  for (const m of readFileSync(new URL("pages/HistoryConcept.tsx", root), "utf-8").matchAll(/["'`](history\.[A-Za-z0-9_.]+)["'`]/g))
    keys.add(m[1]);
  for (const p of [1, 2, 3]) for (const f of ["label", "title", "sub"]) keys.add(`history.part${p}.${f}`);
  for (const p of [1, 2, 3]) keys.add(`history.introPart${p}`);
  for (const c of HISTORY_CARDS) for (const f of ["title", "alt", "p1", "p2"]) keys.add(`history.${c.id}.${f}`);
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the tab's history.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size > 100, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
    // no em dashes (site style), no control characters (a lost backslash turns "\times" into TAB + "imes")
    const bad = [...keys].filter((k) => /[\u2014\u0000-\u001f]/.test(text[k] ?? ""));
    assert.deepEqual(bad, [], `${f}: em dash or control character in ${bad.join(", ")}`);
  }
});

test("cards: unique ids, a picture or a drawing each, credited photos, 1-3 https sources", () => {
  const ids = HISTORY_CARDS.map((c) => c.id);
  assert.equal(new Set(ids).size, ids.length);
  const credits = readFileSync(new URL("CREDITS.md", pub), "utf-8");
  for (const c of HISTORY_CARDS) {
    assert.ok(!!c.photo !== !!c.drawing, `${c.id}: exactly one of photo / drawing`);
    if (c.photo) {
      assert.ok(existsSync(new URL(c.photo.file, pub)), `${c.id}: missing ${c.photo.file}`);
      assert.ok(credits.includes(c.photo.file), `${c.id}: ${c.photo.file} not in CREDITS.md`);
      assert.match(c.photo.licence, /^(Public domain|CC0|CC BY(-SA)? [\d.]+)$/, `${c.id}: licence ${c.photo.licence}`);
      assert.match(c.photo.source, /^https:\/\/commons\.wikimedia\.org\/wiki\/File:/);
    }
    assert.ok(c.sources.length >= 1 && c.sources.length <= 3, `${c.id}: ${c.sources.length} sources`);
    for (const s of c.sources) assert.match(s.url, /^https:\/\//, `${c.id}: ${s.url}`);
  }
});
