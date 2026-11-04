// Notes → paragraphs. Run: node --test frontend/tests (Node ≥ 23 strips the types).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { paragraphs } from "../src/paragraphs.ts";

test("blank lines always split; short text stays one paragraph", () => {
  assert.deepEqual(paragraphs("One.\n\nTwo."), ["One.", "Two."]);
  assert.deepEqual(paragraphs("A short note. Still short."), ["A short note. Still short."]);
});

test("numbered steps and labels open a paragraph; math and abbreviations never split", () => {
  const long = "x".repeat(300);
  const text = `${long} first. (1) Step one, see Phys. Rev. Lett. 98 and $a. B$ inline. (2) Step two. Data: numbers here.`;
  const out = paragraphs(text);
  assert.ok(out.some((p) => p.startsWith("(1) Step one, see Phys. Rev. Lett. 98 and $a. B$ inline.")));
  assert.ok(out.some((p) => p.startsWith("(2) Step two.")));
  assert.ok(out.some((p) => p.startsWith("Data: numbers here.")));
});

test("every library note splits into readable paragraphs in every language", () => {
  const manifest = JSON.parse(readFileSync(new URL("../public/generated/manifest.json", import.meta.url), "utf-8"));
  let checked = 0;
  for (const problem of manifest.problems) {
    const notes = [problem.notes, ...Object.values(problem.i18n ?? {}).map((t: any) => t.notes)].filter(Boolean);
    for (const text of notes as string[]) {
      const out = paragraphs(text);
      assert.equal(out.join(" ").replace(/\s+/g, ""), text.replace(/\s+/g, ""), `${problem.id}: text lost`);
      for (const p of out) assert.ok(p.length < 1400, `${problem.id}: a paragraph of ${p.length} characters`);
      checked++;
    }
  }
  assert.ok(checked > 1000);
});
