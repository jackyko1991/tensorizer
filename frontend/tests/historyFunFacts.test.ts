// The hidden "💡 Did you know?" fun facts on the Tensor history cards: every fact has its text in all 10 languages
// and at least one https source, the MythBusters fact embeds its video from youtube-nocookie.com, and no text uses an
// em dash.
// Run: node --test frontend/tests/*.test.ts
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import { HISTORY_CARDS } from "../src/historyCards.ts";

const dir = new URL("../src/content/", import.meta.url);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));
const facts = HISTORY_CARDS.flatMap((c) => (c.funFacts ?? []).map((f) => ({ card: c.id, ...f })));
// the labels of the badge, the note and the show-all switch
const UI = ["history.fun.title", "history.fun.showAll", "history.fun.videoTitle"];

test("there are fun facts, with unique ids, on Parts I and III at least", () => {
  assert.ok(facts.length >= 8, `only ${facts.length} facts`);
  const ids = facts.map((f) => f.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate fun-fact id");
  for (const part of [1, 3]) assert.ok(HISTORY_CARDS.some((c) => c.part === part && c.funFacts?.length), `no fact in Part ${part}`);
});

test("every fun fact has its text, and the labels, in all 10 languages", () => {
  assert.equal(langs.length, 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const keys = [...UI, ...facts.map((x) => `history.fun.${x.id}`)];
    const missing = keys.filter((k) => typeof text[k] !== "string" || !text[k].trim());
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
    for (const k of ["history.fun.title", "history.fun.showAll"]) assert.ok(text[k].includes("💡"), `${f}: ${k} lost its 💡`);
    const dashed = keys.filter((k) => /[—―]| – /.test(text[k]));
    assert.deepEqual(dashed, [], `${f}: em dash in ${dashed.join(", ")}`);
  }
});

test("every fun fact has at least one https source with a label", () => {
  for (const f of facts) {
    assert.ok(f.sources.length >= 1, `${f.id}: no source`);
    for (const s of f.sources) {
      assert.match(s.url, /^https:\/\/[^\s]+$/, `${f.id}: ${s.url}`);
      assert.ok(s.label.trim() && !/[—―]/.test(s.label), `${f.id}: label "${s.label}"`);
    }
  }
});

// the NVISION 2008 demo, the video chosen by the site's author (https://www.youtube.com/watch?v=WmW6SD-EHVY)
const VIDEO_ID = "WmW6SD-EHVY";

test("only the MythBusters fact embeds a player: youtube-nocookie.com, the verified video id", () => {
  const myth = facts.find((f) => f.id === "mythbusters");
  assert.ok(myth, "no mythbusters fact");
  assert.equal(myth.embed, `https://www.youtube-nocookie.com/embed/${VIDEO_ID}`);
  assert.deepEqual(facts.filter((f) => f.embed).map((f) => f.id), ["mythbusters"]);
  // the page creates the iframe from the data inside the note (rendered only while the note is open)
  const page = readFileSync(new URL("../src/pages/HistoryConcept.tsx", import.meta.url), "utf-8");
  assert.match(page, /f\.embed && \(/);
  assert.doesNotMatch(page, /youtube\.com\/(embed|watch)/, "only the privacy-enhanced youtube-nocookie.com player");
});
