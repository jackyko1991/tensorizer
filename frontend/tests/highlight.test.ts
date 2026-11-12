import { test } from "node:test";
import assert from "node:assert/strict";
import { highlightCode } from "../src/components/highlight.ts";

test("strings, numbers, keywords and calls get colours", () => {
  const h = highlightCode('score = corpus.parse("bach/bwv66.6") if x else 2');
  assert.match(h, /<span class="text-blue-700">corpus\.parse<\/span>/);
  assert.match(h, /<span class="text-emerald-700">"bach\/bwv66\.6"<\/span>/);
  assert.match(h, /<span class="text-violet-700">if<\/span>/);
  assert.match(h, /<span class="text-amber-700">2<\/span>/);
});

test("tinyNotation apostrophes are not strings; escaped entities stay intact", () => {
  assert.equal(highlightCode("c'4 d' e'"), "c'<span class=\"text-amber-700\">4</span> d' e'");
  assert.equal(highlightCode("a &lt; b &amp;&amp; c"), "a &lt; b &amp;&amp; c");
  assert.match(highlightCode("tiny('c4 d')"), /<span class="text-emerald-700">'c4 d'<\/span>/);
});
