import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import katex from "katex";

const sections = JSON.parse(readFileSync(new URL("../src/content/limitations.json", import.meta.url), "utf8"));
test("limitations explanations have complete bilingual teaching steps and renderable equations", () => {
  for (const section of sections) {
    for (const field of ["plain", "analogy", "precise", "example", "legend"]) {
      assert.equal(section[field].length, 2, `${section.id}: ${field}`);
      assert.ok(section[field].every((text: string) => text.trim().length > 0));
    }
    if (section.equation) assert.doesNotThrow(() => katex.renderToString(section.equation, { throwOnError: true }), section.id);
  }
});
