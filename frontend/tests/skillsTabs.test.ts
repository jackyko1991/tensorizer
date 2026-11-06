// Every Coding Skills tab id is passed through by App and rendered by its own branch in SkillsPage
// (a missing branch silently shows the Techniques tab). Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { PAGE_TABS } from "../src/pageRoute.ts";

const read = (p: string) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf-8");

test("App passes any known skills tab through to SkillsPage", () => {
  assert.match(read("App.tsx"), /tab=\{isTab\("skills", subTab\) \? subTab : "techniques"\}/);
});

test("SkillsPage has a render branch for every skills tab but the default", () => {
  const src = read("pages/SkillsPage.tsx");
  for (const t of PAGE_TABS.skills.filter((t) => t !== "techniques"))
    assert.match(src, new RegExp(`tab === "${t}" \\? \\(`), `no branch for ${t}`);
});
