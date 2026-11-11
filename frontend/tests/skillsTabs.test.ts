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

// The sub-pages of Concept and Coding Skills are listed only in their header tab's menu (as AI agent and Further
// reading are); the pages themselves have no tab row any more.
test("Concept and Coding Skills list their sub-pages in the header menu, not in a tab row", () => {
  const app = read("App.tsx");
  assert.match(app, /const MENU_PAGES = \["concept", "skills", "agents", "reading"\] as const;/);
  assert.match(app, /concept: "concept-subpages"/);
  assert.match(app, /skills: "skills-subpages"/);
  // every tab of a menu page becomes an entry: PAGE_TABS drives the menu, so a new tab cannot be forgotten
  assert.match(app, /PAGE_TABS\[target\]\.map\(\(id\) =>/);
  assert.match(app, /href: `#\$\{formatPageRoute\(target, id\)\}`/);
  for (const page of ["pages/ConceptPage.tsx", "pages/SkillsPage.tsx"]) {
    const src = read(page);
    assert.doesNotMatch(src, /role="tablist"/, `${page} still has a tab row`);
    assert.doesNotMatch(src, /tabs-sticky/, `${page} still has a sticky tab row`);
  }
});

test("Concept menu has a label for every concept tab", () => {
  const app = read("App.tsx");
  const labels = app.match(/const CONCEPT_LABEL[^}]+}/)?.[0] ?? "";
  for (const t of PAGE_TABS.concept) assert.match(labels, new RegExp(`\\b${t}: "`), `no label for ${t}`);
});
