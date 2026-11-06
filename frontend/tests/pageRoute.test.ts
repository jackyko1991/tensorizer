// Hash routes of the Concept and Coding Skills sub-tabs. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { formatPageRoute, parsePageRoute, resolveTab } from "../src/pageRoute.ts";

test("parse: page, tab and section; other routes are not tabbed pages", () => {
  assert.deepEqual(parsePageRoute("skills"), { page: "skills", tab: null, section: null });
  assert.deepEqual(parsePageRoute("skills/loops"), { page: "skills", tab: "loops", section: null });
  assert.deepEqual(parsePageRoute("skills/loops/matmul"), { page: "skills", tab: "loops", section: "matmul" });
  assert.deepEqual(parsePageRoute("skills/blas"), { page: "skills", tab: "blas", section: null });
  assert.deepEqual(parsePageRoute("skills/blas/repro"), { page: "skills", tab: "blas", section: "repro" });
  assert.deepEqual(parsePageRoute("concept/dataflow"), { page: "concept", tab: "dataflow", section: null });
  assert.deepEqual(parsePageRoute("concept/"), { page: "concept", tab: null, section: null });
  assert.deepEqual(parsePageRoute("concept/maths"), { page: "concept", tab: "maths", section: null });
  assert.deepEqual(parsePageRoute("concept/maths/roof"), { page: "concept", tab: "maths", section: "roof" });
  for (const r of [null, "", "agents", "workspace", "mechanics/simple_pendulum", "skillsx", "workspace/a/nb/b"])
    assert.equal(parsePageRoute(r), null, String(r));
});

test("resolve: the address wins, an unknown tab falls back to the first, no tab uses the remembered one", () => {
  assert.equal(resolveTab("skills", "loops", "techniques"), "loops");
  assert.equal(resolveTab("skills", "techniques", "loops"), "techniques");
  assert.equal(resolveTab("skills", "bogus", "loops"), "techniques");
  assert.equal(resolveTab("skills", null, "loops"), "loops");
  assert.equal(resolveTab("skills", null, null), "techniques");
  assert.equal(resolveTab("skills", null, "garbage"), "techniques");
  assert.equal(resolveTab("skills", "blas", "loops"), "blas");
  assert.equal(resolveTab("skills", null, "blas"), "blas");
  assert.equal(resolveTab("concept", null, null), "laws");
  assert.equal(resolveTab("concept", "dataflow", null), "dataflow");
  assert.equal(resolveTab("concept", null, "flow"), "dataflow"); // value stored before tabs had URL ids
  assert.equal(resolveTab("concept", null, "law"), "laws");
  assert.equal(resolveTab("concept", "maths", "dataflow"), "maths");
  assert.equal(resolveTab("concept", null, "maths"), "maths");
});

test("format round-trips through parse", () => {
  assert.equal(formatPageRoute("skills", "loops"), "/skills/loops");
  assert.equal(formatPageRoute("skills", "blas", "bench"), "/skills/blas/bench");
  assert.equal(formatPageRoute("concept", "laws", "3"), "/concept/laws/3");
  assert.equal(formatPageRoute("concept", "maths", "penrose"), "/concept/maths/penrose");
  assert.deepEqual(parsePageRoute(formatPageRoute("skills", "loops", "matmul")), {
    page: "skills",
    tab: "loops",
    section: "matmul",
  });
});
