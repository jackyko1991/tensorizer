// Hash routes of the Concept and Coding Skills sub-tabs. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { PAGE_TABS, formatPageRoute, parsePageRoute, resolveTab } from "../src/pageRoute.ts";

test("parse: page, tab and section; other routes are not tabbed pages", () => {
  assert.deepEqual(parsePageRoute("skills"), { page: "skills", tab: null, section: null });
  assert.deepEqual(parsePageRoute("skills/loops"), { page: "skills", tab: "loops", section: null });
  assert.deepEqual(parsePageRoute("skills/loops/matmul"), { page: "skills", tab: "loops", section: "matmul" });
  assert.deepEqual(parsePageRoute("skills/blas"), { page: "skills", tab: "blas", section: null });
  assert.deepEqual(parsePageRoute("skills/blas/repro"), { page: "skills", tab: "blas", section: "repro" });
  assert.deepEqual(parsePageRoute("skills/nvlink"), { page: "skills", tab: "nvlink", section: null });
  assert.deepEqual(parsePageRoute("skills/nvlink/coll"), { page: "skills", tab: "nvlink", section: "coll" });
  assert.deepEqual(parsePageRoute("skills/cpuml"), { page: "skills", tab: "cpuml", section: null });
  assert.deepEqual(parsePageRoute("skills/cpuml/ksvm"), { page: "skills", tab: "cpuml", section: "ksvm" });
  assert.deepEqual(parsePageRoute("concept/dataflow"), { page: "concept", tab: "dataflow", section: null });
  assert.deepEqual(parsePageRoute("concept/"), { page: "concept", tab: null, section: null });
  assert.deepEqual(parsePageRoute("concept/maths"), { page: "concept", tab: "maths", section: null });
  assert.deepEqual(parsePageRoute("concept/maths/roof"), { page: "concept", tab: "dataflow", section: "roof" });
  assert.deepEqual(parsePageRoute("concept/maths/topo"), { page: "concept", tab: "dataflow", section: "topo" });
  assert.deepEqual(parsePageRoute("concept/dataflow/topo"), { page: "concept", tab: "dataflow", section: "topo" });
  assert.deepEqual(parsePageRoute("concept/history"), { page: "concept", tab: "history", section: null });
  assert.deepEqual(parsePageRoute("concept/history/riemann"), { page: "concept", tab: "history", section: "riemann" });
  assert.deepEqual(parsePageRoute("concept/feynman"), { page: "concept", tab: "feynman", section: null });
  assert.deepEqual(parsePageRoute("concept/feynman/example"), { page: "concept", tab: "feynman", section: "example" });
  assert.deepEqual(parsePageRoute("concept/reading"), { page: "concept", tab: "reading", section: null });
  assert.deepEqual(parsePageRoute("concept/reading/gpu"), { page: "concept", tab: "reading", section: "gpu" });
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
  assert.equal(resolveTab("skills", "nvlink", "blas"), "nvlink");
  assert.equal(resolveTab("skills", null, "nvlink"), "nvlink");
  assert.equal(resolveTab("skills", "cpuml", "nvlink"), "cpuml");
  assert.equal(resolveTab("skills", null, "cpuml"), "cpuml");
  assert.equal(resolveTab("concept", null, null), "laws");
  assert.equal(resolveTab("concept", "dataflow", null), "dataflow");
  assert.equal(resolveTab("concept", null, "flow"), "dataflow"); // value stored before tabs had URL ids
  assert.equal(resolveTab("concept", null, "law"), "laws");
  assert.equal(resolveTab("concept", "maths", "dataflow"), "maths");
  assert.equal(resolveTab("concept", null, "maths"), "maths");
  assert.equal(resolveTab("concept", "history", "laws"), "history");
  assert.equal(resolveTab("concept", null, "history"), "history");
  assert.equal(resolveTab("concept", "feynman", "maths"), "feynman");
  assert.equal(resolveTab("concept", null, "feynman"), "feynman");
  assert.equal(resolveTab("concept", "reading", "laws"), "laws"); // Further reading is its own page now
  assert.equal(resolveTab("concept", null, "reading"), "laws");
});

test("format round-trips through parse", () => {
  assert.equal(formatPageRoute("skills", "loops"), "/skills/loops");
  assert.equal(formatPageRoute("skills", "blas", "bench"), "/skills/blas/bench");
  assert.equal(formatPageRoute("skills", "nvlink", "practice"), "/skills/nvlink/practice");
  assert.equal(formatPageRoute("skills", "cpuml", "umap"), "/skills/cpuml/umap");
  assert.equal(formatPageRoute("concept", "laws", "3"), "/concept/laws/3");
  assert.equal(formatPageRoute("concept", "maths", "penrose"), "/concept/maths/penrose");
  assert.equal(formatPageRoute("concept", "history", "gauss"), "/concept/history/gauss");
  assert.equal(formatPageRoute("concept", "feynman", "steps"), "/concept/feynman/steps");
  assert.equal(formatPageRoute("concept", "reading", "history"), "/concept/reading/history");
  assert.deepEqual(parsePageRoute(formatPageRoute("concept", "feynman", "pitfalls")), {
    page: "concept",
    tab: "feynman",
    section: "pitfalls",
  });
  assert.deepEqual(parsePageRoute(formatPageRoute("concept", "history", "einsum")), {
    page: "concept",
    tab: "history",
    section: "einsum",
  });
  assert.deepEqual(parsePageRoute(formatPageRoute("skills", "loops", "matmul")), {
    page: "skills",
    tab: "loops",
    section: "matmul",
  });
});

test("the concept tabs in order (Further reading is a top-level page, #/reading)", () => {
  assert.deepEqual([...PAGE_TABS.concept], ["laws", "feynman", "maths", "dataflow", "limitations", "history"]);
});
