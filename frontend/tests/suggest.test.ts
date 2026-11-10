// The AI panel's suggested prompts follow the problem's state. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { isSkeleton, LIBRARY_PROMPTS, suggestPrompts } from "../src/suggest.ts";
import type { Problem } from "../src/types.ts";

const BLANK_MAIN = '    ax.text(0.5, 0.5, "Empty canvas: write your problem in main.py", ha="center")\n';
const blank = {
  id: "workspace/abc",
  category: "workspace",
  slug: "abc",
  level: "beginner",
  title: "3D pipe flow",
  summary: "simulation of 3d pipeflow",
  order: 0,
  animated: false,
  vector_form: "",
  index_form: "",
  translation: [],
  params: {},
  files: [
    { name: "main.py", code: BLANK_MAIN },
    { name: "equation.py", code: '"""The physics written as tensor operations."""\n' },
  ],
  output: null,
  console: "",
  diagram: null,
} as unknown as Problem;

const keys = (p: Problem, view: Parameters<typeof suggestPrompts>[1] = null) => suggestPrompts(p, view).map((s) => s.key);

test("a blank problem asks for equations and an implementation, with its own title and summary", () => {
  assert.ok(isSkeleton(blank));
  const out = suggestPrompts(blank, null);
  assert.deepEqual(
    out.map((s) => s.key),
    ["agent.sg.equations", "agent.sg.implement", "agent.sg.translation", "agent.sg.params"],
  );
  assert.equal(out[0].vars.title, "3D pipe flow");
  assert.equal(out[0].vars.summary, "simulation of 3d pipeflow");
  assert.ok(!keys(blank).some((k) => k.startsWith("agent.newEx")));
});

test("a written problem without output asks to run it; after a run, to explain the result", () => {
  const done = {
    ...blank,
    vector_form: "a",
    index_form: "b",
    translation: [{ symbol: "x", code: "x", shape: "(N,)", meaning: "" }],
    params: { n: { default: 1 } },
    files: [
      { name: "main.py", code: "def run(params, out_dir): ..." },
      { name: "equation.py", code: "def rhs(y):\n    return -y\n" },
    ],
  } as unknown as Problem;
  assert.ok(!isSkeleton(done));
  assert.deepEqual(keys(done), ["agent.sg.runFix"]);
  const view = { problem: done.id, title: "", edits: {}, params: {}, run: { output: true, failed: false } };
  assert.deepEqual(keys(done, view), ["agent.sg.explainResult"]);
  assert.deepEqual(keys(done, { ...view, run: { output: true, failed: true } }), ["agent.sg.runFix"]);
});

test("an open notebook puts cell prompts first, at most four in all", () => {
  const view = {
    problem: blank.id,
    title: "",
    edits: {},
    params: {},
    workspace: {
      kind: "notebook" as const,
      cells: [
        { id: "1", type: "markdown" as const, source: "# hi" },
        { id: "2", type: "code" as const, source: "print(1)" },
      ],
    },
  };
  const out = suggestPrompts(blank, view);
  assert.equal(out.length, 4);
  assert.deepEqual(out.slice(0, 2).map((s) => s.key), ["agent.sg.explainCell", "agent.sg.plotCell"]);
  assert.equal(out[0].vars.n, "2");
  assert.equal(suggestPrompts(blank, { ...view, workspace: { ...view.workspace, focusedCell: 1 } })[0].vars.n, "1");
});

test("a blank problem after a successful run offers to explain the result, right after its equations", () => {
  const view = { problem: blank.id, title: "", edits: {}, params: {}, run: { output: true, failed: false } };
  assert.deepEqual(keys(blank, view).slice(0, 3), ["agent.sg.equations", "agent.sg.explainResult", "agent.sg.implement"]);
});

test("uploaded data adds a fitting prompt with the file name", () => {
  const out = suggestPrompts({ ...blank, data: [{ name: "img.png", size: 1, type: "" }, { name: "m.csv", size: 1, type: "" }] }, null);
  assert.deepEqual(out[0], { key: "agent.sg.fitData", vars: { title: "3D pipe flow", summary: "simulation of 3d pipeflow", file: "m.csv" } });
});

test("a library problem without content falls back to general prompts", () => {
  assert.deepEqual(keys({ ...blank, category: "mechanics" }), LIBRARY_PROMPTS);
});

const titration = {
  ...blank,
  id: "chemistry/acid_base_titration",
  category: "chemistry",
  title: "Acid-base titration curves",
  animated: false,
  index_form: "\\bar n = \\sum_j j\\,\\alpha_j",
  translation: [{ symbol: "\\alpha_j", code: "alpha", shape: "(A, N)", meaning: "species fraction" }],
  params: { c_acid: { default: 0.1, min: 0.01, max: 1, label: "acid concentration", unit: "mol/L" } },
  files: [{ name: "test.py", code: "def test_x(): pass" }],
  i18n: { "zh-Hant": { title: "酸鹼滴定曲線", labels: { c_acid: "酸濃度" } } },
} as unknown as Problem;

test("a library problem's prompts come from its own content, in the page's language", () => {
  const out = suggestPrompts(titration, null);
  assert.deepEqual(
    out.map((s) => s.key),
    ["agent.sg.libIndex", "agent.sg.libParam", "agent.sg.libTests", "agent.sg.libPlot"], // static: a plot, not motion
  );
  assert.deepEqual(out[1].vars, {
    title: "Acid-base titration curves",
    param: "acid concentration",
    from: "0.1",
    to: "1",
    unit: " mol/L",
  });
  assert.ok(!out.some((s) => s.key === "agent.ex2")); // no "add damping" for chemistry
  const zh = suggestPrompts(titration, null, "zh-Hant");
  assert.equal(zh[0].vars.title, "酸鹼滴定曲線");
  assert.equal(zh[1].vars.param, "酸濃度");
  // no parameters: a symbol from the translation table instead; animated: the animation
  const moving = suggestPrompts({ ...titration, params: {}, animated: true }, null);
  assert.deepEqual(
    moving.map((s) => s.key),
    ["agent.sg.libIndex", "agent.sg.libSymbol", "agent.sg.libTests", "agent.sg.libAnim"],
  );
  assert.equal(moving[1].vars.symbol, "$\\alpha_j$");
});
