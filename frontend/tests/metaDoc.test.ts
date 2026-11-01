// Markdown ↔ meta round trip and the YAML emitter. Run: node --test frontend/tests (Node ≥ 23 strips the types).
import assert from "node:assert/strict";
import { test } from "node:test";
import { fromMarkdown, MarkdownError, pickMd, toMarkdown, toYaml, type MdFields } from "../src/metaDoc.ts";

const sample: MdFields = {
  title: "Damped oscillator $\\ddot{x}$",
  summary: "A mass on a spring with friction.\n\nSecond paragraph with **bold** and `code`.",
  vector_form: "\\ddot{x} = -\\omega^2 x - 2\\gamma \\dot{x}",
  index_form: "\\ddot{x}_b = -\\omega_b^2 x_b",
  translation: [
    { symbol: "x_b", code: "x[b]", shape: "(B,)", meaning: "position of oscillator $b$" },
    { symbol: "|v|", code: "v.abs()", shape: "(B,)", meaning: "speed, a | b pipe" },
    { symbol: "", code: "", shape: "", meaning: "" },
  ],
  notes: "Uses $\\gamma < \\omega$ (underdamped).",
};

test("markdown round trip keeps every field", () => {
  assert.deepEqual(fromMarkdown(toMarkdown(sample)), pickMd(sample));
});

test("empty fields round trip", () => {
  const empty: MdFields = { title: "Untitled problem", summary: "", vector_form: "", index_form: "", translation: [], notes: "" };
  assert.deepEqual(fromMarkdown(toMarkdown(empty)), empty);
});

test("hand-written markdown: equations without $$, missing sections", () => {
  const f = fromMarkdown("# Ball\nfalls.\n## Index form\n\\ddot{y} = -g\n");
  assert.equal(f.title, "Ball");
  assert.equal(f.summary, "falls.");
  assert.equal(f.index_form, "\\ddot{y} = -g");
  assert.equal(f.vector_form, "");
});

test("bad markdown raises a MarkdownError with a line number", () => {
  assert.throws(() => fromMarkdown("no title here"), MarkdownError);
  assert.throws(() => fromMarkdown("# T\n## Mystery\n"), /line 2: unknown section/);
  assert.throws(() => fromMarkdown("# T\n## Translation\n| a | b |\n|---|---|\n| 1 | 2 |\n"), /4 cells/);
});

test("yaml emitter quotes LaTeX and nests params and rows", () => {
  const yaml = toYaml({
    title: "Untitled problem",
    vector_form: "\\ddot{x} = -x",
    animated: false,
    translation: [{ symbol: "x", code: "x", shape: "(N,)", meaning: "it's x" }],
    params: { g: { default: 9.81, min: 0, label: "g", type: "float" }, mode: { default: "a", options: ["a", "b"] } },
    empty: {},
    summary: "two\nlines",
  });
  assert.match(yaml, /^title: Untitled problem$/m);
  assert.match(yaml, /^vector_form: '\\ddot\{x\} = -x'$/m);
  assert.match(yaml, /^ {2}- symbol: x\n {4}code: x\n {4}shape: '\(N,\)'\n {4}meaning: 'it''s x'$/m);
  assert.match(yaml, /^ {2}g:\n {4}default: 9\.81\n {4}min: 0$/m);
  assert.match(yaml, /^ {4}options: \[a, b\]$/m);
  assert.match(yaml, /^empty: \{\}$/m);
  assert.match(yaml, /^summary: \|-\n {2}two\n {2}lines$/m);
});
