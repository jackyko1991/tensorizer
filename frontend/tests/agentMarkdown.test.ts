// Splitting the assistant's answers: code fences, display maths in two notations, inline \( \). Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { copyText, isPython, parseAnswer } from "../src/agentMarkdown.ts";

test("code, display maths (both notations) and inline maths are told apart", () => {
  const text = [
    "The slope \\(\\theta_k\\) is",
    "\\[",
    "\\theta_{k2}=-\\frac{h}{s_k-f}.",
    "\\]",
    "and $$y=h+\\theta s$$ in code:",
    "```python",
    "return h + slope * s[:, None]",
    "```",
    "done.",
  ].join("\n");
  assert.deepEqual(parseAnswer(text), [
    { kind: "md", text: "The slope $\\theta_k$ is\n" },
    { kind: "math", tex: "\\theta_{k2}=-\\frac{h}{s_k-f}." },
    { kind: "md", text: "\nand " },
    { kind: "math", tex: "y=h+\\theta s" },
    { kind: "md", text: " in code:\n" },
    { kind: "code", lang: "python", code: "return h + slope * s[:, None]" },
    { kind: "md", text: "\ndone." },
  ]);
});

test("a fence still streaming in shows as code, and maths inside code is left alone", () => {
  assert.deepEqual(parseAnswer("See:\n```\nx = '$a$' \\(b\\)\ny = 2"), [
    { kind: "md", text: "See:\n" },
    { kind: "code", lang: "", code: "x = '$a$' \\(b\\)\ny = 2" },
  ]);
  assert.ok(isPython("") && isPython("py") && !isPython("yaml"));
});

test("copying an answer keeps its text blocks as written", () => {
  const blocks = [
    { kind: "tool" },
    { kind: "text", text: "First part. " },
    { kind: "run" },
    { kind: "text", text: "\n```py\nx = 1\n```" },
  ];
  assert.equal(copyText(blocks), "First part.\n\n```py\nx = 1\n```");
});
