// Export ▸ notebook for a Workspace problem, built in the browser in the library notebooks' shape. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { workspaceNotebook } from "../src/workspaceNotebook.ts";
import type { Problem } from "../src/types.ts";

const problem = {
  id: "workspace/abc123",
  category: "workspace",
  slug: "abc123",
  title: "RLC circuit",
  files: [
    { name: "main.py", code: "def run(p, out):\n    return out\n" },
    { name: "solve.py", code: "x = 1" },
  ],
} as unknown as Problem;

test("cells: setup, meta.yaml, one %%writefile per file (edits win), then run", () => {
  const nb = workspaceNotebook(problem, "title: RLC\n", { "solve.py": "x = 2\n" }, { R: 1.5, batched: true, n: 3, label: "a" });
  const tags = nb.cells.filter((c) => c.cell_type === "code").map((c) => (c.metadata as { tensorizer: string }).tensorizer);
  assert.deepEqual(tags, ["setup", "file:meta.yaml", "file:main.py", "file:solve.py", "run"]);
  const src = (tag: string) => nb.cells.find((c) => (c.metadata as { tensorizer?: string }).tensorizer === tag)!.source.join("");
  assert.equal(src("file:solve.py"), "%%writefile problems/workspace/abc123/solve.py\nx = 2\n");
  assert.match(src("file:meta.yaml"), /^%%writefile problems\/workspace\/abc123\/meta\.yaml\ntitle: RLC\n$/);
  assert.match(src("setup"), /os\.makedirs\("problems\/workspace\/abc123", exist_ok=True\)/);
  const run = src("run");
  assert.match(run, /"batched": True/); // Python, not JSON
  assert.match(run, /"R": 1\.5/);
  assert.match(run, /import_module\("problems\.workspace\.abc123\.main"\)/);
  assert.equal(nb.nbformat, 4);
  JSON.parse(JSON.stringify(nb)); // serialisable as .ipynb
});
