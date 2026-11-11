/** Workspace problems on Export: their notebook (built in the browser) and the GitHub repository the workspace was
 *  last exported to. Pure (no DOM besides localStorage), unit-tested in tests/workspaceNotebook.test.ts. */
import type { Params, Problem } from "./types";

const lines = (text: string) => text.split(/(?<=\n)/);

/** A Python literal for a parameter value (JSON's true/false/null are not Python). */
function pyLiteral(v: unknown): string {
  if (v === true) return "True";
  if (v === false) return "False";
  if (v === null || v === undefined) return "None";
  if (Array.isArray(v)) return `[${v.map(pyLiteral).join(", ")}]`;
  return JSON.stringify(v);
}

const md = (text: string) => ({ cell_type: "markdown", metadata: {}, source: lines(text) });
const code = (text: string, tag: string) => ({
  cell_type: "code",
  metadata: { tensorizer: tag },
  execution_count: null,
  outputs: [],
  source: lines(text),
});

/** A Workspace problem's notebook, built in the browser in the same shape as the library ones: setup (clone the
 *  shared helpers), one %%writefile cell per file (+ meta.yaml), then the run cell with the current parameters.
 *  Uploaded data files are not included (they are in the .zip). */
export function workspaceNotebook(problem: Problem, meta: string, edits: Record<string, string>, params: Params) {
  const pkg = `problems.${problem.category}.${problem.slug}`;
  const dir = `problems/${problem.category}/${problem.slug}`;
  const files = problem.files.map((f) => ({ name: f.name, code: edits[f.name] ?? f.code }));
  const setup = [
    "import os",
    "import subprocess",
    "import sys",
    "",
    "# Shared helpers (problems/_common) come from the Tensorizer repository; skip the clone if we are already inside it.",
    'if not os.path.isdir("problems"):',
    '    if not os.path.isdir("tensorizer"):',
    '        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/jackyko1991/tensorizer.git"], check=True)',
    '    os.chdir("tensorizer")',
    "sys.path.insert(0, os.getcwd())",
    `os.makedirs(${JSON.stringify(dir)}, exist_ok=True)`,
    `open(${JSON.stringify(`${dir}/__init__.py`)}, "a").close()`,
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "tqdm"], check=True)',
    "",
    "import torch",
    "",
    'print("torch", torch.__version__, "· device:", "cuda" if torch.cuda.is_available() else "cpu")',
    "",
  ].join("\n");
  const body = Object.entries(params)
    .map(([k, v]) => `    ${JSON.stringify(k)}: ${pyLiteral(v)}`)
    .join(",\n");
  const run = [
    "import importlib",
    "from pathlib import Path",
    "",
    "from IPython.display import Image, display",
    "",
    "from problems._common.runtime import load_meta, resolve_params",
    "",
    `params = {\n${body}\n}`,
    "",
    "# Re-import so edits in the %%writefile cells above take effect.",
    `for name in [m for m in sys.modules if m.startswith(${JSON.stringify(pkg)})]:`,
    "    del sys.modules[name]",
    `main = importlib.import_module(${JSON.stringify(`${pkg}.main`)})`,
    `problem_dir = Path(${JSON.stringify(dir)})`,
    'output = main.run(resolve_params(load_meta(problem_dir), params), Path("out"))',
    "display(Image(filename=str(output)))",
    "",
  ].join("\n");
  const cells = [
    md(`# ${problem.title}\n\n*Workspace problem exported from [Tensorizer](https://jackyko1991.github.io/tensorizer/)*`),
    md("## Setup\n\nOn Colab, choose **Runtime → Change runtime type → GPU** to run on a GPU: the code uses it automatically."),
    code(setup, "setup"),
    md("## `meta.yaml`"),
    code(`%%writefile ${dir}/meta.yaml\n${meta.endsWith("\n") ? meta : meta + "\n"}`, "file:meta.yaml"),
    ...files.flatMap((f, i) => [
      md(`## ${i + 1}. \`${f.name}\``),
      code(`%%writefile ${dir}/${f.name}\n${f.code.endsWith("\n") ? f.code : f.code + "\n"}`, `file:${f.name}`),
    ]),
    md("## Run\n\nChange the parameters (or any file above) and run again."),
    code(run, "run"),
  ];
  return {
    cells,
    metadata: {
      kernelspec: { name: "python3", display_name: "Python 3", language: "python" },
      language_info: { name: "python" },
      colab: { provenance: [], name: `${problem.slug}.ipynb` },
      tensorizer: { problem: `${problem.category}/${problem.slug}`, package: pkg },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

/** The GitHub repository the Workspace was last pushed to (GitHubDialog remembers it), or null. */
const LAST_REPO = "tensorizer.ghLastRepo";
export function lastPushedRepo(): { owner: string; repo: string; url: string } | null {
  try {
    const url = localStorage.getItem(LAST_REPO) ?? "";
    const m = url.match(/^https:\/\/github\.com\/([\w.-]+)\/([\w.-]+)\/?$/);
    return m ? { owner: m[1], repo: m[2], url } : null;
  } catch {
    return null;
  }
}
export function rememberPushedRepo(url: string) {
  try {
    localStorage.setItem(LAST_REPO, url);
  } catch {
    // storage blocked: VS Code items stay unavailable
  }
}
