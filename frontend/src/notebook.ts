/** Jupyter notebook export: "Open in Colab" (the committed default) and "Download .ipynb" (your current version). */
import { generatedUrl } from "./api";
import type { Params, Problem } from "./types";
import { workspaceNotebook } from "./workspaceNotebook";

export { lastPushedRepo, rememberPushedRepo } from "./workspaceNotebook";

export const REPO = "jackyko1991/tensorizer";
export const REPO_URL = `https://github.com/${REPO}`;

/** VS Code for the web, opened on the problem's main.py in the GitHub repository (no install needed). */
export const vscodeWebUrl = (problem: Problem) =>
  `https://vscode.dev/github/${REPO}/blob/main/problems/${problem.category}/${problem.slug}/main.py`;

/** Desktop VS Code: clone the repository (VS Code asks where to put it). */
export const vscodeCloneUrl = `vscode://vscode.git/clone?url=${encodeURIComponent(`${REPO_URL}.git`)}`;

export const colabUrl = (problem: Problem) =>
  `https://colab.research.google.com/github/${REPO}/blob/main/notebooks/${problem.category}/${problem.slug}.ipynb`;

interface Cell {
  cell_type: string;
  metadata: { tensorizer?: string };
  source: string[];
}

const lines = (text: string) => text.split(/(?<=\n)/);

/** Download the problem's notebook with the user's edited files and parameter values filled in. */
export async function downloadNotebook(problem: Problem, edits: Record<string, string>, params: Params) {
  if (!problem.notebook) return;
  const nb = await (await fetch(generatedUrl(problem.notebook))).json();
  for (const cell of nb.cells as Cell[]) {
    const tag = cell.metadata.tensorizer ?? "";
    if (tag.startsWith("file:") && edits[tag.slice(5)] !== undefined) {
      const header = cell.source[0]; // "%%writefile problems/<category>/<slug>/<file>\n"
      cell.source = [header, ...lines(edits[tag.slice(5)])];
    } else if (tag === "run") {
      const body = Object.entries(params)
        .map(([k, v]) => `    ${JSON.stringify(k)}: ${JSON.stringify(v)}`)
        .join(",\n");
      cell.source = lines(cell.source.join("").replace(/params = \{\n[\s\S]*?\n\}\n/, `params = {\n${body}\n}\n`));
    }
  }
  save(new Blob([JSON.stringify(nb, null, 1) + "\n"], { type: "application/x-ipynb+json" }), `${problem.slug}.ipynb`);
}

/** Download the standalone pixi project, with the user's edited files and parameter values patched in. */
export async function downloadPixiProject(problem: Problem, edits: Record<string, string>, params: Params) {
  if (!problem.pixi_zip) return;
  const { unzipSync, zipSync, strToU8 } = await import("fflate");
  const files = unzipSync(new Uint8Array(await (await fetch(generatedUrl(problem.pixi_zip))).arrayBuffer()));
  const root = `${problem.slug}/`;
  for (const [name, code] of Object.entries(edits)) {
    files[`${root}problems/${problem.category}/${problem.slug}/${name}`] = strToU8(code);
  }
  files[`${root}params.json`] = strToU8(JSON.stringify(params, null, 2) + "\n");
  save(new Blob([zipSync(files)], { type: "application/zip" }), `${problem.slug}-pixi.zip`);
}

function save(blob: Blob, filename: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function downloadWorkspaceNotebook(problem: Problem, meta: string, edits: Record<string, string>, params: Params) {
  const nb = workspaceNotebook(problem, meta, edits, params);
  save(new Blob([JSON.stringify(nb, null, 1) + "\n"], { type: "application/x-ipynb+json" }), `${problem.slug}.ipynb`);
}

