/** Jupyter notebook export: "Open in Colab" (the committed default) and "Download .ipynb" (your current version). */
import { generatedUrl } from "./api";
import type { Params, Problem } from "./types";

export const REPO = "jackyko1991/tensorizer";
export const REPO_URL = `https://github.com/${REPO}`;

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
  const blob = new Blob([JSON.stringify(nb, null, 1) + "\n"], { type: "application/x-ipynb+json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${problem.slug}.ipynb`;
  a.click();
  URL.revokeObjectURL(a.href);
}
