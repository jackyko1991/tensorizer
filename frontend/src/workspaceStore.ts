/** Workspaces in the browser's IndexedDB (memory when it is unavailable), and .zip export/import. */
import { useCallback, useEffect, useRef, useState } from "react";
import { generatedUrl } from "./api";
import type { Cell, Manifest, Notebook, Problem, Workspace, WsProblem } from "./types";

const DB = "tensorizer";
const STORE = "workspaces";
const DATA_STORE = "data"; // uploaded data files' bytes, keyed "<problem id>/<name>" (kept out of the workspace record)
/** The free trial has one workspace. */
export const MAX_WORKSPACES = 1;
export const CODE_FILES = ["main.py", "equation.py", "tensors.py", "solve.py", "plot.py"];

const memory = new Map<string, Workspace>();
let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  dbPromise ??= new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB, 2);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: "id" });
        if (!db.objectStoreNames.contains(DATA_STORE)) db.createObjectStore(DATA_STORE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null); // private window or storage disabled: keep everything in memory
    }
  });
  return dbPromise;
}

async function readAll(): Promise<Workspace[]> {
  try {
    const db = await openDb();
    if (!db) return [...memory.values()];
    return await new Promise((resolve, reject) => {
      const req = db.transaction(STORE).objectStore(STORE).getAll();
      req.onsuccess = () => resolve(req.result as Workspace[]);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return [...memory.values()];
  }
}

async function write(ws: Workspace): Promise<void> {
  memory.set(ws.id, ws);
  try {
    const db = await openDb();
    if (!db) return;
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(ws);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // quota exceeded or storage gone: the memory copy lasts for this visit
  }
}

// --- uploaded data files -------------------------------------------------------------------------------

export const MAX_DATA_FILE = 20 * 1024 * 1024;
export const MAX_DATA_TOTAL = 100 * 1024 * 1024;
export const DATA_EXTENSIONS = [
  ".csv", ".tsv", ".json", ".npy", ".npz", ".txt", ".dat", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".h5", ".hdf5",
];
const memoryData = new Map<string, Blob>();

/** A file name the server accepts (letters, digits, space, _ . ( ) -), keeping the extension. */
export function dataName(name: string): string {
  const clean = name.normalize("NFC").replace(/[^\p{L}\p{N}_ .()-]/gu, "_").replace(/\.{2,}/g, ".").trim();
  return (/^[\p{L}\p{N}_]/u.test(clean) ? clean : `f_${clean}`).slice(0, 100);
}
export const dataExtOk = (name: string) => DATA_EXTENSIONS.some((e) => name.toLowerCase().endsWith(e));

async function dataTx<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> {
  const db = await openDb();
  if (!db) return undefined;
  return new Promise((resolve, reject) => {
    const req = fn(db.transaction(DATA_STORE, mode).objectStore(DATA_STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
const dataKey = (problem: string, name: string) => `${problem}/${name}`;
export async function putData(problem: string, name: string, blob: Blob): Promise<void> {
  memoryData.set(dataKey(problem, name), blob);
  try {
    await dataTx("readwrite", (s) => s.put(blob, dataKey(problem, name)));
  } catch {
    // storage full or blocked: the memory copy lasts for this visit
  }
}
export async function getData(problem: string, name: string): Promise<Blob | undefined> {
  try {
    return ((await dataTx("readonly", (s) => s.get(dataKey(problem, name)))) as Blob | undefined) ?? memoryData.get(dataKey(problem, name));
  } catch {
    return memoryData.get(dataKey(problem, name));
  }
}
export async function deleteData(problem: string, name: string): Promise<void> {
  memoryData.delete(dataKey(problem, name));
  try {
    await dataTx("readwrite", (s) => s.delete(dataKey(problem, name)));
  } catch {
    // nothing stored
  }
}

export const newId = () => Math.random().toString(36).slice(2, 10);

/** The workspaces (always at least one) and an updater that saves after a short pause. */
export function useWorkspaces() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const pending = useRef(new Map<string, Workspace>());
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    readAll().then((all) => {
      const list = all.length ? all : [{ id: newId(), name: "", problems: [] }];
      setWorkspaces(list);
      setActiveId(list[0].id);
    });
  }, []);

  const flush = useCallback(() => {
    for (const ws of pending.current.values()) void write(ws);
    pending.current.clear();
  }, []);
  useEffect(() => {
    window.addEventListener("beforeunload", flush);
    return () => window.removeEventListener("beforeunload", flush);
  }, [flush]);

  const update = useCallback(
    (fn: (ws: Workspace) => Workspace) =>
      setWorkspaces((list) =>
        list.map((ws) => {
          if (ws.id !== activeId) return ws;
          const next = fn(ws);
          pending.current.set(next.id, next);
          window.clearTimeout(timer.current);
          timer.current = window.setTimeout(flush, 300);
          return next;
        }),
      ),
    [activeId, flush],
  );

  const active = workspaces.find((w) => w.id === activeId) ?? null;
  return { workspaces, active, setActiveId, update };
}

/** Update one problem of a workspace by its local id. */
export const patchProblem = (ws: Workspace, id: string, fn: (p: WsProblem) => WsProblem): Workspace => ({
  ...ws,
  problems: ws.problems.map((p) => (p.id === id ? { ...fn(p), updated: Date.now() } : p)),
});

/** A server entry as a workspace problem, keeping the local id (its URL) stable across sandboxes. */
export function toWsProblem(entry: Problem, meta: string, id = newId(), notebooks: Notebook[] = []): WsProblem {
  return {
    id,
    meta,
    notebooks,
    updated: Date.now(),
    entry: { ...entry, id: `workspace/${id}`, category: "workspace", slug: id },
  };
}

/** Every file the sandbox needs: the five code files and meta.yaml (not yet known for a fresh AI problem). */
export const filesOf = (p: WsProblem): Record<string, string> => ({
  ...Object.fromEntries(p.entry.files.map((f) => [f.name, f.code])),
  ...(p.meta ? { "meta.yaml": p.meta } : {}),
});

/** meta.yaml with a new title (a JSON string is valid YAML). */
export function withTitle(meta: string, title: string): string {
  const line = `title: ${JSON.stringify(title)}`;
  return /^title:.*$/m.test(meta) ? meta.replace(/^title:.*$/m, line) : `${line}\n${meta}`;
}

export const newCell = (type: Cell["type"], source = ""): Cell => ({ id: newId(), type, source });

// --- .zip export / import -----------------------------------------------------------------------------

const slugify = (title: string) =>
  title
    .toLowerCase()
    .replace(/\$[^$]*\$/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/^(\d)/, "p_$1") || "problem";

const lines = (text: string) => text.split(/(?<=\n)/);

/** A notebook as .ipynb: a setup cell makes the problem importable as `problem`, then the user's cells. */
function toIpynb(nb: Notebook, slug: string) {
  const setup = [
    "# Tensorizer: make this problem importable as `problem` (and its default parameters as `params`)",
    "import importlib, sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, str(Path.cwd().parents[1]))  # the project folder",
    `problem = sys.modules["problem"] = importlib.import_module("problems.workspace.${slug}")`,
    "from problems._common.runtime import load_meta, resolve_params",
    "",
    "params = resolve_params(load_meta(Path(problem.__file__).parent))",
  ].join("\n");
  const cells = [
    { cell_type: "code", metadata: { tensorizer: "setup" }, source: lines(setup), outputs: [], execution_count: null },
    ...nb.cells.map((c) =>
      c.type === "markdown"
        ? { cell_type: "markdown", metadata: {}, source: lines(c.source) }
        : {
            cell_type: "code",
            metadata: {},
            source: lines(c.source),
            execution_count: null,
            outputs: [
              ...(c.output?.stdout ? [{ output_type: "stream", name: "stdout", text: lines(c.output.stdout) }] : []),
              ...(c.output?.figures ?? []).map((f) => ({
                output_type: "display_data",
                metadata: {},
                data: { "image/png": f.replace(/^data:image\/png;base64,/, ""), "text/plain": ["<Figure>"] },
              })),
            ],
          },
    ),
  ];
  return {
    cells,
    metadata: { kernelspec: { name: "python3", display_name: "Python 3", language: "python" }, tensorizer: nb.title },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

function pixiToml(name: string, slugs: string[]) {
  const tasks = slugs
    .map((s) => `simulate-${s.replace(/_/g, "-")} = "python -m problems.workspace.${s}.main --out out/${s}"`)
    .join("\n");
  return `[workspace]
name = "${name}"
description = "A Tensorizer workspace (https://github.com/jackyko1991/tensorizer)"
channels = ["conda-forge"]
platforms = ["win-64", "osx-arm64", "linux-64"]

[dependencies]
python = "3.12.*"
pytorch = ">=2.4"
numpy = ">=1.26"
matplotlib = ">=3.8"
pillow = ">=10"
pyyaml = ">=6"
tqdm = ">=4.66"
jupyterlab = ">=4"

[tasks]
# Render each problem into out/<problem>/
${tasks}
# Open the notebooks
notebooks = "jupyter lab notebooks"
`;
}

/** The workspace as a pixi project: problems/workspace/<slug>/ per problem (with its data/), notebooks/<slug>/*.ipynb,
 *  and workspace.json (everything, including cell outputs) so Import restores it exactly. Paths have no root folder. */
export async function workspaceFiles(ws: Workspace, manifest: Manifest, name: string): Promise<Record<string, Uint8Array>> {
  const { unzipSync, strToU8 } = await import("fflate");
  const root = "";
  const files: Record<string, Uint8Array> = {};
  const put = (path: string, text: string) => (files[root + path] = strToU8(text));

  // problems/_common comes from any library problem's pixi project (static, works without the backend).
  const sample = manifest.problems.find((p) => p.pixi_zip);
  if (sample?.pixi_zip) {
    try {
      const zip = unzipSync(new Uint8Array(await (await fetch(generatedUrl(sample.pixi_zip))).arrayBuffer()));
      for (const [path, data] of Object.entries(zip)) {
        const at = path.indexOf("problems/_common/");
        if (at >= 0 && path.endsWith(".py")) files[root + path.slice(at)] = data;
      }
    } catch {
      // without _common the problems still export; pixi run will ask for it
    }
  }
  put("problems/__init__.py", "");
  put("problems/workspace/__init__.py", "");
  const used = new Set<string>();
  const slugs: string[] = [];
  for (const p of ws.problems) {
    let slug = slugify(p.entry.title);
    for (let n = 2; used.has(slug); n++) slug = `${slugify(p.entry.title)}_${n}`;
    used.add(slug);
    slugs.push(slug);
    put(`problems/workspace/${slug}/__init__.py`, "");
    for (const [file, code] of Object.entries(filesOf(p))) put(`problems/workspace/${slug}/${file}`, code);
    for (const d of p.entry.data ?? []) {
      const blob = await getData(p.id, d.name);
      if (blob) files[`problems/workspace/${slug}/data/${d.name}`] = new Uint8Array(await blob.arrayBuffer());
    }
    const titles = new Set<string>();
    for (const nb of p.notebooks) {
      let file = slugify(nb.title) || "notebook";
      for (let n = 2; titles.has(file); n++) file = `${slugify(nb.title) || "notebook"}_${n}`;
      titles.add(file);
      put(`notebooks/${slug}/${file}.ipynb`, JSON.stringify(toIpynb(nb, slug), null, 1) + "\n");
    }
  }
  put("pixi.toml", pixiToml(`tensorizer-${slugify(name).replace(/_/g, "-") || "workspace"}`, slugs));
  put(
    "README.md",
    `# ${name}\n\nA [Tensorizer](https://github.com/jackyko1991/tensorizer) workspace.\n\n` +
      "Install [pixi](https://pixi.sh), then in this folder:\n\n```bash\n" +
      slugs.map((s) => `pixi run simulate-${s.replace(/_/g, "-")}`).join("\n") +
      "\npixi run notebooks\n```\n\nImport this .zip back into Tensorizer from the Workspace menu.\n",
  );
  // exportSlug: where each problem's folder (and data/) is, for Import
  const saved = { ...ws, problems: ws.problems.map((p, i) => ({ ...p, exportSlug: slugs[i] })) };
  put("workspace.json", JSON.stringify({ format: "tensorizer-workspace", version: 1, workspace: saved }) + "\n");
  return files;
}

export async function exportWorkspace(ws: Workspace, manifest: Manifest, name: string) {
  const { zipSync } = await import("fflate");
  const root = `${slugify(name) || "workspace"}/`;
  const files = Object.fromEntries(Object.entries(await workspaceFiles(ws, manifest, name)).map(([k, v]) => [root + k, v]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([zipSync(files)], { type: "application/zip" }));
  a.download = `${root.slice(0, -1)}.zip`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Problems from an exported .zip (workspace.json, or else its problem folders and .ipynb files), with new ids.
 *  Sessions are dropped: sandboxes are re-created from the files on the first run. */
export async function importWorkspace(file: File): Promise<WsProblem[]> {
  const { unzipSync, strFromU8 } = await import("fflate");
  const zip = unzipSync(new Uint8Array(await file.arrayBuffer()));
  const text = (path: string) => strFromU8(zip[path]);
  const fresh = (p: WsProblem): WsProblem => {
    const id = newId();
    return {
      ...p,
      id,
      entry: { ...p.entry, id: `workspace/${id}`, slug: id, session: undefined },
      notebooks: p.notebooks.map((nb) => ({ ...nb, id: newId() })),
    };
  };

  const saved = Object.keys(zip).find((p) => p.endsWith("workspace.json"));
  if (saved) {
    const data = JSON.parse(text(saved));
    if (data?.format === "tensorizer-workspace") {
      const root = saved.slice(0, -"workspace.json".length);
      const out: WsProblem[] = [];
      for (const old of (data.workspace as Workspace).problems as (WsProblem & { exportSlug?: string })[]) {
        const { exportSlug, ...rest } = old;
        const p = fresh(rest);
        const kept = [];
        for (const d of p.entry.data ?? []) {
          const bytes = zip[`${root}problems/workspace/${exportSlug}/data/${d.name}`];
          if (!bytes) continue;
          await putData(p.id, d.name, new Blob([bytes as BlobPart], { type: d.type }));
          kept.push(d);
        }
        out.push({ ...p, entry: { ...p.entry, data: kept } });
      }
      return out;
    }
  }
  // A hand-made project: one problem per problems/workspace/<slug>/meta.yaml.
  const out: WsProblem[] = [];
  for (const path of Object.keys(zip)) {
    const m = path.match(/^(.*?)problems\/workspace\/([^/]+)\/meta\.yaml$/);
    if (!m) continue;
    const [, root, slug] = m;
    const dir = `${root}problems/workspace/${slug}/`;
    const meta = text(path);
    const title = meta.match(/^title:\s*"?(.*?)"?\s*$/m)?.[1] ?? slug;
    const notebooks = Object.keys(zip)
      .filter((p) => p.startsWith(`${root}notebooks/${slug}/`) && p.endsWith(".ipynb"))
      .map((p) => fromIpynb(JSON.parse(text(p)), p.split("/").pop()!.replace(/\.ipynb$/, "")));
    const dataFiles = Object.keys(zip).filter((p) => p.startsWith(`${dir}data/`) && !p.endsWith("/"));
    const entry = {
      title,
      level: "beginner",
      summary: "",
      animated: false,
      vector_form: "",
      index_form: "",
      translation: [],
      params: {},
      order: 0,
      output: null,
      console: "",
      diagram: null,
      files: CODE_FILES.map((name) => ({ name, code: zip[dir + name] ? text(dir + name) : "" })),
    } as unknown as Problem;
    const p = fresh(toWsProblem(entry, meta, slug, notebooks));
    const data = [];
    for (const path of dataFiles) {
      const name = dataName(path.slice(`${dir}data/`.length));
      if (!dataExtOk(name)) continue;
      await putData(p.id, name, new Blob([zip[path] as BlobPart]));
      data.push({ name, size: zip[path].length, type: "" });
    }
    out.push({ ...p, entry: { ...p.entry, data } });
  }
  if (!out.length) throw new Error("no Tensorizer workspace in this .zip");
  return out;
}

function fromIpynb(nb: { cells: { cell_type: string; source: string | string[]; metadata?: { tensorizer?: string } }[]; metadata?: { tensorizer?: string } }, name: string): Notebook {
  const src = (s: string | string[]) => (Array.isArray(s) ? s.join("") : s);
  return {
    id: newId(),
    title: nb.metadata?.tensorizer ?? name,
    cells: nb.cells
      .filter((c) => c.metadata?.tensorizer !== "setup" && (c.cell_type === "code" || c.cell_type === "markdown"))
      .map((c) => newCell(c.cell_type as Cell["type"], src(c.source))),
  };
}
