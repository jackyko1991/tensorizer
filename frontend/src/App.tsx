import { useEffect, useMemo, useRef, useState } from "react";
import {
  assetUrl,
  backendChoice,
  hasJobs,
  health,
  jobActive,
  jobEvents,
  jobSeen,
  listJobs,
  loadManifest,
  markJobSeen,
  OPEN_JOB,
  RunFailed,
  runProblemJob,
  runProblemStream,
  setBackendChoice,
  takeOpenRequest,
  wsBlank,
  wsCopy,
  wsDelete,
  wsFiles,
  wsNotebook,
  wsPutData,
  wsRunStream,
  wsSync,
  type Health,
  type LineEvent,
  type RunResult,
  type TestTally,
  type WsEntry,
} from "./api";
import BackendPanel, { BackendButton } from "./components/BackendPanel";
import JobsButton from "./components/JobsPanel";
import CodePanel from "./components/CodePanel";
import EquationPanel, { FormulationExport } from "./components/EquationPanel";
import ParamsPanel from "./components/ParamsPanel";
import PlotPanel from "./components/PlotPanel";
import Sidebar, { LEVEL_STYLE } from "./components/Sidebar";
import TerminalPanel from "./components/TerminalPanel";
import { LANGS, localize, useLang, type Key, type Lang } from "./i18n";
import { emitContext, emitSelection, onApplyFiles, onApplyNotebook, openAgent, selectionReporter } from "./agentBus";
import AgentPanel from "./components/AgentPanel";
import ExportMenu from "./components/ExportMenu";
import { REPO_URL } from "./notebook";
import ConceptPage from "./pages/ConceptPage";
import { formatPageRoute, isTab, parsePageRoute, resolveTab, TAB_STORE } from "./pageRoute";
import AgentsPage from "./pages/AgentsPage";
import SkillsPage from "./pages/SkillsPage";
import { plainText, RichText } from "./components/Tex";
import NotebookView from "./components/NotebookView";
import DataPanel from "./components/DataPanel";
import SciCard from "./components/SciCard";
import SeriesNav from "./components/SeriesNav";
import { DataSources } from "./components/DataView";
import WorkspacePanel from "./components/WorkspacePanel";
import WorkspaceHome from "./components/WorkspaceHome";
import GitHubDialog from "./components/GitHubDialog";
import FloatingEditor, { EditorButtons, type FloatMode } from "./components/FloatingEditor";
import { Field, InlineEquations, TagInput } from "./components/WsEditor";
import DiagramExport from "./components/DiagramExport";
import PrizeBadges from "./components/PrizeBadges";
import { pictureToSvg, svgToDataUri } from "./diagramExport";
import { metaObject, pickMd, toMarkdown, toYaml, type MetaFields } from "./metaDoc";
import type { Level, Manifest, Notebook, Params, Problem, WsProblem } from "./types";
import {
  dataExtOk,
  dataName,
  deleteData,
  exportProblem,
  exportWorkspace,
  filesOf,
  getData,
  MAX_DATA_FILE,
  MAX_DATA_TOTAL,
  putData,
  importWorkspace,
  workspaceFiles,
  newCell,
  newId,
  patchProblem,
  toWsProblem,
  useWorkspaces,
  withTitle,
} from "./workspaceStore";

const currentHash = () => decodeURIComponent(window.location.hash.replace(/^#\/?/, "")) || null;
const select = (id: string) => (window.location.hash = `/${id}`);
const PAGES = ["problems", "concept", "skills", "agents"] as const;
type Page = (typeof PAGES)[number] | "workspace";
const LEFT_TAB = "tensorizer.leftTab";
const RECENT = "tensorizer.recent";
function stored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // storage blocked: remembered for this page only
  }
}
const TAB_KEY: Record<(typeof PAGES)[number], Key> = {
  problems: "tab.problems",
  concept: "tab.concept",
  skills: "tab.skills",
  agents: "tab.agents",
};

/**
 * Layout (desktop): a full-height workspace with three columns that scroll independently —
 *   1. foldable problem library   2. problem body (pinned title, equations, parameters)
 *   3. code editor, plot and terminal stacked top to bottom.
 * On small screens the columns stack and the library becomes a dropdown.
 */
export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(currentHash);
  const [backend, setBackend] = useState<Health | null | "checking">("checking");
  const [backendOpen, setBackendOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [paramsOpen, setParamsOpen] = useState(true);
  // The Workspace (IndexedDB): problems made blank, copied from the library or created by the AI, with notebooks.
  const { active: ws, update } = useWorkspaces();
  const wsRef = useRef(ws);
  wsRef.current = ws;
  const { lang, setLang, t } = useLang();

  useEffect(() => {
    loadManifest().then(setManifest, (e: Error) => setLoadError(e.message));
    health().then(setBackend);
    const onHash = () => setSelected(currentHash());
    window.addEventListener("hashchange", onHash);
    // Dev server: problem files changed → re-fetch the manifest in place (no page reload).
    const onManifest = () => loadManifest().then(setManifest, () => {});
    import.meta.hot?.on("tensorizer:manifest", onManifest);
    return () => {
      window.removeEventListener("hashchange", onHash);
      import.meta.hot?.off("tensorizer:manifest", onManifest);
    };
  }, []);

  // A problem that moved to another category (e.g. others/tiny_transformer → ai/tiny_transformer): old links follow
  // it by its slug.
  useEffect(() => {
    if (!manifest || !selected || !selected.includes("/") || selected.startsWith("workspace/")) return;
    if (manifest.problems.some((p) => p.id === selected)) return;
    const slug = selected.split("/").pop();
    const moved = manifest.problems.find((p) => p.slug === slug);
    if (moved) window.location.replace(`#/${moved.id}`);
  }, [manifest, selected]);

  // #/concept[/<tab>[/<section>]] and #/skills[/<tab>[/<section>]] are pages with sub-tabs, #/agents and #/workspace
  // pages; any other hash is a problem id (category/slug), a workspace problem (workspace/<id>) or one of its
  // notebooks (workspace/<id>/nb/<notebook>).
  const pageRoute = parsePageRoute(selected);
  const page: Page = pageRoute
    ? pageRoute.page
    : selected === "agents" || selected === "workspace"
      ? selected
      : "problems";
  // The sub-tab: the address wins; #/skills alone opens the remembered tab. A missing or unknown tab is rewritten
  // in place (no new history entry) to the canonical #/<page>/<tab>, so back/forward always land on a tab.
  const subTab = pageRoute ? resolveTab(pageRoute.page, pageRoute.tab, stored(TAB_STORE[pageRoute.page])) : null;
  useEffect(() => {
    if (!pageRoute || !subTab) return;
    store(TAB_STORE[pageRoute.page], subTab);
    if (pageRoute.tab !== subTab) window.location.replace(`#${formatPageRoute(pageRoute.page, subTab)}`);
  }, [pageRoute?.page, pageRoute?.tab, subTab]);
  const openTab = (tab: string) => pageRoute && (window.location.hash = formatPageRoute(pageRoute.page, tab));
  // Left panel: the library (Problems) or the Workspace tree, remembered; opening a workspace problem shows it.
  const [leftTab, setLeftTabState] = useState<"problems" | "workspace">(() =>
    stored(LEFT_TAB) === "workspace" ? "workspace" : "problems",
  );
  const setLeftTab = (tab: "problems" | "workspace") => (setLeftTabState(tab), store(LEFT_TAB, tab));
  const [github, setGithub] = useState(false);
  // Recently viewed library problems, for the Workspace home.
  const [recentIds, setRecentIds] = useState<string[]>(() => {
    try {
      return JSON.parse(stored(RECENT) ?? "[]");
    } catch {
      return [];
    }
  });
  const [lastProblemId, setLastProblemId] = useState<string | null>(null);
  const nbRoute = selected?.match(/^workspace\/([^/]+)\/nb\/([^/]+)$/) ?? null;
  const dataRoute = selected?.match(/^workspace\/([^/]+)\/data$/) ?? null;
  const problemKey = nbRoute ? `workspace/${nbRoute[1]}` : dataRoute ? `workspace/${dataRoute[1]}` : selected;
  const wsProblems = ws?.problems ?? [];
  const problem =
    [...wsProblems.map((p) => p.entry), ...(manifest?.problems ?? [])].find(
      (p) => p.id === (page === "problems" ? problemKey : lastProblemId),
    ) ?? manifest?.problems[0];
  const wsProblem = wsProblems.find((p) => p.entry.id === problem?.id) ?? null;
  const notebook = nbRoute ? wsProblem?.notebooks.find((nb) => nb.id === nbRoute[2]) : undefined;
  useEffect(() => {
    if (page === "problems" && problem) setLastProblemId(problem.id);
  }, [page, problem]);
  useEffect(() => {
    if (page !== "problems" || !problem) return;
    if (problem.category === "workspace") return setLeftTab("workspace");
    setRecentIds((ids) => {
      const next = [problem.id, ...ids.filter((i) => i !== problem.id)].slice(0, 6);
      store(RECENT, JSON.stringify(next));
      return next;
    });
  }, [page, problem?.id]);
  const goTo = (target: Page) =>
    (window.location.hash = target === "problems" ? `/${problem?.id ?? ""}` : `/${target}`);

  // --- Workspace actions (the browser holds the files; the server keeps a disposable sandbox per problem) ---
  const findWs = (id: string) => wsRef.current?.problems.find((p) => p.id === id);
  const addProblem = (p: WsProblem) => {
    update((w) => ({ ...w, problems: [p, ...w.problems] }));
    select(p.entry.id);
  };
  const create = (make: () => Promise<WsEntry>) =>
    make().then(
      (entry) => addProblem(toWsProblem(entry, entry.meta)),
      () => alert(t("workspace.unavailable")),
    );
  /** Push the problem's files to its sandbox (re-created if it expired); returns the live session. */
  async function syncProblem(id: string): Promise<string> {
    const p = findWs(id);
    if (!p) throw new Error("unknown problem");
    const data = p.entry.data ?? [];
    const entry = await wsSync(
      p.entry.session ?? null,
      filesOf(p),
      data.map((d) => d.name),
    );
    // data files the sandbox doesn't have yet (new, or the sandbox was re-created) go up after the files
    const there = new Map(entry.data.map((d) => [d.name, d.size]));
    for (const d of data) {
      if (there.get(d.name) === d.size) continue;
      const blob = await getData(p.id, d.name);
      if (blob) await wsPutData(entry.session, d.name, blob);
    }
    update((w) =>
      patchProblem(w, id, (q) => ({
        ...q,
        meta: q.meta || entry.meta,
        // parsed meta.yaml fields come from the server; the files stay the browser's
        entry: {
          ...entry,
          id: q.entry.id,
          slug: q.entry.slug,
          files: q.entry.files,
          data: q.entry.data,
          output: q.entry.output,
          diagram: entry.diagram ?? q.entry.diagram,
        },
      })),
    );
    return entry.session;
  }
  const patchFiles = (id: string, files: Record<string, string>) =>
    update((w) =>
      patchProblem(w, id, (q) => ({
        ...q,
        meta: files["meta.yaml"] ?? q.meta,
        entry: { ...q.entry, files: q.entry.files.map((f) => (f.name in files ? { ...f, code: files[f.name] } : f)) },
      })),
    );
  /** Fields edited in the view (inline or Markdown): update the entry and regenerate meta.yaml from it. */
  const patchMeta = (id: string, fields: Partial<MetaFields>) =>
    update((w) =>
      patchProblem(w, id, (q) => {
        const entry = { ...q.entry, ...fields };
        return { ...q, entry, meta: toYaml(metaObject(entry)) };
      }),
    );
  function renameProblem(id: string) {
    const p = findWs(id);
    const title = p && prompt(t("ws.renamePrompt"), plainText(p.entry.title))?.trim();
    if (!title) return;
    update((w) => patchProblem(w, id, (q) => ({ ...q, meta: withTitle(q.meta, title), entry: { ...q.entry, title } })));
  }
  function deleteProblem(id: string) {
    const p = findWs(id);
    if (!p || !confirm(t("ws.deleteConfirm").replace("{name}", plainText(p.entry.title)))) return;
    if (p.entry.session) wsDelete(p.entry.session).catch(() => {});
    update((w) => ({ ...w, problems: w.problems.filter((q) => q.id !== id) }));
    if (problemKey === p.entry.id && manifest) select(manifest.problems[0].id);
  }
  function addNotebook(id: string) {
    const p = findWs(id);
    if (!p) return;
    const nb: Notebook = {
      id: newId(),
      title: `${t("ws.notebook")} ${p.notebooks.length + 1}`,
      cells: [
        newCell("markdown", `# ${p.entry.title}\n\n${t("notebook.starter")}`),
        newCell("code", "# from problem import equation, solve, plot\nprint(params)\n"),
      ],
    };
    update((w) => patchProblem(w, id, (q) => ({ ...q, notebooks: [...q.notebooks, nb] })));
    select(`${p.entry.id}/nb/${nb.id}`);
  }
  /** Uploaded data: kept in the browser (IndexedDB), then synced into the sandbox's data/ folder. */
  async function uploadData(id: string, files: File[]) {
    const p = findWs(id);
    if (!p) return;
    const list = [...(p.entry.data ?? [])];
    for (const file of files) {
      const name = dataName(file.name);
      if (!dataExtOk(name)) {
        alert(t("data.badType").replace("{name}", file.name));
        continue;
      }
      if (file.size > MAX_DATA_FILE) {
        alert(t("data.tooBig").replace("{name}", file.name));
        continue;
      }
      const total = list.filter((d) => d.name !== name).reduce((n, d) => n + d.size, 0) + file.size;
      if (total > MAX_DATA_TOTAL) {
        alert(t("data.full"));
        break;
      }
      await putData(id, name, file);
      const entry = { name, size: file.size, type: file.type };
      const at = list.findIndex((d) => d.name === name);
      if (at >= 0) list[at] = entry;
      else list.push(entry);
    }
    update((w) => patchProblem(w, id, (q) => ({ ...q, entry: { ...q.entry, data: list } })));
    await new Promise((r) => setTimeout(r, 0)); // let the store update before syncing
    await syncProblem(id).catch((e: Error) => alert(e.message));
  }
  async function renameData(id: string, name: string) {
    const p = findWs(id);
    const next = p && prompt(t("data.renamePrompt"), name)?.trim();
    if (!p || !next || next === name) return;
    const clean = dataName(next);
    if (!dataExtOk(clean)) return alert(t("data.badType").replace("{name}", next));
    const blob = await getData(id, name);
    if (blob) await putData(id, clean, blob);
    await deleteData(id, name);
    update((w) =>
      patchProblem(w, id, (q) => ({
        ...q,
        entry: {
          ...q.entry,
          data: (q.entry.data ?? [])
            .filter((d) => d.name !== clean)
            .map((d) => (d.name === name ? { ...d, name: clean } : d)),
        },
      })),
    );
    setTimeout(() => syncProblem(id).catch(() => {}), 0);
  }
  async function removeData(id: string, name: string) {
    if (!confirm(t("ws.deleteNotebookConfirm").replace("{name}", name))) return;
    await deleteData(id, name);
    update((w) =>
      patchProblem(w, id, (q) => ({
        ...q,
        entry: { ...q.entry, data: (q.entry.data ?? []).filter((d) => d.name !== name) },
      })),
    );
    setTimeout(() => syncProblem(id).catch(() => {}), 0);
  }
  function renameNotebook(id: string, nbId: string) {
    const nb = findWs(id)?.notebooks.find((n) => n.id === nbId);
    const title = nb && prompt(t("ws.renameNotebookPrompt"), nb.title)?.trim();
    if (title) patchNotebook(id, nbId, (n) => ({ ...n, title }));
  }
  function deleteNotebook(id: string, nbId: string) {
    const p = findWs(id);
    const nb = p?.notebooks.find((n) => n.id === nbId);
    if (!p || !nb || !confirm(t("ws.deleteNotebookConfirm").replace("{name}", nb.title || t("ws.notebook")))) return;
    update((w) => patchProblem(w, id, (q) => ({ ...q, notebooks: q.notebooks.filter((n) => n.id !== nbId) })));
    if (selected === `${p.entry.id}/nb/${nbId}`) select(p.entry.id); // the open notebook → its Source
  }
  const patchNotebook = (id: string, nbId: string, fn: (nb: Notebook) => Notebook) =>
    update((w) =>
      patchProblem(w, id, (q) => ({ ...q, notebooks: q.notebooks.map((nb) => (nb.id === nbId ? fn(nb) : nb)) })),
    );
  // The agent's edits to a Workspace problem (files, re-parsed meta.yaml, notebook cells) → the browser store,
  // so the Inline / Markdown views, the meta.yaml editor, the code editor and the notebook all update.
  useEffect(
    () =>
      onApplyFiles(({ problem: entryId, files, entry }) => {
        const p = wsRef.current?.problems.find((q) => q.entry.id === entryId);
        if (!p) return;
        const parsed = entry
          ? Object.fromEntries(
              (
                [
                  "title",
                  "level",
                  "summary",
                  "animated",
                  "vector_form",
                  "index_form",
                  "translation",
                  "notes",
                  "params",
                  "meta_extra",
                ] as const
              ).map((k) => [k, entry[k]]),
            )
          : {};
        update((w) =>
          patchProblem(w, p.id, (q) => ({
            ...q,
            meta: files["meta.yaml"] ?? q.meta,
            entry: {
              ...q.entry,
              ...parsed,
              diagram: entry?.diagram ?? q.entry.diagram,
              files: q.entry.files.map((f) => (f.name in files ? { ...f, code: files[f.name] } : f)),
            },
          })),
        );
      }),
    [update],
  );
  useEffect(
    () =>
      onApplyNotebook(({ problem: entryId, notebook: nbId, cells }) => {
        const p = wsRef.current?.problems.find((q) => q.entry.id === entryId);
        if (!p) return;
        patchNotebook(p.id, nbId, (nb) => {
          const old = new Map(nb.cells.map((c) => [c.id, c]));
          return {
            ...nb,
            cells: cells.map((c) => {
              const prev = c.id ? old.get(c.id) : undefined;
              // a cell the agent kept: same id; its output stays while its source is unchanged
              if (prev && prev.type === c.type)
                return prev.source === c.source ? prev : { ...prev, source: c.source, output: null };
              return newCell(c.type, c.source);
            }),
          };
        });
      }),
    [update],
  );
  /** The AI created (or changed) a problem: keep it in the workspace, matched by its agent session. */
  function onAiProblem(entry: Problem) {
    const existing = wsRef.current?.problems.find((p) => p.entry.session === entry.session);
    const p = existing
      ? { ...existing, entry: { ...entry, id: existing.entry.id, category: "workspace", slug: existing.entry.slug } }
      : toWsProblem(entry, "");
    if (existing) update((w) => patchProblem(w, p.id, () => p));
    else update((w) => ({ ...w, problems: [p, ...w.problems] }));
    select(p.entry.id);
    if (entry.session)
      wsFiles(entry.session).then(
        (files) => update((w) => patchProblem(w, p.id, (q) => ({ ...q, meta: files["meta.yaml"] ?? q.meta }))),
        () => {},
      );
  }
  async function importZip(file: File) {
    try {
      const problems = await importWorkspace(file);
      update((w) => ({ ...w, problems: [...problems, ...w.problems] }));
      if (problems[0]) select(problems[0].entry.id);
      // parse each meta.yaml on the server once the new problems are in the store
      setTimeout(() => problems.forEach((p) => syncProblem(p.id).catch(() => {})), 0);
    } catch (e) {
      alert(t("ws.importFailed").replace("{error}", e instanceof Error ? e.message : String(e)));
    }
  }
  const workspacePanel = (className: string) =>
    manifest && (
      <WorkspacePanel
        className={className}
        manifest={manifest}
        workspace={ws}
        selected={page === "problems" ? selected : null}
        onSelect={select}
        onNewBlank={() => create(wsBlank)}
        onCopy={(id) => create(() => wsCopy(id))}
        onRename={renameProblem}
        onDelete={deleteProblem}
        onAddNotebook={addNotebook}
        onRenameNotebook={renameNotebook}
        onDeleteNotebook={deleteNotebook}
        onExport={() => ws && manifest && exportWorkspace(ws, manifest, ws.name || t("ws.default"))}
        onExportProblem={(id) => ws && manifest && void exportProblem(ws, manifest, id)}
        onImport={importZip}
        onGitHub={() => setGithub(true)}
      />
    );

  return (
    <div className="flex min-h-dvh flex-col bg-white xl:h-dvh xl:overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1.5 bg-blue-700 px-4 py-2 text-white shadow-sm sm:px-6 lg:h-[52px] lg:flex-nowrap lg:py-0">
        <div className="flex min-w-0 items-center gap-3">
          <button
            onClick={() => (window.location.hash = "/workspace")}
            title={t("home.open")}
            aria-label={t("home.open")}
            className="flex shrink-0 items-center gap-3 rounded-md hover:opacity-90"
            data-testid="logo"
          >
            <img src={`${import.meta.env.BASE_URL}logo-light.svg`} alt="" className="h-8 w-8" />
            <h1 className="text-lg font-semibold tracking-tight">
              Tensor<span className="text-blue-200">izer</span>
            </h1>
          </button>
          <span className="hidden text-sm text-blue-100 xl:inline">{t("tagline")}</span>
        </div>
        {/* On phones the tabs wrap onto their own row under the logo and language menu. */}
        <nav
          className="order-last flex w-full items-center gap-1 overflow-x-auto lg:order-none lg:ml-auto lg:w-auto"
          aria-label="Sections"
        >
          {PAGES.map((target) => (
            <button
              key={target}
              onClick={() => goTo(target)}
              aria-current={page === target ? "page" : undefined}
              className={`h-8 shrink-0 rounded-md px-3 text-sm font-medium transition ${
                page === target
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-blue-100 hover:bg-blue-600 hover:text-white"
              }`}
            >
              {t(TAB_KEY[target])}
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-4 lg:ml-0">
          {hasJobs(backend) && (
            <JobsButton
              titleOf={(id) => {
                const p = manifest?.problems.find((q) => q.id === id);
                return p ? plainText(localize(p, lang).title) : id;
              }}
            />
          )}
          <BackendButton backend={backend} onOpen={() => setBackendOpen(true)} />
          {backendOpen && (
            <BackendPanel
              onClose={() => {
                setBackendOpen(false);
                setBackend("checking"); // the address or device may have changed
                health().then(setBackend);
              }}
              onChoose={(choice) => {
                setBackendChoice(choice);
                setBackend("checking");
                health().then(setBackend);
              }}
            />
          )}
          <select
            value={lang}
            onChange={(e) => setLang(e.target.value as Lang)}
            title={t("language")}
            aria-label={t("language")}
            className="h-8 rounded-md border border-blue-400 bg-blue-600 px-2 text-xs text-white hover:bg-blue-500"
          >
            {LANGS.map((l) => (
              <option key={l.code} value={l.code} className="bg-white text-gray-900">
                {l.name}
              </option>
            ))}
          </select>
        </div>
      </header>

      {loadError && <p className="m-6 rounded-md bg-red-50 p-4 text-sm text-red-700">{loadError}</p>}

      {manifest && page === "concept" && (
        <main className="min-h-0 flex-1">
          <ConceptPage
            manifest={manifest}
            tab={isTab("concept", subTab) ? subTab : "laws"}
            section={pageRoute?.section ?? null}
            onTab={openTab}
          />
        </main>
      )}
      {manifest && page === "skills" && (
        <main className="min-h-0 flex-1">
          <SkillsPage
            manifest={manifest}
            tab={isTab("skills", subTab) ? subTab : "techniques"}
            section={pageRoute?.section ?? null}
            onTab={openTab}
          />
        </main>
      )}
      {page === "agents" && (
        <main className="min-h-0 flex-1">
          <AgentsPage />
        </main>
      )}
      {manifest && page === "workspace" && (
        <main className="min-h-0 flex-1 overflow-y-auto bg-white">
          <WorkspaceHome
            manifest={manifest}
            workspace={ws}
            recent={recentIds.map((id) => manifest.problems.find((p) => p.id === id)).filter((p): p is Problem => !!p)}
            onOpen={select}
            onRenameWorkspace={() => {
              const name = prompt(t("home.renamePrompt"), ws?.name || t("ws.default"))?.trim();
              if (name) update((w) => ({ ...w, name }));
            }}
            onNewBlank={() => create(wsBlank)}
            onCopy={(id) => create(() => wsCopy(id))}
            onAskAI={() => openAgent("new")}
            onRename={renameProblem}
            onDelete={deleteProblem}
            onExport={() => ws && exportWorkspace(ws, manifest, ws.name || t("ws.default"))}
            onImport={importZip}
            onGitHub={() => setGithub(true)}
          />
        </main>
      )}
      {github && ws && manifest && (
        <GitHubDialog
          defaultName={`tensorizer-${
            (ws.name || "workspace")
              .toLowerCase()
              .replace(/[^a-z0-9]+/g, "-")
              .replace(/^-|-$/g, "") || "workspace"
          }`}
          files={() => workspaceFiles(ws, manifest, ws.name || t("ws.default"))}
          onClose={() => setGithub(false)}
        />
      )}

      {manifest && problem && page === "problems" && (
        <div className="flex flex-1 flex-col xl:min-h-0 xl:flex-row">
          <div className="border-b border-gray-200 p-3 xl:hidden">
            <ProblemSelect manifest={manifest} value={problem.id} />
            <details className="mt-2 rounded-md border border-gray-200" open={!!wsProblem}>
              <summary className="cursor-pointer px-3 py-1.5 text-sm font-medium text-gray-700">
                {t("workspace")} ({wsProblems.length})
              </summary>
              {workspacePanel("max-h-[60vh]")}
            </details>
          </div>

          {/* 1. Problem library */}
          <aside
            className={`hidden shrink-0 flex-col border-r border-gray-200 bg-white transition-[width] xl:flex ${
              libraryOpen ? "w-64" : "w-10"
            }`}
          >
            <div className="flex h-9 shrink-0 items-center gap-1 border-b border-gray-200 bg-gray-50 px-1.5">
              {libraryOpen && (
                <div role="tablist" className="grid flex-1 grid-cols-2 gap-1" data-testid="left-tabs">
                  {(["problems", "workspace"] as const).map((tab) => (
                    <button
                      key={tab}
                      role="tab"
                      aria-selected={leftTab === tab}
                      onClick={() => setLeftTab(tab)}
                      className={`rounded-md px-2 py-1 text-xs font-medium transition ${
                        leftTab === tab
                          ? "bg-blue-700 text-white shadow-sm"
                          : "text-gray-600 hover:bg-gray-200 hover:text-gray-900"
                      }`}
                    >
                      {tab === "problems"
                        ? t("tab.problems")
                        : `${t("workspace")}${wsProblems.length ? ` (${wsProblems.length})` : ""}`}
                    </button>
                  ))}
                </div>
              )}
              <button
                onClick={() => setLibraryOpen((o) => !o)}
                title={t(libraryOpen ? "library.fold" : "library.open")}
                className="rounded px-1.5 text-gray-500 hover:bg-gray-200 hover:text-gray-800"
              >
                {libraryOpen ? "«" : "»"}
              </button>
            </div>
            {libraryOpen ? (
              leftTab === "problems" ? (
                <div className="min-h-0 flex-1 overflow-y-auto p-3">
                  <Sidebar
                    manifest={manifest}
                    selected={problem.id}
                    onSelect={select}
                    onFork={(id) => create(() => wsCopy(id))}
                  />
                </div>
              ) : (
                // Workspace: blank problems, library copies and AI-created ones, each with Source, Data, notebooks
                workspacePanel("min-h-0 flex-1")
              )
            ) : (
              <button
                onClick={() => setLibraryOpen(true)}
                className="flex-1 py-3 text-xs font-medium tracking-wide text-gray-500 [writing-mode:vertical-rl] hover:text-gray-800"
              >
                {t("library")}
              </button>
            )}
          </aside>

          {wsProblem && dataRoute ? (
            <DataPanel
              problemId={wsProblem.entry.id}
              localId={wsProblem.id}
              problemTitle={wsProblem.entry.title}
              params={defaults(wsProblem.entry)}
              files={wsProblem.entry.data ?? []}
              onUpload={(files) => uploadData(wsProblem.id, files)}
              onRename={(name) => renameData(wsProblem.id, name)}
              onDelete={(name) => removeData(wsProblem.id, name)}
            />
          ) : wsProblem && nbRoute && notebook ? (
            <NotebookView
              key={notebook.id}
              notebook={notebook}
              problemId={wsProblem.entry.id}
              problemTitle={wsProblem.entry.title}
              params={defaults(wsProblem.entry)}
              canRun={!!backend && backend !== "checking"}
              onChange={(fn) => patchNotebook(wsProblem.id, notebook.id, fn)}
              onDelete={() => {
                update((w) =>
                  patchProblem(w, wsProblem.id, (q) => ({
                    ...q,
                    notebooks: q.notebooks.filter((nb) => nb.id !== notebook.id),
                  })),
                );
                select(wsProblem.entry.id);
              }}
              onRun={async (code, all) => wsNotebook(await syncProblem(wsProblem.id), code, all)}
            />
          ) : (
            <ProblemView
              key={problem.id}
              problem={localize(problem, lang)}
              manifest={manifest}
              backend={backend === "checking" ? null : backend}
              categoryTitle={t(`cat.${problem.category}` as Key) ?? problem.category}
              paramsOpen={paramsOpen}
              onToggleParams={() => setParamsOpen((o) => !o)}
              onFork={wsProblem ? undefined : () => create(() => wsCopy(problem.id))}
              ws={
                wsProblem
                  ? {
                      meta: wsProblem.meta,
                      onFiles: (files) => patchFiles(wsProblem.id, files),
                      onMeta: (fields) => patchMeta(wsProblem.id, fields),
                      allTags: [...new Set(wsProblems.flatMap((q) => q.entry.tags ?? []))].sort(),
                      sync: () => syncProblem(wsProblem.id),
                      onExport: () => ws && manifest && void exportProblem(ws, manifest, wsProblem.id),
                      onDiagram: (svg) => {
                        const id = wsProblem.id;
                        update((w) =>
                          patchProblem(w, id, (q) => ({ ...q, entry: { ...q.entry, diagram: svgToDataUri(svg) } })),
                        );
                        // after the store has the picture: the sandbox checks it (no scripts) and keeps it
                        window.setTimeout(() => syncProblem(id).catch((e: Error) => alert(e.message)), 100);
                      },
                      onOutput: (url) =>
                        update((w) =>
                          patchProblem(w, wsProblem.id, (q) => ({ ...q, entry: { ...q.entry, output: url } })),
                        ),
                    }
                  : undefined
              }
            />
          )}
        </div>
      )}

      <SciCard />

      {manifest && problem && (
        <AgentPanel
          problem={problem}
          backendKey={`${backendChoice()}:${backend === "checking" ? "checking" : backend ? "up" : "down"}`}
          onWorkspaceProblem={onAiProblem}
          onSyncWorkspace={(entryId) => {
            const p = wsRef.current?.problems.find((q) => q.entry.id === entryId);
            return p ? syncProblem(p.id) : Promise.reject(new Error("unknown problem"));
          }}
        />
      )}

      <footer className="sticky bottom-0 z-10 flex h-7 shrink-0 items-center justify-end gap-4 border-t border-gray-200 bg-white/95 px-4 text-xs text-gray-500 backdrop-blur">
        <a
          href="https://buymeacoffee.com/jackyko1991"
          target="_blank"
          rel="noreferrer"
          className="rounded-full bg-yellow-300 px-2.5 py-0.5 font-medium text-gray-900 hover:bg-yellow-400"
        >
          ☕ {t("coffee")}
        </a>
        <span>Apache-2.0</span>
        <a href={REPO_URL} target="_blank" rel="noreferrer" className="flex items-center gap-1.5 hover:text-gray-900">
          <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
          GitHub
        </a>
      </footer>
    </div>
  );
}

function ProblemSelect({ manifest, value }: { manifest: Manifest; value: string }) {
  const { lang, t } = useLang();
  return (
    <select
      className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm"
      value={value}
      onChange={(e) => select(e.target.value)}
    >
      {manifest.categories.flatMap((cat) =>
        manifest.levels.map((level) => {
          const inGroup = manifest.problems.filter((p) => p.category === cat.id && p.level === level);
          return (
            inGroup.length > 0 && (
              <optgroup
                key={`${cat.id}-${level}`}
                label={`${t(`cat.${cat.id}` as Key) ?? cat.title} · ${t(LEVEL_STYLE[level].key)}`}
              >
                {inGroup.map((p) => (
                  <option key={p.id} value={p.id}>
                    {plainText(p.i18n?.[lang]?.title ?? p.title)}
                  </option>
                ))}
              </optgroup>
            )
          );
        }),
      )}
    </select>
  );
}

/** Status dot + label, and a picker: the shared server or a backend running on this computer. */
function defaults(problem: Problem): Params {
  return Object.fromEntries(Object.entries(problem.params).map(([k, s]) => [k, s.default]));
}

const MOBILE_TABS = ["problem", "formulation", "code", "results"] as const;
type MobileTab = (typeof MOBILE_TABS)[number];

interface ProblemViewProps {
  problem: Problem;
  /** For the series step navigator under a library problem's summary. */
  manifest?: Manifest | null;
  backend: Health | null;
  categoryTitle: string;
  paramsOpen: boolean;
  onToggleParams: () => void;
  /** Library problems: copy into the Workspace to edit (library code itself is read-only). */
  onFork?: () => void;
  /** Workspace problems: the files are edited in place (saved in the browser) and synced to a sandbox to run. */
  ws?: {
    meta: string;
    onFiles: (files: Record<string, string>) => void;
    onMeta: (fields: Partial<MetaFields>) => void;
    /** every tag used in the workspace, suggested while typing a new one */
    allTags: string[];
    sync: () => Promise<string>;
    /** A successful run's result, kept as the problem's thumbnail. */
    onOutput?: (url: string) => void;
    /** download the whole problem as one zip (pixi project) */
    onExport: () => void;
    /** a new physical picture (diagram.svg text), uploaded by the user */
    onDiagram: (svg: string) => void;
  };
}

function ProblemView({
  problem,
  manifest,
  backend,
  categoryTitle,
  paramsOpen,
  onToggleParams,
  onFork,
  ws,
}: ProblemViewProps) {
  const [params, setParams] = useState<Params>(() => defaults(problem));
  // lower panel of a library problem: the formulation, or the real data bundled with it
  const [lowerTab, setLowerTab] = useState<"formulation" | "data">("formulation");
  // ⤢ the formulation over the whole page (larger maths, scrolls); Esc or ⤡ brings it back
  const [formulaBig, setFormulaBig] = useState(false);
  useEffect(() => {
    if (!formulaBig) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setFormulaBig(false);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [formulaBig]);
  // A workspace problem's meta.yaml may add or remove parameters: start again from the new defaults.
  const paramSpecs = JSON.stringify(problem.params);
  const lastSpecs = useRef(paramSpecs);
  useEffect(() => {
    if (paramSpecs !== lastSpecs.current) setParams(defaults(problem));
    lastSpecs.current = paramSpecs;
  }, [paramSpecs, problem]);
  // Editing meta.yaml re-parses it on the server after a pause (title, equations and parameters update).
  const [metaError, setMetaError] = useState<string | null>(null);
  // Workspace: the floating Markdown / meta.yaml editor (null = closed; the inline view is always shown)
  const [floating, setFloating] = useState<FloatMode | null>(null);
  const onMeta = (fields: Partial<MetaFields>) => {
    ws?.onMeta(fields);
    setMetaError(null); // meta.yaml is regenerated from the view, so it is valid again
  };
  const syncTimer = useRef<number | undefined>(undefined);
  function editMeta(code: string) {
    ws?.onFiles({ "meta.yaml": code });
    window.clearTimeout(syncTimer.current);
    syncTimer.current = window.setTimeout(
      () =>
        ws?.sync().then(
          () => setMetaError(null),
          (e: Error) => setMetaError(e.message),
        ),
      800,
    );
  }
  useEffect(() => () => window.clearTimeout(syncTimer.current), []);
  const [edits, setEdits] = useState<Record<string, string>>({});
  // Claude's edits arrive here; and the Ask AI panel always knows what is on screen.
  useEffect(() => {
    return onApplyFiles(({ problem: id, files }) => {
      if (id === problem.id && !ws) setEdits((e) => ({ ...e, ...files })); // Workspace: App updates the store
    });
  }, [problem.id, !ws]);
  const [src, setSrc] = useState<string | null>(problem.output ? assetUrl(problem.output) : null);
  const [console, setConsole] = useState(problem.console);
  const [running, setRunning] = useState(false);
  const [verbose, setVerbose] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // test.py's verdict on the last run: the TDD light (null = not run here yet)
  const [tests, setTests] = useState<TestTally | null>(null);
  // drawing progress (GIF frames rendered), 0..1, while a run renders
  const [progress, setProgress] = useState<number | null>(null);
  // Workspace: which editor the user is in, for the AI panel — the floating one when open, otherwise inline.
  const wsMode = floating ?? "inline";
  const markdown =
    ws && wsMode === "markdown" ? toMarkdown(pickMd({ ...problem, notes: problem.notes ?? "" })) : undefined;
  useEffect(() => {
    emitContext({
      problem: problem.id,
      title: plainText(problem.title),
      edits,
      params,
      workspace: ws ? { kind: "source", mode: wsMode, markdown } : undefined,
      run: { output: !!src, failed: !!error },
    });
  }, [problem.id, problem.title, edits, params, !ws, wsMode, markdown, !!src, !!error]);
  const yamlSelection = useMemo(
    () => selectionReporter({ problem: problem.id, file: "meta.yaml", region: "meta.yaml", lang: "yaml" }),
    [problem.id],
  );
  useEffect(() => () => emitSelection(null), [problem.id]); // the meta.yaml editor's highlight goes with the view
  const [hasRun, setHasRun] = useState(false);
  // Phones show one part of the workspace at a time; desktop shows all three columns.
  const [mobileTab, setMobileTab] = useState<MobileTab>("problem");
  const onPhone = (tab: MobileTab, display = "xl:block") => (mobileTab === tab ? "" : `hidden ${display}`);

  // A refreshed manifest may bring a newly rendered default result; show it unless the user ran their own.
  useEffect(() => {
    if (hasRun || running) return;
    setSrc(problem.output ? assetUrl(problem.output) : null);
    setConsole(problem.console);
  }, [problem.output, problem.console, hasRun, running]);

  const inWorkspace = !!ws;
  // Library code is read-only (fork it to the Workspace to change it); Workspace problems run in their own sandbox.
  const editable = inWorkspace;
  const hasEdits = Object.keys(edits).length > 0;

  // Library runs go through detached jobs when the backend has them: the run keeps going if the page closes, and
  // reopening the problem reattaches to it (or shows how it ended). `follow` stops watching, not the job.
  const jobs = !ws && hasJobs(backend);
  const follow = useRef<AbortController | null>(null);
  const runningRef = useRef(running);
  runningRef.current = running;
  useEffect(() => () => follow.current?.abort(), []);

  /** Clear the results and return the handlers that stream a run's output into the terminal. */
  function startOutput() {
    setRunning(true);
    setHasRun(true);
    setMobileTab("results");
    setError(null);
    setTests(null);
    setProgress(null);
    setConsole("");
    // Lines stream in as the run prints; a progress-bar redraw overwrites the previous redraw.
    const lines: string[] = [];
    let lastWasRedraw = false;
    const onLine = ({ text, replace }: LineEvent) => {
      if (lastWasRedraw) lines.pop();
      lines.push(text);
      lastWasRedraw = replace;
      setConsole(lines.join("\n"));
    };
    // While waiting for a free slot on a shared server, one line shows the place in the queue.
    const onQueued = (position: number) =>
      onLine({
        text: t("queue.wait").replace("{n}", String(position)),
        replace: true,
      });
    const controller = new AbortController();
    follow.current?.abort(); // stop watching an earlier job (it keeps running on the server)
    follow.current = controller;
    return { onLine, onQueued, signal: controller.signal };
  }

  async function showResult(pending: Promise<RunResult>, signal: AbortSignal, jobId?: () => string | undefined) {
    const current = () => follow.current?.signal === signal;
    try {
      const result = await pending;
      if (!current()) return;
      setSrc(result.output);
      setConsole(result.console);
      setTests(result.tests ?? null);
      ws?.onOutput?.(result.output);
    } catch (e) {
      if (!current()) return; // another run or job took over the panels
      setError(e instanceof Error ? e.message : String(e));
      if (e instanceof RunFailed && e.console) setConsole(e.console);
    } finally {
      const id = jobId?.();
      if (id && !signal.aborted) markJobSeen(id); // its outcome was shown here
      if (current()) {
        setRunning(false);
        setProgress(null);
      }
    }
  }

  async function run() {
    const { onLine, onQueued, signal } = startOutput();
    const files = hasEdits ? edits : undefined;
    let jobId: string | undefined;
    const pending = ws
      ? ws.sync().then((session) => wsRunStream(session, params, verbose, onLine, onQueued, setProgress))
      : jobs
        ? runProblemJob(problem.id, params, files, verbose, onLine, onQueued, setProgress, (id) => (jobId = id), signal)
        : runProblemStream(problem.id, params, files, verbose, onLine, onQueued, setProgress);
    await showResult(pending, signal, () => jobId);
  }

  /** Show a job's output from its start: live while it runs, or how it ended. */
  function attach(jobId: string) {
    const { onLine, onQueued, signal } = startOutput();
    void showResult(jobEvents(jobId, 0, onLine, onQueued, setProgress, signal), signal, () => jobId);
  }

  // Opening the problem (also after a reload): reattach to its job in progress, or show one that ended while the
  // page was closed; the Jobs panel's "open" asks for a particular job.
  useEffect(() => {
    if (!jobs) return;
    let alive = true;
    const look = (wanted: string | null) =>
      listJobs().then((list) => {
        if (!alive || !list || (runningRef.current && !wanted)) return;
        const job = wanted ? list.find((j) => j.id === wanted) : list.find((j) => j.problem === problem.id);
        if (job && (wanted || jobActive(job) || !jobSeen(job.id))) attach(job.id);
      });
    void look(takeOpenRequest(problem.id));
    const onOpen = () => {
      const wanted = takeOpenRequest(problem.id);
      if (wanted) void look(wanted);
    };
    window.addEventListener(OPEN_JOB, onOpen);
    return () => {
      alive = false;
      window.removeEventListener(OPEN_JOB, onOpen);
    };
  }, [problem.id, jobs]);

  function edit(name: string, code: string) {
    if (ws) return ws.onFiles({ [name]: code });
    const original = problem.files.find((f) => f.name === name)?.code;
    setEdits(({ [name]: _, ...rest }) => (code === original ? rest : { ...rest, [name]: code }));
  }

  const { lang, t } = useLang();
  const level = LEVEL_STYLE[problem.level];
  const wiki = wikiLink(problem, lang);

  return (
    <>
      <div className="sticky top-0 z-10 flex border-b border-gray-200 bg-white xl:hidden" role="tablist">
        {MOBILE_TABS.map((tab) => (
          <button
            key={tab}
            role="tab"
            aria-selected={mobileTab === tab}
            onClick={() => setMobileTab(tab)}
            className={`flex-1 border-b-2 py-2.5 text-sm font-medium ${
              mobileTab === tab ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500"
            }`}
          >
            {t(`mtab.${tab}` as Key)}
          </button>
        ))}
      </div>

      {/* 2. Problem body: title, tags and setup sketch pinned; equations scroll */}
      <main data-ws-inline className="flex min-w-0 flex-col border-r border-gray-200 xl:min-h-0 xl:flex-1">
        <div className={`shrink-0 border-b border-gray-200 bg-white px-5 py-4 ${onPhone("problem")}`}>
          {ws ? (
            <div data-ai-region="summary">
              {/* Workspace problem: edited in place; Markdown / meta.yaml open a floating editor kept in sync */}
              <div className="flex items-start justify-between gap-3">
                <Field
                  value={problem.title}
                  placeholder={t("edit.title")}
                  label={t("edit.title")}
                  className="text-xl font-semibold tracking-tight"
                  inputClass="text-lg font-semibold"
                  render={(v) => (
                    <h2 data-testid="ws-title">
                      <RichText text={v} />
                    </h2>
                  )}
                  onCommit={(title) => title.trim() && onMeta({ title: title.trim() })}
                />
                <EditorButtons open={floating} onOpen={setFloating} />
              </div>
              <div className="mt-1.5 flex flex-wrap items-center gap-2">
                <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
                  {categoryTitle}
                </span>
                <span
                  className={`flex items-center gap-1.5 rounded-full border border-gray-200 py-0.5 pl-2.5 pr-1 text-xs font-medium hover:border-blue-300 ${level.text}`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${level.dot}`} />
                  <select
                    value={problem.level}
                    aria-label={t("edit.level")}
                    title={t("edit.level")}
                    onChange={(e) => onMeta({ level: e.target.value as Level })}
                    className="cursor-pointer bg-transparent focus:outline-none"
                  >
                    {(["beginner", "intermediate", "advanced"] as const).map((l) => (
                      <option key={l} value={l}>
                        {t(LEVEL_STYLE[l].key)}
                      </option>
                    ))}
                  </select>
                </span>
                <button
                  onClick={() => onMeta({ animated: !problem.animated })}
                  title={t("edit.animatedHint")}
                  className="rounded-full border border-gray-200 px-2.5 py-0.5 text-xs text-gray-500 hover:border-blue-300 hover:text-blue-700"
                >
                  {problem.animated ? "🎞" : "🖼"} {t(problem.animated ? "tag.animated" : "tag.static")} ⇄
                </button>
                <TagInput tags={problem.tags ?? []} known={ws.allTags} onChange={(tags) => onMeta({ tags })} />
              </div>
              <Field
                multiline
                value={problem.summary}
                placeholder={t("edit.addSummary")}
                className="mt-2 text-sm leading-relaxed text-gray-600"
                inputClass="text-sm"
                render={(v) => (
                  <p className="whitespace-pre-line">
                    <RichText text={v} />
                  </p>
                )}
                onCommit={(summary) => onMeta({ summary })}
              />
              {floating && (
                <FloatingEditor
                  mode={floating}
                  onMode={setFloating}
                  onClose={() => setFloating(null)}
                  problem={problem}
                  onMeta={onMeta}
                  meta={ws.meta}
                  onMetaText={editMeta}
                  metaError={metaError}
                  onYamlUpdate={yamlSelection}
                />
              )}
            </div>
          ) : (
            <div data-ai-region="summary">
              <div className="flex items-start justify-between gap-3">
                <h2 className="text-xl font-semibold tracking-tight">
                  <RichText text={problem.title} />
                </h2>
                {onFork && (
                  <button
                    onClick={onFork}
                    title={t("fork.hint")}
                    className="flex shrink-0 items-center gap-1.5 rounded-md border border-gray-200 px-2.5 py-1 text-xs font-medium text-gray-600 hover:border-blue-300 hover:text-blue-700"
                  >
                    <ForkIcon /> {t("fork")}
                  </button>
                )}
              </div>
              <div className="mt-1.5 flex flex-wrap items-center gap-2">
                <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
                  {categoryTitle}
                </span>
                <span
                  className={`flex items-center gap-1.5 rounded-full border border-gray-200 px-2.5 py-0.5 text-xs font-medium ${level.text}`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${level.dot}`} />
                  {t(level.key)}
                </span>
                <span className="rounded-full border border-gray-200 px-2.5 py-0.5 text-xs text-gray-500">
                  {t(problem.animated ? "tag.animated" : "tag.static")}
                </span>
                <PrizeBadges tags={problem.tags ?? []} />
              </div>
              <p className="mt-2 text-sm leading-relaxed text-gray-600">
                <RichText text={problem.summary} people />
              </p>
              {manifest && <SeriesNav manifest={manifest} problemId={problem.id} onSelect={select} />}
            </div>
          )}
          {ws && <PictureUpload hasPicture={!!problem.diagram} onPicture={ws.onDiagram} />}
          {problem.diagram && (
            <div className="group relative mx-auto mt-3 w-full max-w-xs">
              <img
                src={assetUrl(problem.diagram)}
                alt={`Sketch of the ${plainText(problem.title)} setup`}
                title={t("setup")}
                className="h-36 w-full object-contain"
              />
              {/* diagram.svg → SVG / PNG / JPG download */}
              <DiagramExport
                src={assetUrl(problem.diagram)}
                name={problem.slug}
                className="absolute top-0 -right-2 opacity-60 group-hover:opacity-100 focus-within:opacity-100"
              />
            </div>
          )}
          {(wiki || problem.papers?.length) && (
            <div className="mt-2 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs">
              {wiki && (
                <a
                  href={wiki.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1.5 text-blue-700 hover:underline"
                >
                  <span className="font-serif font-semibold">W</span>
                  {t("wiki")}: {wiki.title} ↗
                </a>
              )}
              {problem.papers?.map((paper) =>
                paper.url ? (
                  <a
                    key={paper.url}
                    href={paper.url}
                    target="_blank"
                    rel="noreferrer"
                    title={paper.title}
                    className="flex max-w-xs items-center gap-1.5 text-blue-700 hover:underline"
                  >
                    <span>📄</span>
                    <span className="truncate">
                      {t("paper")}: {paper.label}
                    </span>
                    ↗
                  </a>
                ) : (
                  <span
                    key={paper.label}
                    title={paper.title}
                    className="flex max-w-xs items-center gap-1.5 text-gray-600"
                  >
                    <span>📄</span>
                    <span className="truncate">
                      {t("reference")}: {paper.label}
                    </span>
                  </span>
                ),
              )}
            </div>
          )}
        </div>

        <div
          className={`bg-gray-50/60 p-5 xl:min-h-0 xl:flex-1 ${
            !ws && lowerTab === "data" ? "flex flex-col gap-3 xl:flex! xl:overflow-hidden" : "space-y-5 xl:overflow-y-auto"
          } ${onPhone("formulation")}`}
        >
          {ws ? (
            <InlineEquations problem={problem} onMeta={onMeta} />
          ) : (
            <>
              <div
                className="sticky -top-5 z-20 -mx-5 -mt-5 flex items-end gap-1 border-b border-gray-200 bg-gray-50 px-3 pt-2"
                role="tablist"
              >
                {(["formulation", "data"] as const).map((tab) => (
                  <button
                    key={tab}
                    role="tab"
                    aria-selected={lowerTab === tab}
                    onClick={() => setLowerTab(tab)}
                    data-testid={`lower-tab-${tab}`}
                    className={`-mb-px rounded-t-md border px-4 py-1.5 text-xs font-medium ${
                      lowerTab === tab
                        ? "border-gray-200 border-b-white bg-white text-blue-700"
                        : "border-transparent text-gray-500 hover:text-gray-800"
                    }`}
                  >
                    {tab === "formulation"
                      ? `📐 ${t("mtab.formulation")}`
                      : `📊 ${t("dataset.title")} (${problem.datasets?.length ?? 0})`}
                  </button>
                ))}
                {/* in the pinned tab bar, so they stay put while the formulation scrolls */}
                {lowerTab === "formulation" && (
                  <span className="mb-1 ml-auto flex items-center gap-1">
                    <FormulationExport problem={problem} />
                    <button
                      onClick={() => setFormulaBig(true)}
                      title={t("mesh.expand")}
                      aria-label={t("mesh.expand")}
                      data-testid="formula-expand"
                      className="rounded px-1 text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-800"
                    >
                      ⤢
                    </button>
                  </span>
                )}
                {formulaBig && (
                  <div
                    className="fixed inset-3 z-50 flex flex-col overflow-hidden rounded-lg border border-gray-200 bg-gray-50 shadow-2xl"
                    data-testid="formula-big"
                  >
                    <div className="flex shrink-0 items-center justify-between border-b border-gray-200 bg-white px-4 py-1.5 text-xs text-gray-600">
                      <span className="font-medium">📐 {t("mtab.formulation")}</span>
                      <span className="flex items-center gap-1">
                        <FormulationExport problem={problem} />
                        <button
                          onClick={() => setFormulaBig(false)}
                          title={t("mesh.shrink")}
                          aria-label={t("mesh.shrink")}
                          className="rounded px-1 text-gray-500 hover:bg-gray-100 hover:text-gray-800"
                        >
                          ✕
                        </button>
                      </span>
                    </div>
                    <div className="min-h-0 flex-1 overflow-y-auto p-6 text-[1.15rem]">
                      <div className="mx-auto max-w-5xl">
                        <EquationPanel problem={problem} exportable={false} />
                      </div>
                    </div>
                  </div>
                )}
              </div>
              {lowerTab === "formulation" ? (
                <EquationPanel problem={problem} exportable={false} />
              ) : (
                <div className="min-h-80 flex-1 xl:min-h-0">
                  <DataSources datasets={problem.datasets ?? []} fill />
                </div>
              )}
            </>
          )}
        </div>
      </main>

      {/* 3. Code (+ parameters panel), terminal and plot */}
      <section className="flex min-w-0 flex-col xl:min-h-0 xl:w-[54%] xl:shrink-0">
        <CodePanel
          problem={problem.id}
          className={`xl:min-h-0 xl:flex-[1.7] ${onPhone("code", "xl:flex")}`}
          files={problem.files}
          edits={edits}
          editable={editable}
          onEdit={edit}
          onReset={() => setEdits({})}
          onFork={onFork}
          toolbar={<ExportMenu problem={problem} edits={edits} params={params} onWhole={ws?.onExport} />}
          side={
            // display: contents keeps CodePanel's layout; the wrapper only labels the region for the AI panel
            <div data-ai-region="params" className="contents">
              <ParamsPanel
                specs={problem.params}
                values={params}
                onChange={setParams}
                onDefaults={() => setParams(defaults(problem))}
                onRun={run}
                canRun={!!backend}
                running={running}
                hasEdits={hasEdits}
                verbose={verbose}
                onVerbose={setVerbose}
                open={paramsOpen}
                onToggle={onToggleParams}
              />
            </div>
          }
        />
        <div
          className={`flex flex-col border-t border-gray-200 xl:min-h-0 xl:flex-1 xl:flex-row ${onPhone("results", "xl:flex")}`}
        >
          <TerminalPanel
            className="h-40 xl:h-auto xl:w-1/2 xl:shrink-0"
            problem={problem}
            console={console}
            error={error}
            running={running}
            tests={tests}
          />
          <PlotPanel
            className="border-t border-gray-200 xl:min-h-0 xl:flex-1 xl:border-t-0 xl:border-l"
            problem={problem}
            src={src}
            running={running}
            progress={progress}
          />
        </div>
      </section>
    </>
  );
}

/** Workspace: upload the physical picture (SVG, PNG or JPG; a bitmap is wrapped in diagram.svg). */
function PictureUpload({ hasPicture, onPicture }: { hasPicture: boolean; onPicture: (svg: string) => void }) {
  const { t } = useLang();
  const input = useRef<HTMLInputElement>(null);
  const pick = async (file: File | undefined) => {
    if (!file) return;
    try {
      onPicture(await pictureToSvg(file));
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className={hasPicture ? "mt-2 flex justify-end" : "mt-3"}>
      <input
        ref={input}
        type="file"
        accept=".svg,.png,.jpg,.jpeg,image/svg+xml,image/png,image/jpeg"
        className="hidden"
        data-testid="picture-input"
        onChange={(e) => {
          void pick(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {hasPicture ? (
        <button onClick={() => input.current?.click()} className="text-xs text-blue-700 hover:underline">
          ⤒ {t("picture.replace")}
        </button>
      ) : (
        <button
          onClick={() => input.current?.click()}
          className="flex h-24 w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed border-gray-300 text-xs text-gray-500 hover:border-blue-400 hover:text-blue-700"
        >
          <span className="text-lg">🖼</span>
          {t("picture.upload")}
        </button>
      )}
    </div>
  );
}

/** Git-style fork glyph: one branch splitting in two. */
export function ForkIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      <circle cx="4" cy="3" r="1.6" />
      <circle cx="12" cy="3" r="1.6" />
      <circle cx="8" cy="13" r="1.6" />
      <path d="M4 4.6v1.4a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2V4.6M8 8v3.4" />
    </svg>
  );
}

/** The problem's Wikipedia article in the reader's language when one exists, otherwise in English. */
function wikiLink(problem: Problem, lang: Lang): { url: string; title: string } | null {
  if (!problem.wiki) return null;
  const site = lang.startsWith("zh") ? "zh" : lang;
  const local = problem.wiki.titles[site];
  const title = local ?? problem.wiki.titles.en;
  if (!title) return null;
  const host = local ? site : "en";
  const variant = local && lang === "zh-Hant" ? "zh-tw" : local && lang === "zh-Hans" ? "zh-cn" : "wiki";
  return {
    url: `https://${host}.wikipedia.org/${variant}/${encodeURIComponent(title.replace(/ /g, "_"))}`,
    title,
  };
}
