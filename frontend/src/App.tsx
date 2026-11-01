import { useEffect, useState } from "react";
import { generatedUrl, health, loadManifest, RunFailed, runProblemStream, type Health, type LineEvent } from "./api";
import CodePanel from "./components/CodePanel";
import EquationPanel from "./components/EquationPanel";
import ParamsPanel from "./components/ParamsPanel";
import PlotPanel from "./components/PlotPanel";
import Sidebar, { LEVEL_STYLE } from "./components/Sidebar";
import TerminalPanel from "./components/TerminalPanel";
import { LANGS, localize, useLang, type Key, type Lang } from "./i18n";
import { colabUrl, downloadNotebook, REPO_URL } from "./notebook";
import ConceptPage from "./pages/ConceptPage";
import SkillsPage from "./pages/SkillsPage";
import { plainText, RichText } from "./components/Tex";
import type { Manifest, Params, Problem } from "./types";

const currentHash = () => decodeURIComponent(window.location.hash.replace(/^#\/?/, "")) || null;
const select = (id: string) => (window.location.hash = `/${id}`);
const PAGES = ["problems", "concept", "skills"] as const;
type Page = (typeof PAGES)[number];
const TAB_KEY: Record<Page, Key> = { problems: "tab.problems", concept: "tab.concept", skills: "tab.skills" };

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
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [paramsOpen, setParamsOpen] = useState(true);
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

  // #/concept and #/skills are pages; any other hash is a problem id (category/slug).
  const page: Page = selected === "concept" || selected === "skills" ? selected : "problems";
  const [lastProblemId, setLastProblemId] = useState<string | null>(null);
  const problem =
    manifest?.problems.find((p) => p.id === (page === "problems" ? selected : lastProblemId)) ?? manifest?.problems[0];
  useEffect(() => {
    if (page === "problems" && problem) setLastProblemId(problem.id);
  }, [page, problem]);
  const goTo = (target: Page) => (window.location.hash = target === "problems" ? `/${problem?.id ?? ""}` : `/${target}`);

  return (
    <div className="flex min-h-screen flex-col bg-white lg:h-screen">
      <header className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1.5 bg-blue-700 px-4 py-2 text-white shadow-sm sm:px-6 lg:h-[52px] lg:flex-nowrap lg:py-0">
        <div className="flex min-w-0 items-center gap-3">
          <img src={`${import.meta.env.BASE_URL}logo-light.svg`} alt="" className="h-8 w-8" />
          <h1 className="text-lg font-semibold tracking-tight">
            Tensor<span className="text-blue-200">izer</span>
          </h1>
          <span className="hidden text-sm text-blue-100 xl:inline">{t("tagline")}</span>
        </div>
        {/* On phones the tabs wrap onto their own row under the logo and language menu. */}
        <nav className="order-last flex w-full items-center gap-1 overflow-x-auto lg:order-none lg:w-auto" aria-label="Sections">
            {PAGES.map((target) => (
              <button
                key={target}
                onClick={() => goTo(target)}
                aria-current={page === target ? "page" : undefined}
                className={`shrink-0 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  page === target ? "bg-white text-blue-700 shadow-sm" : "text-blue-100 hover:bg-blue-600 hover:text-white"
                }`}
              >
                {t(TAB_KEY[target])}
              </button>
            ))}
          </nav>
        <div className="ml-auto flex items-center gap-4">
          <BackendBadge backend={backend} />
          <select
            value={lang}
            onChange={(e) => setLang(e.target.value as Lang)}
            title={t("language")}
            aria-label={t("language")}
            className="rounded-md border border-blue-400 bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-500"
          >
            {LANGS.map((l) => (
              <option key={l.code} value={l.code}>
                {l.name}
              </option>
            ))}
          </select>
        </div>
      </header>

      {loadError && <p className="m-6 rounded-md bg-red-50 p-4 text-sm text-red-700">{loadError}</p>}

      {manifest && page === "concept" && (
        <main className="min-h-0 flex-1">
          <ConceptPage manifest={manifest} />
        </main>
      )}
      {manifest && page === "skills" && (
        <main className="min-h-0 flex-1">
          <SkillsPage manifest={manifest} />
        </main>
      )}

      {manifest && problem && page === "problems" && (
        <div className="flex flex-1 flex-col lg:min-h-0 lg:flex-row">
          <div className="border-b border-gray-200 p-3 lg:hidden">
            <ProblemSelect manifest={manifest} value={problem.id} />
          </div>

          {/* 1. Problem library */}
          <aside
            className={`hidden shrink-0 flex-col border-r border-gray-200 bg-white transition-[width] lg:flex ${
              libraryOpen ? "w-64" : "w-10"
            }`}
          >
            <div className="flex h-8 shrink-0 items-center justify-between border-b border-gray-200 bg-gray-50 px-2">
              {libraryOpen && (
                <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">{t("library")}</span>
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
              <div className="min-h-0 flex-1 overflow-y-auto p-3">
                <Sidebar manifest={manifest} selected={problem.id} onSelect={select} />
              </div>
            ) : (
              <button
                onClick={() => setLibraryOpen(true)}
                className="flex-1 py-3 text-xs font-medium tracking-wide text-gray-500 [writing-mode:vertical-rl] hover:text-gray-800"
              >
                {t("library")}
              </button>
            )}
          </aside>

          <ProblemView
            key={problem.id}
            problem={localize(problem, lang)}
            backend={backend === "checking" ? null : backend}
            categoryTitle={t(`cat.${problem.category}` as Key) ?? problem.category}
            paramsOpen={paramsOpen}
            onToggleParams={() => setParamsOpen((o) => !o)}
          />
        </div>
      )}

      <footer className="sticky bottom-0 z-10 flex h-7 shrink-0 items-center justify-end gap-4 border-t border-gray-200 bg-white/95 px-4 text-xs text-gray-500 backdrop-blur">
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
              <optgroup key={`${cat.id}-${level}`} label={`${t(`cat.${cat.id}` as Key) ?? cat.title} · ${t(LEVEL_STYLE[level].key)}`}>
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

function BackendBadge({ backend }: { backend: Health | null | "checking" }) {
  const { t } = useLang();
  const [label, color] =
    backend === "checking"
      ? [t("backend.checking"), "bg-gray-300"]
      : backend
        ? [t(backend.allow_code_exec ? "backend.edit" : "backend.online"), "bg-green-500"]
        : [t("backend.offline"), "bg-amber-400"];
  return (
    <span className="hidden items-center gap-2 text-xs text-blue-100 sm:flex">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}

function defaults(problem: Problem): Params {
  return Object.fromEntries(Object.entries(problem.params).map(([k, s]) => [k, s.default]));
}

const MOBILE_TABS = ["problem", "formulation", "code", "results"] as const;
type MobileTab = (typeof MOBILE_TABS)[number];

interface ProblemViewProps {
  problem: Problem;
  backend: Health | null;
  categoryTitle: string;
  paramsOpen: boolean;
  onToggleParams: () => void;
}

function ProblemView({ problem, backend, categoryTitle, paramsOpen, onToggleParams }: ProblemViewProps) {
  const [params, setParams] = useState<Params>(() => defaults(problem));
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [src, setSrc] = useState<string | null>(problem.output ? generatedUrl(problem.output) : null);
  const [console, setConsole] = useState(problem.console);
  const [running, setRunning] = useState(false);
  const [verbose, setVerbose] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasRun, setHasRun] = useState(false);
  // Phones show one part of the workspace at a time; desktop shows all three columns.
  const [mobileTab, setMobileTab] = useState<MobileTab>("problem");
  const onPhone = (tab: MobileTab, display = "lg:block") => (mobileTab === tab ? "" : `hidden ${display}`);

  // A refreshed manifest may bring a newly rendered default result; show it unless the user ran their own.
  useEffect(() => {
    if (hasRun || running) return;
    setSrc(problem.output ? generatedUrl(problem.output) : null);
    setConsole(problem.console);
  }, [problem.output, problem.console, hasRun, running]);

  const editable = !!backend?.allow_code_exec;
  const hasEdits = Object.keys(edits).length > 0;

  async function run() {
    setRunning(true);
    setHasRun(true);
    setMobileTab("results");
    setError(null);
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
    try {
      const result = await runProblemStream(problem.id, params, hasEdits ? edits : undefined, verbose, onLine);
      setSrc(result.output);
      setConsole(result.console);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      if (e instanceof RunFailed && e.console) setConsole(e.console);
    } finally {
      setRunning(false);
    }
  }

  function edit(name: string, code: string) {
    const original = problem.files.find((f) => f.name === name)?.code;
    setEdits(({ [name]: _, ...rest }) => (code === original ? rest : { ...rest, [name]: code }));
  }

  const { t } = useLang();
  const level = LEVEL_STYLE[problem.level];

  return (
    <>
      <div className="sticky top-0 z-10 flex border-b border-gray-200 bg-white lg:hidden" role="tablist">
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
      <main className="flex min-w-0 flex-col border-r border-gray-200 lg:min-h-0 lg:flex-1">
        <div className={`shrink-0 border-b border-gray-200 bg-white px-5 py-4 ${onPhone("problem")}`}>
          <div>
            <h2 className="text-xl font-semibold tracking-tight">
              <RichText text={problem.title} />
            </h2>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">{categoryTitle}</span>
              <span
                className={`flex items-center gap-1.5 rounded-full border border-gray-200 px-2.5 py-0.5 text-xs font-medium ${level.text}`}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${level.dot}`} />
                {t(level.key)}
              </span>
              <span className="rounded-full border border-gray-200 px-2.5 py-0.5 text-xs text-gray-500">
                {t(problem.animated ? "tag.animated" : "tag.static")}
              </span>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-gray-600">
              <RichText text={problem.summary} />
            </p>
          </div>
          {problem.diagram && (
            <img
              src={generatedUrl(problem.diagram)}
              alt={`Sketch of the ${plainText(problem.title)} setup`}
              title={t("setup")}
              className="mx-auto mt-3 h-36 w-full max-w-xs object-contain"
            />
          )}
        </div>

        <div className={`space-y-5 bg-gray-50/60 p-5 lg:min-h-0 lg:flex-1 lg:overflow-y-auto ${onPhone("formulation")}`}>
          <EquationPanel problem={problem} />

        </div>
      </main>

      {/* 3. Code (+ parameters panel), terminal and plot */}
      <section className="flex min-w-0 flex-col lg:min-h-0 lg:w-[54%] lg:shrink-0">
        <CodePanel
          className={`lg:min-h-0 lg:flex-[1.7] ${onPhone("code", "lg:flex")}`}
          files={problem.files}
          edits={edits}
          editable={editable}
          onEdit={edit}
          onReset={() => setEdits({})}
          toolbar={
            <>
              <button
                onClick={() => downloadNotebook(problem, edits, params)}
                title={t("nb.downloadHint")}
                className="hover:text-gray-800"
              >
                ⬇ {t("nb.download")}
              </button>
              <a href={colabUrl(problem)} target="_blank" rel="noreferrer" className="font-medium text-orange-600 hover:text-orange-700">
                {t("nb.colab")} ↗
              </a>
            </>
          }
          side={
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
          }
        />
        <div className={`flex flex-col border-t border-gray-200 lg:min-h-0 lg:flex-1 lg:flex-row ${onPhone("results", "lg:flex")}`}>
          <TerminalPanel
            className="h-40 lg:h-auto lg:w-1/2 lg:shrink-0"
            problem={problem}
            console={console}
            error={error}
            running={running}
          />
          <PlotPanel
            className="border-t border-gray-200 lg:min-h-0 lg:flex-1 lg:border-t-0 lg:border-l"
            problem={problem}
            src={src}
            running={running}
          />
        </div>
      </section>
    </>
  );
}
