import { useEffect, useMemo, useRef, useState } from "react";
import { buildCollections, GROUP_TITLE, GROUPS, ROLE_STYLE } from "../collections";
import { useLang } from "../i18n";
import { downloadPixiProject, vscodeCloneUrl, vscodeWebUrl } from "../notebook";
import ContextMenu from "./ContextMenu";
import DownloadIcon from "./DownloadIcon";
import { prizeLabel } from "./PrizeBadges";
import CategoryIcon from "./CategoryIcon";
import { RichText } from "./Tex";
import type { Level, Manifest, Problem } from "../types";

export const LEVEL_STYLE: Record<
  Level,
  { key: "level.beginner" | "level.intermediate" | "level.advanced"; dot: string; text: string }
> = {
  beginner: { key: "level.beginner", dot: "bg-green-500", text: "text-green-700" },
  intermediate: { key: "level.intermediate", dot: "bg-amber-500", text: "text-amber-700" },
  advanced: { key: "level.advanced", dot: "bg-red-500", text: "text-red-700" },
};

interface Props {
  manifest: Manifest;
  selected: string | null;
  onSelect: (id: string) => void;
  /** right-click → Fork to Workspace */
  onFork?: (id: string) => void;
}

export default function Sidebar({ manifest, selected, onSelect, onFork }: Props) {
  const { lang, t } = useLang();
  const selectedCategory = manifest.problems.find((p) => p.id === selected)?.category;
  const [open, setOpen] = useState<Set<string>>(() => new Set(selectedCategory ? [selectedCategory] : []));

  // Opening a problem from elsewhere (URL, dropdown) expands its topic.
  useEffect(() => {
    if (selectedCategory) setOpen((s) => (s.has(selectedCategory) ? s : new Set(s).add(selectedCategory)));
  }, [selectedCategory]);

  // right-click on a problem: export the whole problem, open it in VS Code, or fork it
  const [menu, setMenu] = useState<{ x: number; y: number; problem: Problem } | null>(null);
  const defaults = (p: Problem) => Object.fromEntries(Object.entries(p.params).map(([k, s]) => [k, s.default]));

  // search: title (your language and English), summary, topic, slug and prize badges; case and accents ignored
  const [query, setQuery] = useState("");
  const box = useRef<HTMLInputElement>(null);
  const fold = (text: string) => text.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
  const index = useMemo(
    () =>
      manifest.problems.map((p) => ({
        p,
        text: fold(
          [
            p.title,
            p.i18n?.[lang]?.title ?? "",
            p.summary,
            p.i18n?.[lang]?.summary ?? "",
            t(`cat.${p.category}` as Parameters<typeof t>[0]) ?? "",
            p.id.replace(/_/g, " "),
            ...(p.tags ?? []).map((tag) => `${tag} ${prizeLabel(tag, t) ?? ""}`),
          ].join(" "),
        ),
      })),
    [manifest, lang, t],
  );
  const words = fold(query).split(/\s+/).filter(Boolean);
  // collections: the index sits under the search box; choosing one lists its problems in order
  const collections = useMemo(() => buildCollections(manifest, lang, t), [manifest, lang, t]);
  const [collection, setCollection] = useState<string | null>(null);
  const current = collections.find((c) => c.id === collection) ?? null;
  const hits = words.length ? index.filter((e) => words.every((w) => e.text.includes(w))).map((e) => e.p) : [];
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement).closest("input, textarea, [contenteditable=true], .cm-editor");
      if (e.key === "/" && !typing) {
        e.preventDefault();
        box.current?.focus();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

  const toggle = (id: string) =>
    setOpen((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const item = (p: Problem) => (
    <li key={p.id}>
      <button
        onClick={() => onSelect(p.id)}
        onContextMenu={(e) => {
          e.preventDefault();
          setMenu({ x: e.clientX, y: e.clientY, problem: p });
        }}
        className={`flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-sm transition ${
          selected === p.id ? "bg-blue-50 font-medium text-blue-700" : "text-gray-700 hover:bg-gray-100"
        }`}
      >
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${LEVEL_STYLE[p.level].dot}`} />
        <span className="min-w-0">
          <RichText text={p.i18n?.[lang]?.title ?? p.title} />
        </span>
      </button>
    </li>
  );

  return (
    <nav className="space-y-1">
      {/* stays put while the list scrolls under it */}
      <div className="sticky -top-3 z-10 -mx-3 -mt-3 mb-2 bg-white px-3 pt-3 pb-1">
        <div className="relative">
          <input
            ref={box}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setQuery("")}
            placeholder={t("search.placeholder")}
            aria-label={t("search.placeholder")}
            data-testid="problem-search"
            className="w-full rounded-md border border-gray-200 bg-white py-1.5 pr-7 pl-7 text-sm placeholder:text-gray-400 focus:border-blue-400 focus:outline-none"
          />
          <svg
            viewBox="0 0 24 24"
            className="pointer-events-none absolute top-2 left-2 h-4 w-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="M20 20l-4-4" />
          </svg>
          {query ? (
            <button
              onClick={() => setQuery("")}
              aria-label={t("search.clear")}
              className="absolute top-1 right-1.5 px-1 text-lg leading-none text-gray-400 hover:text-gray-700"
            >
              ×
            </button>
          ) : (
            <kbd className="absolute top-1.5 right-2 rounded border border-gray-200 px-1 text-[10px] text-gray-400">
              /
            </kbd>
          )}
        </div>
      </div>
      {words.length === 0 && collections.length > 0 && (
        <label className="mb-2 flex items-center gap-1.5 px-1 text-xs text-gray-500" data-testid="collections">
          <span className="shrink-0">📚</span>
          <select
            value={collection ?? ""}
            onChange={(e) => setCollection(e.target.value || null)}
            aria-label={t("coll.title")}
            data-testid="collection-select"
            className="min-w-0 flex-1 rounded-md border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700 focus:border-blue-400 focus:outline-none"
          >
            <option value="">
              {t("coll.title")}: {t("coll.back")}
            </option>
            {GROUPS.filter((g) => collections.some((c) => c.group === g)).map((g) => (
              <optgroup key={g} label={t(GROUP_TITLE[g])}>
                {collections
                  .filter((c) => c.group === g)
                  .map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.title} ({c.items.length})
                    </option>
                  ))}
              </optgroup>
            ))}
          </select>
        </label>
      )}
      {words.length === 0 && current && (
        <div data-testid="collection-list">
          <div className="flex items-center justify-between px-2 pb-1">
            <span className="text-xs font-semibold text-gray-800">{current.title}</span>
            <button onClick={() => setCollection(null)} className="text-[11px] text-blue-600 hover:underline">
              {t("coll.back")}
            </button>
          </div>
          {current.summary && <p className="px-2 pb-1 text-[11px] text-gray-500">{current.summary}</p>}
          <ol>
            {current.items.map((it, i) => (
              <li key={it.problem.id} className="flex items-start gap-1">
                <span className="mt-1.5 w-5 shrink-0 text-right font-mono text-[10px] text-gray-400">
                  {current.kind === "series" ? `${i + 1}.` : it.step ? it.step : ""}
                </span>
                <div className="min-w-0 flex-1">
                  <ul>{item(it.problem)}</ul>
                  {it.role && (
                    <span className={`ml-5 rounded px-1.5 py-0.5 text-[10px] ${ROLE_STYLE[it.role]}`}>
                      {t(`srole.${it.role}` as Parameters<typeof t>[0])}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
      {words.length > 0 && (
        <div data-testid="search-results">
          <p className="px-2 pb-1 text-[11px] text-gray-500">{t("search.count").replace("{n}", String(hits.length))}</p>
          {hits.length === 0 && <p className="px-2 py-2 text-sm text-gray-400">{t("search.none")}</p>}
          {manifest.categories
            .filter((cat) => hits.some((p) => p.category === cat.id))
            .map((cat) => (
              <div key={cat.id} className="mb-2">
                <h3 className="flex items-center gap-1.5 px-2 pt-1 pb-0.5 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">
                  <CategoryIcon id={cat.id} className="h-3.5 w-3.5 text-blue-600" />
                  {t(`cat.${cat.id}` as Parameters<typeof t>[0]) ?? cat.title}
                </h3>
                <ul>{hits.filter((p) => p.category === cat.id).map(item)}</ul>
              </div>
            ))}
        </div>
      )}
      {words.length === 0 &&
        !current &&
        manifest.categories.map((cat) => {
          const isOpen = open.has(cat.id);
          const problems = manifest.problems.filter((p) => p.category === cat.id);
          return (
            <div key={cat.id}>
              <button
                onClick={() => toggle(cat.id)}
                className="flex w-full items-center justify-between rounded-md px-2 py-2 text-left text-sm font-semibold text-gray-800 hover:bg-gray-100"
              >
                <span className="flex items-center gap-2">
                  <CategoryIcon id={cat.id} className="h-4 w-4 shrink-0 text-blue-600" />
                  {t(`cat.${cat.id}` as Parameters<typeof t>[0]) ?? cat.title}
                </span>
                <span className="flex items-center gap-2 text-xs font-normal text-gray-400">
                  {problems.length}
                  <span className={`transition-transform ${isOpen ? "rotate-90" : ""}`}>›</span>
                </span>
              </button>

              {isOpen && (
                <div className="mb-2 ml-2 space-y-2 border-l border-gray-200 pl-2">
                  {manifest.levels.map((level) => {
                    const inLevel = problems.filter((p) => p.level === level);
                    if (inLevel.length === 0) return null;
                    const style = LEVEL_STYLE[level];
                    return (
                      <div key={level}>
                        <h3
                          className={`flex items-center gap-1.5 px-2 pt-1 pb-0.5 text-[11px] font-semibold uppercase tracking-wide ${style.text}`}
                        >
                          <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
                          {t(style.key)}
                        </h3>
                        <ul>
                          {inLevel.map((p) => (
                            <li key={p.id}>
                              <button
                                onClick={() => onSelect(p.id)}
                                onContextMenu={(e) => {
                                  e.preventDefault();
                                  setMenu({ x: e.clientX, y: e.clientY, problem: p });
                                }}
                                className={`w-full rounded-md px-2 py-1 text-left text-sm transition ${
                                  selected === p.id
                                    ? "bg-blue-50 font-medium text-blue-700"
                                    : "text-gray-700 hover:bg-gray-100"
                                }`}
                              >
                                <RichText text={p.i18n?.[lang]?.title ?? p.title} />
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          title={menu.problem.i18n?.[lang]?.title ?? menu.problem.title}
          onClose={() => setMenu(null)}
          items={[
            {
              label: t("nb.pixi"),
              icon: <DownloadIcon />,
              onClick: () => void downloadPixiProject(menu.problem, {}, defaults(menu.problem)),
            },
            { label: `${t("nb.vscode")} ↗`, icon: "⌨", href: vscodeWebUrl(menu.problem) },
            { label: t("nb.vscodeClone"), icon: "⧉", href: vscodeCloneUrl },
            ...(onFork ? [{ label: t("fork"), icon: "⑂", onClick: () => onFork(menu.problem.id) }] : []),
          ]}
        />
      )}
    </nav>
  );
}
