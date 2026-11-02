import { useEffect, useState } from "react";
import { useLang } from "../i18n";
import { downloadPixiProject, vscodeCloneUrl, vscodeWebUrl } from "../notebook";
import ContextMenu from "./ContextMenu";
import DownloadIcon from "./DownloadIcon";
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

  const toggle = (id: string) =>
    setOpen((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <nav className="space-y-1">
      {manifest.categories.map((cat) => {
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
