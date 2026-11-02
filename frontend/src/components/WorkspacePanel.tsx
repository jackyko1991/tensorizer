import { useEffect, useRef, useState, type ReactNode } from "react";
import { DIAGRAM_FORMATS, exportDiagram } from "../diagramExport";
import { useLang, type Key } from "../i18n";
import type { Manifest, Workspace } from "../types";
import { plainText, RichText } from "./Tex";
import DownloadIcon from "./DownloadIcon";

interface Props {
  manifest: Manifest;
  workspace: Workspace | null;
  /** The page's hash route, e.g. "workspace/<id>" (source) or "workspace/<id>/nb/<nb>". */
  selected: string | null;
  onSelect: (route: string) => void;
  onNewBlank: () => Promise<void>;
  onCopy: (libraryId: string) => Promise<void>;
  onRename: (id: string) => void;
  onDelete: (id: string) => void;
  onAddNotebook: (id: string) => void;
  onRenameNotebook: (id: string, notebook: string) => void;
  onDeleteNotebook: (id: string, notebook: string) => void;
  onExport: () => void;
  onImport: (file: File) => void;
  onGitHub: () => void;
  className?: string;
}

/** The Workspace section of the left panel: a workspace picker, then a tree of problems → Source + notebooks. */
export default function WorkspacePanel({
  manifest,
  workspace,
  selected,
  onSelect,
  onNewBlank,
  onCopy,
  onRename,
  onDelete,
  onAddNotebook,
  onRenameNotebook,
  onDeleteNotebook,
  onExport,
  onImport,
  onGitHub,
  className = "",
}: Props) {
  const { lang, t } = useLang();
  const [menu, setMenu] = useState<null | "new" | "library">(null);
  // right-click on a diagram.svg item: export it as SVG / PNG / JPG
  const [fileMenu, setFileMenu] = useState<{ x: number; y: number; src: string; name: string } | null>(null);
  useEffect(() => {
    if (!fileMenu) return;
    const close = () => setFileMenu(null);
    window.addEventListener("click", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", close);
    };
  }, [fileMenu]);
  const [busy, setBusy] = useState(false);
  const [folded, setFolded] = useState<Set<string>>(new Set());
  const fileInput = useRef<HTMLInputElement>(null);
  const problems = workspace?.problems ?? [];

  const create = async (action: () => Promise<void>) => {
    setBusy(true);
    setMenu(null);
    try {
      await action();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`flex min-h-0 flex-col ${className}`}>
      <div className="flex h-8 shrink-0 items-center justify-between border-y border-gray-200 bg-gray-50 px-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">{t("workspace")}</span>
        <button
          onClick={() => setMenu((m) => (m ? null : "new"))}
          disabled={busy}
          title={t("ws.newProblem")}
          aria-expanded={menu !== null}
          className="rounded px-1.5 text-xs font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-50"
        >
          {busy ? "…" : `+ ${t("workspace.newShort")}`}
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {/* Workspace picker: the free trial has one */}
        <div className="mb-2 flex items-center gap-1.5">
          <select
            value={workspace?.id ?? ""}
            onChange={() => {}}
            aria-label={t("ws.picker")}
            className="min-w-0 flex-1 rounded-md border border-gray-300 bg-white px-2 py-1 text-xs text-gray-800"
          >
            {workspace && <option value={workspace.id}>{workspace.name || t("ws.default")}</option>}
            <option disabled value="pro">
              🔒 {t("ws.morePro")}
            </option>
          </select>
          <WsMenu
            onGitHub={onGitHub}
            onExport={onExport}
            onImport={() => fileInput.current?.click()}
            canExport={problems.length > 0}
          />
          <input
            ref={fileInput}
            type="file"
            accept=".zip,application/zip"
            className="hidden"
            data-testid="ws-import"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onImport(file);
              e.target.value = "";
            }}
          />
        </div>

        {menu === "new" && (
          <div className="mb-2 space-y-1 rounded-lg border border-blue-100 bg-blue-50/50 p-1.5 text-sm">
            <MenuItem icon="📄" label={t("ws.blank")} hint={t("ws.blankHint")} onClick={() => create(onNewBlank)} />
            <MenuItem
              icon="📚"
              label={t("ws.fromLibrary")}
              hint={t("ws.fromLibraryHint")}
              onClick={() => setMenu("library")}
            />
            <p className="px-2 pt-0.5 text-[11px] text-gray-500">✨ {t("ws.aiHint")}</p>
          </div>
        )}
        {menu === "library" && (
          <div className="mb-2 rounded-lg border border-blue-100 bg-blue-50/50 p-1.5">
            <select
              autoFocus
              defaultValue=""
              aria-label={t("ws.fromLibrary")}
              onChange={(e) => e.target.value && create(() => onCopy(e.target.value))}
              className="w-full rounded-md border border-gray-300 bg-white px-2 py-1 text-xs"
            >
              <option value="" disabled>
                {t("ws.pickLibrary")}
              </option>
              {manifest.categories.map((cat) => (
                <optgroup key={cat.id} label={t(`cat.${cat.id}` as Key) ?? cat.title}>
                  {manifest.problems
                    .filter((p) => p.category === cat.id)
                    .map((p) => (
                      <option key={p.id} value={p.id}>
                        {plainText(p.i18n?.[lang]?.title ?? p.title)}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
            <button onClick={() => setMenu(null)} className="mt-1 px-1 text-xs text-gray-500 hover:underline">
              {t("ws.cancel")}
            </button>
          </div>
        )}

        {problems.length === 0 ? (
          <p className="px-1 py-1 text-xs text-gray-400">✨ {t("workspace.empty")}</p>
        ) : (
          <ul className="space-y-0.5" data-testid="ws-tree">
            {problems.map((p) => {
              const route = `workspace/${p.id}`;
              const open = !folded.has(p.id);
              const active = selected === route || selected?.startsWith(`${route}/`);
              return (
                <li key={p.id} data-problem={p.id}>
                  <div
                    className={`group flex items-center rounded-md ${active ? "bg-blue-50/60" : "hover:bg-gray-50"}`}
                  >
                    <button
                      onClick={() =>
                        setFolded((s) => {
                          const next = new Set(s);
                          if (open) next.add(p.id);
                          else next.delete(p.id);
                          return next;
                        })
                      }
                      className="px-1 text-xs text-gray-400"
                      aria-label={t(open ? "library.fold" : "library.open")}
                    >
                      <span className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>›</span>
                    </button>
                    <button
                      onClick={() => onSelect(route)}
                      className="min-w-0 flex-1 truncate py-1 text-left text-sm font-medium text-gray-800"
                      data-testid="ws-problem-title"
                    >
                      <RichText text={p.entry.title || t("ws.untitled")} />
                    </button>
                    <span className={`flex shrink-0 ${HOVER_ONLY}`}>
                      <button
                        onClick={() => onRename(p.id)}
                        title={t("ws.rename")}
                        aria-label={t("ws.rename")}
                        className="rounded px-1 text-xs text-gray-500 hover:bg-gray-200"
                      >
                        ✎
                      </button>
                      <button
                        onClick={() => onDelete(p.id)}
                        title={t("ws.delete")}
                        aria-label={t("ws.delete")}
                        className="rounded px-1 text-xs text-gray-500 hover:bg-red-50 hover:text-red-600"
                      >
                        🗑
                      </button>
                    </span>
                  </div>
                  {open && (
                    <ul className="mb-1 ml-3 border-l border-gray-200 pl-2">
                      <TreeItem
                        icon="🧩"
                        label={t("ws.source")}
                        hint={t("ws.sourceHint")}
                        active={selected === route}
                        onClick={() => onSelect(route)}
                      />
                      {p.entry.diagram && (
                        <TreeItem
                          icon="🖼"
                          label="diagram.svg"
                          hint={t("diagram.hint")}
                          active={false}
                          onClick={() => onSelect(route)}
                          onContextMenu={(e) => {
                            e.preventDefault();
                            setFileMenu({
                              x: e.clientX,
                              y: e.clientY,
                              src: p.entry.diagram!,
                              name: plainText(p.entry.title) || "diagram",
                            });
                          }}
                        />
                      )}
                      <TreeItem
                        icon="📊"
                        label={`${t("data.title")}${p.entry.data?.length ? ` (${p.entry.data.length})` : ""}`}
                        hint={t("data.hintShort")}
                        active={selected === `${route}/data`}
                        onClick={() => onSelect(`${route}/data`)}
                      />
                      {p.notebooks.map((nb) => (
                        <TreeItem
                          key={nb.id}
                          icon="📓"
                          label={nb.title || t("ws.notebook")}
                          active={selected === `${route}/nb/${nb.id}`}
                          onClick={() => onSelect(`${route}/nb/${nb.id}`)}
                          actions={
                            <>
                              <button
                                onClick={() => onRenameNotebook(p.id, nb.id)}
                                title={t("ws.renameNotebook")}
                                aria-label={t("ws.renameNotebook")}
                                className="rounded px-1 text-[11px] text-gray-500 hover:bg-gray-200"
                              >
                                ✎
                              </button>
                              <button
                                onClick={() => onDeleteNotebook(p.id, nb.id)}
                                title={t("ws.deleteNotebook")}
                                aria-label={t("ws.deleteNotebook")}
                                className="rounded px-1 text-[11px] text-gray-500 hover:bg-red-50 hover:text-red-600"
                              >
                                🗑
                              </button>
                            </>
                          }
                        />
                      ))}
                      <li>
                        <button
                          onClick={() => onAddNotebook(p.id)}
                          className="w-full rounded-md px-2 py-0.5 text-left text-xs text-blue-700 hover:bg-blue-50"
                        >
                          + {t("ws.addNotebook")}
                        </button>
                      </li>
                    </ul>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
      {fileMenu && (
        <div
          role="menu"
          style={{ left: fileMenu.x, top: fileMenu.y }}
          className="fixed z-50 w-40 rounded-lg border border-gray-200 bg-white p-1 text-xs shadow-lg"
        >
          <p className="px-2 py-1 font-mono text-gray-400">diagram.svg</p>
          {DIAGRAM_FORMATS.map((f) => (
            <button
              key={f}
              role="menuitem"
              onClick={() => void exportDiagram(fileMenu.src, fileMenu.name, f)}
              className="block w-full rounded px-2 py-1 text-left text-gray-700 hover:bg-blue-50"
            >
              {t("diagram.as")} {f.toUpperCase()}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function MenuItem({ icon, label, hint, onClick }: { icon: string; label: string; hint: string; onClick: () => void }) {
  return (
    <button onClick={onClick} className="flex w-full items-start gap-2 rounded-md px-2 py-1 text-left hover:bg-white">
      <span>{icon}</span>
      <span className="min-w-0">
        <span className="block text-sm font-medium text-gray-800">{label}</span>
        <span className="block text-[11px] leading-snug text-gray-500">{hint}</span>
      </span>
    </button>
  );
}

// Row actions: always visible on touch screens, on hover where there is a mouse.
const HOVER_ONLY = "opacity-100 [@media(hover:hover)]:opacity-0 group-hover:opacity-100 focus-within:opacity-100";

function TreeItem(props: {
  icon: string;
  label: string;
  hint?: string;
  active: boolean;
  onClick: () => void;
  onContextMenu?: (e: React.MouseEvent) => void;
  actions?: ReactNode;
}) {
  return (
    <li className={`group flex items-center rounded-md ${props.active ? "bg-blue-50" : "hover:bg-gray-100"}`}>
      <button
        onClick={props.onClick}
        onContextMenu={props.onContextMenu}
        title={props.hint}
        className={`flex min-w-0 flex-1 items-center gap-1.5 truncate px-2 py-0.5 text-left text-xs ${
          props.active ? "font-medium text-blue-700" : "text-gray-600"
        }`}
      >
        <span>{props.icon}</span>
        <span className="truncate">{props.label}</span>
      </button>
      {props.actions && <span className={`flex shrink-0 ${HOVER_ONLY}`}>{props.actions}</span>}
    </li>
  );
}

/** ⋯ menu: export / import a .zip, and cloud save (Pro). */
function WsMenu({
  onExport,
  onImport,
  onGitHub,
  canExport,
}: {
  onExport: () => void;
  onImport: () => void;
  onGitHub: () => void;
  canExport: boolean;
}) {
  const { t } = useLang();
  const [open, setOpen] = useState(false);
  const item = "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs hover:bg-blue-50";
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        title={t("ws.menu")}
        aria-label={t("ws.menu")}
        aria-expanded={open}
        className="rounded-md border border-gray-300 px-1.5 py-0.5 text-xs text-gray-600 hover:bg-gray-100"
      >
        ⋯
      </button>
      {open && (
        <div
          className="absolute right-0 z-30 mt-1 w-52 rounded-lg border border-gray-200 bg-white p-1 shadow-lg"
          onClick={() => setOpen(false)}
        >
          <button
            onClick={onExport}
            disabled={!canExport}
            title={t("ws.exportHint")}
            className={`${item} disabled:opacity-40`}
          >
            <DownloadIcon /> {t("ws.export")}
          </button>
          <button onClick={onImport} title={t("ws.importHint")} className={item}>
            ⬆ {t("ws.import")}
          </button>
          <button
            onClick={onGitHub}
            disabled={!canExport}
            className={`${item} disabled:opacity-40`}
            data-testid="ws-github"
          >
            <svg viewBox="0 0 16 16" className="h-3 w-3" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.81.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.67.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
            {t("gh.menu")}
          </button>
          <button disabled title={t("ws.cloudHint")} className={`${item} cursor-not-allowed text-gray-400`}>
            ☁ {t("ws.cloud")}
            <span className="ml-auto rounded-full bg-amber-100 px-1.5 text-[10px] font-semibold text-amber-700">
              Pro
            </span>
          </button>
          <p className="px-2.5 pt-1 text-[10px] text-gray-400">💾 {t("ws.savedLocally")}</p>
        </div>
      )}
    </div>
  );
}
