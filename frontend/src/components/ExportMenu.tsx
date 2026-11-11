import { useEffect, useRef, useState } from "react";
import { useLang } from "../i18n";
import {
  colabUrl,
  downloadNotebook,
  downloadPixiProject,
  downloadWorkspaceNotebook,
  lastPushedRepo,
  vscodeCloneUrl,
  vscodeWebUrl,
} from "../notebook";
import type { Params, Problem } from "../types";
import DownloadIcon from "./DownloadIcon";

interface Props {
  problem: Problem;
  edits: Record<string, string>;
  params: Params;
  /** Workspace problems: download the whole problem as one zip (the library links below point at GitHub files). */
  onWhole?: () => void;
  /** Workspace problems: their meta.yaml, written into the notebook built in the browser */
  wsMeta?: string;
  /** Workspace problems: open as a folder in desktop VS Code through the backend, and pull the edits back */
  folder?: { online: boolean; remote: boolean; path: string; onOpen: () => void; onPull: () => void };
}

/** "Export ▾": open in Colab or VS Code, download the notebook, or download the whole problem as a pixi project zip.
 *  Library and Workspace problems get the same menu. A Workspace problem has no copy in Tensorizer's repository, so
 *  its notebook is built in the browser (Colab: download, then upload) and VS Code opens the GitHub repository the
 *  workspace was last exported to. */
export default function ExportMenu({ problem, edits, params, onWhole, wsMeta = "", folder }: Props) {
  const { t } = useLang();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", close);
    };
  }, [open]);

  const inWorkspace = !!onWhole;
  const repo = open && inWorkspace ? lastPushedRepo() : null;
  const item = "flex w-full flex-col items-start rounded-md px-3 py-2 text-left hover:bg-blue-50";
  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="rounded-md border border-gray-300 bg-white px-2.5 py-0.5 font-medium text-gray-700 hover:bg-gray-50"
      >
        {t("export")} ▾
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 z-30 mt-1 w-64 rounded-lg border border-gray-200 bg-white p-1 shadow-lg"
        >
          {inWorkspace ? (
            <>
              <button
                role="menuitem"
                className={item}
                onClick={() => {
                  setOpen(false);
                  downloadWorkspaceNotebook(problem, wsMeta, edits, params);
                  window.open("https://colab.research.google.com/", "_blank", "noopener");
                }}
              >
                <span className="font-medium text-orange-600">{t("nb.colab")} ↗</span>
                <span className="text-[11px] text-gray-500">{t("nb.colabWsHint")}</span>
              </button>
              {/* VS Code needs the files on disk: the backend writes them (only greyed while no backend is connected) */}
              {folder?.online ? (
                <>
                  <button role="menuitem" className={item} onClick={() => (setOpen(false), folder.onOpen())}>
                    <span className="font-medium text-blue-700">{t("nb.vscode")} ↗</span>
                    <span className="text-[11px] text-gray-500">
                      {folder.remote ? t("vs.openRemoteHint") : t("vs.openHint").replace("{path}", folder.path)}
                    </span>
                  </button>
                  <button role="menuitem" className={item} onClick={() => (setOpen(false), folder.onPull())}>
                    <span className="font-medium text-blue-700">⟲ {t("vs.pull")}</span>
                    <span className="text-[11px] text-gray-500">{t("vs.pullHint")}</span>
                  </button>
                </>
              ) : (
                (["nb.vscode", "vs.pull"] as const).map((key) => (
                  <div key={key} role="menuitem" aria-disabled="true" className={`${item} cursor-default hover:bg-transparent`}>
                    <span className="font-medium text-gray-400">{t(key)}</span>
                    <span className="text-[11px] text-gray-500">{t("vs.noBackend")}</span>
                  </div>
                ))
              )}
              {repo && (
                <a
                  role="menuitem"
                  href={`vscode://vscode.git/clone?url=${encodeURIComponent(`${repo.url}.git`)}`}
                  className={item}
                  onClick={() => setOpen(false)}
                >
                  <span className="font-medium text-blue-700">{t("nb.vscodeClone")}</span>
                  <span className="text-[11px] text-gray-500">{t("nb.vscodeWsHint").replace("{repo}", repo.repo)}</span>
                </a>
              )}
            </>
          ) : (
            <>
              <a
                role="menuitem"
                href={colabUrl(problem)}
                target="_blank"
                rel="noreferrer"
                className={item}
                onClick={() => setOpen(false)}
              >
                <span className="font-medium text-orange-600">{t("nb.colab")} ↗</span>
                <span className="text-[11px] text-gray-500">{t("nb.colabHint")}</span>
              </a>
              <a
                role="menuitem"
                href={vscodeWebUrl(problem)}
                target="_blank"
                rel="noreferrer"
                className={item}
                onClick={() => setOpen(false)}
              >
                <span className="font-medium text-blue-700">{t("nb.vscode")} ↗</span>
                <span className="text-[11px] text-gray-500">{t("nb.vscodeHint")}</span>
              </a>
              <a role="menuitem" href={vscodeCloneUrl} className={item} onClick={() => setOpen(false)}>
                <span className="font-medium text-blue-700">{t("nb.vscodeClone")}</span>
                <span className="text-[11px] text-gray-500">{t("nb.vscodeCloneHint")}</span>
              </a>
            </>
          )}
          <div className="my-1 border-t border-gray-100" />
          {inWorkspace ? (
            <button
              role="menuitem"
              className={item}
              onClick={() => (setOpen(false), downloadWorkspaceNotebook(problem, wsMeta, edits, params))}
            >
              <span className="flex items-center gap-1.5 font-medium text-gray-800">
                <DownloadIcon /> {t("nb.download")}
              </span>
              <span className="text-[11px] text-gray-500">{t("nb.downloadWsHint")}</span>
            </button>
          ) : (
            <button
              role="menuitem"
              className={item}
              onClick={() => (setOpen(false), downloadNotebook(problem, edits, params))}
            >
              <span className="flex items-center gap-1.5 font-medium text-gray-800">
                <DownloadIcon /> {t("nb.download")}
              </span>
              <span className="text-[11px] text-gray-500">{t("nb.downloadHint")}</span>
            </button>
          )}
          <button
            role="menuitem"
            className={item}
            onClick={() => (setOpen(false), onWhole ? onWhole() : downloadPixiProject(problem, edits, params))}
          >
            <span className="flex items-center gap-1.5 font-medium text-gray-800">
              <DownloadIcon /> {t("nb.pixi")}
            </span>
            <span className="text-[11px] text-gray-500">{t(inWorkspace ? "nb.wholeHint" : "nb.pixiHint")}</span>
          </button>
        </div>
      )}
    </div>
  );
}
