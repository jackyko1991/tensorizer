import { useEffect, useRef, useState } from "react";
import { useLang } from "../i18n";
import { colabUrl, downloadNotebook, downloadPixiProject, vscodeCloneUrl, vscodeWebUrl } from "../notebook";
import type { Params, Problem } from "../types";

interface Props {
  problem: Problem;
  edits: Record<string, string>;
  params: Params;
  /** Workspace problems: download the whole problem as one zip (the library links below point at GitHub files). */
  onWhole?: () => void;
}

/** "Export ▾": open in Colab or VS Code, download the notebook, or download the whole problem as a pixi project zip. */
export default function ExportMenu({ problem, edits, params, onWhole }: Props) {
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
      {open && onWhole && (
        <div
          role="menu"
          className="absolute right-0 z-30 mt-1 w-64 rounded-lg border border-gray-200 bg-white p-1 shadow-lg"
        >
          <button role="menuitem" className={item} onClick={() => (setOpen(false), onWhole())}>
            <span className="font-medium text-gray-800">⬇ {t("nb.pixi")}</span>
            <span className="text-[11px] text-gray-500">{t("nb.wholeHint")}</span>
          </button>
        </div>
      )}
      {open && !onWhole && (
        <div
          role="menu"
          className="absolute right-0 z-30 mt-1 w-64 rounded-lg border border-gray-200 bg-white p-1 shadow-lg"
        >
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
          <div className="my-1 border-t border-gray-100" />
          <button
            role="menuitem"
            className={item}
            onClick={() => (setOpen(false), downloadNotebook(problem, edits, params))}
          >
            <span className="font-medium text-gray-800">⬇ {t("nb.download")}</span>
            <span className="text-[11px] text-gray-500">{t("nb.downloadHint")}</span>
          </button>
          <button
            role="menuitem"
            className={item}
            onClick={() => (setOpen(false), downloadPixiProject(problem, edits, params))}
          >
            <span className="font-medium text-gray-800">⬇ {t("nb.pixi")}</span>
            <span className="text-[11px] text-gray-500">{t("nb.pixiHint")}</span>
          </button>
        </div>
      )}
    </div>
  );
}
