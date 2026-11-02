import { useEffect, useRef, useState } from "react";
import { DIAGRAM_FORMATS, exportDiagram } from "../diagramExport";
import { useLang } from "../i18n";
import DownloadIcon from "./DownloadIcon";

/** A small download button next to the physical picture: download diagram.svg as SVG, PNG or JPG. */
export default function DiagramExport({
  src,
  name,
  className = "",
}: {
  src: string;
  name: string;
  className?: string;
}) {
  const { t } = useLang();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [open]);

  return (
    <div ref={box} className={className}>
      <button
        onClick={() => setOpen((o) => !o)}
        title={t("diagram.export")}
        aria-label={t("diagram.export")}
        className="rounded-md border border-gray-200 bg-white px-1.5 py-0.5 text-xs text-gray-600 shadow-sm hover:border-blue-300 hover:text-blue-700"
      >
        <DownloadIcon />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 z-20 mt-1 w-36 rounded-lg border border-gray-200 bg-white p-1 text-xs shadow-lg"
        >
          <p className="px-2 py-1 font-mono text-gray-400">diagram.svg</p>
          {DIAGRAM_FORMATS.map((f) => (
            <button
              key={f}
              role="menuitem"
              onClick={() => {
                setOpen(false);
                void exportDiagram(src, name, f);
              }}
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
