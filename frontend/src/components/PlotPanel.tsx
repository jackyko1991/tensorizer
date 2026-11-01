import { useEffect, useState } from "react";
import { useLang } from "../i18n";
import { plainText, RichText } from "./Tex";
import type { Problem } from "../types";

interface Props {
  problem: Problem;
  src: string | null;
  running: boolean;
  className?: string;
}

export default function PlotPanel({ problem, src, running, className = "" }: Props) {
  const { t } = useLang();
  const [zoomed, setZoomed] = useState(false);

  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setZoomed(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomed]);

  return (
    <section className={`flex min-w-0 flex-col bg-white ${className}`}>
      <PanelBar title={t("plot")}>
        {running && <span className="animate-pulse text-blue-600">{t("plot.rendering")}</span>}
        {src && (
          <button onClick={() => setZoomed(true)} className="text-gray-500 hover:text-gray-800" title="Zoom (click the plot)">
            ⤢ {t("plot.zoom")}
          </button>
        )}
      </PanelBar>
      <div className="flex min-h-48 flex-1 items-center justify-center overflow-hidden p-2 lg:min-h-0">
        {src ? (
          <img
            src={src}
            alt={`${plainText(problem.title)} result`}
            onClick={() => setZoomed(true)}
            className={`max-h-full max-w-full cursor-zoom-in object-contain ${running ? "opacity-40" : ""}`}
          />
        ) : (
          <span className="text-sm text-gray-400">{t("plot.none")}</span>
        )}
      </div>

      {zoomed && src && (
        <div
          onClick={() => setZoomed(false)}
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-gray-900/60 p-6 backdrop-blur-sm"
        >
          <figure className="flex max-h-full max-w-full flex-col rounded-xl bg-white p-4 shadow-2xl">
            <img src={src} alt={`${plainText(problem.title)} result`} className="min-h-0 max-h-[80vh] max-w-[85vw] object-contain" />
            <figcaption className="mt-2 flex justify-between gap-6 text-xs text-gray-500">
              <span>
                <RichText text={problem.title} />
              </span>
              <span>{t("plot.close")}</span>
            </figcaption>
          </figure>
        </div>
      )}
    </section>
  );
}

export function PanelBar({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="flex h-8 shrink-0 items-center justify-between border-b border-gray-200 bg-gray-50 px-3 text-xs">
      <span className="font-semibold uppercase tracking-wide text-gray-500">{title}</span>
      <span className="flex items-center gap-3">{children}</span>
    </div>
  );
}
