import { useEffect, useState } from "react";
import { DataSources } from "./DataView";
import { useLang } from "../i18n";
import { plainText, RichText } from "./Tex";
import type { Dataset, Problem } from "../types";

interface Props {
  problem: Problem;
  src: string | null;
  running: boolean;
  /** share of the animation's frames drawn so far (0..1) while rendering, if known */
  progress?: number | null;
  className?: string;
}

/** The Results panel. While a run is drawing: its progress. Afterwards the same browser as the Data tab: a thin icon
 *  rail opens the list of results, the animation first, then any extra plots, 3D result meshes and tables the run
 *  wrote next to it. */
export default function PlotPanel({ problem, src, running, progress = null, className = "" }: Props) {
  const { t } = useLang();
  const [zoomed, setZoomed] = useState(false);

  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setZoomed(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomed]);

  const main: Dataset | null = src
    ? {
        name: src.split("?")[0].split("/").pop() || "output",
        kind: "image",
        url: src,
        size: 0,
        comments: [],
        columns: [],
        rows: 0,
      }
    : null;

  return (
    <section className={`flex min-w-0 flex-col bg-white ${className}`}>
      <PanelBar title={t("results")}>
        {running && (
          <span className="text-blue-600" data-testid="plot-progress">
            <span className="animate-pulse">{t("plot.rendering")}</span>
            {progress !== null && <span className="ml-1 tabular-nums">{Math.round(progress * 100)}%</span>}
          </span>
        )}
      </PanelBar>
      {running && progress !== null && (
        <div className="h-0.5 shrink-0 bg-blue-100">
          <div className="h-full bg-blue-600 transition-[width] duration-200" style={{ width: `${progress * 100}%` }} />
        </div>
      )}
      {main && !running ? (
        <div className="h-[70vh] min-h-72 shrink-0 xl:h-auto xl:min-h-0 xl:flex-1 xl:shrink" data-testid="results-browser">
          {/* a definite height on phones (the fill layout sizes to it); on desktop the panel's flex height */}
          <DataSources datasets={[main, ...(problem.results ?? [])]} rail={["files"]} listTitle={t("results")} fill />
        </div>
      ) : (
        <div className="flex min-h-48 flex-1 items-center justify-center overflow-hidden p-2 xl:min-h-0">
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
      )}

      {zoomed && src && (
        <div
          onClick={() => setZoomed(false)}
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-gray-900/60 p-6 backdrop-blur-sm"
        >
          <figure className="flex max-h-full max-w-full flex-col rounded-xl bg-white p-4 shadow-2xl">
            <img
              src={src}
              alt={`${plainText(problem.title)} result`}
              className="min-h-0 max-h-[80vh] max-w-[85vw] object-contain"
            />
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
