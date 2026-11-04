import { lazy, Suspense, useEffect, useState } from "react";
import DataView from "./DataView";
import { useLang } from "../i18n";
import { plainText, RichText } from "./Tex";
import type { Problem } from "../types";
import { assetUrl } from "../api";

const MeshView = lazy(() => import("./MeshView"));
const ICON = { image: "🖼", mesh: "🧊", table: "📊", text: "📄", array: "🔢", other: "📦" } as const;

interface Props {
  problem: Problem;
  src: string | null;
  running: boolean;
  /** share of the animation's frames drawn so far (0..1) while rendering, if known */
  progress?: number | null;
  className?: string;
}

export default function PlotPanel({ problem, src, running, progress = null, className = "" }: Props) {
  const { t } = useLang();
  const [zoomed, setZoomed] = useState(false);
  const results = problem.results ?? [];
  const [tab, setTab] = useState<string>(""); // "" = the animation
  const result = results.find((r) => r.name === tab);

  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setZoomed(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomed]);

  return (
    <section className={`flex min-w-0 flex-col bg-white ${className}`}>
      <PanelBar title={t("plot")}>
        {running && (
          <span className="text-blue-600" data-testid="plot-progress">
            <span className="animate-pulse">{t("plot.rendering")}</span>
            {progress !== null && <span className="ml-1 tabular-nums">{Math.round(progress * 100)}%</span>}
          </span>
        )}
        {src && (
          <button
            onClick={() => setZoomed(true)}
            className="text-gray-500 hover:text-gray-800"
            title="Zoom (click the plot)"
          >
            ⤢ {t("plot.zoom")}
          </button>
        )}
      </PanelBar>
      {running && progress !== null && (
        <div className="h-0.5 shrink-0 bg-blue-100">
          <div className="h-full bg-blue-600 transition-[width] duration-200" style={{ width: `${progress * 100}%` }} />
        </div>
      )}
      {results.length > 0 && (
        <div
          className="flex shrink-0 items-end gap-1 overflow-x-auto border-b border-gray-200 bg-gray-50 px-2 pt-1"
          role="tablist"
        >
          {[
            { name: "", label: `🎞 ${t("plot.animation")}` },
            ...results.map((r) => ({ name: r.name, label: `${ICON[r.kind]} ${r.name}` })),
          ].map((x) => (
            <button
              key={x.name || "animation"}
              role="tab"
              aria-selected={tab === x.name}
              onClick={() => setTab(x.name)}
              data-testid={`result-tab-${x.name || "animation"}`}
              className={`-mb-px shrink-0 rounded-t-md border px-3 py-1 font-mono text-[11px] ${
                tab === x.name
                  ? "border-gray-200 border-b-white bg-white text-blue-700"
                  : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              {x.label}
            </button>
          ))}
        </div>
      )}
      {result ? (
        <div
          className={`min-h-72 flex-1 xl:min-h-0 ${result.kind === "mesh" ? "relative" : "overflow-auto p-2"}`}
          data-testid="result-view"
        >
          {result.kind === "mesh" ? (
            <Suspense
              fallback={
                <div className="flex h-72 items-center justify-center">
                  <span className="h-8 w-8 animate-spin rounded-full border-2 border-gray-200 border-t-blue-600" />
                </div>
              }
            >
              <MeshView url={assetUrl(result.url)} legend={result.legend} fill />
            </Suspense>
          ) : result.kind === "image" ? (
            <img
              src={assetUrl(result.url)}
              alt={result.name}
              className="mx-auto max-h-full max-w-full object-contain"
            />
          ) : (
            <DataView dataset={result} tall />
          )}
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
