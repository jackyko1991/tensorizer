import { useEffect, useLayoutEffect, useRef, useState } from "react";
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
  const [big, setBig] = useBigPanel(); // ⤢ the results over the whole page

  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setZoomed(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomed]);

  // phones: the results browser ends exactly at the footer, so the Results tab fits the screen without scrolling
  const box = useRef<HTMLDivElement>(null);
  const [fitH, setFitH] = useState<number | null>(null);
  useLayoutEffect(() => {
    const fit = () => {
      const el = box.current;
      if (!el || window.matchMedia("(min-width: 1280px)").matches || el.offsetParent === null) return setFitH(null);
      const top = el.getBoundingClientRect().top + window.scrollY;
      const footer = document.querySelector("footer")?.getBoundingClientRect().height ?? 0;
      setFitH(Math.max(240, Math.floor(window.innerHeight - top - footer)));
    };
    fit();
    const ro = new ResizeObserver(fit);
    if (box.current?.parentElement) ro.observe(box.current.parentElement);
    window.addEventListener("resize", fit);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", fit);
    };
  }, [src, running]);

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
    <section className={`flex min-w-0 flex-col bg-white ${className} ${big ? BIG_PANEL : ""}`}>
      <PanelBar title={t("results")}>
        {running && (
          <span className="text-blue-600" data-testid="plot-progress">
            <span className="animate-pulse">{t("plot.rendering")}</span>
            {progress !== null && <span className="ml-1 tabular-nums">{Math.round(progress * 100)}%</span>}
          </span>
        )}
        <ExpandButton big={big} onChange={setBig} testId="results-expand" />
      </PanelBar>
      {running && progress !== null && (
        <div className="h-0.5 shrink-0 bg-blue-100">
          <div className="h-full bg-blue-600 transition-[width] duration-200" style={{ width: `${progress * 100}%` }} />
        </div>
      )}
      {main && !running ? (
        <div
          ref={box}
          style={fitH && !big ? { height: fitH } : undefined}
          className={big ? "min-h-0 flex-1" : "h-[50vh] shrink-0 xl:h-auto xl:min-h-0 xl:flex-1 xl:shrink"}
          data-testid="results-browser"
        >
          {/* a definite height on phones (the fill layout sizes to it); on desktop the panel's flex height */}
          <DataSources datasets={[main, ...(problem.results ?? [])]} rail={["files"]} listTitle={t("results")} fill expandable={false} />
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

/** A panel shown over the whole page: the same element grows (its state and scroll are kept); Esc brings it back. */
export const BIG_PANEL = "fixed! inset-3 z-50 h-auto! w-auto! rounded-lg border border-gray-200 shadow-2xl";

export function useBigPanel(): [boolean, (big: boolean) => void] {
  const [big, setBig] = useState(false);
  useEffect(() => {
    if (!big) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setBig(false);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [big]);
  return [big, setBig];
}

/** ⤢ / ✕ at the right end of a PanelBar: the same place on every panel. */
export function ExpandButton({ big, onChange, testId }: { big: boolean; onChange: (big: boolean) => void; testId?: string }) {
  const { t } = useLang();
  return (
    <button
      onClick={() => onChange(!big)}
      title={t(big ? "mesh.shrink" : "mesh.expand")}
      aria-label={t(big ? "mesh.shrink" : "mesh.expand")}
      aria-pressed={big}
      data-testid={testId}
      className="rounded px-1 text-sm leading-none text-gray-500 hover:bg-gray-200 hover:text-gray-800"
    >
      {big ? "✕" : "⤢"}
    </button>
  );
}
