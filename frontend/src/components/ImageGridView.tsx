/** An image dataset (.npz with uint8 (N,H,W) images and their labels, e.g. an MNIST subset): a paged grid of
 *  pixel-crisp thumbnails with their labels, a class filter, a page size and a split chooser (x_train / x_test);
 *  clicking a thumbnail enlarges it (Esc closes, ← → step). The archive is fetched when shown and kept for the
 *  session. */
import { useEffect, useMemo, useRef, useState } from "react";
import { useLang } from "../i18n";
import { parseImageNpz, rgba, type ImageSet } from "../npy";

const cache = new Map<string, Promise<ImageSet[]>>();

function load(url: string): Promise<ImageSet[]> {
  if (!cache.has(url)) {
    const p = fetch(url)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((b) => parseImageNpz(new Uint8Array(b)));
    p.catch(() => cache.delete(url)); // retry next time
    cache.set(url, p);
  }
  return cache.get(url)!;
}

const PAGE_SIZES = [24, 48, 96, 192];

/** One image on a canvas of its own pixel size; CSS scales it with nearest-neighbour sampling. */
function Pixels({
  set,
  index,
  className,
  width,
}: {
  set: ImageSet;
  index: number;
  className?: string;
  /** CSS width (the height follows the aspect ratio). */
  width?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (ctx) ctx.putImageData(new ImageData(rgba(set, index), set.width, set.height), 0, 0);
  }, [set, index]);
  return (
    <canvas
      ref={ref}
      width={set.width}
      height={set.height}
      className={className}
      style={{ imageRendering: "pixelated", aspectRatio: `${set.width} / ${set.height}`, width }}
    />
  );
}

export default function ImageGridView({ url, fill = false }: { url: string; fill?: boolean }) {
  const { t } = useLang();
  const [sets, setSets] = useState<ImageSet[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [which, setWhich] = useState(0);
  const [cls, setCls] = useState<number | null>(null);
  const [pageSize, setPageSize] = useState(48);
  const [page, setPage] = useState(0);
  const [zoom, setZoom] = useState<number | null>(null); // position in the filtered list
  const grid = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    setSets(null);
    setError(null);
    setWhich(0);
    load(url).then(
      (s) => live && setSets(s),
      (e) => live && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, [url]);

  const set = sets?.[Math.min(which, sets.length - 1)] ?? null;
  const classes = useMemo(() => (set?.labels ? [...new Set(set.labels)].sort((a, b) => a - b) : []), [set]);
  // indices of the images that pass the class filter
  const shown = useMemo(() => {
    if (!set) return [];
    const all = Array.from({ length: set.count }, (_, i) => i);
    return cls === null || !set.labels ? all : all.filter((i) => set.labels![i] === cls);
  }, [set, cls]);
  const pages = Math.max(1, Math.ceil(shown.length / pageSize));
  const current = Math.min(page, pages - 1);
  const visible = shown.slice(current * pageSize, (current + 1) * pageSize);
  const goPage = (p: number) => {
    setPage(Math.max(0, Math.min(pages - 1, p)));
    grid.current?.scrollTo({ top: 0 });
  };

  // Esc closes the enlarged image before the expanded page hears it (capture phase); ← → step through the filter
  useEffect(() => {
    if (zoom === null) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") setZoom(null);
      else if (e.key === "ArrowLeft") setZoom((z) => (z === null ? z : Math.max(0, z - 1)));
      else if (e.key === "ArrowRight") setZoom((z) => (z === null ? z : Math.min(shown.length - 1, z + 1)));
      else return;
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [zoom, shown.length]);

  if (error)
    return (
      <p className="p-3 text-xs text-red-600" data-testid="images-error">
        {t("images.failed")} {error}
      </p>
    );
  if (!set)
    return (
      <div className={`flex items-center justify-center gap-2 text-xs text-gray-400 ${fill ? "h-full" : "h-48"}`}>
        <span className="h-6 w-6 animate-spin rounded-full border-2 border-gray-200 border-t-blue-600" />
        {t("images.loading")}
      </div>
    );

  const select = "rounded border border-gray-200 bg-white px-1 py-0.5 text-xs text-gray-700";
  const zoomed = zoom !== null ? shown[Math.min(zoom, shown.length - 1)] : null;
  return (
    <div className={`flex min-h-0 flex-col ${fill ? "h-full" : ""}`} data-testid="image-grid">
      <div className="sticky top-0 z-10 flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-gray-100 bg-white px-3 py-1.5 text-xs text-gray-600">
        {sets!.length > 1 && (
          <label className="flex items-center gap-1">
            {t("images.split")}
            <select
              className={select}
              value={which}
              onChange={(e) => {
                setWhich(Number(e.target.value));
                setCls(null);
                setPage(0);
              }}
            >
              {sets!.map((s, i) => (
                <option key={s.name} value={i}>
                  {s.name} ({s.count})
                </option>
              ))}
            </select>
          </label>
        )}
        {set.labels && (
          <label className="flex items-center gap-1">
            {t("images.class")}
            <select
              className={select}
              value={cls ?? ""}
              data-testid="images-class"
              onChange={(e) => {
                setCls(e.target.value === "" ? null : Number(e.target.value));
                setPage(0);
              }}
            >
              <option value="">{t("images.all")}</option>
              {classes.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="flex items-center gap-1">
          {t("images.perPage")}
          <select
            className={select}
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(0);
            }}
          >
            {PAGE_SIZES.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <span className="text-gray-500" data-testid="images-count">
          {t("images.summary")
            .replace("{n}", String(set.count))
            .replace("{h}", String(set.height))
            .replace("{w}", String(set.width))}
          {classes.length > 0 && `, ${t("images.classes").replace("{k}", String(classes.length))}`}
          {cls !== null && ` · ${t("images.shown").replace("{n}", String(shown.length))}`}
        </span>
        <span className="ml-auto flex items-center gap-1 whitespace-nowrap">
          <button
            onClick={() => goPage(current - 1)}
            disabled={current === 0}
            title={t("images.prev")}
            aria-label={t("images.prev")}
            className="rounded px-1.5 hover:bg-gray-100 disabled:opacity-30"
          >
            ‹
          </button>
          <span className="tabular-nums">
            {current + 1} / {pages}
          </span>
          <button
            onClick={() => goPage(current + 1)}
            disabled={current >= pages - 1}
            title={t("images.next")}
            aria-label={t("images.next")}
            className="rounded px-1.5 hover:bg-gray-100 disabled:opacity-30"
          >
            ›
          </button>
        </span>
      </div>
      <div
        ref={grid}
        className={`grid content-start gap-1.5 p-2 ${fill ? "min-h-0 flex-1 overflow-auto" : ""}`}
        style={{ gridTemplateColumns: "repeat(auto-fill, minmax(4rem, 1fr))" }}
      >
        {visible.map((i, k) => (
          <button
            key={i}
            onClick={() => setZoom(current * pageSize + k)}
            title={`#${i}${set.labels ? ` · ${t("images.label")} ${set.labels[i]}` : ""}`}
            className="group flex flex-col items-center rounded border border-gray-200 bg-white p-1 hover:border-blue-400"
            data-testid="image-thumb"
          >
            <Pixels set={set} index={i} className="w-full bg-black" />
            <span className="mt-0.5 flex w-full justify-between font-mono text-[10px] leading-tight text-gray-500">
              <span className="font-semibold text-gray-800">{set.labels ? set.labels[i] : ""}</span>
              <span className="text-gray-400">#{i}</span>
            </span>
          </button>
        ))}
      </div>
      {zoomed !== null && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-4"
          onClick={() => setZoom(null)}
          role="dialog"
          aria-modal="true"
          data-testid="image-zoom"
        >
          <figure
            className="flex max-h-full flex-col items-center gap-2 rounded-lg bg-white p-3 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <Pixels
              set={set}
              index={zoomed}
              className="bg-black"
              width={`min(85vw, ${((70 * set.width) / set.height).toFixed(2)}vh, ${set.width * 24}px)`}
            />
            <figcaption className="flex w-full items-center gap-3 text-sm text-gray-700">
              {set.labels && (
                <span>
                  {t("images.label")} <b className="font-mono">{set.labels[zoomed]}</b>
                </span>
              )}
              <span>
                {t("images.index")} <span className="font-mono">{zoomed}</span>
              </span>
              <span className="text-xs text-gray-400">
                {set.name} · {set.height}×{set.width}
              </span>
              <button
                onClick={() => setZoom(null)}
                title={t("images.close")}
                aria-label={t("images.close")}
                className="ml-auto rounded px-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-800"
              >
                ✕
              </button>
            </figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}
