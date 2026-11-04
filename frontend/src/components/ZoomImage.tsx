import { useLayoutEffect, useRef, useState } from "react";
import { useLang } from "../i18n";

// Zoom levels as multiples of the fitted size (1 = the whole image fills the frame, centred).
const LEVELS = [1, 1.5, 2, 3, 4, 6];

/** An image in the expanded viewer: click zooms in at the pointer, Shift+click or right-click zooms out; − / + too. */
export default function ZoomImage({ src, alt }: { src: string; alt: string }) {
  const { t } = useLang();
  const box = useRef<HTMLDivElement>(null);
  const img = useRef<HTMLImageElement>(null);
  const [level, setLevel] = useState(0);
  const [fitW, setFitW] = useState(0); // width of the image when it just fits the frame (px)
  // the point (fractions of the image) to keep under the pointer after a zoom, and where the pointer was
  const anchor = useRef<{
    fx: number;
    fy: number;
    px: number;
    py: number;
  } | null>(null);
  const scale = LEVELS[level];

  const fitWidth = () => {
    const b = box.current;
    const i = img.current;
    if (!b || !i || !i.naturalWidth) return 0;
    return Math.min(
      b.clientWidth,
      (b.clientHeight * i.naturalWidth) / i.naturalHeight,
    );
  };

  useLayoutEffect(() => {
    const a = anchor.current;
    const b = box.current;
    const i = img.current;
    if (!a || !b || !i) return;
    anchor.current = null;
    b.scrollLeft = i.offsetLeft + a.fx * i.offsetWidth - a.px;
    b.scrollTop = i.offsetTop + a.fy * i.offsetHeight - a.py;
  }, [level]);

  const zoom = (dir: 1 | -1, e?: { clientX: number; clientY: number }) => {
    const next = Math.min(LEVELS.length - 1, Math.max(0, level + dir));
    if (next === level) return;
    const b = box.current!;
    const i = img.current!;
    const br = b.getBoundingClientRect();
    // where the drawn image is now (at fit it is centred inside the box)
    const w = level === 0 ? fitWidth() : i.offsetWidth;
    const h = (w * i.naturalHeight) / Math.max(i.naturalWidth, 1);
    const left =
      level === 0
        ? br.left + (b.clientWidth - w) / 2
        : i.getBoundingClientRect().left;
    const top =
      level === 0
        ? br.top + (b.clientHeight - h) / 2
        : i.getBoundingClientRect().top;
    const cx = e?.clientX ?? br.left + br.width / 2;
    const cy = e?.clientY ?? br.top + br.height / 2;
    anchor.current = {
      fx: Math.min(1, Math.max(0, (cx - left) / w)),
      fy: Math.min(1, Math.max(0, (cy - top) / h)),
      px: cx - br.left,
      py: cy - br.top,
    };
    if (level === 0) setFitW(w);
    setLevel(next);
  };

  return (
    <div className="absolute inset-0">
      <div
        ref={box}
        className="h-full w-full overflow-auto"
        data-testid="zoom-box"
      >
        {level === 0 ? (
          <img
            ref={img}
            src={src}
            alt={alt}
            title={t("image.zoomHint")}
            draggable={false}
            onClick={(e) => zoom(e.shiftKey ? -1 : 1, e)}
            onContextMenu={(e) => e.preventDefault()}
            data-testid="zoom-image"
            className="h-full w-full cursor-zoom-in object-contain select-none"
          />
        ) : (
          <div className="flex min-h-full w-max min-w-full">
            <img
              ref={img}
              src={src}
              alt={alt}
              title={t("image.zoomHint")}
              draggable={false}
              onClick={(e) => zoom(e.shiftKey ? -1 : 1, e)}
              onContextMenu={(e) => {
                e.preventDefault();
                zoom(-1, e);
              }}
              data-testid="zoom-image"
              className={`m-auto max-w-none select-none ${
                level === LEVELS.length - 1
                  ? "cursor-zoom-out"
                  : "cursor-zoom-in"
              }`}
              style={{ width: fitW * scale }}
            />
          </div>
        )}
      </div>
      <div className="absolute right-2 bottom-2 flex items-center gap-0.5 rounded-md border border-gray-200 bg-white/90 p-0.5 text-xs text-gray-700 shadow-sm">
        <button
          onClick={() => zoom(-1)}
          disabled={level === 0}
          aria-label={t("image.zoomOut")}
          title={t("image.zoomOut")}
          className="w-6 rounded hover:bg-gray-100 disabled:opacity-30"
        >
          −
        </button>
        <button
          onClick={() => setLevel(0)}
          className="min-w-12 rounded px-1 tabular-nums hover:bg-gray-100"
          title={t("image.fit")}
          data-testid="zoom-level"
        >
          {level === 0 ? t("image.fit") : `${Math.round(scale * 100)}%`}
        </button>
        <button
          onClick={() => zoom(1)}
          disabled={level === LEVELS.length - 1}
          aria-label={t("image.zoomIn")}
          title={t("image.zoomIn")}
          className="w-6 rounded hover:bg-gray-100 disabled:opacity-30"
        >
          +
        </button>
      </div>
    </div>
  );
}
