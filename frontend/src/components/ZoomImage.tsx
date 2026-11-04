import { useLayoutEffect, useRef, useState } from "react";
import { useLang } from "../i18n";

const STEP = 1.5;
const MAX = 8; // × the image's own pixels

/** An image in the expanded viewer. Top right: zoom-in / zoom-out mode toggle, Fit, 100 % (native pixels) and the
 * current zoom. A click on the image zooms by the mode at the pointer (Shift inverts it); right-click zooms out. */
export default function ZoomImage({ src, alt }: { src: string; alt: string }) {
  const { t } = useLang();
  const box = useRef<HTMLDivElement>(null);
  const img = useRef<HTMLImageElement>(null);
  const [z, setZ] = useState<number | null>(null); // × natural size; null = fit the frame
  const [mode, setMode] = useState<"in" | "out">("in");
  const [, setLoaded] = useState(0);
  // the point (fractions of the image) to keep under the pointer after a zoom, and where the pointer was
  const anchor = useRef<{ fx: number; fy: number; px: number; py: number } | null>(null);

  const natural = () => ({ w: img.current?.naturalWidth || 1, h: img.current?.naturalHeight || 1 });
  const fitScale = () => {
    const b = box.current;
    const n = natural();
    if (!b) return 1;
    return Math.min(b.clientWidth / n.w, b.clientHeight / n.h);
  };
  const current = () => z ?? fitScale();

  useLayoutEffect(() => {
    const a = anchor.current;
    const b = box.current;
    const i = img.current;
    if (!a || !b || !i) return;
    anchor.current = null;
    b.scrollLeft = i.offsetLeft + a.fx * i.offsetWidth - a.px;
    b.scrollTop = i.offsetTop + a.fy * i.offsetHeight - a.py;
  }, [z]);

  const setScale = (next: number | null, e?: { clientX: number; clientY: number }) => {
    const b = box.current;
    if (!b) return;
    const s = current();
    const n = natural();
    const br = b.getBoundingClientRect();
    const w = n.w * s;
    const h = n.h * s;
    // where the drawn image is now: centred while it is smaller than the frame, else scrolled
    const left = br.left + Math.max(0, (b.clientWidth - w) / 2) - (w > b.clientWidth ? b.scrollLeft : 0);
    const top = br.top + Math.max(0, (b.clientHeight - h) / 2) - (h > b.clientHeight ? b.scrollTop : 0);
    const cx = e?.clientX ?? br.left + b.clientWidth / 2;
    const cy = e?.clientY ?? br.top + b.clientHeight / 2;
    anchor.current = {
      fx: Math.min(1, Math.max(0, (cx - left) / w)),
      fy: Math.min(1, Math.max(0, (cy - top) / h)),
      px: cx - br.left,
      py: cy - br.top,
    };
    setZ(next);
  };

  const zoom = (dir: "in" | "out", e?: { clientX: number; clientY: number }) => {
    const s = current();
    const fit = fitScale();
    if (dir === "in") return setScale(Math.min(MAX, s * STEP), e);
    // going out always shrinks: it snaps to "fit" when it passes it from above, then keeps going down to a floor
    const floor = Math.min(fit, 1) / 4;
    if (s <= floor * 1.01) return;
    const next = Math.max(floor, s / STEP);
    setScale(s > fit * 1.01 && next < fit ? null : next, e);
  };

  const s = current();
  // ⧉ copy the image (the frame on screen, for a GIF) to the clipboard as PNG
  const [copied, setCopied] = useState<"" | "ok" | "fail">("");
  const copy = async () => {
    const i = img.current;
    try {
      if (!i || !navigator.clipboard?.write || typeof ClipboardItem === "undefined") throw new Error("no clipboard");
      const c = document.createElement("canvas");
      c.width = i.naturalWidth;
      c.height = i.naturalHeight;
      c.getContext("2d")!.drawImage(i, 0, 0);
      const blob = await new Promise<Blob>((ok, bad) => c.toBlob((b) => (b ? ok(b) : bad(new Error("png"))), "image/png"));
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      setCopied("ok");
    } catch {
      setCopied("fail");
    }
    window.setTimeout(() => setCopied(""), 1500);
  };
  const btn = (active: boolean) =>
    `rounded px-1.5 py-0.5 ${active ? "bg-blue-600 text-white" : "hover:bg-gray-100"}`;
  return (
    <div className="absolute inset-0">
      <div ref={box} className="h-full w-full overflow-auto" data-testid="zoom-box">
        <div className="flex min-h-full w-max min-w-full">
          <img
            ref={img}
            src={src}
            alt={alt}
            title={t("image.zoomHint")}
            draggable={false}
            onLoad={() => {
              setLoaded((k) => k + 1);
              setZ(fitScale() >= 1 ? 1 : null); // open at 100 %, or fit when the image is bigger than the frame
            }}
            onClick={(e) => zoom(e.shiftKey ? (mode === "in" ? "out" : "in") : mode, e)}
            onContextMenu={(e) => {
              e.preventDefault();
              zoom("out", e);
            }}
            data-testid="zoom-image"
            className={`m-auto max-w-none select-none ${mode === "in" ? "cursor-zoom-in" : "cursor-zoom-out"}`}
            style={{ width: natural().w * s, height: natural().h * s }}
          />
        </div>
      </div>
      <div
        className="absolute top-2 right-4 flex items-center gap-0.5 rounded-md border border-gray-200 bg-white/95 p-0.5 text-xs text-gray-700 shadow-sm"
        data-testid="zoom-bar"
      >
        <button
          onClick={() => setMode("in")}
          aria-pressed={mode === "in"}
          title={t("image.zoomIn")}
          aria-label={t("image.zoomIn")}
          data-testid="zoom-mode-in"
          className={btn(mode === "in")}
        >
          🔍+
        </button>
        <button
          onClick={() => setMode("out")}
          aria-pressed={mode === "out"}
          title={t("image.zoomOut")}
          aria-label={t("image.zoomOut")}
          data-testid="zoom-mode-out"
          className={btn(mode === "out")}
        >
          🔍−
        </button>
        <span className="mx-1 h-4 w-px bg-gray-200" />
        <button onClick={() => setScale(null)} className={btn(z === null)} data-testid="zoom-fit">
          {t("image.fit")}
        </button>
        <button onClick={() => setScale(1)} className={btn(z === 1)} data-testid="zoom-100">
          100%
        </button>
        <span className="mx-1 h-4 w-px bg-gray-200" />
        <button
          onClick={copy}
          className={btn(false)}
          title={t("image.copyHint")}
          data-testid="zoom-copy"
        >
          {copied === "ok" ? `✓ ${t("formula.copied")}` : copied === "fail" ? t("formula.copyFail") : `⧉ ${t("formula.copy")}`}
        </button>
        <span className="w-12 text-right tabular-nums text-gray-500" data-testid="zoom-level">
          {Math.round(s * 100)}%
        </span>
      </div>
    </div>
  );
}
