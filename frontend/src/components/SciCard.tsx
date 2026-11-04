/** Easter egg: hover (mouse) or tap (touch) a scientist's name in problem text for a pixel-portrait card.
 *  Names are plain spans with a dotted underline (RichText `people`); this one listener shows a fixed-position card
 *  in a portal, so the text never moves. Tap anywhere else, or press Escape, to close it. */
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLang } from "../i18n";
import { portrait, SCIENTISTS, type Scientist } from "../scientists";

const BY_ID = new Map(SCIENTISTS.map((s) => [s.id, s]));
const WIDTH = 288;

function Portrait({ s }: { s: Scientist }) {
  const rects = useMemo(() => {
    const out: { x: number; y: number; w: number; c: string }[] = [];
    portrait(s).forEach((row, y) => {
      let x = 0;
      while (x < 16) {
        const c = row[x];
        let w = 1;
        while (x + w < 16 && row[x + w] === c) w++;
        if (c) out.push({ x, y, w, c });
        x += w;
      }
    });
    return out;
  }, [s]);
  return (
    <svg
      viewBox="0 0 16 16"
      width={48}
      height={48}
      shapeRendering="crispEdges"
      className="shrink-0 rounded bg-[#f5f1e6]"
    >
      {rects.map((r, i) => (
        <rect key={i} x={r.x} y={r.y} width={r.w} height={1} fill={r.c} />
      ))}
    </svg>
  );
}

export default function SciCard() {
  const { lang, t } = useLang();
  const [open, setOpen] = useState<{ s: Scientist; rect: DOMRect } | null>(null);
  const hideTimer = useRef<number>(0);

  useEffect(() => {
    const target = (e: Event) => (e.target as HTMLElement | null)?.closest?.("[data-sci]") as HTMLElement | null;
    const show = (el: HTMLElement) => {
      const s = BY_ID.get(el.dataset.sci ?? "");
      if (s) setOpen({ s, rect: el.getBoundingClientRect() });
    };
    const over = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      const el = target(e);
      if (el) {
        window.clearTimeout(hideTimer.current);
        show(el);
      } else if ((e.target as HTMLElement)?.closest?.("[data-sci-card]")) {
        window.clearTimeout(hideTimer.current);
      }
    };
    const out = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      if (target(e) || (e.target as HTMLElement)?.closest?.("[data-sci-card]")) {
        window.clearTimeout(hideTimer.current);
        hideTimer.current = window.setTimeout(() => setOpen(null), 160);
      }
    };
    const click = (e: MouseEvent) => {
      const el = target(e);
      if (el) {
        e.preventDefault();
        show(el);
      } else if (!(e.target as HTMLElement)?.closest?.("[data-sci-card]")) setOpen(null);
    };
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(null);
    const close = () => setOpen(null);
    document.addEventListener("pointerover", over);
    document.addEventListener("pointerout", out);
    document.addEventListener("click", click);
    document.addEventListener("keydown", key);
    window.addEventListener("scroll", close, true);
    return () => {
      document.removeEventListener("pointerover", over);
      document.removeEventListener("pointerout", out);
      document.removeEventListener("click", click);
      document.removeEventListener("keydown", key);
      window.removeEventListener("scroll", close, true);
    };
  }, []);

  if (!open) return null;
  const { s, rect } = open;
  const left = Math.min(Math.max(8, rect.left + rect.width / 2 - WIDTH / 2), window.innerWidth - WIDTH - 8);
  const above = rect.top > 140;
  const style = above ? { left, bottom: window.innerHeight - rect.top + 6 } : { left, top: rect.bottom + 6 };
  return createPortal(
    <div
      data-sci-card
      role="tooltip"
      style={{ ...style, width: WIDTH }}
      className="fixed z-[60] flex gap-3 rounded-lg border border-gray-200 bg-white p-3 text-left shadow-lg"
    >
      <Portrait s={s} />
      <div className="min-w-0 text-xs">
        <p className="font-semibold text-gray-900">
          {s.name} <span className="font-normal text-gray-500">({s.years})</span>
        </p>
        <p className="mt-1 leading-snug text-gray-600 italic" title={t("sci.joke")}>
          {s.meme[lang] ?? s.meme.en}
        </p>
      </div>
    </div>,
    document.body,
  );
}
