import { useEffect, useRef, useState } from "react";

/**
 * One tooltip for the whole site, in the same white card style as the series hover card. Any element with a
 * `title` keeps working as before: on hover its title moves to `data-tip` (so the browser's black tooltip never
 * shows) and this card appears below it (above when there is no room). Line breaks in a title are kept.
 */
export default function HoverTip() {
  const [tip, setTip] = useState<{ text: string; x: number; y: number; above: boolean } | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const current = useRef<Element | null>(null);
  const card = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const hide = () => {
      window.clearTimeout(timer.current);
      current.current = null;
      setTip(null);
    };
    const over = (e: MouseEvent) => {
      const el = (e.target as Element | null)?.closest?.("[title], [data-tip]");
      if (!el || el instanceof SVGElement) return;
      const title = el.getAttribute("title");
      if (title) {
        el.setAttribute("data-tip", title); // React may set title again later: picked up on the next hover
        el.removeAttribute("title");
      }
      if (el === current.current) return;
      const text = el.getAttribute("data-tip");
      window.clearTimeout(timer.current);
      current.current = el;
      setTip(null);
      if (!text) return;
      timer.current = window.setTimeout(() => {
        const r = el.getBoundingClientRect();
        const above = r.bottom + 90 > window.innerHeight;
        setTip({ text, x: r.left + r.width / 2, y: above ? r.top - 6 : r.bottom + 6, above });
      }, 350);
    };
    const out = (e: MouseEvent) => {
      const to = e.relatedTarget as Node | null;
      if (current.current && to && current.current.contains(to)) return;
      if (current.current && e.target instanceof Node && current.current.contains(e.target)) hide();
    };
    document.addEventListener("mouseover", over, true);
    document.addEventListener("mouseout", out, true);
    document.addEventListener("mousedown", hide, true);
    window.addEventListener("scroll", hide, true);
    window.addEventListener("blur", hide);
    return () => {
      document.removeEventListener("mouseover", over, true);
      document.removeEventListener("mouseout", out, true);
      document.removeEventListener("mousedown", hide, true);
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("blur", hide);
    };
  }, []);

  // keep the card inside the window
  const [dx, setDx] = useState(0);
  useEffect(() => {
    if (!tip || !card.current) return setDx(0);
    const r = card.current.getBoundingClientRect();
    const left = tip.x - r.width / 2;
    setDx(Math.max(8 - left, Math.min(0, window.innerWidth - 8 - (left + r.width))));
  }, [tip]);

  if (!tip) return null;
  return (
    <div
      ref={card}
      role="tooltip"
      data-testid="hover-tip"
      className="pointer-events-none fixed z-[100] max-w-xs rounded-lg border border-gray-200 bg-white px-2.5 py-1.5 text-xs leading-relaxed whitespace-pre-line text-gray-700 shadow-lg"
      style={{
        left: tip.x + dx,
        top: tip.y,
        transform: `translate(-50%, ${tip.above ? "-100%" : "0"})`,
      }}
    >
      {tip.text}
    </div>
  );
}
