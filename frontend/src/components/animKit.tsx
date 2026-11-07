/**
 * Shared pieces of the Coding Skills animations (For loop vs tensor, Techniques): site colours, SVG text and
 * cell primitives, the play/pause clock (pauses off-screen, starts paused under prefers-reduced-motion) and
 * the control bar.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import { usePageText } from "../content/pages";

/** Site colours: gray idle, blue tensor / done (as the chart's CPU bars), amber for the current element. */
export const COL = {
  idle: "#f3f4f6",
  idleLine: "#d1d5db",
  done: "#93c5fd",
  doneLine: "#60a5fa",
  blue: "#2563eb",
  blueLight: "#dbeafe",
  cur: "#f59e0b",
  curLight: "#fde68a",
  text: "#374151",
  muted: "#6b7280",
  green: "#16a34a",
  greenLight: "#dcfce7",
  red: "#dc2626",
  redLight: "#fee2e2",
  violet: "#7c3aed",
  violetLight: "#ede9fe",
};

export const SPEEDS = [0.5, 1, 2, 4];

/** Blue sequential scale for a value in [0, 1]. */
export function blueScale(v: number): string {
  const t = Math.min(Math.max(v, 0), 1);
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(239, 30)},${lerp(246, 64)},${lerp(255, 175)})`; // blue-50 -> blue-800
}

export type Kind = "sans" | "mono" | "math";

/** Font of a label: the page's sans text, the CodeBlock code font, or KaTeX's fonts for math. */
export const FAMILY: Record<Kind, string> = { sans: "inherit", mono: "var(--font-code)", math: "KaTeX_Main, serif" };

/**
 * A tiny TeX-like renderer for SVG labels, so they look like the $...$ math in the card text: letters in
 * KaTeX_Math italic, digits and operators upright in KaTeX_Main, `_i` / `_{ij}` subscripts, `^2` superscripts.
 */
export function mathSpans(src: string): ReactNode[] {
  const out: ReactNode[] = [];
  let shift = 0; // current dy offset in em
  let k = 0;
  const run = (text: string, script: 0 | 1 | -1) => {
    const target = script === 1 ? 0.35 : script === -1 ? -0.45 : 0;
    for (const ch of text) {
      const letter = /\p{L}/u.test(ch);
      out.push(
        <tspan
          key={k++}
          dy={target !== shift ? `${(target - shift) * (script ? 1 / 0.72 : 1)}em` : undefined}
          fontFamily={letter ? "KaTeX_Math, serif" : "KaTeX_Main, serif"}
          fontStyle={letter ? "italic" : "normal"}
          fontSize={script ? "72%" : undefined}
        >
          {ch === " " ? " " : ch}
        </tspan>,
      );
      shift = target;
    }
  };
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if ((c === "_" || c === "^") && i + 1 < src.length) {
      let arg = src[i + 1];
      if (arg === "{") {
        const end = src.indexOf("}", i + 2);
        arg = src.slice(i + 2, end);
        i = end;
      } else i++;
      run(arg, c === "_" ? 1 : -1);
    } else run(c, 0);
  }
  return out;
}

export function Txt({ x, y, children, size = 8, color = COL.text, anchor = "start", weight, kind = "sans", opacity, pre }: {
  x: number;
  y: number;
  children: ReactNode;
  size?: number;
  color?: string;
  anchor?: "start" | "middle" | "end";
  weight?: number;
  kind?: Kind;
  opacity?: number;
  /** keep runs of spaces (code indentation) */
  pre?: boolean;
}) {
  const body = kind === "math" ? mathSpans(Array.isArray(children) ? children.join("") : String(children)) : children;
  return (
    <text
      x={x}
      y={y}
      fontSize={size}
      fill={color}
      textAnchor={anchor}
      style={{ fontFamily: FAMILY[kind] }}
      fontWeight={weight}
      opacity={opacity}
      xmlSpace={pre ? "preserve" : undefined}
    >
      {body}
    </text>
  );
}

export function Cell({ x, y, s, fill, stroke = COL.idleLine, sw = 0.75, label, labelColor = COL.text, size = 8, dash, opacity }: {
  x: number;
  y: number;
  s: number;
  fill: string;
  stroke?: string;
  sw?: number;
  label?: string | number;
  labelColor?: string;
  size?: number;
  dash?: string;
  opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      <rect x={x} y={y} width={s} height={s} rx={1.5} fill={fill} stroke={stroke} strokeWidth={sw} strokeDasharray={dash} />
      {label != null && (
        <Txt x={x + s / 2} y={y + s / 2 + size * 0.36} size={size} anchor="middle" color={labelColor}>
          {label}
        </Txt>
      )}
    </g>
  );
}

/**
 * The animation clock: tick g runs 0 .. cycle and wraps. Pauses while the element is off-screen, starts paused
 * under prefers-reduced-motion.
 */
export function useAnimClock(cycle: number, tickMs: number) {
  const root = useRef<HTMLDivElement>(null);
  const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const [playing, setPlaying] = useState(!reduced);
  const [visible, setVisible] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [g, setG] = useState(0);

  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return setVisible(true);
    const ob = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0 });
    ob.observe(el);
    return () => ob.disconnect();
  }, []);

  useEffect(() => {
    if (!playing || !visible) return;
    const h = window.setInterval(() => setG((v) => (v + 1) % (cycle + 1)), tickMs / speed);
    return () => window.clearInterval(h);
  }, [playing, visible, speed, tickMs, cycle]);

  const step = () => {
    setPlaying(false);
    setG((v) => (v + 1) % (cycle + 1));
  };
  return { root, g, setG, playing, setPlaying, speed, setSpeed, step };
}

export const btn =
  "rounded-md border border-gray-300 bg-white px-2.5 py-1 text-xs text-gray-700 hover:bg-gray-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";

/** Play / pause, single step, restart and speed. */
export function AnimControls({ clock }: { clock: ReturnType<typeof useAnimClock> }) {
  const T = usePageText();
  const { playing, setPlaying, setG, speed, setSpeed, step } = clock;
  return (
    <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
      <button type="button" className={btn} onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
        {playing ? `❚❚ ${T("loops.anim.pause")}` : `▶ ${T("loops.anim.play")}`}
      </button>
      <button type="button" className={btn} onClick={step}>
        {T("loops.anim.step")} ›
      </button>
      <button type="button" className={btn} onClick={() => setG(0)}>
        ↺ {T("loops.anim.restart")}
      </button>
      <label className="flex items-center gap-1.5 text-xs text-gray-600">
        {T("loops.anim.speed")}
        <select
          className="rounded-md border border-gray-300 bg-white px-1.5 py-1 text-xs"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        >
          {SPEEDS.map((v) => (
            <option key={v} value={v}>
              {v}×
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
