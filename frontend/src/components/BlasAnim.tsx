/**
 * Animations of the BLAS tab (one or more per card) and the shared pieces the NVLink tab reuses: the step
 * animation frame (SVG picture, numbered step list, caption, controls) and the plain line-by-line notes under a
 * code block. Plans and numbers: ../blasNvlinkAnim.ts.
 */
import { type ReactNode } from "react";
import { usePageText, type PageKey } from "../content/pages";
import {
  ATOMIC_PARTS,
  ATOMIC_RUNS,
  BLAS_CODE,
  BLAS_PLAN,
  NV_CODE,
  OVERHEAD,
  SUM_INPUTS,
  accumulate,
  crossover,
  fmt9,
  gemmLoads,
  levelCost,
  modelGflops,
  poolTime,
  stepAt,
  sumOrder,
  totalTicks,
  type Add,
  type BlasCase,
  type StepPlan,
  type SumOrder,
} from "../blasNvlinkAnim";
import { AnimControls, COL, Cell, Txt, blueScale, useAnimClock } from "./animKit";
import { RichText } from "./Tex";

export const W = 280; // SVG user units, scaled to the column width
const HOLD = 6; // ticks the finished picture stays before the cycle restarts

export interface DrawProps {
  step: number;
  /** tick inside the step, 0 .. steps[step] - 1 */
  t: number;
  /** progress through the step, (0, 1] */
  f: number;
  /** the case's own short labels: k("flops") = T("<prefix>.anim.<case>.flops") */
  k: (label: string) => string;
  T: (key: PageKey) => string;
}

export const fill = (s: string, v: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (m, key) => String(v[key] ?? m));

export function Arrow({ x1, y1, x2, y2, color = COL.cur, w = 1.4, dash, head = 4 }: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color?: string;
  w?: number;
  dash?: string;
  head?: number;
}) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  if (Math.hypot(x2 - x1, y2 - y1) < 0.5) return null;
  const hx = (s: number) => x2 - head * Math.cos(a + s * 0.45);
  const hy = (s: number) => y2 - head * Math.sin(a + s * 0.45);
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2 - 0.6 * head * Math.cos(a)} y2={y2 - 0.6 * head * Math.sin(a)} stroke={color} strokeWidth={w} strokeDasharray={dash} />
      <path d={`M ${x2} ${y2} L ${hx(1)} ${hy(1)} L ${hx(-1)} ${hy(-1)} Z`} fill={color} />
    </g>
  );
}

/** A rounded box with a centred label. */
export function Box({ x, y, w, h, label, fill: bg = "#fff", stroke = "#94a3b8", color = COL.text, size = 7, kind = "sans", weight, sw = 0.9, dash }: {
  x: number;
  y: number;
  w: number;
  h: number;
  label?: ReactNode;
  fill?: string;
  stroke?: string;
  color?: string;
  size?: number;
  kind?: "sans" | "mono" | "math";
  weight?: number;
  sw?: number;
  dash?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={3} fill={bg} stroke={stroke} strokeWidth={sw} strokeDasharray={dash} />
      {label != null && (
        <Txt x={x + w / 2} y={y + h / 2 + size * 0.36} size={size} anchor="middle" color={color} kind={kind} weight={weight}>
          {label}
        </Txt>
      )}
    </g>
  );
}

/** Point at fraction u of a polyline. */
export function along(pts: [number, number][], u: number): [number, number] {
  const seg = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]));
  let d = Math.min(Math.max(u, 0), 1) * seg.reduce((a, b) => a + b, 0);
  for (let i = 0; i < seg.length; i++) {
    if (d <= seg[i] || i === seg.length - 1) {
      const r = seg[i] ? d / seg[i] : 0;
      return [pts[i][0] + r * (pts[i + 1][0] - pts[i][0]), pts[i][1] + r * (pts[i + 1][1] - pts[i][1])];
    }
    d -= seg[i];
  }
  return pts[pts.length - 1];
}

/** Log-scale map of v in [lo, hi] onto [a, b]. */
export const logMap = (lo: number, hi: number, a: number, b: number) => (v: number) =>
  a + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (b - a);

/**
 * One step animation: the SVG picture, the numbered list of its steps (the current one highlighted), a caption
 * and the shared play / step / restart / speed controls. Text: `<prefix>.anim.<id>.title / .s<k> / .caption`.
 */
export function StepAnim({ prefix, id, plan, h, draw }: {
  prefix: "blas" | "nvlink";
  id: string;
  plan: StepPlan;
  h: number;
  draw: (p: DrawProps) => ReactNode;
}) {
  const T = usePageText();
  const key = (s: string) => `${prefix}.anim.${id}.${s}` as PageKey;
  const k = (s: string) => T(key(s));
  const clock = useAnimClock(totalTicks(plan) + HOLD, plan.tick);
  const { step, t, f } = stepAt(plan, clock.g);
  const n = plan.steps.length;
  const caption = k("caption");
  return (
    <div ref={clock.root} className="mt-4" data-testid={`${prefix}-anim-${id}`}>
      <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <figcaption className="flex items-center justify-between gap-2 border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">
          <span className="min-w-0">
            <RichText text={k("title")} />
          </span>
          {n > 1 && (
            <span className="shrink-0 font-mono tabular-nums text-blue-700">
              {step + 1}/{n}
            </span>
          )}
        </figcaption>
        <div className="grid md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <svg viewBox={`0 0 ${W} ${h}`} className="mx-auto block h-auto w-full max-w-[560px] p-1" role="img" aria-label={caption.replace(/[$`\\*]/g, "")}>
            {draw({ step, t, f, k, T })}
          </svg>
          <ol className="space-y-1 border-t border-gray-100 p-3 text-xs leading-snug md:border-t-0 md:border-l">
            {plan.steps.map((_, i) => (
              <li
                key={i}
                className={`flex gap-2 rounded px-1.5 py-1 ${i === step ? "bg-blue-50 text-gray-900" : "text-gray-500"}`}
                aria-current={i === step ? "step" : undefined}
              >
                <span
                  className={`mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${
                    i === step ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-600"
                  }`}
                >
                  {i + 1}
                </span>
                <span className="min-w-0">
                  <RichText text={k(`s${i}`)} />
                </span>
              </li>
            ))}
          </ol>
        </div>
        <p className="border-t border-gray-100 px-3 py-2 text-xs leading-snug text-gray-600">
          <RichText text={caption} />
        </p>
      </figure>
      <AnimControls clock={clock} />
    </div>
  );
}

/** Plain explanation of a code block, one note per line or group of lines (`<prefix>.anim.code.<id>.<k>`). */
export function CodeNotes({ prefix, id }: { prefix: "blas" | "nvlink"; id: string }) {
  const T = usePageText();
  const n = (prefix === "blas" ? (BLAS_CODE as Record<string, number>) : (NV_CODE as Record<string, number>))[id] ?? 0;
  return (
    <div className="mt-2 rounded-lg border border-gray-200 bg-gray-50/70 px-3 py-2" data-testid={`${prefix}-code-notes-${id}`}>
      <div className="text-xs font-semibold text-gray-600">{T(`${prefix}.anim.codeNotes` as PageKey)}</div>
      <ul className="mt-1 list-disc space-y-1 pl-4 text-sm leading-relaxed text-gray-700">
        {Array.from({ length: n }, (_, i) => (
          <li key={i}>
            <RichText text={T(`${prefix}.anim.code.${id}.${i}` as PageKey)} />
          </li>
        ))}
      </ul>
    </div>
  );
}

// --- 1. BLAS levels -------------------------------------------------------------------------------------

function Counters({ x, y, flops, moved, k }: { x: number; y: number; flops: number; moved: number; k: DrawProps["k"] }) {
  const ratio = moved ? flops / moved : 0;
  return (
    <g>
      <Txt x={x} y={y} size={7} color={COL.muted}>
        {k("flops")}
      </Txt>
      <Txt x={x} y={y + 10} size={9} weight={600} color={COL.blue} kind="mono">
        {flops}
      </Txt>
      <Txt x={x} y={y + 24} size={7} color={COL.muted}>
        {k("moved")}
      </Txt>
      <Txt x={x} y={y + 34} size={9} weight={600} color={COL.text} kind="mono">
        {moved}
      </Txt>
      <Txt x={x} y={y + 48} size={7} color={COL.muted}>
        {k("ratio")}
      </Txt>
      <Txt x={x} y={y + 58} size={9} weight={600} color={COL.cur} kind="mono">
        {ratio.toFixed(2)}
      </Txt>
    </g>
  );
}

function Levels({ step, t, f, k, T }: DrawProps) {
  const title = (lvl: number, routine: string) => (
    <>
      <Txt x={8} y={12} size={8.5} weight={600}>
        {T(`blas.levels.l${lvl}` as PageKey)}
      </Txt>
      <Txt x={W - 8} y={12} size={8} anchor="end" kind="mono" color={COL.blue}>
        {routine}
      </Txt>
    </>
  );
  if (step === 0) {
    const n = 8;
    const i = Math.min(t, n - 1);
    const done = Math.min(t + 1, n);
    const s = 18;
    const x0 = 26;
    const row = (y: number, name: string, write?: boolean) => (
      <g>
        <Txt x={x0 - 6} y={y + 12} size={9} anchor="end" kind="math">
          {name}
        </Txt>
        {Array.from({ length: n }, (_, j) => (
          <Cell
            key={j}
            x={x0 + j * (s + 2)}
            y={y}
            s={s}
            fill={j === i && t < n ? COL.curLight : j < done ? (write ? COL.done : COL.blueLight) : COL.idle}
            stroke={j === i && t < n ? COL.cur : COL.idleLine}
            label={`${name}${j}`}
            size={6}
          />
        ))}
      </g>
    );
    return (
      <g>
        {title(1, "axpy:  y ← αx + y")}
        {row(34, "x")}
        {row(62, "y", true)}
        <Txt x={x0} y={100} size={7} color={COL.muted}>
          {`y${i} ← α·x${i} + y${i}`}
        </Txt>
        <Counters x={200} y={36} flops={2 * done} moved={3 * done} k={k} />
      </g>
    );
  }
  if (step === 1) {
    const n = 6;
    const p = Math.min(Math.floor(f * n * n), n * n - 1);
    const ci = Math.floor(p / n);
    const cj = p % n;
    const s = 14;
    const ax = 20;
    const ay = 26;
    return (
      <g>
        {title(2, "gemv:  y ← Ax + y")}
        <Txt x={ax + (n * (s + 1)) / 2} y={ay + n * (s + 1) + 10} size={9} anchor="middle" kind="math">
          A
        </Txt>
        {Array.from({ length: n * n }, (_, q) => {
          const r = Math.floor(q / n);
          const c = q % n;
          return <Cell key={q} x={ax + c * (s + 1)} y={ay + r * (s + 1)} s={s} fill={q === p ? COL.curLight : q < p ? COL.done : COL.idle} stroke={q === p ? COL.cur : COL.idleLine} />;
        })}
        {Array.from({ length: n }, (_, j) => (
          <Cell key={`x${j}`} x={120} y={ay + j * (s + 1)} s={s} fill={j === cj ? COL.curLight : COL.blueLight} stroke={j === cj ? COL.cur : COL.idleLine} />
        ))}
        <Txt x={127} y={ay + n * (s + 1) + 10} size={9} anchor="middle" kind="math">
          x
        </Txt>
        {Array.from({ length: n }, (_, j) => (
          <Cell key={`y${j}`} x={150} y={ay + j * (s + 1)} s={s} fill={j === ci ? COL.curLight : j < ci ? COL.done : COL.idle} stroke={j === ci ? COL.cur : COL.idleLine} />
        ))}
        <Txt x={157} y={ay + n * (s + 1) + 10} size={9} anchor="middle" kind="math">
          y
        </Txt>
        <Counters x={200} y={36} flops={2 * (p + 1)} moved={p + 1} k={k} />
      </g>
    );
  }
  if (step === 2) {
    const n = 4;
    const e = Math.min(t, n * n - 1);
    const ci = Math.floor(e / n);
    const cj = e % n;
    const s = 13;
    const doneE = e + 1;
    const usesA = (r: number) => (r < ci ? n : r === ci ? cj + 1 : 0);
    const usesB = (c: number) => ci + (c <= cj ? 1 : 0);
    const mat = (x0: number, name: string, cell: (r: number, c: number) => ReactNode) => (
      <g>
        <Txt x={x0 + (n * (s + 1)) / 2} y={30} size={9} anchor="middle" kind="math">
          {name}
        </Txt>
        {Array.from({ length: n * n }, (_, q) => cell(Math.floor(q / n), q % n))}
      </g>
    );
    const y0 = 36;
    return (
      <g>
        {title(3, "gemm:  C ← AB + C")}
        {mat(10, "A", (r, c) => (
          <Cell key={`a${r}${c}`} x={10 + c * (s + 1)} y={y0 + r * (s + 1)} s={s} fill={blueScale((usesA(r) / n) * 0.7)} stroke={r === ci ? COL.cur : COL.idleLine} sw={r === ci ? 1.3 : 0.75} label={usesA(r)} size={6} labelColor={usesA(r) > 2 ? "#fff" : COL.text} />
        ))}
        <Txt x={73} y={y0 + 32} size={9} anchor="middle" kind="math">
          ×
        </Txt>
        {mat(80, "B", (r, c) => (
          <Cell key={`b${r}${c}`} x={80 + c * (s + 1)} y={y0 + r * (s + 1)} s={s} fill={blueScale((usesB(c) / n) * 0.7)} stroke={c === cj ? COL.cur : COL.idleLine} sw={c === cj ? 1.3 : 0.75} label={usesB(c)} size={6} labelColor={usesB(c) > 2 ? "#fff" : COL.text} />
        ))}
        <Txt x={143} y={y0 + 32} size={9} anchor="middle" kind="math">
          =
        </Txt>
        {mat(150, "C", (r, c) => {
          const q = r * n + c;
          return <Cell key={`c${r}${c}`} x={150 + c * (s + 1)} y={y0 + r * (s + 1)} s={s} fill={q === e ? COL.curLight : q < e ? COL.done : COL.idle} stroke={q === e ? COL.cur : COL.idleLine} />;
        })}
        <Txt x={10} y={y0 + 4 * (s + 1) + 14} size={6.5} color={COL.muted}>
          {k("reuse")}
        </Txt>
        <Counters x={222} y={36} flops={2 * n * doneE} moved={levelCost(3, n).moved} k={k} />
      </g>
    );
  }
  // summary: FLOPs per value at n = 1000, log axis
  const n = 1000;
  const X = logMap(0.1, 1000, 74, 266);
  const vals = ([1, 2, 3] as const).map((l) => levelCost(l, n).flops / levelCost(l, n).moved);
  return (
    <g>
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={18} y2={104} stroke="#e5e7eb" />
          <Txt x={X(v)} y={114} size={7} anchor="middle" color={COL.muted}>
            {String(v)}
          </Txt>
        </g>
      ))}
      {vals.map((v, i) => {
        const y = 26 + i * 26;
        const end = X(0.1) + (X(v) - X(0.1)) * Math.min(1, f * 1.5);
        return (
          <g key={i}>
            <Txt x={70} y={y + 11} size={7} anchor="end">
              {T(`blas.levels.l${i + 1}` as PageKey)}
            </Txt>
            <rect x={X(0.1)} y={y} width={Math.max(0, end - X(0.1))} height={16} rx={2} fill={i === 2 ? COL.blue : "#9ca3af"} />
            <Txt x={end + 3} y={y + 11} size={7.5} kind="mono">
              {v < 1 ? v.toFixed(2) : String(v)}
            </Txt>
          </g>
        );
      })}
      <Txt x={170} y={128} size={7} anchor="middle" color={COL.text}>
        {k("axis")}
      </Txt>
    </g>
  );
}

// --- 1. GEMM tiling, packing, micro-kernel, threads ------------------------------------------------------

const CORE_COL = [COL.blue, COL.cur, COL.green, COL.violet];

function Gemm({ step, t, f, k }: DrawProps) {
  const n = 8;
  const b = 4;
  const s = 6;
  const A = { x: 12, y: 70 };
  const B = { x: 72, y: 12 };
  const C = { x: 72, y: 70 };
  // tile of C in focus: cycles in step 0, tile (0, 0) afterwards
  const tile = step === 0 ? Math.min(Math.floor(t / 2), 3) : 0;
  const ti = Math.floor(tile / 2);
  const tj = tile % 2;
  const kk = step === 2 ? Math.min(Math.floor(t / 2), n - 1) : -1;
  const aFill = (r: number, c: number) => {
    if (step === 3) return blueScale(0.15);
    if (step === 2) return Math.floor(r / b) === 0 ? (c === kk ? COL.curLight : COL.blueLight) : COL.idle;
    return Math.floor(r / b) === ti ? COL.curLight : COL.idle;
  };
  const bFill = (r: number, c: number) => {
    if (step === 3) return blueScale(0.15);
    if (step === 2) return Math.floor(c / b) === 0 ? (r === kk ? COL.curLight : COL.blueLight) : COL.idle;
    return Math.floor(c / b) === tj ? COL.curLight : COL.idle;
  };
  const cFill = (r: number, c: number) => {
    const tr = Math.floor(r / b);
    const tc = Math.floor(c / b);
    if (step === 3) {
      const core = tr * 2 + tc;
      return core < 4 ? blueScale(0.1 + 0.6 * f) : COL.idle;
    }
    if (step === 2) return tr === 0 && tc === 0 ? blueScale(0.1 + (0.6 * (kk + 1)) / n) : COL.idle;
    return tr === ti && tc === tj ? COL.cur : COL.idle;
  };
  const mat = (o: { x: number; y: number }, name: string, fillOf: (r: number, c: number) => string, tag: string) => (
    <g>
      {Array.from({ length: n * n }, (_, q) => {
        const r = Math.floor(q / n);
        const c = q % n;
        return <rect key={`${tag}${q}`} x={o.x + c * s} y={o.y + r * s} width={s} height={s} fill={fillOf(r, c)} stroke="#fff" strokeWidth={0.5} />;
      })}
      <line x1={o.x + b * s} x2={o.x + b * s} y1={o.y} y2={o.y + n * s} stroke="#475569" strokeWidth={0.8} />
      <line x1={o.x} x2={o.x + n * s} y1={o.y + b * s} y2={o.y + b * s} stroke="#475569" strokeWidth={0.8} />
      <rect x={o.x} y={o.y} width={n * s} height={n * s} fill="none" stroke="#475569" strokeWidth={0.8} />
      <Txt x={o.x - 4} y={o.y + 10} size={9} anchor="end" kind="math">
        {name}
      </Txt>
    </g>
  );
  const px = 140;
  return (
    <g>
      {mat(A, "A", aFill, "a")}
      {mat(B, "B", bFill, "b")}
      {mat(C, "C", cFill, "c")}
      {step === 3 &&
        [0, 1, 2, 3].map((q) => (
          <rect key={q} x={C.x + (q % 2) * b * s + 0.8} y={C.y + Math.floor(q / 2) * b * s + 0.8} width={b * s - 1.6} height={b * s - 1.6} fill="none" stroke={CORE_COL[q]} strokeWidth={1.6} />
        ))}
      {step <= 1 && (
        <rect x={C.x + tj * b * s} y={C.y + ti * b * s} width={b * s} height={b * s} fill="none" stroke={COL.cur} strokeWidth={1.6} />
      )}
      {/* right panel */}
      {step === 1 && (
        <g>
          <Box x={px} y={14} w={134} h={60} fill={COL.greenLight} stroke={COL.green} />
          <Txt x={px + 4} y={24} size={7} color={COL.green} weight={600}>
            {k("cache")}
          </Txt>
          {Array.from({ length: 32 }, (_, q) => {
            const shown = q < Math.ceil(f * 32);
            return (
              <g key={q}>
                <rect x={px + 3 + q * 4} y={32} width={3.4} height={12} fill={shown ? COL.curLight : "#fff"} stroke={COL.idleLine} strokeWidth={0.3} />
                <rect x={px + 3 + q * 4} y={52} width={3.4} height={12} fill={shown ? COL.curLight : "#fff"} stroke={COL.idleLine} strokeWidth={0.3} />
              </g>
            );
          })}
          <Txt x={px + 3} y={50} size={5.5} color={COL.muted} kind="mono">
            A[0:4, :]
          </Txt>
          <Txt x={px + 3} y={70} size={5.5} color={COL.muted} kind="mono">
            B[:, 0:4]
          </Txt>
          <Arrow x1={A.x + 48} y1={A.y + 6} x2={px - 2} y2={38} dash="2 2" />
          <Arrow x1={B.x + 50} y1={B.y + 30} x2={px - 2} y2={58} dash="2 2" />
        </g>
      )}
      {step === 2 && (
        <g>
          <Box x={px + 20} y={14} w={96} h={70} fill={COL.violetLight} stroke={COL.violet} />
          <Txt x={px + 68} y={24} size={7} anchor="middle" color={COL.violet} weight={600}>
            {k("regs")}
          </Txt>
          {Array.from({ length: 16 }, (_, q) => (
            <Cell key={q} x={px + 44 + (q % 4) * 12} y={30 + Math.floor(q / 4) * 12} s={11} fill={blueScale(0.1 + (0.6 * (kk + 1)) / n)} stroke={COL.violet} sw={0.5} />
          ))}
          <Txt x={px + 68} y={98} size={7.5} anchor="middle" kind="mono" color={COL.text}>
            {`k = ${kk}:  C += a[:, k] · b[k, :]`}
          </Txt>
          <Txt x={px + 68} y={110} size={7} anchor="middle" kind="mono" color={COL.cur}>
            {`16 FMA × ${kk + 1}`}
          </Txt>
        </g>
      )}
      {step === 3 &&
        [0, 1, 2, 3].map((q) => (
          <g key={q}>
            <Txt x={px + 8} y={26 + q * 18} size={7} color={CORE_COL[q]} weight={600}>
              {fill(k("core"), { i: q })}
            </Txt>
            <rect x={px + 52} y={19 + q * 18} width={80} height={9} rx={2} fill={COL.idle} />
            <rect x={px + 52} y={19 + q * 18} width={80 * f} height={9} rx={2} fill={CORE_COL[q]} />
          </g>
        ))}
      {step === 0 && (
        <g>
          {[0, 1, 2, 3].map((q) => (
            <Box key={q} x={px + 30 + (q % 2) * 40} y={20 + Math.floor(q / 2) * 34} w={36} h={30} fill={q === tile ? COL.curLight : COL.idle} stroke={q === tile ? COL.cur : COL.idleLine} label={`C${Math.floor(q / 2)}${q % 2}`} kind="mono" size={8} />
          ))}
        </g>
      )}
      <Txt x={12} y={140} size={7} color={COL.muted}>
        {k("ramNaive")}
      </Txt>
      <Txt x={W - 8} y={140} size={7.5} anchor="end" kind="mono">
        {`2n³ = ${gemmLoads(n)}`}
      </Txt>
      <Txt x={12} y={152} size={7} color={COL.text} weight={600}>
        {k("ramTiled")}
      </Txt>
      <Txt x={W - 8} y={152} size={7.5} anchor="end" kind="mono" color={COL.green} weight={600}>
        {`2n³/4 = ${gemmLoads(n, b)}`}
      </Txt>
    </g>
  );
}

// --- 2. how PyTorch and NumPy reach a BLAS ---------------------------------------------------------------

const LIBS = ["oneMKL", "OpenBLAS", "Accelerate", "cuBLAS / cuBLASLt", "rocBLAS / hipBLASLt"];
/** path per step: input row, front end (0 torch, 1 numpy), library edges [lib, solid?] */
const ROUTES: { input: number; fe: number; libs: [number, boolean][] }[] = [
  { input: 0, fe: 0, libs: [[0, true], [1, false]] },
  { input: 1, fe: 0, libs: [[3, true]] },
  { input: 2, fe: 0, libs: [[4, true]] },
  { input: 3, fe: 1, libs: [[1, true], [0, false], [2, false]] },
];

function Dispatch({ step, f, k }: DrawProps) {
  const inY = (i: number) => 14 + i * 28;
  const feY = [42, 98];
  const libY = (i: number) => 8 + i * 25;
  const IN = { x: 4, w: 82, h: 20 };
  const FE = { x: 104, w: 70, h: 20 };
  const LB = { x: 192, w: 84, h: 18 };
  const inputs = ["cpu", "cuda", "rocm", "numpy"];
  const fes = ["torch.matmul", "numpy.matmul"];
  const r = ROUTES[step];
  const curve = (x1: number, y1: number, x2: number, y2: number) => `M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`;
  const allIn = inputs.map((_, i) => ({ i, fe: i < 3 ? 0 : 1 }));
  // [front end, library, usual build (solid) or other builds (dashed)]
  const allLib: [number, number, boolean][] = [[0, 0, true], [0, 1, false], [0, 3, true], [0, 4, true], [1, 1, true], [1, 0, false], [1, 2, false]];
  const active = (fe: number, lib: number) => r.fe === fe && r.libs.find(([l]) => l === lib);
  // the moving dot: input -> front end -> first (solid) library
  const p1: [number, number] = [IN.x + IN.w, inY(r.input) + IN.h / 2];
  const p2: [number, number] = [FE.x, feY[r.fe] + FE.h / 2];
  const p3: [number, number] = [FE.x + FE.w, feY[r.fe] + FE.h / 2];
  const p4: [number, number] = [LB.x, libY(r.libs[0][0]) + LB.h / 2];
  const dot = along([p1, p2, p3, p4], f);
  return (
    <g>
      {allIn.map(({ i, fe }) => (
        <path key={`e${i}`} d={curve(IN.x + IN.w, inY(i) + IN.h / 2, FE.x, feY[fe] + FE.h / 2)} fill="none" stroke={i === r.input ? COL.cur : COL.idleLine} strokeWidth={i === r.input ? 1.8 : 0.9} />
      ))}
      {allLib.map(([fe, lib, solid]) => {
        const a = active(fe, lib);
        return (
          <path
            key={`l${fe}${lib}`}
            d={curve(FE.x + FE.w, feY[fe] + FE.h / 2, LB.x, libY(lib) + LB.h / 2)}
            fill="none"
            stroke={a ? COL.cur : COL.idleLine}
            strokeWidth={a ? 1.8 : 0.9}
            strokeDasharray={solid ? undefined : "3 2"}
          />
        );
      })}
      <circle cx={dot[0]} cy={dot[1]} r={3} fill={COL.cur} stroke="#fff" strokeWidth={1} />
      {inputs.map((name, i) => (
        <Box key={name} x={IN.x} y={inY(i)} w={IN.w} h={IN.h} label={k(name)} size={6.5} fill={i === r.input ? COL.curLight : "#fff"} stroke={i === r.input ? COL.cur : "#94a3b8"} />
      ))}
      {fes.map((name, i) => (
        <Box key={name} x={FE.x} y={feY[i]} w={FE.w} h={FE.h} label={name} kind="mono" size={6.5} fill={i === r.fe ? COL.blueLight : "#fff"} stroke={i === r.fe ? COL.blue : "#94a3b8"} />
      ))}
      {LIBS.map((name, i) => {
        const a = r.libs.find(([l]) => l === i);
        return <Box key={name} x={LB.x} y={libY(i)} w={LB.w} h={LB.h} label={name} kind="mono" size={6} fill={a ? (a[1] ? COL.greenLight : "#fff") : "#fff"} stroke={a ? COL.green : "#94a3b8"} dash={a && !a[1] ? "3 2" : undefined} />;
      })}
      <line x1={4} x2={20} y1={136} y2={136} stroke={COL.muted} strokeDasharray="3 2" />
      <Txt x={24} y={138.5} size={6.5} color={COL.muted}>
        {k("built")}
      </Txt>
    </g>
  );
}

// --- 3. threads ------------------------------------------------------------------------------------------

function Threads({ step, f, k }: DrawProps) {
  const cores = 6;
  const tiles = 12;
  const threads = [1, 6, 12][step];
  const X = (v: number) => 46 + (v / 12) * 222;
  const laneY = (c: number) => 12 + c * 15;
  const now = f * 12;
  const finish = poolTime(tiles, threads, cores);
  // segments per core: [start, end, colour]
  const segs: [number, number, string][][] = Array.from({ length: cores }, () => []);
  if (threads === 1) for (let q = 0; q < tiles; q++) segs[0].push([q, q + 0.94, COL.blue]);
  else if (threads === cores) for (let c = 0; c < cores; c++) for (let q = 0; q < 2; q++) segs[c].push([q, q + 0.94, COL.blue]);
  else
    for (let c = 0; c < cores; c++) {
      const slices = 8;
      const gap = (finish - 2) / (slices - 1);
      for (let q = 0; q < slices; q++) {
        const st = q * (0.25 + gap);
        segs[c].push([st, st + 0.25, q % 2 ? COL.violet : COL.blue]);
      }
    }
  return (
    <g>
      {[0, 2, 4, 6, 8, 10, 12].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={8} y2={laneY(cores)} stroke="#e5e7eb" />
          <Txt x={X(v)} y={laneY(cores) + 8} size={6.5} anchor="middle" color={COL.muted}>
            {String(v)}
          </Txt>
        </g>
      ))}
      {segs.map((row, c) => (
        <g key={c}>
          <Txt x={42} y={laneY(c) + 8} size={6.5} anchor="end">
            {fill(k("core"), { i: c })}
          </Txt>
          <rect x={X(0)} y={laneY(c)} width={X(12) - X(0)} height={11} fill={COL.idle} />
          {row.map(([a, b2, col], q) =>
            a < now ? <rect key={q} x={X(a)} y={laneY(c) + 1} width={Math.max(0, X(Math.min(b2, now)) - X(a))} height={9} rx={1} fill={col} /> : null,
          )}
        </g>
      ))}
      <line x1={X(now)} x2={X(now)} y1={8} y2={laneY(cores)} stroke={COL.cur} strokeWidth={1.2} />
      {now >= finish && (
        <g>
          <line x1={X(finish)} x2={X(finish)} y1={8} y2={laneY(cores)} stroke={COL.green} strokeWidth={1.2} strokeDasharray="3 2" />
          <Txt x={Math.min(X(finish) + 3, W - 60)} y={laneY(cores) + 20} size={7.5} color={COL.green} weight={600}>
            {fill(k("time"), { t: String(finish) })}
          </Txt>
        </g>
      )}
      <Txt x={157} y={laneY(cores) + 32} size={6.5} anchor="middle" color={COL.text}>
        {k("axis")}
      </Txt>
    </g>
  );
}

// --- 4. summation order ----------------------------------------------------------------------------------

const ORDERS: SumOrder[] = ["seq", "pair", "lanes"];
const SUMS = ORDERS.map((o) => sumOrder(o));
const REF = SUM_INPUTS.map(Math.fround).reduce((a, b) => a + b, 0); // float64 sum of the same float32 inputs

function Order({ step, t, k }: DrawProps) {
  const ix = (i: number) => 26 + i * 33;
  const adds = SUMS[step];
  const shown = Math.min(t + 1, adds.length);
  // positions of the partial sums (node ids 8..14)
  const pos = new Map<number, [number, number]>();
  SUM_INPUTS.forEach((_, i) => pos.set(i, [ix(i), 22]));
  let level = new Map<number, number>();
  adds.forEach((a: Add, j) => {
    const id = 8 + j;
    const [lx] = pos.get(a.left)!;
    const [rx] = pos.get(a.right)!;
    let p: [number, number];
    if (step === 0) p = [rx, 46 + j * 13];
    else if (step === 1) {
      const lv = Math.max(level.get(a.left) ?? 0, level.get(a.right) ?? 0) + 1;
      level.set(id, lv);
      p = [(lx + rx) / 2, 26 + lv * 26];
    } else p = j < 6 ? [rx, 46 + Math.floor(j / 2) * 16] : [(lx + rx) / 2, 104];
    pos.set(id, p);
  });
  const bw = 28;
  const last = adds[adds.length - 1].value;
  return (
    <g>
      <Txt x={8} y={10} size={7} color={COL.muted}>
        {k("inputs")}
      </Txt>
      <Txt x={W - 8} y={10} size={7.5} anchor="end" weight={600} color={COL.blue}>
        {k(ORDERS[step])}
      </Txt>
      {adds.slice(0, shown).map((a, j) => {
        const [x, y] = pos.get(8 + j)!;
        return (
          <g key={`e${j}`}>
            {[a.left, a.right].map((c) => {
              const [cx, cy] = pos.get(c)!;
              return <line key={c} x1={cx} y1={cy + 6} x2={x} y2={y - 6} stroke={j === shown - 1 ? COL.cur : COL.doneLine} strokeWidth={j === shown - 1 ? 1.4 : 0.8} />;
            })}
          </g>
        );
      })}
      {SUM_INPUTS.map((v, i) => (
        <Box key={i} x={ix(i) - 14} y={16} w={28} h={12} label={String(v)} size={6.5} kind="mono" fill={COL.blueLight} stroke={COL.doneLine} />
      ))}
      {adds.slice(0, shown).map((a, j) => {
        const [x, y] = pos.get(8 + j)!;
        const final = j === adds.length - 1;
        const w = final ? 46 : bw;
        return (
          <Box
            key={`n${j}`}
            x={Math.min(x - w / 2, W - 2 - w)}
            y={y - 6}
            w={w}
            h={12}
            label={final ? fmt9(a.value) : String(Number(a.value.toPrecision(4)))}
            size={6.2}
            kind="mono"
            fill={j === shown - 1 ? COL.curLight : final ? COL.greenLight : "#fff"}
            stroke={j === shown - 1 ? COL.cur : final ? COL.green : COL.idleLine}
            weight={final ? 600 : undefined}
          />
        );
      })}
      {/* results so far */}
      {ORDERS.map((o, i) => {
        const done = i < step || (i === step && shown === adds.length);
        const v = SUMS[i][SUMS[i].length - 1].value;
        const x = i % 2 ? 146 : 8;
        const y = 146 + Math.floor(i / 2) * 12;
        return (
          <g key={o} opacity={done ? 1 : 0.25}>
            <Txt x={x} y={y} size={7} color={COL.muted}>
              {`${k(o)}:`}
            </Txt>
            <Txt x={x + 126} y={y} size={7.5} anchor="end" kind="mono" weight={600} color={v === last && i === step ? COL.cur : COL.text}>
              {done ? fmt9(v) : "…"}
            </Txt>
          </g>
        );
      })}
      <Txt x={146} y={158} size={7} color={COL.muted}>
        {`${k("ref")}:`}
      </Txt>
      <Txt x={272} y={158} size={7.5} anchor="end" kind="mono" color={COL.green}>
        {REF.toFixed(8)}
      </Txt>
    </g>
  );
}

// --- 4. atomics vs a fixed order -------------------------------------------------------------------------

function Atomics({ step, t, k }: DrawProps) {
  const det = step === 1;
  const runs = det ? [ATOMIC_RUNS.fixed, ATOMIC_RUNS.fixed] : ATOMIC_RUNS.atomic;
  const run = t < 6 ? 0 : 1;
  const j = Math.min(t % 6, 4); // 0: empty, 1..4 blocks arrived
  const order = runs[run];
  const sums = accumulate(order);
  const bx = (i: number) => 14 + i * 66;
  const accX = 90;
  const accY = 74;
  const sender = j >= 1 && j <= 4 ? order[j - 1] : -1;
  const results = runs.map((o, r) => (r < run || (r === run && j === 4) ? accumulate(o)[3] : null));
  const differ = results[0] != null && results[1] != null && fmt9(results[0]) !== fmt9(results[1]);
  return (
    <g>
      {ATOMIC_PARTS.map((v, i) => {
        const pos = order.indexOf(i);
        const sent = pos < j;
        return (
          <g key={i}>
            <Box x={bx(i)} y={14} w={56} h={26} fill={i === sender ? COL.curLight : sent ? COL.idle : COL.blueLight} stroke={i === sender ? COL.cur : COL.doneLine} />
            <Txt x={bx(i) + 28} y={24} size={6.5} anchor="middle" color={COL.muted}>
              {fill(k("block"), { i })}
            </Txt>
            <Txt x={bx(i) + 28} y={35} size={7.5} anchor="middle" kind="mono">
              {String(v)}
            </Txt>
            {sent && (
              <Txt x={bx(i) + 52} y={50} size={7} anchor="end" kind="mono" color={COL.cur}>
                {`#${pos + 1}`}
              </Txt>
            )}
          </g>
        );
      })}
      {sender >= 0 && <Arrow x1={bx(sender) + 28} y1={41} x2={accX + 50} y2={accY - 2} w={1.6} />}
      <Box x={accX} y={accY} w={100} h={26} fill={COL.violetLight} stroke={COL.violet} />
      <Txt x={accX + 50} y={accY + 10} size={6.5} anchor="middle" color={COL.violet}>
        {k("acc")}
      </Txt>
      <Txt x={accX + 50} y={accY + 21} size={8} anchor="middle" kind="mono" weight={600}>
        {j === 0 ? "0" : fmt9(sums[j - 1])}
      </Txt>
      {[0, 1].map((r) => (
        <g key={r} opacity={results[r] == null && r !== run ? 0.3 : 1}>
          <Txt x={14} y={120 + r * 13} size={7.5} weight={r === run ? 600 : undefined}>
            {fill(k("run"), { r: r + 1 })}
          </Txt>
          <Txt x={110} y={120 + r * 13} size={7} kind="mono" color={COL.muted}>
            {runs[r].join(" → ")}
          </Txt>
          <Txt x={W - 10} y={120 + r * 13} size={8} anchor="end" kind="mono" weight={600} color={results[r] == null ? COL.muted : differ ? COL.red : det ? COL.green : COL.text}>
            {results[r] == null ? "…" : fmt9(results[r]!)}
          </Txt>
        </g>
      ))}
    </g>
  );
}

// --- 5. fixed costs: the illustrative throughput model ---------------------------------------------------

function Overhead({ step, f, k }: DrawProps) {
  const x0 = 40;
  const x1 = 268;
  const y0 = 18;
  const y1 = 126;
  const X = logMap(16, 4096, x0, x1);
  const Y = logMap(0.1, 30000, y1, y0);
  const ns = Array.from({ length: 41 }, (_, i) => 16 * 2 ** (i / 5));
  const path = (t0: number, peak: number, upto: number) =>
    ns
      .filter((n) => n <= upto)
      .map((n, i) => `${i ? "L" : "M"} ${X(n).toFixed(1)} ${Y(modelGflops(n, t0, peak)).toFixed(1)}`)
      .join(" ");
  const nNow = 16 * 2 ** (8 * f);
  const { cpu, gpu } = OVERHEAD;
  const cross = crossover();
  return (
    <g>
      <Txt x={x0 - 30} y={10} size={7} color={COL.text}>
        {k("y")}
      </Txt>
      {[0.1, 1, 10, 100, 1000, 10000].map((v) => (
        <g key={v}>
          <line x1={x0} x2={x1} y1={Y(v)} y2={Y(v)} stroke="#e5e7eb" />
          <Txt x={x0 - 3} y={Y(v) + 2.5} size={6.5} anchor="end" color={COL.muted}>
            {v >= 1000 ? `${v / 1000}k` : String(v)}
          </Txt>
        </g>
      ))}
      {[16, 64, 256, 1024, 4096].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={y0} y2={y1} stroke="#f3f4f6" />
          <Txt x={X(v)} y={y1 + 9} size={6.5} anchor="middle" color={COL.muted}>
            {String(v)}
          </Txt>
        </g>
      ))}
      <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke={COL.idleLine} strokeWidth={0.6} />
      <line x1={x0} x2={x1} y1={Y(cpu.peak)} y2={Y(cpu.peak)} stroke={COL.blue} strokeDasharray="4 3" strokeWidth={1} />
      <Txt x={x0 + 3} y={Y(cpu.peak) - 3} size={6.5} color={COL.blue}>
        {k("ideal")}
      </Txt>
      {step >= 1 && (
        <g>
          <path d={path(cpu.t0, cpu.peak, step === 1 ? nNow : 4096)} fill="none" stroke={COL.blue} strokeWidth={1.8} />
          {step === 1 && <circle cx={X(nNow)} cy={Y(modelGflops(nNow, cpu.t0, cpu.peak))} r={3} fill={COL.cur} />}
          <Txt x={X(1024)} y={Y(cpu.peak) + 10} size={7} color={COL.blue} weight={600}>
            {k("cpu")}
          </Txt>
        </g>
      )}
      {step === 2 && (
        <g>
          <line x1={x0} x2={x1} y1={Y(gpu.peak)} y2={Y(gpu.peak)} stroke={COL.violet} strokeDasharray="4 3" strokeWidth={1} />
          <path d={path(gpu.t0, gpu.peak, nNow)} fill="none" stroke={COL.violet} strokeWidth={1.8} />
          <circle cx={X(nNow)} cy={Y(modelGflops(nNow, gpu.t0, gpu.peak))} r={3} fill={COL.cur} />
          <Txt x={X(1024)} y={Y(gpu.peak) + 10} size={7} color={COL.violet} weight={600}>
            {k("gpu")}
          </Txt>
          {nNow >= cross && (
            <g>
              <line x1={X(cross)} x2={X(cross)} y1={y0} y2={y1} stroke={COL.cur} strokeDasharray="2 2" />
              <Txt x={X(cross) + 3} y={y1 - 4} size={7} kind="mono" color={COL.cur}>
                {`n ≈ ${cross}`}
              </Txt>
            </g>
          )}
        </g>
      )}
      <Txt x={(x0 + x1) / 2} y={y1 + 22} size={7} anchor="middle" color={COL.text}>
        {k("x")}
      </Txt>
    </g>
  );
}

// --- 6. timing a GPU call --------------------------------------------------------------------------------

function Timing({ step, f, k }: DrawProps) {
  const X = (v: number) => 40 + (v / 10) * 228;
  const cpuY = 34;
  const gpuY = 62;
  const now = f * 10;
  const bar = (a: number, b: number, y: number, col: string, label?: string, light = false) =>
    a < now ? (
      <g>
        <rect x={X(a)} y={y} width={Math.max(0, X(Math.min(b, now)) - X(a))} height={14} rx={2} fill={col} opacity={light ? 0.45 : 1} />
        {label && now >= (a + b) / 2 && (
          <Txt x={(X(a) + X(b)) / 2} y={X(b) - X(a) < 30 ? y - 3 : y + 9.5} size={6.5} anchor="middle" color={X(b) - X(a) < 30 ? COL.text : "#fff"}>
            {label}
          </Txt>
        )}
      </g>
    ) : null;
  const bracket = (a: number, b: number, y: number, col: string, label: string) =>
    now >= b ? (
      <g>
        <line x1={X(a)} x2={X(b)} y1={y} y2={y} stroke={col} strokeWidth={1.4} />
        <line x1={X(a)} x2={X(a)} y1={y - 4} y2={y + 4} stroke={col} strokeWidth={1.4} />
        <line x1={X(b)} x2={X(b)} y1={y - 4} y2={y + 4} stroke={col} strokeWidth={1.4} />
        <Txt x={(X(a) + X(b)) / 2} y={y + 11} size={6.5} anchor="middle" color={col} weight={600}>
          {label}
        </Txt>
      </g>
    ) : null;
  const code = ["t0 = time(); C = A @ B; t1 = time()", "t0 = time(); C = A @ B; torch.cuda.synchronize(); t1 = time()", ""][step];
  return (
    <g>
      <Txt x={8} y={12} size={step === 1 ? 6.3 : 7} kind="mono" color={COL.blue}>
        {code}
      </Txt>
      <Txt x={36} y={cpuY + 10} size={7} anchor="end" weight={600}>
        {k("cpu")}
      </Txt>
      <Txt x={36} y={gpuY + 10} size={7} anchor="end" weight={600}>
        {k("gpu")}
      </Txt>
      <rect x={X(0)} y={cpuY} width={X(10) - X(0)} height={14} fill={COL.idle} />
      <rect x={X(0)} y={gpuY} width={X(10) - X(0)} height={14} fill={COL.idle} />
      {step < 2 && (
        <g>
          {bar(0.5, 1.2, cpuY, COL.cur, k("launch"))}
          {bar(1.2, 8.2, gpuY, COL.violet, k("kernel"))}
          {step === 1 && bar(1.2, 8.2, cpuY, "#9ca3af", undefined, true)}
          {step === 0 ? bracket(0.5, 1.3, 98, COL.red, k("measured")) : bracket(0.5, 8.4, 98, COL.green, k("measured"))}
        </g>
      )}
      {step === 2 && (
        <g>
          {bar(0.2, 0.5, cpuY, COL.cur)}
          {bar(0.5, 3.4, gpuY, "#9ca3af", k("warm"))}
          {[
            [3.6, 5.6],
            [5.8, 7.7],
            [7.9, 9.9],
          ].map(([a, b], i) => (
            <g key={i}>
              {bar(a, a + 0.2, cpuY, COL.cur)}
              {bar(a + 0.2, b, gpuY, COL.violet, k("kernel"))}
              {bracket(a, b, 98, COL.green, `${(b - a).toFixed(1)}`)}
            </g>
          ))}
          {now >= 9.9 && (
            <Txt x={X(7.8)} y={124} size={7.5} anchor="middle" weight={600} color={COL.green}>
              {`${k("median")}: 2.0`}
            </Txt>
          )}
        </g>
      )}
      <line x1={X(now)} x2={X(now)} y1={cpuY - 4} y2={gpuY + 18} stroke={COL.cur} strokeWidth={1} />
      {[0, 2, 4, 6, 8, 10].map((v) => (
        <Txt key={v} x={X(v)} y={gpuY + 26} size={6.5} anchor="middle" color={COL.muted}>
          {String(v)}
        </Txt>
      ))}
      <Txt x={154} y={140} size={7} anchor="middle">
        {k("axis")}
      </Txt>
    </g>
  );
}

const PICTURES: Record<BlasCase, { draw: (p: DrawProps) => ReactNode; h: number }> = {
  levels: { draw: (p) => <Levels {...p} />, h: 134 },
  gemm: { draw: (p) => <Gemm {...p} />, h: 158 },
  dispatch: { draw: (p) => <Dispatch {...p} />, h: 144 },
  threads: { draw: (p) => <Threads {...p} />, h: 138 },
  order: { draw: (p) => <Order {...p} />, h: 164 },
  atomics: { draw: (p) => <Atomics {...p} />, h: 142 },
  overhead: { draw: (p) => <Overhead {...p} />, h: 154 },
  timing: { draw: (p) => <Timing {...p} />, h: 146 },
};

/** One animation of the BLAS tab. */
export default function BlasAnim({ id }: { id: BlasCase }) {
  const pic = PICTURES[id];
  return <StepAnim prefix="blas" id={id} plan={BLAS_PLAN[id]} h={pic.h} draw={pic.draw} />;
}
