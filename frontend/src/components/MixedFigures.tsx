/**
 * Static and lightly interactive figures of Coding Skills ▸ Mixed precision, redrawn after NVIDIA's
 * mixed-precision training guide and Micikevicius et al. 2018 (shapes schematic, see each note):
 * the activation-gradient histogram against FP16's range (with a loss-scale slider), D = A·B + C on a matrix unit,
 * the FP16-multiply / FP32-accumulate graph, and training loss with and without loss scaling.
 */
import { useState } from "react";
import { COL, Txt } from "./animKit";
import { RichText } from "./Tex";
import { usePageText, type PageKey } from "../content/pages";
import { gradHistogram, histogramSplit, lossCurves } from "../chipsAnim";
import { fmt, roundTo, runningSum } from "../chipsModel";

const GUIDE = "https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html";
const PAPER = "https://arxiv.org/abs/1710.03740";
const PTX = "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions";

const Src = ({ href, label }: { href: string; label: string }) => (
  <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-700 hover:underline">
    {label}
    <span aria-hidden> ↗</span>
  </a>
);

function Frame({ title, testid, note, links = [], children }: { title: string; testid: string; note: string; links?: [string, string][]; children: React.ReactNode }) {
  return (
    <figure className="mt-4 overflow-hidden rounded-lg border border-gray-200 bg-white" data-testid={testid}>
      <figcaption className="border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">{title}</figcaption>
      {children}
      <p className="border-t border-gray-100 px-3 py-1.5 text-[11px] leading-relaxed text-gray-500">
        <RichText text={note} />
        {links.map(([href, label]) => (
          <span key={href}>
            {" "}
            <Src href={href} label={label} />
          </span>
        ))}
      </p>
    </figure>
  );
}

// --- 1. activation gradients against FP16's range, and what a loss scale does ---------------------------------

const HW = 320;
const pct = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

export function GradientHistogram() {
  const T = usePageText();
  const k = (s: string) => T(s as PageKey);
  const [s, setS] = useState(0);
  const bins = gradHistogram();
  const split = histogramSplit(bins, s);
  const L = 30;
  const R = HW - 8;
  const top = 26;
  const bot = 136;
  const U0 = -66; // the zero bar sits at u = -65; bins run k = -64 … 16
  const U1 = 17;
  const xs = (u: number) => L + ((u - U0) / (U1 - U0)) * (R - L);
  const bw = xs(1) - xs(0);
  const ys = (p: number) => bot - ((Math.log2(Math.max(p, 1 / 512)) + 9) / 15) * (bot - top);
  const yTicks = [-9, -6, -3, 0, 3, 6];
  const xTicks = [-60, -48, -36, -24, -14, 0, 15];
  const line = (u: number, c: string) => <line x1={xs(u)} x2={xs(u)} y1={top - 10} y2={bot} stroke={c} strokeWidth={1} />;
  const span = (u0: number, u1: number, y: number, text: string, c: string) => (
    <g>
      <line x1={xs(u0) + 1} x2={xs(u1) - 1} y1={y} y2={y} stroke={c} strokeWidth={0.6} markerStart="url(#gh-a)" markerEnd="url(#gh-a)" />
      <Txt x={(xs(u0) + xs(u1)) / 2} y={y - 2} size={5.6} anchor="middle" color={c}>
        {text}
      </Txt>
    </g>
  );
  return (
    <Frame
      title={k("mixed.fig.hist.title")}
      testid="mixed-histogram"
      note={k("mixed.fig.hist.note")}
      links={[[PAPER, "Micikevicius et al. 2018"], [GUIDE, "NVIDIA"]]}
    >
      <svg viewBox={`0 0 ${HW} 162`} className="mx-auto block h-auto w-full max-w-[680px] p-1" role="img" aria-label={k("mixed.fig.hist.title")}>
        <defs>
          <marker id="gh-a" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M0,0 L6,3 L0,6 z" fill="context-stroke" />
          </marker>
        </defs>
        {yTicks.map((e) => (
          <g key={e}>
            <line x1={L} x2={R} y1={ys(2 ** e)} y2={ys(2 ** e)} stroke="#eef0f3" strokeWidth={0.5} />
            <Txt x={L - 2} y={ys(2 ** e) + 1.8} size={5} anchor="end" color={COL.muted}>
              {e >= 0 ? String(2 ** e) : `1/${2 ** -e}`}
            </Txt>
          </g>
        ))}
        {/* the exact zeros: unchanged by any scale */}
        <rect x={xs(-65)} y={ys(split.zero)} width={bw - 0.4} height={bot - ys(split.zero)} fill={COL.muted} opacity={0.55} />
        {bins.map((b) => {
          const u = b.k + s;
          const lost = u < -24;
          const sub = !lost && u < -14;
          const over = u > 15;
          const c = over ? COL.red : lost ? COL.redLight : sub ? COL.curLight : "#86b85a";
          const stroke = over ? COL.red : lost ? COL.red : sub ? COL.cur : "#4d7c2a";
          return (
            <rect
              key={b.k}
              x={xs(u)}
              y={ys(b.p)}
              width={bw - 0.4}
              height={bot - ys(b.p)}
              fill={c}
              stroke={stroke}
              strokeWidth={0.3}
              strokeDasharray={lost ? "1 1" : undefined}
              style={{ transition: "x 300ms ease" }}
            />
          );
        })}
        {line(-24, COL.red)}
        {line(-14, COL.blue)}
        {line(16, "#111827")}
        {span(-64, -24, top - 4, k("mixed.fig.hist.zero"), COL.red)}
        {span(-24, -14, top - 4, k("mixed.fig.hist.sub"), COL.blue)}
        {span(-24, 16, top - 13, k("mixed.fig.hist.range"), "#111827")}
        <line x1={L} x2={R} y1={bot} y2={bot} stroke={COL.idleLine} strokeWidth={0.6} />
        <Txt x={xs(-65) + bw / 2} y={bot + 7} size={5} anchor="middle" color={COL.muted}>
          0
        </Txt>
        {xTicks.map((t) => (
          <Txt key={t} x={xs(t) + (t === 15 ? bw : 0)} y={bot + 7} size={5} anchor="middle" color={COL.muted}>
            {t === 15 ? "16" : String(t)}
          </Txt>
        ))}
        <Txt x={(L + R) / 2} y={bot + 15} size={5.8} anchor="middle">
          {k("mixed.fig.hist.x")}
        </Txt>
        <text transform={`translate(7 ${(top + bot) / 2}) rotate(-90)`} fontSize={5.6} textAnchor="middle" fill={COL.text}>
          {k("mixed.fig.hist.y")}
        </text>
      </svg>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 pb-2 text-xs text-gray-700">
        <label className="flex items-center gap-2">
          <span className="font-medium">{k("mixed.fig.hist.scale")}</span>
          <input type="range" min={0} max={20} value={s} onChange={(e) => setS(Number(e.target.value))} className="w-36 accent-blue-600" aria-label={k("mixed.fig.hist.scale")} />
          <span className="font-mono tabular-nums">
            2<sup>{s}</sup> = {(2 ** s).toLocaleString()}
          </span>
        </label>
        <span className="tabular-nums" data-testid="mixed-histogram-lost">
          <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm border border-red-600 bg-red-100 align-middle" />
          {k("mixed.fig.hist.lost").replace("{p}", pct(split.lost))}
        </span>
        <span className="tabular-nums">
          <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm border border-amber-500 bg-amber-200 align-middle" />
          {k("mixed.fig.hist.subshare").replace("{p}", pct(split.sub))}
        </span>
        {split.over > 0 && (
          <span className="font-medium text-red-700 tabular-nums">{k("mixed.fig.hist.over").replace("{p}", pct(split.over))}</span>
        )}
      </div>
    </Frame>
  );
}

// --- 2. training loss with and without loss scaling ------------------------------------------------------------

export function LossCurves() {
  const T = usePageText();
  const k = (s: string) => T(s as PageKey);
  const curves = lossCurves();
  const L = 26;
  const R = HW - 8;
  const top = 8;
  const bot = 118;
  const xs = (t: number) => L + (t / 2000) * (R - L);
  const ys = (v: number) => bot - ((Math.min(v, 5) - 2.5) / 2.5) * (bot - top);
  const path = (pts: [number, number][]) => pts.map(([t, v], i) => `${i ? "L" : "M"}${xs(t).toFixed(1)},${ys(v).toFixed(1)}`).join("");
  const series = [
    { id: "s1", pts: curves.scale1, c: "#9ca3af", w: 0.8 },
    { id: "fp32", pts: curves.fp32, c: "#111827", w: 0.8 },
    { id: "s128", pts: curves.scale128, c: "#65a30d", w: 0.8 },
  ];
  return (
    <Frame title={k("mixed.fig.loss.title")} testid="mixed-loss" note={k("mixed.fig.loss.note")} links={[[GUIDE, "NVIDIA"]]}>
      <svg viewBox={`0 0 ${HW} 140`} className="mx-auto block h-auto w-full max-w-[680px] p-1" role="img" aria-label={k("mixed.fig.loss.title")}>
        {[2.5, 3, 3.5, 4, 4.5, 5].map((v) => (
          <g key={v}>
            <line x1={L} x2={R} y1={ys(v)} y2={ys(v)} stroke="#eef0f3" strokeWidth={0.5} />
            <Txt x={L - 2} y={ys(v) + 1.8} size={5} anchor="end" color={COL.muted}>
              {v.toFixed(1)}
            </Txt>
          </g>
        ))}
        {[0, 500, 1000, 1500, 2000].map((t) => (
          <Txt key={t} x={xs(t)} y={bot + 7} size={5} anchor="middle" color={COL.muted}>
            {t === 0 ? "0" : `${t}K`}
          </Txt>
        ))}
        <Txt x={(L + R) / 2} y={bot + 15} size={5.8} anchor="middle">
          {k("mixed.fig.loss.x")}
        </Txt>
        <text transform={`translate(6 ${(top + bot) / 2}) rotate(-90)`} fontSize={5.6} textAnchor="middle" fill={COL.text}>
          {k("mixed.fig.loss.y")}
        </text>
        {series.map((c) => (
          <path key={c.id} d={path(c.pts)} fill="none" stroke={c.c} strokeWidth={c.w} strokeLinejoin="round" />
        ))}
        {series.map((c, i) => (
          <g key={c.id}>
            <line x1={R - 112} x2={R - 100} y1={top + 8 + i * 9} y2={top + 8 + i * 9} stroke={c.c} strokeWidth={1.6} />
            <Txt x={R - 97} y={top + 10 + i * 9} size={5.6}>
              {k(`mixed.fig.loss.${c.id}`)}
            </Txt>
          </g>
        ))}
      </svg>
    </Frame>
  );
}

// --- 3. D = A·B + C on a matrix unit, and the formats each generation takes -----------------------------------

const MMA_ROWS: [string, string, string][] = [
  ["Volta (2017)", "FP16", "FP16, FP32"],
  ["Turing (2018)", "INT8, INT4", "INT32"],
  ["Ampere (2020)", "BF16, TF32", "FP32"],
  ["Hopper (2022)", "FP8 (E4M3, E5M2)", "FP16, FP32"],
  ["Blackwell (2024)", "FP6, FP4 (block-scaled)", "FP32"],
];

function Grid({ x, y, name, fill, stroke }: { x: number; y: number; name: string; fill: string; stroke: string }) {
  const c = 11;
  return (
    <g>
      {Array.from({ length: 16 }, (_, i) => (
        <rect key={i} x={x + (i % 4) * c} y={y + Math.floor(i / 4) * c} width={c - 0.8} height={c - 0.8} rx={0.8} fill={fill} stroke={stroke} strokeWidth={0.4} />
      ))}
      {Array.from({ length: 16 }, (_, i) => (
        <Txt key={i} x={x + (i % 4) * c + c / 2 - 0.4} y={y + Math.floor(i / 4) * c + c / 2 + 1.6} size={3.8} anchor="middle" color="#fff" weight={600}>
          {`${name}${Math.floor(i / 4)}${i % 4}`}
        </Txt>
      ))}
    </g>
  );
}

export function MmaFigure() {
  const T = usePageText();
  const k = (s: string) => T(s as PageKey);
  const y = 8;
  return (
    <Frame title={k("mixed.fig.mma.title")} testid="mixed-mma" note={k("mixed.fig.mma.note")}>
      <svg viewBox={`0 0 ${HW} 62`} className="mx-auto block h-auto w-full max-w-[620px] p-1" role="img" aria-label={k("mixed.fig.mma.title")}>
        <Txt x={20} y={y + 26} size={14} weight={700}>
          D
        </Txt>
        <Txt x={36} y={y + 26} size={12}>
          =
        </Txt>
        <Grid x={52} y={y} name="A" fill="#0f9d8a" stroke="#0b6b5e" />
        <Txt x={104} y={y + 25} size={11}>
          ×
        </Txt>
        <Grid x={116} y={y} name="B" fill="#7c3aed" stroke="#5b21b6" />
        <Txt x={168} y={y + 25} size={12} weight={700}>
          +
        </Txt>
        <Grid x={182} y={y} name="C" fill="#65a30d" stroke="#3f6212" />
        <Txt x={240} y={y + 14} size={5.6} color={COL.muted}>
          {k("mixed.fig.mma.one")}
        </Txt>
        <Txt x={240} y={y + 23} size={5.6} color={COL.muted}>
          {k("mixed.fig.mma.cycle")}
        </Txt>
      </svg>
      <div className="overflow-x-auto px-3 pb-2">
        <table className="w-full text-left text-xs">
          <thead className="text-gray-600">
            <tr>
              <th className="py-1 pr-3 font-semibold">{k("mixed.fig.mma.gen")}</th>
              <th className="py-1 pr-3 font-semibold">
                <span className="text-teal-700">A</span>, <span className="text-violet-700">B</span>
              </th>
              <th className="py-1 font-semibold">
                <span className="text-lime-700">C</span>, D {k("mixed.fig.mma.acc")}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 font-mono">
            {MMA_ROWS.map(([g, ab, cd]) => (
              <tr key={g}>
                <td className="py-1 pr-3 font-sans text-gray-900">{g}</td>
                <td className="py-1 pr-3">{ab}</td>
                <td className="py-1">{cd}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-1 text-[11px] text-gray-500">
          <Src href={GUIDE} label="NVIDIA mixed-precision training guide" /> · <Src href={PTX} label="PTX ISA: mma" />
        </p>
      </div>
    </Frame>
  );
}

// --- 4. one multiply-accumulate: FP16 in, exact product, FP32 sum ---------------------------------------------

function Reg({ x, y, label, color }: { x: number; y: number; label: string; color: string }) {
  return (
    <g>
      <rect x={x} y={y} width={22} height={11} rx={1.5} fill={color} />
      <Txt x={x + 11} y={y + 7.6} size={5.6} anchor="middle" color="#fff" weight={600}>
        {label}
      </Txt>
    </g>
  );
}

export function MacGraph() {
  const T = usePageText();
  const k = (s: string) => T(s as PageKey);
  const brown = "#9a3412";
  const a = (x1: number, y1: number, x2: number, y2: number) => (
    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={COL.text} strokeWidth={0.7} markerEnd="url(#mac-a)" />
  );
  return (
    <Frame title={k("mixed.fig.mac.title")} testid="mixed-mac" note={k("mixed.fig.mac.note")} links={[[GUIDE, "NVIDIA"]]}>
      <svg viewBox={`0 0 ${HW} 92`} className="mx-auto block h-auto w-full max-w-[620px] p-1" role="img" aria-label={k("mixed.fig.mac.title")}>
        <defs>
          <marker id="mac-a" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto">
            <path d="M0,0 L6,3 L0,6 z" fill={COL.text} />
          </marker>
        </defs>
        {[
          [36, "in"],
          [118, "mul"],
          [200, "add"],
          [284, "out"],
        ].map(([x, id]) => (
          <Txt key={id} x={x as number} y={10} size={5.8} anchor="middle" weight={600}>
            {k(`mixed.fig.mac.${id}`)}
          </Txt>
        ))}
        <Reg x={25} y={30} label="F16" color={brown} />
        <Reg x={25} y={56} label="F16" color={brown} />
        {a(47, 35.5, 112, 45)}
        {a(47, 61.5, 112, 49)}
        <circle cx={118} cy={47} r={7} fill="#86b85a" stroke="#3f6212" strokeWidth={0.6} />
        <Txt x={118} y={49.6} size={8} anchor="middle" weight={700}>
          ×
        </Txt>
        {a(125, 47, 192, 47)}
        <circle cx={200} cy={47} r={7} fill="#86b85a" stroke="#3f6212" strokeWidth={0.6} />
        <Txt x={200} y={49.8} size={8} anchor="middle" weight={700}>
          +
        </Txt>
        {a(193, 24, 197, 39)}
        {a(207, 24, 203, 39)}
        <Txt x={200} y={21} size={4.8} anchor="middle" color={COL.muted}>
          {k("mixed.fig.mac.more")}
        </Txt>
        <Reg x={189} y={72} label="F32" color={brown} />
        {a(200, 72, 200, 55)}
        {a(207, 47, 272, 47)}
        <Reg x={273} y={41.5} label="F32" color={brown} />
      </svg>
    </Frame>
  );
}

// --- 5. the three problems of half precision, and their fixes (Mixed precision ▸ Why) --------------------------
// Every number is computed with the site's own rounding (chipsModel.roundTo / runningSum), not drawn by hand.

const PW = 320;
const F16 = fmt("fp16");
const F32 = fmt("fp32");

/** gradients on a log2 axis against FP16's range; solved: the same gradients times the loss scale S = 2^10 */
function UnderflowFig({ solved, k }: { solved: boolean; k: (s: string) => string }) {
  const grads = [-31, -28, -26, -22, -18, -14, -11].map((e) => 1.3 * 2 ** e);
  const S = solved ? 2 ** 10 : 1;
  const L = 14;
  const R = PW - 10;
  const lo = -34;
  const hi = 17;
  const xs = (e: number) => L + ((e - lo) / (hi - lo)) * (R - L);
  return (
    <svg viewBox={`0 0 ${PW} 64`} className="mx-auto mt-2 block h-auto w-full max-w-[520px]" aria-hidden>
      <rect x={xs(-24)} y={16} width={xs(16) - xs(-24)} height={18} rx={2} fill={COL.greenLight} />
      <rect x={L} y={16} width={xs(-24) - L} height={18} rx={2} fill={COL.redLight} />
      <Txt x={(xs(-24) + xs(16)) / 2} y={12} size={6} anchor="middle" color={COL.green}>
        {k("mixed.why.fig.range")}
      </Txt>
      <Txt x={(L + xs(-24)) / 2} y={12} size={6} anchor="middle" color={COL.red}>
        → 0
      </Txt>
      {grads.map((g, i) => {
        const v = g * S;
        const lost = roundTo(v, F16) === 0;
        const x = xs(Math.log2(v));
        return (
          <g key={i}>
            {solved && <line x1={xs(Math.log2(g))} x2={x - 3} y1={25} y2={25} stroke={COL.idleLine} strokeWidth={0.6} />}
            <circle cx={x} cy={25} r={3.2} fill={lost ? "#fff" : COL.green} stroke={lost ? COL.red : COL.green} strokeWidth={1} />
            {lost && <path d={`M${x - 2} ${23} l4 4 M${x + 2} ${23} l-4 4`} stroke={COL.red} strokeWidth={0.9} />}
          </g>
        );
      })}
      {[-32, -24, -16, -8, 0, 8, 16].map((e) => (
        <Txt key={e} x={xs(e)} y={46} size={5.2} anchor="middle" color={COL.muted}>
          {`2^${e}`}
        </Txt>
      ))}
      <Txt x={R} y={58} size={6} anchor="end" color={solved ? COL.green : COL.red} weight={600}>
        {solved ? `× S = 2^10: ${grads.filter((g) => roundTo(g * S, F16) !== 0).length}/${grads.length} ✓` : `${grads.filter((g) => roundTo(g, F16) === 0).length}/${grads.length} → 0`}
      </Txt>
    </svg>
  );
}

/** the FP16 numbers next to 1 (0.000977 apart): w + 0.0001 rounds back to w; solved: an FP32 master weight keeps it */
function UpdateFig({ solved }: { solved: boolean }) {
  const f = solved ? F32 : F16;
  const w = 1;
  const u = 0.0001;
  const got = roundTo(w + u, f);
  const L = 20;
  const R = PW - 20;
  const span = 2 ** -10 * 1.15; // a little more than one FP16 step
  const xs = (v: number) => L + ((v - w + 0.06 * span) / span) * (R - L) * 0.92;
  const ticks = solved ? Array.from({ length: 12 }, (_, i) => w + (i * span) / 11) : [w, w + 2 ** -10];
  return (
    <svg viewBox={`0 0 ${PW} 58`} className="mx-auto mt-2 block h-auto w-full max-w-[520px]" aria-hidden>
      <line x1={L} x2={R} y1={24} y2={24} stroke={COL.idleLine} strokeWidth={0.8} />
      {ticks.map((t, i) => (
        <line key={i} x1={xs(t)} x2={xs(t)} y1={solved ? 20 : 17} y2={solved ? 28 : 31} stroke={solved ? COL.blue : COL.text} strokeWidth={solved ? 0.5 : 1} />
      ))}
      {!solved && (
        <>
          <Txt x={xs(w)} y={41} size={6} anchor="middle">
            1
          </Txt>
          <Txt x={xs(w + 2 ** -10)} y={41} size={6} anchor="middle">
            1.000977
          </Txt>
        </>
      )}
      <circle cx={xs(w + u)} cy={24} r={2.6} fill="#fff" stroke={COL.cur} strokeWidth={1} />
      <path d={`M${xs(w + u)} 13 L${xs(got)} 21`} stroke={solved ? COL.green : COL.red} strokeWidth={0.8} markerEnd="" />
      <circle cx={xs(got)} cy={24} r={3} fill={solved ? COL.green : COL.red} />
      <Txt x={xs(w + u) + 4} y={11} size={6} color={COL.cur}>
        1 + 0.0001
      </Txt>
      <Txt x={R} y={54} size={6.5} anchor="end" color={solved ? COL.green : COL.red} weight={600}>
        {`${f.name}: 1 + 0.0001 → ${got === 1 ? "1" : got.toFixed(4)}`}
      </Txt>
    </svg>
  );
}

/** a running sum of 0.1: FP16 stalls at 256 (steps there are 0.25 wide); solved: the FP32 accumulator keeps going */
function SumFig({ solved, k }: { solved: boolean; k: (s: string) => string }) {
  const n = 4000;
  const values = Array.from({ length: n }, () => 0.1);
  const got = runningSum(values, solved ? F32 : F16, solved ? "wide" : "naive");
  const L = 26;
  const R = PW - 10;
  const top = 6;
  const bot = 46;
  const xs = (i: number) => L + (i / n) * (R - L);
  const ys = (v: number) => bot - (v / 420) * (bot - top);
  const path = (f: (i: number) => number) =>
    Array.from({ length: 81 }, (_, j) => Math.min(n - 1, j * 50)).map((i, j) => `${j ? "L" : "M"}${xs(i).toFixed(1)},${ys(f(i)).toFixed(1)}`).join("");
  const last = got[n - 1];
  return (
    <svg viewBox={`0 0 ${PW} 62`} className="mx-auto mt-2 block h-auto w-full max-w-[520px]" aria-hidden>
      {[0, 256, 400].map((v) => (
        <g key={v}>
          <line x1={L} x2={R} y1={ys(v)} y2={ys(v)} stroke="#eef0f3" strokeWidth={0.5} />
          <Txt x={L - 3} y={ys(v) + 1.8} size={5} anchor="end" color={COL.muted}>
            {v}
          </Txt>
        </g>
      ))}
      <path d={path((i) => 0.1 * (i + 1))} fill="none" stroke={COL.idleLine} strokeWidth={1.4} strokeDasharray="3 2" />
      <path d={path((i) => got[i])} fill="none" stroke={solved ? COL.green : COL.red} strokeWidth={1.3} />
      <Txt x={R} y={ys(400) - 1.5} size={5.5} anchor="end" color={COL.muted}>
        {k("mixed.why.fig.exact")}
      </Txt>
      <Txt x={(L + R) / 2} y={bot + 8} size={5.5} anchor="middle" color={COL.muted}>
        {`${k("mixed.why.fig.steps")}: + 0.1 × ${n}`}
      </Txt>
      <Txt x={R} y={bot + 15} size={6.5} anchor="end" color={solved ? COL.green : COL.red} weight={600}>
        {`${solved ? "FP32" : "FP16"}: ${Number(last.toFixed(1))}`}
      </Txt>
    </svg>
  );
}

/** the figure for problem n of Mixed precision ▸ Why, before (problem) or after (solution) the fix */
export function ProblemFig({ n, solved }: { n: 1 | 2 | 3; solved: boolean }) {
  const T = usePageText();
  const k = (s: string) => T(s as PageKey);
  return (
    <div data-testid={`mixed-problem-fig-${n}${solved ? "-solved" : ""}`}>
      {n === 1 ? <UnderflowFig solved={solved} k={k} /> : n === 2 ? <UpdateFig solved={solved} /> : <SumFig solved={solved} k={k} />}
    </div>
  );
}
