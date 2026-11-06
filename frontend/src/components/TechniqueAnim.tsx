import { type ReactNode } from "react";
import { usePageText, type PageKey } from "../content/pages";
import {
  AG,
  BATCH,
  FFT_T,
  LAP_KERNEL,
  LA_B,
  LA_H,
  PLAN,
  RK,
  TDD_LIGHT,
  agBackward,
  agForward,
  batchTraj,
  bcA,
  bcB,
  broadcastOp,
  cabs,
  cexp,
  cmul,
  contractTerms,
  conv2d,
  cxC,
  cxE,
  cxPhased,
  cxPsi,
  cxTime,
  cxX,
  deviceOps,
  devX,
  dft,
  eigh2,
  einsum,
  esA,
  esB,
  esCell,
  fftPsi0,
  fftShift,
  fftX,
  fftk,
  freeEvolve,
  isEigen,
  laDir,
  matVec2,
  rk4Path,
  rk4Stages,
  rkExact,
  rkF,
  rows2,
  solve2,
  stPath,
  stU,
  stageAt,
  stepCount,
  totalFrames,
  type C,
  type TechCase,
} from "../techAnim";
import { AnimControls, COL, Cell, Txt, blueScale, useAnimClock } from "./animKit";
import { RichText } from "./Tex";

const W = 280; // SVG user units, scaled to the column width
const HOLD = 8; // ticks the finished picture stays before the cycle restarts
const BATCH_COL = [COL.blue, COL.cur, COL.green, COL.violet];

type Stage = ReturnType<typeof stageAt>;
interface PicProps {
  g: number;
  s: Stage;
  T: (k: PageKey) => string;
}

const fmt = (v: number, d = 2) => {
  const r = Number(v.toFixed(d));
  return (Object.is(r, -0) ? 0 : r).toString().replace("-", "−");
};

/** Diverging fill: blue for positive, amber for negative values, |v| / max in [0, 1]. */
function divScale(v: number, max: number): string {
  const t = Math.min(Math.abs(v) / max, 1);
  if (v >= 0) return blueScale(t * 0.8);
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(255, 245)},${lerp(251, 158)},${lerp(235, 11)})`; // amber-50 -> amber-500
}

function Arrow({ x1, y1, x2, y2, color, w = 1.4, dash, head = 4 }: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  w?: number;
  dash?: string;
  head?: number;
}) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  const len = Math.hypot(x2 - x1, y2 - y1);
  if (len < 0.5) return null;
  const hx = (s: number) => x2 - head * Math.cos(a + s * 0.45);
  const hy = (s: number) => y2 - head * Math.sin(a + s * 0.45);
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2 - 0.6 * head * Math.cos(a)} y2={y2 - 0.6 * head * Math.sin(a)} stroke={color} strokeWidth={w} strokeDasharray={dash} />
      <path d={`M ${x2} ${y2} L ${hx(1)} ${hy(1)} L ${hx(-1)} ${hy(-1)} Z`} fill={color} />
    </g>
  );
}

/** Plot frame with fixed limits; returns the data -> SVG maps. Axis labels carry units (a.u. if none). */
function frame(x0: number, y0: number, w: number, h: number, xl: [number, number], yl: [number, number]) {
  const X = (v: number) => x0 + ((v - xl[0]) / (xl[1] - xl[0])) * w;
  const Y = (v: number) => y0 + h - ((v - yl[0]) / (yl[1] - yl[0])) * h;
  return { X, Y };
}

function Axes({ x0, y0, w, h, xl, yl, xlab, ylab, xt, yt }: {
  x0: number;
  y0: number;
  w: number;
  h: number;
  xl: [number, number];
  yl: [number, number];
  xlab: string;
  ylab: string;
  xt?: number[];
  yt?: number[];
}) {
  const { X, Y } = frame(x0, y0, w, h, xl, yl);
  return (
    <g>
      <rect x={x0} y={y0} width={w} height={h} fill="none" stroke={COL.idleLine} strokeWidth={0.6} />
      {(xt ?? xl).map((v) => (
        <Txt key={`x${v}`} x={X(v)} y={y0 + h + 8} size={6.5} anchor="middle" color={COL.muted} kind="math">
          {fmt(v, 1)}
        </Txt>
      ))}
      {(yt ?? yl).map((v) => (
        <Txt key={`y${v}`} x={x0 - 2} y={Y(v) + 2.3} size={6.5} anchor="end" color={COL.muted} kind="math">
          {fmt(v, 1)}
        </Txt>
      ))}
      <Txt x={x0 + w} y={y0 + h + 16} size={7.5} anchor="end" color={COL.muted} kind="math">
        {xlab}
      </Txt>
      <Txt x={x0 + 2} y={y0 - 3} size={7.5} color={COL.muted} kind="math">
        {ylab}
      </Txt>
    </g>
  );
}

const poly = (pts: [number, number][]) => pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");

// ---- 1. TDD ---------------------------------------------------------------------------------------------

function Tdd({ s }: PicProps) {
  const testLines = ["def test_energy():", "  e0, e1 = energy(...)", "  drift = abs(e1/e0 - 1)", "  assert drift < 1e-5"];
  const code = ["def solve(m, x0, v0, dt):", "  ... RK4 steps ...", "  return xs, vs"];
  const codeRef = ["def solve(m, x0, v0, dt):", "  xs = rk4_scan(f, y0)", "  return xs, vs"];
  const nTest = s.step === 0 ? s.t + 1 : 4;
  const nCode = s.step < 2 ? 0 : s.step === 2 ? Math.min(s.t + 1, 3) : 3;
  const light = TDD_LIGHT[s.step];
  const lines = s.step === 4 ? codeRef : code;
  const file = (y: number, h: number, name: string, body: string[], n: number, hot: number) => (
    <g>
      <rect x={6} y={y} width={140} height={h} rx={3} fill="white" stroke={COL.idleLine} />
      <rect x={6} y={y} width={140} height={12} rx={3} fill={COL.idle} />
      <Txt x={11} y={y + 9} size={7.5} kind="mono" color={COL.muted}>
        {name}
      </Txt>
      {body.slice(0, n).map((l, i) => (
        <g key={i}>
          {i === hot && <rect x={8} y={y + 15 + i * 11} width={136} height={10} fill={COL.blueLight} />}
          <Txt x={11} y={y + 23 + i * 11} size={7.5} kind="mono" pre>
            {l}
          </Txt>
        </g>
      ))}
    </g>
  );
  const chip = (i: number, label: string, color: string, light: string) => {
    const on = (s.step < 2 ? 0 : s.step < 4 ? 1 : 2) === i;
    return (
      <g key={i}>
        <rect x={150 + i * 43} y={120} width={40} height={16} rx={8} fill={on ? light : "white"} stroke={on ? color : COL.idleLine} strokeWidth={on ? 1.4 : 0.7} />
        <Txt x={170 + i * 43} y={131} size={7.5} anchor="middle" kind="mono" color={on ? color : COL.muted}>
          {label}
        </Txt>
      </g>
    );
  };
  return (
    <>
      {file(6, 62, "test.py", testLines, nTest, s.step === 0 ? s.t : -1)}
      {file(76, 52, "solve.py", lines, nCode, s.step === 2 ? s.t : s.step === 4 ? 1 : -1)}
      <rect x={196} y={8} width={44} height={68} rx={8} fill={COL.text} />
      <circle cx={218} cy={26} r={11} fill={light === "red" ? COL.red : "#4b5563"} />
      <circle cx={218} cy={56} r={11} fill={light === "green" ? COL.green : "#4b5563"} />
      <Txt x={218} y={92} size={8} anchor="middle" kind="mono" color={light === "red" ? COL.red : light === "green" ? COL.green : COL.muted}>
        {light === "red" ? (s.step === 1 ? "1 failed" : "…") : light === "green" ? "1 passed" : "pytest"}
      </Txt>
      <Txt x={218} y={106} size={8} anchor="middle" kind="math" color={COL.muted}>
        {s.step === 1 ? "no solver" : s.step >= 3 ? "drift = 4·10^{−7}" : ""}
      </Txt>
      {chip(0, "test", COL.red, COL.redLight)}
      {chip(1, "code", COL.green, COL.greenLight)}
      {chip(2, "refactor", COL.blue, COL.blueLight)}
    </>
  );
}

// ---- 2. tensors and devices ----------------------------------------------------------------------------

function Tensors({ s, T }: PicProps) {
  const cs = 18;
  const cpuX = 26;
  const gpuX = 170;
  const ty = 40;
  const p = s.step === 0 ? 0 : s.step === 1 ? (s.t + 1) / s.n : 1;
  const tx = cpuX + (gpuX - cpuX) * p;
  const ops = deviceOps(devX);
  const op = s.step === 2 ? Math.floor(s.t / 2) : -1;
  const cur = op >= 0 ? ops[op].t : devX;
  const hot = s.step === 0 ? s.t : -1;
  const shapeLbl = cur.shape.length === 2 ? "(2, 3)" : "(2,)";
  const strideLbl = cur.shape.length === 2 ? "stride (3, 1)" : "stride (1,)";
  const box = (x: number, label: string, on: boolean) => (
    <g>
      <rect x={x} y={8} width={122} height={114} rx={5} fill={on ? COL.blueLight : COL.idle} fillOpacity={on ? 0.5 : 0.6} stroke={on ? COL.blue : COL.idleLine} />
      <Txt x={x + 6} y={19} size={8} weight={600} color={on ? COL.blue : COL.muted}>
        {label}
      </Txt>
    </g>
  );
  const rows = cur.shape.length === 2 ? rows2(cur) : [cur.data];
  return (
    <>
      {box(8, "CPU", p < 1)}
      {box(152, "GPU  cuda:0", p >= 1)}
      {s.step === 1 && <Arrow x1={118} y1={64} x2={164} y2={64} color={COL.blue} w={1.6} />}
      <Txt x={tx} y={ty - 6} size={8} kind="mono">
        {`x: ${shapeLbl}`}
      </Txt>
      {rows.map((row, i) =>
        row.map((v, j) => {
          const k = i * row.length + j;
          return (
            <Cell
              key={`${i}${j}`}
              x={tx + j * cs}
              y={ty + i * cs}
              s={cs - 2}
              fill={k === hot ? COL.curLight : op >= 0 && s.t % 2 === 0 ? COL.blue : COL.done}
              stroke={k === hot ? COL.cur : COL.doneLine}
              sw={k === hot ? 1.5 : 0.75}
              label={v}
              labelColor={op >= 0 && s.t % 2 === 0 ? "white" : COL.text}
              size={8}
            />
          );
        }),
      )}
      <Txt x={tx} y={ty + 2 * cs + 10} size={7.5} kind="mono" color={COL.muted}>
        {strideLbl}
      </Txt>
      {/* the same numbers as one block of memory */}
      {cur.data.map((v, k) => (
        <Cell
          key={k}
          x={tx - 8 + k * 13}
          y={ty + 2 * cs + 16}
          s={12}
          fill={k === hot ? COL.curLight : "white"}
          stroke={k === hot ? COL.cur : COL.idleLine}
          sw={k === hot ? 1.5 : 0.6}
          label={v}
          size={6.5}
        />
      ))}
      <Txt x={8} y={136} size={8} kind="math" color={hot >= 0 ? COL.cur : COL.muted}>
        {hot >= 0 ? `x[${Math.floor(hot / 3)}, ${hot % 3}] → 3·${Math.floor(hot / 3)} + ${hot % 3} = ${hot}` : ""}
      </Txt>
      <Txt x={152} y={136} size={8} kind="mono" color={COL.blue}>
        {op >= 0 ? ops[op].label : s.step === 1 ? "x = x.to(device)" : ""}
      </Txt>
      <Txt x={8} y={150} size={7.5} color={COL.muted}>
        {T("skills.anim.tensors.copies").replace("{n}", p >= 1 ? "1" : "0")}
      </Txt>
    </>
  );
}

// ---- 3. broadcasting -----------------------------------------------------------------------------------

function Broadcasting({ s, T }: PicProps) {
  const cs = 20;
  const ax = 10;
  const rx = 42;
  const ry = 34;
  const a = bcA.data;
  const b = bcB.data;
  const u = rows2(broadcastOp(bcA, bcB, (p, q) => p + q));
  const stretch = s.step === 1 ? (s.t + 1) / s.n : s.step === 2 ? 1 : 0;
  const nCols = Math.round(stretch * 4);
  const nRows = Math.round(stretch * 3);
  const computed = s.step === 2;
  // shape table: columns = axes, compared from the right
  const tx = 150;
  const col = (d: number) => tx + 30 + d * 34;
  const checked = (d: number) => s.step > 0 || (s.step === 0 && ((d === 1 && s.t >= 1) || (d === 0 && s.t >= 2)));
  const focus = (d: number) => s.step === 0 && ((d === 1 && s.t === 1) || (d === 0 && s.t === 2));
  const shapes: [string, number[]][] = [
    ["a", [3, 1]],
    ["b", [1, 4]],
    ["a+b", [3, 4]],
  ];
  return (
    <>
      <Txt x={ax + cs / 2 - 1} y={ry - 6} size={8} anchor="middle" kind="math">a</Txt>
      <Txt x={rx - 6} y={ry - cs + 4} size={8} anchor="end" kind="math">b</Txt>
      {/* a: (3, 1) column and its virtual copies */}
      {a.map((v, i) => (
        <Cell key={`a${i}`} x={ax} y={ry + i * cs} s={cs - 2} fill={COL.curLight} stroke={COL.cur} label={v} size={8} />
      ))}
      {/* b: (1, 4) row above the table */}
      {b.map((v, j) => (
        <Cell key={`b${j}`} x={rx + j * cs} y={ry - cs - 2} s={cs - 2} fill={COL.violetLight} stroke={COL.violet} label={v} size={8} />
      ))}
      {u.map((row, i) =>
        row.map((v, j) => {
          const va = j < nCols; // stretched a covers this cell
          const vb = i < nRows;
          return (
            <g key={`u${i}${j}`}>
              <Cell x={rx + j * cs} y={ry + i * cs} s={cs - 2} fill={computed ? COL.blue : "white"} stroke={computed ? COL.blue : COL.idleLine} label={computed ? v : undefined} labelColor="white" size={8} />
              {!computed && va && (
                <Txt x={rx + j * cs + 5} y={ry + i * cs + 8} size={6.5} anchor="middle" color={COL.cur}>
                  {a[i]}
                </Txt>
              )}
              {!computed && vb && (
                <Txt x={rx + j * cs + 13} y={ry + i * cs + 15} size={6.5} anchor="middle" color={COL.violet}>
                  {b[j]}
                </Txt>
              )}
            </g>
          );
        }),
      )}
      {/* the virtual stretch: dashed outlines, nothing copied */}
      {stretch > 0 && !computed && (
        <g>
          <rect x={rx - 1} y={ry - 1} width={nCols * cs} height={3 * cs - 1} fill={COL.cur} fillOpacity={0.08} stroke={COL.cur} strokeDasharray="3 2" strokeWidth={0.9} />
          <rect x={rx - 1} y={ry - 1} width={4 * cs - 1} height={nRows * cs} fill={COL.violet} fillOpacity={0.06} stroke={COL.violet} strokeDasharray="3 2" strokeWidth={0.9} />
        </g>
      )}
      {stretch > 0 && (
        <Txt x={ax} y={ry + 3 * cs + 24} size={7.5} kind="mono" color={COL.muted}>
          {computed ? "u = a + b   # (3, 4)" : T("skills.anim.broadcasting.virtual")}
        </Txt>
      )}
      {/* shape table */}
      {shapes.map(([name, sh], r) => (
        <g key={name}>
          <Txt x={tx + 24} y={ry - 6 + r * 20 + 12} size={8} anchor="end" kind="mono" color={r === 2 ? COL.blue : COL.text}>
            {name}
          </Txt>
          {sh.map((v, d) => {
            const show = r < 2 || checked(d);
            const stretched = r < 2 && v === 1 && checked(d);
            return (
              <Cell
                key={d}
                x={col(d)}
                y={ry - 6 + r * 20}
                s={16}
                fill={!show ? "white" : r === 2 ? COL.blueLight : stretched ? COL.idle : "white"}
                stroke={focus(d) ? COL.cur : r === 2 && show ? COL.blue : COL.idleLine}
                sw={focus(d) ? 1.5 : 0.7}
                label={show ? (stretched ? `1→${shapes[2][1][d]}` : v) : undefined}
                labelColor={stretched ? COL.muted : COL.text}
                size={stretched ? 6 : 8}
              />
            );
          })}
        </g>
      ))}
      <Txt x={col(0) + 8} y={ry - 12} size={7} anchor="middle" color={COL.muted} kind="mono">dim 0</Txt>
      <Txt x={col(1) + 8} y={ry - 12} size={7} anchor="middle" color={COL.muted} kind="mono">dim 1</Txt>
      {s.step >= 1 && (
        <g>
          <Txt x={tx + 4} y={ry + 62} size={7.5} kind="mono" color={COL.cur}>a: stride (1, 0)</Txt>
          <Txt x={tx + 4} y={ry + 73} size={7.5} kind="mono" color={COL.violet}>b: stride (0, 1)</Txt>
        </g>
      )}
    </>
  );
}

// ---- 4. einsum / contraction ---------------------------------------------------------------------------

const IDX_COL: Record<string, string> = { i: COL.blue, j: COL.cur, k: COL.green };

function Einsum({ s, T }: PicProps) {
  const cs = 17;
  const A = rows2(esA);
  const B = rows2(esB);
  const Cm = rows2(einsum("ij,jk->ik", esA, esB));
  const ax = 22;
  const ay = 58;
  const bx = 104;
  const by = 42;
  const cx = 186;
  const cy = 58;
  const lit = s.step === 0 ? "ijk"[s.t] : null; // index named in stage 0
  const cell = s.step === 1 ? esCell(s.t) : null;
  const filled = (i: number, k: number) => s.step === 2 || (s.step === 1 && i * 2 + k <= s.t);
  const fade = s.step === 2 ? Math.max(0, 1 - (s.t + 1) / 3) : 1; // the summed index j disappearing
  const spec = "ij,jk->ik";
  const letterOp = (c: string) => (c === "j" ? fade : 1);
  const idxLbl = (x: number, y: number, c: string) => (
    <Txt key={`${c}${x},${y}`} x={x} y={y} size={8.5} anchor="middle" kind="math" color={IDX_COL[c]} weight={lit === c ? 700 : undefined} opacity={c === "j" ? Math.max(fade, 0.25) : 1}>
      {c}
    </Txt>
  );
  // Penrose diagram
  const py = 156;
  const merge = s.step === 2 ? Math.min((s.t + 1) / 3, 1) : 0;
  const pA = 88 + merge * 18;
  const pB = 160 - merge * 18;
  const terms = cell ? contractTerms(esA, esB, cell[0], cell[1]) : null;
  return (
    <>
      <text x={W / 2} y={16} fontSize={11} textAnchor="middle" style={{ fontFamily: "var(--font-code)" }}>
        {[...spec].map((c, n) => (
          <tspan key={n} fill={IDX_COL[c] ?? COL.text} fontWeight={lit === c ? 700 : 400} opacity={letterOp(c)}>
            {c}
          </tspan>
        ))}
      </text>
      {/* A: rows i, columns j */}
      <Txt x={ax + 1.5 * cs} y={ay - 14} size={8.5} anchor="middle" kind="math">A</Txt>
      {[0, 1, 2].map((j) => idxLbl(ax + j * cs + cs / 2 - 1, ay - 4, "j"))}
      {[0, 1].map((i) => idxLbl(ax - 7, ay + i * cs + 11, "i"))}
      {A.map((row, i) =>
        row.map((v, j) => {
          const on = cell && cell[0] === i;
          return (
            <Cell key={`a${i}${j}`} x={ax + j * cs} y={ay + i * cs} s={cs - 2} fill={on ? COL.blueLight : COL.idle} stroke={on ? COL.cur : COL.idleLine} sw={on ? 1.4 : 0.7} label={v} size={8} />
          );
        }),
      )}
      {/* B: rows j, columns k */}
      <Txt x={bx + cs} y={by - 14} size={8.5} anchor="middle" kind="math">B</Txt>
      {[0, 1].map((k) => idxLbl(bx + k * cs + cs / 2 - 1, by - 4, "k"))}
      {[0, 1, 2].map((j) => idxLbl(bx - 7, by + j * cs + 11, "j"))}
      {B.map((row, j) =>
        row.map((v, k) => {
          const on = cell && cell[1] === k;
          return (
            <Cell key={`b${j}${k}`} x={bx + k * cs} y={by + j * cs} s={cs - 2} fill={on ? COL.greenLight : COL.idle} stroke={on ? COL.cur : COL.idleLine} sw={on ? 1.4 : 0.7} label={v} size={8} />
          );
        }),
      )}
      {/* C: rows i, columns k */}
      <Txt x={cx - 12} y={cy + cs + 3} size={9} anchor="middle" kind="math">=</Txt>
      <Txt x={cx + cs} y={cy - 14} size={8.5} anchor="middle" kind="math">C</Txt>
      {[0, 1].map((k) => idxLbl(cx + k * cs + cs / 2 - 1, cy - 4, "k"))}
      {[0, 1].map((i) => idxLbl(cx + 2 * cs + 6, cy + i * cs + 11, "i"))}
      {Cm.map((row, i) =>
        row.map((v, k) => {
          const now = cell && cell[0] === i && cell[1] === k;
          const on = filled(i, k);
          return (
            <Cell key={`c${i}${k}`} x={cx + k * cs} y={cy + i * cs} s={cs - 2} fill={now ? COL.blue : on ? COL.done : "white"} stroke={now ? COL.blue : on ? COL.doneLine : COL.idleLine} label={on ? v : undefined} labelColor={now ? "white" : COL.text} size={8} />
          );
        }),
      )}
      <Txt x={W / 2} y={110} size={8.5} anchor="middle" kind="math" color={cell ? COL.text : COL.muted}>
        {cell && terms
          ? `C_{${cell[0]}${cell[1]}} = ${terms.map(([p, q]) => `${p}·${q}`).join(" + ")} = ${Cm[cell[0]][cell[1]]}`
          : s.step === 2
            ? "C_{ik} = Σ_j A_{ij} B_{jk}"
            : ""}
      </Txt>
      {/* Penrose: boxes with one leg per index; the joined leg j is the contraction */}
      <Txt x={W / 2} y={128} size={7.5} anchor="middle" color={COL.muted}>
        {T("skills.anim.einsum.penrose")}
      </Txt>
      <line x1={pA - 26} y1={py} x2={pA - 9} y2={py} stroke={COL.blue} strokeWidth={1.6} />
      <line x1={pB + 9} y1={py} x2={pB + 26} y2={py} stroke={COL.green} strokeWidth={1.6} />
      {merge < 1 && (
        <line x1={pA + 9} y1={py} x2={pB - 9} y2={py} stroke={COL.cur} strokeWidth={s.step === 1 || lit === "j" ? 3 : 1.6} opacity={1 - merge} />
      )}
      {merge < 1 ? (
        <>
          <rect x={pA - 9} y={py - 9} width={18} height={18} rx={2} fill={COL.idle} stroke={COL.text} strokeWidth={0.9} />
          <rect x={pB - 9} y={py - 9} width={18} height={18} rx={2} fill={COL.idle} stroke={COL.text} strokeWidth={0.9} />
          <Txt x={pA} y={py + 3} size={8.5} anchor="middle" kind="math">A</Txt>
          <Txt x={pB} y={py + 3} size={8.5} anchor="middle" kind="math">B</Txt>
          <Txt x={(pA + pB) / 2} y={py - 4} size={8} anchor="middle" kind="math" color={COL.cur} opacity={1 - merge}>j</Txt>
        </>
      ) : (
        <>
          <rect x={pA - 9} y={py - 9} width={pB - pA + 18} height={18} rx={2} fill={COL.blueLight} stroke={COL.blue} strokeWidth={1} />
          <Txt x={(pA + pB) / 2} y={py + 3} size={8.5} anchor="middle" kind="math">C</Txt>
        </>
      )}
      <Txt x={pA - 22} y={py - 4} size={8} anchor="middle" kind="math" color={COL.blue}>i</Txt>
      <Txt x={pB + 22} y={py - 4} size={8} anchor="middle" kind="math" color={COL.green}>k</Txt>
    </>
  );
}

// ---- 5. batching ---------------------------------------------------------------------------------------

function Batching({ s, T }: PicProps) {
  const cs = 13;
  const bx = 16;
  const gap = s.step === 0 ? 14 : s.step === 1 ? 14 * (1 - (s.t + 1) / s.n) : 0;
  const by = 30;
  const step = s.step === 2 ? s.t + 1 : 0;
  const Y = batchTraj[Math.min(step, BATCH.T)];
  const { X, Y: PY } = frame(108, 14, 160, 96, [0, 10], [0, 5]);
  const flash = s.step === 2;
  return (
    <>
      {Y.map((row, b) => (
        <g key={b}>
          {s.step === 0 && (
            <Txt x={bx - 3} y={by + b * (cs + gap) + 8} size={7.5} anchor="end" kind="math" color={BATCH_COL[b]}>
              {`y_${b}`}
            </Txt>
          )}
          {row.map((_, j) => (
            <Cell
              key={j}
              x={bx + j * cs}
              y={by + b * (cs + gap)}
              s={cs - 1}
              fill={flash ? COL.blueLight : "white"}
              stroke={BATCH_COL[b]}
              sw={0.9}
            />
          ))}
          <circle cx={bx + 4 * cs + 6} cy={by + b * (cs + gap) + 5} r={2.5} fill={BATCH_COL[b]} />
        </g>
      ))}
      {["x", "y", "v_x", "v_y"].map((l, j) => (
        <Txt key={l} x={bx + j * cs + 5} y={by - 4} size={7} anchor="middle" kind="math" color={COL.muted}>
          {l}
        </Txt>
      ))}
      {s.step >= 1 && (
        <g opacity={s.step === 1 ? (s.t + 1) / s.n : 1}>
          <rect x={bx - 2} y={by - 2} width={4 * cs + 3} height={4 * (cs + gap) - gap + 3} rx={2} fill="none" stroke={COL.blue} strokeWidth={1.2} />
          <Txt x={bx - 5} y={by + 2 * cs + 3} size={8} anchor="end" kind="math" color={COL.blue}>B</Txt>
          <Txt x={bx} y={by + 4 * (cs + gap) - gap + 12} size={8} kind="mono" color={COL.blue}>(B, 4)</Txt>
        </g>
      )}
      {s.step === 0 && (
        <Txt x={bx} y={by + 4 * (cs + 14) + 4} size={7.5} kind="mono" color={COL.muted}>4 × (4,)</Txt>
      )}
      <Axes x0={108} y0={14} w={160} h={96} xl={[0, 10]} yl={[0, 5]} xlab="x (m)" ylab="y (m)" xt={[0, 5, 10]} yt={[0, 5]} />
      {Y.map((_, b) => {
        const pts = batchTraj.slice(0, step + 1).map((S) => [X(S[b][0]), PY(S[b][1])] as [number, number]);
        const last = pts[pts.length - 1];
        return (
          <g key={b}>
            <polyline points={poly(pts)} fill="none" stroke={BATCH_COL[b]} strokeWidth={1} opacity={0.8} />
            <circle cx={last[0]} cy={last[1]} r={3} fill={BATCH_COL[b]} stroke="white" strokeWidth={0.8} />
          </g>
        );
      })}
      <Txt x={108} y={140} size={8} kind="mono" color={COL.blue}>
        {s.step === 2 ? "y = step(y)" : ""}
      </Txt>
      <Txt x={268} y={140} size={8} anchor="end" kind="mono" color={COL.blue}>
        {T("skills.anim.batching.calls").replace("{n}", String(step))}
      </Txt>
    </>
  );
}

// ---- 6. RK4 --------------------------------------------------------------------------------------------

function Rk4({ s }: PicProps) {
  const x0 = 26;
  const y0 = 12;
  const w = 168;
  const h = 110;
  const { X, Y } = frame(x0, y0, w, h, [0, 2.6], [0, 3]);
  const path = rk4Path(rkF, RK.y0, RK.h, RK.steps);
  const n = Math.floor(s.idx / 5);
  const sub = s.step; // 0..3 = k1..k4, 4 = the combined step
  const st = rk4Stages(rkF, path[n][0], path[n][1], RK.h);
  const exact = Array.from({ length: 53 }, (_, i) => {
    const t = (2.6 * i) / 52;
    return [X(t), Y(rkExact(t))] as [number, number];
  });
  const seg = (t: number, y: number, k: number) => {
    const d = 0.2;
    return { x1: X(t - d), y1: Y(y - k * d), x2: X(t + d), y2: Y(y + k * d) };
  };
  const tx = 202;
  return (
    <>
      <Axes x0={x0} y0={y0} w={w} h={h} xl={[0, 2.6]} yl={[0, 3]} xlab="t (a.u.)" ylab="y (a.u.)" xt={[0, 0.8, 1.6, 2.4]} yt={[0, 1, 2, 3]} />
      <polyline points={poly(exact)} fill="none" stroke={COL.muted} strokeWidth={0.9} strokeDasharray="3 2" />
      {path.slice(0, n + 1).map(([t, y], i) => (
        <g key={i}>
          {i > 0 && <line x1={X(path[i - 1][0])} y1={Y(path[i - 1][1])} x2={X(t)} y2={Y(y)} stroke={COL.blue} strokeWidth={1.4} />}
          <circle cx={X(t)} cy={Y(y)} r={2.6} fill={COL.blue} />
        </g>
      ))}
      {st.pts.slice(0, Math.min(sub, 3) + 1).map(([t, y], q) => {
        const now = q === sub;
        return (
          <g key={q} opacity={sub === 4 ? 0.35 : now ? 1 : 0.55}>
            {q > 0 && <line x1={X(path[n][0])} y1={Y(path[n][1])} x2={X(t)} y2={Y(y)} stroke={COL.cur} strokeWidth={0.7} strokeDasharray="2 2" />}
            <line {...seg(t, y, st.k[q])} stroke={COL.cur} strokeWidth={now ? 2 : 1.2} />
            <circle cx={X(t)} cy={Y(y)} r={now ? 2.6 : 1.8} fill={COL.cur} />
            <Txt x={X(t) + 3} y={Y(y) - 4} size={7} kind="math" color="#b45309">{`k_${q + 1}`}</Txt>
          </g>
        );
      })}
      {sub === 4 && (
        <Arrow x1={X(path[n][0])} y1={Y(path[n][1])} x2={X(path[n + 1][0])} y2={Y(path[n + 1][1])} color={COL.blue} w={1.8} />
      )}
      <Txt x={tx} y={20} size={8} kind="math" color={COL.muted}>{`n = ${n},  h = ${RK.h}`}</Txt>
      {st.k.map((k, q) => (
        <Txt key={q} x={tx} y={36 + q * 13} size={8} kind="math" color={q <= sub ? (q === sub ? "#b45309" : COL.text) : COL.idleLine}>
          {`k_${q + 1} = ${q <= sub ? fmt(k) : "?"}`}
        </Txt>
      ))}
      <Txt x={tx} y={96} size={8} kind="math" color={sub === 4 ? COL.blue : COL.idleLine}>
        {`y_{${n + 1}} = ${sub === 4 ? fmt(st.y1, 3) : "?"}`}
      </Txt>
      <Txt x={tx} y={110} size={7.5} kind="math" color={COL.muted}>
        {`e^{sin t} = ${fmt(rkExact(path[n + 1][0]), 3)}`}
      </Txt>
      <Txt x={tx} y={124} size={7.5} kind="mono" color={COL.muted}>1 : 2 : 2 : 1</Txt>
    </>
  );
}

// ---- 7. linear algebra ---------------------------------------------------------------------------------

function Linalg({ s }: PicProps) {
  const c = 70;
  const sc = 18;
  const P = (v: number[]) => [c + v[0] * sc, c - v[1] * sc] as const;
  const { E, V } = eigh2(LA_H);
  const ell = Array.from({ length: 65 }, (_, i) => {
    const w = matVec2(LA_H, laDir(i / 4));
    return P(w) as unknown as [number, number];
  });
  const v = s.step === 0 ? laDir(s.t) : null;
  const hv = v ? matVec2(LA_H, v) : null;
  const eig = v ? isEigen(LA_H, v, 1e-9) : false;
  const lam = hv && v ? hv[0] * v[0] + hv[1] * v[1] : 0;
  const x = solve2(LA_H, LA_B);
  const tx = 146;
  const vec = (w: number[], color: string, width = 1.6, dash?: string) => {
    const [x2, y2] = P(w);
    return <Arrow x1={c} y1={c} x2={x2} y2={y2} color={color} w={width} dash={dash} />;
  };
  return (
    <>
      <line x1={8} y1={c} x2={132} y2={c} stroke={COL.idleLine} strokeWidth={0.6} />
      <line x1={c} y1={8} x2={c} y2={132} stroke={COL.idleLine} strokeWidth={0.6} />
      <Txt x={132} y={c + 10} size={7} anchor="end" kind="math" color={COL.muted}>v_1 (a.u.)</Txt>
      <Txt x={c + 3} y={14} size={7} kind="math" color={COL.muted}>v_2 (a.u.)</Txt>
      <circle cx={c} cy={c} r={sc} fill="none" stroke={COL.muted} strokeWidth={0.7} strokeDasharray="2 2" />
      <polyline points={poly(ell)} fill={COL.blueLight} fillOpacity={0.35} stroke={COL.doneLine} strokeWidth={0.8} />
      {v && hv && (
        <>
          {vec(hv, eig ? COL.green : COL.blue, eig ? 2 : 1.6)}
          {vec(v, eig ? COL.green : COL.cur, 1.6)}
        </>
      )}
      {s.step === 1 &&
        [0, 1].map((k) => {
          const u = [V[0][k], V[1][k]];
          return (
            <g key={k}>
              {vec([E[k] * u[0], E[k] * u[1]], COL.green, 2)}
              {vec([-E[k] * u[0], -E[k] * u[1]], COL.green, 0.8, "2 2")}
            </g>
          );
        })}
      {s.step === 2 && (
        <>
          {vec(LA_B, COL.violet, 1.8)}
          {s.t >= 1 && vec(x, COL.cur, 1.8)}
          {s.t >= 2 && (
            <line x1={P(x)[0]} y1={P(x)[1]} x2={P(LA_B)[0]} y2={P(LA_B)[1]} stroke={COL.muted} strokeWidth={0.7} strokeDasharray="2 2" />
          )}
        </>
      )}
      {/* the matrix and the numbers */}
      <Txt x={tx} y={22} size={9} kind="math">H =</Txt>
      <path d={`M ${tx + 24} 10 l -2 0 l 0 26 l 2 0 M ${tx + 54} 10 l 2 0 l 0 26 l -2 0`} fill="none" stroke={COL.text} strokeWidth={0.8} />
      <Txt x={tx + 31} y={20} size={8.5} kind="math" pre>2   1</Txt>
      <Txt x={tx + 31} y={32} size={8.5} kind="math" pre>1   2</Txt>
      {v && hv && (
        <>
          <Txt x={tx} y={54} size={8} kind="math" color={eig ? COL.green : "#b45309"}>{`v = (${fmt(v[0])}, ${fmt(v[1])})`}</Txt>
          <Txt x={tx} y={68} size={8} kind="math" color={eig ? COL.green : COL.blue}>{`Hv = (${fmt(hv[0])}, ${fmt(hv[1])})`}</Txt>
          <Txt x={tx} y={86} size={8.5} kind="math" weight={600} color={eig ? COL.green : COL.muted}>
            {eig ? `Hv = ${fmt(lam, 0)}v` : "Hv ∦ v"}
          </Txt>
        </>
      )}
      {s.step === 1 && (
        <>
          <Txt x={tx} y={54} size={8} kind="math" color={COL.green}>{`E = (${fmt(E[0])}, ${fmt(E[1])})`}</Txt>
          <Txt x={tx} y={68} size={8} kind="math" color={COL.green}>{`ψ_1 = (${fmt(V[0][0])}, ${fmt(V[1][0])})`}</Txt>
          <Txt x={tx} y={82} size={8} kind="math" color={COL.green}>{`ψ_2 = (${fmt(V[0][1])}, ${fmt(V[1][1])})`}</Txt>
        </>
      )}
      {s.step === 2 && (
        <>
          <Txt x={tx} y={54} size={8} kind="math" color={COL.violet}>{`b = (${fmt(LA_B[0])}, ${fmt(LA_B[1])})`}</Txt>
          {s.t >= 1 && <Txt x={tx} y={68} size={8} kind="math" color="#b45309">{`x = (${fmt(x[0])}, ${fmt(x[1])})`}</Txt>}
          {s.t >= 2 && <Txt x={tx} y={86} size={8.5} kind="math" weight={600} color={COL.text}>Hx = b ✓</Txt>}
        </>
      )}
      <Txt x={tx} y={118} size={7.5} kind="mono" color={COL.blue}>
        {s.step === 2 ? "linalg.solve(H, b)" : s.step === 1 ? "linalg.eigh(H)" : "H @ v"}
      </Txt>
    </>
  );
}

// ---- 8. stencils / conv2d ------------------------------------------------------------------------------

function Stencils({ s }: PicProps) {
  const cs = 13;
  const ux = 18;
  const uy = 30;
  const kx = 132;
  const ky = 52;
  const ox = 180;
  const n = stU.length;
  const lap = conv2d(stU, LAP_KERNEL, 1);
  const visited = s.step === 0 ? stPath.slice(0, s.t + 1) : [];
  const cur = s.step === 0 ? stPath[s.t] : null;
  const all = s.step === 1;
  const isVisited = (i: number, j: number) => all || visited.some(([a, b]) => a === i && b === j);
  const at = (i: number, j: number) => (i < 0 || j < 0 || i >= n || j >= n ? 0 : stU[i][j]);
  return (
    <>
      <Txt x={ux + (n * cs) / 2} y={14} size={8.5} anchor="middle" kind="math">u</Txt>
      <Txt x={kx + 1.5 * cs} y={ky - 8} size={8} anchor="middle" kind="math">k · h^2</Txt>
      <Txt x={ox + (n * cs) / 2} y={14} size={8.5} anchor="middle" kind="math">∇^2u</Txt>
      {/* zero padding ring */}
      <rect x={ux - cs} y={uy - cs} width={(n + 2) * cs} height={(n + 2) * cs} fill="none" stroke={COL.idleLine} strokeDasharray="2 2" strokeWidth={0.7} />
      {stU.map((row, i) =>
        row.map((v, j) => (
          <Cell key={`u${i}${j}`} x={ux + j * cs} y={uy + i * cs} s={cs - 1} fill={blueScale(v / 4)} stroke={COL.idleLine} sw={0.4} label={v} labelColor={v >= 2 ? "white" : COL.text} size={7} />
        )),
      )}
      {all && <rect x={ux - 1} y={uy - 1} width={n * cs + 1} height={n * cs + 1} fill="none" stroke={COL.blue} strokeWidth={1.6} />}
      {cur && (
        <rect x={ux + (cur[1] - 1) * cs - 1} y={uy + (cur[0] - 1) * cs - 1} width={3 * cs + 1} height={3 * cs + 1} fill={COL.cur} fillOpacity={0.12} stroke={COL.cur} strokeWidth={1.6} />
      )}
      {LAP_KERNEL.map((row, a) =>
        row.map((v, b) => (
          <Cell
            key={`k${a}${b}`}
            x={kx + b * cs}
            y={ky + a * cs}
            s={cs - 1}
            fill={v < 0 ? COL.curLight : v > 0 ? COL.blueLight : "white"}
            stroke={cur || all ? COL.cur : COL.idleLine}
            sw={0.6}
            label={fmt(v)}
            size={7}
          />
        )),
      )}
      {stU.map((row, i) =>
        row.map((_, j) => {
          const on = isVisited(i, j);
          const now = cur && cur[0] === i && cur[1] === j;
          return (
            <Cell
              key={`o${i}${j}`}
              x={ox + j * cs}
              y={uy + i * cs}
              s={cs - 1}
              fill={on ? divScale(lap[i][j], 4) : "white"}
              stroke={now ? COL.cur : COL.idleLine}
              sw={now ? 1.6 : 0.4}
              label={on ? fmt(lap[i][j]) : undefined}
              size={7}
            />
          );
        }),
      )}
      <Txt x={W / 2} y={uy + n * cs + 26} size={8.5} anchor="middle" kind="math" color={cur ? COL.text : COL.muted}>
        {cur
          ? `${at(cur[0] - 1, cur[1])} + ${at(cur[0] + 1, cur[1])} + ${at(cur[0], cur[1] - 1)} + ${at(cur[0], cur[1] + 1)} − 4·${at(cur[0], cur[1])} = ${fmt(lap[cur[0]][cur[1]])}`
          : ""}
      </Txt>
      <Txt x={W / 2} y={uy + n * cs + 40} size={7.5} anchor="middle" kind="mono" color={COL.blue}>
        {all ? "conv2d(u, k, padding=1)  # all 49 at once" : ""}
      </Txt>
    </>
  );
}

// ---- 9. FFT --------------------------------------------------------------------------------------------

const PSI_K = dft(fftPsi0);
const K = fftk(PSI_K.length);
const PSI_T = freeEvolve(fftPsi0, FFT_T);
const phaseHue = (z: C) => `hsl(${((Math.atan2(z[1], z[0]) * 180) / Math.PI + 360) % 360},70%,50%)`;

function Fft({ s, T }: PicProps) {
  const top = frame(30, 12, 240, 44, [-16, 15], [-1, 1]);
  const bot = frame(30, 86, 240, 40, [-Math.PI, Math.PI], [0, 4.5]);
  const tNow = s.step === 2 ? (FFT_T * (s.t + 1)) / s.n : s.step === 3 ? FFT_T : 0;
  const spec = s.step >= 1 ? PSI_K.map((v, j) => cmul(v, cexp(-0.5 * K[j] ** 2 * tNow))) : null;
  const psi = s.step === 3 ? PSI_T : fftPsi0;
  const line = (p: C[], f: (z: C) => number) => poly(fftX.map((x, j) => [top.X(x), top.Y(f(p[j]))]));
  const ks = fftShift(K);
  const sp = spec ? fftShift(spec) : null;
  const bw = 240 / 32;
  return (
    <>
      <Axes x0={30} y0={12} w={240} h={44} xl={[-16, 15]} yl={[-1, 1]} xlab="x (a.u.)" ylab="ψ (a.u.)" xt={[-16, 0, 15]} yt={[-1, 0, 1]} />
      {s.step === 3 && <polyline points={line(fftPsi0, (z) => z[0])} fill="none" stroke={COL.idleLine} strokeWidth={0.9} strokeDasharray="2 2" />}
      <g opacity={s.step === 1 || s.step === 2 ? 0.35 : 1}>
        <polyline points={line(psi, cabs)} fill="none" stroke={COL.muted} strokeWidth={0.7} strokeDasharray="3 1.5" />
        <polyline points={line(psi, (z) => z[0])} fill="none" stroke={COL.blue} strokeWidth={1.3} />
      </g>
      <Txt x={262} y={22} size={7} anchor="end" kind="math" color={COL.blue}>Re ψ</Txt>
      <Txt x={262} y={31} size={7} anchor="end" kind="math" color={COL.muted}>|ψ|</Txt>
      <Txt x={14} y={78} size={8} kind="mono" color={s.step === 1 ? COL.blue : s.step === 3 ? COL.green : COL.muted}>
        {s.step === 1 ? "↓ fft" : s.step === 3 ? "↑ ifft" : s.step === 2 ? "" : ""}
      </Txt>
      {s.step === 2 && (
        <Txt x={112} y={78} size={8.5} kind="math" color={COL.text}>{`× e^{−ik²t/2},  t = ${fmt(tNow, 1)}`}</Txt>
      )}
      <Axes x0={30} y0={86} w={240} h={40} xl={[-Math.PI, Math.PI]} yl={[0, 4.5]} xlab="k (a.u.)" ylab="" xt={[]} yt={[0, 4]} />
      <Txt x={bot.X(-Math.PI)} y={134} size={6.5} anchor="middle" kind="math" color={COL.muted}>−π</Txt>
      <Txt x={bot.X(0)} y={134} size={6.5} anchor="middle" kind="math" color={COL.muted}>0</Txt>
      <Txt x={bot.X(Math.PI)} y={134} size={6.5} anchor="middle" kind="math" color={COL.muted}>π</Txt>
      <Txt x={32} y={83} size={7.5} kind="math" color={COL.muted}>|ψ_k| (a.u.)</Txt>
      {sp &&
        sp.map((z, j) => {
          const h = 126 - bot.Y(cabs(z));
          return <rect key={j} x={bot.X(ks[j]) - bw / 2 + 0.5} y={126 - h} width={bw - 1} height={h} fill={phaseHue(z)} />;
        })}
      {/* phase colour key, -pi .. pi */}
      <defs>
        <linearGradient id="fft-phase" x1="0" x2="1">
          {[0, 60, 120, 180, 240, 300, 360].map((d) => (
            <stop key={d} offset={d / 360} stopColor={`hsl(${(d + 180) % 360},70%,50%)`} />
          ))}
        </linearGradient>
      </defs>
      <rect x={182} y={146} width={60} height={5} fill="url(#fft-phase)" />
      <Txt x={180} y={151} size={6.5} anchor="end" kind="math" color={COL.muted}>−π</Txt>
      <Txt x={244} y={151} size={6.5} kind="math" color={COL.muted}>π (rad)</Txt>
      <Txt x={30} y={151} size={7} color={COL.muted}>
        {T("skills.anim.fft.phase")}
      </Txt>
    </>
  );
}

// ---- 10. autograd --------------------------------------------------------------------------------------

function Autograd({ s }: PicProps) {
  const { r, phi } = agForward(AG.x);
  const gr = agBackward(AG.x);
  const nodes = [
    { x: 36, label: "x", val: `(${AG.x[0]}, ${AG.x[1]})`, grad: `(${fmt(gr.dx[0], 3)}, ${fmt(gr.dx[1], 3)})`, gl: "∂φ/∂x" },
    { x: 136, label: "r", val: fmt(r), grad: fmt(gr.dr), gl: "∂φ/∂r" },
    { x: 236, label: "φ", val: fmt(phi), grad: "1", gl: "∂φ/∂φ" },
  ];
  const ny = 26;
  const fwd = (i: number) => s.step > 0 || s.t >= i;
  const bwd = (i: number) => s.step > 1 || (s.step === 1 && s.t >= 2 - i);
  const ops = ["|x|", "−GM/r"];
  // force picture
  const c: [number, number] = [52, 128];
  const sc = 4.5;
  const F = gr.dx.map((v) => -v);
  return (
    <>
      {ops.map((o, i) => (
        <g key={o}>
          <Arrow x1={nodes[i].x + 24} y1={ny} x2={nodes[i + 1].x - 24} y2={ny} color={fwd(i + 1) ? COL.blue : COL.idleLine} w={1.3} />
          <Txt x={(nodes[i].x + nodes[i + 1].x) / 2} y={ny - 6} size={7.5} anchor="middle" kind="math" color={COL.muted}>
            {o}
          </Txt>
          {bwd(i) && (
            <path
              d={`M ${nodes[i + 1].x - 6} ${ny + 34} Q ${(nodes[i].x + nodes[i + 1].x) / 2} ${ny + 50} ${nodes[i].x + 8} ${ny + 34}`}
              fill="none"
              stroke={COL.cur}
              strokeWidth={1.3}
              markerEnd="url(#ag-head)"
            />
          )}
        </g>
      ))}
      <defs>
        <marker id="ag-head" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 6 3 L 0 6 Z" fill={COL.cur} />
        </marker>
      </defs>
      {nodes.map((nd, i) => (
        <g key={nd.label}>
          <rect x={nd.x - 22} y={ny - 11} width={44} height={22} rx={11} fill={fwd(i) ? COL.blueLight : "white"} stroke={fwd(i) ? COL.blue : COL.idleLine} strokeWidth={s.step === 0 && s.t === i ? 1.8 : 0.9} />
          <Txt x={nd.x} y={ny + 3.5} size={10} anchor="middle" kind="math">
            {nd.label}
          </Txt>
          {fwd(i) && (
            <Txt x={nd.x} y={ny + 22} size={8} anchor="middle" kind="math" color={COL.blue}>
              {nd.val}
            </Txt>
          )}
          {bwd(i) && (
            <>
              <Txt x={nd.x} y={ny + 58} size={7.5} anchor="middle" kind="math" color="#b45309">
                {nd.gl}
              </Txt>
              <Txt x={nd.x} y={ny + 70} size={8} anchor="middle" kind="math" color="#b45309">
                {nd.grad}
              </Txt>
            </>
          )}
        </g>
      ))}
      {/* the force: minus the gradient, pointing at the mass */}
      <g opacity={s.step === 2 ? 1 : 0.25}>
        <circle cx={c[0] - AG.x[0] * sc} cy={c[1] + AG.x[1] * sc} r={3} fill={COL.text} />
        <Txt x={c[0] - AG.x[0] * sc - 5} y={c[1] + AG.x[1] * sc + 3} size={7} anchor="end" kind="math" color={COL.muted}>M</Txt>
        <circle cx={c[0]} cy={c[1]} r={2.5} fill={COL.blue} />
        {s.step === 2 && <Arrow x1={c[0]} y1={c[1]} x2={c[0] + F[0] * 400 * sc / 5} y2={c[1] - F[1] * 400 * sc / 5} color={COL.green} w={1.8} />}
      </g>
      <Txt x={84} y={124} size={8} kind="math" color={s.step === 2 ? COL.green : COL.idleLine}>
        {`F = −∇φ = (${fmt(F[0], 3)}, ${fmt(F[1], 3)})`}
      </Txt>
      <Txt x={84} y={138} size={7.5} kind="mono" color={COL.blue}>
        {s.step === 0 ? "phi = -G*M / x.norm()" : "autograd.grad(phi, x)"}
      </Txt>
      <Txt x={84} y={150} size={7} kind="math" color={COL.muted}>GM = 1</Txt>
    </>
  );
}

// ---- 11. complex phases --------------------------------------------------------------------------------

const CX_P0 = cxPsi(0).prob;

function Complex({ s }: PicProps) {
  const t = cxTime(s.t);
  const ph = cxPhased(t);
  const { prob } = cxPsi(t);
  const cmax = Math.max(...cxC);
  const R = 15;
  const { X, Y } = frame(30, 82, 236, 46, [-5, 5], [0, 0.7]);
  return (
    <>
      {ph.map((z, n) => {
        const cx = 34 + n * 58;
        const cy = 30;
        const len = (R * cxC[n]) / cmax;
        return (
          <g key={n}>
            <circle cx={cx} cy={cy} r={R} fill="white" stroke={COL.idleLine} strokeWidth={0.8} />
            <line x1={cx - R - 2} y1={cy} x2={cx + R + 2} y2={cy} stroke={COL.idle} strokeWidth={0.6} />
            <Arrow x1={cx} y1={cy} x2={cx + (len * z[0]) / cxC[n]} y2={cy - (len * z[1]) / cxC[n]} color={[COL.blue, COL.cur, COL.green][n]} w={1.6} />
            <Txt x={cx} y={cy + R + 10} size={7.5} anchor="middle" kind="math">{`E_${n} = ${cxE(n)}`}</Txt>
          </g>
        );
      })}
      <Txt x={196} y={22} size={8.5} kind="math">{`t = ${fmt(t, 2)}`}</Txt>
      <Txt x={196} y={36} size={7.5} kind="math" color={COL.muted}>{"c_n exp(−iE_n t)"}</Txt>
      <Txt x={196} y={50} size={7} kind="mono" color={COL.blue}>c*exp(-1j*E*t)</Txt>
      <Axes x0={30} y0={82} w={236} h={46} xl={[-5, 5]} yl={[0, 0.7]} xlab="x (a.u.)" ylab="|ψ|^2 (a.u.)" xt={[-5, 0, 5]} yt={[0, 0.6]} />
      <polyline points={poly(cxX.map((x, j) => [X(x), Y(CX_P0[j])]))} fill="none" stroke={COL.idleLine} strokeWidth={0.9} strokeDasharray="2 2" />
      <polyline points={poly(cxX.map((x, j) => [X(x), Y(prob[j])]))} fill={COL.blueLight} fillOpacity={0.5} stroke={COL.blue} strokeWidth={1.3} />
    </>
  );
}

const PICTURES: Record<TechCase, { draw: (p: PicProps) => ReactNode; h: number }> = {
  tdd: { draw: (p) => <Tdd {...p} />, h: 142 },
  tensors: { draw: (p) => <Tensors {...p} />, h: 156 },
  broadcasting: { draw: (p) => <Broadcasting {...p} />, h: 124 },
  einsum: { draw: (p) => <Einsum {...p} />, h: 172 },
  batching: { draw: (p) => <Batching {...p} />, h: 146 },
  rk4: { draw: (p) => <Rk4 {...p} />, h: 144 },
  linalg: { draw: (p) => <Linalg {...p} />, h: 140 },
  stencils: { draw: (p) => <Stencils {...p} />, h: 166 },
  fft: { draw: (p) => <Fft {...p} />, h: 156 },
  autograd: { draw: (p) => <Autograd {...p} />, h: 156 },
  complex: { draw: (p) => <Complex {...p} />, h: 146 },
};

/**
 * One animation under a Techniques card: an SVG picture of what the operation does to shapes and values, the
 * list of its steps (the current one highlighted), a caption and the shared controls.
 */
export default function TechniqueAnim({ id }: { id: TechCase }) {
  const T = usePageText();
  const cycle = totalFrames(id) + HOLD;
  const clock = useAnimClock(cycle, PLAN[id].tick);
  const { root, g } = clock;
  const s = stageAt(id, g);
  const n = stepCount(id);
  const pic = PICTURES[id];
  const caption = T(`skills.anim.${id}.caption` as PageKey);
  return (
    <div ref={root} className="mt-4" data-testid={`tech-anim-${id}`}>
      <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <figcaption className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">
          <span>{T("skills.anim.title")}</span>
          {n > 1 && (
            <span className="font-mono tabular-nums text-blue-700" data-testid={`tech-anim-step-${id}`}>
              {s.step + 1}/{n}
            </span>
          )}
        </figcaption>
        <div className="grid md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <svg
            viewBox={`0 0 ${W} ${pic.h}`}
            className="mx-auto block h-auto w-full max-w-[540px] p-1"
            role="img"
            aria-label={caption.replace(/[$`\\]/g, "")}
          >
            {pic.draw({ g, s, T })}
          </svg>
          <ol className="space-y-1 border-t border-gray-100 p-3 text-xs leading-snug md:border-t-0 md:border-l">
            {Array.from({ length: n }, (_, k) => (
              <li
                key={k}
                className={`flex gap-2 rounded px-1.5 py-1 ${k === s.step ? "bg-blue-50 text-gray-900" : "text-gray-500"}`}
                aria-current={k === s.step ? "step" : undefined}
              >
                <span
                  className={`mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${
                    k === s.step ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-600"
                  }`}
                >
                  {k + 1}
                </span>
                <span>
                  <RichText text={T(`skills.anim.${id}.s${k}` as PageKey)} />
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
