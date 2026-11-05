import { useEffect, useRef, useState, type ReactNode } from "react";
import { usePageText, type PageKey } from "../content/pages";
import {
  DIFF_SHIFTS,
  PEND,
  SIZE,
  diffU0,
  diffusionLoop,
  diffusionTensor,
  dotData,
  dotLoopSum,
  loopCount,
  loopFrames,
  matmul,
  mcData,
  mmData,
  mmIJK,
  mmLoop,
  mmTile,
  pairData,
  pairDistances,
  pairIJ,
  pendLoop,
  pendTraj,
  piEstimate,
  tensorCount,
  tensorFrames,
  type AnimCase,
} from "../loopAnim";
import { RichText } from "./Tex";

/** Site colours: gray idle, blue tensor / done (as the chart's CPU bars), amber for the loop's current element. */
const COL = {
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
};

const W = 240; // SVG user units; scaled to the panel width
const HOLD = 12; // ticks the finished picture stays before the cycle restarts
const SPEEDS = [0.5, 1, 2, 4];

/** Blue sequential scale for a value in [0, 1]. */
function blueScale(v: number): string {
  const t = Math.min(Math.max(v, 0), 1);
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(239, 30)},${lerp(246, 64)},${lerp(255, 175)})`; // blue-50 -> blue-800
}

type Kind = "sans" | "mono" | "math";

/** Font of a label: the page's sans text, the CodeBlock code font, or KaTeX's fonts for math. */
const FAMILY: Record<Kind, string> = { sans: "inherit", mono: "var(--font-code)", math: "KaTeX_Main, serif" };

/**
 * A tiny TeX-like renderer for SVG labels, so they look like the $...$ math in the card text: letters in
 * KaTeX_Math italic, digits and operators upright in KaTeX_Main, `_i` / `_{ij}` subscripts, `^2` superscripts.
 */
function mathSpans(src: string): ReactNode[] {
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
          {ch === " " ? " " : ch}
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

function Txt({ x, y, children, size = 8, color = COL.text, anchor = "start", weight, kind = "sans" }: {
  x: number;
  y: number;
  children: ReactNode;
  size?: number;
  color?: string;
  anchor?: "start" | "middle" | "end";
  weight?: number;
  kind?: Kind;
}) {
  const body = kind === "math" ? mathSpans(Array.isArray(children) ? children.join("") : String(children)) : children;
  return (
    <text x={x} y={y} fontSize={size} fill={color} textAnchor={anchor} style={{ fontFamily: FAMILY[kind] }} fontWeight={weight}>
      {body}
    </text>
  );
}

function Cell({ x, y, s, fill, stroke = COL.idleLine, sw = 0.75, label, labelColor = COL.text, size = 8 }: {
  x: number;
  y: number;
  s: number;
  fill: string;
  stroke?: string;
  sw?: number;
  label?: string | number;
  labelColor?: string;
  size?: number;
}) {
  return (
    <g>
      <rect x={x} y={y} width={s} height={s} rx={1.5} fill={fill} stroke={stroke} strokeWidth={sw} />
      {label != null && (
        <Txt x={x + s / 2} y={y + s / 2 + size * 0.36} size={size} anchor="middle" color={labelColor}>
          {label}
        </Txt>
      )}
    </g>
  );
}

// ---- one picture per case; `f` = ticks done on that side ----------------------------------------------

function Dot({ side, f }: { side: "loop" | "tensor"; f: number }) {
  const { x, y } = dotData;
  const s = 15;
  const gap = 2.5;
  const x0 = 22;
  const cx = (i: number) => x0 + i * (s + gap);
  const cur = side === "loop" ? f - 1 : -1; // pass just made
  const state = (i: number) =>
    side === "loop" ? (i === cur ? "cur" : i < cur ? "done" : "idle") : f >= 1 ? (f === 1 ? "flash" : "done") : "idle";
  const fillOf = (st: string) => (st === "cur" ? COL.curLight : st === "flash" ? COL.blueLight : st === "done" ? COL.done : COL.idle);
  const lineOf = (st: string) => (st === "cur" ? COL.cur : st === "flash" ? COL.blue : st === "done" ? COL.doneLine : COL.idleLine);
  const sum = side === "loop" ? dotLoopSum(x, y, f) : f >= 2 ? dotLoopSum(x, y, SIZE.dot) : null;
  const sx = W / 2;
  const sy = 104;
  return (
    <>
      <Txt x={6} y={18} kind="math">x</Txt>
      <Txt x={6} y={38} kind="math">y</Txt>
      <Txt x={1} y={67} kind="math" color={COL.muted}>x·y</Txt>
      {x.map((_, i) => {
        const st = state(i);
        const showProd = side === "loop" ? i <= cur : f >= 1;
        return (
          <g key={i}>
            <Cell x={cx(i)} y={7} s={s} fill={fillOf(st)} stroke={lineOf(st)} sw={st === "cur" ? 1.5 : 0.75} label={x[i]} />
            <Cell x={cx(i)} y={27} s={s} fill={fillOf(st)} stroke={lineOf(st)} sw={st === "cur" ? 1.5 : 0.75} label={y[i]} />
            <Cell
              x={cx(i)}
              y={56}
              s={s}
              fill={showProd ? fillOf(st) : "white"}
              stroke={showProd ? lineOf(st) : COL.idleLine}
              sw={st === "cur" ? 1.5 : 0.75}
              label={showProd ? x[i] * y[i] : undefined}
            />
            {/* tensor: one reduction into s (fan-in); loop: only the current product flows into s */}
            {((side === "tensor" && f >= 2) || (side === "loop" && i === cur)) && (
              <line
                x1={cx(i) + s / 2}
                y1={56 + s}
                x2={sx}
                y2={sy - 12}
                stroke={side === "loop" ? COL.cur : COL.blue}
                strokeWidth={side === "loop" ? 1.4 : 0.8}
                opacity={side === "loop" ? 1 : 0.7}
              />
            )}
          </g>
        );
      })}
      <rect
        x={sx - 34}
        y={sy - 12}
        width={68}
        height={18}
        rx={3}
        fill={sum == null ? COL.idle : side === "loop" && f < SIZE.dot ? "#fffbeb" : COL.blueLight}
        stroke={sum == null ? COL.idleLine : side === "loop" && f < SIZE.dot ? COL.cur : COL.blue}
      />
      <Txt x={sx} y={sy + 1} anchor="middle" kind="math">
        s = {sum ?? "?"}
      </Txt>
      <Txt x={W - 4} y={sy + 22} anchor="end" kind="mono" color={side === "loop" ? COL.cur : COL.blue}>
        {side === "loop" ? `i = ${cur >= 0 ? cur : "–"}` : "torch.dot(x, y)"}
      </Txt>
    </>
  );
}

function Pairwise({ side, f }: { side: "loop" | "tensor"; f: number }) {
  const x = pairData;
  const n = SIZE.pairwise;
  const D = pairDistances(x);
  const s = 12;
  const gx = 46;
  const gy = 30;
  const k = side === "loop" ? f - 1 : -1;
  const [ci, cj] = k >= 0 ? pairIJ(k) : [-1, -1];
  const vecCell = (i: number, which: "col" | "row") => {
    const hot = side === "loop" && (which === "col" ? i === ci : i === cj);
    const on = side === "tensor" && f >= 1;
    return { fill: hot ? COL.curLight : on ? COL.blueLight : COL.idle, stroke: hot ? COL.cur : on ? COL.blue : COL.idleLine };
  };
  return (
    <>
      <Txt x={gx - 30} y={gy - 6} kind="math" color={COL.muted}>x_i</Txt>
      <Txt x={gx - 12} y={gy - 18} kind="math" color={COL.muted}>x_j</Txt>
      {x.map((_, i) => (
        <g key={i}>
          <Cell x={gx - 18} y={gy + i * s} s={s - 1} {...vecCell(i, "col")} />
          <Cell x={gx + i * s} y={gy - 16} s={s - 1} {...vecCell(i, "row")} />
        </g>
      ))}
      {Array.from({ length: n * n }, (_, p) => {
        const [i, j] = pairIJ(p);
        let fill = "white";
        let stroke = COL.idleLine;
        let sw = 0.5;
        if (side === "loop") {
          if (p === k) [fill, stroke, sw] = [COL.cur, COL.cur, 1.2];
          else if (p < k) fill = blueScale(D[i][j] / 9);
        } else if (f >= 2) fill = blueScale(D[i][j] / 9);
        return <Cell key={p} x={gx + j * s} y={gy + i * s} s={s - 1} fill={fill} stroke={stroke} sw={sw} />;
      })}
      {/* broadcasting: the column and the row stretched over the whole grid in one go */}
      {side === "tensor" && f === 1 && (
        <g opacity={0.55}>
          {x.map((_, i) => (
            <g key={i}>
              <rect x={gx} y={gy + i * s + 2} width={n * s - 1} height={s - 5} fill={COL.blueLight} stroke={COL.blue} strokeWidth={0.4} />
              <rect x={gx + i * s + 2} y={gy} width={s - 5} height={n * s - 1} fill="none" stroke={COL.blue} strokeWidth={0.6} />
            </g>
          ))}
        </g>
      )}
      <Txt x={gx + n * s + 8} y={gy + 10} kind="math" color={COL.muted}>{"d_{ij} = |x_i − x_j|"}</Txt>
      <Txt x={gx + n * s + 8} y={gy + 28} kind="math" color={COL.muted}>{`N = ${n}`}</Txt>
      <Txt x={gx + n * s + 8} y={gy + 44} kind="math" color={COL.muted}>{`N^2 = ${n * n}`}</Txt>
      <Txt x={side === "loop" ? gx + n * s + 8 : gx - 30} y={side === "loop" ? gy + 68 : gy + n * s + 13} kind="mono" color={side === "loop" ? COL.cur : COL.blue}>
        {side === "loop" ? (k >= 0 ? `i = ${ci}, j = ${cj}` : "") : f >= 1 ? "x[:, None] - x[None, :]" : ""}
      </Txt>
    </>
  );
}

function Matmul({ side, f }: { side: "loop" | "tensor"; f: number }) {
  const { A, B } = mmData;
  const n = SIZE.matmul;
  const Cfull = matmul(A, B);
  const s = 14;
  const ax = 8;
  const ay = 72;
  const bx = 70;
  const by = 8;
  const cx = bx;
  const cy = ay;
  const loopC = side === "loop" ? mmLoop(A, B, f) : null;
  const [i, j, k] = side === "loop" && f >= 1 && f <= n ** 3 ? mmIJK(f - 1) : [-1, -1, -1];
  const tile = side === "tensor" ? f - 1 : -1; // tile being filled
  const inTile = (r: number, c: number) => side === "tensor" && f >= 1 && f <= 4 && mmTile(r, c) === tile;
  const tileRows = (r: number) => side === "tensor" && f >= 1 && f <= 4 && Math.floor(r / (n / 2)) === Math.floor(tile / 2);
  const tileCols = (c: number) => side === "tensor" && f >= 1 && f <= 4 && Math.floor(c / (n / 2)) === tile % 2;
  return (
    <>
      <Txt x={ax + (n * s) / 2} y={ay - 6} anchor="middle" kind="math">A</Txt>
      <Txt x={bx - 8} y={by + (n * s) / 2 + 3} anchor="end" kind="math">B</Txt>
      <Txt x={cx + n * s + 6} y={cy + (n * s) / 2 + 3} kind="math">C = AB</Txt>
      {A.map((row, r) =>
        row.map((v, c) => {
          let fill = COL.idle;
          let stroke = COL.idleLine;
          if (side === "loop" && r === i) [fill, stroke] = c === k ? [COL.cur, COL.cur] : [COL.curLight, COL.idleLine];
          if (tileRows(r)) fill = COL.blueLight;
          return <Cell key={`a${r}${c}`} x={ax + c * s} y={ay + r * s} s={s - 1} fill={fill} stroke={stroke} label={v} />;
        }),
      )}
      {B.map((row, r) =>
        row.map((v, c) => {
          let fill = COL.idle;
          let stroke = COL.idleLine;
          if (side === "loop" && c === j) [fill, stroke] = r === k ? [COL.cur, COL.cur] : [COL.curLight, COL.idleLine];
          if (tileCols(c)) fill = COL.blueLight;
          return <Cell key={`b${r}${c}`} x={bx + c * s} y={by + r * s} s={s - 1} fill={fill} stroke={stroke} label={v} />;
        }),
      )}
      {Cfull.map((row, r) =>
        row.map((v, c) => {
          let fill = "white";
          let stroke = COL.idleLine;
          let label: number | undefined;
          if (side === "loop") {
            const val = loopC![r][c];
            if (r === i && c === j) [fill, stroke, label] = [COL.curLight, COL.cur, val ?? undefined];
            else if (val != null) [fill, label] = [COL.done, val];
          } else if (inTile(r, c)) [fill, stroke, label] = [COL.blue, COL.blue, v];
          else if (f >= 1 && mmTile(r, c) < tile) [fill, label] = [COL.done, v];
          else if (f > 4) [fill, label] = [COL.done, v];
          return (
            <Cell
              key={`c${r}${c}`}
              x={cx + c * s}
              y={cy + r * s}
              s={s - 1}
              fill={fill}
              stroke={stroke}
              sw={r === i && c === j ? 1.4 : 0.75}
              label={label}
              labelColor={fill === COL.blue ? "white" : COL.text}
            />
          );
        }),
      )}
      <Txt x={cx + n * s + 6} y={cy + n * s + 4} kind="mono" color={side === "loop" ? COL.cur : COL.blue}>
        {side === "loop"
          ? i >= 0
            ? `i, j, k = ${i}, ${j}, ${k}`
            : ""
          : f >= 1 && f <= 4
            ? `A @ B  [${tile + 1}/4]`
            : f > 4
              ? "A @ B"
              : ""}
      </Txt>
    </>
  );
}

function Pendulum({ side, f }: { side: "loop" | "tensor"; f: number }) {
  const B = SIZE.pendB;
  const T = SIZE.pendT;
  const L = 44;
  const dx = 30;
  const x0 = (W - dx * (B - 1)) / 2;
  const py = 16;
  const lp = side === "loop" ? pendLoop(f) : null;
  const tt = side === "loop" ? lp!.t : Math.min(f, T);
  return (
    <>
      <line x1={x0 - 18} y1={py} x2={x0 + dx * (B - 1) + 18} y2={py} stroke={COL.muted} strokeWidth={1.5} />
      {Array.from({ length: B }, (_, b) => {
        const step = side === "loop" ? lp!.steps[b] : Math.min(f, T);
        const th = pendTraj[step][b][0];
        const px = x0 + b * dx;
        const bx = px + L * Math.sin(th);
        const by = py + L * Math.cos(th);
        const cur = side === "loop" && lp!.current === b;
        const ahead = side === "loop" ? step > tt : f >= 1;
        const color = cur ? COL.cur : side === "tensor" ? (f >= 1 && f <= T ? COL.blue : COL.doneLine) : ahead ? COL.doneLine : "#9ca3af";
        return (
          <g key={b}>
            <line x1={px} y1={py} x2={bx} y2={by} stroke={cur ? COL.cur : COL.text} strokeWidth={cur ? 1.6 : 1} />
            <circle cx={px} cy={py} r={1.5} fill={COL.text} />
            <circle cx={bx} cy={by} r={cur ? 6 : 5} fill={color} stroke={cur ? "#b45309" : "white"} strokeWidth={1} />
            <Txt x={px} y={py + L + 22} anchor="middle" kind="mono" color={COL.muted}>{b}</Txt>
          </g>
        );
      })}
      <Txt x={8} y={py + L + 38} kind="math">
        t = {tt}/{T}
      </Txt>
      <Txt x={W - 6} y={py + L + 38} anchor="end" kind="mono" color={side === "loop" ? COL.cur : COL.blue}>
        {side === "loop"
          ? lp!.current >= 0
            ? `b = ${lp!.current}`
            : ""
          : f >= 1 && f <= T
            ? "y: (B, 2)"
            : ""}
      </Txt>
      <Txt x={8} y={py + L + 52} kind="mono" color={COL.muted}>
        {`dt = ${PEND.dt}`}
      </Txt>
    </>
  );
}

function Diffusion({ side, f }: { side: "loop" | "tensor"; f: number }) {
  const n = SIZE.diffusion;
  const s = 13;
  const gx = 8;
  const gy = 8;
  const m = n - 2;
  const k = side === "loop" ? f - 1 : -1;
  const ci = k >= 0 && k < m * m ? 1 + Math.floor(k / m) : -1;
  const cj = k >= 0 && k < m * m ? 1 + (k % m) : -1;
  const u = side === "loop" ? diffusionLoop(diffU0, f) : f >= 5 ? diffusionTensor(diffU0) : diffU0;
  const updated = (i: number, j: number) =>
    i > 0 && j > 0 && i < n - 1 && j < n - 1 && (side === "loop" ? (i - 1) * m + (j - 1) < f : f >= 5);
  const shift = side === "tensor" && f >= 1 && f <= 4 ? DIFF_SHIFTS[f - 1] : null;
  const nb = (i: number, j: number) => ci >= 0 && Math.abs(i - ci) + Math.abs(j - cj) === 1;
  const tx = gx + n * s + 10;
  return (
    <>
      {u.map((row, i) =>
        row.map((v, j) => {
          const isCur = i === ci && j === cj;
          const stroke = isCur || nb(i, j) ? COL.cur : updated(i, j) ? COL.blue : COL.idleLine;
          return (
            <rect
              key={`${i}-${j}`}
              x={gx + j * s}
              y={gy + i * s}
              width={s - 1}
              height={s - 1}
              rx={1}
              fill={blueScale(v)}
              stroke={stroke}
              strokeWidth={isCur ? 2 : nb(i, j) ? 1.3 : updated(i, j) ? 0.9 : 0.5}
              strokeDasharray={nb(i, j) ? "2 1" : undefined}
            />
          );
        }),
      )}
      {/* the fixed boundary */}
      <rect x={gx - 1} y={gy - 1} width={n * s + 1} height={n * s + 1} fill="none" stroke={COL.muted} strokeWidth={0.6} />
      <rect x={gx + s - 1} y={gy + s - 1} width={m * s + 1} height={m * s + 1} fill="none" stroke={COL.muted} strokeWidth={0.6} strokeDasharray="3 2" />
      {shift && (
        <rect
          x={gx + (1 + shift[1]) * s - 1}
          y={gy + (1 + shift[0]) * s - 1}
          width={m * s + 1}
          height={m * s + 1}
          fill={COL.blue}
          fillOpacity={0.15}
          stroke={COL.blue}
          strokeWidth={1.6}
        />
      )}
      <Txt x={tx} y={gy + 10} kind="math" color={COL.muted}>a = 0.2</Txt>
      <Txt x={tx} y={gy + 26} kind="math" color={COL.muted}>{`n = ${n}`}</Txt>
      <Txt x={tx} y={gy + 48} kind="mono" color={side === "loop" ? COL.cur : COL.blue}>
        {side === "loop" ? (ci >= 0 ? `i = ${ci}, j = ${cj}` : "") : shift ? shift[2] : f >= 5 ? "new = c + a·lap" : ""}
      </Txt>
      {side === "tensor" && f >= 1 && f <= 5 && (
        <g>
          {DIFF_SHIFTS.map((_, q) => (
            <rect key={q} x={tx + q * 13} y={gy + 58} width={10} height={10} rx={1.5} fill={q < f ? COL.blue : COL.idle} stroke={COL.idleLine} strokeWidth={0.5} />
          ))}
        </g>
      )}
    </>
  );
}

function McPi({ side, f }: { side: "loop" | "tensor"; f: number }) {
  const sz = 104;
  const x0 = 10;
  const y0 = 8;
  const N = SIZE.mc;
  const shown = side === "loop" ? Math.min(f, N) : f >= 1 ? N : 0;
  const est = piEstimate(mcData, shown);
  const tx = x0 + sz + 14;
  return (
    <>
      <rect x={x0} y={y0} width={sz} height={sz} fill="white" stroke={COL.muted} strokeWidth={0.8} />
      <path d={`M ${x0} ${y0} A ${sz} ${sz} 0 0 1 ${x0 + sz} ${y0 + sz} L ${x0} ${y0 + sz} Z`} fill={COL.blueLight} fillOpacity={0.5} stroke={COL.doneLine} strokeWidth={0.8} />
      {mcData.slice(0, shown).map(([px, py], i) => {
        const inside = px * px + py * py <= 1;
        const cur = side === "loop" && i === shown - 1 && f <= N;
        return (
          <circle
            key={i}
            cx={x0 + px * sz}
            cy={y0 + sz - py * sz}
            r={cur ? 3.5 : 1.8}
            fill={cur ? COL.cur : inside ? (side === "tensor" && f === 1 ? COL.blue : "#1d4ed8") : "#9ca3af"}
            stroke={cur ? "#b45309" : "none"}
          />
        );
      })}
      <Txt x={tx} y={y0 + 14} kind="math" color={COL.muted}>x^2 + y^2 ≤ 1</Txt>
      <Txt x={tx} y={y0 + 34} kind="math">{`N = ${shown}`}</Txt>
      <Txt x={tx} y={y0 + 54} size={9} kind="math" color={shown ? COL.text : COL.muted}>
        π ≈ {shown ? est.toFixed(3) : "?"}
      </Txt>
      <Txt x={tx} y={y0 + 70} kind="math" color={COL.muted}>{`π = ${Math.PI.toFixed(3)}…`}</Txt>
      <Txt x={tx} y={y0 + 90} kind="mono" color={side === "loop" ? COL.cur : COL.blue}>
        {side === "loop" ? (shown && f <= N ? `k = ${shown - 1}` : "") : shown ? "(N,) bool" : ""}
      </Txt>
    </>
  );
}

const PICTURES: Record<AnimCase, { draw: (p: { side: "loop" | "tensor"; f: number }) => ReactNode; h: number }> = {
  dot: { draw: (p) => <Dot {...p} />, h: 132 },
  pairwise: { draw: (p) => <Pairwise {...p} />, h: 146 },
  matmul: { draw: (p) => <Matmul {...p} />, h: 140 },
  pendulum: { draw: (p) => <Pendulum {...p} />, h: 120 },
  diffusion: { draw: (p) => <Diffusion {...p} />, h: 120 },
  mc_pi: { draw: (p) => <McPi {...p} />, h: 120 },
};

/** Base tick length (ms at 1x): every loop run takes a few seconds, whatever its number of passes. */
const tickMs = (c: AnimCase) => Math.min(320, Math.max(90, 8000 / loopFrames(c)));

const btn =
  "rounded-md border border-gray-300 bg-white px-2.5 py-1 text-xs text-gray-700 hover:bg-gray-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";

/**
 * Loop vs tensor, side by side: the loop side takes one tick per element, the tensor side one vectorised
 * call. Loops forever; starts paused under prefers-reduced-motion; pauses while off-screen.
 */
export default function LoopVsTensorAnim({ id }: { id: AnimCase }) {
  const T = usePageText();
  const root = useRef<HTMLDivElement>(null);
  const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const [playing, setPlaying] = useState(!reduced);
  const [visible, setVisible] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [g, setG] = useState(0);
  const cycle = loopFrames(id) + HOLD;

  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return setVisible(true);
    const ob = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0 });
    ob.observe(el);
    return () => ob.disconnect();
  }, []);

  useEffect(() => {
    if (!playing || !visible) return;
    const h = window.setInterval(() => setG((v) => (v + 1) % (cycle + 1)), tickMs(id) / speed);
    return () => window.clearInterval(h);
  }, [playing, visible, speed, id, cycle]);

  const lf = Math.min(g, loopFrames(id));
  const tf = Math.min(g, tensorFrames(id));
  const pic = PICTURES[id];
  const panel = (side: "loop" | "tensor", f: number, count: string, done: boolean) => (
    <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      <figcaption className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-3 py-1.5 font-mono text-xs text-gray-600">
        <span>{T(side === "loop" ? "loops.loop" : "loops.tensor")}</span>
        {done && <span className={side === "loop" ? "text-gray-500" : "text-blue-700"}>✓</span>}
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${pic.h}`}
        className="block h-auto w-full"
        role="img"
        aria-label={T(`loops.anim.${id}.${side}` as PageKey).replace(/[$`\\]/g, "")}
      >
        {pic.draw({ side, f })}
      </svg>
      <div className="flex flex-wrap items-baseline justify-between gap-x-2 border-t border-gray-100 px-3 py-1.5 text-xs">
        <span className={`font-mono tabular-nums ${side === "loop" ? "text-amber-700" : "text-blue-700"}`} data-testid={`anim-count-${side}`}>
          {count}
        </span>
      </div>
      <p className="px-3 pb-2 text-xs leading-snug text-gray-600">
        <RichText text={T(`loops.anim.${id}.${side}` as PageKey)} />
      </p>
    </figure>
  );

  return (
    <div ref={root} className="mt-3" data-testid={`loops-anim-${id}`}>
      <div className="grid gap-3 md:grid-cols-2">
        {panel("loop", lf, T("loops.anim.loopSteps").replace("{n}", String(loopCount(id, lf))), lf >= loopFrames(id))}
        {panel(
          "tensor",
          tf,
          T(id === "pendulum" ? "loops.anim.tensorSteps" : "loops.anim.tensorCalls").replace("{n}", String(tensorCount(id, tf))),
          tf >= tensorFrames(id),
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button type="button" className={btn} onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
          {playing ? `❚❚ ${T("loops.anim.pause")}` : `▶ ${T("loops.anim.play")}`}
        </button>
        <button
          type="button"
          className={btn}
          onClick={() => {
            setPlaying(false);
            setG((v) => (v + 1) % (cycle + 1));
          }}
        >
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
      <p className="mt-2 text-[11px] leading-snug text-gray-500">
        <RichText text={T("loops.anim.note")} />
      </p>
    </div>
  );
}
