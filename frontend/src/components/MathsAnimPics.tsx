/**
 * Pictures of the Tensor maths animations, part 1: shared helpers, "What a tensor is" and "Einstein summation".
 * Each picture draws one tick of its plan (mathsAnim.ts) in a 280-unit wide SVG; words come from maths.anim.*.
 */
import { type ReactNode } from "react";
import { type PageKey } from "../content/pages";
import {
  ADDRESS,
  BATCH_SHAPE,
  DOT,
  EX_A,
  LINES_PER_M,
  MM,
  MV,
  OUTER,
  PHOTO_H,
  PHOTO_SHAPE,
  PHOTO_W,
  RANK_DIMS,
  RANK_PICK,
  RECEIPT,
  ROT_A,
  ROT_R,
  ROT_V,
  STICK_M,
  TR,
  UNITS,
  UPDOWN_UNIT,
  BMM,
  lower,
  matMul,
  matVec,
  metPhi,
  mStage,
  numel,
  photoPixel,
  rot,
  rotAngle,
  rotMatrix,
  slanted,
  upper,
} from "../mathsAnim";
import { COL, Cell, FAMILY, Txt, mathSpans } from "./animKit";

export const W = 280;
export type Stage = ReturnType<typeof mStage>;
export interface PicProps {
  g: number;
  s: Stage;
  T: (k: PageKey) => string;
}

/** Number with at most d decimals, a real minus sign and no "-0". */
export const fmt = (v: number, d = 2) => {
  const r = Number(v.toFixed(d));
  return (Object.is(r, -0) ? 0 : r).toString().replace("-", "−");
};
/** Integer with thin-space thousands (921 600). */
export const fint = (n: number) => Math.round(n).toLocaleString("en").replace(/,/g, " ");

export function Arrow({ x1, y1, x2, y2, color, w = 1.2, dash, head = 4 }: {
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

/** Math symbol followed by upright text, e.g. "I" + " (FLOP/byte)". The math part must not end in a script. */
export function MU({ x, y, m, u, size = 7, color = COL.text, anchor = "start", weight }: {
  x: number;
  y: number;
  m: string;
  u: string;
  size?: number;
  color?: string;
  anchor?: "start" | "middle" | "end";
  weight?: number;
}) {
  return (
    <text x={x} y={y} fontSize={size} fill={color} textAnchor={anchor} fontWeight={weight}>
      {mathSpans(m)}
      <tspan style={{ fontFamily: FAMILY.sans }}>{u}</tspan>
    </text>
  );
}

/** A small matrix of cells with optional values and per-cell fill. */
export function Mat({ x, y, m, cs = 16, fill, stroke, label, size = 7, opacity }: {
  x: number;
  y: number;
  m: (number | string)[][];
  cs?: number;
  fill?: (r: number, c: number) => string | undefined;
  stroke?: (r: number, c: number) => string | undefined;
  label?: (r: number, c: number) => string | undefined;
  size?: number;
  opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      {m.map((row, r) =>
        row.map((v, c) => {
          const f = fill?.(r, c);
          const st = stroke?.(r, c);
          const lbl = label ? label(r, c) : typeof v === "number" ? fmt(v) : v;
          return (
            <Cell
              key={`${r}-${c}`}
              x={x + c * cs}
              y={y + r * cs}
              s={cs - 2}
              fill={f ?? "white"}
              stroke={st ?? COL.idleLine}
              sw={st ? 1.4 : 0.75}
              label={lbl}
              labelColor={f === COL.blue ? "white" : COL.text}
              size={size}
            />
          );
        }),
      )}
    </g>
  );
}

/** The shape line under an einsum picture, and which letters are summed or kept. */
export function Shapes({ y, shapes, summed, free, T, on }: {
  y: number;
  shapes: string;
  summed: string;
  free: string;
  T: PicProps["T"];
  on: boolean;
}) {
  return (
    <g>
      {on && <rect x={4} y={y - 9} width={W - 8} height={25} rx={3} fill={COL.blueLight} opacity={0.6} />}
      <Txt x={10} y={y} size={7.5} kind="mono" color={on ? COL.blue : COL.text} weight={on ? 600 : undefined}>
        {shapes}
      </Txt>
      <Txt x={10} y={y + 11} size={6.5} color={COL.cur}>
        {summed ? `${T("maths.anim.summed")}: ${summed}` : ""}
      </Txt>
      <Txt x={W - 10} y={y + 11} size={6.5} anchor="end" color={COL.blue}>
        {free ? `${T("maths.anim.free")}: ${free}` : ""}
      </Txt>
    </g>
  );
}

// ---- 1. what a tensor is ------------------------------------------------------------------------------

function House({ x, y, s, on }: { x: number; y: number; s: number; on: boolean }) {
  const f = on ? COL.curLight : "white";
  const st = on ? COL.cur : COL.idleLine;
  return (
    <g>
      <path d={`M ${x - 1} ${y + s * 0.42} L ${x + s / 2} ${y} L ${x + s + 1} ${y + s * 0.42} Z`} fill={f} stroke={st} strokeWidth={0.8} />
      <rect x={x} y={y + s * 0.42} width={s} height={s * 0.58} fill={f} stroke={st} strokeWidth={0.8} />
    </g>
  );
}

export function Address({ s, T }: PicProps) {
  const px = [4, 98, 190];
  const titles: PageKey[] = ["maths.anim.address.village", "maths.anim.address.flats", "maths.anim.address.city"];
  const code = ["v[3]", "A[2, 1]", "T[1, 2, 1]"];
  const level = (p: number) => (p === s.step ? s.t : p < s.step ? 3 : 0);
  const op = (p: number) => (p === s.step ? 1 : 0.4);
  const hot = COL.curLight;
  return (
    <>
      {[0, 1, 2].map((p) => {
        const x = px[p];
        const L = level(p);
        return (
          <g key={p} opacity={op(p)}>
            <Txt x={x + 44} y={12} size={7.5} anchor="middle" weight={600}>
              {T(titles[p])}
            </Txt>
            {p === 0 && (
              <>
                {Array.from({ length: ADDRESS[0].shape[0] }, (_, k) => (
                  <House key={k} x={x + 2 + k * 14} y={40} s={12} on={L >= 1 && k === ADDRESS[0].at[0]} />
                ))}
                <Arrow x1={x + 2} y1={62} x2={x + 86} y2={62} color={COL.blue} />
                <Txt x={x + 44} y={73} size={6.5} anchor="middle" color={COL.blue}>
                  {T("maths.anim.address.house")}
                </Txt>
              </>
            )}
            {p === 1 && (
              <>
                <path d={`M ${x + 20} 30 L ${x + 47.5} 21 L ${x + 75} 30 Z`} fill={COL.idle} stroke={COL.idleLine} strokeWidth={0.8} />
                {Array.from({ length: 4 }, (_, r) =>
                  Array.from({ length: 5 }, (_, c) => {
                    const cell = L >= 2 && r === 2 && c === 1;
                    const row = L >= 1 && r === 2;
                    return (
                      <rect
                        key={`${r}${c}`}
                        x={x + 20 + c * 11}
                        y={30 + r * 9}
                        width={11}
                        height={9}
                        fill={cell ? hot : row ? COL.blueLight : "white"}
                        stroke={cell ? COL.cur : COL.idleLine}
                        strokeWidth={cell ? 1.2 : 0.6}
                      />
                    );
                  }),
                )}
                <Arrow x1={x + 14} y1={30} x2={x + 14} y2={66} color={COL.blue} />
                <text x={x + 9} y={48} fontSize={6.5} fill={COL.blue} textAnchor="middle" transform={`rotate(-90 ${x + 9} 48)`}>
                  {T("maths.anim.address.floor")}
                </text>
                <Arrow x1={x + 20} y1={72} x2={x + 75} y2={72} color={COL.blue} />
                <Txt x={x + 47.5} y={82} size={6.5} anchor="middle" color={COL.blue}>
                  {T("maths.anim.address.door")}
                </Txt>
              </>
            )}
            {p === 2 && (
              <>
                {[0, 1, 2].map((b) => {
                  const bx = x + 2 + b * 29;
                  const bOn = L >= 1 && b === 1;
                  return (
                    <g key={b}>
                      <rect x={bx - 1} y={33} width={27} height={34} fill={bOn ? COL.blueLight : "none"} stroke={bOn ? COL.blue : "none"} strokeWidth={0.8} />
                      {Array.from({ length: 4 }, (_, r) =>
                        Array.from({ length: 5 }, (_, c) => {
                          const cell = bOn && L >= 3 && r === 2 && c === 1;
                          const row = bOn && L >= 2 && r === 2;
                          return (
                            <rect
                              key={`${r}${c}`}
                              x={bx + c * 5}
                              y={34 + r * 8}
                              width={5}
                              height={8}
                              fill={cell ? hot : row ? COL.done : "white"}
                              stroke={cell ? COL.cur : COL.idleLine}
                              strokeWidth={cell ? 1.1 : 0.5}
                            />
                          );
                        }),
                      )}
                    </g>
                  );
                })}
                <Arrow x1={x + 2} y1={26} x2={x + 85} y2={26} color={COL.blue} />
                <Txt x={x + 44} y={22} size={6.5} anchor="middle" color={COL.blue}>
                  {T("maths.anim.address.building")}
                </Txt>
              </>
            )}
            <Txt x={x + 44} y={100} size={8} anchor="middle" kind="mono" color={L >= p + 1 ? COL.cur : COL.text} weight={600}>
              {code[p]}
            </Txt>
            <Txt x={x + 44} y={112} size={6.5} anchor="middle" kind="mono" color={COL.muted}>
              {`shape (${ADDRESS[p].shape.join(", ")}${p === 0 ? "," : ""})`}
            </Txt>
          </g>
        );
      })}
    </>
  );
}

export function Rotate({ g, s, T }: PicProps) {
  const th = rotAngle(g);
  const R = rot(th);
  const w = matVec(R, ROT_V);
  const A2 = rotMatrix(R, ROT_A);
  const sc = 22;
  const L = 55;
  const cy = 86;
  const e1 = [Math.cos(th), -Math.sin(th)]; // SVG directions of the turned axes
  const e2 = [-Math.sin(th), -Math.cos(th)];
  const axes = (cx: number) => (
    <g>
      <line x1={cx - L} y1={cy} x2={cx + L} y2={cy} stroke={COL.idleLine} strokeDasharray="2 2" strokeWidth={0.7} />
      <line x1={cx} y1={cy - L} x2={cx} y2={cy + L} stroke={COL.idleLine} strokeDasharray="2 2" strokeWidth={0.7} />
      <line x1={cx - L * e1[0]} y1={cy - L * e1[1]} x2={cx + L * e1[0]} y2={cy + L * e1[1]} stroke={COL.blue} strokeWidth={0.9} />
      <line x1={cx - L * e2[0]} y1={cy - L * e2[1]} x2={cx + L * e2[0]} y2={cy + L * e2[1]} stroke={COL.blue} strokeWidth={0.9} />
      <Txt x={cx + (L + 6) * e1[0]} y={cy + (L + 6) * e1[1] + 3} size={7} anchor="middle" kind="math" color={COL.blue}>
        {"x'_1"}
      </Txt>
      <Txt x={cx + (L + 5) * e2[0] - 4} y={cy + (L + 5) * e2[1] + 2} size={7} anchor="end" kind="math" color={COL.blue}>
        {"x'_2"}
      </Txt>
    </g>
  );
  const cx1 = 70;
  const tip = [cx1 + ROT_V[0] * sc, cy - ROT_V[1] * sc];
  const f1 = [cx1 + w[0] * sc * e1[0], cy + w[0] * sc * e1[1]];
  const f2 = [cx1 + w[1] * sc * e2[0], cy + w[1] * sc * e2[1]];
  const cx2 = 210;
  const deg = Math.round((th * 180) / Math.PI);
  return (
    <>
      <Txt x={140} y={12} size={7.5} anchor="middle" kind="math" color={COL.muted}>
        {`θ = ${deg}°`}
      </Txt>
      <g opacity={s.step < 2 ? 1 : 0.45}>
        <Txt x={cx1} y={12} size={7.5} anchor="middle" weight={600}>
          {T("maths.anim.rotate.vector")}
        </Txt>
        {axes(cx1)}
        <line x1={tip[0]} y1={tip[1]} x2={f1[0]} y2={f1[1]} stroke={COL.cur} strokeDasharray="2 1.5" strokeWidth={0.8} />
        <line x1={tip[0]} y1={tip[1]} x2={f2[0]} y2={f2[1]} stroke={COL.cur} strokeDasharray="2 1.5" strokeWidth={0.8} />
        <circle cx={f1[0]} cy={f1[1]} r={1.8} fill={COL.cur} />
        <circle cx={f2[0]} cy={f2[1]} r={1.8} fill={COL.cur} />
        <Arrow x1={cx1} y1={cy} x2={tip[0]} y2={tip[1]} color={COL.text} w={1.6} />
        <Txt x={tip[0] + 4} y={tip[1] + 2} size={8} kind="math">
          v
        </Txt>
        <Txt x={cx1} y={152} size={7} anchor="middle" kind="math" color={s.step === 1 ? COL.cur : COL.text}>
          {`v'_1 = ${fmt(w[0])},  v'_2 = ${fmt(w[1])}`}
        </Txt>
        <Txt x={cx1} y={163} size={7} anchor="middle" kind="math" color={COL.green}>
          {`|v| = ${fmt(Math.hypot(w[0], w[1]))}`}
        </Txt>
      </g>
      <g opacity={s.step === 2 ? 1 : 0.45}>
        <Txt x={cx2} y={12} size={7.5} anchor="middle" weight={600}>
          {T("maths.anim.rotate.matrix")}
        </Txt>
        {axes(cx2)}
        <ellipse cx={cx2} cy={cy} rx={ROT_A[0][0] * 20} ry={ROT_A[1][1] * 20} fill={COL.violetLight} fillOpacity={0.6} stroke={COL.violet} strokeWidth={1.3} />
        <Txt x={cx2} y={152} size={7} anchor="middle" kind="math" color={s.step === 2 ? COL.cur : COL.text}>
          {`A'_{11} = ${fmt(A2[0][0])},  A'_{12} = ${fmt(A2[0][1])},  A'_{22} = ${fmt(A2[1][1])}`}
        </Txt>
        <Txt x={cx2} y={163} size={7} anchor="middle" kind="math" color={COL.green}>
          {`A'_{11} + A'_{22} = ${fmt(A2[0][0] + A2[1][1])}`}
        </Txt>
      </g>
    </>
  );
}

export function Updown({ s, T }: PicProps) {
  const ui = UPDOWN_UNIT[s.step];
  const f = UNITS[ui].f;
  const v1 = upper(f);
  const w1 = lower(f);
  const X0 = 16;
  const PX = 100; // px per metre
  const major = f === 1 ? 1 : f === 10 ? 5 : 50;
  const minor = f === 100 ? 5 : 1;
  const ticks: ReactNode[] = [];
  for (let k = 0; (k * PX) / f <= 210 + 1e-6; k += minor) {
    const x = X0 + (k * PX) / f;
    const big = k % major === 0;
    ticks.push(<line key={k} x1={x} y1={16} x2={x} y2={16 + (big ? 7 : 3.5)} stroke={COL.text} strokeWidth={big ? 0.8 : 0.4} />);
    if (big && (k * PX) / f <= 200 + 1e-6)
      ticks.push(
        <Txt key={`l${k}`} x={x} y={36} size={6} anchor="middle" color={COL.muted} kind="math">
          {String(k)}
        </Txt>,
      );
  }
  // contour lines every 1/3 m, offset half a spacing: 6 of them on the 2 m stick
  const lines = Array.from({ length: 7 }, (_, k) => X0 + ((k + 0.5) * PX) / LINES_PER_M).filter((x) => x < X0 + 210);
  const LX = (v: number) => 34 + ((Math.log10(v) + 2) / 5) * 186;
  const pair = s.step >= 3;
  const row = (y: number, v: number, label: string, color: string, on: boolean) => (
    <g>
      {on && <rect x={2} y={y - 7} width={W - 4} height={11} rx={2} fill={COL.greenLight} />}
      <line x1={34} y1={y} x2={220} y2={y} stroke={COL.idle} strokeWidth={0.8} />
      <Txt x={30} y={y + 2.5} size={7} anchor="end" kind="math" color={color}>
        {label}
      </Txt>
      <circle cx={LX(v)} cy={y} r={3} fill={color} />
      <Txt x={228} y={y + 2.5} size={7.5} kind="math" color={color} weight={on ? 700 : undefined}>
        {`= ${fmt(v, 2)}`}
      </Txt>
    </g>
  );
  return (
    <>
      <Txt x={X0} y={10} size={6.5} color={COL.muted}>
        {T("maths.anim.updown.ruler")}
      </Txt>
      <rect x={X0} y={16} width={210} height={14} fill="white" stroke={COL.idleLine} strokeWidth={0.6} />
      {ticks}
      <Arrow x1={X0} y1={44} x2={X0 + STICK_M * PX} y2={44} color={COL.blue} w={2.2} head={5} />
      <Txt x={W - 2} y={26} size={7} anchor="end" color={COL.text}>
        {`${T("maths.anim.updown.unit")}: 1 ${UNITS[ui].name}`}
      </Txt>
      <Txt x={232} y={46} size={8} kind="math" color={COL.blue} weight={600}>
        {`v^1 = ${fmt(v1)}`}
      </Txt>
      <Txt x={X0} y={58} size={6.5} color={COL.muted}>
        {T("maths.anim.updown.hill")}
      </Txt>
      <rect x={X0} y={62} width={210} height={34} fill={COL.greenLight} stroke={COL.idleLine} strokeWidth={0.6} />
      <rect x={X0} y={62} width={PX / f} height={34} fill={COL.curLight} fillOpacity={0.5} stroke={COL.cur} strokeWidth={0.8} strokeDasharray="2 1" />
      {lines.map((x, k) => (
        <path key={k} d={`M ${x} 62 Q ${x + 5} 70 ${x} 79 T ${x} 96`} fill="none" stroke={COL.green} strokeWidth={1} />
      ))}
      <line x1={X0} y1={79} x2={X0 + STICK_M * PX} y2={79} stroke={COL.blue} strokeWidth={1.4} />
      <Txt x={232} y={84} size={8} kind="math" color={COL.cur} weight={600}>
        {`w_1 = ${fmt(w1)}`}
      </Txt>
      {/* log gauge: the two components move apart, the pairing stays at 6 */}
      {row(116, v1, "v^1", COL.blue, false)}
      {row(129, w1, "w_1", COL.cur, false)}
      {row(142, v1 * w1, "w_1 v^1", COL.green, pair)}
      <line x1={34} y1={150} x2={220} y2={150} stroke={COL.text} strokeWidth={0.6} />
      {[-2, -1, 0, 1, 2, 3].map((k) => (
        <g key={k}>
          <line x1={LX(10 ** k)} y1={150} x2={LX(10 ** k)} y2={153} stroke={COL.text} strokeWidth={0.6} />
          <Txt x={LX(10 ** k)} y={160} size={6} anchor="middle" color={COL.muted} kind="math">
            {fmt(10 ** k, 2)}
          </Txt>
        </g>
      ))}
      <Txt x={127} y={170} size={6.5} anchor="middle" color={COL.muted}>
        {T("maths.anim.updown.axis")}
      </Txt>
      {s.step === 4 && (
        <Txt x={140} y={182} size={7.5} anchor="middle" kind="math" color={COL.violet}>
          {"e_i · e_j = δ_{ij}  ⇒  v^i = v_i"}
        </Txt>
      )}
    </>
  );
}

export function Metric({ s, T }: PicProps) {
  const phi = metPhi(s.step, s.t, s.n);
  const m = slanted(phi);
  const O = [40, 128];
  const sc = 78;
  const P = (x: number, y: number) => [O[0] + x * sc, O[1] - y * sc];
  const v = P(m.up[0] + m.up[1] * m.e2[0], m.up[1] * m.e2[1]);
  const a1 = P(m.up[0], 0);
  const b2 = P(m.up[1] * m.e2[0], m.up[1] * m.e2[1]);
  const d1 = P(m.down[0], 0);
  const d2 = P(m.down[1] * m.e2[0], m.down[1] * m.e2[1]);
  const ax2 = P(1.75 * m.e2[0], 1.75 * m.e2[1]);
  const deg = Math.round((phi * 180) / Math.PI);
  return (
    <>
      <line x1={O[0] - 10} y1={O[1]} x2={O[0] + 1.85 * sc} y2={O[1]} stroke={COL.idleLine} strokeWidth={0.8} />
      <line x1={O[0]} y1={O[1]} x2={ax2[0]} y2={ax2[1]} stroke={COL.idleLine} strokeWidth={0.8} />
      <Arrow x1={O[0]} y1={O[1]} x2={P(1, 0)[0]} y2={P(1, 0)[1]} color={COL.text} w={1.3} />
      <Arrow x1={O[0]} y1={O[1]} x2={P(m.e2[0], m.e2[1])[0]} y2={P(m.e2[0], m.e2[1])[1]} color={COL.text} w={1.3} />
      <Txt x={P(1, 0)[0]} y={O[1] + 10} size={7.5} anchor="middle" kind="math">
        {"e_1"}
      </Txt>
      <Txt x={P(m.e2[0], m.e2[1])[0] - 9} y={P(m.e2[0], m.e2[1])[1]} size={7.5} anchor="middle" kind="math">
        {"e_2"}
      </Txt>
      {/* upper components: walk along the axes (parallelogram) */}
      <line x1={v[0]} y1={v[1]} x2={a1[0]} y2={a1[1]} stroke={COL.blue} strokeDasharray="3 2" strokeWidth={0.9} />
      <line x1={v[0]} y1={v[1]} x2={b2[0]} y2={b2[1]} stroke={COL.blue} strokeDasharray="3 2" strokeWidth={0.9} />
      <circle cx={a1[0]} cy={a1[1]} r={2.4} fill={COL.blue} />
      <circle cx={b2[0]} cy={b2[1]} r={2.4} fill={COL.blue} />
      {/* lower components: perpendicular projections */}
      <line x1={v[0]} y1={v[1]} x2={d1[0]} y2={d1[1]} stroke={COL.cur} strokeDasharray="1 1.5" strokeWidth={1} />
      <line x1={v[0]} y1={v[1]} x2={d2[0]} y2={d2[1]} stroke={COL.cur} strokeDasharray="1 1.5" strokeWidth={1} />
      <rect x={d1[0] - 2} y={d1[1] - 2} width={4} height={4} fill={COL.cur} />
      <rect x={d2[0] - 2} y={d2[1] - 2} width={4} height={4} fill={COL.cur} />
      <Arrow x1={O[0]} y1={O[1]} x2={v[0]} y2={v[1]} color={COL.violet} w={1.8} head={5} />
      <Txt x={v[0] + 5} y={v[1] - 2} size={8.5} kind="math" color={COL.violet}>
        v
      </Txt>
      <Txt x={186} y={20} size={8} kind="math">{`φ = ${deg}°`}</Txt>
      <Txt x={186} y={38} size={7.5} kind="math" color={COL.blue}>{`v^1 = ${fmt(m.up[0])},  v^2 = ${fmt(m.up[1])}`}</Txt>
      <Txt x={186} y={52} size={7.5} kind="math" color={COL.cur}>{`v_1 = ${fmt(m.down[0])},  v_2 = ${fmt(m.down[1])}`}</Txt>
      <g opacity={s.step === 2 ? 1 : 0.55}>
        <Txt x={186} y={72} size={7.5} kind="math" color={COL.violet}>
          {`g_{12} = e_1 · e_2 = ${fmt(m.g[0][1])}`}
        </Txt>
        <Txt x={186} y={86} size={7.5} kind="math" color={COL.violet}>
          {"v_i = g_{ij} v^j"}
        </Txt>
      </g>
      <line x1={186} y1={106} x2={198} y2={106} stroke={COL.blue} strokeDasharray="3 2" strokeWidth={1} />
      <Txt x={202} y={108.5} size={6.5} color={COL.blue}>
        {T("maths.anim.metric.up")}
      </Txt>
      <line x1={186} y1={120} x2={198} y2={120} stroke={COL.cur} strokeDasharray="1 1.5" strokeWidth={1.2} />
      <Txt x={202} y={122.5} size={6.5} color={COL.cur}>
        {T("maths.anim.metric.down")}
      </Txt>
    </>
  );
}

const CH_COL: [number, number, number][] = [
  [220, 38, 38],
  [22, 163, 74],
  [37, 99, 235],
];
const chFill = (c: number, v: number) => {
  const [r, g, b] = CH_COL[c];
  const mix = (base: number) => Math.round(255 + (base - 255) * v);
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
};

function Plane({ x, y, c, cs }: { x: number; y: number; c: number; cs: number }) {
  const cells = [];
  for (let r = 0; r < PHOTO_H; r++)
    for (let k = 0; k < PHOTO_W; k++)
      cells.push(<rect key={`${r}-${k}`} x={x + k * cs} y={y + r * cs} width={cs} height={cs} fill={chFill(c, photoPixel(r, k)[c])} stroke="white" strokeWidth={0.3} />);
  return (
    <g>
      {cells}
      <rect x={x} y={y} width={PHOTO_W * cs} height={PHOTO_H * cs} fill="none" stroke={`rgb(${CH_COL[c].join(",")})`} strokeWidth={0.9} />
    </g>
  );
}

export function Photo({ s, T }: PicProps) {
  const cs = 6.5;
  const x0 = 26;
  const y0 = 50;
  const nCh = s.step === 0 ? 1 : s.step === 1 ? Math.min(3, s.t + 2) : 3;
  const batch = s.step >= 2;
  const nB = s.step === 2 ? Math.min(5, s.t + 1) : s.step > 2 ? 5 : 0;
  const shape = s.step === 0 ? [480, 640] : batch ? BATCH_SHAPE : PHOTO_SHAPE;
  const wP = PHOTO_W * cs;
  const hP = PHOTO_H * cs;
  const fin = s.step === 3;
  return (
    <>
      {[2, 1, 0]
        .filter((c) => c < nCh)
        .map((c) => (
          <Plane key={c} x={x0 + c * 9} y={y0 - c * 8} c={c} cs={cs} />
        ))}
      <Arrow x1={x0 - 6} y1={y0} x2={x0 - 6} y2={y0 + hP} color={COL.text} />
      <text x={x0 - 11} y={y0 + hP / 2} fontSize={6.5} fill={COL.text} textAnchor="middle" transform={`rotate(-90 ${x0 - 11} ${y0 + hP / 2})`}>
        {`${T("maths.anim.photo.rows")} 480`}
      </text>
      <Arrow x1={x0} y1={y0 + hP + 6} x2={x0 + wP} y2={y0 + hP + 6} color={COL.text} />
      <Txt x={x0 + wP / 2} y={y0 + hP + 15} size={6.5} anchor="middle">
        {`${T("maths.anim.photo.cols")} 640`}
      </Txt>
      {nCh > 1 && (
        <>
          <Arrow x1={x0 + wP + 3} y1={y0 - 2} x2={x0 + wP + 21} y2={y0 - 18} color={COL.text} />
          <Txt x={x0 + wP + 22} y={y0 - 20} size={6.5}>
            {`${T("maths.anim.photo.ch")} 3`}
          </Txt>
        </>
      )}
      {batch && (
        <>
          {Array.from({ length: nB }, (_, k) => {
            const bx = 128 + k * 26;
            return (
              <g key={k}>
                {[2, 1, 0].map((c) => (
                  <rect key={c} x={bx + c * 3} y={62 - c * 3} width={16} height={12} fill={chFill(c, 0.55)} stroke={`rgb(${CH_COL[c].join(",")})`} strokeWidth={0.6} />
                ))}
              </g>
            );
          })}
          {nB >= 5 && (
            <Txt x={262} y={72} size={9} anchor="middle" color={COL.muted}>
              …
            </Txt>
          )}
          <Arrow x1={x0 + wP + 34} y1={88} x2={272} y2={88} color={COL.violet} w={1.4} />
          <Txt x={200} y={98} size={6.5} anchor="middle" color={COL.violet}>
            {`${T("maths.anim.photo.batch")} 32`}
          </Txt>
        </>
      )}
      {fin && <rect x={4} y={120} width={W - 8} height={44} rx={3} fill={COL.blueLight} opacity={0.6} />}
      <Txt x={10} y={132} size={7.5} kind="mono" color={fin ? COL.blue : COL.text} weight={600}>
        {`x.shape = (${shape.join(", ")})`}
      </Txt>
      <Txt x={10} y={145} size={7.5} kind="mono" color={fin ? COL.blue : COL.text} weight={600}>
        {`x.ndim  = ${shape.length}`}
      </Txt>
      <Txt x={10} y={158} size={7} color={COL.muted}>
        {`${fint(numel(shape))} ${T("maths.anim.photo.numbers")}`}
      </Txt>
    </>
  );
}

export function EqRank({ s, T }: PicProps) {
  const r = s.step < 4 ? s.step + 1 : 4;
  const pick = s.step === 4;
  const dims = RANK_DIMS.slice(0, r);
  return (
    <>
      <Txt x={10} y={14} size={7.5} kind="mono">
        {`x = torch.zeros(${dims.join(", ")})`}
      </Txt>
      <Txt x={W - 10} y={14} size={8.5} kind="math" anchor="end" color={COL.blue} weight={600}>
        {`r = ${r}`}
      </Txt>
      {dims.map((d, k) => {
        const x = 8 + k * 67;
        const on = k === s.step;
        const bx = (v: number) => x + 6 + (v / (d - 1)) * 48;
        return (
          <g key={k}>
            <rect x={x} y={24} width={62} height={50} rx={4} fill={on ? COL.blueLight : "white"} stroke={on ? COL.blue : COL.idleLine} strokeWidth={on ? 1.2 : 0.7} />
            <Txt x={x + 31} y={37} size={8.5} anchor="middle" kind="math" color={COL.blue}>
              {`i_${k + 1}`}
            </Txt>
            <line x1={x + 6} y1={50} x2={x + 54} y2={50} stroke={COL.idleLine} strokeWidth={2} strokeLinecap="round" />
            {pick && (
              <>
                <circle cx={bx(RANK_PICK[k])} cy={50} r={3} fill={COL.cur} />
                <Txt x={bx(RANK_PICK[k])} y={46} size={6.5} anchor="middle" kind="mono" color={COL.cur}>
                  {String(RANK_PICK[k])}
                </Txt>
              </>
            )}
            <Txt x={x + 31} y={66} size={6.5} anchor="middle" kind="mono" color={COL.muted}>
              {`0 … ${d - 1}`}
            </Txt>
          </g>
        );
      })}
      <Txt x={10} y={92} size={7.5} kind="mono">
        {`x.ndim = ${r},  x.shape = (${dims.join(", ")}${r === 1 ? "," : ""})`}
      </Txt>
      <Txt x={10} y={105} size={7} color={COL.text}>
        {`${T("maths.anim.eqRank.entries")}: ${dims.join(" · ")} = ${fint(numel(dims))}`}
      </Txt>
      {pick && (
        <Txt x={10} y={120} size={7.5} kind="mono" color={COL.cur} weight={600}>
          {`x[${RANK_PICK.join(", ")}] = 0.0`}
        </Txt>
      )}
    </>
  );
}

export function EqRotate({ s }: PicProps) {
  const top = s.step < 2;
  const i = s.step === 1 ? 1 : 0;
  const v2 = matVec(ROT_R, ROT_V);
  const term = top ? s.t : 3;
  const A2 = rotMatrix(ROT_R, ROT_A);
  const filled = (r: number) => s.step === 2 || r < i || (r === i && s.t >= 2);
  return (
    <>
      <g opacity={top ? 1 : 0.4}>
        <Txt x={18} y={14} size={8} kind="math">R</Txt>
        <Mat
          x={14}
          y={20}
          m={ROT_R}
          cs={22}
          fill={(r, c) => (top && r === i ? (c === term ? COL.curLight : COL.blueLight) : undefined)}
          stroke={(r, c) => (top && r === i && c === term ? COL.cur : undefined)}
        />
        <Txt x={70} y={14} size={8} kind="math">v</Txt>
        <Mat x={66} y={20} m={ROT_V.map((x) => [x])} cs={22} fill={(r) => (top && r === term ? COL.curLight : undefined)} />
        <Txt x={98} y={46} size={10} anchor="middle" color={COL.muted}>=</Txt>
        <Txt x={112} y={14} size={8} kind="math">{"v'"}</Txt>
        <Mat
          x={108}
          y={20}
          m={v2.map((x) => [x])}
          cs={22}
          fill={(r) => (filled(r) ? (top && r === i ? COL.cur : COL.done) : undefined)}
          label={(r) => (filled(r) ? fmt(v2[r]) : "")}
        />
        {top && (
          <>
            <Txt x={142} y={30} size={7.5} kind="math">
              {`v'_${i + 1} = R_{${i + 1}1} v_1 + R_{${i + 1}2} v_2`}
            </Txt>
            <Txt x={142} y={44} size={7.5} kind="math" color={COL.cur}>
              {`= ${fmt(ROT_R[i][0])}·${ROT_V[0]}${term >= 1 ? ` + ${fmt(ROT_R[i][1])}·${ROT_V[1]}` : ""}${term >= 2 ? ` = ${fmt(v2[i])}` : ""}`.replace("+ −", "− ")}
            </Txt>
          </>
        )}
        <Txt x={142} y={60} size={7} kind="mono" color={COL.blue}>
          v2 = R @ v
        </Txt>
      </g>
      <g opacity={top ? 0.35 : 1}>
        {(["R", "A", "R^T"] as const).map((n, k) => (
          <Txt key={n} x={14 + k * 42} y={84} size={8} kind="math">
            {n}
          </Txt>
        ))}
        <Mat x={10} y={90} m={ROT_R} cs={18} size={6} fill={() => (!top && s.t >= 1 ? COL.blueLight : undefined)} />
        <Mat x={52} y={90} m={ROT_A} cs={18} size={6} />
        <Mat x={94} y={90} m={[[0.8, -0.6], [0.6, 0.8]]} cs={18} size={6} fill={() => (!top && s.t >= 2 ? COL.blueLight : undefined)} />
        <Txt x={140} y={110} size={10} anchor="middle" color={COL.muted}>=</Txt>
        <Txt x={154} y={84} size={8} kind="math">{"A'"}</Txt>
        <Mat x={150} y={90} m={A2} cs={24} size={6.5} fill={() => (!top && s.t >= 3 ? COL.done : undefined)} label={(r, c) => (!top && s.t >= 3 ? fmt(A2[r][c]) : "")} />
        <Txt x={204} y={104} size={7} kind="math" color={COL.green}>
          {"tr A' = 1.64 + 1.36"}
        </Txt>
        <Txt x={204} y={116} size={7} kind="math" color={COL.green}>
          {"= 3 = tr A"}
        </Txt>
        <Txt x={10} y={140} size={7} kind="mono" color={COL.blue}>
          A2 = R @ A @ R.T
        </Txt>
      </g>
    </>
  );
}

// ---- 2. Einstein summation ----------------------------------------------------------------------------

export function Receipt({ s, T }: PicProps) {
  const cx = [42, 92, 134, 176];
  const ys = [50, 64, 78];
  const done = (r: number) => s.step > 0 || r <= s.t;
  const nSum = s.step === 0 ? 0 : s.step === 1 ? s.t + 1 : 3;
  const prods = RECEIPT.q.map((q, i) => q * RECEIPT.p[i]);
  const total = prods.slice(0, nSum).reduce((a, b) => a + b, 0);
  return (
    <>
      <rect x={24} y={6} width={176} height={116} rx={3} fill="white" stroke={COL.idleLine} />
      <Txt x={cx[0]} y={20} size={6.5} anchor="middle" color={COL.muted}>{T("maths.anim.receipt.item")}</Txt>
      <Txt x={cx[1]} y={20} size={6.5} anchor="middle" color={COL.muted}>{T("maths.anim.receipt.qty")}</Txt>
      <Txt x={cx[2]} y={20} size={6.5} anchor="middle" color={COL.muted}>{T("maths.anim.receipt.price")}</Txt>
      <Txt x={cx[0]} y={32} size={7.5} anchor="middle" kind="math">i</Txt>
      <Txt x={cx[1]} y={32} size={7.5} anchor="middle" kind="math">{"q_i"}</Txt>
      <Txt x={cx[2]} y={32} size={7.5} anchor="middle" kind="math">{"p_i"}</Txt>
      <Txt x={cx[3]} y={32} size={7.5} anchor="middle" kind="math">{"q_i p_i"}</Txt>
      {ys.map((y, r) => (
        <g key={r}>
          {s.step === 0 && r === s.t && <rect x={28} y={y - 9} width={168} height={13} rx={2} fill={COL.curLight} opacity={0.7} />}
          <Txt x={cx[0]} y={y} size={7.5} anchor="middle" kind="math">{String(r + 1)}</Txt>
          <Txt x={cx[1]} y={y} size={7.5} anchor="middle" kind="math">{fmt(RECEIPT.q[r])}</Txt>
          <Txt x={cx[2]} y={y} size={7.5} anchor="middle" kind="math">{fmt(RECEIPT.p[r])}</Txt>
          {done(r) && (
            <Txt x={cx[3]} y={y} size={7.5} anchor="middle" kind="math" color={COL.cur} weight={600}>
              {fmt(prods[r])}
            </Txt>
          )}
        </g>
      ))}
      <line x1={30} y1={88} x2={194} y2={88} stroke={COL.text} strokeWidth={0.6} strokeDasharray="2 1.5" />
      <Txt x={cx[0]} y={104} size={7.5} anchor="middle" weight={600}>{T("maths.anim.receipt.total")}</Txt>
      {nSum > 0 && (
        <Txt x={194} y={104} size={7.5} anchor="end" kind="math" color={COL.blue} weight={600}>
          {`${prods.slice(0, nSum).map((p) => fmt(p)).join(" + ")} = ${fmt(total)}`}
        </Txt>
      )}
      <Txt x={242} y={40} size={9} anchor="middle" kind="math">{"q_i p_i"}</Txt>
      <Arrow x1={242} y1={48} x2={242} y2={70} color={s.step >= 1 ? COL.cur : COL.idleLine} w={1.4} />
      <Txt x={242} y={84} size={9} anchor="middle" kind="math" color={COL.blue} weight={600}>
        {s.step >= 1 ? fmt(total) : "?"}
      </Txt>
      {s.step === 2 && (
        <>
          <Txt x={242} y={100} size={6.5} anchor="middle" color={COL.cur}>{`${T("maths.anim.summed")}: i`}</Txt>
          <Txt x={242} y={113} size={6.5} anchor="middle" kind="mono" color={COL.blue}>{'"i,i->"'}</Txt>
        </>
      )}
    </>
  );
}

export function EinEx({ s }: PicProps) {
  const top = s.step !== 1;
  const term = s.step === 1 ? s.t : -1;
  return (
    <>
      <g opacity={top ? 1 : 0.4}>
        <Txt x={22} y={12} size={8} kind="math">A</Txt>
        <Mat x={18} y={16} m={EX_A} cs={20} fill={(r, c) => (top && r === c ? COL.curLight : undefined)} stroke={(r, c) => (top && r === c ? COL.cur : undefined)} />
        <Txt x={70} y={28} size={8} kind="math">{"A_{ii} = A_{11} + A_{22}"}</Txt>
        <Txt x={70} y={42} size={8} kind="math" color={COL.cur} weight={600}>
          {s.step === 0 && s.t === 0 ? "= 1 + 4" : "= 1 + 4 = 5"}
        </Txt>
        {s.step === 2 && (
          <Txt x={18} y={66} size={7} kind="mono" color={COL.blue}>
            {'torch.einsum("ii->", A)   # 5'}
          </Txt>
        )}
      </g>
      <g opacity={s.step === 1 ? 1 : 0.4}>
        <Txt x={22} y={84} size={8} kind="math">A</Txt>
        <Mat
          x={18}
          y={88}
          m={EX_A}
          cs={20}
          fill={(r, c) => (s.step === 1 && r === 0 ? (c === term ? COL.curLight : COL.blueLight) : undefined)}
          stroke={(r, c) => (r === 0 && c === term ? COL.cur : undefined)}
        />
        <Txt x={66} y={110} size={9} anchor="middle" color={COL.muted}>·</Txt>
        <Txt x={78} y={84} size={8} kind="math">A</Txt>
        <Mat
          x={74}
          y={88}
          m={EX_A}
          cs={20}
          fill={(r, c) => (s.step === 1 && c === 0 ? (r === term ? COL.curLight : COL.blueLight) : undefined)}
          stroke={(r, c) => (c === 0 && r === term ? COL.cur : undefined)}
        />
        <Txt x={126} y={100} size={8} kind="math">{"(AA)_{11} = A_{1j} A_{j1}"}</Txt>
        <Txt x={126} y={114} size={8} kind="math" color={COL.cur} weight={600}>
          {s.step === 1 ? ["= 1·1", "= 1·1 + 2·3", "= 1·1 + 2·3 = 7"][s.t] : "= 1·1 + 2·3 = 7"}
        </Txt>
      </g>
    </>
  );
}

export function EqDot({ s, T }: PicProps) {
  const n = s.step === 0 ? s.t + 1 : 3;
  const terms = DOT.x.map((x, i) => x * DOT.y[i]);
  const cur = s.step === 0 ? s.t : -1;
  const hi = (c: number) => (c === cur ? COL.curLight : c < n ? COL.blueLight : undefined);
  return (
    <>
      <Txt x={14} y={30} size={8} kind="math" anchor="middle">x</Txt>
      <Mat x={24} y={18} m={[DOT.x]} cs={20} fill={(_, c) => hi(c)} />
      <Txt x={14} y={52} size={8} kind="math" anchor="middle">y</Txt>
      <Mat x={24} y={40} m={[DOT.y]} cs={20} fill={(_, c) => hi(c)} />
      <Txt x={10} y={80} size={7.5} kind="math">{"x_i y_i"}</Txt>
      <Mat x={36} y={68} m={[terms]} cs={20} fill={(_, c) => (c < n ? (c === cur ? COL.cur : COL.done) : undefined)} label={(_, c) => (c < n ? fmt(terms[c]) : "")} />
      <Txt x={108} y={22} size={8} kind="math" color={COL.muted}>{"s = x_1 y_1 + x_2 y_2 + x_3 y_3"}</Txt>
      <Txt x={108} y={42} size={8.5} kind="math" color={COL.cur} weight={600}>
        {`s = ${terms.slice(0, n).map((t) => fmt(t)).join(" + ")}${n === 3 ? ` = ${fmt(terms.reduce((a, b) => a + b, 0))}` : ""}`}
      </Txt>
      <Txt x={108} y={62} size={7} kind="mono" color={COL.blue}>
        {'einsum("i,i->", x, y)'}
      </Txt>
      <Shapes y={104} shapes="(3,), (3,)  ->  ()" summed="i" free="" T={T} on={s.step === 1} />
    </>
  );
}

export function EqMatvec({ s, T }: PicProps) {
  const t = s.step === 0 ? s.t : 6;
  const i = Math.min(Math.floor(t / 2), 2);
  const j = t % 2;
  const on = s.step === 0;
  const y = matVec(MV.A, MV.x);
  const done = (r: number) => r < i || (r === i && (j === 1 || !on)) || !on;
  const partial = on && j === 0 ? MV.A[i][0] * MV.x[0] : y[i];
  return (
    <>
      <Txt x={10} y={50} size={7} kind="math" color={COL.blue}>i</Txt>
      <Txt x={34} y={14} size={7} kind="math" color={COL.cur}>j</Txt>
      <Mat
        x={18}
        y={18}
        m={MV.A}
        cs={17}
        fill={(r, c) => (on && r === i ? (c === j ? COL.curLight : COL.blueLight) : undefined)}
        stroke={(r, c) => (on && r === i && c === j ? COL.cur : undefined)}
      />
      <Txt x={66} y={14} size={8} kind="math">x</Txt>
      <Mat x={62} y={26} m={MV.x.map((v) => [v])} cs={17} fill={(r) => (on && r === j ? COL.curLight : undefined)} />
      <Txt x={88} y={46} size={10} anchor="middle" color={COL.muted}>=</Txt>
      <Txt x={102} y={14} size={8} kind="math">y</Txt>
      <Mat
        x={98}
        y={18}
        m={y.map((v) => [v])}
        cs={17}
        fill={(r) => (on && r === i ? COL.cur : done(r) ? COL.done : undefined)}
        label={(r) => (on && r === i ? fmt(partial) : done(r) ? fmt(y[r]) : "")}
      />
      <Txt x={128} y={30} size={7.5} kind="math">
        {`y_${i + 1} = A_{${i + 1}1} x_1 + A_{${i + 1}2} x_2`}
      </Txt>
      <Txt x={128} y={44} size={7.5} kind="math" color={COL.cur} weight={600}>
        {`= ${MV.A[i][0]}·${MV.x[0]}${!on || j === 1 ? ` + ${MV.A[i][1]}·${MV.x[1]} = ${fmt(y[i])}` : ""}`}
      </Txt>
      <Txt x={128} y={62} size={7} kind="mono" color={COL.blue}>
        {'einsum("ij,j->i", A, x)'}
      </Txt>
      <Shapes y={100} shapes="(3, 2), (2,)  ->  (3,)" summed="j" free="i" T={T} on={!on} />
    </>
  );
}

export function EqMatmul({ s, T }: PicProps) {
  const on = s.step === 0;
  const t = on ? s.t : 4;
  const i = Math.min(t >> 1, 1);
  const k = t & 1;
  const C = matMul(MM.A, MM.B);
  const done = (r: number, c: number) => !on || r * 2 + c < t;
  const terms = MM.A[i].map((a, j) => `${a}·${MM.B[j][k]}`).join(" + ");
  return (
    <>
      <Txt x={6} y={38} size={7} kind="math" color={COL.blue}>i</Txt>
      <Txt x={36} y={20} size={7} kind="math" color={COL.cur}>j</Txt>
      <Mat x={12} y={24} m={MM.A} cs={16} fill={(r) => (on && r === i ? COL.blueLight : undefined)} />
      <Txt x={64} y={42} size={9} anchor="middle" color={COL.muted}>·</Txt>
      <Txt x={84} y={12} size={7} kind="math" color={COL.blue}>k</Txt>
      <Mat x={72} y={16} m={MM.B} cs={16} fill={(_, c) => (on && c === k ? COL.blueLight : undefined)} />
      <Txt x={112} y={42} size={10} anchor="middle" color={COL.muted}>=</Txt>
      <Mat
        x={122}
        y={24}
        m={C}
        cs={16}
        fill={(r, c) => (on && r === i && c === k ? COL.cur : done(r, c) ? COL.done : undefined)}
        label={(r, c) => (done(r, c) || (on && r === i && c === k) ? fmt(C[r][c]) : "")}
      />
      <Txt x={162} y={24} size={7.5} kind="math">
        {`C_{${i + 1}${k + 1}} = A_{${i + 1}j} B_{j${k + 1}}`}
      </Txt>
      <Txt x={162} y={38} size={7.5} kind="math" color={COL.cur}>{`= ${terms}`}</Txt>
      <Txt x={162} y={52} size={7.5} kind="math" color={COL.cur} weight={600}>{`= ${fmt(C[i][k])}`}</Txt>
      <Txt x={162} y={68} size={7} kind="mono" color={COL.blue}>
        {'einsum("ij,jk->ik")'}
      </Txt>
      <Shapes y={100} shapes="(2, 3), (3, 2)  ->  (2, 2)" summed="j" free="i, k" T={T} on={!on} />
    </>
  );
}

export function EqTrace({ s, T }: PicProps) {
  const n = s.step === 0 ? s.t + 1 : 3;
  const d = TR.map((r, i) => r[i]);
  return (
    <>
      <Txt x={22} y={12} size={8} kind="math">A</Txt>
      <Mat
        x={18}
        y={16}
        m={TR}
        cs={20}
        fill={(r, c) => (r === c && r < n ? (s.step === 0 && r === s.t ? COL.cur : COL.curLight) : undefined)}
        stroke={(r, c) => (r === c ? COL.cur : undefined)}
      />
      <Txt x={96} y={30} size={8} kind="math">{"t = A_{11} + A_{22} + A_{33}"}</Txt>
      <Txt x={96} y={46} size={8.5} kind="math" color={COL.cur} weight={600}>
        {`t = ${d.slice(0, n).join(" + ")}${n === 3 ? ` = ${d.reduce((a, b) => a + b, 0)}` : ""}`}
      </Txt>
      <Txt x={96} y={64} size={7} kind="mono" color={COL.blue}>
        {'einsum("ii->", A)'}
      </Txt>
      <Shapes y={100} shapes="(3, 3)  ->  ()" summed="i" free="" T={T} on={s.step === 1} />
    </>
  );
}

export function EqOuter({ s, T }: PicProps) {
  const n = s.step === 0 ? s.t + 1 : 9;
  const P = OUTER.u.map((a) => OUTER.v.map((b) => a * b));
  const cur = s.step === 0 ? s.t : -1;
  const ci = Math.floor(Math.max(cur, 0) / 3);
  const cj = Math.max(cur, 0) % 3;
  return (
    <>
      <Txt x={24} y={14} size={8} kind="math">u</Txt>
      <Mat x={18} y={32} m={OUTER.u.map((x) => [x])} cs={18} fill={(r) => (cur >= 0 && r === ci ? COL.curLight : undefined)} />
      <Txt x={34} y={26} size={8} kind="math">v</Txt>
      <Mat x={44} y={12} m={[OUTER.v]} cs={18} fill={(_, c) => (cur >= 0 && c === cj ? COL.curLight : undefined)} />
      <Mat
        x={44}
        y={32}
        m={P}
        cs={18}
        fill={(r, c) => (r * 3 + c === cur ? COL.cur : r * 3 + c < n ? COL.done : undefined)}
        label={(r, c) => (r * 3 + c < n ? fmt(P[r][c]) : "")}
      />
      <Txt x={112} y={36} size={8} kind="math">{"P_{ij} = u_i v_j"}</Txt>
      {cur >= 0 && (
        <Txt x={112} y={52} size={8} kind="math" color={COL.cur} weight={600}>
          {`P_{${ci + 1}${cj + 1}} = ${OUTER.u[ci]}·${OUTER.v[cj]} = ${P[ci][cj]}`}
        </Txt>
      )}
      <Txt x={112} y={70} size={7} kind="mono" color={COL.blue}>
        {'einsum("i,j->ij", u, v)'}
      </Txt>
      <Shapes y={102} shapes="(3,), (3,)  ->  (3, 3)" summed="" free="i, j" T={T} on={s.step === 1} />
    </>
  );
}

export function EqBatch({ s, T }: PicProps) {
  const C = BMM.A.map((A, b) => matMul(A, BMM.B[b]));
  const cb = s.step < 2 ? s.step : -1;
  const t = s.step < 2 ? s.t : 4;
  return (
    <>
      {[0, 1].map((b) => {
        const y = 10 + b * 44;
        const active = b === cb;
        const done = (r: number, c: number) => b < cb || cb < 0 || (active && r * 2 + c < t);
        const i = t >> 1;
        const k = t & 1;
        return (
          <g key={b} opacity={active || cb < 0 ? 1 : 0.4}>
            <Txt x={6} y={y + 17} size={7.5} kind="math" color={COL.violet}>{`b = ${b + 1}`}</Txt>
            <Mat x={38} y={y} m={BMM.A[b]} cs={15} size={6.5} fill={(r) => (active && r === i ? COL.blueLight : undefined)} />
            <Txt x={74} y={y + 18} size={9} anchor="middle" color={COL.muted}>·</Txt>
            <Mat x={80} y={y} m={BMM.B[b]} cs={15} size={6.5} fill={(_, c) => (active && c === k ? COL.blueLight : undefined)} />
            <Txt x={117} y={y + 19} size={9} anchor="middle" color={COL.muted}>=</Txt>
            <Mat
              x={124}
              y={y}
              m={C[b]}
              cs={15}
              size={6.5}
              fill={(r, c) => (active && r === i && c === k ? COL.cur : done(r, c) ? COL.done : undefined)}
              label={(r, c) => (done(r, c) || (active && r === i && c === k) ? fmt(C[b][r][c]) : "")}
            />
            {active && t < 4 && (
              <Txt x={162} y={y + 17} size={7} kind="math" color={COL.cur}>
                {`C_{${b + 1}${i + 1}${k + 1}} = ${BMM.A[b][i].map((a, j) => `${a}·${BMM.B[b][j][k]}`).join(" + ")} = ${C[b][i][k]}`}
              </Txt>
            )}
          </g>
        );
      })}
      <Txt x={162} y={96} size={7} kind="mono" color={COL.blue}>
        torch.bmm(A, B)
      </Txt>
      <Shapes y={116} shapes="(2, 2, 2), (2, 2, 2)  ->  (2, 2, 2)" summed="j" free="b, i, k" T={T} on={s.step === 2} />
    </>
  );
}
