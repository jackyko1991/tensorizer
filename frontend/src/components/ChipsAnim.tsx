/**
 * Animations of the Chips tab (Coding Skills ▸ Chips), on the step frame shared with the BLAS and NVLink tabs.
 * Plans and numbers: ../chipsAnim.ts; number formats: ../chipsModel.ts.
 */
import { COL, Txt } from "./animKit";
import { Arrow, Box, StepAnim, W, fill, type DrawProps } from "./BlasAnim";
import { BITS_STEPS, BITS_VALUE, BUDGETS, CHIPS_PLAN, FEED, MIXED_PLAN, MMA, SCALE_GRADS, bitsStep, busyFraction, lossScaling, scalarCycles, stallAt, sumCurves, type ChipsCase, type MixedCase } from "../chipsAnim";
import { allValues, bitsOf, fmt } from "../chipsModel";

const fmtNum = (v: number, d = 6) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : Number(v.toPrecision(d)).toString());

// --- bits: π in FP32, BF16, FP16, FP8, FP4 ------------------------------------------------------------------

function Bits({ step, f, k }: DrawProps) {
  const { f: format, value, error } = bitsStep(step);
  const b = bitsOf(BITS_VALUE, format);
  const cells = [{ bits: [b.sign], color: COL.red, light: COL.redLight }, { bits: b.exp, color: COL.blue, light: COL.blueLight }, { bits: b.man, color: COL.green, light: COL.greenLight }];
  const total = 1 + format.e + format.m;
  const shown = 1 + b.exp.length + b.man.length;
  const size = Math.min(9, 252 / shown);
  let x = 14;
  const reveal = Math.ceil(f * shown);
  let idx = 0;
  return (
    <g>
      <Txt x={14} y={16} size={9} weight={600}>
        {format.name}
      </Txt>
      <Txt x={W - 14} y={16} size={7.5} anchor="end" color={COL.muted}>
        {fill(k("bitsTotal"), { n: total })}
      </Txt>
      {cells.map((c, ci) =>
        c.bits.map((bit, j) => {
          const on = idx++ < reveal;
          const cx = x;
          x += size;
          return (
            <g key={`${ci}-${j}`}>
              <rect x={cx} y={26} width={size - 0.8} height={14} rx={1.5} fill={on ? c.light : COL.idle} stroke={on ? c.color : COL.idleLine} strokeWidth={0.6} />
              {size >= 7 && on && (
                <Txt x={cx + size / 2 - 0.4} y={36} size={Math.min(7, size - 1)} anchor="middle" kind="mono" color={c.color}>
                  {bit}
                </Txt>
              )}
            </g>
          );
        }),
      )}
      {format.m > b.man.length && (
        <Txt x={x + 3} y={36} size={7} color={COL.muted}>
          …
        </Txt>
      )}
      <Txt x={14} y={54} size={7.5} color={COL.red}>
        {k("sign")}
      </Txt>
      <Txt x={60} y={54} size={7.5} color={COL.blue}>
        {fill(k("exp"), { n: format.e })}
      </Txt>
      <Txt x={150} y={54} size={7.5} color={COL.green}>
        {fill(k("man"), { n: format.m })}
      </Txt>
      <Txt x={14} y={78} size={8.5}>
        {fill(k("value"), { v: fmtNum(value, 10) })}
      </Txt>
      <Txt x={14} y={94} size={8.5} color={error > 1e-3 ? COL.red : COL.green}>
        {fill(k("error"), { e: error === 0 ? "0" : error.toExponential(1) })}
      </Txt>
      {/* the error, as a bar on a log scale from 1e-8 to 1 */}
      <rect x={14} y={104} width={W - 28} height={6} rx={3} fill={COL.idle} />
      <rect x={14} y={104} width={Math.max(2, ((Math.log10(Math.max(error, 1e-8)) + 8) / 8) * (W - 28))} height={6} rx={3} fill={error > 1e-3 ? COL.red : COL.cur} />
      <Txt x={14} y={120} size={6.5} color={COL.muted}>
        10⁻⁸
      </Txt>
      <Txt x={W - 14} y={120} size={6.5} anchor="end" color={COL.muted}>
        1
      </Txt>
      <Txt x={W / 2} y={120} size={6.5} anchor="middle" color={COL.muted}>
        {fill(k("pi"), { n: BITS_STEPS.length })}
      </Txt>
    </g>
  );
}

// --- grid: where the values are -------------------------------------------------------------------------

function Grid({ step, f, k }: DrawProps) {
  const lines: { id: string; y: number }[] = [
    { id: "fp4e2m1", y: 30 },
    { id: "fp6e2m3", y: 62 },
    { id: "fp8e4m3", y: 94 },
  ];
  const x0 = 16;
  const x1 = W - 12;
  const X = (v: number) => x0 + (v / 6) * (x1 - x0);
  if (step === 3) {
    // near 1: BF16 has 2^7 values per binade, FP16 2^10
    const lo = 1;
    const hi = 1.0625;
    const Xn = (v: number) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
    // in [1, 2) consecutive values are 2^-m apart
    const ticks = (id: string) => {
      const step = 2 ** -fmt(id).m;
      return Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step);
    };
    return (
      <g>
        <Txt x={x0} y={16} size={8} weight={600}>
          {k("near1Title")}
        </Txt>
        {(["bf16", "fp16"] as const).map((id, r) => {
          const y = 44 + r * 40;
          const t = ticks(id);
          const shown = Math.ceil(f * t.length);
          return (
            <g key={id}>
              <line x1={x0} y1={y} x2={x1} y2={y} stroke={COL.idleLine} />
              {t.slice(0, shown).map((v) => (
                <line key={v} x1={Xn(v)} y1={y - 5} x2={Xn(v)} y2={y + 5} stroke={r ? COL.violet : COL.green} strokeWidth={0.7} />
              ))}
              <Txt x={x0} y={y - 9} size={7.5} color={r ? COL.violet : COL.green}>
                {fill(k("near1Line"), { name: fmt(id).name, n: t.length - 1 })}
              </Txt>
            </g>
          );
        })}
        <Txt x={x0} y={120} size={6.5} color={COL.muted}>
          1
        </Txt>
        <Txt x={x1} y={120} size={6.5} anchor="end" color={COL.muted}>
          1.0625
        </Txt>
      </g>
    );
  }
  return (
    <g>
      {lines.slice(0, step + 1).map(({ id, y }, r) => {
        const vals = allValues(fmt(id)).filter((v) => v <= 6);
        const shown = r === step ? Math.ceil(f * vals.length) : vals.length;
        const colour = [COL.red, COL.cur, COL.blue][r];
        return (
          <g key={id}>
            <line x1={x0} y1={y} x2={x1} y2={y} stroke={COL.idleLine} />
            {vals.slice(0, shown).map((v) => (
              <circle key={v} cx={X(v)} cy={y} r={r === 2 ? 1 : 1.8} fill={colour} />
            ))}
            <Txt x={x0} y={y - 7} size={7.5} color={colour}>
              {fill(k("line"), { name: fmt(id).name, n: allValues(fmt(id)).length })}
            </Txt>
          </g>
        );
      })}
      {[0, 1, 2, 3, 4, 5, 6].map((v) => (
        <Txt key={v} x={X(v)} y={116} size={6.5} anchor="middle" color={COL.muted}>
          {v}
        </Txt>
      ))}
    </g>
  );
}

// --- sum: 0.1 added 4096 times in FP16 ------------------------------------------------------------------

const CURVES = sumCurves();
const STALL = stallAt(CURVES.naive);

function Sum({ step, f, k }: DrawProps) {
  const x0 = 30;
  const x1 = W - 10;
  const y0 = 108;
  const y1 = 14;
  const n = CURVES.exact.length;
  const top = CURVES.exact[n - 1];
  const X = (i: number) => x0 + (i / (n - 1)) * (x1 - x0);
  const Y = (v: number) => y0 - (v / top) * (y0 - y1);
  const path = (c: number[], upto: number) =>
    c
      .slice(0, upto)
      .filter((_, i) => i % 16 === 0 || i === upto - 1)
      .map((v, j, arr) => `${j ? "L" : "M"}${X(j === arr.length - 1 ? upto - 1 : j * 16).toFixed(1)},${Y(v).toFixed(1)}`)
      .join(" ");
  const series: { key: "naive" | "wide" | "kahan"; color: string }[] = [
    { key: "naive", color: COL.red },
    { key: "wide", color: COL.blue },
    { key: "kahan", color: COL.green },
  ];
  return (
    <g>
      <line x1={x0} y1={y0} x2={x1} y2={y0} stroke={COL.idleLine} />
      <line x1={x0} y1={y0} x2={x0} y2={y1} stroke={COL.idleLine} />
      <path d={path(CURVES.exact, n)} fill="none" stroke={COL.muted} strokeWidth={0.8} strokeDasharray="3 2" />
      {series.slice(0, step + 1).map((s, i) => {
        const upto = i === step ? Math.max(2, Math.ceil(f * n)) : n;
        return <path key={s.key} d={path(CURVES[s.key], upto)} fill="none" stroke={s.color} strokeWidth={1.4} />;
      })}
      {step >= 0 && (
        <g>
          <line x1={X(STALL)} y1={y0} x2={X(STALL)} y2={Y(CURVES.naive[STALL])} stroke={COL.red} strokeWidth={0.5} strokeDasharray="2 2" />
          <Txt x={X(STALL) + 3} y={Y(CURVES.naive[STALL]) + 10} size={6.5} color={COL.red}>
            {fill(k("stall"), { i: STALL, v: fmtNum(CURVES.naive[STALL], 4) })}
          </Txt>
        </g>
      )}
      <Txt x={x0 - 3} y={Y(top) + 3} size={6.5} anchor="end" color={COL.muted}>
        {fmtNum(top, 4)}
      </Txt>
      <Txt x={x0 - 3} y={y0} size={6.5} anchor="end" color={COL.muted}>
        0
      </Txt>
      <Txt x={x1} y={y0 + 10} size={6.5} anchor="end" color={COL.muted}>
        {fill(k("steps"), { n })}
      </Txt>
      {series.slice(0, step + 1).map((s, i) => (
        <Txt key={s.key} x={x0 + 4} y={y1 + 2 + i * 9} size={7} color={s.color}>
          {fill(k(s.key), { v: fmtNum(CURVES[s.key][i === step ? Math.max(1, Math.ceil(f * n)) - 1 : n - 1], 5) })}
        </Txt>
      ))}
    </g>
  );
}

// --- budget: who gets the silicon -----------------------------------------------------------------------

const BLOCK_COL = { cpu: [COL.blue, COL.blueLight], gpu: [COL.green, COL.greenLight], npu: [COL.violet, COL.violetLight], cache: [COL.cur, COL.curLight], io: ["#64748b", "#e2e8f0"] } as const;

function Budget({ step, f, k }: DrawProps) {
  const b = BUDGETS[step];
  const x0 = 20;
  const w = W - 40;
  const h = 70;
  let x = x0;
  return (
    <g>
      <Txt x={x0} y={16} size={8.5} weight={600}>
        {k(`chip${step}`)}
      </Txt>
      <rect x={x0 - 2} y={24} width={w + 4} height={h + 4} rx={4} fill="#fff" stroke="#334155" strokeWidth={1} />
      {b.blocks.map((bl) => {
        const bw = bl.share * w * Math.min(1, f * 1.4);
        const cx = x;
        x += bl.share * w;
        const [c, light] = BLOCK_COL[bl.kind];
        return (
          <g key={bl.kind}>
            <rect x={cx} y={26} width={Math.max(0, bw - 1)} height={h} rx={2} fill={light} stroke={c} strokeWidth={0.7} />
            {bl.share * w > 24 && f > 0.5 && (
              <Txt x={cx + (bl.share * w) / 2} y={26 + h / 2 + 3} size={7} anchor="middle" color={c} weight={600}>
                {k(bl.kind)}
              </Txt>
            )}
          </g>
        );
      })}
      <Txt x={x0} y={112} size={7} color={COL.text}>
        {k(`note${step}`)}
      </Txt>
      <Txt x={W - 20} y={122} size={6.5} anchor="end" color={COL.muted}>
        {k("schematic")}
      </Txt>
    </g>
  );
}

// --- feed: memory bandwidth vs arithmetic units ---------------------------------------------------------

function Feed({ step, t, k }: DrawProps) {
  const c = FEED[step];
  const busy = busyFraction(c.gbs, c.tflops);
  const pipe = 4 + (Math.log10(c.gbs) - 2) * 9; // pipe thickness grows with log bandwidth
  const lanes = 8;
  const lit = Math.max(1, Math.round(busy * lanes * 8));
  return (
    <g>
      <Box x={10} y={40} w={52} h={44} label={k(`mem${step}`)} fill={COL.violetLight} stroke={COL.violet} size={7} />
      <rect x={62} y={62 - pipe / 2} width={70} height={pipe} fill={COL.blueLight} stroke={COL.blue} strokeWidth={0.6} />
      {Array.from({ length: 5 }, (_, i) => {
        const px = 62 + ((t * 6 + i * 14) % 70);
        return <rect key={i} x={px} y={62 - Math.min(3, pipe / 3)} width={4} height={Math.min(6, (pipe * 2) / 3)} fill={COL.blue} />;
      })}
      <Txt x={97} y={62 - pipe / 2 - 4} size={7} anchor="middle" color={COL.blue}>
        {fill(k("gbs"), { v: c.gbs })}
      </Txt>
      {/* 8 × 8 arithmetic units; the lit ones have data */}
      {Array.from({ length: 64 }, (_, i) => {
        const r = Math.floor(i / 8);
        const col = i % 8;
        const on = i < lit && (t + i) % 3 !== 0;
        return <rect key={i} x={140 + col * 15} y={22 + r * 11} width={12} height={8} rx={1.5} fill={on ? COL.curLight : COL.idle} stroke={on ? COL.cur : COL.idleLine} strokeWidth={0.5} />;
      })}
      <Txt x={200} y={16} size={7} anchor="middle">
        {fill(k("alu"), { v: c.tflops })}
      </Txt>
      <Txt x={200} y={122} size={7.5} anchor="middle" color={busy < 0.05 ? COL.red : COL.text} weight={600}>
        {fill(k("busy"), { p: (busy * 100).toFixed(busy < 0.1 ? 1 : 0) })}
      </Txt>
    </g>
  );
}

// --- mma: scalar lanes vs a matrix unit -----------------------------------------------------------------

function Mma({ step, t, f, k }: DrawProps) {
  const n = MMA.n;
  const total = n ** 3;
  const scalar = scalarCycles(n, MMA.scalarPerCycle);
  const done = step === 0 ? Math.min(total, Math.ceil(f * scalar) * MMA.scalarPerCycle) : total;
  const cell = 9;
  const grid = (x: number, y: number, label: string, lit: number, color: string) => (
    <g>
      {Array.from({ length: n * n }, (_, i) => (
        <rect key={i} x={x + (i % n) * cell} y={y + Math.floor(i / n) * cell} width={cell - 1} height={cell - 1} rx={1} fill={i < lit ? color : COL.idle} stroke={COL.idleLine} strokeWidth={0.4} />
      ))}
      <Txt x={x + (n * cell) / 2} y={y - 4} size={7} anchor="middle" kind="math">
        {label}
      </Txt>
    </g>
  );
  const cOut = step === 0 ? Math.floor(done / n) : step === 1 ? Math.ceil(f * n * n) : n * n;
  return (
    <g>
      {grid(14, 22, "A", n * n, COL.blueLight)}
      {grid(62, 22, "B", n * n, COL.greenLight)}
      {grid(118, 22, "C", cOut, COL.curLight)}
      <Txt x={56} y={42} size={9} anchor="middle">
        ×
      </Txt>
      <Txt x={110} y={42} size={9} anchor="middle">
        =
      </Txt>
      {step === 0 && (
        <g>
          {Array.from({ length: MMA.scalarPerCycle }, (_, i) => (
            <Box key={i} x={176 + i * 24} y={24} w={20} h={14} label="FMA" size={6} fill={(t + i) % 2 ? COL.curLight : "#fff"} stroke={COL.cur} />
          ))}
          <Txt x={176} y={56} size={7.5}>
            {fill(k("fma"), { lanes: MMA.scalarPerCycle, cycles: scalar })}
          </Txt>
        </g>
      )}
      {step >= 1 && (
        <g>
          <Box x={176} y={22} w={92} h={22} label={k("unit")} size={7} fill={step === 1 && f < 1 ? COL.curLight : COL.greenLight} stroke={COL.green} weight={600} />
          <Txt x={176} y={56} size={7.5}>
            {fill(k("mma"), { n, ops: total })}
          </Txt>
        </g>
      )}
      <Arrow x1={152} y1={40} x2={172} y2={33} color={COL.muted} w={0.8} />
      {step === 2 && (
        <g>
          <Txt x={14} y={86} size={7.5} weight={600}>
            {k("names")}
          </Txt>
          <Txt x={14} y={100} size={7} color={COL.muted}>
            {k("names2")}
          </Txt>
        </g>
      )}
      <Txt x={14} y={118} size={7} color={COL.muted}>
        {fill(k("per"), { ops: total, scalar, mma: 1 })}
      </Txt>
    </g>
  );
}

// --- amp: one mixed-precision training step -------------------------------------------------------------

function Amp({ step, f, k }: DrawProps) {
  // stations of the loop, left to right then back: master (FP32) → copy (FP16/BF16) → forward → loss × S → backward → unscale → update
  const st = [
    { x: 8, y: 20, key: "master", fp32: true },
    { x: 96, y: 20, key: "copy", fp32: false },
    { x: 184, y: 20, key: "fwd", fp32: false },
    { x: 184, y: 66, key: "loss", fp32: true },
    { x: 96, y: 66, key: "bwd", fp32: false },
    { x: 8, y: 66, key: "unscale", fp32: true },
  ];
  const cur = Math.min(step, st.length - 1);
  return (
    <g>
      {st.map((s, i) => {
        const on = i === cur;
        const done = i < cur;
        const bg = on ? COL.curLight : done ? (s.fp32 ? COL.blueLight : COL.greenLight) : "#fff";
        return <Box key={s.key} x={s.x} y={s.y} w={84} h={30} label={k(s.key)} fill={bg} stroke={s.fp32 ? COL.blue : COL.green} size={6.6} weight={on ? 600 : undefined} />;
      })}
      <Arrow x1={92} y1={35} x2={96} y2={35} color={COL.muted} w={0.8} />
      <Arrow x1={180} y1={35} x2={184} y2={35} color={COL.muted} w={0.8} />
      <Arrow x1={226} y1={50} x2={226} y2={66} color={COL.muted} w={0.8} />
      <Arrow x1={184} y1={81} x2={180} y2={81} color={COL.muted} w={0.8} />
      <Arrow x1={96} y1={81} x2={92} y2={81} color={COL.muted} w={0.8} />
      <Arrow x1={50} y1={66} x2={50} y2={50} color={step === 5 ? COL.cur : COL.muted} w={step === 5 ? 1.4 : 0.8} />
      {step === 5 && f > 0.5 && (
        <Txt x={56} y={60} size={6.5} color={COL.cur}>
          {k("update")}
        </Txt>
      )}
      <rect x={8} y={106} width={8} height={8} rx={1} fill={COL.blueLight} stroke={COL.blue} strokeWidth={0.6} />
      <Txt x={20} y={113} size={6.8}>
        {k("fp32")}
      </Txt>
      <rect x={130} y={106} width={8} height={8} rx={1} fill={COL.greenLight} stroke={COL.green} strokeWidth={0.6} />
      <Txt x={142} y={113} size={6.8}>
        {k("low")}
      </Txt>
    </g>
  );
}

// --- scale: dynamic loss scaling over training ----------------------------------------------------------

const SCALES = lossScaling(SCALE_GRADS);

function Scale({ step, f, k }: DrawProps) {
  const n = SCALE_GRADS.length;
  const shown = step === 0 ? Math.max(1, Math.ceil(f * n)) : n;
  const x0 = 34;
  const x1 = W - 10;
  const X = (i: number) => x0 + (i / (n - 1)) * (x1 - x0);
  // log2 of the scale (top panel) and of the scaled gradient vs FP16's range (bottom panel)
  const ls = SCALES.map((s) => Math.log2(s.scale));
  const lo = Math.min(...ls) - 1;
  const hi = Math.max(...ls) + 1;
  const Ys = (v: number) => 54 - ((v - lo) / (hi - lo)) * 40;
  const Yg = (v: number) => 118 - ((Math.log2(v) + 24) / 40) * 44; // 2^-24 … 2^16
  return (
    <g>
      <Txt x={x0} y={10} size={7} color={COL.blue}>
        {k("scale")}
      </Txt>
      <path d={ls.slice(0, shown).map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Ys(v).toFixed(1)}`).join(" ")} fill="none" stroke={COL.blue} strokeWidth={1.3} />
      {SCALES.slice(0, shown).map((s, i) =>
        s.overflow ? (
          <g key={i}>
            <circle cx={X(i)} cy={Ys(ls[i])} r={2.6} fill={COL.redLight} stroke={COL.red} strokeWidth={0.8} />
          </g>
        ) : null,
      )}
      <Txt x={x1} y={10} size={6.5} anchor="end" color={COL.red}>
        {k("skip")}
      </Txt>
      {step === 1 && (
        <g>
          <line x1={x0} y1={Yg(65504)} x2={x1} y2={Yg(65504)} stroke={COL.red} strokeWidth={0.6} strokeDasharray="3 2" />
          <Txt x={x0 + 2} y={Yg(65504) - 2} size={6} color={COL.red}>
            {k("fp16max")}
          </Txt>
          <line x1={x0} y1={Yg(2 ** -24)} x2={x1} y2={Yg(2 ** -24)} stroke={COL.muted} strokeWidth={0.5} strokeDasharray="2 2" />
          {SCALE_GRADS.map((g, i) => (
            <g key={i}>
              <circle cx={X(i)} cy={Yg(Math.max(g * 1e-4, 2 ** -24))} r={1.4} fill={COL.muted} />
              <circle cx={X(i)} cy={Yg(Math.min(g * 1e-4 * SCALES[i].scale, 2 ** 16))} r={1.8} fill={SCALES[i].overflow ? COL.red : COL.green} />
            </g>
          ))}
          <Txt x={x0} y={70} size={7} color={COL.green}>
            {k("grads")}
          </Txt>
        </g>
      )}
      <Txt x={x1} y={126} size={6.5} anchor="end" color={COL.muted}>
        {fill(k("iter"), { n })}
      </Txt>
    </g>
  );
}

const CHIPS_PICTURES: Record<ChipsCase, { h: number; draw: (p: DrawProps) => React.ReactNode }> = {
  budget: { h: 128, draw: Budget },
  feed: { h: 128, draw: Feed },
  mma: { h: 124, draw: Mma },
};
const MIXED_PICTURES: Record<MixedCase, { h: number; draw: (p: DrawProps) => React.ReactNode }> = {
  bits: { h: 126, draw: Bits },
  grid: { h: 124, draw: Grid },
  sum: { h: 124, draw: Sum },
  amp: { h: 120, draw: Amp },
  scale: { h: 130, draw: Scale },
};

export default function ChipsAnim({ id }: { id: ChipsCase }) {
  const pic = CHIPS_PICTURES[id];
  return <StepAnim prefix="chips" id={id} plan={CHIPS_PLAN[id]} h={pic.h} draw={pic.draw} />;
}

export function MixedAnim({ id }: { id: MixedCase }) {
  const pic = MIXED_PICTURES[id];
  return <StepAnim prefix="mixed" id={id} plan={MIXED_PLAN[id]} h={pic.h} draw={pic.draw} />;
}
