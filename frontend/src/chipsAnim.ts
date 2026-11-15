// Step plans and numbers of the Chips tab's animations (Coding Skills ▸ Chips), on the step frame shared with the
// BLAS and NVLink tabs. Pure, unit-tested (frontend/tests/chipsAnim.test.ts).
import { fmt, roundTo, runningSum, type FloatFormat } from "./chipsModel.ts";
import type { StepPlan } from "./blasNvlinkAnim.ts";

/** Coding Skills ▸ Mixed precision */
export const MIXED_PLAN = {
  bits: { steps: [10, 10, 10, 10, 10], tick: 420, labels: ["sign", "exp", "man", "value", "error", "bitsTotal", "pi"] },
  grid: { steps: [10, 10, 10, 12], tick: 420, labels: ["line", "near1Title", "near1Line"] },
  sum: { steps: [14, 14, 14], tick: 260, labels: ["naive", "wide", "kahan", "stall", "steps"] },
  amp: { steps: [8, 8, 8, 8, 8, 8], tick: 420, labels: ["master", "copy", "fwd", "loss", "bwd", "unscale", "update", "fp32", "low"] },
  scale: { steps: [16, 16], tick: 240, labels: ["scale", "skip", "grads", "iter", "fp16max"] },
} as const satisfies Record<string, StepPlan>;
export type MixedCase = keyof typeof MIXED_PLAN;

/** Coding Skills ▸ Chips */
export const CHIPS_PLAN = {
  feed: { steps: [12, 12, 12], tick: 300, labels: ["mem0", "mem1", "mem2", "alu", "busy", "gbs"] },
  mma: { steps: [12, 12, 10], tick: 380, labels: ["fma", "mma", "unit", "names", "names2", "per"] },
} as const satisfies Record<string, StepPlan>;
export type ChipsCase = keyof typeof CHIPS_PLAN;

/**
 * Dynamic loss scaling, as in PyTorch's GradScaler: gradients are computed for loss × S; if any is inf/NaN in FP16
 * the step is skipped and S halves; after `interval` good steps in a row S doubles. `grads` are the unscaled
 * largest gradient magnitudes per iteration. Returns, per iteration, the scale used and whether it overflowed.
 */
export function lossScaling(grads: number[], s0 = 2 ** 16, interval = 4, max = 65504): { scale: number; overflow: boolean }[] {
  let s = s0;
  let good = 0;
  return grads.map((g) => {
    const overflow = g * s > max;
    const out = { scale: s, overflow };
    if (overflow) {
      s /= 2;
      good = 0;
    } else if (++good >= interval) {
      s *= 2;
      good = 0;
    }
    return out;
  });
}
/** a made-up but typical gradient history for the animation: large early, then small and slowly shrinking */
export const SCALE_GRADS = [1.2, 0.8, 0.6, 0.3, 0.05, 0.02, 0.02, 0.015, 0.01, 0.012, 0.008, 0.006, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003, 0.0025, 0.002];

/** the formats the "bits" animation steps through, and the value it squeezes */
export const BITS_STEPS = ["fp32", "bf16", "fp16", "fp8e4m3", "fp4e2m1"] as const;
export const BITS_VALUE = Math.PI;

export function bitsStep(i: number): { f: FloatFormat; value: number; error: number } {
  const f = fmt(BITS_STEPS[Math.min(Math.max(i, 0), BITS_STEPS.length - 1)]);
  const value = roundTo(BITS_VALUE, f);
  return { f, value, error: Math.abs(value - BITS_VALUE) / BITS_VALUE };
}

/** the running sums the "sum" animation draws: N copies of 0.1 in FP16, three ways */
export const SUM_N = 4096;
export const SUM_X = 0.1;
export function sumCurves(n = SUM_N): { naive: number[]; wide: number[]; kahan: number[]; exact: number[] } {
  const xs = Array(n).fill(SUM_X);
  const f = fmt("fp16");
  return {
    naive: runningSum(xs, f, "naive"),
    wide: runningSum(xs, f, "wide"),
    kahan: runningSum(xs, f, "kahan"),
    exact: xs.map((_, i) => (i + 1) * SUM_X),
  };
}
/** the step count where naive FP16 summation stops growing (adding 0.1 no longer changes the total) */
export function stallAt(curve: number[]): number {
  for (let i = 1; i < curve.length; i++) if (curve[i] === curve[i - 1]) return i;
  return curve.length;
}

/**
 * Schematic floorplans for the "budget" animation: shares of the die (or package) given to each block. These are
 * drawn proportions to make one point (who gets the silicon), not measurements; the picture says "schematic".
 */
export const BUDGETS: { id: string; blocks: { kind: "cpu" | "gpu" | "npu" | "io" | "cache"; share: number }[] }[] = [
  { id: "laptop", blocks: [{ kind: "cpu", share: 0.38 }, { kind: "gpu", share: 0.22 }, { kind: "npu", share: 0.1 }, { kind: "cache", share: 0.12 }, { kind: "io", share: 0.18 }] },
  { id: "applemax", blocks: [{ kind: "cpu", share: 0.2 }, { kind: "gpu", share: 0.45 }, { kind: "npu", share: 0.05 }, { kind: "cache", share: 0.1 }, { kind: "io", share: 0.2 }] },
  { id: "datacenter", blocks: [{ kind: "gpu", share: 0.78 }, { kind: "cache", share: 0.1 }, { kind: "io", share: 0.12 }] },
];

/**
 * "feed": how busy the arithmetic units can be when every multiply needs fresh data from memory (LLM decoding,
 * about 1 FLOP per byte of weights in BF16): busy = min(1, bandwidth × intensity / peak).
 */
/** three chips from the Chips tab's data (content/chips/chips.json: bandwidth and dense BF16; a test keeps them equal) */
export const FEED = [
  { id: "amd-radeon-890m", gbs: 120, tflops: 23.8 },
  { id: "nvidia-rtx-4090", gbs: 1008, tflops: 165.2 },
  { id: "nvidia-h100-sxm", gbs: 3350, tflops: 989.5 },
] as const;
export const DECODE_INTENSITY = 1; // FLOP per byte: 2 FLOP per BF16 weight (2 bytes)
export const busyFraction = (gbs: number, tflops: number, intensity = DECODE_INTENSITY) =>
  Math.min(1, (gbs * 1e9 * intensity) / (tflops * 1e12));

/** "mma": multiply-adds per step for scalar lanes vs one 4×4×4 matrix-unit operation */
export const MMA = { n: 4, scalarPerCycle: 4 };
export const scalarCycles = (n: number, lanes: number) => Math.ceil(n ** 3 / lanes);

// --- Mixed precision figures (components/MixedFigures.tsx) --------------------------------------------------------

/**
 * Schematic histogram of activation-gradient magnitudes, after Micikevicius et al. 2018 (Fig. 3, Multibox SSD):
 * about two thirds exact zeros, the rest spread over 2^-60 … 2^-11 with a peak near 2^-29. Bin k holds the share (%)
 * of all values with magnitude in [2^k, 2^(k+1)). The shape is illustrative, not the paper's data.
 */
export const HIST_ZERO = 67;
export function gradHistogram(): { k: number; p: number }[] {
  const raw: { k: number; p: number }[] = [];
  for (let k = -60; k <= -11; k++) {
    const d = k + 29;
    const sigma = d < 0 ? 4.6 : 5.2;
    raw.push({ k, p: Math.exp(-(d * d) / (2 * sigma * sigma)) });
  }
  const total = raw.reduce((a, b) => a + b.p, 0);
  return raw.map((b) => ({ k: b.k, p: (b.p / total) * (100 - HIST_ZERO) }));
}

/** FP16 thresholds: below 2^-24 a value rounds to zero, below 2^-14 it is subnormal, from 2^16 it overflows */
export const FP16_ZERO = -24;
export const FP16_SUBNORMAL = -14;
export const FP16_OVERFLOW = 16;

/**
 * Where the nonzero values land after multiplying by a loss scale 2^s, as percentages of the nonzero values:
 * lost (become zero), sub (subnormal: fewer significant bits), over (beyond FP16's largest value).
 */
export function histogramSplit(bins: { k: number; p: number }[], s: number) {
  const nonzero = bins.reduce((a, b) => a + b.p, 0);
  const share = (f: (u: number) => boolean) => (100 * bins.filter((b) => f(b.k + s)).reduce((a, b) => a + b.p, 0)) / nonzero;
  return {
    zero: HIST_ZERO,
    lost: share((u) => u < FP16_ZERO),
    sub: share((u) => u >= FP16_ZERO && u < FP16_SUBNORMAL),
    over: share((u) => u >= FP16_OVERFLOW),
  };
}

/** deterministic noise in [-1, 1] */
const noise = (i: number, seed: number) => {
  const x = Math.sin(i * 12.9898 + seed * 78.233) * 43758.5453;
  return 2 * (x - Math.floor(x)) - 1;
};

/**
 * Schematic training-loss curves after NVIDIA's mixed-precision guide (bigLSTM): FP32, mixed precision with no loss
 * scaling (diverges once gradients underflow), and with a loss scale of 128 (tracks FP32). Iterations in thousands.
 */
export function lossCurves() {
  const base = (t: number) => 2.75 + 0.75 * Math.exp(-t / 700) + 2.25 * Math.exp(-t / 35);
  const pts = (end: number, seed: number, extra: (t: number) => number = () => 0) => {
    const out: [number, number][] = [];
    for (let i = 0, t = 0; t <= end; i++, t += 4) out.push([t, base(t) + extra(t) + 0.035 * noise(i, seed)]);
    return out;
  };
  const drift = (t: number) => (t < 380 ? 0 : 1.15 * ((t - 380) / 520) ** 1.4);
  return {
    fp32: pts(1800, 1),
    scale1: pts(900, 2, (t) => drift(t) * (1 + 0.12 * noise(Math.round(t), 5))),
    scale128: pts(2000, 3),
  };
}
