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
