// The Chips tab's arithmetic, pure and unit-tested (frontend/tests/chipsModel.test.ts): number formats (rounding a
// value to FP64 … FP4, their bits, every value a small format can hold), what low precision does to a running sum
// (naive, FP32 accumulation, Kahan), and the two numbers derived from a chip's spec sheet (the roofline's ridge point
// and the memory-bound ceiling of LLM decoding).

/** A binary floating-point format: 1 sign bit, `e` exponent bits, `m` stored mantissa bits. */
export interface FloatFormat {
  id: string;
  name: string;
  e: number;
  m: number;
  /** what happens above the largest finite value: IEEE formats overflow to infinity, OCP FP8 E4M3 and FP4 saturate */
  overflow: "inf" | "saturate";
  /** E4M3 (OCP) spends the top exponent on normal numbers except one NaN code, so its largest value is 448, not 240 */
  maxOverride?: number;
}

export const FORMATS: FloatFormat[] = [
  { id: "fp64", name: "FP64", e: 11, m: 52, overflow: "inf" },
  { id: "fp32", name: "FP32", e: 8, m: 23, overflow: "inf" },
  { id: "tf32", name: "TF32", e: 8, m: 10, overflow: "inf" },
  { id: "bf16", name: "BF16", e: 8, m: 7, overflow: "inf" },
  { id: "fp16", name: "FP16", e: 5, m: 10, overflow: "inf" },
  { id: "fp8e5m2", name: "FP8 E5M2", e: 5, m: 2, overflow: "inf" },
  { id: "fp8e4m3", name: "FP8 E4M3", e: 4, m: 3, overflow: "saturate", maxOverride: 448 },
  { id: "fp6e3m2", name: "FP6 E3M2", e: 3, m: 2, overflow: "saturate", maxOverride: 28 },
  { id: "fp6e2m3", name: "FP6 E2M3", e: 2, m: 3, overflow: "saturate", maxOverride: 7.5 },
  { id: "fp4e2m1", name: "FP4 E2M1", e: 2, m: 1, overflow: "saturate", maxOverride: 6 },
];
export const fmt = (id: string): FloatFormat => {
  const f = FORMATS.find((x) => x.id === id);
  if (!f) throw new Error(`unknown format ${id}`);
  return f;
};

export const bias = (f: FloatFormat) => 2 ** (f.e - 1) - 1;
/** smallest normal exponent */
export const emin = (f: FloatFormat) => 1 - bias(f);
/** largest finite value */
export function maxValue(f: FloatFormat): number {
  if (f.maxOverride !== undefined) return f.maxOverride;
  const emax = 2 ** f.e - 2 - bias(f);
  return (2 - 2 ** -f.m) * 2 ** emax;
}
export const minNormal = (f: FloatFormat) => 2 ** emin(f);
export const minSubnormal = (f: FloatFormat) => 2 ** (emin(f) - f.m);
/** the gap between 1 and the next value: 2^-m */
export const epsilon = (f: FloatFormat) => 2 ** -f.m;

/** round half to even */
function rne(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
}

/** `x` rounded to the nearest value of format `f` (ties to even), with its overflow rule */
export function roundTo(x: number, f: FloatFormat): number {
  if (f.id === "fp64" || !Number.isFinite(x) || x === 0) return x;
  if (f.id === "fp32") return Math.fround(x);
  const s = Math.sign(x);
  const a = Math.abs(x);
  const e = Math.max(Math.floor(Math.log2(a)), emin(f));
  const q = 2 ** (e - f.m);
  let r = rne(a / q) * q;
  const max = maxValue(f);
  if (r > max) r = f.overflow === "inf" && a >= max + 2 ** (Math.floor(Math.log2(max)) - f.m - 1) ? Infinity : max;
  return s * r;
}

/** sign, exponent and mantissa bits of `x` in format `f` (after rounding), most significant first */
export function bitsOf(x: number, f: FloatFormat): { sign: number; exp: number[]; man: number[]; value: number } {
  const v = roundTo(x, f);
  const sign = v < 0 || Object.is(v, -0) ? 1 : 0;
  const a = Math.abs(v);
  const width = (n: number, w: number) => Array.from({ length: w }, (_, i) => (n >> (w - 1 - i)) & 1);
  if (!Number.isFinite(a)) return { sign, exp: width(2 ** f.e - 1, f.e), man: width(0, Math.min(f.m, 30)), value: v };
  if (a === 0) return { sign, exp: width(0, f.e), man: width(0, Math.min(f.m, 30)), value: v };
  let e = Math.floor(Math.log2(a));
  let frac: number;
  let field: number;
  if (e < emin(f)) {
    field = 0; // subnormal
    frac = a / 2 ** emin(f);
    e = emin(f);
  } else {
    field = e + bias(f);
    frac = a / 2 ** e - 1;
  }
  // mantissa bits: show at most 30 (FP64 has 52; the display truncates, the value doesn't)
  const shown = Math.min(f.m, 30);
  const man = Array.from({ length: shown }, (_, i) => Math.floor(frac * 2 ** (i + 1)) % 2);
  return { sign, exp: width(field, f.e), man, value: v };
}

/** every non-negative finite value of a small format (8 bits or fewer), ascending */
export function allValues(f: FloatFormat): number[] {
  if (f.e + f.m > 7) throw new Error(`${f.name} has too many values to list`);
  const out = new Set<number>();
  for (let field = 0; field < 2 ** f.e; field++)
    for (let man = 0; man < 2 ** f.m; man++) {
      const v = field === 0 ? man * 2 ** (emin(f) - f.m) : (1 + man / 2 ** f.m) * 2 ** (field - bias(f));
      if (v <= maxValue(f)) out.add(v);
    }
  return [...out].sort((a, b) => a - b);
}

/** relative rounding error of `x` in format `f` */
export const relError = (x: number, f: FloatFormat) => Math.abs(roundTo(x, f) - x) / Math.abs(x);

export type SumMode = "naive" | "wide" | "kahan";
/**
 * The running total of `values` added one by one, every number held in `f`:
 * "naive" rounds the total to `f` after each addition; "wide" keeps the total in FP32 (what tensor cores do for
 * FP16/BF16 inputs); "kahan" stays in `f` but carries the lost low bits in a compensation term.
 * Returns the total after each step.
 */
export function runningSum(values: number[], f: FloatFormat, mode: SumMode): number[] {
  const r = (x: number) => roundTo(x, f);
  const out: number[] = [];
  let s = 0;
  let c = 0;
  for (const v0 of values) {
    const v = r(v0);
    if (mode === "naive") s = r(s + v);
    else if (mode === "wide") s = Math.fround(s + v);
    else {
      const y = r(v - c);
      const t = r(s + y);
      c = r(r(t - s) - y);
      s = t;
    }
    out.push(s);
  }
  return out;
}

/** arithmetic intensity (FLOP per byte) where the roofline's slope meets the flat compute ceiling */
export const ridgePoint = (peakTflops: number, bandwidthGBs: number) => (peakTflops * 1e12) / (bandwidthGBs * 1e9);

/**
 * Upper bound on tokens per second when decoding one sequence of an LLM: every token reads every weight once, so
 * the time per token is at least (parameters × bytes per parameter) / memory bandwidth.
 */
export const decodeCeiling = (bandwidthGBs: number, paramsBillions: number, bytesPerParam: number) =>
  bandwidthGBs / (paramsBillions * bytesPerParam);

/** bytes per parameter of common weight formats */
export const BYTES: Record<string, number> = { fp32: 4, bf16: 2, fp16: 2, fp8: 1, int8: 1, int4: 0.5, fp4: 0.5 };
