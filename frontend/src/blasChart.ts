// Pure helpers of the BLAS tab's throughput chart (no DOM), unit-tested in frontend/tests/blasChart.test.ts.

/** FLOPs of an n x n by n x n matrix product (the convention 2 n^3, as in scripts/blas_benchmark.py). */
export const gemmFlops = (n: number): number => 2 * n ** 3;

/** Throughput in GFLOP/s of an n x n matrix product that took `seconds`. */
export const gflops = (n: number, seconds: number): number => gemmFlops(n) / seconds / 1e9;

/** Arithmetic intensity of float GEMM in FLOP per byte: 2n^3 over A, B read and C read + written (4 n^2 values). */
export const gemmIntensity = (n: number, bytesPerValue: number): number => gemmFlops(n) / (4 * n * n * bytesPerValue);

/** Roofline: attainable GFLOP/s = min(peak, intensity x bandwidth). */
export const roofline = (intensity: number, peakGflops: number, bandwidthGBs: number): number =>
  Math.min(peakGflops, intensity * bandwidthGBs);

/** Map v in [lo, hi] (log scale) to [a, b]; values outside are clamped. */
export function logScale(lo: number, hi: number, a: number, b: number): (v: number) => number {
  const l0 = Math.log10(lo);
  const l1 = Math.log10(hi);
  return (v) => {
    const t = (Math.log10(Math.min(Math.max(v, lo), hi)) - l0) / (l1 - l0);
    return a + t * (b - a);
  };
}

/** Powers of ten from lo to hi inclusive (lo, hi rounded outwards to decades). */
export function decades(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(10 ** e);
  return out;
}

/** 12345 -> "12k", 1500 -> "1.5k", 335.2 -> "335", 7.4 -> "7.4". */
export function fmtGflops(x: number): string {
  if (x >= 10000) return `${Math.round(x / 1000)}k`;
  if (x >= 1000) return `${(x / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return x >= 100 ? String(Math.round(x)) : x >= 10 ? x.toFixed(0) : x.toFixed(1);
}

/** Share of the estimated peak, as a whole percentage; null without a peak. */
export const percentOfPeak = (gf: number, peak: number | null | undefined): number | null =>
  peak ? Math.round((100 * gf) / peak) : null;

/** SVG path through the points (x, y), skipping none. */
export const linePath = (pts: [number, number][]): string =>
  pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
