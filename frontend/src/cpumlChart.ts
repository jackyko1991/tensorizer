// Pure helpers of the "Migrating CPU-only ML to tensors" tab (no DOM), unit-tested in frontend/tests/cpumlChart.test.ts.

/** Speed-up axis: 10^LO_DEC .. 10^HI_DEC times, log scale, with 1x (as fast as scikit-learn) inside. */
export const LO_DEC = -2;
export const HI_DEC = 3;

/** Position of a speed-up on the axis, in percent of its width (clamped to the ends). */
export function speedupPos(x: number): number {
  const e = Math.log10(Math.max(x, 1e-12));
  const c = Math.min(Math.max(e, LO_DEC), HI_DEC);
  return (100 * (c - LO_DEC)) / (HI_DEC - LO_DEC);
}

/** The bar from 1x to x: { left, width } in percent; it grows right when the tensor code is faster, left when slower. */
export function barSpan(x: number): { left: number; width: number } {
  const one = speedupPos(1);
  const p = speedupPos(x);
  return { left: Math.min(one, p), width: Math.abs(p - one) };
}

/** Axis ticks: every power of ten from 10^LO_DEC to 10^HI_DEC. */
export function ticks(): number[] {
  return Array.from({ length: HI_DEC - LO_DEC + 1 }, (_, k) => 10 ** (LO_DEC + k));
}

/** 0.01 -> "0.01×", 1 -> "1×", 1000 -> "1000×". */
export function fmtTick(v: number): string {
  return `${v >= 1 ? Math.round(v) : v}×`;
}

/** Speed-up as text: 2.345 -> "×2.3", 0.0712 -> "×0.071", 154.2 -> "×154". */
export function fmtX(x: number): string {
  return x >= 10 ? `×${Math.round(x).toLocaleString("en")}` : `×${x.toPrecision(2)}`;
}

/** Seconds as text: 2.5 -> "2.5 s", 0.0123 -> "12 ms", 4.2e-5 -> "42 µs". */
export function fmtTime(s: number): string {
  if (s >= 1) return `${s.toPrecision(2)} s`;
  const [v, unit] = s >= 1e-3 ? [s * 1e3, "ms"] : [s * 1e6, "µs"];
  return `${v >= 100 ? Math.round(v) : v.toPrecision(2)} ${unit}`;
}

/** A metric value for the agreement table: fractions as percent, small numbers in scientific notation. */
export function fmtMetric(v: number | null | undefined, kind: "pct" | "num" | "sci" | "int" | "time"): string {
  if (v == null || !Number.isFinite(v)) return "–";
  if (kind === "pct") return `${(100 * v).toFixed(1)} %`;
  if (kind === "int") return String(Math.round(v));
  if (kind === "time") return fmtTime(v);
  if (kind === "sci") {
    if (v === 0) return "0";
    const [m, e] = v.toExponential(1).split("e");
    return `${m}e${Number(e)}`;
  }
  return Math.abs(v) >= 100 ? v.toFixed(2) : v.toPrecision(6);
}
