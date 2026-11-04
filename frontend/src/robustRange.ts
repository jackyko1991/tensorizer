// Colour range for a scalar shown over many files/steps: the [p, 1−p] quantiles instead of min/max, so a few extreme
// values (one collateral vessel, one stagnation point) do not wash out the colours of everything else. Values outside
// the range are drawn saturated, and the legend marks those ends with ≤ / ≥.

export interface Range {
  lo: number;
  hi: number;
  min: number;
  max: number;
  /** True when some values lie below lo / above hi (the legend shows "≤ lo" / "≥ hi"). */
  clipLow: boolean;
  clipHigh: boolean;
}

const MAX_SAMPLE = 200_000;

/** Quantile range over all chunks (finite values only), from an even-stride sample of at most 200 000 values.
 * Falls back to [min, max] when the quantiles coincide but the data are not constant (a mostly-constant field). */
export function robustRange(chunks: ArrayLike<number>[], p = 0.02): Range | null {
  let n = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const c of chunks)
    for (let i = 0; i < c.length; i++) {
      const v = c[i];
      if (!Number.isFinite(v)) continue;
      n++;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  if (n === 0) return null;
  if (min === max) return { lo: min, hi: max, min, max, clipLow: false, clipHigh: false };
  const stride = Math.max(1, Math.ceil(n / MAX_SAMPLE));
  const sample: number[] = [];
  let k = 0;
  for (const c of chunks)
    for (let i = 0; i < c.length; i++) {
      const v = c[i];
      if (!Number.isFinite(v)) continue;
      if (k++ % stride === 0) sample.push(v);
    }
  sample.sort((a, b) => a - b);
  const at = (q: number) => sample[Math.min(sample.length - 1, Math.max(0, Math.round(q * (sample.length - 1))))];
  let lo = at(p);
  let hi = at(1 - p);
  if (!(hi > lo)) {
    lo = min;
    hi = max;
  }
  return { lo, hi, min, max, clipLow: min < lo, clipHigh: max > hi };
}
