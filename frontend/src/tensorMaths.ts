// Pure helpers of the Concept page's "Tensor maths" tab (no DOM), unit-tested in frontend/tests/tensorMaths.test.ts.

/** Multiplications of an (a x b) by (b x c) matrix product: one multiply per (i, j, k) triple. */
export const matmulMults = (a: number, b: number, c: number): number => a * b * c;

/**
 * Cost (scalar multiplications) of a chain of matrix products, multiplied in the order given by `split`.
 * `dims` has n + 1 entries for n matrices (matrix i is dims[i] x dims[i+1]); `split` is a nested pair of
 * indices such as [[0, 1], 2] for (A B) C or [0, [1, 2]] for A (B C).
 */
export type Order = number | [Order, Order];
export function chainCost(dims: number[], order: Order): { cost: number; rows: number; cols: number } {
  if (typeof order === "number") return { cost: 0, rows: dims[order], cols: dims[order + 1] };
  const l = chainCost(dims, order[0]);
  const r = chainCost(dims, order[1]);
  if (l.cols !== r.rows) throw new Error(`shapes do not match: ${l.cols} vs ${r.rows}`);
  return { cost: l.cost + r.cost + matmulMults(l.rows, l.cols, r.cols), rows: l.rows, cols: r.cols };
}

/** The cheapest parenthesisation of a matrix chain (classic O(n^3) dynamic programme) and its cost. */
export function bestChain(dims: number[]): { cost: number; order: Order } {
  const n = dims.length - 1;
  const cost: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const cut: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let len = 2; len <= n; len++)
    for (let i = 0; i + len - 1 < n; i++) {
      const j = i + len - 1;
      cost[i][j] = Infinity;
      for (let k = i; k < j; k++) {
        const c = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1];
        if (c < cost[i][j]) {
          cost[i][j] = c;
          cut[i][j] = k;
        }
      }
    }
  const build = (i: number, j: number): Order => (i === j ? i : [build(i, cut[i][j]), build(cut[i][j] + 1, j)]);
  return { cost: n ? cost[0][n - 1] : 0, order: n ? build(0, n - 1) : 0 };
}

/** "(AB)C"-style text of an order, naming matrices A, B, C, ... */
export function orderText(order: Order, top = true): string {
  if (typeof order === "number") return String.fromCharCode(65 + order);
  const s = orderText(order[0], false) + orderText(order[1], false);
  return top ? s : `(${s})`;
}

/** Roofline (Williams, Waterman & Patterson 2009): attainable FLOP/s = min(peak, bandwidth x intensity). */
export const attainable = (intensity: number, peak: number, bandwidth: number): number =>
  Math.min(peak, bandwidth * intensity);

/** Ridge point: the intensity (FLOP/byte) above which a kernel can reach the peak. */
export const ridge = (peak: number, bandwidth: number): number => peak / bandwidth;

/**
 * FLOPs and bytes moved (each operand read once, the result written once) of common kernels on n-sized data,
 * for `b` bytes per number. Intensity = flops / bytes.
 */
export function kernelCost(kind: "add" | "dot" | "gemv" | "gemm" | "conv", n: number, b = 4, ch = 64) {
  switch (kind) {
    case "add": // z = x + y: n adds, read x, y, write z
      return { flops: n, bytes: 3 * n * b };
    case "dot": // x . y: n multiplies + n adds, read x, y
      return { flops: 2 * n, bytes: 2 * n * b };
    case "gemv": // y = A x: 2n^2 FLOPs, read A (n^2) and x, write y
      return { flops: 2 * n * n, bytes: (n * n + 2 * n) * b };
    case "gemm": // C = A B: 2n^3 FLOPs, read A, B, write C
      return { flops: 2 * n ** 3, bytes: 3 * n * n * b };
    case "conv": // 3x3 convolution, ch -> ch channels on an n-pixel image: read input and weights, write output
      return { flops: 2 * 9 * ch * ch * n, bytes: (2 * ch * n + 9 * ch * ch) * b };
  }
}

export const intensity = (c: { flops: number; bytes: number }): number => c.flops / c.bytes;

/**
 * Bytes moved by a chain of k element-wise operations on n numbers: run one by one, every step reads its input
 * and writes its output (2kn numbers); fused into one kernel, the chain reads once and writes once (2n).
 */
export const elementwiseBytes = (k: number, n: number, b = 4, fused = false): number => (fused ? 2 : 2 * k) * n * b;
