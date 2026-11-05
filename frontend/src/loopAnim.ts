/**
 * Pure step logic of the "For loop vs tensor" animations (Coding Skills tab): what each side shows after
 * a given number of ticks. One loop tick is one pass of the Python loop body (one element); the tensor side
 * finishes in one vectorised call (a few ticks only where the picture splits that call into stages:
 * products then sum, broadcast then fill, BLAS tiles, shifted slices). Drawn by components/LoopVsTensorAnim.
 */

export type AnimCase = "dot" | "pairwise" | "matmul" | "pendulum" | "diffusion" | "mc_pi";

export const ANIM_CASES: AnimCase[] = ["dot", "pairwise", "matmul", "pendulum", "diffusion", "mc_pi"];

/** Problem sizes used in the pictures (small, so every element is visible). */
export const SIZE = {
  dot: 12, // N
  pairwise: 8, // N -> N^2 = 64 distances
  matmul: 4, // n -> n^3 = 64 multiply-adds
  pendB: 6, // B pendulums
  pendT: 10, // time steps
  diffusion: 8, // n x n grid with a fixed boundary -> (n-2)^2 = 36 interior cells
  mc: 100, // N darts
} as const;

/** Ticks the loop side needs: one per element (N, N^2, n^3, B*T, (n-2)^2, N). */
export function loopFrames(c: AnimCase): number {
  switch (c) {
    case "dot":
      return SIZE.dot;
    case "pairwise":
      return SIZE.pairwise ** 2;
    case "matmul":
      return SIZE.matmul ** 3;
    case "pendulum":
      return SIZE.pendB * SIZE.pendT;
    case "diffusion":
      return (SIZE.diffusion - 2) ** 2;
    case "mc_pi":
      return SIZE.mc;
  }
}

/** Ticks the tensor side's picture uses (one call; split into stages for the eye; T steps for the pendulums). */
export function tensorFrames(c: AnimCase): number {
  switch (c) {
    case "dot":
      return 2; // all products, then the sum
    case "pairwise":
      return 2; // broadcast column and row, then all N^2 distances
    case "matmul":
      return 4; // 2 x 2 tiles of C, one call
    case "pendulum":
      return SIZE.pendT; // the time loop stays: one vectorised RK4 step per t
    case "diffusion":
      return 5; // four shifted slices, then the update
    case "mc_pi":
      return 1; // all darts at once
  }
}

/** Counter under the loop side: loop-body passes done after f ticks. */
export const loopCount = (c: AnimCase, f: number) => Math.min(Math.max(f, 0), loopFrames(c));

/** Counter under the tensor side: vectorised calls made after f ticks (one per time step for the pendulums). */
export function tensorCount(c: AnimCase, f: number): number {
  const g = Math.min(Math.max(f, 0), tensorFrames(c));
  return c === "pendulum" ? g : g > 0 ? 1 : 0;
}

/** Deterministic PRNG (mulberry32), so the pictures and the tests use fixed data. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ints = (r: () => number, n: number, lo: number, hi: number) =>
  Array.from({ length: n }, () => lo + Math.floor(r() * (hi - lo + 1)));

// ---- dot product ---------------------------------------------------------------------------------------

export const dotData = (() => {
  const r = rng(7);
  return { x: ints(r, SIZE.dot, 1, 5), y: ints(r, SIZE.dot, 1, 5) };
})();

/** Running sum after f loop passes: s = sum_{i<f} x_i y_i. */
export function dotLoopSum(x: number[], y: number[], f: number): number {
  let s = 0;
  for (let i = 0; i < Math.min(f, x.length); i++) s += x[i] * y[i];
  return s;
}

/** Pairwise sum (the reduction tree a vectorised kernel uses); equals the loop sum for integers. */
export function treeSum(v: number[]): number {
  let a = v.slice();
  while (a.length > 1) {
    const b: number[] = [];
    for (let i = 0; i < a.length; i += 2) b.push(a[i] + (a[i + 1] ?? 0));
    a = b;
  }
  return a[0] ?? 0;
}

// ---- pairwise distances --------------------------------------------------------------------------------

export const pairData = (() => {
  const r = rng(11);
  return Array.from({ length: SIZE.pairwise }, () => Math.round(r() * 90) / 10); // positions 0 .. 9
})();

/** (i, j) of the loop pass number k (row-major double loop). */
export const pairIJ = (k: number, n: number = SIZE.pairwise) => [Math.floor(k / n), k % n] as const;

/** Distances by broadcasting: column x[:, None] minus row x[None, :]. */
export const pairDistances = (x: number[]) => x.map((xi) => x.map((xj) => Math.abs(xi - xj)));

// ---- matrix multiply -----------------------------------------------------------------------------------

export const mmData = (() => {
  const r = rng(5);
  const n = SIZE.matmul;
  const m = () => Array.from({ length: n }, () => ints(r, n, 0, 3));
  return { A: m(), B: m() };
})();

/** (i, j, k) of loop pass number f in the triple loop i -> j -> k. */
export function mmIJK(f: number, n: number = SIZE.matmul): [number, number, number] {
  return [Math.floor(f / (n * n)), Math.floor(f / n) % n, f % n];
}

/** C after f multiply-adds of the triple loop (partial sums in the cell being worked on, null if untouched). */
export function mmLoop(A: number[][], B: number[][], f: number): (number | null)[][] {
  const n = A.length;
  const C: (number | null)[][] = A.map(() => Array(n).fill(null));
  for (let p = 0; p < Math.min(f, n ** 3); p++) {
    const [i, j, k] = mmIJK(p, n);
    C[i][j] = (C[i][j] ?? 0) + A[i][k] * B[k][j];
  }
  return C;
}

export const matmul = (A: number[][], B: number[][]) =>
  A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));

/** Tile (2 x 2 blocks, row-major) that cell (i, j) of C belongs to, for the "BLAS works in tiles" picture. */
export const mmTile = (i: number, j: number, n: number = SIZE.matmul) => {
  const h = n / 2;
  return Math.floor(i / h) * 2 + Math.floor(j / h);
};

// ---- pendulums (RK4) -----------------------------------------------------------------------------------

export const PEND = { g: 9.81, L: 1, dt: 0.1 };

/** One RK4 step for one pendulum, as in pendulum_loop. */
export function rk4Scalar(th: number, om: number, dt = PEND.dt, gL = PEND.g / PEND.L): [number, number] {
  const f = (a: number, b: number): [number, number] => [b, -gL * Math.sin(a)];
  const k1 = f(th, om);
  const k2 = f(th + 0.5 * dt * k1[0], om + 0.5 * dt * k1[1]);
  const k3 = f(th + 0.5 * dt * k2[0], om + 0.5 * dt * k2[1]);
  const k4 = f(th + dt * k3[0], om + dt * k3[1]);
  return [th + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), om + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])];
}

/** One RK4 step for all B pendulums at once on a (B, 2) state, written stage by stage like pendulum_tensor. */
export function rk4Batch(y: [number, number][], dt = PEND.dt, gL = PEND.g / PEND.L): [number, number][] {
  const f = (s: [number, number][]) => s.map(([a, b]) => [b, -gL * Math.sin(a)] as [number, number]);
  const add = (s: [number, number][], k: [number, number][], h: number) =>
    s.map(([a, b], i) => [a + h * k[i][0], b + h * k[i][1]] as [number, number]);
  const k1 = f(y);
  const k2 = f(add(y, k1, 0.5 * dt));
  const k3 = f(add(y, k2, 0.5 * dt));
  const k4 = f(add(y, k3, dt));
  return y.map(([a, b], i) => [
    a + (dt / 6) * (k1[i][0] + 2 * k2[i][0] + 2 * k3[i][0] + k4[i][0]),
    b + (dt / 6) * (k1[i][1] + 2 * k2[i][1] + 2 * k3[i][1] + k4[i][1]),
  ]);
}

/** traj[t][b] = (theta, omega) of pendulum b after t steps, t = 0 .. T. */
export const pendTraj: [number, number][][] = (() => {
  const y0: [number, number][] = Array.from({ length: SIZE.pendB }, (_, b) => [0.3 + (0.7 * b) / (SIZE.pendB - 1), 0]);
  const out = [y0];
  for (let t = 0; t < SIZE.pendT; t++) out.push(rk4Batch(out[t]));
  return out;
})();

/** Loop side after f ticks: time step t, the pendulum b being advanced, and each pendulum's step count. */
export function pendLoop(f: number, B: number = SIZE.pendB, T: number = SIZE.pendT) {
  const g = Math.min(Math.max(f, 0), B * T);
  const t = Math.floor(g / B); // completed time steps for every pendulum
  const done = g % B; // pendulums already advanced to t + 1
  const steps = Array.from({ length: B }, (_, b) => (b < done ? t + 1 : t));
  return { t: g === B * T ? T : t, current: g === B * T ? -1 : done, steps };
}

// ---- diffusion stencil ---------------------------------------------------------------------------------

export const DIFF_A = 0.2;

export const diffU0: number[][] = (() => {
  const n = SIZE.diffusion;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === 0) return 1; // hot top edge, fixed
      const d2 = (i - 4.5) ** 2 + (j - 2.5) ** 2;
      return d2 < 2.5 ? 0.9 : 0;
    }),
  );
})();

/** Loop side: the first `count` interior cells (row-major) updated with the 5-point stencil, the rest unchanged. */
export function diffusionLoop(u: number[][], count: number, a = DIFF_A): number[][] {
  const n = u.length;
  const out = u.map((r) => r.slice());
  let k = 0;
  for (let i = 1; i < n - 1; i++)
    for (let j = 1; j < n - 1; j++) {
      if (k++ >= count) return out;
      const lap = u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4 * u[i][j];
      out[i][j] = u[i][j] + a * lap;
    }
  return out;
}

/** u[r0:r0+h, c0:c0+w] as a new array. */
const slice2 = (u: number[][], r0: number, c0: number, h: number, w: number) =>
  Array.from({ length: h }, (_, i) => u[r0 + i].slice(c0, c0 + w));

/** Tensor side: the interior update from four shifted slices, as in diffusion_tensor. */
export function diffusionTensor(u: number[][], a = DIFF_A): number[][] {
  const n = u.length;
  const m = n - 2;
  const c = slice2(u, 1, 1, m, m);
  const down = slice2(u, 2, 1, m, m); // u[2:, 1:-1]
  const up = slice2(u, 0, 1, m, m); // u[:-2, 1:-1]
  const right = slice2(u, 1, 2, m, m); // u[1:-1, 2:]
  const left = slice2(u, 1, 0, m, m); // u[1:-1, :-2]
  const out = u.map((r) => r.slice());
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++)
      out[i + 1][j + 1] = c[i][j] + a * (down[i][j] + up[i][j] + right[i][j] + left[i][j] - 4 * c[i][j]);
  return out;
}

/** The four shifted slices in tensor-frame order: [row offset, column offset, label]. */
export const DIFF_SHIFTS: [number, number, string][] = [
  [1, 0, "u[2:, 1:-1]"],
  [-1, 0, "u[:-2, 1:-1]"],
  [0, 1, "u[1:-1, 2:]"],
  [0, -1, "u[1:-1, :-2]"],
];

// ---- Monte Carlo pi ------------------------------------------------------------------------------------

export function darts(n: number, seed: number): [number, number][] {
  const r = rng(seed);
  return Array.from({ length: n }, () => [r(), r()]);
}

/** 4 x (fraction of the first k darts inside the quarter circle); NaN before the first dart. */
export function piEstimate(d: [number, number][], k: number): number {
  const m = Math.min(k, d.length);
  let hits = 0;
  for (let i = 0; i < m; i++) if (d[i][0] ** 2 + d[i][1] ** 2 <= 1) hits++;
  return m ? (4 * hits) / m : NaN;
}

export const mcData = darts(SIZE.mc, 2024);
