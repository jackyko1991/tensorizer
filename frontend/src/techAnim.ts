/**
 * Pure step logic of the Coding Skills "Techniques" animations: which stage a tick belongs to and the small
 * tensor computations each picture shows (broadcasting, einsum contraction, batching, RK4 stages, eigh / solve,
 * conv2d stencils, FFT evolution, autograd, complex phases). Drawn by components/TechniqueAnim; every
 * computation is checked against an explicit reference in tests/techAnim.test.ts.
 */

export type TechCase =
  | "tdd"
  | "tensors"
  | "broadcasting"
  | "einsum"
  | "batching"
  | "rk4"
  | "linalg"
  | "stencils"
  | "fft"
  | "autograd"
  | "complex";

export const TECH_CASES: TechCase[] = [
  "tdd",
  "tensors",
  "broadcasting",
  "einsum",
  "batching",
  "rk4",
  "linalg",
  "stencils",
  "fft",
  "autograd",
  "complex",
];

/** Stage plan of one animation: [step caption index, ticks], and the tick length in ms at 1x. */
interface Plan {
  stages: [number, number][];
  tick: number;
}

const RK4_STEPS = 3;

export const PLAN: Record<TechCase, Plan> = {
  tdd: { stages: [[0, 4], [1, 3], [2, 4], [3, 3], [4, 3]], tick: 500 },
  tensors: { stages: [[0, 6], [1, 5], [2, 6]], tick: 500 },
  broadcasting: { stages: [[0, 3], [1, 4], [2, 3]], tick: 700 },
  einsum: { stages: [[0, 3], [1, 4], [2, 4]], tick: 850 },
  batching: { stages: [[0, 2], [1, 3], [2, 12]], tick: 420 },
  rk4: {
    stages: Array.from({ length: RK4_STEPS * 5 }, (_, i) => [i % 5, 1] as [number, number]),
    tick: 750,
  },
  linalg: { stages: [[0, 16], [1, 3], [2, 4]], tick: 420 },
  stencils: { stages: [[0, 6], [1, 3]], tick: 750 },
  fft: { stages: [[0, 2], [1, 2], [2, 5], [3, 3]], tick: 700 },
  autograd: { stages: [[0, 3], [1, 3], [2, 2]], tick: 900 },
  complex: { stages: [[0, 48]], tick: 150 },
};

/** Number of distinct step captions (skills.anim.<id>.s0 .. s<n-1>). */
export const stepCount = (c: TechCase) => Math.max(...PLAN[c].stages.map(([k]) => k)) + 1;

/** Ticks of one run (the picture then holds before the cycle restarts). */
export const totalFrames = (c: TechCase) => PLAN[c].stages.reduce((s, [, n]) => s + n, 0);

/** Stage of tick g (clamped to the last tick): index in the plan, step caption, tick within it, its length. */
export function stageAt(c: TechCase, g: number): { idx: number; step: number; t: number; n: number } {
  const st = PLAN[c].stages;
  let r = Math.min(Math.max(Math.floor(g), 0), totalFrames(c) - 1);
  for (let idx = 0; idx < st.length; idx++) {
    const [step, n] = st[idx];
    if (r < n) return { idx, step, t: r, n };
    r -= n;
  }
  const last = st.length - 1;
  return { idx: last, step: st[last][0], t: st[last][1] - 1, n: st[last][1] };
}

// ---- dense tensors -------------------------------------------------------------------------------------

/** A row-major dense tensor. */
export interface Tensor {
  shape: number[];
  data: number[];
}

/** Contiguous (row-major) strides, in elements: shape (2, 3) -> (3, 1). */
export function contiguousStrides(shape: number[]): number[] {
  const s = Array(shape.length).fill(1);
  for (let d = shape.length - 2; d >= 0; d--) s[d] = s[d + 1] * shape[d + 1];
  return s;
}

/** Memory offset of a multi-index: sum_d idx_d stride_d. */
export const flatOffset = (idx: number[], strides: number[]) => idx.reduce((o, i, d) => o + i * strides[d], 0);

/** Multi-index of the k-th element in row-major order. */
export function unravel(k: number, shape: number[]): number[] {
  const idx = Array(shape.length).fill(0);
  for (let d = shape.length - 1; d >= 0; d--) {
    idx[d] = k % shape[d];
    k = Math.floor(k / shape[d]);
  }
  return idx;
}

export const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** Tensor from nested arrays (rectangular). */
export function tensor(nested: unknown): Tensor {
  const shape: number[] = [];
  let x: unknown = nested;
  while (Array.isArray(x)) {
    shape.push(x.length);
    x = x[0];
  }
  return { shape, data: (nested as unknown[]).flat(Infinity) as number[] };
}

/** Nested 2-D array of a (R, C) tensor. */
export const rows2 = (t: Tensor) =>
  Array.from({ length: t.shape[0] }, (_, i) => t.data.slice(i * t.shape[1], (i + 1) * t.shape[1]));

// ---- tensors and devices -------------------------------------------------------------------------------

export const devX: Tensor = tensor([
  [1, 2, 3],
  [4, 5, 6],
]);

/** The ops the picture runs on the device: x*2, then +1, then .sum(dim=1) -> shape (2,). */
export function deviceOps(x: Tensor): { label: string; t: Tensor }[] {
  const y = { shape: x.shape, data: x.data.map((v) => v * 2) };
  const z = { shape: x.shape, data: y.data.map((v) => v + 1) };
  return [
    { label: "y = x * 2", t: y },
    { label: "z = y + 1", t: z },
    { label: "s = z.sum(dim=1)", t: sumDim(z, 1) },
  ];
}

/** Sum over one axis of a 2-D tensor. */
export function sumDim(t: Tensor, dim: 0 | 1): Tensor {
  const r = rows2(t);
  return dim === 1
    ? { shape: [t.shape[0]], data: r.map((row) => row.reduce((a, b) => a + b, 0)) }
    : { shape: [t.shape[1]], data: r[0].map((_, j) => r.reduce((a, row) => a + row[j], 0)) };
}

// ---- broadcasting --------------------------------------------------------------------------------------

/** PyTorch / NumPy broadcast shape: align from the right, sizes must match or be 1. Throws otherwise. */
export function broadcastShape(a: number[], b: number[]): number[] {
  const n = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let k = 1; k <= n; k++) {
    const x = a[a.length - k] ?? 1;
    const y = b[b.length - k] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new Error(`shapes (${a}) and (${b}) do not broadcast`);
    out.unshift(Math.max(x, y));
  }
  return out;
}

/** Strides of the expanded view (no copy): size-1 or missing axes get stride 0. */
export function expandStrides(shape: number[], target: number[]): number[] {
  const st = contiguousStrides(shape);
  return target.map((size, d) => {
    const k = d - (target.length - shape.length);
    if (k < 0) return 0;
    return shape[k] === 1 && size !== 1 ? 0 : st[k];
  });
}

/** Element-wise op with broadcasting, read through the stride-0 views. */
export function broadcastOp(a: Tensor, b: Tensor, op: (x: number, y: number) => number): Tensor {
  const shape = broadcastShape(a.shape, b.shape);
  const sa = expandStrides(a.shape, shape);
  const sb = expandStrides(b.shape, shape);
  const data = Array.from({ length: numel(shape) }, (_, k) => {
    const idx = unravel(k, shape);
    return op(a.data[flatOffset(idx, sa)], b.data[flatOffset(idx, sb)]);
  });
  return { shape, data };
}

export const bcA: Tensor = tensor([[1], [2], [3]]); // (A, 1) = (3, 1)
export const bcB: Tensor = tensor([[10, 20, 30, 40]]); // (1, L) = (1, 4)

// ---- einsum --------------------------------------------------------------------------------------------

/** torch.einsum on small dense tensors, by brute force over every index assignment. */
export function einsum(spec: string, ...ts: Tensor[]): Tensor {
  const [lhs, rhs] = spec.replace(/\s/g, "").split("->");
  const ins = lhs.split(",");
  if (ins.length !== ts.length) throw new Error("operand count");
  const size: Record<string, number> = {};
  ins.forEach((s, n) =>
    [...s].forEach((c, d) => {
      if (size[c] != null && size[c] !== ts[n].shape[d]) throw new Error(`size of ${c}`);
      size[c] = ts[n].shape[d];
    }),
  );
  const all = Object.keys(size);
  const outIdx = [...rhs];
  const outShape = outIdx.map((c) => size[c]);
  const outSt = contiguousStrides(outShape);
  const data = Array(numel(outShape)).fill(0);
  const strides = ts.map((t) => contiguousStrides(t.shape));
  const total = numel(all.map((c) => size[c]));
  for (let k = 0; k < total; k++) {
    const v = unravel(
      k,
      all.map((c) => size[c]),
    );
    const at: Record<string, number> = {};
    all.forEach((c, i) => (at[c] = v[i]));
    let p = 1;
    ts.forEach((t, n) => (p *= t.data[flatOffset([...ins[n]].map((c) => at[c]), strides[n])]));
    data[flatOffset(outIdx.map((c) => at[c]), outSt)] += p;
  }
  return { shape: outShape, data };
}

export const esA: Tensor = tensor([
  [1, 2, 0],
  [0, 1, 3],
]); // (i, j) = (2, 3)
export const esB: Tensor = tensor([
  [2, 1],
  [1, 0],
  [0, 2],
]); // (j, k) = (3, 2)

/** Output cell order of the contraction stage: (i, k) row-major. */
export const esCell = (t: number) => [Math.floor(t / 2), t % 2] as const;

/** The products A_ij B_jk summed into C_ik (the terms shown for one output cell). */
export function contractTerms(A: Tensor, B: Tensor, i: number, k: number): [number, number][] {
  const a = rows2(A);
  const b = rows2(B);
  return a[i].map((v, j) => [v, b[j][k]] as [number, number]);
}

/** Levi-Civita symbol eps_ijk as a (3, 3, 3) tensor. */
export function leviCivita(): Tensor {
  const data = Array(27).fill(0);
  const set = (i: number, j: number, k: number, v: number) => (data[i * 9 + j * 3 + k] = v);
  [[0, 1, 2], [1, 2, 0], [2, 0, 1]].forEach(([i, j, k]) => set(i, j, k, 1));
  [[0, 2, 1], [2, 1, 0], [1, 0, 2]].forEach(([i, j, k]) => set(i, j, k, -1));
  return { shape: [3, 3, 3], data };
}

// ---- batching ------------------------------------------------------------------------------------------

export const BATCH = { B: 4, v0: 10, g: 9.81, dt: 0.2, T: 12 };
export const batchAngles = [15, 35, 55, 75]; // degrees

/** Initial states (x, y, vx, vy), one row per launch angle. */
export const batchY0 = (): number[][] =>
  batchAngles.map((a) => {
    const r = (a * Math.PI) / 180;
    return [0, 0, BATCH.v0 * Math.cos(r), BATCH.v0 * Math.sin(r)];
  });

/** One semi-implicit Euler step of one projectile; on hitting the ground it stops there. */
export function projStep(s: number[], dt = BATCH.dt, g = BATCH.g): number[] {
  const vy = s[3] - g * dt;
  const x = s[0] + s[2] * dt;
  const y = s[1] + vy * dt;
  return y < 0 ? [x, 0, 0, 0] : [x, y, s[2], vy];
}

/** The same step on the whole (B, 4) batch, written column-wise like tensor code (with a torch.where mask). */
export function projStepBatch(Y: number[][], dt = BATCH.dt, g = BATCH.g): number[][] {
  const col = (j: number) => Y.map((r) => r[j]);
  const vy = col(3).map((v) => v - g * dt);
  const x = col(0).map((v, b) => v + col(2)[b] * dt);
  const y = col(1).map((v, b) => v + vy[b] * dt);
  const landed = y.map((v) => v < 0);
  return Y.map((_, b) => (landed[b] ? [x[b], 0, 0, 0] : [x[b], y[b], Y[b][2], vy[b]]));
}

/** traj[t] = (B, 4) state after t batched steps, t = 0 .. T. */
export const batchTraj: number[][][] = (() => {
  const out = [batchY0()];
  for (let t = 0; t < BATCH.T; t++) out.push(projStepBatch(out[t]));
  return out;
})();

// ---- RK4 -----------------------------------------------------------------------------------------------

export const rkF = (t: number, y: number) => Math.cos(t) * y; // dy/dt = y cos t
export const rkExact = (t: number) => Math.exp(Math.sin(t)); // y(0) = 1
export const RK = { h: 0.8, y0: 1, steps: RK4_STEPS };

/** The four samples of one classical RK4 step: points (t, y) where f is evaluated, slopes k, and y_{n+1}. */
export function rk4Stages(f: (t: number, y: number) => number, t: number, y: number, h: number) {
  const k1 = f(t, y);
  const p2: [number, number] = [t + h / 2, y + (h / 2) * k1];
  const k2 = f(...p2);
  const p3: [number, number] = [t + h / 2, y + (h / 2) * k2];
  const k3 = f(...p3);
  const p4: [number, number] = [t + h, y + h * k3];
  const k4 = f(...p4);
  return {
    pts: [[t, y], p2, p3, p4] as [number, number][],
    k: [k1, k2, k3, k4],
    y1: y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4),
  };
}

/** (t_n, y_n) for n = 0 .. steps. */
export function rk4Path(f: (t: number, y: number) => number, y0: number, h: number, steps: number): [number, number][] {
  const out: [number, number][] = [[0, y0]];
  for (let n = 0; n < steps; n++) {
    const [t, y] = out[n];
    out.push([t + h, rk4Stages(f, t, y, h).y1]);
  }
  return out;
}

// ---- linear algebra ------------------------------------------------------------------------------------

export const LA_H: [[number, number], [number, number]] = [
  [2, 1],
  [1, 2],
];
export const LA_B: [number, number] = [2.5, 2];

/** torch.linalg.eigh of a symmetric 2 x 2 matrix: ascending eigenvalues, unit eigenvectors as columns. */
export function eigh2(M: number[][]): { E: [number, number]; V: [[number, number], [number, number]] } {
  const [a, b, c] = [M[0][0], M[0][1], M[1][1]];
  const m = (a + c) / 2;
  const d = Math.hypot((a - c) / 2, b);
  const E: [number, number] = [m - d, m + d];
  const vec = (l: number): [number, number] => {
    const v: [number, number] = Math.abs(b) > 1e-12 ? [b, l - a] : a <= c === (l === E[0]) ? [1, 0] : [0, 1];
    const n = Math.hypot(...v);
    return [v[0] / n, v[1] / n];
  };
  const v1 = vec(E[0]);
  const v2 = vec(E[1]);
  return { E, V: [[v1[0], v2[0]], [v1[1], v2[1]]] };
}

/** torch.linalg.solve for 2 x 2 (Cramer's rule). */
export function solve2(A: number[][], b: number[]): [number, number] {
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  return [(b[0] * A[1][1] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - b[0] * A[1][0]) / det];
}

export const matVec2 = (A: number[][], v: number[]): [number, number] => [
  A[0][0] * v[0] + A[0][1] * v[1],
  A[1][0] * v[0] + A[1][1] * v[1],
];

/** Unit vector at the t-th of 16 directions (the probing stage). */
export const laDir = (t: number): [number, number] => [Math.cos((t * Math.PI) / 8), Math.sin((t * Math.PI) / 8)];

/** True when Hv is parallel to v (v is an eigenvector). */
export function isEigen(M: number[][], v: number[], tol = 1e-9): boolean {
  const w = matVec2(M, v);
  return Math.abs(w[0] * v[1] - w[1] * v[0]) < tol;
}

// ---- stencils ------------------------------------------------------------------------------------------

export const LAP_KERNEL = [
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0],
];

export const stU: number[][] = Array.from({ length: 7 }, (_, i) =>
  Array.from({ length: 7 }, (_, j) => Math.max(0, 3 - Math.abs(i - 3) - Math.abs(j - 3))),
);

/** torch.nn.functional.conv2d (cross-correlation) of one channel with zero padding `pad`. */
export function conv2d(u: number[][], k: number[][], pad: number): number[][] {
  const H = u.length;
  const W = u[0].length;
  const kh = k.length;
  const kw = k[0].length;
  const at = (i: number, j: number) => (i < 0 || j < 0 || i >= H || j >= W ? 0 : u[i][j]);
  const oh = H + 2 * pad - kh + 1;
  const ow = W + 2 * pad - kw + 1;
  return Array.from({ length: oh }, (_, i) =>
    Array.from({ length: ow }, (_, j) => {
      let s = 0;
      for (let a = 0; a < kh; a++) for (let b = 0; b < kw; b++) s += k[a][b] * at(i - pad + a, j - pad + b);
      return s;
    }),
  );
}

/** Output cells the sliding window visits in the first stage (row, column). */
export const stPath: [number, number][] = [
  [1, 1],
  [1, 4],
  [2, 3],
  [3, 3],
  [4, 2],
  [5, 5],
];

// ---- FFT -----------------------------------------------------------------------------------------------

export const FFT_N = 32;
export const FFT_T = 6; // free-evolution time shown

/** Complex numbers as [re, im]. */
export type C = [number, number];
export const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
export const cexp = (phi: number): C => [Math.cos(phi), Math.sin(phi)];
export const cabs = (a: C) => Math.hypot(a[0], a[1]);

/** Discrete Fourier transform, torch.fft.fft convention (no normalisation forward). */
export function dft(x: C[], inverse = false): C[] {
  const n = x.length;
  const sgn = inverse ? 1 : -1;
  return Array.from({ length: n }, (_, k) => {
    let re = 0;
    let im = 0;
    for (let j = 0; j < n; j++) {
      const [c, s] = cexp((sgn * 2 * Math.PI * j * k) / n);
      re += x[j][0] * c - x[j][1] * s;
      im += x[j][0] * s + x[j][1] * c;
    }
    return inverse ? ([re / n, im / n] as C) : ([re, im] as C);
  });
}

/** torch.fft.fftfreq(n, d) times 2 pi: angular wavenumbers in FFT order. */
export const fftk = (n: number, d = 1) =>
  Array.from({ length: n }, (_, j) => (2 * Math.PI * (j < Math.ceil(n / 2) ? j : j - n)) / (n * d));

/** Grid x_j = -N/2 + j (dx = 1) and a Gaussian packet moving right. */
export const fftX = Array.from({ length: FFT_N }, (_, j) => j - FFT_N / 2);
export const fftPsi0: C[] = fftX.map((x) => {
  const a = Math.exp(-((x + 6) ** 2) / (2 * 1.6 ** 2));
  return [a * Math.cos(1.2 * x), a * Math.sin(1.2 * x)];
});

/** Free evolution exp(-i k^2 t / 2) applied in Fourier space (the kinetic step of split-step). */
export function freeEvolve(psi: C[], t: number): C[] {
  const k = fftk(psi.length);
  const ph = dft(psi).map((v, j) => cmul(v, cexp(-0.5 * k[j] ** 2 * t)));
  return dft(ph, true);
}

/** Spectral derivative d/dx: multiply by i k between the two transforms. */
export function spectralDerivative(f: number[], d = 1): number[] {
  const k = fftk(f.length, d);
  const F = dft(f.map((v) => [v, 0] as C)).map((v, j) => cmul(v, [0, k[j]]));
  return dft(F, true).map((v) => v[0]);
}

/** fftshift order (negative k first) for drawing. */
export const fftShift = <T,>(a: T[]) => [...a.slice(Math.ceil(a.length / 2)), ...a.slice(0, Math.ceil(a.length / 2))];

// ---- autograd ------------------------------------------------------------------------------------------

export const AG = { GM: 1, x: [3, 4] as [number, number] };

/** Forward pass: r = |x|, phi = -GM / r. */
export function agForward(x: number[], GM = AG.GM) {
  const r = Math.hypot(...x);
  return { r, phi: -GM / r };
}

/** Backward pass by the chain rule: dphi/dphi = 1, dphi/dr = GM / r^2, dphi/dx = dphi/dr * x / r. */
export function agBackward(x: number[], GM = AG.GM) {
  const { r } = agForward(x, GM);
  const dr = GM / r ** 2;
  return { dphi: 1, dr, dx: x.map((v) => (dr * v) / r) };
}

// ---- complex phases ------------------------------------------------------------------------------------

export const CX = { nx: 61, L: 5, c: [0.8, 0.5, 0.33] };
export const cxE = (n: number) => n + 0.5; // harmonic oscillator, hbar = omega = m = 1

/** Hermite functions phi_n(x), n = 0, 1, 2 (orthonormal). */
export function hoBasis(x: number, n: number): number {
  const g = Math.exp(-(x * x) / 2) / Math.PI ** 0.25;
  if (n === 0) return g;
  if (n === 1) return Math.SQRT2 * x * g;
  return ((2 * x * x - 1) / Math.SQRT2) * g;
}

export const cxX = Array.from({ length: CX.nx }, (_, j) => -CX.L + (2 * CX.L * j) / (CX.nx - 1));

/** Normalised coefficients c_n. */
export const cxC = (() => {
  const n = Math.hypot(...CX.c);
  return CX.c.map((v) => v / n);
})();

/** c_n e^{-i E_n t}: the element-wise phase factor applied to the coefficient vector. */
export const cxPhased = (t: number): C[] => cxC.map((c, n) => cmul([c, 0], cexp(-cxE(n) * t)));

/** psi_t = einsum("xn,n->x", phi, c * exp(-1j * E * t)) and prob = |psi_t|^2 on the grid. */
export function cxPsi(t: number): { psi: C[]; prob: number[] } {
  const a = cxPhased(t);
  const psi = cxX.map((x) => {
    let re = 0;
    let im = 0;
    a.forEach((v, n) => {
      const p = hoBasis(x, n);
      re += p * v[0];
      im += p * v[1];
    });
    return [re, im] as C;
  });
  return { psi, prob: psi.map((v) => v[0] ** 2 + v[1] ** 2) };
}

/** Time of tick t of the single stage: one full period 2 pi in 48 ticks. */
export const cxTime = (t: number) => (2 * Math.PI * t) / PLAN.complex.stages[0][1];

// ---- TDD -----------------------------------------------------------------------------------------------

/** Light after each TDD step: written test (no light yet), red run, code written (still red), green, refactor. */
export const TDD_LIGHT: ("none" | "red" | "green")[] = ["none", "red", "red", "green", "green"];
