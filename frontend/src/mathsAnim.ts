/**
 * Pure step logic of the Concept page's "Tensor maths" animations: which stage a tick belongs to, where each
 * animation sits on the page, and the small computations the pictures show (unit changes of upper and lower
 * components, a slanted basis and its metric, rotations, contractions, contraction order, the roofline).
 * Drawn by components/MathsAnim; every computation is checked against a reference in tests/mathsAnim.test.ts.
 */

export type MathsCase =
  | "address"
  | "rotate"
  | "updown"
  | "metric"
  | "photo"
  | "eqRank"
  | "eqRotate"
  | "receipt"
  | "einEx"
  | "eqDot"
  | "eqMatvec"
  | "eqMatmul"
  | "eqTrace"
  | "eqOuter"
  | "eqBatch"
  | "wire"
  | "eqLoop"
  | "eqMps"
  | "chain"
  | "eqChain"
  | "van"
  | "fusion"
  | "eqRoof"
  | "ranks";

/** Stage plan: [step caption index, ticks], and the tick length in ms at 1x. */
interface Plan {
  stages: [number, number][];
  tick: number;
}

export const MATHS_PLAN: Record<MathsCase, Plan> = {
  address: { stages: [[0, 4], [1, 4], [2, 4]], tick: 700 },
  rotate: { stages: [[0, 8], [1, 8], [2, 8]], tick: 260 },
  updown: { stages: [[0, 4], [1, 4], [2, 4], [3, 4], [4, 4]], tick: 700 },
  metric: { stages: [[0, 6], [1, 10], [2, 8]], tick: 300 },
  photo: { stages: [[0, 3], [1, 4], [2, 6], [3, 3]], tick: 600 },
  eqRank: { stages: [[0, 2], [1, 2], [2, 2], [3, 2], [4, 4]], tick: 700 },
  eqRotate: { stages: [[0, 3], [1, 3], [2, 4]], tick: 900 },
  receipt: { stages: [[0, 3], [1, 3], [2, 2]], tick: 800 },
  einEx: { stages: [[0, 3], [1, 3], [2, 2]], tick: 900 },
  eqDot: { stages: [[0, 3], [1, 3]], tick: 900 },
  eqMatvec: { stages: [[0, 6], [1, 3]], tick: 700 },
  eqMatmul: { stages: [[0, 4], [1, 3]], tick: 1000 },
  eqTrace: { stages: [[0, 3], [1, 3]], tick: 900 },
  eqOuter: { stages: [[0, 9], [1, 3]], tick: 450 },
  eqBatch: { stages: [[0, 4], [1, 4], [2, 3]], tick: 700 },
  wire: { stages: [[0, 3], [1, 3], [2, 3]], tick: 850 },
  eqLoop: { stages: [[0, 4], [1, 2], [2, 3]], tick: 900 },
  eqMps: { stages: [[0, 2], [1, 3], [2, 3], [3, 4]], tick: 900 },
  chain: { stages: [[0, 2], [1, 4], [2, 4], [3, 3]], tick: 800 },
  eqChain: { stages: [[0, 12], [1, 2], [2, 4]], tick: 320 },
  van: { stages: [[0, 12], [1, 12]], tick: 300 },
  fusion: { stages: [[0, 9], [1, 3], [2, 3]], tick: 650 },
  eqRoof: { stages: [[0, 13], [1, 4]], tick: 450 },
  ranks: { stages: [[0, 2], [1, 4], [2, 3]], tick: 800 },
};

export const MATHS_CASES = Object.keys(MATHS_PLAN) as MathsCase[];

/** Where each animation sits: section, and the Feynman part it follows ("eq" = inside an equation card). */
export const MATHS_PLACE: Record<MathsCase, { section: string; after: string }> = {
  address: { section: "tensor", after: "analogy" },
  rotate: { section: "tensor", after: "precise" },
  updown: { section: "tensor", after: "updown" },
  metric: { section: "tensor", after: "updown" },
  photo: { section: "tensor", after: "example" },
  eqRank: { section: "tensor", after: "eq" },
  eqRotate: { section: "tensor", after: "eq" },
  receipt: { section: "einsum", after: "analogy" },
  einEx: { section: "einsum", after: "example" },
  eqDot: { section: "einsum", after: "eq" },
  eqMatvec: { section: "einsum", after: "eq" },
  eqMatmul: { section: "einsum", after: "eq" },
  eqTrace: { section: "einsum", after: "eq" },
  eqOuter: { section: "einsum", after: "eq" },
  eqBatch: { section: "einsum", after: "eq" },
  wire: { section: "penrose", after: "example" },
  eqLoop: { section: "penrose", after: "eq" },
  eqMps: { section: "penrose", after: "eq" },
  chain: { section: "topo", after: "precise" },
  eqChain: { section: "topo", after: "eq" },
  van: { section: "roof", after: "analogy" },
  fusion: { section: "roof", after: "fusion" },
  eqRoof: { section: "roof", after: "eq" },
  ranks: { section: "recap", after: "mis" },
};

/** Animations of a section placed after a given part, in page order. */
export const animsAfter = (section: string, after: string) =>
  MATHS_CASES.filter((c) => MATHS_PLACE[c].section === section && MATHS_PLACE[c].after === after);

/** Number of distinct step captions (maths.anim.<id>.s0 .. s<n-1>). */
export const mStepCount = (c: MathsCase) => Math.max(...MATHS_PLAN[c].stages.map(([k]) => k)) + 1;

/** Ticks of one run (the picture then holds before the cycle restarts). */
export const mFrames = (c: MathsCase) => MATHS_PLAN[c].stages.reduce((s, [, n]) => s + n, 0);

/** Stage of tick g (clamped): index in the plan, step caption, tick within it, its length. */
export function mStage(c: MathsCase, g: number): { idx: number; step: number; t: number; n: number } {
  const st = MATHS_PLAN[c].stages;
  let r = Math.min(Math.max(Math.floor(g), 0), mFrames(c) - 1);
  for (let idx = 0; idx < st.length; idx++) {
    const [step, n] = st[idx];
    if (r < n) return { idx, step, t: r, n };
    r -= n;
  }
  const last = st.length - 1;
  return { idx: last, step: st[last][0], t: st[last][1] - 1, n: st[last][1] };
}

/** First tick of a step (for jump buttons). */
export function mStepStart(c: MathsCase, step: number): number {
  let g = 0;
  for (const [k, n] of MATHS_PLAN[c].stages) {
    if (k === step) return g;
    g += n;
  }
  return 0;
}

// ---- small linear algebra -----------------------------------------------------------------------------

export const matVec = (A: number[][], v: number[]) => A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
export const matMul = (A: number[][], B: number[][]) =>
  A.map((row) => B[0].map((_, k) => row.reduce((s, a, j) => s + a * B[j][k], 0)));
export const transpose = (A: number[][]) => A[0].map((_, j) => A.map((row) => row[j]));
export const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
/** Rows of a flat (..., C) array. */
export const rows = (t: { shape: number[]; data: number[] }) => {
  const c = t.shape[t.shape.length - 1];
  return Array.from({ length: t.data.length / c }, (_, r) => t.data.slice(r * c, r * c + c));
};

/** Matrix rank by Gaussian elimination with partial pivoting. */
export function matrixRank(M: number[][], tol = 1e-9): number {
  const A = M.map((r) => [...r]);
  let rank = 0;
  for (let c = 0; c < A[0].length && rank < A.length; c++) {
    let p = rank;
    for (let r = rank + 1; r < A.length; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < tol) continue;
    [A[rank], A[p]] = [A[p], A[rank]];
    for (let r = rank + 1; r < A.length; r++) {
      const f = A[r][c] / A[rank][c];
      for (let k = c; k < A[0].length; k++) A[r][k] -= f * A[rank][k];
    }
    rank++;
  }
  return rank;
}

// ---- 1. what a tensor is ------------------------------------------------------------------------------

/** The address picture: shape and the picked position at each rank. */
export const ADDRESS = [
  { shape: [6], at: [3] },
  { shape: [4, 5], at: [2, 1] },
  { shape: [3, 4, 5], at: [1, 2, 1] },
];

/** Rotating the axes by theta: components in the turned frame, v'_i = R_ij v_j. */
export const rot = (th: number): number[][] => [
  [Math.cos(th), Math.sin(th)],
  [-Math.sin(th), Math.cos(th)],
];
export const ROT_MAX = Math.atan2(3, 4); // 36.87 degrees: cos 0.8, sin 0.6
/** Angle of the axes at tick g of the rotate animation: 0 -> ROT_MAX -> 0 over one run. */
export const rotAngle = (g: number) => (ROT_MAX * (1 - Math.cos((2 * Math.PI * Math.min(g, mFrames("rotate"))) / mFrames("rotate")))) / 2;
export const ROT_V = [1, 2];
export const ROT_A = [
  [2, 0],
  [0, 1],
];
/** A'_ij = R_ik R_jl A_kl. */
export const rotMatrix = (R: number[][], A: number[][]) => matMul(matMul(R, A), transpose(R));

/** The rotation the equation card evaluates: cos = 0.8, sin = 0.6 (exact decimals). */
export const ROT_R = [
  [0.8, 0.6],
  [-0.6, 0.8],
];

/** Index up or down: a stick of 2 m crosses contour lines spaced 1/3 m apart. */
export const UNITS = [
  { name: "m", f: 1 },
  { name: "dm", f: 10 },
  { name: "cm", f: 100 },
];
export const STICK_M = 2; // length of the stick in metres
export const LINES_PER_M = 3; // contour lines crossed per metre
/** Upper (contravariant) component: number of units along the stick; grows as the unit shrinks. */
export const upper = (f: number) => STICK_M * f;
/** Lower (covariant) component: lines crossed per unit; shrinks with the unit. */
export const lower = (f: number) => LINES_PER_M / f;
/**
 * The hill of the updown picture: a round hill whose flanks fall 3 m per m (a cone with a rounded top),
 * z = 6.5 - 3r. Contour L (height L m) is the circle r = (6.5 - L) / 3, so the straight walk from the summit
 * outwards meets one every 1/3 m.
 */
export const HILL = { top: 6.5, rMax: 6.5 / LINES_PER_M, levels: [1, 2, 3, 4, 5, 6] };
export function hillZ(x: number, y: number): number {
  const r = Math.hypot(x, y);
  const z = r >= 0.1 ? HILL.top - LINES_PER_M * r : HILL.top - 0.15 - 15 * r * r; // smooth cap, same slope at 0.1
  return Math.max(0, z);
}
export const contourR = (L: number) => (HILL.top - L) / LINES_PER_M;
/** Where the walk (from the summit along +x, the stick length) crosses the contours, nearest first. */
export const crossings = (len = STICK_M) =>
  HILL.levels
    .map(contourR)
    .sort((a, b) => a - b)
    .filter((r) => r <= len);
/** Unit index shown at each updown step: m, dm, cm, then cm held. */
export const UPDOWN_UNIT = [0, 1, 2, 2, 2];

/** A slanted basis e1 = (1, 0), e2 = (cos phi, sin phi) and a fixed vector v. */
export const MET_V = [1.3, 1.0];
export const metPhi = (step: number, t: number, n: number) => {
  const lo = (60 * Math.PI) / 180;
  if (step === 0) return Math.PI / 2;
  if (step === 1) return Math.PI / 2 - ((Math.PI / 2 - lo) * (t + 1)) / n;
  return lo;
};
export function slanted(phi: number, v = MET_V) {
  const e1 = [1, 0];
  const e2 = [Math.cos(phi), Math.sin(phi)];
  // v = a e1 + b e2  ->  b = v_y / sin phi, a = v_x - b cos phi
  const b = v[1] / e2[1];
  const a = v[0] - b * e2[0];
  const g = [
    [dot(e1, e1), dot(e1, e2)],
    [dot(e2, e1), dot(e2, e2)],
  ];
  return { e1, e2, up: [a, b], down: [dot(v, e1), dot(v, e2)], g };
}

/** The photo: a tiny 6 x 8 RGB picture (sky, sun, hill) standing in for 480 x 640. */
export const PHOTO_H = 6;
export const PHOTO_W = 8;
export function photoPixel(r: number, c: number): [number, number, number] {
  const sun = (r - 1) ** 2 + (c - 6) ** 2 <= 1.2;
  const hill = r >= 4 - Math.round(Math.sin((c / 7) * Math.PI) * 1.4);
  if (sun) return [1, 0.85, 0.2];
  if (hill) return [0.2, 0.65, 0.25];
  return [0.45, 0.7, 1];
}
export const PHOTO_SHAPE = [480, 640, 3];
export const BATCH_SHAPE = [32, 480, 640, 3];
export const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** The rank equation card: indices added one by one, then one element picked. */
export const RANK_DIMS = [32, 480, 640, 3];
export const RANK_PICK = [5, 100, 200, 1];

// ---- 2. Einstein summation ----------------------------------------------------------------------------

export const RECEIPT = { q: [2, 1, 3], p: [1.5, 4, 0.5] };
export const EX_A = [
  [1, 2],
  [3, 4],
];
export const DOT = { x: [1, 2, 3], y: [4, 0, 2] };
export const MV = {
  A: [
    [1, 2],
    [0, 3],
    [2, 1],
  ],
  x: [2, 1],
};
export const MM = {
  A: [
    [1, 2, 0],
    [0, 1, 3],
  ],
  B: [
    [1, 0],
    [2, 1],
    [0, 1],
  ],
};
export const TR = [
  [1, 2, 0],
  [3, 4, 1],
  [0, 5, 2],
];
export const OUTER = { u: [1, 2, 3], v: [1, 0, 2] };
export const BMM = {
  A: [
    [
      [1, 2],
      [0, 1],
    ],
    [
      [2, 0],
      [1, 1],
    ],
  ],
  B: [
    [
      [1, 0],
      [1, 1],
    ],
    [
      [0, 1],
      [2, 1],
    ],
  ],
};
/** Running sum of the products a_i b_i after k terms. */
export const runningSum = (a: number[], b: number[], k: number) => a.slice(0, k).reduce((s, x, i) => s + x * b[i], 0);

// ---- 3. Penrose ---------------------------------------------------------------------------------------

export const LOOP = {
  A: [
    [1, 2],
    [3, 4],
  ],
  B: [
    [0, 1],
    [1, 2],
  ],
};
/** Terms A_ij B_ji of tr(AB), in the order (i, j) = (0,0), (0,1), (1,0), (1,1). */
export const loopTerms = () =>
  [0, 1].flatMap((i) => [0, 1].map((j) => ({ i, j, a: LOOP.A[i][j], b: LOOP.B[j][i] })));

export const MPS = {
  A: [
    [1, 2],
    [0, 1],
  ],
  B: [
    [
      [1, 0],
      [0, 1],
    ],
    [
      [2, 1],
      [1, 0],
    ],
  ],
  C: [
    [1, 1],
    [0, 2],
  ],
};
/** The four terms of T_000 = sum over a, b of A_0a B_a0b C_b0. */
export const mpsTerms = () =>
  [0, 1].flatMap((a) => [0, 1].map((b) => ({ a, b, v: MPS.A[0][a] * MPS.B[a][0][b] * MPS.C[b][0] })));

// ---- 4. contraction order -----------------------------------------------------------------------------

export const CHAIN_DIMS = [10, 100, 5, 50];
/** Steps of the two bracketings: the product's shape and its multiplications. */
export function chainSteps(dims = CHAIN_DIMS) {
  const [a, b, c, d] = dims;
  return {
    left: [
      { name: "AB", rows: a, cols: c, cost: a * b * c },
      { name: "(AB)C", rows: a, cols: d, cost: a * c * d },
    ],
    right: [
      { name: "BC", rows: b, cols: d, cost: b * c * d },
      { name: "A(BC)", rows: a, cols: d, cost: a * b * d },
    ],
  };
}
/** The small product counted one multiplication at a time: (2 x 3) times (3 x 2). */
export const COUNT = { a: 2, b: 3, c: 2 };
/** Output cell and inner index of multiplication number k. */
export const countAt = (k: number) => ({ i: Math.floor(k / (COUNT.b * COUNT.c)), k2: Math.floor(k / COUNT.b) % COUNT.c, j: k % COUNT.b });

// ---- 5. roofline --------------------------------------------------------------------------------------

/** An illustrative machine: 200 GFLOP/s peak, 50 GB/s memory, ridge at 4 FLOP/byte. */
export const MACH = { P: 200, beta: 50 };
export const att = (I: number) => Math.min(MACH.P, MACH.beta * I);
/** I at tick t of the sweep: 13 log steps from 0.01 to 100 FLOP/byte. */
export const sweepI = (t: number) => 10 ** (-2 + (4 * Math.min(t, 12)) / 12);
/** Share of time the arithmetic units and the memory are busy at intensity I. */
export const busy = (I: number) => ({ alu: att(I) / MACH.P, mem: att(I) / (MACH.beta * I) });
export const VAN_I = [0.25, 40]; // salad, meal

/** y = relu(a * x + b): numbers moved and FLOPs per element, step by step and fused. */
export const FUSION = {
  unfused: { moved: 6, flops: 3 },
  fused: { moved: 2, flops: 3 },
};
export const intensityOf = (k: { moved: number; flops: number }, bytes = 4) => k.flops / (k.moved * bytes);

// ---- 6. recap -----------------------------------------------------------------------------------------

export const RANK1 = { u: [1, 2, 3, 4], v: [2, 1, 0, 3] };
export const rank1Matrix = () => RANK1.u.map((a) => RANK1.v.map((b) => a * b));
