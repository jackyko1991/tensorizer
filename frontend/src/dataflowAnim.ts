/**
 * Pure step logic of the Concept "Tensor dataflow" animations (components/DataflowAnim): which stage a tick
 * belongs to and the small computations each picture shows (pair separations, a contraction lowered to a
 * matrix product, float32 / float64 digits, an autograd graph, the device choice). Checked against explicit
 * references in tests/dataflowAnim.test.ts.
 */

/** Cards of the tab with an animation here (s1 and mem use TensorLab / PairMemory, s6 is a table). */
export type FlowCase = "s2" | "s3" | "s4" | "s5" | "s7";
export const FLOW_CASES: FlowCase[] = ["s2", "s3", "s4", "s5", "s7"];

/** Stage plan of one animation: [step caption index, ticks], and the tick length in ms at 1x. */
interface Plan {
  stages: [number, number][];
  tick: number;
}

export const PAIR_N = 4; // particles in the vectorise picture

export const PLAN: Record<FlowCase, Plan> = {
  s2: { stages: [[0, PAIR_N * PAIR_N], [1, 5], [2, 5]], tick: 420 },
  s3: { stages: [[0, 3], [1, 6], [2, 4], [3, 6]], tick: 600 },
  s4: { stages: [[0, 4], [1, 5], [2, 4], [3, 5]], tick: 750 },
  s5: { stages: [[0, 4], [1, 4], [2, 8], [3, 4]], tick: 700 },
  s7: { stages: [[0, 4], [1, 4], [2, 4], [3, 4]], tick: 800 },
};

/** Number of distinct step captions (flow.anim.<id>.s0 .. s<n-1>). */
export const stepCount = (c: FlowCase) => Math.max(...PLAN[c].stages.map(([k]) => k)) + 1;

/** Ticks of one run (the picture then holds before the cycle restarts). */
export const totalFrames = (c: FlowCase) => PLAN[c].stages.reduce((s, [, n]) => s + n, 0);

/** Stage of tick g (clamped to the last tick): index in the plan, step caption, tick within it, its length. */
export function stageAt(c: FlowCase, g: number): { idx: number; step: number; t: number; n: number } {
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

// ---- s2: pair separations ------------------------------------------------------------------------------

/** Loop order of the double loop `for a: for b:`: pass k visits pair (a, b). */
export const pairAB = (k: number): [number, number] => [Math.floor(k / PAIR_N), k % PAIR_N];

/** Strides of `x[None, :, :].expand(N, N, 3)` and `x[:, None, :].expand(N, N, 3)` for a contiguous (N, 3) x. */
export const VIEW_STRIDES = { rows: [0, 3, 1], cols: [3, 0, 1] } as const;

/** Flat offset in x's buffer read by element [a, b, i] of a view with the given strides. */
export const viewOffset = (strides: readonly number[], a: number, b: number, i: number) =>
  a * strides[0] + b * strides[1] + i * strides[2];

// ---- s4: einsum "abi,b->ai" as a matrix product --------------------------------------------------------

export const CON = { A: 3, B: 4, I: 2 };
/** K[a][b][i], small integers so the sums fit in a cell. */
export const conK: number[][][] = Array.from({ length: CON.A }, (_, a) =>
  Array.from({ length: CON.B }, (_, b) => Array.from({ length: CON.I }, (_, i) => ((a + 2 * b + 3 * i) % 4) - 1)),
);
export const conM = [1, 2, 0, 1];

/** Reference: F_ai = sum_b K_abi m_b, written as loops. */
export function einsumLoop(K: number[][][], m: number[]): number[][] {
  return K.map((Ka) => Array.from({ length: CON.I }, (_, i) => Ka.reduce((s, Kab, b) => s + Kab[i] * m[b], 0)));
}

/** K laid out as a (A·I, B) matrix: row r = a·I + i (permute to (a, i, b), then view). */
export const asMatrix = (K: number[][][]): number[][] =>
  Array.from({ length: CON.A * CON.I }, (_, r) => K[Math.floor(r / CON.I)].map((Kab) => Kab[r % CON.I]));

/** Matrix times vector, then reshaped back to (A, I). */
export function einsumGemm(K: number[][][], m: number[]): number[][] {
  const y = asMatrix(K).map((row) => row.reduce((s, v, b) => s + v * m[b], 0));
  return Array.from({ length: CON.A }, (_, a) => y.slice(a * CON.I, a * CON.I + CON.I));
}

/** The decimal expansion of 0.1 as stored in float32 and float64, split into correct digits and the error. */
export function digitsOf(dtype: "float32" | "float64"): { ok: string; bad: string } {
  const s = dtype === "float32" ? Math.fround(0.1).toPrecision(18) : (0.1).toPrecision(22);
  let k = 2; // after "0."
  while (k < s.length && s[k] === (k === 2 ? "1" : "0")) k++;
  return { ok: s.slice(0, k), bad: s.slice(k) };
}

export const FLOAT_BITS = { float32: [1, 8, 23], float64: [1, 11, 52] } as const; // sign, exponent, mantissa

// ---- s5: autograd graph and fusion -----------------------------------------------------------------------

export const AG5 = { x: [1, 0.5, 0], GM: 1 };
/** Forward values r = |x|, phi = -GM / r and the backward ones dphi/dr = GM / r^2, grad = dphi/dr * x / r. */
export function agGraph(x = AG5.x, GM = AG5.GM) {
  const r = Math.hypot(...x);
  const phi = -GM / r;
  const dphidr = GM / (r * r);
  const drdx = x.map((v) => v / r);
  const grad = drdx.map((v) => dphidr * v);
  return { r, phi, dphidr, drdx, grad };
}

export const FUSE_OPS = 4; // element-wise operations in the fusion picture
/** Memory trips (one read and one write per kernel): eager runs FUSE_OPS kernels, compiled runs one. */
export const memTrips = (fused: boolean) => (fused ? 2 : 2 * FUSE_OPS);

// ---- s7: device choice ------------------------------------------------------------------------------------

export type Dev = "mps" | "cuda" | "cpu";
/** Mirror of problems/_common/runtime.py device(). */
export function chooseDevice(choice: string, has: { mps: boolean; cuda: boolean }): Dev {
  if (choice === "mps" && has.mps) return "mps";
  if ((choice === "auto" || choice === "cuda") && has.cuda) return "cuda";
  return "cpu";
}

/** The four scenarios the flow chart walks through: machine label index, the panel's choice, what is present. */
export const SCENARIOS: { machine: 0 | 1 | 2; choice: string; has: { mps: boolean; cuda: boolean } }[] = [
  { machine: 0, choice: "auto", has: { mps: false, cuda: true } },
  { machine: 1, choice: "mps", has: { mps: true, cuda: false } },
  { machine: 2, choice: "auto", has: { mps: false, cuda: false } },
  { machine: 2, choice: "cuda", has: { mps: false, cuda: false } },
];

/** Outcome of the two checks in the flow chart: [MPS check passes, CUDA check passes]. */
export const checks = (choice: string, has: { mps: boolean; cuda: boolean }): [boolean, boolean] => [
  choice === "mps" && has.mps,
  (choice === "auto" || choice === "cuda") && has.cuda,
];
