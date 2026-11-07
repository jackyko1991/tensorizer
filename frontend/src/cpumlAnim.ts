/**
 * Pure step logic of the "CPU-only ML to tensors" animations: the stage plan of each card and the toy tensor
 * computations the pictures show (PCA by centring and SVD, the batched linear SVM, the RBF kernel and the
 * projected dual ascent of the kernel SVM, logistic regression's gradient, kNN by cdist and topk, k-means by
 * argmin and scatter-add). Drawn by components/CpumlAnim; checked in tests/cpumlAnim.test.ts.
 */

export type CpumlCase = "pca" | "linsvm" | "ksvm" | "logreg" | "knn" | "kmeans";
export const CPUML_ANIM_CASES: CpumlCase[] = ["pca", "linsvm", "ksvm", "logreg", "knn", "kmeans"];

/** Number of explained lines of each card's tensor code (cpuml.anim.<id>.code0 .. code<n-1>). */
export const CODE_LINES: Record<CpumlCase, number> = { pca: 4, linsvm: 5, ksvm: 6, logreg: 4, knn: 4, kmeans: 5 };

export type M = number[][];

// ---- small dense helpers ---------------------------------------------------------------------------------

export const matmul = (A: M, B: M): M => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
export const transpose = (A: M): M => A[0].map((_, j) => A.map((r) => r[j]));
export const sqDist = (a: number[], b: number[]) => a.reduce((s, v, d) => s + (v - b[d]) ** 2, 0);
/** torch.cdist: (M, D), (N, D) -> (M, N) Euclidean distances. */
export const cdist = (A: M, B: M): M => A.map((a) => B.map((b) => Math.sqrt(sqDist(a, b))));
/** argmin / argmax of a row, the first one on ties (as torch). */
export const argmin = (r: number[]) => r.reduce((b, v, i) => (v < r[b] ? i : b), 0);
export const argmax = (r: number[]) => r.reduce((b, v, i) => (v > r[b] ? i : b), 0);
const r1 = (v: number) => Math.round(v * 10) / 10;

// ---- stage plans -----------------------------------------------------------------------------------------

/** [step caption index, ticks, round] (round: the k-means / dual-ascent iteration the stage belongs to). */
type Stage = [number, number, number?];
interface Plan {
  stages: Stage[];
  tick: number;
}

// ---- PCA: centre by broadcasting, then one SVD ----------------------------------------------------------

/** Eight points of an elongated cloud (rounded to 0.1 so the matrix shows exact values). */
export const PCA_X: M = (() => {
  const t = [-2.3, -1.6, -0.9, -0.3, 0.3, 0.9, 1.5, 2.4];
  const s = [0.3, -0.5, 0.5, -0.3, 0.2, -0.5, 0.45, -0.15];
  const a = Math.PI / 6;
  return t.map((ti, i) => [r1(3 + ti * Math.cos(a) - s[i] * Math.sin(a)), r1(2.5 + ti * Math.sin(a) + s[i] * Math.cos(a))]);
})();

export const colMean = (X: M) => X[0].map((_, d) => X.reduce((s, r) => s + r[d], 0) / X.length);

/** SVD of a centred (N, 2) matrix through the eigen-decomposition of Xc^T Xc: S descending, Vh rows with x >= 0. */
export function svd2(Xc: M): { U: M; S: number[]; Vh: M } {
  const [[a, b], [, c]] = matmul(transpose(Xc), Xc);
  const m = (a + c) / 2;
  const r = Math.hypot((a - c) / 2, b);
  const lam = [m + r, Math.max(m - r, 0)];
  const th = 0.5 * Math.atan2(2 * b, a - c);
  let v1 = [Math.cos(th), Math.sin(th)];
  if (v1[0] < 0) v1 = [-v1[0], -v1[1]];
  let v2 = [-v1[1], v1[0]];
  if (v2[0] < 0 || (v2[0] === 0 && v2[1] < 0)) v2 = [-v2[0], -v2[1]];
  const S = lam.map(Math.sqrt);
  const Vh = [v1, v2];
  const U = Xc.map((row) => Vh.map((v, k) => (row[0] * v[0] + row[1] * v[1]) / S[k]));
  return { U, S, Vh };
}

export const PCA = (() => {
  const mean = colMean(PCA_X);
  const Xc = PCA_X.map((r) => r.map((v, d) => v - mean[d]));
  const { U, S, Vh } = svd2(Xc);
  const variance = S.map((s) => (s * s) / (PCA_X.length - 1));
  const tot = variance[0] + variance[1];
  const evr = variance.map((v) => v / tot);
  const scores = Xc.map((r) => r[0] * Vh[0][0] + r[1] * Vh[0][1]); // Xc @ Vh[:1].T, (N,)
  return { mean, Xc, U, S, Vh, variance, evr, scores };
})();

// ---- shared toy classification data: 4 points, 2 features, 3 classes -----------------------------------------

export const LIN_X: M = [
  [1.0, 0.4],
  [0.2, 1.2],
  [-0.9, -0.5],
  [0.8, -1.1],
];
export const LIN_Y = [0, 1, 2, 0];
export const N_CLS = 3;
export const oneHot = (y: number[], k: number): M => y.map((c) => Array.from({ length: k }, (_, j) => (j === c ? 1 : 0)));

// ---- linear SVM: all K one-vs-rest squared hinges in one (D+1, K) matrix -----------------------------------

export const SVM_C = 1;
export const SVM_W0: M = [
  [0.4, -0.2, -0.3],
  [-0.1, 0.5, -0.2],
  [0.0, -0.1, 0.1],
];

export const appendOnes = (X: M): M => X.map((r) => [...r, 1]);
/** 2 * (y[:, None] == arange(K)) - 1: +1 own class, -1 the rest. */
export const signY = (y: number[], k: number): M => oneHot(y, k).map((r) => r.map((v) => 2 * v - 1));

export function svmParts(W: M, C = SVM_C) {
  const Xb = appendOnes(LIN_X);
  const Y = signY(LIN_Y, N_CLS);
  const S = matmul(Xb, W);
  const H = S.map((r, n) => r.map((v, k) => Math.max(0, 1 - Y[n][k] * v)));
  const loss = 0.5 * W.flat().reduce((s, v) => s + v * v, 0) + C * H.flat().reduce((s, v) => s + v * v, 0);
  // dJ/dW = W - 2C Xb^T (Y o H)
  const YH = H.map((r, n) => r.map((v, k) => Y[n][k] * v));
  const XtYH = matmul(transpose(Xb), YH);
  const G = W.map((r, d) => r.map((v, k) => v - 2 * C * XtYH[d][k]));
  return { Xb, Y, S, H, loss, G };
}

/** The minimiser the optimiser reaches (gradient descent run to convergence: same objective, same minimiser as L-BFGS). */
export const SVM_WSTAR: M = (() => {
  let W = SVM_W0.map((r) => [...r]);
  for (let it = 0; it < 20000; it++) {
    const { G } = svmParts(W);
    W = W.map((r, d) => r.map((v, k) => v - 0.05 * G[d][k]));
  }
  return W;
})();

// ---- logistic regression: matmul, softmax, gradient step ----------------------------------------------

export const LR_C = 1;
export const LR_ETA = 0.5;
export const LR_W0: M = [
  [0.5, -0.3, 0.2],
  [-0.4, 0.6, 0.1],
];
export const LR_B = [0.1, 0, -0.1];

export const softmaxRows = (Z: M): M =>
  Z.map((r) => {
    const mx = Math.max(...r);
    const e = r.map((v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });

export function lrParts(W: M, b = LR_B, C = LR_C) {
  const XW = matmul(LIN_X, W);
  const Z = XW.map((r) => r.map((v, k) => v + b[k]));
  const P = softmaxRows(Z);
  const Y = oneHot(LIN_Y, N_CLS);
  const PY = P.map((r, n) => r.map((v, k) => v - Y[n][k]));
  const XtPY = matmul(transpose(LIN_X), PY);
  // gradient of the scaled loss J / C: X^T (P - Y) + W / C
  const G = W.map((r, d) => r.map((v, k) => XtPY[d][k] + v / C));
  const ce = P.reduce((s, r, n) => s - Math.log(r[LIN_Y[n]]), 0);
  const loss = (0.5 * W.flat().reduce((s, v) => s + v * v, 0)) / C + ce; // J / C
  return { XW, Z, P, Y, PY, G, loss };
}

export const LR_W1: M = (() => {
  const { G } = lrParts(LR_W0);
  return LR_W0.map((r, d) => r.map((v, k) => v - LR_ETA * G[d][k]));
})();

// ---- kernel SVM: broadcast kernel matrix, projected dual ascent -----------------------------------------

export const KS_X: M = [
  [0.8, 1.5],
  [0.6, 1.8],
  [0.8, 0.9],
  [2.5, 1.3],
  [1.7, 1.9],
];
export const KS_Y = [1, 1, 1, -1, -1];
export const KS_C = 1;
/** scikit-learn's gamma="scale": 1 / (D var(X)), the variance over every entry. */
export const KS_GAMMA = (() => {
  const v = KS_X.flat();
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return 1 / (KS_X[0].length * (v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length));
})();

/** Closest point to v with 0 <= a <= C and sum(y a) = 0: a = clip(v - mu y, 0, C), mu by bisection. */
export function project(v: number[], y: number[], C: number): number[] {
  const h = (mu: number) => v.reduce((s, vi, i) => s + y[i] * Math.min(Math.max(vi - mu * y[i], 0), C), 0);
  let lo = -(Math.max(...v.map(Math.abs)) + C);
  let hi = -lo;
  for (let it = 0; it < 100; it++) {
    const mid = (lo + hi) / 2;
    if (h(mid) > 0) lo = mid;
    else hi = mid;
  }
  const mu = (lo + hi) / 2;
  return v.map((vi, i) => Math.min(Math.max(vi - mu * y[i], 0), C));
}

/** Dual-ascent iterations the picture shows (the last one is converged). */
export const KS_SNAPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 400];

export const KS = (() => {
  const sq = KS_X.map((r) => r[0] * r[0] + r[1] * r[1]);
  const gram = matmul(KS_X, transpose(KS_X));
  const D2 = sq.map((a, i) => sq.map((b, j) => Math.max(a + b - 2 * gram[i][j], 0)));
  const K = D2.map((r) => r.map((v) => Math.exp(-KS_GAMMA * v)));
  const Q = K.map((r, i) => r.map((v, j) => v * KS_Y[i] * KS_Y[j]));
  const Qv = (a: number[]) => Q.map((r) => r.reduce((s, q, j) => s + q * a[j], 0));
  let v = KS_Y.map(() => 1);
  for (let it = 0; it < 30; it++) {
    v = Qv(v);
    const n = Math.hypot(...v);
    v = v.map((x) => x / n);
  }
  const L = Qv(v).reduce((s, x, i) => s + x * v[i], 0);
  const dual = (a: number[]) => a.reduce((s, x) => s + x, 0) - 0.5 * Qv(a).reduce((s, x, i) => s + x * a[i], 0);
  // FISTA, as the card's code; keep snapshots at the iterations of KS_SNAPS
  let a = KS_Y.map(() => 0);
  let z = [...a];
  let t = 1;
  const snaps: { it: number; a: number[]; dual: number }[] = [{ it: 0, a: [...a], dual: 0 }];
  for (let it = 1; it <= KS_SNAPS[KS_SNAPS.length - 1]; it++) {
    const Qz = Qv(z);
    const an = project(z.map((zi, i) => zi + (1 - Qz[i]) / L), KS_Y, KS_C);
    const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
    z = an.map((x, i) => x + ((t - 1) / tn) * (x - a[i]));
    a = an;
    t = tn;
    if (KS_SNAPS.includes(it)) snaps.push({ it, a: [...a], dual: dual(a) });
  }
  // decision values f = K @ (a * y); b from the margin support vectors (0 < a < C), as the card's code
  const f = K.map((r) => r.reduce((s, k, j) => s + k * a[j] * KS_Y[j], 0));
  const free = a.map((x) => x > 1e-3 * KS_C && x < (1 - 1e-3) * KS_C);
  const nFree = free.filter(Boolean).length;
  const b = KS_Y.reduce((s, yi, i) => s + (free[i] ? yi - f[i] : 0), 0) / nFree;
  return { sq, gram, D2, K, Q, L, snaps, dual, alpha: a, f, b, free };
})();

// ---- kNN: cdist, topk, one-hot votes, argmax -----------------------------------------------------------

export const KNN_TR: M = [
  [0.8, 1.0],
  [1.6, 0.6],
  [1.2, 2.0],
  [3.4, 1.0],
  [4.4, 1.6],
  [4.0, 0.4],
  [2.2, 3.8],
  [3.4, 4.4],
];
export const KNN_YTR = [0, 0, 0, 1, 1, 1, 2, 2];
export const KNN_TE: M = [
  [1.9, 1.5],
  [3.6, 2.2],
  [2.6, 3.0],
];
export const KNN_K = 3;

/** topk(k, largest=False) of a row: indices of the k smallest, nearest first (ties: lower index). */
export const topkSmallest = (r: number[], k: number) =>
  r.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k).map(([, i]) => i);

export const KNN = (() => {
  const D = cdist(KNN_TE, KNN_TR);
  const idx = D.map((r) => topkSmallest(r, KNN_K));
  const lab = idx.map((r) => r.map((i) => KNN_YTR[i]));
  const votes = lab.map((r) => Array.from({ length: N_CLS }, (_, c) => r.filter((l) => l === c).length));
  const pred = votes.map(argmax);
  return { D, idx, lab, votes, pred };
})();

// ---- k-means: cdist + argmin, index_add_ + bincount ----------------------------------------------------

export const KM_X: M = [
  [1.5, 0.7],
  [1.2, 1.8],
  [4.5, 1.0],
  [0.8, 1.2],
  [5.3, 1.6],
  [4.9, 0.6],
  [2.6, 4.5],
  [3.4, 5.2],
  [3.1, 4.2],
];
export const KM_K = 3;

export interface KmRound {
  centers: M; // (K, 2) used for the distances
  D: M; // (N, K)
  labels: number[]; // (N,)
  sums: M; // (K, 2) per-cluster sums of the points
  counts: number[]; // (K,)
  next: M; // (K, 2) the cluster means
}

/** Lloyd's algorithm as the card's code, from the first K points; one entry per assignment, the last converged. */
export function kmeansRun(X: M, K: number, maxIter = 20): KmRound[] {
  let centers = X.slice(0, K).map((r) => [...r]);
  const out: KmRound[] = [];
  let prev: number[] | null = null;
  for (let it = 0; it <= maxIter; it++) {
    const D = cdist(X, centers);
    const labels = D.map(argmin);
    const sums = Array.from({ length: K }, () => X[0].map(() => 0));
    const counts = Array(K).fill(0);
    labels.forEach((l, n) => {
      X[n].forEach((v, d) => (sums[l][d] += v));
      counts[l]++;
    });
    const next = sums.map((s, j) => s.map((v) => v / Math.max(counts[j], 1)));
    out.push({ centers, D, labels, sums, counts, next });
    if (prev && labels.every((l, n) => l === prev![n])) break;
    prev = labels;
    centers = next;
  }
  return out;
}

export const KM = kmeansRun(KM_X, KM_K);

// ---- the plans -----------------------------------------------------------------------------------------

export const PLAN: Record<CpumlCase, Plan> = {
  pca: { stages: [[0, 3], [1, 5], [2, 4], [3, 5]], tick: 650 },
  linsvm: { stages: [[0, 3], [1, 3], [2, 4], [3, 4], [4, 3], [5, 4]], tick: 750 },
  ksvm: {
    stages: [[0, 3], [1, 3], [2, 3], [3, 3], ...KS.snaps.slice(1).map((_, i) => [4, 1, i + 1] as Stage), [5, 4]],
    tick: 700,
  },
  logreg: { stages: [[0, 4], [1, 3], [2, 3], [3, 3], [4, 3], [5, 4]], tick: 750 },
  knn: { stages: [[0, 3], [1, 3], [2, 3], [3, 4]], tick: 800 },
  kmeans: {
    stages: KM.flatMap((_, r): Stage[] =>
      r < KM.length - 1
        ? [[0, 2, r], [1, 2, r], [2, 2, r], [3, 3, r]]
        : [[0, 2, r], [1, 2, r], [4, 4, r]],
    ),
    tick: 600,
  },
};

/** Number of step captions (cpuml.anim.<id>.s0 .. s<n-1>). */
export const stepCount = (c: CpumlCase) => Math.max(...PLAN[c].stages.map(([k]) => k)) + 1;

/** Ticks of one run (the picture then holds before the cycle restarts). */
export const totalFrames = (c: CpumlCase) => PLAN[c].stages.reduce((s, [, n]) => s + n, 0);

/** Stage of tick g, clamped to the last tick: step caption, tick within it, its length, the round. */
export function stageAt(c: CpumlCase, g: number): { idx: number; step: number; t: number; n: number; round: number } {
  const st = PLAN[c].stages;
  let r = Math.min(Math.max(Math.floor(g), 0), totalFrames(c) - 1);
  for (let idx = 0; idx < st.length; idx++) {
    const [step, n, round = 0] = st[idx];
    if (r < n) return { idx, step, t: r, n, round };
    r -= n;
  }
  const last = st[st.length - 1];
  return { idx: st.length - 1, step: last[0], t: last[1] - 1, n: last[1], round: last[2] ?? 0 };
}
