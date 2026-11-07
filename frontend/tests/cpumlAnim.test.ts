// The "CPU-only ML to tensors" animations, plot explanations and code notes: the toy tensor computations
// against explicit references, every card wired up, every key in all languages. Run: node --test frontend/tests/*.test.ts
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import {
  CODE_LINES,
  CPUML_ANIM_CASES,
  KM,
  KM_K,
  KM_X,
  KNN,
  KNN_K,
  KNN_TE,
  KNN_TR,
  KNN_YTR,
  KS,
  KS_C,
  KS_GAMMA,
  KS_X,
  KS_Y,
  LIN_X,
  LR_ETA,
  LR_W0,
  LR_W1,
  PCA,
  PCA_X,
  PLAN,
  SVM_W0,
  SVM_WSTAR,
  cdist,
  lrParts,
  matmul,
  project,
  stageAt,
  stepCount,
  svmParts,
  topkSmallest,
  totalFrames,
  transpose,
} from "../src/cpumlAnim.ts";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const close = (a: number, b: number, tol = 1e-9, msg?: string) => assert.ok(Math.abs(a - b) <= tol, msg ?? `${a} != ${b} (tol ${tol})`);

/** Every key the animations, code notes and plot explanations need. */
function animKeys(): string[] {
  const keys = ["cpuml.anim.title", "cpuml.anim.codeTitle", "cpuml.plot.title", "cpuml.plot.axis", "cpuml.plot.common"];
  for (const c of CPUML_ANIM_CASES) {
    keys.push(`cpuml.anim.${c}.caption`, `cpuml.plot.${c}`);
    for (let k = 0; k < stepCount(c); k++) keys.push(`cpuml.anim.${c}.s${k}`);
    for (let k = 0; k < CODE_LINES[c]; k++) keys.push(`cpuml.anim.${c}.code${k}`);
  }
  return keys;
}

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("every algorithm card has an animation, code notes and a plot explanation", () => {
  const tab = read("pages/CpumlTab.tsx");
  const cases = tab.match(/const CASES = \[([^\]]+)\]/)![1].match(/"(\w+)"/g)!.map((s) => s.slice(1, -1));
  assert.deepEqual(cases, CPUML_ANIM_CASES);
  assert.match(tab, /<CpumlAnim id=\{c\} \/>/);
  assert.match(tab, /<CpumlCodeNotes id=\{c\} lines=\{CODE_LINES\[c\]\} \/>/);
  assert.match(tab, /cpuml\.plot\.\$\{c\}/);
  assert.match(tab, /cpuml\.plot\.common/);
  assert.match(tab, /cpuml\.plot\.axis/);
  const anim = read("components/CpumlAnim.tsx");
  for (const c of CPUML_ANIM_CASES) assert.match(anim, new RegExp(`\\b${c}: \\{ draw:`), `picture for ${c}`);
});

test("every new key exists, non-empty, in English and in all translations; no em dashes", () => {
  const keys = animKeys();
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = keys.filter((k) => typeof text[k] !== "string" || !text[k].trim());
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
    const dash = keys.filter((k) => /[—―]| – /.test(text[k]));
    assert.deepEqual(dash, [], `${f}: em dash in ${dash.join(", ")}`);
    // the {k} / {knn} / {clusters} placeholders survive translation
    for (const k of keys) for (const ph of text[k].match(/\{\w+\}/g) ?? []) assert.ok(JSON.parse(readFileSync(new URL("pages.en.json", dir), "utf-8"))[k].includes(ph), `${f} ${k}: ${ph}`);
  }
  for (const src of ["components/CpumlAnim.tsx", "cpumlAnim.ts"]) assert.doesNotMatch(read(src), /[—―]/, src);
});

test("plans: stageAt walks every stage in order, each step caption is reached", () => {
  for (const c of CPUML_ANIM_CASES) {
    const n = totalFrames(c);
    assert.ok(n >= 8 && n <= 60, `${c}: ${n} ticks`);
    assert.ok(PLAN[c].tick >= 300, c);
    const seen = new Set<number>();
    let prev = 0;
    for (let g = 0; g < n; g++) {
      const s = stageAt(c, g);
      assert.ok(s.idx >= prev && s.t < s.n, `${c} @ ${g}`);
      prev = s.idx;
      seen.add(s.step);
    }
    assert.equal(seen.size, stepCount(c), c);
    assert.deepEqual(stageAt(c, n + 5), stageAt(c, n - 1));
  }
});

test("PCA: centring, SVD and explained variance", () => {
  const n = PCA_X.length;
  for (const v of PCA_X.flat()) close(v * 10, Math.round(v * 10), 1e-9); // one decimal, as displayed
  for (let d = 0; d < 2; d++) close(PCA.Xc.reduce((s, r) => s + r[d], 0), 0, 1e-12);
  // covariance eigen-pairs: C v = var v, with C = Xc^T Xc / (N - 1)
  const C = matmul(transpose(PCA.Xc), PCA.Xc).map((r) => r.map((v) => v / (n - 1)));
  for (let k = 0; k < 2; k++) {
    const v = PCA.Vh[k];
    close(Math.hypot(...v), 1, 1e-12);
    for (let d = 0; d < 2; d++) close(C[d][0] * v[0] + C[d][1] * v[1], PCA.variance[k] * v[d], 1e-9);
  }
  close(PCA.Vh[0][0] * PCA.Vh[1][0] + PCA.Vh[0][1] * PCA.Vh[1][1], 0, 1e-12);
  assert.ok(PCA.S[0] > PCA.S[1]);
  // Xc = U diag(S) Vh
  PCA.Xc.forEach((r, i) => r.forEach((v, d) => close(PCA.U[i][0] * PCA.S[0] * PCA.Vh[0][d] + PCA.U[i][1] * PCA.S[1] * PCA.Vh[1][d], v, 1e-9)));
  close(PCA.evr[0] + PCA.evr[1], 1, 1e-12);
  assert.ok(PCA.evr[0] > 0.9 && PCA.evr[0] < 0.95, "the caption says 94 %");
});

test("linear SVM: hinge, gradient (finite differences) and the minimiser", () => {
  const p = svmParts(SVM_W0);
  assert.deepEqual(p.Xb.map((r) => r[2]), [1, 1, 1, 1]);
  assert.deepEqual(p.Y[0], [1, -1, -1]);
  p.H.flat().forEach((h) => assert.ok(h >= 0));
  const eps = 1e-6;
  for (let d = 0; d < 3; d++)
    for (let k = 0; k < 3; k++) {
      const Wp = SVM_W0.map((r) => [...r]);
      const Wm = SVM_W0.map((r) => [...r]);
      Wp[d][k] += eps;
      Wm[d][k] -= eps;
      close(p.G[d][k], (svmParts(Wp).loss - svmParts(Wm).loss) / (2 * eps), 1e-5, `grad ${d}${k}`);
    }
  const q = svmParts(SVM_WSTAR);
  q.G.flat().forEach((g) => close(g, 0, 1e-8));
  assert.ok(q.loss < p.loss);
  assert.ok(q.H.flat().some((h) => h === 0), "some points end beyond the margin");
});

test("logistic regression: softmax rows, gradient (finite differences), the step lowers the loss", () => {
  const p = lrParts(LR_W0);
  p.P.forEach((r) => close(r.reduce((a, b) => a + b, 0), 1, 1e-12));
  const eps = 1e-6;
  for (let d = 0; d < 2; d++)
    for (let k = 0; k < 3; k++) {
      const Wp = LR_W0.map((r) => [...r]);
      const Wm = LR_W0.map((r) => [...r]);
      Wp[d][k] += eps;
      Wm[d][k] -= eps;
      close(p.G[d][k], (lrParts(Wp).loss - lrParts(Wm).loss) / (2 * eps), 1e-6, `grad ${d}${k}`);
    }
  LR_W1.forEach((r, d) => r.forEach((v, k) => close(v, LR_W0[d][k] - LR_ETA * p.G[d][k], 1e-12)));
  assert.ok(lrParts(LR_W1).loss < p.loss);
  assert.equal(LIN_X.length, p.Z.length);
});

test("kernel SVM: kernel by broadcasting, projection, converged dual with one zero multiplier", () => {
  KS.D2.forEach((r, i) => r.forEach((v, j) => close(v, (KS_X[i][0] - KS_X[j][0]) ** 2 + (KS_X[i][1] - KS_X[j][1]) ** 2, 1e-12)));
  KS.K.forEach((r, i) => close(r[i], 1, 1e-12));
  close(KS.K[0][1], Math.exp(-KS_GAMMA * KS.D2[0][1]), 1e-12);
  // projection: feasible, and no feasible point nearby is closer
  const v = [0.9, -0.2, 1.7, 0.4, 0.3];
  const a = project(v, KS_Y, KS_C);
  close(a.reduce((s, x, i) => s + x * KS_Y[i], 0), 0, 1e-9);
  a.forEach((x) => assert.ok(x >= 0 && x <= KS_C));
  const dist = (b: number[]) => b.reduce((s, x, i) => s + (x - v[i]) ** 2, 0);
  for (const [i, j] of [[0, 3], [1, 4], [2, 3]]) {
    for (const e of [1e-3, -1e-3]) {
      const b = [...a];
      b[i] += e; // y_i = +1, y_j = -1: moving both keeps sum(y a) = 0
      b[j] += e;
      if (b.every((x) => x >= 0 && x <= KS_C)) assert.ok(dist(b) >= dist(a) - 1e-12);
    }
  }
  // the dual rises over the shown iterations and the final point is optimal (KKT within tolerance)
  const duals = KS.snaps.map((s) => s.dual);
  assert.ok(duals[duals.length - 1] >= Math.max(...duals) - 1e-9);
  close(KS.alpha[0], 0, 1e-6);
  close(KS.alpha[4], KS_C, 1e-6);
  assert.equal(KS.free.filter(Boolean).length, 3);
  const margins = KS.f.map((f, i) => KS_Y[i] * (f + KS.b));
  KS.free.forEach((fr, i) => fr && close(margins[i], 1, 1e-3));
  assert.ok(margins[0] >= 1 - 1e-3, "alpha = 0: outside the margin");
  assert.ok(margins[4] <= 1 + 1e-3, "alpha = C: inside the margin");
});

test("kNN: cdist, topk and the vote against brute force", () => {
  KNN.D.forEach((r, m) => r.forEach((d, n) => close(d, Math.hypot(KNN_TE[m][0] - KNN_TR[n][0], KNN_TE[m][1] - KNN_TR[n][1]), 1e-12)));
  assert.deepEqual(topkSmallest([3, 1, 2, 1], 2), [1, 3]);
  KNN.idx.forEach((row, m) => {
    const sorted = [...KNN.D[m]].sort((a, b) => a - b);
    assert.deepEqual(row.map((n) => KNN.D[m][n]), sorted.slice(0, KNN_K));
    const votes = [0, 0, 0];
    row.forEach((n) => votes[KNN_YTR[n]]++);
    assert.deepEqual(KNN.votes[m], votes);
    assert.equal(KNN.pred[m], votes.indexOf(Math.max(...votes)));
  });
  assert.deepEqual(KNN.pred, [0, 1, 2]);
  assert.deepEqual(KNN.votes[2], [1, 0, 2], "one mixed vote");
});

test("k-means: Lloyd from the first K points, means, convergence", () => {
  assert.deepEqual(KM[0].centers, KM_X.slice(0, KM_K));
  assert.ok(KM.length >= 3, "at least two moves before convergence");
  KM.forEach((r) => {
    r.labels.forEach((l, n) => assert.equal(l, r.D[n].indexOf(Math.min(...r.D[n]))));
    assert.equal(r.counts.reduce((a, b) => a + b, 0), KM_X.length);
    r.next.forEach((c, j) => {
      const mine = KM_X.filter((_, n) => r.labels[n] === j);
      c.forEach((v, d) => close(v, mine.reduce((s, p) => s + p[d], 0) / mine.length, 1e-12));
    });
  });
  for (let i = 1; i < KM.length; i++) assert.deepEqual(KM[i].centers, KM[i - 1].next);
  assert.deepEqual(KM[KM.length - 1].labels, KM[KM.length - 2].labels, "converged: labels unchanged");
  assert.deepEqual(cdist([[0, 0]], [[3, 4]]), [[5]]);
});
