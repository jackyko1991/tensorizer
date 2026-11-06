// Step logic of the Coding Skills "Techniques" animations, each against an explicit reference.
// Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  AG,
  BATCH,
  LAP_KERNEL,
  LA_B,
  LA_H,
  PLAN,
  RK,
  TDD_LIGHT,
  TECH_CASES,
  agBackward,
  agForward,
  batchTraj,
  batchY0,
  bcA,
  bcB,
  broadcastOp,
  broadcastShape,
  cabs,
  contiguousStrides,
  contractTerms,
  conv2d,
  cxPhased,
  cxPsi,
  cxTime,
  cxX,
  deviceOps,
  devX,
  dft,
  eigh2,
  einsum,
  esA,
  esB,
  expandStrides,
  fftPsi0,
  flatOffset,
  freeEvolve,
  isEigen,
  laDir,
  leviCivita,
  matVec2,
  projStep,
  projStepBatch,
  rk4Path,
  rk4Stages,
  rkExact,
  rkF,
  rows2,
  solve2,
  spectralDerivative,
  stageAt,
  stepCount,
  stU,
  sumDim,
  tensor,
  totalFrames,
  unravel,
  type C,
} from "../src/techAnim.ts";

const close = (a: number, b: number, tol = 1e-9, msg?: string) =>
  assert.ok(Math.abs(a - b) <= tol, msg ?? `${a} != ${b} (tol ${tol})`);

test("every technique has a plan; stageAt walks the stages in order and clamps", () => {
  assert.equal(TECH_CASES.length, 11);
  for (const c of TECH_CASES) {
    const n = totalFrames(c);
    assert.ok(n >= 6 && n <= 60, `${c}: ${n} ticks`);
    assert.ok(PLAN[c].tick >= 100, c);
    const seen = new Set<number>();
    let prevIdx = 0;
    for (let g = 0; g < n; g++) {
      const s = stageAt(c, g);
      assert.ok(s.idx >= prevIdx && s.t < s.n, `${c} @ ${g}`);
      prevIdx = s.idx;
      seen.add(s.step);
    }
    assert.equal(seen.size, stepCount(c), c);
    assert.deepEqual(stageAt(c, 10 ** 6), stageAt(c, n - 1));
    assert.deepEqual(stageAt(c, -3), stageAt(c, 0));
  }
  assert.deepEqual(stageAt("broadcasting", 4), { idx: 1, step: 1, t: 1, n: 4 });
});

test("TDD: the test exists and fails before the code; green only after the code", () => {
  assert.equal(TDD_LIGHT.length, stepCount("tdd"));
  assert.equal(TDD_LIGHT[0], "none");
  assert.ok(TDD_LIGHT.indexOf("red") < TDD_LIGHT.indexOf("green"));
  assert.equal(TDD_LIGHT.at(-1), "green");
});

test("tensors: strides, offsets and the device ops", () => {
  assert.deepEqual(contiguousStrides([2, 3]), [3, 1]);
  assert.deepEqual(contiguousStrides([4, 2, 5]), [10, 5, 1]);
  for (let k = 0; k < 6; k++) assert.equal(flatOffset(unravel(k, [2, 3]), [3, 1]), k);
  assert.deepEqual(unravel(4, [2, 3]), [1, 1]);
  const ops = deviceOps(devX);
  assert.deepEqual(ops[0].t.data, [2, 4, 6, 8, 10, 12]);
  assert.deepEqual(ops[1].t.data, [3, 5, 7, 9, 11, 13]);
  assert.deepEqual(ops[2].t, { shape: [2], data: [15, 33] });
  assert.deepEqual(sumDim(devX, 0), { shape: [3], data: [5, 7, 9] });
});

test("broadcasting: shapes, stride-0 views and values equal the explicit reference", () => {
  assert.deepEqual(broadcastShape([3, 1], [1, 4]), [3, 4]);
  assert.deepEqual(broadcastShape([3, 1], [4]), [3, 4]);
  assert.deepEqual(broadcastShape([5, 1, 3], [1, 5, 3]), [5, 5, 3]); // pair separations (N, N, 3)
  assert.throws(() => broadcastShape([3, 2], [4]));
  assert.deepEqual(expandStrides([3, 1], [3, 4]), [1, 0]);
  assert.deepEqual(expandStrides([1, 4], [3, 4]), [0, 1]);
  assert.deepEqual(expandStrides([4], [2, 3, 4]), [0, 0, 1]);
  const u = broadcastOp(bcA, bcB, (a, b) => a + b);
  assert.deepEqual(u.shape, [3, 4]);
  const ref = [1, 2, 3].map((a) => [10, 20, 30, 40].map((b) => a + b));
  assert.deepEqual(rows2(u), ref);
  // (N, 1, 2) - (1, N, 2): every pair separation
  const x = [[0, 0], [1, 2], [4, 1]];
  const r = broadcastOp(tensor(x.map((p) => [p])), tensor([x]), (a, b) => a - b);
  assert.deepEqual(r.shape, [3, 3, 2]);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let d = 0; d < 2; d++) assert.equal(r.data[i * 6 + j * 2 + d], x[i][d] - x[j][d]);
});

test("einsum: contraction equals explicit sums (matmul, dot, cross product, trace)", () => {
  const C = einsum("ij,jk->ik", esA, esB);
  const a = rows2(esA);
  const b = rows2(esB);
  const ref = a.map((row) => b[0].map((_, k) => row.reduce((s, v, j) => s + v * b[j][k], 0)));
  assert.deepEqual(rows2(C), ref);
  assert.deepEqual(rows2(C), [[4, 1], [1, 6]]);
  for (let i = 0; i < 2; i++)
    for (let k = 0; k < 2; k++)
      assert.equal(contractTerms(esA, esB, i, k).reduce((s, [p, q]) => s + p * q, 0), ref[i][k]);
  const v = tensor([[1, 2, 2], [3, 0, 4]]);
  assert.deepEqual(einsum("bj,bj->b", v, v).data, [9, 25]);
  const vv = tensor([[1, 2, 3]]);
  const B = tensor([0, 0, 1]);
  assert.deepEqual(einsum("ijk,bj,k->bi", leviCivita(), vv, B).data, [2, -1, 0]); // (1,2,3) x (0,0,1)
  assert.equal(einsum("ii->", tensor([[1, 2], [3, 4]])).data[0], 5);
});

test("batching: one batched step equals the per-item step for every projectile", () => {
  let Y = batchY0();
  assert.equal(Y.length, BATCH.B);
  for (let t = 0; t < BATCH.T; t++) {
    const per = Y.map((s) => projStep(s));
    Y = projStepBatch(Y);
    assert.deepEqual(Y, per);
    assert.deepEqual(batchTraj[t + 1], Y);
  }
  // all landed by the end, steeper ones flew higher
  assert.ok(batchTraj[BATCH.T].every((s) => s[1] === 0));
  const peak = (b: number) => Math.max(...batchTraj.map((Y) => Y[b][1]));
  assert.ok(peak(0) < peak(1) && peak(1) < peak(2) && peak(2) < peak(3));
});

test("RK4: four slopes, the weighted step, and fourth-order accuracy", () => {
  const h = 0.1;
  const s = rk4Stages((_t, y) => y, 0, 1, h);
  close(s.y1, 1 + h + h ** 2 / 2 + h ** 3 / 6 + h ** 4 / 24, 1e-15);
  assert.equal(s.pts.length, 4);
  close(s.pts[1][0], h / 2, 1e-15);
  const err = (hh: number) => {
    const n = Math.round(2.4 / hh);
    const p = rk4Path(rkF, RK.y0, hh, n);
    return Math.abs(p[n][1] - rkExact(p[n][0]));
  };
  const ratio = err(0.2) / err(0.1);
  assert.ok(ratio > 10 && ratio < 24, `error ratio ${ratio}`);
  const shown = rk4Path(rkF, RK.y0, RK.h, RK.steps);
  for (const [t, y] of shown) close(y, rkExact(t), 0.05);
});

test("linalg: eigh gives Hv = λv (ascending), solve gives Hx = b", () => {
  const { E, V } = eigh2(LA_H);
  close(E[0], 1);
  close(E[1], 3);
  for (let c = 0; c < 2; c++) {
    const v = [V[0][c], V[1][c]];
    const w = matVec2(LA_H, v);
    close(w[0], E[c] * v[0]);
    close(w[1], E[c] * v[1]);
    close(Math.hypot(v[0], v[1]), 1);
  }
  const x = solve2(LA_H, LA_B);
  const hx = matVec2(LA_H, x);
  close(hx[0], LA_B[0]);
  close(hx[1], LA_B[1]);
  const eig = Array.from({ length: 16 }, (_, t) => t).filter((t) => isEigen(LA_H, laDir(t)));
  assert.deepEqual(eig, [2, 6, 10, 14]); // 45°, 135°, 225°, 315°
  const g = eigh2([[3, 0], [0, 1]]);
  assert.deepEqual(g.E, [1, 3]);
});

test("stencils: conv2d with the Laplacian kernel equals the explicit 5-point sum", () => {
  const lap = conv2d(stU, LAP_KERNEL, 1);
  const n = stU.length;
  const at = (i: number, j: number) => (i < 0 || j < 0 || i >= n || j >= n ? 0 : stU[i][j]);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      assert.equal(lap[i][j], at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1) - 4 * at(i, j));
  const flat = conv2d([[2, 2, 2], [2, 2, 2], [2, 2, 2]], LAP_KERNEL, 0);
  assert.deepEqual(flat, [[0]]); // constant field: no curvature
  assert.equal(conv2d(stU, LAP_KERNEL, 0).length, n - 2);
});

test("FFT: inverse undoes forward, d/dx is multiplication by ik, evolution keeps the norm", () => {
  const back = dft(dft(fftPsi0), true);
  back.forEach((v, j) => {
    close(v[0], fftPsi0[j][0], 1e-10);
    close(v[1], fftPsi0[j][1], 1e-10);
  });
  const n = 32;
  const L = 2 * Math.PI;
  const d = L / n;
  const f = Array.from({ length: n }, (_, j) => Math.sin(3 * j * d));
  spectralDerivative(f, d).forEach((v, j) => close(v, 3 * Math.cos(3 * j * d), 1e-9));
  const norm = (p: C[]) => p.reduce((s, v) => s + v[0] ** 2 + v[1] ** 2, 0);
  close(norm(freeEvolve(fftPsi0, 6)), norm(fftPsi0), 1e-9);
  // the packet spreads: its peak drops
  const peak = (p: C[]) => Math.max(...p.map(cabs));
  assert.ok(peak(freeEvolve(fftPsi0, 6)) < 0.9 * peak(fftPsi0));
});

test("autograd: backward pass equals finite differences of the forward pass", () => {
  const { r, phi } = agForward(AG.x);
  close(r, 5);
  close(phi, -0.2);
  const g = agBackward(AG.x);
  close(g.dr, 0.04);
  const h = 1e-6;
  for (let d = 0; d < 2; d++) {
    const xp = [...AG.x];
    const xm = [...AG.x];
    xp[d] += h;
    xm[d] -= h;
    close(g.dx[d], (agForward(xp).phi - agForward(xm).phi) / (2 * h), 1e-8);
  }
  close(g.dx[0], 0.024);
  close(g.dx[1], 0.032);
});

test("complex: unit phases e^{-iEt}, conserved norm, periodic probability", () => {
  const dx = cxX[1] - cxX[0];
  const p0 = cxPsi(0).prob.reduce((s, v) => s + v * dx, 0);
  close(p0, 1, 1e-3);
  for (const t of [0.3, 1, 2.5, 4]) {
    cxPhased(t).forEach((v, n) => close(Math.atan2(v[1], v[0]), Math.atan2(-Math.sin((n + 0.5) * t), Math.cos((n + 0.5) * t)) , 1e-12));
    const p = cxPsi(t).prob.reduce((s, v) => s + v * dx, 0);
    close(p, p0, 1e-9);
  }
  const a = cxPsi(cxTime(5)).prob;
  const b = cxPsi(cxTime(5) + 2 * Math.PI).prob;
  a.forEach((v, j) => close(v, b[j], 1e-9));
  // the density moves: not stationary
  const c = cxPsi(Math.PI / 2).prob;
  assert.ok(Math.max(...a.map((v, j) => Math.abs(v - c[j]))) > 0.01);
});
