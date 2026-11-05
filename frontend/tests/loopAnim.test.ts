// Step logic of the "For loop vs tensor" animations. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ANIM_CASES,
  SIZE,
  darts,
  diffU0,
  diffusionLoop,
  diffusionTensor,
  dotData,
  dotLoopSum,
  loopCount,
  loopFrames,
  matmul,
  mmData,
  mmLoop,
  pairDistances,
  pairIJ,
  pendLoop,
  pendTraj,
  piEstimate,
  rk4Batch,
  rk4Scalar,
  tensorCount,
  tensorFrames,
  treeSum,
} from "../src/loopAnim.ts";

test("the loop needs one tick per element, the tensor side one call", () => {
  const want = { dot: 12, pairwise: 64, matmul: 64, pendulum: 60, diffusion: 36, mc_pi: 100 };
  for (const c of ANIM_CASES) {
    assert.equal(loopFrames(c), want[c], c);
    assert.equal(loopCount(c, 10 ** 6), want[c], c);
    const end = tensorCount(c, 10 ** 6);
    assert.equal(end, c === "pendulum" ? SIZE.pendT : 1, c); // pendulums: one vectorised step per t
    assert.ok(tensorFrames(c) <= 5 || c === "pendulum", c);
    assert.equal(tensorCount(c, 0), 0);
  }
  // per time step the loop pays B passes, the tensor one
  assert.equal(loopFrames("pendulum") / tensorFrames("pendulum"), SIZE.pendB);
});

test("dot product: loop running sum and the reduction tree agree", () => {
  const { x, y } = dotData;
  const prods = x.map((v, i) => v * y[i]);
  assert.equal(dotLoopSum(x, y, SIZE.dot), treeSum(prods));
  assert.equal(dotLoopSum(x, y, 3), prods[0] + prods[1] + prods[2]);
  assert.equal(treeSum([1, 2, 3, 4, 5]), 15);
});

test("pairwise: the double loop visits every (i, j) once, row-major", () => {
  const seen = new Set<string>();
  for (let k = 0; k < SIZE.pairwise ** 2; k++) seen.add(pairIJ(k).join(","));
  assert.equal(seen.size, 64);
  assert.deepEqual(pairIJ(9), [1, 1]);
  const d = pairDistances([0, 3, 7]);
  assert.deepEqual(d, [[0, 3, 7], [3, 0, 4], [7, 4, 0]]);
});

test("matmul: the triple loop after n^3 passes equals A @ B, and is partial before", () => {
  const { A, B } = mmData;
  assert.deepEqual(mmLoop(A, B, SIZE.matmul ** 3), matmul(A, B));
  const half = mmLoop(A, B, 2); // two multiply-adds into C[0][0]
  assert.equal(half[0][0], A[0][0] * B[0][0] + A[0][1] * B[1][0]);
  assert.equal(half[0][1], null);
});

test("pendulums: the batched RK4 step equals B scalar steps; the loop advances one pendulum per tick", () => {
  const y = pendTraj[0];
  const batch = rk4Batch(y);
  y.forEach(([th, om], b) => {
    const [a, w] = rk4Scalar(th, om);
    assert.ok(Math.abs(a - batch[b][0]) < 1e-12 && Math.abs(w - batch[b][1]) < 1e-12);
  });
  const s = pendLoop(SIZE.pendB + 2);
  assert.equal(s.t, 1);
  assert.equal(s.current, 2);
  assert.deepEqual(s.steps.slice(0, 3), [2, 2, 1]);
  assert.equal(pendLoop(loopFrames("pendulum")).t, SIZE.pendT);
  // energy roughly conserved over the short run (RK4, dt = 0.1)
  const E = ([th, om]: [number, number]) => 0.5 * om * om - 9.81 * Math.cos(th);
  for (let b = 0; b < SIZE.pendB; b++)
    assert.ok(Math.abs(E(pendTraj[SIZE.pendT][b]) - E(pendTraj[0][b])) < 1e-3);
});

test("diffusion: shifted slices give the same update as the stencil loop, on a tiny grid too", () => {
  assert.deepEqual(diffusionTensor(diffU0), diffusionLoop(diffU0, 36));
  const u = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  // reference by hand, a = 0.2: centre 1 + 0.2(-4) = 0.2; its interior neighbours get 0.2
  const ref = [
    [0, 0, 0, 0],
    [0, 0.2, 0.2, 0],
    [0, 0.2, 0, 0],
    [0, 0, 0, 0],
  ];
  const t = diffusionTensor(u);
  t.forEach((row, i) => row.forEach((v, j) => assert.ok(Math.abs(v - ref[i][j]) < 1e-12, `${i},${j}`)));
  assert.deepEqual(diffusionLoop(u, 4), t);
  assert.deepEqual(diffusionLoop(u, 0), u); // nothing updated yet
});

test("Monte Carlo pi converges on fixed seeds", () => {
  for (const seed of [1, 2, 3]) {
    const d = darts(200_000, seed);
    assert.ok(Math.abs(piEstimate(d, d.length) - Math.PI) < 0.02, `seed ${seed}`);
  }
  assert.ok(Number.isNaN(piEstimate(darts(3, 1), 0)));
});
