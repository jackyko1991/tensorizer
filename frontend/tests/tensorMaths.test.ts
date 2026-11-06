// Pure helpers of the Concept page's Tensor maths tab. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  attainable,
  bestChain,
  chainCost,
  elementwiseBytes,
  intensity,
  kernelCost,
  orderText,
  ridge,
} from "../src/tensorMaths.ts";

const DIMS = [10, 100, 5, 50]; // A 10x100, B 100x5, C 5x50

test("contraction order changes the cost: (AB)C vs A(BC)", () => {
  assert.equal(chainCost(DIMS, [[0, 1], 2]).cost, 10 * 100 * 5 + 10 * 5 * 50); // 7 500
  assert.equal(chainCost(DIMS, [0, [1, 2]]).cost, 100 * 5 * 50 + 10 * 100 * 50); // 75 000
  const r = chainCost(DIMS, [[0, 1], 2]);
  assert.deepEqual([r.rows, r.cols], [10, 50]);
  assert.throws(() => chainCost([2, 3, 4, 5], [[0, 2], 1]));
});

test("the dynamic programme finds the cheapest order", () => {
  const b = bestChain(DIMS);
  assert.equal(b.cost, 7500);
  assert.equal(orderText(b.order), "(AB)C");
  assert.equal(orderText([0, [1, 2]]), "A(BC)");
  // four matrices: compare with brute force over the five orders
  const d = [30, 35, 15, 5, 10];
  const orders = [
    [[[0, 1], 2], 3],
    [[0, [1, 2]], 3],
    [[0, 1], [2, 3]],
    [0, [[1, 2], 3]],
    [0, [1, [2, 3]]],
  ] as const;
  const brute = Math.min(...orders.map((o) => chainCost(d, o as never).cost));
  assert.equal(bestChain(d).cost, brute);
  assert.equal(bestChain(d).cost, 9375);
  assert.equal(bestChain([4, 7]).cost, 0);
});

test("roofline: the slope below the ridge, the flat roof above", () => {
  assert.equal(attainable(1, 1000, 50), 50);
  assert.equal(attainable(100, 1000, 50), 1000);
  assert.equal(ridge(1000, 50), 20);
  assert.equal(attainable(ridge(1000, 50), 1000, 50), 1000);
});

test("arithmetic intensity of common kernels (float32)", () => {
  assert.equal(intensity(kernelCost("add", 1e6)), 1 / 12);
  assert.equal(intensity(kernelCost("dot", 1e6)), 1 / 4);
  assert.ok(Math.abs(intensity(kernelCost("gemv", 4096)) - 0.5) < 1e-3);
  assert.equal(intensity(kernelCost("gemm", 600)), 100); // n / 6 for float32
  assert.equal(intensity(kernelCost("gemm", 600, 8)), 50); // n / 12 for float64
  assert.ok(intensity(kernelCost("conv", 224 * 224)) > 100);
  // intensity of GEMM grows with n, the others do not
  assert.ok(intensity(kernelCost("gemm", 2048)) > 10 * intensity(kernelCost("gemm", 128)));
  assert.equal(intensity(kernelCost("dot", 10)), intensity(kernelCost("dot", 1e7)));
});

test("fusing k element-wise steps moves k times fewer bytes", () => {
  assert.equal(elementwiseBytes(3, 1000), 24000);
  assert.equal(elementwiseBytes(3, 1000, 4, true), 8000);
});
