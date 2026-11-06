// BLAS tab chart helpers. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  decades,
  fmtGflops,
  gemmFlops,
  gemmIntensity,
  gflops,
  linePath,
  logScale,
  percentOfPeak,
  roofline,
} from "../src/blasChart.ts";

test("FLOPs and GFLOP/s of an n x n matrix product", () => {
  assert.equal(gemmFlops(2), 16);
  assert.equal(gemmFlops(1000), 2e9);
  assert.equal(gflops(1000, 2), 1);
});

test("GEMM intensity grows like n; the roofline caps it at the peak", () => {
  assert.equal(gemmIntensity(256, 8), 16); // float64: n / 16 FLOP per byte
  assert.equal(gemmIntensity(256, 4), 32);
  assert.equal(roofline(0.1, 400, 40), 4); // memory-bound
  assert.equal(roofline(100, 400, 40), 400); // compute-bound
});

test("log scale maps decades evenly and clamps", () => {
  const s = logScale(10, 1000, 0, 100);
  assert.equal(s(10), 0);
  assert.equal(s(100), 50);
  assert.equal(s(1000), 100);
  assert.equal(s(1), 0);
  assert.equal(s(1e6), 100);
  const inv = logScale(1, 100, 200, 0); // SVG y grows downwards
  assert.equal(inv(10), 100);
});

test("decade ticks, number format, percent of peak, path", () => {
  assert.deepEqual(decades(5, 20000), [1, 10, 100, 1000, 10000, 100000]);
  assert.deepEqual(decades(10, 1000), [10, 100, 1000]);
  assert.equal(fmtGflops(10116), "10k");
  assert.equal(fmtGflops(1500), "1.5k");
  assert.equal(fmtGflops(1000), "1k");
  assert.equal(fmtGflops(335.2), "335");
  assert.equal(fmtGflops(47.6), "48");
  assert.equal(fmtGflops(7.44), "7.4");
  assert.equal(percentOfPeak(335, 749), 45);
  assert.equal(percentOfPeak(335, null), null);
  assert.equal(linePath([[0, 1], [2.25, 3]]), "M0.0,1.0L2.3,3.0");
});
