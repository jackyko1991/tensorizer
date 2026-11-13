// The Chips tab's number formats and derived chip numbers. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { allValues, bitsOf, decodeCeiling, epsilon, fmt, maxValue, minNormal, minSubnormal, ridgePoint, roundTo, runningSum } from "../src/chipsModel.ts";

const close = (a: number, b: number, rel = 1e-9) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b), `${a} vs ${b}`);

test("largest values match the standards (IEEE 754, OCP OFP8, OCP MX)", () => {
  close(maxValue(fmt("fp16")), 65504);
  close(maxValue(fmt("bf16")), 3.3895313892515355e38);
  close(maxValue(fmt("fp32")), 3.4028234663852886e38);
  close(maxValue(fmt("tf32")), 3.4011621342146535e38);
  close(maxValue(fmt("fp8e4m3")), 448);
  close(maxValue(fmt("fp8e5m2")), 57344);
  close(maxValue(fmt("fp6e3m2")), 28);
  close(maxValue(fmt("fp6e2m3")), 7.5);
  close(maxValue(fmt("fp4e2m1")), 6);
});

test("smallest values and epsilon", () => {
  close(minNormal(fmt("fp16")), 6.103515625e-5);
  close(minSubnormal(fmt("fp16")), 5.960464477539063e-8);
  close(minNormal(fmt("bf16")), 1.1754943508222875e-38);
  close(minSubnormal(fmt("fp8e4m3")), 2 ** -9);
  close(minSubnormal(fmt("fp8e5m2")), 2 ** -16);
  close(minSubnormal(fmt("fp4e2m1")), 0.5);
  close(epsilon(fmt("fp16")), 2 ** -10);
  close(epsilon(fmt("bf16")), 2 ** -7);
  close(epsilon(fmt("fp32")), 2 ** -23);
});

test("rounding: ties to even, subnormals, overflow to infinity or saturation", () => {
  assert.equal(roundTo(0.1, fmt("fp16")), 0.0999755859375);
  assert.equal(roundTo(0.1, fmt("bf16")), 0.10009765625);
  assert.equal(roundTo(Math.PI, fmt("bf16")), 3.140625);
  assert.equal(roundTo(Math.PI, fmt("fp8e4m3")), 3.25);
  assert.equal(roundTo(Math.PI, fmt("fp4e2m1")), 3);
  assert.equal(roundTo(2049, fmt("fp16")), 2048); // tie: 2048 is even
  assert.equal(roundTo(2051, fmt("fp16")), 2052);
  assert.equal(roundTo(1e-7, fmt("fp16")), 1.1920928955078125e-7); // subnormal step 2^-24
  assert.equal(roundTo(70000, fmt("fp16")), Infinity);
  assert.equal(roundTo(1000, fmt("fp8e4m3")), 448);
  assert.equal(roundTo(-9, fmt("fp4e2m1")), -6);
  assert.equal(roundTo(Math.PI, fmt("fp32")), Math.fround(Math.PI));
  assert.equal(roundTo(Math.PI, fmt("fp64")), Math.PI);
});

test("the small formats list exactly their values", () => {
  assert.deepEqual(allValues(fmt("fp4e2m1")), [0, 0.5, 1, 1.5, 2, 3, 4, 6]);
  const e4m3 = allValues(fmt("fp8e4m3"));
  assert.equal(e4m3.at(-1), 448);
  assert.equal(e4m3.length, 127); // 128 codes for magnitudes, the last one is NaN
  assert.equal(allValues(fmt("fp8e5m2")).at(-1), 57344);
  for (const v of e4m3) assert.equal(roundTo(v, fmt("fp8e4m3")), v);
});

test("bits of a value", () => {
  const one = bitsOf(1, fmt("fp16"));
  assert.deepEqual([one.sign, one.exp.join(""), one.man.join("")], [0, "01111", "0000000000"]);
  const pi = bitsOf(Math.PI, fmt("bf16"));
  assert.deepEqual([pi.sign, pi.exp.join(""), pi.man.join("")], [0, "10000000", "1001001"]);
  const neg = bitsOf(-6, fmt("fp4e2m1"));
  assert.deepEqual([neg.sign, neg.exp.join(""), neg.man.join("")], [1, "11", "1"]);
  const tiny = bitsOf(2 ** -9, fmt("fp8e4m3")); // smallest subnormal
  assert.deepEqual([tiny.exp.join(""), tiny.man.join("")], ["0000", "001"]);
});

test("summing 10,000 × 0.1: FP16 stalls, FP32 accumulation and Kahan stay close", () => {
  const xs = Array(10000).fill(0.1);
  const exact = 1000;
  const naive = runningSum(xs, fmt("fp16"), "naive").at(-1)!;
  const wide = runningSum(xs, fmt("fp16"), "wide").at(-1)!;
  const kahan = runningSum(xs, fmt("fp16"), "kahan").at(-1)!;
  assert.ok(naive < 300, `naive FP16 total ${naive}`); // the total stops growing once 0.1 is under half a step
  assert.ok(Math.abs(wide - exact * (0.0999755859375 / 0.1)) < 0.05, `wide ${wide}`); // only the input rounding is left
  assert.ok(Math.abs(kahan - 999.7) < 2, `kahan ${kahan}`);
});

test("ridge point and decoding ceiling", () => {
  assert.ok(Math.abs(ridgePoint(989, 3350) - 295.2) < 0.1); // H100 SXM dense BF16 / HBM3
  assert.ok(Math.abs(decodeCeiling(3350, 7, 2) - 239.3) < 0.1);
  assert.ok(Math.abs(decodeCeiling(100, 7, 2) - 7.14) < 0.01);
});
