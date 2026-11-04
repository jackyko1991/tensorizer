// Robust colour range for the 3D viewer. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { robustRange } from "../src/robustRange.ts";

test("one extreme value no longer sets the range, and the high end is marked as clipped", () => {
  const body = Float32Array.from({ length: 1000 }, (_, i) => 1 + (6 * i) / 999); // 1 … 7 Pa
  const r = robustRange([body, Float32Array.of(37.8)])!;
  assert.ok(r.hi < 7.1 && r.hi > 6.5, `hi ${r.hi}`);
  assert.ok(r.lo < 1.3 && r.lo >= 1, `lo ${r.lo}`);
  assert.equal(r.max, Float32Array.of(37.8)[0]);
  assert.equal(r.clipHigh, true);
});

test("a constant field keeps its value and is not clipped", () => {
  const r = robustRange([new Float32Array(50).fill(2)])!;
  assert.deepEqual([r.lo, r.hi, r.clipLow, r.clipHigh], [2, 2, false, false]);
});

test("mostly constant data fall back to min/max instead of an empty range", () => {
  const v = new Float32Array(1000).fill(0);
  v[999] = 5;
  const r = robustRange([v])!;
  assert.deepEqual([r.lo, r.hi], [0, 5]);
  assert.equal(r.clipHigh, false);
});

test("NaN and infinities are ignored; empty input gives null", () => {
  const r = robustRange([Float32Array.of(NaN, 1, 2, Infinity, 3)])!;
  assert.equal(r.min, 1);
  assert.equal(r.max, 3);
  assert.equal(robustRange([new Float32Array(0)]), null);
});
