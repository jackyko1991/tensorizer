// Pure helpers of the "Migrating CPU-only ML to tensors" tab. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { barSpan, fmtMetric, fmtTick, fmtTime, fmtX, HI_DEC, LO_DEC, speedupPos, ticks } from "../src/cpumlChart.ts";

test("the speed-up axis is logarithmic and clamped", () => {
  assert.equal(speedupPos(10 ** LO_DEC), 0);
  assert.equal(speedupPos(10 ** HI_DEC), 100);
  assert.equal(speedupPos(1e-9), 0);
  assert.equal(speedupPos(1e9), 100);
  const one = speedupPos(1);
  assert.ok(Math.abs(speedupPos(10) - one - (one - speedupPos(0.1))) < 1e-9, "symmetric around 1x");
});

test("bars start at 1x and grow right when faster, left when slower", () => {
  const one = speedupPos(1);
  assert.deepEqual(barSpan(1), { left: one, width: 0 });
  const fast = barSpan(100);
  assert.equal(fast.left, one);
  assert.ok(Math.abs(fast.left + fast.width - speedupPos(100)) < 1e-9);
  const slow = barSpan(0.1);
  assert.ok(Math.abs(slow.left - speedupPos(0.1)) < 1e-9);
  assert.ok(Math.abs(slow.left + slow.width - one) < 1e-9);
});

test("ticks cover every decade", () => {
  assert.equal(ticks().length, HI_DEC - LO_DEC + 1);
  assert.deepEqual(ticks().map(fmtTick), ["0.01×", "0.1×", "1×", "10×", "100×", "1000×"]);
});

test("numbers are formatted for the page", () => {
  assert.equal(fmtX(2.345), "×2.3");
  assert.equal(fmtX(0.0712), "×0.071");
  assert.equal(fmtX(1542), "×1,542");
  assert.equal(fmtTime(2.5), "2.5 s");
  assert.equal(fmtTime(0.0123), "12 ms");
  assert.equal(fmtTime(4.2e-5), "42 µs");
  assert.equal(fmtMetric(0.868, "pct"), "86.8 %");
  assert.equal(fmtMetric(7.46e-8, "sci"), "7.5e-8");
  assert.equal(fmtMetric(763, "int"), "763");
  assert.equal(fmtMetric(18.954182, "num"), "18.9542");
  assert.equal(fmtMetric(349.40698, "num"), "349.41");
  assert.equal(fmtMetric(null, "num"), "–");
});
