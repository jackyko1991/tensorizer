// The Chips tab's animation plans and numbers. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { BITS_STEPS, BUDGETS, CHIPS_PLAN, FEED, MIXED_PLAN, MMA, SCALE_GRADS, bitsStep, busyFraction, lossScaling, scalarCycles, stallAt, sumCurves } from "../src/chipsAnim.ts";

test("bits: π loses precision step by step, FP4 holds 3", () => {
  const errs = BITS_STEPS.map((_, i) => bitsStep(i).error);
  for (let i = 1; i < errs.length; i++) assert.ok(errs[i] >= errs[i - 1] * 0.5, `step ${i}`); // broadly growing
  assert.equal(bitsStep(4).value, 3);
  assert.equal(bitsStep(1).value, 3.140625);
  assert.ok(bitsStep(0).error < 1e-7);
});

test("sum: naive FP16 stalls, FP32 accumulation and Kahan track the exact total", () => {
  const c = sumCurves(4096);
  const stall = stallAt(c.naive);
  assert.ok(stall > 100 && stall < 4096, `stall at ${stall}`);
  assert.ok(c.naive.at(-1)! < 300);
  assert.ok(Math.abs(c.wide.at(-1)! - 409.6) < 1);
  assert.ok(Math.abs(c.kahan.at(-1)! - 409.6) < 1);
});

test("budget shares add up to one per chip", () => {
  for (const b of BUDGETS) assert.ok(Math.abs(b.blocks.reduce((s, x) => s + x.share, 0) - 1) < 1e-9, b.id);
});

test("feed: decoding keeps a datacenter GPU's units mostly idle, a laptop's less so", () => {
  const busy = FEED.map((f) => busyFraction(f.gbs, f.tflops));
  assert.ok(busy.every((b) => b > 0 && b <= 1));
  assert.ok(busy[2] < 0.01); // H100: 3.35 TB/s feeds under 1% of 989 TFLOP/s at 1 FLOP/byte
});

test("mma: a 4×4×4 product is 64 multiply-adds", () => {
  assert.equal(scalarCycles(MMA.n, MMA.scalarPerCycle), 16);
  for (const p of [...Object.values(CHIPS_PLAN), ...Object.values(MIXED_PLAN)]) assert.ok(p.steps.length >= 2 && p.tick > 0);
});

test("dynamic loss scaling: halve and skip on overflow, double after good steps", () => {
  const r = lossScaling([1, 1, 0.01, 0.01, 0.01, 0.01, 0.01], 2 ** 16, 4);
  // 1 × 2^16 > 65504 overflows (skip, halve); four good steps at 2^15, then back to 2^16, where 0.01 × 2^16 is fine
  assert.deepEqual(r.map((x) => x.overflow), [true, false, false, false, false, false, false]);
  assert.deepEqual(r.map((x) => Math.log2(x.scale)), [16, 15, 15, 15, 15, 16, 16]);
  const sim = lossScaling(SCALE_GRADS);
  assert.ok(sim.some((x) => x.overflow) && sim.some((x) => !x.overflow));
  for (const x of sim) if (!x.overflow) assert.ok(x.scale * 0.002 > 6e-8); // scaled small gradients stay above FP16's subnormals
});

test("feed uses the table's own numbers", () => {
  const chips = JSON.parse(readFileSync(new URL("../src/content/chips/chips.json", import.meta.url), "utf-8")).chips as { id: string; bandwidthGBs: number; bf16TF: number }[];
  for (const f of FEED) {
    const c = chips.find((x) => x.id === f.id);
    assert.ok(c, f.id);
    assert.equal(c.bandwidthGBs, f.gbs, f.id);
    assert.equal(c.bf16TF, f.tflops, f.id);
  }
});
