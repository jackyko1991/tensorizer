// NVLink tab helpers: collective cost factors, the ring all-reduce simulation, the DGX-1 mesh. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MESH_EDGES,
  busFactor,
  fmtSeconds,
  hops,
  popcount,
  ringAllReduce,
  ringAllReduceSeconds,
  ringLatencySteps,
  treeLatencySteps,
} from "../src/nvlinkModel.ts";

test("bus bandwidth factors as in nccl-tests PERFORMANCE.md", () => {
  assert.equal(busFactor("allreduce", 8), 1.75);
  assert.equal(busFactor("allreduce", 2), 1);
  assert.equal(busFactor("allgather", 4), 0.75);
  assert.equal(busFactor("reducescatter", 4), 0.75);
  assert.equal(busFactor("broadcast", 8), 1);
  assert.equal(busFactor("allreduce", 1), 0);
});

test("ring all-reduce time: 2(N-1)/N * S / B (+ latency)", () => {
  const GB = 1e9;
  assert.ok(Math.abs(ringAllReduceSeconds(GB, 8, 32 * GB) - 0.0546875) < 1e-12); // PCIe Gen4 x16
  assert.ok(Math.abs(ringAllReduceSeconds(GB, 8, 450 * GB) - 1.75 / 450) < 1e-12); // NVLink 4
  assert.ok(Math.abs(ringAllReduceSeconds(GB, 8, 450 * GB, 1e-5) - (1.75 / 450 + 14e-5)) < 1e-12);
  assert.equal(ringAllReduceSeconds(GB, 1, GB), 0);
  assert.equal(fmtSeconds(0.0546875), "54.7 ms");
  assert.equal(fmtSeconds(1.75 / 450), "3.9 ms");
  assert.equal(fmtSeconds(2), "2.00 s");
  assert.equal(fmtSeconds(0.0002), "200 µs");
});

test("tree latency grows like log N, ring like N", () => {
  assert.equal(ringLatencySteps(8), 14);
  assert.equal(treeLatencySteps(8), 6);
  assert.equal(treeLatencySteps(1024), 20);
  assert.ok(treeLatencySteps(72) < ringLatencySteps(72));
});

test("ring all-reduce simulation ends with the full sum everywhere", () => {
  for (const n of [2, 3, 4, 8]) {
    const states = ringAllReduce(n);
    const full = (1 << n) - 1;
    assert.equal(states.length, 2 * (n - 1) + 1);
    // after reduce-scatter GPU i owns the full chunk (i + 1) mod n, and only that one
    const rs = states[n - 1].held;
    for (let g = 0; g < n; g++)
      for (let c = 0; c < n; c++) assert.equal(rs[g][c] === full, c === (g + 1) % n, `n=${n} gpu ${g} chunk ${c}`);
    for (const row of states[states.length - 1].held) assert.deepEqual(row, Array(n).fill(full));
    // every step: each GPU sends exactly one chunk to its right neighbour
    for (const s of states.slice(1)) {
      assert.equal(s.sends.length, n);
      for (const t of s.sends) assert.equal(t.to, (t.from + 1) % n);
    }
  }
  assert.equal(popcount(0b1011), 3);
});

test("DGX-1 hybrid cube-mesh: 16 links, 4 per GPU, GPU 0 to GPU 5 takes two hops", () => {
  assert.equal(MESH_EDGES.length, 16);
  for (let g = 0; g < 8; g++) assert.equal(MESH_EDGES.filter(([a, b]) => a === g || b === g).length, 4, `GPU ${g}`);
  assert.equal(hops(MESH_EDGES, 0, 1), 1);
  assert.equal(hops(MESH_EDGES, 0, 4), 1);
  assert.equal(hops(MESH_EDGES, 0, 5), 2);
  for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) assert.ok(hops(MESH_EDGES, a, b) <= 2);
});
