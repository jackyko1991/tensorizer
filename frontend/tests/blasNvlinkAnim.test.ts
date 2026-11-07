// BLAS and NVLink tab animations and code notes: every card has its animation, every code block its plain
// explanation, every new key exists in all languages, and the numbers the pictures show are right.
// Run: node --test frontend/tests/blasNvlinkAnim.test.ts
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import {
  ATOMIC_RUNS,
  BLAS_CARDS,
  BLAS_CODE,
  BLAS_PLAN,
  H100_LINKS,
  NV_CARDS,
  NV_CODE,
  NV_GENS,
  NV_PLAN,
  PCIE_GENS,
  SUM_INPUTS,
  accumulate,
  crossover,
  fmt9,
  gemmLoads,
  genTotal,
  levelCost,
  modelGflops,
  OVERHEAD,
  poolTime,
  stepAt,
  stepMs,
  sumOrder,
  totalTicks,
  type StepPlan,
} from "../src/blasNvlinkAnim.ts";
import { hops, MESH_EDGES, ringAllReduceSeconds, ringLatencySteps, treeLatencySteps } from "../src/nvlinkModel.ts";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const TABS = {
  blas: { src: read("pages/BlasTab.tsx"), plan: BLAS_PLAN as Record<string, StepPlan>, cards: BLAS_CARDS, code: BLAS_CODE, comp: "BlasAnim" },
  nvlink: { src: read("pages/NvlinkTab.tsx"), plan: NV_PLAN as Record<string, StepPlan>, cards: NV_CARDS, code: NV_CODE, comp: "NvlinkAnim" },
};

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));
const load = (f: string) => JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;

function newKeys(): string[] {
  const keys: string[] = [];
  for (const [p, t] of Object.entries(TABS)) {
    keys.push(`${p}.anim.codeNotes`);
    for (const [id, plan] of Object.entries(t.plan)) {
      keys.push(`${p}.anim.${id}.title`, `${p}.anim.${id}.caption`);
      plan.steps.forEach((_, k) => keys.push(`${p}.anim.${id}.s${k}`));
      for (const l of plan.labels) keys.push(`${p}.anim.${id}.${l}`);
    }
    for (const [id, n] of Object.entries(t.code)) for (let k = 0; k < n; k++) keys.push(`${p}.anim.code.${id}.${k}`);
  }
  return keys;
}

/** Text of each <Card id="..."> block of a tab. */
function cardBodies(src: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of src.matchAll(/<Card id="(\w+)">([\s\S]*?)<\/Card>/g)) out[m[1]] = m[2];
  return out;
}

test("every card of both tabs shows its animations, and every animation is on its card", () => {
  for (const [p, t] of Object.entries(TABS)) {
    const sections = t.src.match(/const SECTIONS = \[([^\]]+)\]/)![1].match(/"(\w+)"/g)!.map((s) => s.slice(1, -1));
    assert.deepEqual(Object.keys(t.cards).sort(), [...sections].sort(), p);
    const bodies = cardBodies(t.src);
    const all = new Set<string>();
    for (const s of sections) {
      assert.ok(t.cards[s].length >= 1, `${p}.${s} has no animation`);
      for (const id of t.cards[s]) {
        assert.match(bodies[s], new RegExp(`<${t.comp} id="${id}"`), `${p}.${s} lacks ${id}`);
        all.add(id);
      }
    }
    assert.deepEqual([...all].sort(), Object.keys(t.plan).sort(), `${p}: plan and cards differ`);
  }
});

test("every code block is followed by its plain explanation", () => {
  for (const [p, t] of Object.entries(TABS)) {
    const blocks = (t.src.match(/<CodeBlock /g) ?? []).length + (t.src.match(/<pre /g) ?? []).length;
    const notes = [...t.src.matchAll(/<CodeNotes prefix="(\w+)" id="(\w+)"/g)];
    assert.equal(notes.length, blocks, `${p}: ${blocks} code blocks, ${notes.length} notes`);
    assert.deepEqual(notes.map((m) => m[2]).sort(), Object.keys(t.code).sort(), p);
    for (const m of notes) assert.equal(m[1], p);
  }
});

test("every new key exists, non-empty and without em dashes, in all 10 languages", () => {
  const keys = newKeys();
  assert.ok(keys.length > 150, `only ${keys.length} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = load(f);
    const missing = keys.filter((k) => typeof text[k] !== "string" || !text[k].trim());
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
    const dash = keys.filter((k) => /—/.test(text[k]));
    assert.deepEqual(dash, [], `${f}: em dash in ${dash.join(", ")}`);
  }
  for (const s of [read("components/BlasAnim.tsx"), read("components/NvlinkAnim.tsx"), read("blasNvlinkAnim.ts")]) assert.doesNotMatch(s, /—/);
});

test("stepAt walks every plan in order and clamps", () => {
  for (const t of Object.values(TABS))
    for (const [id, plan] of Object.entries(t.plan)) {
      const n = totalTicks(plan);
      assert.ok(plan.tick >= 200 && n >= 8 && n <= 80, id);
      let prev = 0;
      const seen = new Set<number>();
      for (let g = 0; g < n; g++) {
        const s = stepAt(plan, g);
        assert.ok(s.step >= prev && s.f > 0 && s.f <= 1, `${id} @ ${g}`);
        prev = s.step;
        seen.add(s.step);
      }
      assert.equal(seen.size, plan.steps.length, id);
      assert.deepEqual(stepAt(plan, 10 ** 6), stepAt(plan, n - 1));
      assert.deepEqual(stepAt(plan, -1), stepAt(plan, 0));
    }
});

test("BLAS levels: FLOPs per value 2/3, 2 and n/2, as in the tab's table", () => {
  const r = (l: 1 | 2 | 3, n: number) => levelCost(l, n).flops / levelCost(l, n).moved;
  assert.equal(r(1, 1000), 2 / 3);
  assert.equal(r(2, 1000), 2);
  assert.equal(r(3, 1000), 500);
  assert.equal(gemmLoads(8), 1024);
  assert.equal(gemmLoads(8, 4), 256);
});

test("summation order: three orders give three different float32 sums of the same eight numbers", () => {
  const res = (["seq", "pair", "lanes"] as const).map((o) => sumOrder(o));
  for (const adds of res) assert.equal(adds.length, SUM_INPUTS.length - 1);
  const finals = res.map((a) => fmt9(a[a.length - 1].value));
  assert.deepEqual(finals, ["4.00000048", "4.00000000", "3.99999976"]);
  // each order is a sum of all eight inputs exactly once
  for (const adds of res) {
    const leaves = new Set<number>();
    for (const a of adds) for (const x of [a.left, a.right]) if (x < 8) leaves.add(x);
    assert.equal(leaves.size, 8);
  }
});

test("atomics: two arrival orders differ in the last bits, the fixed order repeats", () => {
  const [r1, r2] = ATOMIC_RUNS.atomic.map((o) => accumulate(o).at(-1)!);
  assert.notEqual(fmt9(r1), fmt9(r2));
  for (const o of ATOMIC_RUNS.atomic) assert.deepEqual([...o].sort(), [0, 1, 2, 3]);
  assert.equal(fmt9(accumulate(ATOMIC_RUNS.fixed).at(-1)!), "2.00000000");
});

test("fixed-cost model: rises to the peak; the GPU wins only above a crossover; threads model", () => {
  const { cpu, gpu } = OVERHEAD;
  assert.ok(modelGflops(16, cpu.t0, cpu.peak) < 0.1 * cpu.peak);
  assert.ok(modelGflops(4096, cpu.t0, cpu.peak) > 0.99 * cpu.peak);
  const c = crossover();
  assert.ok(c >= 32 && c <= 512, `crossover ${c}`);
  assert.equal(poolTime(12, 1, 6), 12);
  assert.equal(poolTime(12, 6, 6), 2);
  assert.ok(poolTime(12, 12, 6) > 2);
});

test("NVLink numbers match the tab's table and links", () => {
  const src = TABS.nvlink.src;
  for (const g of NV_GENS) {
    const m = src.match(new RegExp(`gen: "${g.gen}", gpu: "[^"]*", year: \\d+, links: (\\d+), perDir: (\\d+), total: (\\d+)`));
    assert.ok(m, g.gen);
    assert.deepEqual([Number(m[1]), Number(m[2]), Number(m[3])], [g.links, g.perDir, genTotal(g)], g.gen);
  }
  for (const p of PCIE_GENS) assert.match(src, new RegExp(`gen: "${p.gen} x16", perDir: "[\\d.]+", total: ${p.total}`), p.gen);
  assert.match(src, new RegExp(`name: "NVLink 4 \\(H100\\)", perDir: ${H100_LINKS.nvlink4}`));
  assert.match(src, new RegExp(`name: "PCIe Gen5 x16", perDir: ${H100_LINKS.pcie5}`));
  // topology: 16 pairs one hop, 12 two hops in the cube-mesh
  const counts = [0, 0, 0];
  for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) counts[hops(MESH_EDGES, a, b)]++;
  assert.deepEqual(counts, [0, 16, 12]);
  assert.equal(ringLatencySteps(8), 14);
  assert.equal(treeLatencySteps(8), 6);
  // worked example: compute 50 ms + all-reduce of 1 GB over 8 GPUs
  const pcie = ringAllReduceSeconds(1e9, 8, 32e9) * 1e3;
  const nvl = ringAllReduceSeconds(1e9, 8, 450e9) * 1e3;
  assert.ok(stepMs(50, pcie) > 100 && stepMs(50, nvl) < 55);
  assert.ok(stepMs(50, pcie, true) < stepMs(50, pcie));
});
