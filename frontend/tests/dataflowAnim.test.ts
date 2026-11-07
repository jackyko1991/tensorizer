// The Concept "Tensor dataflow" tab's animations: every card has an illustration, every flow.anim.* / flow.code.*
// key they use exists in English and in every translation, and the step logic matches explicit references.
// Run: node --test frontend/tests/*.test.ts
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import {
  CON,
  FLOAT_BITS,
  FLOW_CASES,
  PAIR_N,
  SCENARIOS,
  VIEW_STRIDES,
  agGraph,
  asMatrix,
  checks,
  chooseDevice,
  conK,
  conM,
  digitsOf,
  einsumGemm,
  einsumLoop,
  memTrips,
  pairAB,
  stageAt,
  stepCount,
  totalFrames,
  viewOffset,
} from "../src/dataflowAnim.ts";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const page = read("pages/DataflowConcept.tsx");
const anim = read("components/DataflowAnim.tsx");

function usedKeys(): Set<string> {
  const keys = new Set<string>();
  for (const src of [anim, page])
    for (const m of src.matchAll(/["'`](flow\.(?:anim|code)\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  for (const c of FLOW_CASES) {
    keys.add(`flow.anim.${c}.caption`);
    for (let k = 0; k < stepCount(c); k++) keys.add(`flow.anim.${c}.s${k}`);
  }
  for (const c of codeNotes()) keys.add(`flow.code.${c}`);
  for (let k = 0; k < 3; k++) keys.add(`flow.anim.s7.m${k}`);
  return keys;
}

const sections = (): string[] => JSON.parse(page.match(/const SECTIONS = (\[[^\]]*\])/)![1]);
const codeNotes = (): string[] => JSON.parse(page.match(/const CODE_NOTES = (\[[^\]]*\])/)![1]);

const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));

test("the animations' flow.anim.* and flow.code.* keys exist in pages.en.json and in every translation", () => {
  const keys = usedKeys();
  assert.ok(keys.size >= 60, `found only ${keys.size} keys`);
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  for (const f of langs) {
    const text = JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
});

test("every card of the tab has an illustration", () => {
  const ids = sections();
  assert.deepEqual(ids, ["s1", "s2", "mem", "s3", "s4", "s5", "s6", "s7"]);
  const own: Record<string, string> = { s1: "<TensorLab", mem: "<PairMemory", s6: "<FrameworkTable" };
  for (const id of ids) {
    if (own[id]) assert.ok(page.includes(own[id]), `${id} lost ${own[id]}`);
    else assert.ok((FLOW_CASES as string[]).includes(id), `${id} has no DataflowAnim case`);
  }
  assert.ok(page.includes("<DataflowAnim"));
  // every case has a picture in the component
  for (const c of FLOW_CASES) assert.match(anim, new RegExp(`\\b${c}: \\{ draw:`), `no picture for ${c}`);
});

test("the stage plans are consistent", () => {
  for (const c of FLOW_CASES) {
    const n = stepCount(c);
    const seen = new Set<number>();
    for (let g = 0; g < totalFrames(c); g++) seen.add(stageAt(c, g).step);
    assert.equal(seen.size, n, `${c}: every caption is shown`);
    assert.deepEqual(stageAt(c, 10_000), stageAt(c, totalFrames(c) - 1));
  }
  assert.equal(stageAt("s2", PAIR_N * PAIR_N - 1).step, 0);
  assert.equal(stageAt("s2", PAIR_N * PAIR_N).step, 1);
});

test("s2: the loop visits every pair once; the broadcast views read x with stride 0", () => {
  const seen = new Set<string>();
  for (let k = 0; k < PAIR_N * PAIR_N; k++) seen.add(pairAB(k).join(","));
  assert.equal(seen.size, PAIR_N * PAIR_N);
  assert.deepEqual(pairAB(PAIR_N + 2), [1, 2]);
  // x[None, :, :] at [a, b, i] reads x[b, i]; x[:, None, :] reads x[a, i]
  for (let a = 0; a < PAIR_N; a++)
    for (let b = 0; b < PAIR_N; b++)
      for (let i = 0; i < 3; i++) {
        assert.equal(viewOffset(VIEW_STRIDES.rows, a, b, i), 3 * b + i);
        assert.equal(viewOffset(VIEW_STRIDES.cols, a, b, i), 3 * a + i);
      }
});

test("s4: einsum abi,b->ai equals the matrix product of the reshaped K", () => {
  assert.deepEqual(einsumGemm(conK, conM), einsumLoop(conK, conM));
  const M = asMatrix(conK);
  assert.equal(M.length, CON.A * CON.I);
  assert.equal(M[0].length, CON.B);
  assert.equal(M[3][2], conK[1][2][1]); // row a·I + i
  const F = einsumLoop(conK, conM);
  assert.equal(F[2][1], conK[2].reduce((s, Kb, b) => s + Kb[1] * conM[b], 0));
  // all cells small enough to print
  for (const row of M) for (const v of row) assert.ok(Math.abs(v) <= 3);
});

test("s4: 0.1 is right to about 7 digits in float32 and about 16 in float64", () => {
  const f32 = digitsOf("float32");
  const f64 = digitsOf("float64");
  assert.equal(f32.ok + f32.bad, Math.fround(0.1).toPrecision(18));
  assert.equal(f64.ok + f64.bad, (0.1).toPrecision(22));
  const sig = (s: string) => s.length - 2; // digits after "0."
  assert.ok(sig(f32.ok) >= 7 && sig(f32.ok) <= 9, f32.ok);
  assert.ok(sig(f64.ok) >= 16 && sig(f64.ok) <= 18, f64.ok);
  assert.equal(FLOAT_BITS.float32.reduce((a, b) => a + b), 32);
  assert.equal(FLOAT_BITS.float64.reduce((a, b) => a + b), 64);
});

test("s5: the backward pass gives grad phi = GM x / r^3", () => {
  const { r, phi, grad } = agGraph([1, 0.5, 0], 1);
  assert.ok(Math.abs(r - Math.sqrt(1.25)) < 1e-12);
  assert.ok(Math.abs(phi + 1 / r) < 1e-12);
  const want = [1, 0.5, 0].map((v) => v / r ** 3);
  grad.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-12));
  // finite difference
  const h = 1e-6;
  const f = (x: number[]) => -1 / Math.hypot(...x);
  assert.ok(Math.abs((f([1 + h, 0.5, 0]) - f([1 - h, 0.5, 0])) / (2 * h) - grad[0]) < 1e-6);
  assert.ok(memTrips(true) < memTrips(false));
});

test("s7: the flow chart follows runtime.py's device()", () => {
  const runtime = readFileSync(new URL("../../tensorizer_physics/core/runtime.py", import.meta.url), "utf-8");
  assert.ok(runtime.includes('if choice == "mps" and torch.backends.mps.is_available():'));
  assert.ok(runtime.includes('if choice in ("auto", "cuda") and torch.cuda.is_available():'));
  const want = ["cuda", "mps", "cpu", "cpu"];
  SCENARIOS.forEach((s, k) => {
    assert.equal(chooseDevice(s.choice, s.has), want[k]);
    const [m, c] = checks(s.choice, s.has);
    assert.equal(m ? "mps" : c ? "cuda" : "cpu", want[k]);
  });
});

test("no em dashes in the animation sources", () => {
  for (const src of [anim, read("dataflowAnim.ts")]) assert.doesNotMatch(src, /[—―]| – /);
});
