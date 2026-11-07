// The Tensor maths animations: step plans, page placement, every computation against an explicit reference,
// and every maths.anim.* text key in all languages. Run: node --test frontend/tests/*.test.ts
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import {
  BMM,
  CHAIN_DIMS,
  COUNT,
  DOT,
  EX_A,
  FUSION,
  HILL,
  LOOP,
  MATHS_CASES,
  MATHS_PLACE,
  MATHS_PLAN,
  MM,
  MPS,
  MV,
  OUTER,
  RANK_DIMS,
  RECEIPT,
  ROT_A,
  ROT_MAX,
  ROT_R,
  ROT_V,
  TR,
  UNITS,
  att,
  busy,
  chainSteps,
  contourR,
  crossings,
  hillZ,
  countAt,
  intensityOf,
  loopTerms,
  lower,
  mFrames,
  mStage,
  mStepCount,
  mStepStart,
  matMul,
  matVec,
  matrixRank,
  metPhi,
  mpsTerms,
  numel,
  rank1Matrix,
  rot,
  rotAngle,
  rotMatrix,
  rows,
  runningSum,
  slanted,
  sweepI,
  upper,
} from "../src/mathsAnim.ts";
import { chainCost } from "../src/tensorMaths.ts";
import { einsum, tensor } from "../src/techAnim.ts";

const ein = (spec: string, ...xs: unknown[]) => einsum(spec, ...xs.map((x) => tensor(x)));

const close = (a: number, b: number, tol = 1e-9, msg?: string) =>
  assert.ok(Math.abs(a - b) <= tol, msg ?? `${a} != ${b} (tol ${tol})`);

const src = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, src), "utf-8");

test("every animation has a plan; mStage walks the stages in order and clamps", () => {
  assert.equal(MATHS_CASES.length, 24);
  for (const c of MATHS_CASES) {
    const n = mFrames(c);
    assert.ok(n >= 4 && n <= 40, `${c}: ${n} ticks`);
    assert.ok(MATHS_PLAN[c].tick >= 200, c);
    let prev = 0;
    const seen = new Set<number>();
    for (let g = 0; g < n; g++) {
      const s = mStage(c, g);
      assert.ok(s.idx >= prev && s.t < s.n, `${c} @ ${g}`);
      prev = s.idx;
      seen.add(s.step);
    }
    assert.equal(seen.size, mStepCount(c), c);
    assert.deepEqual(mStage(c, 1e6), mStage(c, n - 1));
    assert.equal(mStage(c, mStepStart(c, mStepCount(c) - 1)).step, mStepCount(c) - 1);
  }
});

test("every section has an animation, and every equation card has its own", () => {
  for (const s of ["tensor", "einsum", "penrose", "topo", "roof", "recap"])
    assert.ok(MATHS_CASES.some((c) => MATHS_PLACE[c].section === s), s);
  const page = read("pages/MathsConcept.tsx");
  const eqBlock = page.slice(page.indexOf("const EQS"), page.indexOf("// External references"));
  const cards = [...eqBlock.matchAll(/\{ tex: String/g)].length;
  const anims = [...eqBlock.matchAll(/anim: "(\w+)"/g)].map((m) => m[1]);
  assert.equal(anims.length, cards, "one anim per equation card");
  const eqCases = MATHS_CASES.filter((c) => MATHS_PLACE[c].after === "eq");
  assert.deepEqual([...anims].sort(), [...eqCases].sort());
  // the page renders the placed animations
  assert.match(page, /animsAfter\(/);
  assert.match(page, /<MathsAnim /);
});

test("index up or down: components move in opposite directions, the pairing stays put", () => {
  assert.deepEqual(UNITS.map((u) => upper(u.f)), [2, 20, 200]);
  for (const u of UNITS) close(upper(u.f) * lower(u.f), 6);
  close(lower(100), 0.03);
  // the drawn hill: 6 contours on the 2 m walk, 1/3 m apart, each ring at its own height
  const xs = crossings();
  assert.equal(xs.length, upper(1) * lower(1));
  for (let k = 1; k < xs.length; k++) close(xs[k] - xs[k - 1], 1 / 3);
  for (const L of HILL.levels)
    for (const t of [0, 1, 2.5, 4]) close(hillZ(contourR(L) * Math.cos(t), contourR(L) * Math.sin(t)), L);
  close(hillZ(0.1, 0), hillZ(0.0999999, 0), 1e-5); // the rounded top joins the flank
  assert.equal(hillZ(HILL.rMax + 0.1, 0), 0);
  // slanted axes: v_i = g_ij v^j, and at 90 degrees the two kinds coincide
  for (const phi of [Math.PI / 2, metPhi(1, 4, 10), metPhi(2, 0, 8)]) {
    const m = slanted(phi);
    const v = [m.up[0] * m.e1[0] + m.up[1] * m.e2[0], m.up[0] * m.e1[1] + m.up[1] * m.e2[1]];
    close(v[0], 1.3);
    close(v[1], 1.0);
    const gv = matVec(m.g, m.up);
    close(gv[0], m.down[0]);
    close(gv[1], m.down[1]);
  }
  const r = slanted(Math.PI / 2);
  close(r.up[0], r.down[0]);
  close(r.up[1], r.down[1]);
  const s = slanted(metPhi(2, 0, 8));
  assert.ok(Math.abs(s.up[1] - s.down[1]) > 0.2);
});

test("rotations: one R per index; lengths and traces do not change", () => {
  close(Math.cos(ROT_MAX), 0.8);
  const R = rot(ROT_MAX);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) close(R[i][j], ROT_R[i][j]);
  const v2 = matVec(ROT_R, ROT_V);
  close(v2[0], 2);
  close(v2[1], 1);
  const A2 = rotMatrix(ROT_R, ROT_A);
  close(A2[0][0], 1.64);
  close(A2[0][1], -0.48);
  close(A2[1][1], 1.36);
  // explicit A'_ij = R_ik R_jl A_kl
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) {
      let s = 0;
      for (let k = 0; k < 2; k++) for (let l = 0; l < 2; l++) s += ROT_R[i][k] * ROT_R[j][l] * ROT_A[k][l];
      close(A2[i][j], s);
    }
  for (let g = 0; g <= mFrames("rotate"); g++) {
    const th = rotAngle(g);
    assert.ok(th >= -1e-12 && th <= ROT_MAX + 1e-12);
    const w = matVec(rot(th), ROT_V);
    close(Math.hypot(w[0], w[1]), Math.hypot(1, 2));
    const B = rotMatrix(rot(th), ROT_A);
    close(B[0][0] + B[1][1], 3);
  }
  close(rotAngle(0), 0);
});

test("shapes: the photo and the batch", () => {
  assert.equal(numel([480, 640, 3]), 921600);
  assert.equal(numel(RANK_DIMS), 29491200);
});

test("einsum examples equal explicit sums", () => {
  assert.equal(runningSum(RECEIPT.q, RECEIPT.p, 3), 8.5);
  assert.equal(EX_A[0][0] + EX_A[1][1], 5);
  assert.equal(matMul(EX_A, EX_A)[0][0], 7);
  assert.equal(ein("i,i->", DOT.x, DOT.y).data[0], runningSum(DOT.x, DOT.y, 3));
  assert.equal(runningSum(DOT.x, DOT.y, 3), 10);
  assert.deepEqual(ein("ij,j->i", MV.A, MV.x).data, matVec(MV.A, MV.x));
  assert.deepEqual(matVec(MV.A, MV.x), [4, 3, 5]);
  assert.deepEqual(rows(ein("ij,jk->ik", MM.A, MM.B)), matMul(MM.A, MM.B));
  assert.deepEqual(matMul(MM.A, MM.B), [[5, 2], [2, 4]]);
  assert.equal(ein("ii->", TR).data[0], 7);
  assert.deepEqual(rows(ein("i,j->ij", OUTER.u, OUTER.v)), OUTER.u.map((a) => OUTER.v.map((b) => a * b)));
  const C = ein("bij,bjk->bik", BMM.A, BMM.B);
  assert.deepEqual(C.shape, [2, 2, 2]);
  for (let b = 0; b < 2; b++) assert.deepEqual(rows(C).slice(2 * b, 2 * b + 2), matMul(BMM.A[b], BMM.B[b]));
});

test("Penrose: the trace loop and the 3-site MPS", () => {
  const terms = loopTerms();
  const s = terms.reduce((a, t) => a + t.a * t.b, 0);
  const AB = matMul(LOOP.A, LOOP.B);
  const BA = matMul(LOOP.B, LOOP.A);
  assert.equal(s, AB[0][0] + AB[1][1]);
  assert.equal(s, BA[0][0] + BA[1][1]);
  assert.equal(s, 13);
  const T = ein("ia,ajb,bk->ijk", MPS.A, MPS.B, MPS.C);
  assert.deepEqual(T.shape, [2, 2, 2]);
  assert.equal(mpsTerms().reduce((a, t) => a + t.v, 0), T.data[0]);
  const AB3 = ein("ia,ajb->ijb", MPS.A, MPS.B);
  assert.deepEqual(AB3.shape, [2, 2, 2]);
  const nested = [0, 1].map((i) => [0, 1].map((j) => [0, 1].map((b) => AB3.data[i * 4 + j * 2 + b])));
  assert.deepEqual(ein("ijb,bk->ijk", nested, MPS.C).data, T.data);
});

test("contraction order: the costs match tensorMaths and the counted product", () => {
  const st = chainSteps();
  assert.equal(st.left[0].cost + st.left[1].cost, chainCost(CHAIN_DIMS, [[0, 1], 2]).cost);
  assert.equal(st.right[0].cost + st.right[1].cost, chainCost(CHAIN_DIMS, [0, [1, 2]]).cost);
  assert.equal(st.left[0].cost + st.left[1].cost, 7500);
  assert.equal(st.right[0].cost + st.right[1].cost, 75000);
  const n = COUNT.a * COUNT.b * COUNT.c;
  assert.equal(n, 12);
  assert.equal(n, MATHS_PLAN.eqChain.stages[0][1]);
  const seen = new Set<string>();
  for (let k = 0; k < n; k++) {
    const p = countAt(k);
    assert.ok(p.i < COUNT.a && p.k2 < COUNT.c && p.j < COUNT.b);
    seen.add(`${p.i}${p.k2}${p.j}`);
  }
  assert.equal(seen.size, n);
});

test("roofline: min(P, beta I), utilisation and fusion intensity", () => {
  close(att(4), 200);
  close(att(1), 50);
  close(att(100), 200);
  close(sweepI(0), 0.01);
  close(sweepI(12), 100);
  close(busy(0.25).mem, 1);
  close(busy(0.25).alu, 12.5 / 200);
  close(busy(40).alu, 1);
  close(busy(40).mem, 0.1);
  close(intensityOf(FUSION.unfused), 1 / 8);
  close(intensityOf(FUSION.fused), 3 / 8);
});

test("recap: an outer product has tensor rank 2 but matrix rank 1", () => {
  const M = rank1Matrix();
  assert.equal(matrixRank(M), 1);
  assert.equal(matrixRank([[1, 0], [0, 1]]), 2);
  assert.equal(matrixRank(MM.A), 2);
});

// ---- text ---------------------------------------------------------------------------------------------

const dir = new URL("content/", src);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));
const load = (f: string) => JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
const ANIM_FILES = ["components/MathsAnim.tsx", "components/MathsAnimPics.tsx", "components/MathsAnimPics2.tsx"];

function animKeys(): Set<string> {
  const keys = new Set<string>(["maths.anim.title"]);
  for (const c of MATHS_CASES) {
    keys.add(`maths.anim.${c}.caption`);
    for (let k = 0; k < mStepCount(c); k++) keys.add(`maths.anim.${c}.s${k}`);
  }
  for (const f of ANIM_FILES)
    for (const m of read(f).matchAll(/["'`](maths\.anim\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  return keys;
}

test("every maths.anim.* key exists in English and in all translations", () => {
  const keys = animKeys();
  assert.ok(keys.size > 100, `only ${keys.size} keys`);
  assert.ok(langs.length >= 10);
  for (const f of langs) {
    const text = load(f);
    const missing = [...keys].filter((k) => typeof text[k] !== "string" || !text[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
  }
  // the pictures use T("maths.anim.<id>.<word>") for words, through a helper that only takes known keys
  const en = load("pages.en.json");
  for (const k of keys) assert.ok(en[k], k);
});

test("the animation sources have no em dashes outside comments", () => {
  for (const f of [...ANIM_FILES, "mathsAnim.ts"]) {
    const code = read(f)
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/(^|[\s;,{}()])\/\/.*$/gm, "$1");
    assert.ok(!/[—―]| – /.test(code), f);
  }
});
