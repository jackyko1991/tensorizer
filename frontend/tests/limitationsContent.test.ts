import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import katex from "katex";
import { fallingLift, jointProbabilities, loopRadii, magneticLoop, LIMITATION_EQUATIONS, EARTH_TIDE } from "../src/limitationsModel.ts";

// The Limits of tensors tab reads limitations.* keys from pages.<lang>.json. English and both Chinese scripts are
// complete; the other languages fall back to English until they are translated.
const COMPLETE = ["en", "zh-Hant", "zh-Hans"];
const read = (lang: string) => JSON.parse(readFileSync(new URL(`../src/content/pages.${lang}.json`, import.meta.url), "utf8")) as Record<string, string>;
const en = read("en");
const keys = Object.keys(en).filter((k) => k.startsWith("limitations."));
const source = ["pages/LimitationsConcept.tsx", "components/LimitationsFigure.tsx"].map((f) => readFileSync(new URL(`../src/${f}`, import.meta.url), "utf8")).join("\n");

test("limitations text is complete in English and Chinese, with valid inline mathematics", () => {
  assert.ok(keys.length > 140, `found only ${keys.length} keys`);
  for (const lang of COMPLETE) {
    const text = read(lang);
    const missing = keys.filter((k) => typeof text[k] !== "string" || !text[k].trim());
    assert.deepEqual(missing, [], `${lang} lacks ${missing.join(", ")}`);
    for (const k of keys) {
      const math = (s: string) => [...s.matchAll(/\$([^$]+)\$/g)].map((m) => m[1]);
      for (const m of math(text[k])) assert.doesNotThrow(() => katex.renderToString(m, { throwOnError: true }), `${lang} ${k}: ${m}`);
      assert.deepEqual(math(text[k]).sort(), math(en[k]).sort(), `${lang} ${k}: inline math differs from English`);
      assert.equal(text[k].split("|").length, en[k].split("|").length, `${lang} ${k}: table cells`);
      assert.equal((text[k].match(/\*\*\(\d\)\*\*/g) ?? []).length, (en[k].match(/\*\*\(\d\)\*\*/g) ?? []).length, `${lang} ${k}: list items`);
    }
  }
});

test("every section has a title and every equation tab has a label", () => {
  for (const s of ["scope", "spinors", "gravity", "connections", "topology", "entanglement", "frontiers", "recap", "bridge"]) {
    assert.ok(en[`limitations.${s}.title`], s);
    assert.ok(source.includes(`"${s}"`), `${s} is rendered`);
  }
  for (const [s, eqs] of Object.entries(LIMITATION_EQUATIONS)) {
    eqs.forEach((tex, i) => {
      assert.doesNotThrow(() => katex.renderToString(tex, { throwOnError: true }), `${s} ${i}`);
      if (s !== "bridge") assert.ok(en[`limitations.${s}.eq${i + 1}`], `${s}.eq${i + 1}`);
    });
  }
});

test("a falling lift cancels the pull but not the tide", () => {
  assert.deepEqual(fallingLift(0), { dropMetres: 0, squeezeMetres: 0 });
  const two = fallingLift(2);
  assert.ok(Math.abs(two.dropMetres - 19.62) < 1e-9);
  assert.ok(Math.abs(EARTH_TIDE - 1.5414e-6) < 1e-9);
  assert.ok(Math.abs(two.squeezeMetres - 3.083e-6) < 1e-9);
  let previous = fallingLift(0);
  for (let t = 0.1; t <= 2; t += 0.1) {
    const now = fallingLift(t);
    assert.ok(now.dropMetres > previous.dropMetres && now.squeezeMetres > previous.squeezeMetres);
    previous = now;
  }
});

test("the two particle preparations have equal marginals and different joint outcomes",()=>{
  for(const paired of [false,true]){
    const p=jointProbabilities(paired);
    assert.equal(p.flat().reduce((a,b)=>a+b,0),1);
    for(let i=0;i<2;i++){
      assert.equal(p[i][0]+p[i][1],.5);
      assert.equal(p[0][i]+p[1][i],.5);
    }
  }
  assert.notDeepEqual(jointProbabilities(true),jointProbabilities(false));
});
test("loop contraction reaches a point only on the unpunctured sheet",()=>{
  assert.deepEqual(loopRadii(0),[51,51]);
  assert.deepEqual(loopRadii(100),[0,32]);
  let previous=[51,51];
  for(let percent=0;percent<=100;percent++){
    const radius=loopRadii(percent);
    assert.ok(radius[0]>=0 && radius[1]>28);
    assert.ok(radius[0]<=previous[0] && radius[1]<=previous[1]);
    previous=radius;
  }
});

test("fixed enclosed flux gives the electron phase after each winding",()=>{
 const start=magneticLoop(0),one=magneticLoop(360),two=magneticLoop(720);
 assert.ok(Math.abs(start.fluxWebers-1.033916924e-15)<1e-24);
 assert.equal(start.fluxWebers,one.fluxWebers);
 assert.equal(start.fluxWebers,two.fluxWebers);
 assert.equal(start.phaseDegrees,0);
 assert.equal(one.phaseDegrees,-90);
 assert.equal(one.phaseRadians,-Math.PI/2);
 assert.equal(two.phaseDegrees,-180);
 assert.equal(magneticLoop(90).phaseDegrees,-22.5);
});
