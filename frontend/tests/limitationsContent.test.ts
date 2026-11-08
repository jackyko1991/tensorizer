import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import katex from "katex";
import { jointProbabilities, loopRadii, LIMITATION_EQUATIONS } from "../src/limitationsModel.ts";

const content = JSON.parse(readFileSync(new URL("../src/content/limitations.json", import.meta.url), "utf8"));
test("limitations stories retain bilingual copy and valid optional mathematics", () => {
  assert.equal(new Set(content.sections.map((s: {id:string})=>s.id)).size,7);
  const visit=(value: unknown):void=>{
    if(Array.isArray(value)){
      if(value.every(v=>typeof v==="string")){
        assert.equal(value.length,2,"localized text has English and Chinese");
        for(const s of value){
          assert.ok(s.trim());
          for(const m of s.matchAll(/\$([^$]+)\$/g)) assert.doesNotThrow(()=>katex.renderToString(m[1],{throwOnError:true}));
        }
      }else value.forEach(visit);
    }else if(value&&typeof value==="object")Object.values(value).forEach(visit);
  };
  visit(content);
  for(const equation of Object.values(LIMITATION_EQUATIONS))assert.doesNotThrow(()=>katex.renderToString(equation,{throwOnError:true}));
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
