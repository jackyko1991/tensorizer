import {test} from "node:test";
import assert from "node:assert/strict";
import {tensorPieces,EXAMPLE_INPUT} from "../src/tensorDecomposition.ts";
const near=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test("four read-and-send pieces reproduce the worked example",()=>{
 const d=tensorPieces(0);
 assert.deepEqual(d.input,[1,2]);
 d.terms.forEach((t,i)=>t.vector.forEach((x,j)=>near(x,[[2,0],[2,0],[0,0],[0,6]][i][j])));
 assert.deepEqual(d.result,[4,6]);
});
test("changing the basis preserves the complete tensor, input and output",()=>{
 for(const angle of [0,13,37,45,72,90]){
  const d=tensorPieces(angle);
  for(let r=0;r<2;r++){
   near(d.basis.reduce((s,e,i)=>s+e[r]*d.input[i],0),EXAMPLE_INPUT[r]);
   for(let c=0;c<2;c++){
    const reconstructed=d.terms.reduce((s,t)=>s+t.weight*d.basis[t.i][r]*d.basis[t.j][c],0);
    near(reconstructed,[[2,1],[0,3]][r][c]);
   }
  }
  near(d.result[0],4); near(d.result[1],6);
 }
});
