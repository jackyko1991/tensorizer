import assert from "node:assert/strict";
import {test} from "node:test";
import {readFileSync} from "node:fs";
import {TILE_A,TILE_B,TILE_C,TILE_D,scalarTile,productCell,tensorCoreFrame} from "../src/tensorCoreModel.ts";
test("scalar FMAs and one tile MMA describe the same 4x4 result",()=>{
 const reference=TILE_A.map((row,i)=>TILE_B[0].map((_,j)=>TILE_C[i][j]+row.reduce((sum,a,k)=>sum+a*TILE_B[k][j],0)));
 assert.deepEqual(TILE_D,reference);assert.equal(TILE_D[0][0],7);
 assert.deepEqual(scalarTile(0),TILE_C);assert.deepEqual(scalarTile(64),reference);
 for(let q=0;q<64;q++){
  const before=scalarTile(q),after=scalarTile(q+1),{i,j,k}=productCell(q);
  assert.equal(after[i][j]-before[i][j],TILE_A[i][k]*TILE_B[k][j]);
  for(let r=0;r<4;r++)for(let c=0;c<4;c++)if(r!==i||c!==j)assert.equal(after[r][c],before[r][c]);
 }
 assert.equal(tensorCoreFrame(4).scalarUpdates,1);assert.equal(tensorCoreFrame(67).scalarUpdates,64);
 assert.equal(tensorCoreFrame(68).tileActive,true);assert.equal(tensorCoreFrame(70).tileDone,true);
 assert.equal(tensorCoreFrame(72).phase,3);
});
test("the Tensor Core example has all ten localized explanations",()=>{
 for(const lang of ["en","zh-Hant","zh-Hans","fr","de","es","ja","ru","ko","hi"]){
  const d=JSON.parse(readFileSync(new URL(`../src/content/pages.${lang}.json`,import.meta.url),"utf8"));
  for(const key of ["tabVector","tabHardware","intro","title","s0","s1","s2","s3","caption","result"])assert.ok(d["flow.tensorcore."+key],lang+":"+key);
 }
});
