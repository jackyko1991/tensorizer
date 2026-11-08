export const TILE_N=4;
export const TILE_A=[[1,2,0,1],[0,1,2,0],[2,0,1,1],[1,1,0,2]];
export const TILE_B=[[1,0,2,1],[2,1,0,1],[0,2,1,0],[1,0,1,2]];
export const TILE_C=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i===j?1:0));
export const TILE_PHASE_STARTS=[0,4,68,72];
export const TILE_CYCLE=87;
export function productCell(index:number){const q=Math.max(0,Math.min(63,Math.floor(index)));return {i:Math.floor(q/16),j:Math.floor(q/4)%4,k:q%4};}
export function scalarTile(completed:number){
 const result=TILE_C.map(row=>[...row]);
 for(let q=0;q<Math.max(0,Math.min(64,Math.floor(completed)));q++){const {i,j,k}=productCell(q);result[i][j]+=TILE_A[i][k]*TILE_B[k][j];}
 return result;
}
export const TILE_D=scalarTile(64);
export function tensorCoreFrame(g:number){
 const tick=Math.max(0,Math.min(TILE_CYCLE,Math.floor(g)));
 const phase=tick<4?0:tick<68?1:tick<72?2:3;
 const scalarUpdates=phase===0?0:Math.min(64,tick-3),tileDone=tick>=70;
 return {phase,scalarUpdates,tileDone,active:phase===1?productCell(scalarUpdates-1):null,tileActive:phase===2&&!tileDone};
}
