export type MemoryScale="linear"|"log";
export const MEMORY_LINEAR_MAX=256e9;
export const MEMORY_LOG_MIN=1;
export const MEMORY_LOG_MAX=1e12;
export function memoryPercent(bytes:number,scale:MemoryScale):number{
 if(bytes<=0)return 0;
 const fraction=scale==='linear'?bytes/MEMORY_LINEAR_MAX:(Math.log10(bytes)-Math.log10(MEMORY_LOG_MIN))/(Math.log10(MEMORY_LOG_MAX)-Math.log10(MEMORY_LOG_MIN));
 return Math.min(100,Math.max(0,100*fraction));
}
export const MEMORY_TICKS={linear:[0,64e9,128e9,192e9,256e9],log:[1,1e3,1e6,1e9,1e12]};
