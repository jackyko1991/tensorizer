import {test} from "node:test";
import assert from "node:assert/strict";
import {memoryPercent,MEMORY_TICKS,MEMORY_LINEAR_MAX} from "../src/pairMemory.ts";
test("linear memory positions reflect actual byte ratios, including small allocations",()=>{
 assert.equal(memoryPercent(8e9,'linear'),3.125);assert.equal(memoryPercent(24e9,'linear'),9.375);assert.equal(memoryPercent(80e9,'linear'),31.25);
 for(const bytes of [432,864,12e6,12e9,120e9])assert.ok(Math.abs(memoryPercent(2*bytes,'linear')-2*memoryPercent(bytes,'linear'))<1e-10);
 assert.equal(memoryPercent(MEMORY_LINEAR_MAX,'linear'),100);assert.equal(memoryPercent(0,'linear'),0);assert.equal(memoryPercent(1e15,'linear'),100);
});
test("log memory positions use equal distances for equal byte ratios",()=>{
 assert.deepEqual(MEMORY_TICKS.log.map(b=>memoryPercent(b,'log')),[0,25,50,75,100]);
 for(const bytes of [1,1e3,1e6,1e9])assert.ok(Math.abs(memoryPercent(bytes*10,'log')-memoryPercent(bytes,'log')-100/12)<1e-10);
 assert.ok(memoryPercent(432,'log')>0);assert.equal(memoryPercent(0,'log'),0);
});
