// Plans and pure helpers (no DOM) of the BLAS and NVLink & NVSwitch tab animations (components/BlasAnim.tsx,
// components/NvlinkAnim.tsx), unit-tested in frontend/tests/blasNvlinkAnim.test.ts.

/** One animation: ticks per step (the step list beside the picture has one entry per step), base tick (ms at 1x)
 * and the short SVG labels it reads from the page text (`<prefix>.anim.<case>.<label>`). */
export interface StepPlan {
  steps: number[];
  tick: number;
  labels: readonly string[];
}

export const BLAS_PLAN = {
  levels: { steps: [12, 12, 16, 10], tick: 420, labels: ["flops", "moved", "ratio", "axis", "reuse"] },
  gemm: { steps: [8, 12, 16, 8], tick: 380, labels: ["cache", "regs", "core", "ramNaive", "ramTiled"] },
  dispatch: { steps: [8, 8, 8, 8], tick: 380, labels: ["cpu", "cuda", "rocm", "numpy", "built"] },
  threads: { steps: [12, 12, 12], tick: 380, labels: ["core", "time", "axis"] },
  order: { steps: [9, 9, 9], tick: 420, labels: ["inputs", "seq", "pair", "lanes", "ref"] },
  atomics: { steps: [12, 12], tick: 420, labels: ["block", "acc", "run"] },
  overhead: { steps: [8, 12, 12], tick: 320, labels: ["x", "y", "cpu", "gpu", "ideal"] },
  timing: { steps: [10, 10, 12], tick: 360, labels: ["cpu", "gpu", "launch", "kernel", "measured", "axis", "warm", "median"] },
} as const satisfies Record<string, StepPlan>;
export type BlasCase = keyof typeof BLAS_PLAN;

export const NV_PLAN = {
  path: { steps: [10, 10, 14], tick: 380, labels: ["host", "sw", "gpu", "axis", "pcie", "nvl"] },
  parallel: { steps: [12, 12, 12], tick: 420, labels: ["gpu", "dp", "tp", "pp", "batch", "grad", "layer", "act"] },
  gens: { steps: [5, 5, 5, 5, 5, 8], tick: 380, labels: ["axis", "link"] },
  hops: { steps: [8, 10, 28], tick: 300, labels: ["mesh", "switch", "one", "two"] },
  colls: { steps: [10, 10, 10, 10], tick: 380, labels: ["before", "after", "gpu"] },
  tree: { steps: [15, 7], tick: 420, labels: ["ring", "tree", "steps"] },
  ddp: { steps: [8, 10, 14, 8], tick: 420, labels: ["gpu", "batch", "fwd", "bwd", "bucket", "step"] },
  p2p: { steps: [10, 10, 12], tick: 380, labels: ["host", "sw", "gpu", "ideal", "staged"] },
  overlap: { steps: [10, 10, 12], tick: 380, labels: ["axis", "compute", "comm", "pcie", "nvl", "fwd", "bwd"] },
} as const satisfies Record<string, StepPlan>;
export type NvCase = keyof typeof NV_PLAN;

/** Which animations each card of the tab shows (section id of the tab -> cases). */
export const BLAS_CARDS: Record<string, BlasCase[]> = {
  what: ["levels", "gemm"],
  libs: ["dispatch"],
  which: ["threads"],
  repro: ["order", "atomics"],
  bench: ["overhead"],
  method: ["timing"],
};
export const NV_CARDS: Record<string, NvCase[]> = {
  why: ["path", "parallel"],
  gens: ["gens"],
  topo: ["hops"],
  coll: ["colls", "tree"],
  code: ["ddp", "p2p"],
  practice: ["overlap"],
};

/** Plain explanations under each code block: code block id -> number of notes (`<prefix>.anim.code.<id>.<k>`). */
export const BLAS_CODE = { torch: 5, numpy: 2, threads: 3, order: 2, tol: 3, det: 4, bench: 2 } as const;
export const NV_CODE = { ddp: 6, p2p: 4, smi: 4, gloo: 4 } as const;

export const totalTicks = (p: StepPlan) => p.steps.reduce((a, b) => a + b, 0);

/** Step index, tick inside the step and progress f in (0, 1] of tick g (clamped to the last step's end). */
export function stepAt(p: StepPlan, g: number): { step: number; t: number; f: number } {
  let r = Math.min(Math.max(Math.floor(g), 0), totalTicks(p) - 1);
  for (let k = 0; k < p.steps.length; k++) {
    if (r < p.steps[k]) return { step: k, t: r, f: (r + 1) / p.steps[k] };
    r -= p.steps[k];
  }
  return { step: p.steps.length - 1, t: 0, f: 1 };
}

// --- BLAS -----------------------------------------------------------------------------------------------

/** FLOPs and values moved of the three BLAS levels for size n (as in the tab's table). */
export function levelCost(level: 1 | 2 | 3, n: number): { flops: number; moved: number } {
  if (level === 1) return { flops: 2 * n, moved: 3 * n };
  if (level === 2) return { flops: 2 * n * n, moved: n * n };
  return { flops: 2 * n ** 3, moved: 4 * n * n };
}

/** Values read from main memory for C = AB (n x n): unblocked every product reads one A and one B value
 * (2n^3); with b x b tiles kept in cache, each loaded value is reused b times (2n^3 / b). */
export const gemmLoads = (n: number, b = 1) => (2 * n ** 3) / b;

/** The eight float32 inputs of the summation-order animation. */
export const SUM_INPUTS = [0.1, 0.7, 0.3, 0.9, 0.2, 0.6, 0.4, 0.8];
export type SumOrder = "seq" | "pair" | "lanes";

/** One rounded float32 addition: node `out` = left + right (node ids: 0..7 inputs, 8.. partial sums). */
export interface Add {
  left: number;
  right: number;
  value: number;
}

/** The additions of a float32 sum in three orders: left to right, pairwise (a tree, as threads split it) and
 * two SIMD lanes (even and odd elements summed side by side, then combined). Each partial sum is rounded. */
export function sumOrder(order: SumOrder, xs: number[] = SUM_INPUTS): Add[] {
  const v = xs.map(Math.fround);
  const vals = [...v];
  const out: Add[] = [];
  const add = (a: number, b: number) => {
    const value = Math.fround(vals[a] + vals[b]);
    out.push({ left: a, right: b, value });
    vals.push(value);
    return vals.length - 1;
  };
  if (order === "seq") {
    let s = 0;
    for (let i = 1; i < v.length; i++) s = add(s, i);
  } else if (order === "pair") {
    let level = v.map((_, i) => i);
    while (level.length > 1) {
      const next: number[] = [];
      for (let i = 0; i < level.length; i += 2) next.push(add(level[i], level[i + 1]));
      level = next;
    }
  } else {
    let e = 0;
    let o = 1;
    for (let i = 2; i < v.length; i += 2) {
      e = add(e, i);
      o = add(o, i + 1);
    }
    add(e, o);
  }
  return out;
}

/** 4.0000005 -> "4.00000048": nine significant digits show the last float32 bits. */
export const fmt9 = (x: number) => x.toPrecision(9);

/** Partial sums of the four thread blocks of the atomics animation, and the arrival orders of two runs. */
export const ATOMIC_PARTS = [0.2, 0.6, 0.4, 0.8];
export const ATOMIC_RUNS: { atomic: number[][]; fixed: number[] } = { atomic: [[1, 3, 0, 2], [2, 0, 3, 1]], fixed: [0, 1, 2, 3] };

/** Running float32 sums of the partial sums added in `order`. */
export function accumulate(order: number[], parts: number[] = ATOMIC_PARTS): number[] {
  const out: number[] = [];
  let s = 0;
  for (const i of order) {
    s = Math.fround(s + Math.fround(parts[i]));
    out.push(s);
  }
  return out;
}

/** Illustrative fixed-cost model of a GEMM call: t = t0 + 2n^3 / P; returns GFLOP/s. */
export const OVERHEAD = { cpu: { t0: 5e-6, peak: 400 }, gpu: { t0: 2e-5, peak: 10000 } } as const;
export function modelGflops(n: number, t0: number, peak: number): number {
  const flops = 2 * n ** 3;
  return flops / (t0 + flops / (peak * 1e9)) / 1e9;
}
/** Smallest power-of-two n (16..8192) at which the GPU model beats the CPU model. */
export function crossover(): number {
  for (let n = 16; n <= 8192; n *= 2)
    if (modelGflops(n, OVERHEAD.gpu.t0, OVERHEAD.gpu.peak) > modelGflops(n, OVERHEAD.cpu.t0, OVERHEAD.cpu.peak)) return n;
  return -1;
}

/** Thread-pool animation (illustrative): 12 tiles of C on 6 cores; returns the finish time in tile units. With
 * more threads than cores each core time-slices, and every switch costs `switchCost` of a tile (cache refills). */
export function poolTime(tiles: number, threads: number, cores: number, switchCost = 0.25): number {
  const used = Math.min(threads, cores);
  const ideal = tiles / used;
  if (threads <= cores) return Math.ceil(tiles / threads);
  const perCore = threads / cores; // threads sharing a core
  return ideal + (perCore - 1) * ideal * switchCost;
}

// --- NVLink ---------------------------------------------------------------------------------------------

/** H100 SXM links, GB/s in one direction (half of the both-direction totals in the tab's table). */
export const H100_LINKS = { pcie5: 64, nvlink4: 450 } as const;

/** Ideal time (s) to move `bytes` over a link of `gbs` GB/s in one direction. */
export const moveSeconds = (bytes: number, gbs: number) => bytes / (gbs * 1e9);
export const GIB = 2 ** 30;

/** NVLink generations (links per GPU, GB/s per link one direction) and PCIe x16 (GB/s both directions), as in
 * the tab's table; per GPU both directions = links * 2 * perDir. */
export const NV_GENS = [
  { gen: "NVLink 1", gpu: "P100", links: 4, perDir: 20 },
  { gen: "NVLink 2", gpu: "V100", links: 6, perDir: 25 },
  { gen: "NVLink 3", gpu: "A100", links: 12, perDir: 25 },
  { gen: "NVLink 4", gpu: "H100", links: 18, perDir: 25 },
  { gen: "NVLink 5", gpu: "B200", links: 18, perDir: 50 },
] as const;
export const PCIE_GENS = [
  { gen: "PCIe Gen3", total: 32 },
  { gen: "PCIe Gen4", total: 64 },
  { gen: "PCIe Gen5", total: 128 },
  { gen: "PCIe Gen6", total: 256 },
] as const;
export const genTotal = (g: { links: number; perDir: number }) => g.links * 2 * g.perDir;

/** A shortest path a -> b (list of GPUs) over an undirected link list (breadth-first search); [] if none. */
export function shortestPath(edges: [number, number][], a: number, b: number): number[] {
  const prev = new Map<number, number>([[a, -1]]);
  const queue = [a];
  while (queue.length) {
    const u = queue.shift()!;
    if (u === b) break;
    for (const [x, y] of edges) {
      const v = x === u ? y : y === u ? x : -1;
      if (v >= 0 && !prev.has(v)) {
        prev.set(v, u);
        queue.push(v);
      }
    }
  }
  if (!prev.has(b)) return [];
  const path = [b];
  while (path[0] !== a) path.unshift(prev.get(path[0])!);
  return path;
}

/** All 28 pairs (a < b) of 8 GPUs, in order. */
export const PAIRS: [number, number][] = Array.from({ length: 8 }, (_, a) => Array.from({ length: 7 - a }, (_, j) => [a, a + 1 + j] as [number, number])).flat();

/** Binary tree of N nodes as a heap: parent of k is (k - 1) >> 1; depth of node k. */
export const treeDepth = (k: number) => Math.floor(Math.log2(k + 1));

/** Step time (ms) of the worked example: compute, then (or overlapped with backward) the all-reduce. */
export function stepMs(computeMs: number, commMs: number, overlap = false, bwdShare = 2 / 3): number {
  if (!overlap) return computeMs + commMs;
  const fwd = computeMs * (1 - bwdShare);
  return fwd + Math.max(computeMs * bwdShare, commMs);
}
