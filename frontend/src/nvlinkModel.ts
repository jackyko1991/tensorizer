// Pure helpers of the NVLink & NVSwitch tab (no DOM), unit-tested in frontend/tests/nvlinkModel.test.ts.

export type Collective = "allreduce" | "allgather" | "reducescatter" | "broadcast";

/**
 * Bytes each GPU sends, as a multiple of the tensor size S, for the bandwidth-optimal (ring) algorithm on N GPUs:
 * NCCL's "bus bandwidth" factors (nccl-tests doc/PERFORMANCE.md). S is the full tensor (all-reduce, broadcast),
 * the gathered output (all-gather) or the input before scattering (reduce-scatter).
 */
export function busFactor(op: Collective, n: number): number {
  if (n <= 1) return 0;
  if (op === "allreduce") return (2 * (n - 1)) / n;
  if (op === "broadcast") return 1;
  return (n - 1) / n;
}

/**
 * Ring all-reduce time (seconds) of `bytes` over `n` GPUs whose links move `bytesPerSec` per direction:
 * t = 2(N-1)/N * S / B + 2(N-1) * alpha (alpha = per-step latency, default 0: the bandwidth term only).
 */
export function ringAllReduceSeconds(bytes: number, n: number, bytesPerSec: number, alpha = 0): number {
  if (n <= 1) return 0;
  return busFactor("allreduce", n) * (bytes / bytesPerSec) + 2 * (n - 1) * alpha;
}

/** Latency term of a (binary) tree all-reduce: up the tree and down again, 2 * ceil(log2 N) steps. */
export function treeLatencySteps(n: number): number {
  return n <= 1 ? 0 : 2 * Math.ceil(Math.log2(n));
}

/** Latency term of the ring all-reduce: N-1 reduce-scatter steps plus N-1 all-gather steps. */
export function ringLatencySteps(n: number): number {
  return n <= 1 ? 0 : 2 * (n - 1);
}

/** One transfer of a ring step: GPU `from` sends chunk `chunk` to GPU (from + 1) mod N. */
export interface Send {
  from: number;
  to: number;
  chunk: number;
}

export interface RingState {
  /** step number: 0 = start, 1..N-1 reduce-scatter, N..2(N-1) all-gather */
  step: number;
  phase: "start" | "reduce-scatter" | "all-gather";
  /** held[gpu][chunk] = bit mask of the GPUs whose values are summed into that chunk */
  held: number[][];
  /** the transfers that produced this state (empty at the start) */
  sends: Send[];
}

/**
 * Ring all-reduce of a tensor split into N chunks on N GPUs, simulated step by step. In reduce-scatter step k
 * (k = 0..N-2) GPU i sends chunk (i - k) mod N to GPU i+1, which adds it to its own; afterwards GPU i holds the
 * full sum of chunk (i + 1) mod N. In all-gather step k GPU i forwards chunk (i + 1 - k) mod N, which the
 * receiver copies. Returns the states 0..2(N-1).
 */
export function ringAllReduce(n: number): RingState[] {
  const mod = (a: number) => ((a % n) + n) % n;
  let held = Array.from({ length: n }, (_, g) => Array.from({ length: n }, () => 1 << g));
  const states: RingState[] = [{ step: 0, phase: "start", held, sends: [] }];
  for (let s = 1; s <= 2 * (n - 1); s++) {
    const rs = s <= n - 1;
    const k = rs ? s - 1 : s - n;
    const sends = Array.from({ length: n }, (_, i) => ({ from: i, to: mod(i + 1), chunk: rs ? mod(i - k) : mod(i + 1 - k) }));
    const next = held.map((row) => row.slice());
    for (const t of sends) next[t.to][t.chunk] = rs ? next[t.to][t.chunk] | held[t.from][t.chunk] : held[t.from][t.chunk];
    held = next;
    states.push({ step: s, phase: rs ? "reduce-scatter" : "all-gather", held, sends });
  }
  return states;
}

/** Number of set bits (how many GPUs' values a chunk holds). */
export function popcount(mask: number): number {
  let c = 0;
  for (let m = mask; m; m &= m - 1) c++;
  return c;
}

/** 0.0039 -> "3.9 ms", 0.0547 -> "54.7 ms", 1.2 -> "1.20 s". */
export function fmtSeconds(t: number): string {
  if (t >= 1) return `${t.toFixed(2)} s`;
  const ms = t * 1e3;
  return ms >= 100 ? `${Math.round(ms)} ms` : ms >= 1 ? `${ms.toFixed(1)} ms` : `${(ms * 1e3).toFixed(0)} µs`;
}

/** DGX-1 hybrid cube-mesh: two fully connected quads (GPUs 0-3, 4-7) plus a link from GPU i to GPU i + 4. */
export const MESH_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3],
  [4, 5], [5, 6], [6, 7], [7, 4], [4, 6], [5, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/** Fewest links between GPUs a and b of an undirected link list (breadth-first search); -1 if unreachable. */
export function hops(edges: [number, number][], a: number, b: number): number {
  const dist = new Map<number, number>([[a, 0]]);
  const queue = [a];
  while (queue.length) {
    const u = queue.shift()!;
    if (u === b) return dist.get(u)!;
    for (const [x, y] of edges) {
      const v = x === u ? y : y === u ? x : -1;
      if (v >= 0 && !dist.has(v)) {
        dist.set(v, dist.get(u)! + 1);
        queue.push(v);
      }
    }
  }
  return -1;
}
