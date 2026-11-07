import { useState, type ReactNode } from "react";
import { AnimControls, COL, Txt, blueScale, useAnimClock } from "../components/animKit";
import { CodeNotes } from "../components/BlasAnim";
import CodeBlock from "../components/CodeBlock";
import NvlinkAnim from "../components/NvlinkAnim";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { MESH_EDGES, busFactor, fmtSeconds, popcount, ringAllReduce, ringAllReduceSeconds, type Collective } from "../nvlinkModel";

/** Every section of the tab, in order: numbered 1..N in the table of contents and on the cards. */
const SECTIONS = ["why", "gens", "topo", "coll", "code", "practice"] as const;
const num = (key: (typeof SECTIONS)[number]) => SECTIONS.indexOf(key) + 1;

const fill = (s: string, v: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (m, k) => String(v[k] ?? m));

/** NVIDIA (and NCCL / PyTorch) documents every number on the tab is taken from. */
const SRC = {
  p100: { name: "Tesla P100 technical overview", url: "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/nvidia-teslap100-techoverview.pdf" },
  volta: { name: "Volta architecture whitepaper", url: "https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf" },
  ampere: { name: "NVIDIA Ampere Architecture In-Depth", url: "https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/" },
  a100: { name: "A100 product page", url: "https://www.nvidia.com/en-us/data-center/a100/" },
  hopper: { name: "NVIDIA Hopper Architecture In-Depth", url: "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/" },
  h100: { name: "H100 product page", url: "https://www.nvidia.com/en-us/data-center/h100/" },
  nvlink: { name: "NVLink and NVLink Switch", url: "https://www.nvidia.com/en-us/data-center/nvlink/" },
  bwUltra: { name: "Inside NVIDIA Blackwell Ultra", url: "https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/" },
  b200: { name: "HGX reference architecture: components", url: "https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html" },
  nvswitch: { name: "NVSwitch accelerates DGX-2", url: "https://developer.nvidia.com/blog/nvswitch-accelerates-nvidia-dgx2/" },
  gtc18: { name: "GTC Japan 2018: DGX-2 and NVSwitch", url: "https://www.nvidia.com/content/apac/gtc/ja/pdf/2018/1050.pdf" },
  nvl72: { name: "GB200 NVL72", url: "https://www.nvidia.com/en-us/data-center/gb200-nvl72/" },
  rtx2080: { name: "GeForce RTX 2080 Ti", url: "https://www.nvidia.com/content/nvidiaGDC/zz/en_ZZ/geforce/graphics-cards/rtx-2080-ti.html" },
  rtx3090: { name: "GeForce RTX 3090 / 3090 Ti", url: "https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/" },
  rtx4090: { name: "GeForce RTX 4090", url: "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/" },
  ncclPerf: { name: "nccl-tests PERFORMANCE.md", url: "https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md" },
  ncclEnv: { name: "NCCL environment variables", url: "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html" },
  ncclTree: { name: "Massively scale your deep learning training with NCCL 2.4", url: "https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/" },
  dist: { name: "torch.distributed", url: "https://docs.pytorch.org/docs/stable/distributed.html" },
  ddp: { name: "DistributedDataParallel", url: "https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html" },
  torchrun: { name: "torchrun", url: "https://docs.pytorch.org/docs/stable/elastic/run.html" },
  peer: { name: "torch.cuda.can_device_access_peer", url: "https://docs.pytorch.org/docs/stable/generated/torch.cuda.can_device_access_peer.html" },
  smi: { name: "nvidia-smi", url: "https://docs.nvidia.com/deploy/nvidia-smi/index.html" },
  cudaMpi: { name: "An Introduction to CUDA-Aware MPI", url: "https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/" },
  nvshmem: { name: "NVSHMEM", url: "https://docs.nvidia.com/nvshmem/index.html" },
} as const;
type SrcId = keyof typeof SRC;

/** Card heading with the numbered blue circle, as on the Concept page. */
function Heading({ n, text }: { n: number; text: string }) {
  return (
    <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
        {n}
      </span>
      <RichText text={text} />
    </h2>
  );
}

function Card({ id, children }: { id: string; children: ReactNode }) {
  return (
    <section id={`nvlink-${id}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6" data-testid={`nvlink-${id}`}>
      {children}
    </section>
  );
}

/** An external documentation link, opening in a new tab. */
function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-700 hover:underline">
      {children}
      <span aria-hidden> ↗</span>
    </a>
  );
}

/** "Sources:" followed by links to the documents. */
function Sources({ ids, label = "nvlink.sources" }: { ids: SrcId[]; label?: PageKey }) {
  const T = usePageText();
  return (
    <p className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs" data-testid="nvlink-sources">
      <span className="text-gray-500">{T(label)}</span>
      {ids.map((id) => (
        <Ext key={id} href={SRC[id].url}>
          {SRC[id].name}
        </Ext>
      ))}
    </p>
  );
}

const P = "mt-3 text-sm leading-relaxed text-gray-700";

// --- 1. why: one H100's three bandwidths -----------------------------------------------------------------

const BARS: { key: PageKey; gbs: number; color: string }[] = [
  { key: "nvlink.why.bar.pcie", gbs: 128, color: "#9ca3af" },
  { key: "nvlink.why.bar.nvlink", gbs: 900, color: COL.blue },
  { key: "nvlink.why.bar.hbm", gbs: 3350, color: COL.violet },
];
const BAR_MAX = 3500; // fixed axis limit, GB/s

function BandwidthBars() {
  const T = usePageText();
  const W = 360;
  const m = { l: 92, r: 44, t: 6, b: 30 };
  const row = 24;
  const H = m.t + BARS.length * row + m.b;
  const X = (v: number) => m.l + (v / BAR_MAX) * (W - m.l - m.r);
  return (
    <figure className="mt-4 rounded-lg border border-gray-200 bg-gray-50/60 p-3 text-xs" data-testid="nvlink-bars">
      <div className="mx-auto max-w-lg">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={T("nvlink.why.bar.caption")}>
          {[0, 1000, 2000, 3000].map((v) => (
            <g key={v}>
              <line x1={X(v)} x2={X(v)} y1={m.t} y2={H - m.b} stroke="#e5e7eb" />
              <text x={X(v)} y={H - m.b + 11} textAnchor="middle" fontSize="9" fill="#6b7280">
                {v.toLocaleString("en")}
              </text>
            </g>
          ))}
          {BARS.map((b, i) => {
            const y = m.t + i * row + 4;
            return (
              <g key={b.key}>
                <title>{`${T(b.key)}: ${b.gbs.toLocaleString("en")} GB/s`}</title>
                <text x={m.l - 6} y={y + 11} textAnchor="end" fontSize="9" fill="#111827">
                  {T(b.key)}
                </text>
                <rect x={X(0)} y={y} width={X(b.gbs) - X(0)} height={16} rx={2} fill={b.color} />
                <text x={X(b.gbs) + 4} y={y + 11} fontSize="9" fill="#374151">
                  {b.gbs.toLocaleString("en")}
                </text>
              </g>
            );
          })}
          <text x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="#4b5563">
            {T("nvlink.why.bar.x")}
          </text>
        </svg>
      </div>
      <figcaption className="mt-2 text-[11px] leading-relaxed text-gray-500">
        <RichText text={T("nvlink.why.bar.caption")} />
      </figcaption>
    </figure>
  );
}

// --- 2. generations --------------------------------------------------------------------------------------

/** NVLink per GPU: links, GB/s per link in one direction, GB/s per GPU with both directions added, memory GB/s. */
const GENS: { id: string; gen: string; gpu: string; year: number; links: number; perDir: number; total: number; mem: string; src: SrcId[] }[] = [
  { id: "nv1", gen: "NVLink 1", gpu: "Tesla P100 (Pascal)", year: 2016, links: 4, perDir: 20, total: 160, mem: "732", src: ["p100", "volta"] },
  { id: "nv2", gen: "NVLink 2", gpu: "Tesla V100 (Volta)", year: 2017, links: 6, perDir: 25, total: 300, mem: "900", src: ["volta"] },
  { id: "nv3", gen: "NVLink 3", gpu: "A100 (Ampere)", year: 2020, links: 12, perDir: 25, total: 600, mem: "1,555 (40 GB) / 2,039 (80 GB SXM)", src: ["ampere", "a100"] },
  { id: "nv4", gen: "NVLink 4", gpu: "H100 (Hopper)", year: 2022, links: 18, perDir: 25, total: 900, mem: "3,350 (SXM)", src: ["hopper", "h100"] },
  { id: "nv5", gen: "NVLink 5", gpu: "B200 (Blackwell)", year: 2024, links: 18, perDir: 50, total: 1800, mem: "8,000", src: ["nvlink", "bwUltra", "b200"] },
];
/** PCIe x16: GB/s in one direction and with both directions added. */
const PCIE: { id: string; gen: string; perDir: string; total: number; src: SrcId[] }[] = [
  { id: "g3", gen: "PCIe Gen3 x16", perDir: "15.75", total: 32, src: ["ampere"] },
  { id: "g4", gen: "PCIe Gen4 x16", perDir: "31.5", total: 64, src: ["ampere", "a100"] },
  { id: "g5", gen: "PCIe Gen5 x16", perDir: "64", total: 128, src: ["hopper"] },
  { id: "g6", gen: "PCIe Gen6 x16", perDir: "128", total: 256, src: ["bwUltra"] },
];

const GEN_HEAD = ["gen", "gpu", "links", "perDir", "total", "mem", "src"] as const;

function SrcLinks({ ids }: { ids: SrcId[] }) {
  return (
    <span className="flex flex-col gap-0.5">
      {ids.map((id) => (
        <Ext key={id} href={SRC[id].url}>
          {SRC[id].name}
        </Ext>
      ))}
    </span>
  );
}

function GenTable() {
  const T = usePageText();
  const n = (v: number) => v.toLocaleString("en");
  return (
    <details className="group mt-4 rounded-lg border border-gray-200 bg-white" data-testid="nvlink-gen-table">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2 text-sm font-medium text-gray-800 [&::-webkit-details-marker]:hidden">
        <span className="text-xs text-gray-500 group-open:hidden" aria-hidden>▸</span>
        <span className="hidden text-xs text-gray-500 group-open:inline" aria-hidden>▾</span>
        {T("nvlink.gens.table.title")}
      </summary>
      {/* phones: one compact block per row instead of a sideways-scrolling table */}
      <ul className="divide-y divide-gray-100 border-t border-gray-100 text-xs text-gray-700 md:hidden">
        {GENS.map((g) => (
          <li key={g.id} className="space-y-1 px-4 py-2.5" data-testid={`nvlink-gen-card-${g.id}`}>
            <div className="font-medium text-gray-900">
              {g.gen} · {g.gpu}, {g.year}
            </div>
            <div>
              {fill(T("nvlink.gens.card"), { links: g.links, perDir: g.perDir, total: n(g.total), mem: g.mem })}
            </div>
            <div className="text-gray-500">
              <SrcLinks ids={g.src} />
            </div>
          </li>
        ))}
        {PCIE.map((p) => (
          <li key={p.id} className="space-y-1 bg-gray-50/60 px-4 py-2.5">
            <div className="font-medium text-gray-900">{p.gen}</div>
            <div>{fill(T("nvlink.gens.cardPcie"), { perDir: p.perDir, total: p.total })}</div>
            <div className="text-gray-500">
              <SrcLinks ids={p.src} />
            </div>
          </li>
        ))}
      </ul>
      <div className="hidden overflow-x-auto border-t border-gray-100 md:block">
        <table className="w-full min-w-[48rem] text-left text-xs">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              {GEN_HEAD.map((h) => (
                <th key={h} className="px-3 py-1.5 align-bottom font-semibold">
                  {T(`nvlink.gens.table.${h}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 text-gray-700">
            {GENS.map((g) => (
              <tr key={g.id} data-testid={`nvlink-gen-${g.id}`}>
                <td className="px-3 py-1.5 align-top font-medium whitespace-nowrap text-gray-900">{g.gen}</td>
                <td className="px-3 py-1.5 align-top">
                  {g.gpu}, {g.year}
                </td>
                <td className="px-3 py-1.5 align-top tabular-nums">{g.links}</td>
                <td className="px-3 py-1.5 align-top tabular-nums">{g.perDir}</td>
                <td className="px-3 py-1.5 align-top font-medium tabular-nums">{n(g.total)}</td>
                <td className="px-3 py-1.5 align-top tabular-nums">{g.mem}</td>
                <td className="px-3 py-1.5 align-top">
                  <SrcLinks ids={g.src} />
                </td>
              </tr>
            ))}
            {PCIE.map((p) => (
              <tr key={p.id} className="bg-gray-50/60">
                <td className="px-3 py-1.5 align-top font-medium whitespace-nowrap text-gray-900">{p.gen}</td>
                <td className="px-3 py-1.5 align-top text-gray-500">{T("nvlink.gens.table.pcieGpu")}</td>
                <td className="px-3 py-1.5 align-top text-gray-500">x16</td>
                <td className="px-3 py-1.5 align-top tabular-nums">{p.perDir}</td>
                <td className="px-3 py-1.5 align-top font-medium tabular-nums">{p.total}</td>
                <td className="px-3 py-1.5 align-top text-gray-500">–</td>
                <td className="px-3 py-1.5 align-top">
                  <SrcLinks ids={p.src} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-gray-100 px-4 py-2 text-xs leading-relaxed text-gray-600">
        <RichText text={T("nvlink.gens.table.note")} />
      </p>
    </details>
  );
}

// --- 3. topology diagram ---------------------------------------------------------------------------------

function TopologyDiagram() {
  const T = usePageText();
  const W = 380;
  const H = 190;
  // left panel: front face (0-3) and back face (4-7) of a cube
  const front: [number, number][] = [[28, 70], [108, 70], [108, 150], [28, 150]];
  const pos = [...front, ...front.map(([x, y]) => [x + 50, y - 40] as [number, number])];
  const hot = new Set(["0-4", "4-5"]); // GPU 0 -> GPU 5 needs two hops in the mesh
  const r = 11;
  // right panel: 8 GPUs around an NVSwitch bar
  const sx = (i: number) => 214 + (i % 4) * 46;
  const sy = (i: number) => (i < 4 ? 58 : 158);
  return (
    <figure className="mt-4 rounded-lg border border-gray-200 bg-gray-50/60 p-3 text-xs" data-testid="nvlink-topo-svg">
      <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto w-full max-w-xl" role="img" aria-label={T("nvlink.topo.svg.caption")}>
        <Txt x={92} y={14} size={10} anchor="middle" weight={600}>
          {T("nvlink.topo.svg.mesh")}
        </Txt>
        {MESH_EDGES.map(([a, b]) => {
          const isHot = hot.has(`${a}-${b}`);
          return (
            <line
              key={`${a}-${b}`}
              x1={pos[a][0]}
              y1={pos[a][1]}
              x2={pos[b][0]}
              y2={pos[b][1]}
              stroke={isHot ? COL.cur : "#93c5fd"}
              strokeWidth={isHot ? 3 : 1.6}
            />
          );
        })}
        {pos.map(([x, y], i) => (
          <g key={i}>
            <rect x={x - r} y={y - r} width={2 * r} height={2 * r} rx={3} fill={i === 0 || i === 5 ? COL.curLight : "#fff"} stroke="#334155" strokeWidth={1.2} />
            <Txt x={x} y={y + 3} size={8} anchor="middle">
              {i}
            </Txt>
          </g>
        ))}
        <Txt x={92} y={182} size={8.5} anchor="middle" color={COL.muted}>
          {T("nvlink.topo.svg.meshHops")}
        </Txt>

        <line x1={196} x2={196} y1={20} y2={176} stroke="#e5e7eb" />

        <Txt x={283} y={14} size={10} anchor="middle" weight={600}>
          {T("nvlink.topo.svg.switch")}
        </Txt>
        {Array.from({ length: 8 }, (_, i) => (
          <line
            key={`s${i}`}
            x1={sx(i)}
            x2={sx(i)}
            y1={sy(i)}
            y2={108}
            stroke={i === 0 || i === 5 ? COL.cur : "#93c5fd"}
            strokeWidth={i === 0 || i === 5 ? 3 : 1.6}
          />
        ))}
        <rect x={204} y={96} width={158} height={24} rx={4} fill="#334155" />
        <Txt x={283} y={112} size={9} anchor="middle" color="#fff" weight={600}>
          NVSwitch
        </Txt>
        {Array.from({ length: 8 }, (_, i) => (
          <g key={`g${i}`}>
            <rect x={sx(i) - r} y={sy(i) - r} width={2 * r} height={2 * r} rx={3} fill={i === 0 || i === 5 ? COL.curLight : "#fff"} stroke="#334155" strokeWidth={1.2} />
            <Txt x={sx(i)} y={sy(i) + 3} size={8} anchor="middle">
              {i}
            </Txt>
          </g>
        ))}
        <Txt x={283} y={182} size={8.5} anchor="middle" color={COL.muted}>
          {T("nvlink.topo.svg.switchHops")}
        </Txt>
      </svg>
      <figcaption className="mt-2 text-[11px] leading-relaxed text-gray-500">
        <RichText text={T("nvlink.topo.svg.caption")} />
      </figcaption>
    </figure>
  );
}

/** Systems of section 3: a badge (hover: the numbers) and a paragraph. */
const SYSTEMS: { id: string; name: string; src: SrcId[] }[] = [
  { id: "dgx1", name: "DGX-1", src: ["volta", "p100"] },
  { id: "dgx2", name: "DGX-2 + NVSwitch", src: ["nvswitch", "gtc18"] },
  { id: "nvl72", name: "NVLink Switch System / GB200 NVL72", src: ["hopper", "nvl72", "nvlink"] },
  { id: "geforce", name: "GeForce", src: ["rtx2080", "rtx3090", "rtx4090"] },
];

// --- 4. collectives: ring all-reduce animation ------------------------------------------------------------

const RING_N = 4;
const RING = ringAllReduce(RING_N);
const LAST = RING.length - 1;

function RingAnim() {
  const T = usePageText();
  const clock = useAnimClock(LAST + 1, 1500); // one extra tick holding the result
  const hold = clock.g > LAST; // the extra tick: result, no arrows
  const st = RING[Math.min(clock.g, LAST)];
  const sends = hold ? [] : st.sends;
  const W = 380;
  const colW = 76;
  const gap = 18;
  const x0 = 12;
  const top = 36;
  const ch = 22;
  const gx = (g: number) => x0 + g * (colW + gap);
  const cy = (c: number) => top + 16 + c * (ch + 4);
  const H = cy(RING_N) + 26;
  const label = (mask: number) =>
    popcount(mask) === RING_N
      ? "Σ"
      : Array.from({ length: RING_N }, (_, i) => i)
          .filter((i) => mask & (1 << i))
          .map((i) => "abcd"[i])
          .join("+");
  const phase =
    st.phase === "start"
      ? T("nvlink.anim.start")
      : fill(T(st.phase === "reduce-scatter" ? "nvlink.anim.rs" : "nvlink.anim.ag"), {
          k: st.phase === "reduce-scatter" ? st.step : st.step - (RING_N - 1),
          n: RING_N - 1,
        });
  return (
    <figure className="mt-4 rounded-lg border border-gray-200 bg-gray-50/60 p-3 text-xs" data-testid="nvlink-ring-anim" ref={clock.root}>
      <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto w-full max-w-xl" role="img" aria-label={T("nvlink.anim.caption")}>
        <defs>
          <marker id="nv-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={COL.cur} />
          </marker>
        </defs>
        <Txt x={W / 2} y={12} size={10.5} anchor="middle" weight={600}>
          {hold ? T("nvlink.anim.done") : phase}
        </Txt>
        {Array.from({ length: RING_N }, (_, g) => (
          <g key={g}>
            <rect x={gx(g)} y={top - 4} width={colW} height={RING_N * (ch + 4) + 20} rx={5} fill="#fff" stroke="#cbd5e1" />
            <Txt x={gx(g) + colW / 2} y={top + 8} size={9.5} anchor="middle" weight={600}>
              {fill(T("nvlink.anim.gpu"), { i: g, v: "abcd"[g] })}
            </Txt>
            {st.held[g].map((mask, c) => {
              const k = popcount(mask);
              const sent = sends.some((s) => s.to === g && s.chunk === c);
              return (
                <g key={c}>
                  <rect
                    x={gx(g) + 6}
                    y={cy(c)}
                    width={colW - 12}
                    height={ch}
                    rx={2}
                    fill={blueScale((k - 1) / (RING_N - 1) * 0.8)}
                    stroke={sent ? COL.cur : COL.idleLine}
                    strokeWidth={sent ? 2 : 0.75}
                  />
                  <Txt x={gx(g) + 10} y={cy(c) + 14.5} size={8} color={k > 2 ? "#fff" : COL.muted}>
                    {`${c}`}
                  </Txt>
                  <Txt x={gx(g) + colW - 9} y={cy(c) + 15} size={10} anchor="end" color={k > 2 ? "#fff" : COL.text} kind="mono">
                    {label(mask)}
                  </Txt>
                </g>
              );
            })}
          </g>
        ))}
        {sends.map((s) => {
          const y = cy(s.chunk) + ch / 2;
          if (s.to > s.from)
            return (
              <line key={s.from} x1={gx(s.from) + colW - 4} x2={gx(s.to) + 4} y1={y} y2={y} stroke={COL.cur} strokeWidth={1.8} markerEnd="url(#nv-arrow)" />
            );
          // the ring closes: last GPU back to GPU 0, under the boxes
          const yb = H - 8;
          return (
            <path
              key={s.from}
              d={`M${gx(s.from) + colW / 2},${top + RING_N * (ch + 4) + 16} L${gx(s.from) + colW / 2},${yb} L${gx(0) + colW / 2},${yb} L${gx(0) + colW / 2},${top + RING_N * (ch + 4) + 18}`}
              fill="none"
              stroke={COL.cur}
              strokeWidth={1.8}
              markerEnd="url(#nv-arrow)"
            />
          );
        })}
      </svg>
      <AnimControls clock={clock} />
      <figcaption className="mt-2 text-[11px] leading-relaxed text-gray-500">
        <RichText text={T("nvlink.anim.caption")} />
      </figcaption>
    </figure>
  );
}

const COLLECTIVES: { id: Collective; fn: string; before: string; after: string; factor: string }[] = [
  { id: "allreduce", fn: "dist.all_reduce", before: "[S]", after: "[S]", factor: String.raw`\tfrac{2(N-1)}{N}` },
  { id: "allgather", fn: "dist.all_gather_into_tensor", before: "[S/N]", after: "[S]", factor: String.raw`\tfrac{N-1}{N}` },
  { id: "reducescatter", fn: "dist.reduce_scatter_tensor", before: "[S]", after: "[S/N]", factor: String.raw`\tfrac{N-1}{N}` },
  { id: "broadcast", fn: "dist.broadcast", before: "[S]", after: "[S]", factor: "1" },
];

// --- 5. programming --------------------------------------------------------------------------------------

const CODE_DDP = `# train.py - one process per GPU; start with:  torchrun --nproc_per_node=8 train.py
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")        # rank / world size come from torchrun's environment
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)

# a collective on a plain tensor: afterwards every GPU holds the same sum
g = torch.full((4,), float(rank), device="cuda")
dist.all_reduce(g, op=dist.ReduceOp.SUM)       # with 8 GPUs: tensor([28., 28., 28., 28.])

# data parallel training: DDP all-reduces the gradients during backward()
model = DDP(torch.nn.Linear(1024, 1024).cuda(), device_ids=[rank])
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
x = torch.randn(64, 1024, device="cuda")
loss = model(x).square().mean()
loss.backward()                                 # gradients summed over the GPUs (NCCL, NVLink if present)
opt.step()
dist.destroy_process_group()`;

const CODE_P2P = `import torch
print(torch.cuda.device_count())
print(torch.cuda.can_device_access_peer(0, 1))  # True: GPU 0 can read / write GPU 1's memory directly

x = torch.randn(256, 1024, 1024, device="cuda:0")   # 1 GiB of float32
y = x.to("cuda:1")                              # device-to-device copy: NVLink or PCIe peer-to-peer if
torch.cuda.synchronize()                        # available, otherwise staged through host memory`;

const CODE_SMI = `nvidia-smi topo -m              # matrix of GPU-GPU / GPU-NIC links: NV# = # NVLinks, PIX/PXB/PHB/NODE/SYS = PCIe paths
nvidia-smi nvlink --status      # per-link state and speed of each GPU's NVLinks

NCCL_DEBUG=INFO torchrun --nproc_per_node=8 train.py   # NCCL logs the topology, rings / trees and transports it chose
NCCL_P2P_DISABLE=1 ...          # experiment: forbid direct GPU-GPU paths and compare the speed`;

const CODE_GLOO = `# runs anywhere, even without a GPU: a "distributed" job of one process on the gloo (CPU) backend
import torch, torch.distributed as dist

dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
t = torch.arange(4.0)
dist.all_reduce(t)                              # sum over 1 process: unchanged
print(t, dist.get_world_size())                 # tensor([0., 1., 2., 3.]) 1
dist.destroy_process_group()`;

// --- 6. practice: the cost model --------------------------------------------------------------------------

/** Links of the calculator, GB/s per GPU in one direction (half of the both-direction totals above). */
const LINKS: { id: string; name: string; perDir: number }[] = [
  { id: "pcie4", name: "PCIe Gen4 x16", perDir: 32 },
  { id: "pcie5", name: "PCIe Gen5 x16", perDir: 64 },
  { id: "nvl3", name: "NVLink 3 (A100)", perDir: 300 },
  { id: "nvl4", name: "NVLink 4 (H100)", perDir: 450 },
  { id: "nvl5", name: "NVLink 5 (B200)", perDir: 900 },
];
const GB = 1e9;
const sec = (gb: number, n: number, perDir: number) => ringAllReduceSeconds(gb * GB, n, perDir * GB);

function CostCalculator() {
  const T = usePageText();
  const [gb, setGb] = useState(1);
  const [n, setN] = useState(8);
  const times = LINKS.map((l) => sec(gb, n, l.perDir));
  const tMax = Math.max(...times);
  return (
    <div className="mt-4 rounded-lg border border-gray-200 bg-white p-4" data-testid="nvlink-calc">
      <h3 className="text-sm font-semibold text-gray-900">{T("nvlink.calc.title")}</h3>
      <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-gray-700">
        <label className="flex items-center gap-1.5">
          {T("nvlink.calc.size")}
          <select className="rounded-md border border-gray-300 bg-white px-1.5 py-1" value={gb} onChange={(e) => setGb(Number(e.target.value))} data-testid="nvlink-calc-size">
            {[0.1, 0.5, 1, 2, 10, 140].map((v) => (
              <option key={v} value={v}>
                {v} GB
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          {T("nvlink.calc.gpus")}
          <select className="rounded-md border border-gray-300 bg-white px-1.5 py-1" value={n} onChange={(e) => setN(Number(e.target.value))} data-testid="nvlink-calc-n">
            {[2, 4, 8, 16, 72].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <span className="text-gray-500">
          <Tex tex={String.raw`\tfrac{2(N-1)}{N}\,S = ${(busFactor("allreduce", n) * gb).toFixed(3).replace(/\.?0+$/, "")}\ \text{GB}`} />
        </span>
      </div>
      <table className="mt-3 w-full text-left text-xs">
        <thead className="text-gray-600">
          <tr>
            <th className="py-1 pr-2 font-semibold">{T("nvlink.calc.link")}</th>
            <th className="py-1 pr-2 text-right font-semibold">{T("nvlink.calc.bw")}</th>
            <th className="py-1 pr-2 text-right font-semibold">{T("nvlink.calc.time")}</th>
            <th className="hidden w-2/5 py-1 sm:table-cell" />
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 text-gray-700">
          {LINKS.map((l, i) => (
            <tr key={l.id} data-testid={`nvlink-calc-${l.id}`}>
              <td className="py-1 pr-2 whitespace-nowrap">{l.name}</td>
              <td className="py-1 pr-2 text-right tabular-nums">{l.perDir}</td>
              <td className="py-1 pr-2 text-right font-medium tabular-nums">{fmtSeconds(times[i])}</td>
              <td className="hidden py-1 sm:table-cell">
                <div className="h-2.5 rounded-sm" style={{ width: `${Math.max(1, (100 * times[i]) / tMax)}%`, background: l.id.startsWith("pcie") ? "#9ca3af" : COL.blue }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] leading-relaxed text-gray-500">
        <RichText text={T("nvlink.calc.note")} />
      </p>
    </div>
  );
}

// --- the tab --------------------------------------------------------------------------------------------

export default function NvlinkTab() {
  const T = usePageText();
  const h3 = "mt-5 text-sm font-semibold text-gray-900";
  const worked = {
    pcie: fmtSeconds(sec(1, 8, 32)),
    nvl: fmtSeconds(sec(1, 8, 450)),
    ratio: Math.round(sec(1, 8, 32) / sec(1, 8, 450)),
  };

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
      <PageToc
        scrollRoot="skills-scroll"
        items={SECTIONS.map((key, i) => ({ id: `nvlink-${key}`, mark: i + 1, label: T(`nvlink.${key}.title` as PageKey) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10">
        <header>
          <p className="text-sm font-medium text-blue-700">{T("skills.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("nvlink.title")}</h1>
          <p className="mt-3 text-gray-600">
            <RichText text={T("nvlink.intro")} />
          </p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900" data-testid="nvlink-note">
            <RichText text={T("nvlink.howto")} />
          </p>
        </header>

        <Card id="why">
          <Heading n={num("why")} text={T("nvlink.why.title")} />
          <p className={P}>
            <RichText text={T("nvlink.why.text")} />
          </p>
          <BandwidthBars />
          <NvlinkAnim id="path" />
          <h3 className={h3}>{T("nvlink.why.timeTitle")}</h3>
          <ul className="mt-2 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-3">
            {(["dp", "tp", "pp"] as const).map((k) => (
              <li key={k} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`nvlink.why.${k}`)} />
              </li>
            ))}
          </ul>
          <NvlinkAnim id="parallel" />
          <Sources ids={["hopper", "h100"]} />
        </Card>

        <Card id="gens">
          <Heading n={num("gens")} text={T("nvlink.gens.title")} />
          <p className={P}>
            <RichText text={T("nvlink.gens.text")} />
          </p>
          <NvlinkAnim id="gens" />
          <GenTable />
        </Card>

        <Card id="topo">
          <Heading n={num("topo")} text={T("nvlink.topo.title")} />
          <p className={P}>
            <RichText text={T("nvlink.topo.text")} />
          </p>
          <TopologyDiagram />
          <NvlinkAnim id="hops" />
          <ul className="mt-4 space-y-3 text-sm leading-relaxed text-gray-700">
            {SYSTEMS.map((s) => (
              <li key={s.id} className="rounded-lg bg-gray-50 p-3" data-testid={`nvlink-sys-${s.id}`}>
                <span
                  className="mr-2 inline-flex cursor-help rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700"
                  title={T(`nvlink.topo.${s.id}.tip` as PageKey)}
                  data-tip-heading={s.name}
                  data-testid="nvlink-chip"
                >
                  {s.name}
                </span>
                <RichText text={T(`nvlink.topo.${s.id}.text` as PageKey)} />
                <Sources ids={s.src} />
              </li>
            ))}
          </ul>
        </Card>

        <Card id="coll">
          <Heading n={num("coll")} text={T("nvlink.coll.title")} />
          <p className={P}>
            <RichText text={T("nvlink.coll.text")} />
          </p>
          <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full min-w-[36rem] text-left text-xs" data-testid="nvlink-coll-table">
              <thead className="bg-gray-50 text-gray-600">
                <tr>
                  {(["op", "what", "shape", "factor"] as const).map((h) => (
                    <th key={h} className="px-3 py-1.5 font-semibold">
                      {T(`nvlink.coll.table.${h}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 text-gray-700">
                {COLLECTIVES.map((c) => (
                  <tr key={c.id}>
                    <td className="px-3 py-1.5 align-top">
                      <div className="font-medium text-gray-900">{T(`nvlink.coll.${c.id}.name` as PageKey)}</div>
                      <div className="font-mono text-[11px] text-gray-500">{c.fn}</div>
                    </td>
                    <td className="px-3 py-1.5 align-top">
                      <RichText text={T(`nvlink.coll.${c.id}.what` as PageKey)} />
                    </td>
                    <td className="px-3 py-1.5 align-top font-mono whitespace-nowrap">
                      {c.before} → {c.after}
                    </td>
                    <td className="px-3 py-1.5 align-top">
                      <Tex tex={c.factor} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <NvlinkAnim id="colls" />
          <RingAnim />
          <div className="mt-4 rounded-lg border border-gray-200 bg-white px-4 py-3" data-testid="nvlink-eq">
            <h3 className="text-xs font-semibold tracking-wide text-gray-500">{T("nvlink.coll.eqTitle")}</h3>
            <div className="mt-1 text-base">
              <Tex tex={String.raw`t_\text{ring} = \underbrace{2(N-1)\,\alpha}_\text{latency} + \underbrace{\frac{2(N-1)}{N}\cdot\frac{S}{B}}_\text{bandwidth}`} display />
            </div>
            <p className="mt-2 text-sm leading-relaxed text-gray-700">
              <RichText text={T("nvlink.coll.eqText")} />
            </p>
          </div>
          <h3 className={h3}>{T("nvlink.coll.treeTitle")}</h3>
          <p className="mt-2 text-sm leading-relaxed text-gray-700">
            <RichText text={T("nvlink.coll.tree")} />
          </p>
          <NvlinkAnim id="tree" />
          <h3 className={h3}>{T("nvlink.coll.ncclTitle")}</h3>
          <p className="mt-2 text-sm leading-relaxed text-gray-700">
            <RichText text={T("nvlink.coll.nccl")} />
          </p>
          <Sources ids={["ncclPerf", "ncclTree", "ncclEnv"]} />
        </Card>

        <Card id="code">
          <Heading n={num("code")} text={T("nvlink.code.title")} />
          <p className={P}>
            <RichText text={T("nvlink.code.text")} />
          </p>
          <p className="mt-3 rounded-lg bg-amber-50 px-4 py-2.5 text-sm leading-relaxed text-amber-950" data-testid="nvlink-shown">
            <RichText text={T("nvlink.code.shown")} />
          </p>
          <div className="mt-4 space-y-4">
            <div>
              <CodeBlock code={CODE_DDP} title={T("nvlink.code.ddp")} />
              <CodeNotes prefix="nvlink" id="ddp" />
              <NvlinkAnim id="ddp" />
              <Sources ids={["dist", "ddp", "torchrun"]} label="nvlink.docs" />
            </div>
            <div>
              <CodeBlock code={CODE_P2P} title={T("nvlink.code.p2p")} />
              <CodeNotes prefix="nvlink" id="p2p" />
              <NvlinkAnim id="p2p" />
              <Sources ids={["peer"]} label="nvlink.docs" />
            </div>
            <div>
              <CodeBlock code={CODE_SMI} title={T("nvlink.code.smi")} />
              <CodeNotes prefix="nvlink" id="smi" />
              <Sources ids={["smi", "ncclEnv"]} label="nvlink.docs" />
            </div>
            <div>
              <CodeBlock code={CODE_GLOO} title={T("nvlink.code.gloo")} />
              <CodeNotes prefix="nvlink" id="gloo" />
              <p className="mt-2 text-sm leading-relaxed text-gray-700">
                <RichText text={T("nvlink.code.glooText")} />
              </p>
            </div>
          </div>
          <h3 className={h3}>{T("nvlink.code.moreTitle")}</h3>
          <p className="mt-2 text-sm leading-relaxed text-gray-700">
            <RichText text={T("nvlink.code.more")} />
          </p>
          <Sources ids={["cudaMpi", "nvshmem"]} label="nvlink.docs" />
        </Card>

        <Card id="practice">
          <Heading n={num("practice")} text={T("nvlink.practice.title")} />
          <div className="mt-3 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-2">
            <div className="rounded-lg bg-blue-50/60 p-3">
              <h3 className="font-semibold text-blue-900">{T("nvlink.practice.mattersTitle")}</h3>
              <p className="mt-1">
                <RichText text={T("nvlink.practice.matters")} />
              </p>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <h3 className="font-semibold text-gray-900">{T("nvlink.practice.fineTitle")}</h3>
              <p className="mt-1">
                <RichText text={T("nvlink.practice.fine")} />
              </p>
            </div>
          </div>
          <p className={P} data-testid="nvlink-worked">
            <RichText text={fill(T("nvlink.practice.worked"), worked)} />
          </p>
          <NvlinkAnim id="overlap" />
          <CostCalculator />
          <p className={P}>
            <RichText text={T("nvlink.practice.caveat")} />
          </p>
        </Card>
      </article>
    </div>
  );
}
