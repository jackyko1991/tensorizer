import { useEffect, useRef, useState } from "react";
import Tex, { RichText } from "../components/Tex";
import MathsAnim from "../components/MathsAnim";
import AnalogyPicture from "../components/AnalogyPicture";
import { usePageText, type PageKey } from "../content/pages";
import { attainable, intensity, kernelCost, ridge } from "../tensorMaths";
import { animsAfter, type MathsCase } from "../mathsAnim";
import bench from "../content/blas_bench.json";
import { FEYNMAN, Feynman, EqCards, Links, Figure } from "./MathsConcept";

const C = { grid: "#e5e7eb", muted: "#6b7280", leg: "#1f2937" };
const EQUATIONS: { tex: string; code: string; anim: MathsCase }[] = [
    { tex: String.raw`P_\text{att} = \min(P,\ \beta I), \qquad I = \frac{\text{FLOPs}}{\text{bytes moved}}, \qquad I^* = \frac{P}{\beta}`, code: "f = torch.compile(lambda x: torch.relu(a * x + b))   # one fused kernel instead of three", anim: "eqRoof" },
  ];
const REFERENCES = [
    { label: "S. Williams, A. Waterman & D. Patterson (2009), Roofline: an insightful visual performance model for multicore architectures, Commun. ACM 52(4), 65–76", url: "https://doi.org/10.1145/1498765.1498785" },
    { label: "Wikipedia: Roofline model", url: "https://en.wikipedia.org/wiki/Roofline_model" },
    { label: "PyTorch: torch.compile", url: "https://docs.pytorch.org/docs/stable/generated/torch.compile.html" },
  ];

export default function RooflineCard({ number }: { number: number }) {
  const T = usePageText();
  return <section id="flow-roof" className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6">
    <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{number}</span>
      {T("maths.roof.title")}
    </h2>
    <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
      {FEYNMAN.map(f=><div key={f} className="space-y-4">
        <Feynman kind={f} text={T(`maths.roof.${f}` as PageKey)} label={T(`maths.label.${f}` as PageKey)}
          illustration={f==="analogy" ? <AnalogyPicture kind="roof"/> : undefined}/>
        {animsAfter("roof",f).map(id=><MathsAnim key={id} id={id}/>)}
        {f==="precise" && <><Roofline/><p><RichText text={T("maths.roof.ops")}/></p></>}
        {f==="example" && <><p><RichText text={T("maths.roof.fusion")}/></p>
          {animsAfter("roof","fusion").map(id=><MathsAnim key={id} id={id}/>)}</>}
      </div>)}
      <aside className="rounded-lg border border-blue-100 bg-blue-50/50 p-4"><h3 className="mb-2 font-semibold text-gray-900">{T("maths.recap.mis")}</h3><p><RichText text={T("maths.roof.misconception")}/></p></aside>
      <EqCards items={EQUATIONS} title={T("maths.label.code")}/>
      <Links items={REFERENCES} title={T("maths.label.links")}/>
    </div>
  </section>;
}

// This machine: best measured float32 GEMM (BLAS tab) and theoretical peaks, in GFLOP/s; bandwidths in GB/s are
// nominal (not measured): dual-channel DDR4-3200 assumed for the CPU, the GTX 1080 Ti's specification for the GPU.
const best = (id: keyof typeof bench.results) => Math.max(...bench.results[id].rows.map((r) => r.gflops));
const MACHINE = {
  cpu: { name: bench.machine.cpu, gemm: best("cpu_float32_all"), peak: bench.peak.cpu.float32.all, bw: 51.2, color: "#2563eb" },
  gpu: { name: bench.machine.gpu, gemm: best("cuda_float32"), peak: bench.peak.cuda.float32, bw: 484, color: "#c2410c" },
};
const CONV_PIXELS = 56 * 56;

const tick = (v: number) => (v >= 1000 ? `${v / 1000}k` : String(v));

function Roofline() {
  const T = usePageText();
  const [n, setN] = useState(512);
  const W = 560;
  const H = 300;
  const m = { l: 48, r: 14, t: 14, b: 26 };
  const [x0, x1, y0, y1] = [0.01, 1000, 0.1, 100000];
  const X = (v: number) => m.l + ((Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0))) * (W - m.l - m.r);
  const Y = (v: number) => H - m.b - ((Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (H - m.t - m.b);
  // on a narrow screen the chart shrinks: enlarge its text and keep only the essential labels
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(W);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth || W));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const narrow = width < 450;
  const k = Math.min(1.7, Math.max(1, (0.72 * W) / width));
  const fs = (v: number) => v * k;
  type Pt = { label: string; i: number; pos: "above" | "below" | "left" | "right"; gemm?: boolean; fuse?: boolean };
  const pts: Pt[] = [
    { label: "x + y", i: intensity(kernelCost("add", 1e6)), pos: "above" },
    { label: "", i: 3 / 24, pos: "left", fuse: true },
    { label: "dot", i: intensity(kernelCost("dot", 1e6)), pos: "above" },
    { label: "relu(a·x+b), fused", i: 3 / 8, pos: "right", fuse: true },
    { label: "gemv", i: intensity(kernelCost("gemv", 4096)), pos: "above" },
    { label: "conv 3×3", i: intensity(kernelCost("conv", CONV_PIXELS)), pos: "above" },
    { label: `GEMM n=${n}`, i: intensity(kernelCost("gemm", n)), pos: "below", gemm: true },
  ];
  const cpu = MACHINE.cpu;
  const gpu = MACHINE.gpu;
  const machineText = T("maths.roof.machine")
    .replace("{cpu}", `${cpu.name}: ${Math.round(cpu.gemm)} GFLOP/s`)
    .replace("{gpu}", `${gpu.name}: ${(gpu.gemm / 1000).toFixed(1)} TFLOP/s`);
  const roofPath = (peak: number, bw: number) =>
    `M${X(x0)},${Y(attainable(x0, peak, bw))} L${X(ridge(peak, bw))},${Y(peak)} L${X(x1)},${Y(peak)}`;
  const bound = (i: number) => (i < ridge(cpu.gemm, cpu.bw) ? T("maths.roof.memory") : T("maths.roof.compute"));
  const iOf = (label: string) => pts.find((p) => p.label === label)!.i;
  const table: { name: string; flops: string; bytes: string; ai: string; i: number }[] = [
    { name: "x + y", flops: "n", bytes: "12n", ai: String.raw`\tfrac{1}{12}`, i: iOf("x + y") },
    { name: "dot", flops: "2n", bytes: "8n", ai: String.raw`\tfrac{1}{4}`, i: iOf("dot") },
    { name: "gemv", flops: "2n^2", bytes: "4(n^2 + 2n)", ai: String.raw`\approx \tfrac{1}{2}`, i: iOf("gemv") },
    { name: `GEMM, n = ${n}`, flops: "2n^3", bytes: "12n^2", ai: String.raw`\tfrac{n}{6} \approx ${Math.round(n / 6)}`, i: iOf(`GEMM n=${n}`) },
    { name: "conv 3×3, C = 64", flops: String.raw`18\,C^2 p`, bytes: String.raw`4(2Cp + 9C^2)`, ai: String.raw`\approx ${Math.round(iOf("conv 3×3"))}`, i: iOf("conv 3×3") },
  ];
  // slope of the memory roofs on screen, for the labels written along them
  const slope = (Math.atan2(Y(1) - Y(10), X(10) - X(1)) * 180) / Math.PI;
  return (
    <div className="space-y-3">
      <Figure caption={T("maths.roof.caption")} testid="maths-roofline">
        <div className="mx-auto max-w-xl" ref={wrap}>
          <div className="mb-1 text-xs text-gray-600">
            <RichText text={T("maths.roof.y")} />
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="roofline">
            {[0.1, 1, 10, 100, 1000, 10000, 100000].map((v) => (
              <g key={`y${v}`}>
                <line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} stroke={C.grid} />
                <text x={m.l - 5} y={Y(v) + 3.5} textAnchor="end" fontSize={fs(10.5)} fill={C.muted}>
                  {tick(v)}
                </text>
              </g>
            ))}
            {[0.01, 0.1, 1, 10, 100, 1000].map((v) => (
              <g key={`x${v}`}>
                <line x1={X(v)} x2={X(v)} y1={m.t} y2={H - m.b} stroke="#f3f4f6" />
                <text x={X(v)} y={H - m.b + 6 + fs(8)} textAnchor="middle" fontSize={fs(10.5)} fill={C.muted}>
                  {tick(v)}
                </text>
              </g>
            ))}
            {[cpu, gpu].map((d) => {
              const lx = X(0.013);
              const ly = Y(attainable(0.013, d.gemm, d.bw)) + (d === cpu ? 6 + fs(9) : -7); // CPU below its line, GPU above
              return (
                <g key={d.name}>
                  <line x1={X(ridge(d.peak, d.bw))} x2={X(x1)} y1={Y(d.peak)} y2={Y(d.peak)} stroke={d.color} strokeDasharray="2 3" opacity="0.7" />
                  <line x1={X(ridge(d.gemm, d.bw))} x2={X(ridge(d.gemm, d.bw))} y1={Y(d.gemm)} y2={H - m.b} stroke={d.color} strokeDasharray="3 3" opacity="0.5" />
                  <text x={X(ridge(d.gemm, d.bw)) + 3} y={H - m.b - 4} fontSize={fs(9.5)} fill={d.color}>
                    {narrow ? "" : `${T("maths.roof.ridge")} `}
                    {ridge(d.gemm, d.bw).toFixed(1)}
                  </text>
                  <path d={roofPath(d.gemm, d.bw)} fill="none" stroke={d.color} strokeWidth="2.2" />
                  <text x={lx} y={ly} fontSize={fs(10.5)} fill={d.color} fontWeight="600" transform={`rotate(${-slope} ${lx} ${ly})`}>
                    {narrow ? (d === cpu ? "CPU" : "GPU") : T(d === cpu ? "maths.roof.cpu" : "maths.roof.gpu")}
                  </text>
                  {!narrow && (
                    <text x={X(x1) - 4} y={Y(d.peak) - 4} fontSize={fs(9)} fill={d.color} textAnchor="end">
                      {T("maths.roof.peak")}
                    </text>
                  )}
                </g>
              );
            })}
            {!narrow && (
              <>
                <text x={X(0.6)} y={Y(0.25)} fontSize={fs(10.5)} fill={C.muted} textAnchor="middle">
                  ← {T("maths.roof.memory")}
                </text>
                <text x={X(1000) - 4} y={Y(0.25)} fontSize={fs(10.5)} fill={C.muted} textAnchor="end">
                  {T("maths.roof.compute")} →
                </text>
              </>
            )}
            {/* fusion: the same element-wise chain, three kernels -> one, moves right along the CPU slope */}
            <line
              x1={X(3 / 24) + 6}
              y1={Y(attainable(3 / 24, cpu.gemm, cpu.bw)) + 12}
              x2={X(3 / 8) - 6}
              y2={Y(attainable(3 / 8, cpu.gemm, cpu.bw)) + 12}
              stroke="#059669"
              strokeWidth="1.5"
              markerEnd="url(#maths-fuse)"
            />
            <defs>
              <marker id="maths-fuse" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill="#059669" />
              </marker>
            </defs>
            {pts.map((p) => {
              const cx = X(p.i);
              const cy = Y(attainable(p.i, cpu.gemm, cpu.bw));
              const at = {
                above: { x: cx, y: cy - 8, a: "middle" },
                below: { x: cx, y: cy + 6 + fs(9), a: "middle" },
                left: { x: cx - 2, y: cy + 10 + fs(9), a: "end" },
                right: { x: cx + 2, y: cy + 10 + fs(9), a: "start" },
              }[p.pos] as { x: number; y: number; a: "middle" | "start" | "end" };
              return (
                <g key={p.i}>
                  {[cpu, gpu].map((d) => (
                    <circle
                      key={d.name}
                      cx={cx}
                      cy={Y(attainable(p.i, d.gemm, d.bw))}
                      r={p.gemm ? 5 : 3.8}
                      fill={d === cpu ? (p.gemm ? "#1e3a8a" : d.color) : "#fff"}
                      stroke={d.color}
                      strokeWidth="1.6"
                    />
                  ))}
                  {(!narrow || p.gemm) && p.label && (
                    <text
                      x={at.x}
                      y={at.y}
                      fontSize={fs(10)}
                      fill={p.fuse ? "#047857" : C.leg}
                      textAnchor={at.a}
                      fontWeight={p.gemm ? 700 : 400}
                    >
                      {p.label}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
          <div className="text-center text-xs text-gray-600">
            <RichText text={T("maths.roof.x")} />
          </div>
          <label className="mt-3 flex items-center gap-3 text-xs text-gray-700">
            <span className="shrink-0">
              <RichText text={T("maths.roof.slider")} />
            </span>
            <input
              type="range"
              min={4}
              max={12}
              step={1}
              value={Math.log2(n)}
              onChange={(e) => setN(2 ** Number(e.target.value))}
              className="flex-1"
              data-testid="maths-gemm-n"
            />
            <span className="w-12 text-right font-mono">{n}</span>
          </label>
        </div>
      </Figure>
      <p className="text-xs text-gray-600">
        <RichText text={machineText} />
      </p>
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-left text-xs">
          <thead className="bg-gray-50 text-gray-500">
            <tr>
              <th className="px-3 py-2 font-medium">{T("maths.roof.table.kernel")}</th>
              <th className="px-3 py-2 font-medium">{T("maths.roof.table.flops")}</th>
              <th className="px-3 py-2 font-medium">{T("maths.roof.table.bytes")}</th>
              <th className="px-3 py-2 font-medium">{T("maths.roof.table.intensity")}</th>
              <th className="px-3 py-2 font-medium">{T("maths.roof.table.bound")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {table.map((r) => (
              <tr key={r.name}>
                <td className="px-3 py-2 font-mono text-gray-800">{r.name}</td>
                <td className="px-3 py-2"><Tex tex={r.flops} /></td>
                <td className="px-3 py-2"><Tex tex={r.bytes} /></td>
                <td className="px-3 py-2"><Tex tex={r.ai} /></td>
                <td className={`px-3 py-2 ${r.i < ridge(cpu.gemm, cpu.bw) ? "text-amber-700" : "text-blue-700"}`}>{bound(r.i)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

