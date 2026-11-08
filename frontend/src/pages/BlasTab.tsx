import type { ReactNode } from "react";
import { decades, fmtGflops, gemmIntensity, linePath, logScale, percentOfPeak, roofline } from "../blasChart";
import BlasAnim, { CodeNotes } from "../components/BlasAnim";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import bench from "../content/blas_bench.json";
import { usePageText, type PageKey } from "../content/pages";

/** One measured size of a configuration in blas_bench.json (written by scripts/blas_benchmark.py). */
interface Row {
  n: number;
  seconds: number;
  gflops: number;
  gflops_range: number[];
}
interface Config {
  id: string;
  device: string;
  dtype: string;
  threads: number | null;
  rows: Row[];
}
interface Bench {
  machine: {
    cpu: string;
    cores: number | null;
    threads: number;
    gpu: string | null;
    torch: string;
    date: string;
    passes: number;
    cpu_load_percent?: (number | null)[];
  };
  blas: { BLAS_INFO: string | null; LAPACK_INFO: string | null; cuda: string | null };
  peak: {
    cpu: { float32: { "1t": number; all: number }; float64: { "1t": number; all: number }; assumption: string } | null;
    cuda: { float32: number; float64: number; assumption: string } | null;
  };
  agreement: { reproducibility: Record<string, number | boolean> };
  results: Record<string, Config>;
}

const B = bench as unknown as Bench;

/** Every section of the tab, in order: numbered 1..N in the table of contents and on the cards. */
const SECTIONS = ["what", "libs", "which", "repro", "bench", "method"] as const;
const num = (key: (typeof SECTIONS)[number]) => SECTIONS.indexOf(key) + 1;

/** Axis tick: 0.01 -> "0.01", 1000 -> "1k". */
const fmtTick = (v: number) => (v >= 1000 ? fmtGflops(v) : String(v));

/** 8.1e-6 -> "$8.1 \times 10^{-6}$" (for RichText). */
const texExp = (x: number) => {
  const [mant, exp] = x.toExponential(1).split("e");
  return `$${mant} \\times 10^{${Number(exp)}}$`;
};

const fill = (s: string, v: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (m, k) => String(v[k] ?? m));

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
    <section id={`blas-${id}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6" data-testid={`blas-${id}`}>
      {children}
    </section>
  );
}

/** An external documentation link, styled like the site's other external links. */
function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-700 hover:underline">
      {children}
      <span aria-hidden> ↗</span>
    </a>
  );
}

// --- 1. what BLAS is ------------------------------------------------------------------------------------

const LEVELS: { key: PageKey; op: string; routines: string; flops: string; data: string; ai: string }[] = [
  { key: "blas.levels.l1", op: String.raw`y \leftarrow \alpha x + y`, routines: "axpy, dot, nrm2", flops: "2n", data: "3n", ai: String.raw`\tfrac{2}{3}` },
  { key: "blas.levels.l2", op: String.raw`y \leftarrow \alpha A x + \beta y`, routines: "gemv, trsv", flops: "2n^2", data: String.raw`n^2`, ai: "2" },
  { key: "blas.levels.l3", op: String.raw`C \leftarrow \alpha A B + \beta C`, routines: "gemm, trsm, syrk", flops: "2n^3", data: "4n^2", ai: String.raw`\tfrac{n}{2}` },
];

const EQS: { key: PageKey; tex: string }[] = [
  { key: "blas.eq.flops", tex: String.raw`\text{FLOPs} = 2n^3` },
  { key: "blas.eq.rate", tex: String.raw`\text{GFLOP/s} = \frac{2n^3}{t \cdot 10^9}` },
  { key: "blas.eq.ai", tex: String.raw`I = \frac{2n^3}{4n^2 \cdot 8\,\text{B}} = \frac{n}{16}\ \text{FLOP/B}` },
  { key: "blas.eq.roof", tex: String.raw`P_\text{att} = \min\left(P,\ \beta I\right)` },
];

/** Log-log roofline with illustrative numbers and where the BLAS levels sit on it. */
function Roofline() {
  const T = usePageText();
  const P = 400;
  const BW = 40;
  const W = 360;
  const H = 210;
  const m = { l: 40, r: 12, t: 12, b: 26 };
  const X = logScale(0.01, 1000, m.l, W - m.r);
  const Y = logScale(0.1, 1000, H - m.b, m.t);
  const ridge = P / BW;
  const roof = linePath([0.01, ridge, 1000].map((i) => [X(i), Y(roofline(i, P, BW))]));
  // labels below-right of the points on the slope, below the point on the flat roof
  const pts: { i: number; label: string; flat?: boolean }[] = [
    { i: 2 / 24, label: "daxpy" },
    { i: 2 / 8, label: "dgemv" },
    { i: gemmIntensity(64, 8), label: "dgemm, n = 64" },
    { i: gemmIntensity(2048, 8), label: "n = 2048", flat: true },
  ];
  const tick = fmtTick;
  const slope = (Math.atan2(Y(1) - Y(10), X(10) - X(1)) * 180) / Math.PI;
  const mx = X(0.012);
  const my = Y(roofline(0.012, P, BW)) - 6;
  return (
    <figure className="mt-4 rounded-lg border border-gray-200 bg-gray-50/60 p-3 text-xs" data-testid="blas-roofline">
      <div className="mx-auto max-w-lg">
      <div className="mb-1 text-gray-600">
        <RichText text={T("blas.roof.y")} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={T("blas.roof.caption")}>
        {decades(0.1, 1000).map((v) => (
          <g key={`y${v}`}>
            <line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} stroke="#e5e7eb" />
            <text x={m.l - 4} y={Y(v) + 3} textAnchor="end" fontSize="9" fill="#6b7280">
              {tick(v)}
            </text>
          </g>
        ))}
        {decades(0.01, 1000).map((v) => (
          <g key={`x${v}`}>
            <line x1={X(v)} x2={X(v)} y1={m.t} y2={H - m.b} stroke="#f3f4f6" />
            <text x={X(v)} y={H - m.b + 11} textAnchor="middle" fontSize="9" fill="#6b7280">
              {tick(v)}
            </text>
          </g>
        ))}
        <line x1={X(ridge)} x2={X(ridge)} y1={Y(P)} y2={H - m.b} stroke="#9ca3af" strokeDasharray="3 3" />
        <path d={roof} fill="none" stroke="#111827" strokeWidth="2" />
        <text x={mx} y={my} fontSize="9" fill="#374151" transform={`rotate(${-slope} ${mx} ${my})`}>
          {T("blas.roof.memory")}
        </text>
        <text x={X(100)} y={Y(P) - 6} fontSize="9" fill="#374151" textAnchor="middle">
          {T("blas.roof.compute")}
        </text>
        {pts.map((p) => {
          const y = Y(roofline(p.i, P, BW));
          return (
            <g key={p.label}>
              <circle cx={X(p.i)} cy={y} r="4" fill="#2a78d6" stroke="#fff" strokeWidth="2" />
              <text
                x={p.flat ? X(p.i) : X(p.i) + 6}
                y={y + 14}
                textAnchor={p.flat ? "middle" : "start"}
                fontSize="9"
                fill="#111827"
              >
                {p.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="text-center text-gray-600">
        <RichText text={T("blas.roof.x")} />
      </div>
      </div>
      <figcaption className="mt-2 text-[11px] leading-relaxed text-gray-500">
        <RichText text={T("blas.roof.caption")} />
      </figcaption>
    </figure>
  );
}

// --- 2. libraries ---------------------------------------------------------------------------------------

/** [id, name, official documentation, licence]; hardware / notes / route are page text (blas.lib.<id>.*). */
const LIBS: { id: string; name: string; url: string; licence: string; extra?: { name: string; url: string } }[] = [
  { id: "netlib", name: "Reference BLAS (Netlib)", url: "https://www.netlib.org/blas/", licence: "BSD-style (netlib)" },
  { id: "openblas", name: "OpenBLAS", url: "https://www.openmathlib.org/OpenBLAS/docs/", licence: "BSD-3-Clause" },
  {
    id: "mkl",
    name: "Intel oneMKL",
    url: "https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/current/overview.html",
    licence: "Intel Simplified Software License (proprietary, redistributable)",
  },
  { id: "aocl", name: "AMD AOCL-BLAS", url: "https://www.amd.com/en/developer/aocl/dense.html", licence: "BSD-3-Clause (some files MIT)" },
  { id: "blis", name: "BLIS", url: "https://github.com/flame/blis/tree/master/docs", licence: "BSD-3-Clause" },
  { id: "accelerate", name: "Apple Accelerate (vecLib)", url: "https://developer.apple.com/documentation/accelerate/blas", licence: "Proprietary (part of macOS)" },
  {
    id: "cublas",
    name: "cuBLAS",
    url: "https://docs.nvidia.com/cuda/cublas/",
    licence: "Proprietary (NVIDIA EULA, redistributable)",
    extra: { name: "cuBLASLt", url: "https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api" },
  },
  {
    id: "rocblas",
    name: "rocBLAS",
    url: "https://rocm.docs.amd.com/projects/rocBLAS/en/latest/",
    licence: "MIT",
    extra: { name: "hipBLASLt", url: "https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/" },
  },
  { id: "onednn", name: "oneDNN", url: "https://uxlfoundation.github.io/oneDNN/", licence: "Apache-2.0" },
];

function LibTable() {
  const T = usePageText();
  const head = ["lib", "hw", "licence", "notes", "via"] as const;
  return (
    <details className="group mt-4 rounded-lg border border-gray-200 bg-white" data-testid="blas-lib-table">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2 text-sm font-medium text-gray-800 [&::-webkit-details-marker]:hidden">
        <span className="text-xs text-gray-500 group-open:hidden" aria-hidden>▸</span>
        <span className="hidden text-xs text-gray-500 group-open:inline" aria-hidden>▾</span>
        {T("blas.libs.table.title")}
      </summary>
      {/* phones: one compact block per library instead of a sideways-scrolling table */}
      <ul className="divide-y divide-gray-100 border-t border-gray-100 text-xs text-gray-700 md:hidden">
        {LIBS.map((l) => (
          <li key={l.id} className="space-y-1 px-4 py-2.5" data-testid={`blas-lib-card-${l.id}`}>
            <div className="font-medium text-gray-900">
              <Ext href={l.url}>{l.name}</Ext>
              {l.extra && (
                <>
                  {" / "}
                  <Ext href={l.extra.url}>{l.extra.name}</Ext>
                </>
              )}
            </div>
            <div className="text-gray-500">
              {T(`blas.lib.${l.id}.hw` as PageKey)} · {l.licence}
            </div>
            <p>
              <RichText text={T(`blas.lib.${l.id}.notes` as PageKey)} />
            </p>
            <p>
              <span className="text-gray-500">{T("blas.libs.table.via")}: </span>
              <RichText text={T(`blas.lib.${l.id}.via` as PageKey)} />
            </p>
          </li>
        ))}
      </ul>
      <div className="hidden overflow-x-auto border-t border-gray-100 md:block">
        <table className="w-full min-w-[48rem] text-left text-xs">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              {head.map((h) => (
                <th key={h} className="px-3 py-1.5 font-semibold">
                  {T(`blas.libs.table.${h}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 text-gray-700">
            {LIBS.map((l) => (
              <tr key={l.id} data-testid={`blas-lib-${l.id}`}>
                <td className="px-3 py-1.5 align-top font-medium whitespace-nowrap text-gray-900">
                  <Ext href={l.url}>{l.name}</Ext>
                  {l.extra && (
                    <>
                      {" / "}
                      <Ext href={l.extra.url}>{l.extra.name}</Ext>
                    </>
                  )}
                </td>
                <td className="px-3 py-1.5 align-top">{T(`blas.lib.${l.id}.hw` as PageKey)}</td>
                <td className="px-3 py-1.5 align-top">{l.licence}</td>
                <td className="px-3 py-1.5 align-top">
                  <RichText text={T(`blas.lib.${l.id}.notes` as PageKey)} />
                </td>
                <td className="px-3 py-1.5 align-top">
                  <RichText text={T(`blas.lib.${l.id}.via` as PageKey)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-1.5 border-t border-gray-100 px-4 py-2 text-xs leading-relaxed text-gray-600">
        <p>
          <RichText text={T("blas.libs.lapack")} /> <Ext href="https://www.netlib.org/lapack/">LAPACK</Ext>
        </p>
        <p>
          <RichText text={T("blas.libs.sparse")} /> <Ext href="https://docs.nvidia.com/cuda/cusparse/">cuSPARSE</Ext>
        </p>
      </div>
    </details>
  );
}

// --- 3. which BLAS --------------------------------------------------------------------------------------

const CODE_TORCH = `import torch
print(torch.__config__.show())          # build settings: look for BLAS_INFO, LAPACK_INFO, MKL, MKL-DNN
torch.backends.mkl.is_available()       # True: linked against Intel oneMKL
torch.backends.mkldnn.is_available()    # True: oneDNN kernels for convolutions etc.
torch.get_num_threads()                 # size of PyTorch's intra-op thread pool`;

const CODE_NUMPY = `import numpy
numpy.show_config()                     # "Build Dependencies" -> blas / lapack: name, version`;

const CODE_THREADS = `# in the shell, before Python starts (read once, when each library loads)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python run.py

# inside Python
import torch
torch.set_num_threads(4)                # PyTorch's pool (and MKL's, in an MKL build)`;

const DOCS = {
  torch: [
    { name: "torch.__config__", url: "https://docs.pytorch.org/docs/stable/config_mod.html" },
    { name: "torch.backends", url: "https://docs.pytorch.org/docs/stable/backends.html" },
  ],
  numpy: [{ name: "numpy.show_config", url: "https://numpy.org/doc/stable/reference/generated/numpy.show_config.html" }],
  threads: [{ name: "torch.set_num_threads", url: "https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html" }],
  tol: [{ name: "torch.testing", url: "https://docs.pytorch.org/docs/stable/testing.html" }],
  det: [{ name: "Reproducibility (PyTorch notes)", url: "https://docs.pytorch.org/docs/stable/notes/randomness.html" }],
};

function DocLinks({ links }: { links: { name: string; url: string }[] }) {
  const T = usePageText();
  return (
    <p className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-xs">
      <span className="text-gray-500">{T("blas.which.docs")}</span>
      {links.map((l) => (
        <Ext key={l.url} href={l.url}>
          <span className="font-mono">{l.name}</span>
        </Ext>
      ))}
    </p>
  );
}

// --- 4. reproducibility ---------------------------------------------------------------------------------

const CODE_ORDER = `>>> (0.1 + 0.2) + 0.3
0.6000000000000001
>>> 0.1 + (0.2 + 0.3)
0.6`;

const CODE_TOL = `import torch
torch.testing.assert_close(got, want)    # float32 defaults: rtol 1.3e-6, atol 1e-5
torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)   # long float32 sums
assert (dE <= 1e-6 * E.abs().max()).all()   # relative to the size of the values`;

const CODE_DET = `import os, torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # cuBLAS: before the first CUDA call
torch.use_deterministic_algorithms(True)          # error if an op has no deterministic version
torch.backends.cudnn.benchmark = False            # no autotuned (possibly different) kernel choice
torch.manual_seed(0)`;

// --- 5. measured chart ----------------------------------------------------------------------------------

const BLUE = "#2a78d6"; // float32
const ORANGE = "#eb6834"; // float64

interface Series {
  id: string;
  color: string;
  hollow: boolean;
}

const CPU_SERIES: Series[] = [
  { id: "cpu_float32_all", color: BLUE, hollow: false },
  { id: "cpu_float32_1t", color: BLUE, hollow: true },
  { id: "cpu_float64_all", color: ORANGE, hollow: false },
  { id: "cpu_float64_1t", color: ORANGE, hollow: true },
];
const GPU_SERIES: Series[] = [
  { id: "cuda_float32", color: BLUE, hollow: false },
  { id: "cuda_float64", color: ORANGE, hollow: false },
];

const Y_LO = 5;
const Y_HI = 30000; // same fixed limits in both panels, so CPU and GPU compare directly

function Panel({ title, series, peaks }: { title: string; series: Series[]; peaks: { value: number; color: string }[] }) {
  const T = usePageText();
  const present = series.filter((s) => B.results[s.id]);
  if (!present.length) return null;
  const sizes = B.results[present[0].id].rows.map((r) => r.n);
  const W = 340;
  const H = 250;
  const m = { l: 36, r: 12, t: 10, b: 24 };
  const X = logScale(sizes[0], sizes[sizes.length - 1], m.l, W - m.r);
  const Y = logScale(Y_LO, Y_HI, H - m.b, m.t);
  const label = (id: string) => fill(T(`blas.series.${id}` as PageKey), { threads: B.machine.threads });
  return (
    <figure className="min-w-0 rounded-lg border border-gray-200 bg-white p-3" data-testid="blas-chart-panel">
      <figcaption className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="text-sm font-medium text-gray-900">{title}</span>
        <span className="text-[11px] text-gray-500">
          <RichText text={T("blas.chart.y")} />
        </span>
      </figcaption>
      <ul className="mb-1 flex min-h-[2.25rem] flex-wrap content-start gap-x-3 gap-y-1 text-[11px] text-gray-700">
        {present.map((s) => (
          <li key={s.id} className="flex items-center gap-1.5">
            <svg width="22" height="10" aria-hidden>
              <line x1="0" x2="22" y1="5" y2="5" stroke={s.color} strokeWidth={s.hollow ? 1.25 : 2} />
              <circle cx="11" cy="5" r="3.5" fill={s.hollow ? "#fff" : s.color} stroke={s.color} strokeWidth="1.5" />
            </svg>
            {label(s.id)}
          </li>
        ))}
        <li className="flex items-center gap-1.5 text-gray-500">
          <svg width="22" height="10" aria-hidden>
            <line x1="0" x2="22" y1="5" y2="5" stroke="#6b7280" strokeWidth="1.25" strokeDasharray="4 3" />
          </svg>
          {T("blas.chart.peak")}
        </li>
      </ul>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={title}>
        {decades(Y_LO, Y_HI)
          .filter((v) => v >= Y_LO && v <= Y_HI)
          .map((v) => (
            <g key={v}>
              <line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} stroke="#e5e7eb" />
              <text x={m.l - 4} y={Y(v) + 3} textAnchor="end" fontSize="9" fill="#6b7280">
                {fmtGflops(v)}
              </text>
            </g>
          ))}
        {sizes.map((n) => (
          <text key={n} x={X(n)} y={H - m.b + 12} textAnchor="middle" fontSize="9" fill="#6b7280">
            {n}
          </text>
        ))}
        <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="#9ca3af" />
        {peaks.map((p) => (
          <g key={`${p.color}${p.value}`}>
            <line x1={m.l} x2={W - m.r} y1={Y(p.value)} y2={Y(p.value)} stroke={p.color} strokeWidth="1.25" strokeDasharray="4 3" opacity="0.8" />
            <text x={W - m.r} y={Y(p.value) - 3} textAnchor="end" fontSize="8.5" fill="#4b5563">
              {fmtGflops(p.value)}
            </text>
          </g>
        ))}
        {present.map((s) => {
          const rows = B.results[s.id].rows;
          return (
            <g key={s.id}>
              <path
                d={linePath(rows.map((r) => [X(r.n), Y(r.gflops)]))}
                fill="none"
                stroke={s.color}
                strokeWidth={s.hollow ? 1.25 : 2}
              />
              {rows.map((r) => {
                const [lo, hi] = r.gflops_range;
                const x = X(r.n);
                return (
                  <g key={r.n} data-testid="blas-error-bar">
                    <title>{`${label(s.id)}, n = ${r.n}: ${fmtGflops(r.gflops)} GFLOP/s (${fmtGflops(lo)}–${fmtGflops(hi)})`}</title>
                    {/* error bar |-| standing upright: the spread over passes, with end caps */}
                    <line x1={x} x2={x} y1={Y(lo)} y2={Y(hi)} stroke="#1f2937" strokeWidth="1.25" />
                    <line x1={x - 3} x2={x + 3} y1={Y(lo)} y2={Y(lo)} stroke="#1f2937" strokeWidth="1.25" />
                    <line x1={x - 3} x2={x + 3} y1={Y(hi)} y2={Y(hi)} stroke="#1f2937" strokeWidth="1.25" />
                    <circle cx={x} cy={Y(r.gflops)} r="3.5" fill={s.hollow ? "#fff" : s.color} stroke={s.color} strokeWidth="1.5" />
                    <circle cx={x} cy={Y(r.gflops)} r="9" fill="transparent" />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <div className="text-center text-[11px] text-gray-600">
        <RichText text={T("blas.chart.x")} />
      </div>
    </figure>
  );
}

/** Peak (GFLOP/s) estimated for a configuration, or null. */
function peakOf(id: string): number | null {
  const c = B.results[id];
  if (!c) return null;
  const dtype = c.dtype as "float32" | "float64";
  if (c.device === "cuda") return B.peak.cuda?.[dtype] ?? null;
  const p = B.peak.cpu?.[dtype];
  return p ? (c.threads === 1 ? p["1t"] : p.all) : null;
}

function BenchTable() {
  const T = usePageText();
  const ids = [...CPU_SERIES, ...GPU_SERIES].map((s) => s.id).filter((id) => B.results[id]);
  if (!ids.length) return null;
  const sizes = B.results[ids[0]].rows.map((r) => r.n);
  const last = sizes[sizes.length - 1];
  return (
    <details className="group mt-3 rounded-lg border border-gray-200 bg-white" data-testid="blas-bench-table">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2 text-sm font-medium text-gray-800 [&::-webkit-details-marker]:hidden">
        <span className="text-xs text-gray-500 group-open:hidden" aria-hidden>▸</span>
        <span className="hidden text-xs text-gray-500 group-open:inline" aria-hidden>▾</span>
        {T("blas.chart.table")}
      </summary>
      <div className="overflow-x-auto border-t border-gray-100">
        <table className="w-full min-w-[36rem] text-right text-xs tabular-nums">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              <th className="px-3 py-1.5 text-left font-semibold">{T("blas.chart.config")}</th>
              {sizes.map((n) => (
                <th key={n} className="px-2 py-1.5 font-semibold">
                  n = {n}
                </th>
              ))}
              <th className="px-3 py-1.5 font-semibold">
                {T("blas.chart.pct")} (n = {last})
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 text-gray-700">
            {ids.map((id) => {
              const c = B.results[id];
              const pct = percentOfPeak(c.rows[c.rows.length - 1].gflops, peakOf(id));
              const where = c.device === "cuda" ? "GPU" : "CPU";
              return (
                <tr key={id}>
                  <td className="px-3 py-1.5 text-left whitespace-nowrap text-gray-900">
                    {where}, {fill(T(`blas.series.${id}` as PageKey), { threads: B.machine.threads })}
                  </td>
                  {c.rows.map((r) => (
                    <td key={r.n} className="px-2 py-1.5">
                      {fmtGflops(r.gflops)}
                    </td>
                  ))}
                  <td className="px-3 py-1.5">{pct == null ? "–" : `${pct} %`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

// --- the tab --------------------------------------------------------------------------------------------

export default function BlasTab() {
  const T = usePageText();
  const m = B.machine;
  const load = (m.cpu_load_percent ?? []).filter((x): x is number => x != null);
  const repro = B.agreement.reproducibility;
  const cpuPeak = B.peak.cpu;
  const gpuPeak = B.peak.cuda;
  const h3 = "text-xs font-semibold tracking-wide text-gray-500"; // no uppercase: it would change the TeX

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
      <PageToc
        scrollRoot="skills-scroll"
        items={SECTIONS.map((key, i) => ({ id: `blas-${key}`, mark: i + 1, label: T(`blas.${key}.title` as PageKey) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">{T("blas.title")}</h1>
          <p className="mt-3 text-gray-600">
            <RichText text={T("blas.intro")} />
          </p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">
            <RichText text={T("blas.howto")} />
          </p>
        </header>

        <Card id="what">
          <Heading n={num("what")} text={T("blas.what.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.what.text")} />
          </p>
          <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full min-w-[34rem] text-left text-xs" data-testid="blas-levels">
              <thead className="bg-gray-50 text-gray-600">
                <tr>
                  {(["level", "op", "routines", "flops", "data", "ai"] as const).map((h) => (
                    <th key={h} className="px-3 py-1.5 font-semibold">
                      {T(`blas.levels.${h}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 text-gray-700">
                {LEVELS.map((l) => (
                  <tr key={l.key}>
                    <td className="px-3 py-1.5 font-medium whitespace-nowrap text-gray-900">{T(l.key)}</td>
                    <td className="px-3 py-1.5 whitespace-nowrap">
                      <Tex tex={l.op} />
                    </td>
                    <td className="px-3 py-1.5 font-mono">{l.routines}</td>
                    <td className="px-3 py-1.5">
                      <Tex tex={l.flops} />
                    </td>
                    <td className="px-3 py-1.5">
                      <Tex tex={l.data} />
                    </td>
                    <td className="px-3 py-1.5">
                      <Tex tex={l.ai} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <BlasAnim id="levels" />
          <h3 className="mt-5 text-sm font-semibold text-gray-900">{T("blas.what.fastTitle")}</h3>
          <ul className="mt-2 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-3">
            {(["blocking", "kernel", "threads"] as const).map((k) => (
              <li key={k} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`blas.what.${k}`)} />
              </li>
            ))}
          </ul>
          <BlasAnim id="gemm" />
          <div className="mt-4 rounded-lg border border-gray-200 bg-white px-4 py-3" data-testid="blas-eq">
            <div className="grid gap-x-6 gap-y-3 md:grid-cols-2">
              {EQS.map((e) => (
                <div key={e.key} className="min-w-0">
                  <h3 className={h3}>
                    <RichText text={T(e.key)} />
                  </h3>
                  <div className="mt-1 text-base">
                    <Tex tex={e.tex} display />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-2 border-t border-gray-100 pt-2 text-xs text-gray-600">
              <RichText text={T("blas.eq.roofNote")} />
            </p>
          </div>
          <Roofline />
        </Card>

        <Card id="libs">
          <Heading n={num("libs")} text={T("blas.libs.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.libs.text")} />
          </p>
          <BlasAnim id="dispatch" />
          <LibTable />
        </Card>

        <Card id="which">
          <Heading n={num("which")} text={T("blas.which.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.which.text")} />
          </p>
          <div className="mt-4 space-y-4">
            <div>
              <CodeBlock code={CODE_TORCH} title={T("blas.which.torch")} />
              <CodeNotes prefix="blas" id="torch" />
              <DocLinks links={DOCS.torch} />
            </div>
            <div>
              <CodeBlock code={CODE_NUMPY} title={T("blas.which.numpy")} />
              <CodeNotes prefix="blas" id="numpy" />
              <DocLinks links={DOCS.numpy} />
            </div>
            <div>
              <CodeBlock code={CODE_THREADS} title={T("blas.which.threads")} />
              <CodeNotes prefix="blas" id="threads" />
              <DocLinks links={DOCS.threads} />
              <p className="mt-2 text-sm leading-relaxed text-gray-700">
                <RichText text={T("blas.which.threadsText")} />
              </p>
              <BlasAnim id="threads" />
            </div>
          </div>
          <p className="mt-4 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm leading-relaxed text-blue-900" data-testid="blas-project">
            <RichText text={T("blas.which.project")} />
          </p>
        </Card>

        <Card id="repro">
          <Heading n={num("repro")} text={T("blas.repro.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.repro.text")} />
          </p>
          <div className="mt-4">
            <CodeBlock code={CODE_ORDER} title={T("blas.repro.order")} />
            <CodeNotes prefix="blas" id="order" />
          </div>
          <BlasAnim id="order" />
          {typeof repro.cpu_vs_cuda_float32_max_abs_diff === "number" && (
            <p className="mt-3 text-sm leading-relaxed text-gray-700">
              <RichText
                text={fill(T("blas.repro.measured"), {
                  n: Number(repro.n),
                  threads: m.threads,
                  diff: texExp(repro.cpu_vs_cuda_float32_max_abs_diff),
                })}
              />
            </p>
          )}
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.repro.tol")} />
          </p>
          <div className="mt-3">
            <CodeBlock code={CODE_TOL} title={T("blas.repro.tolTitle")} />
            <CodeNotes prefix="blas" id="tol" />
            <DocLinks links={DOCS.tol} />
          </div>
          <p className="mt-4 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.repro.det")} />
          </p>
          <div className="mt-3">
            <CodeBlock code={CODE_DET} title={T("blas.repro.detTitle")} />
            <CodeNotes prefix="blas" id="det" />
            <DocLinks links={DOCS.det} />
          </div>
          <BlasAnim id="atomics" />
          <div className="mt-5 rounded-lg bg-amber-50 px-4 py-3 text-sm leading-relaxed text-amber-950" data-testid="blas-ci">
            <h3 className="font-semibold">{T("blas.repro.ci.title")}</h3>
            <p className="mt-1">
              <RichText text={T("blas.repro.ci.text")} />
            </p>
            <ul className="mt-2 list-disc space-y-1.5 pl-5">
              <li>
                <RichText text={T("blas.repro.ci.hopfield")} />
              </li>
              <li>
                <RichText text={T("blas.repro.ci.batched")} />
              </li>
            </ul>
            <p className="mt-2">
              <RichText text={T("blas.repro.ci.lesson")} />
            </p>
          </div>
        </Card>

        <Card id="bench">
          <Heading n={num("bench")} text={T("blas.bench.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.bench.text")} />
          </p>
          <BlasAnim id="overhead" />
          <div className="mt-4 grid gap-3 md:grid-cols-2" data-testid="blas-chart">
            <Panel
              title={T("blas.chart.cpu")}
              series={CPU_SERIES}
              peaks={cpuPeak ? [{ value: cpuPeak.float32.all, color: BLUE }, { value: cpuPeak.float64.all, color: ORANGE }] : []}
            />
            <Panel
              title={T("blas.chart.gpu")}
              series={GPU_SERIES}
              peaks={gpuPeak ? [{ value: gpuPeak.float32, color: BLUE }, { value: gpuPeak.float64, color: ORANGE }] : []}
            />
          </div>
          <p className="mt-2 text-[11px] text-gray-500">{T("blas.chart.range")}</p>
          <BenchTable />
          <p className="mt-4 text-sm leading-relaxed text-gray-700">
            <RichText text={T("blas.bench.observe")} />
          </p>
        </Card>

        <Card id="method">
          <Heading n={num("method")} text={T("blas.method.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText
              text={fill(T("blas.machine"), {
                cpu: m.cpu,
                cores: m.cores ?? m.threads,
                threads: m.threads,
                gpu: m.gpu ? fill(T("blas.machineGpu"), { gpu: m.gpu }) : "",
                torch: m.torch,
                blas: [B.blas.BLAS_INFO && `BLAS_INFO=${B.blas.BLAS_INFO}`, B.blas.cuda && `CUDA ${B.blas.cuda}`].filter(Boolean).join(", "),
                date: m.date,
                passes: m.passes,
              })}
            />
          </p>
          <BlasAnim id="timing" />
          {load.length > 0 && Math.max(...load) >= 10 && (
            <p className="mt-2 text-sm leading-relaxed text-amber-800">
              {fill(T("blas.load"), { lo: Math.round(Math.min(...load)), hi: Math.round(Math.max(...load)) })}
            </p>
          )}
          {(cpuPeak || gpuPeak) && (
            <p className="mt-2 text-sm leading-relaxed text-gray-700">
              {fill(T("blas.peakNote"), { cpu: cpuPeak?.assumption ?? "–", gpu: gpuPeak?.assumption ?? "–" })}
            </p>
          )}
          <p className="mt-2 text-sm leading-relaxed text-gray-700">{T("blas.caveat")}</p>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-gray-900 px-3 py-2 font-mono text-xs text-gray-100">
            pixi run -e cuda python scripts/blas_benchmark.py
          </pre>
          <CodeNotes prefix="blas" id="bench" />
        </Card>
      </article>
    </div>
  );
}
