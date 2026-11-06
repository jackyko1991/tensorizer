import type { ReactNode } from "react";
import CodeBlock from "../components/CodeBlock";
import HwChip from "../components/HwChip";
import LoopVsTensorAnim from "../components/LoopVsTensorAnim";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import bench from "../content/loop_bench.json";
import { usePageText, type PageKey } from "../content/pages";
import type { AnimCase } from "../loopAnim";

/** One measured size of a case in loop_bench.json (written by scripts/loop_benchmark.py). */
interface BenchRow {
  n: number;
  loop_s: number;
  tensor_cpu_s: number;
  tensor_gpu_s: number | null;
  speedup_cpu: number;
  speedup_gpu: number | null;
  speedup_cpu_range?: number[];
  speedup_gpu_range?: number[] | null;
}

interface Bench {
  machine: {
    cpu: string;
    threads: number;
    gpu: string | null;
    torch: string;
    date: string;
    passes: number;
    cpu_load_percent?: (number | null)[];
  };
  units: Record<string, string>;
  code: Record<string, { loop: string; tensor: string }>;
  cases: Record<string, BenchRow[]>;
}

const B = bench as unknown as Bench;

/** The six comparisons, in page order, with the problems that use the tensor version. */
const CASES: { id: string; usedIn: string[] }[] = [
  { id: "dot", usedIn: [] },
  { id: "pairwise", usedIn: ["mechanics/n_body_gravity"] },
  { id: "matmul", usedIn: [] },
  { id: "pendulum", usedIn: ["mechanics/simple_pendulum"] },
  { id: "diffusion", usedIn: ["thermodynamics/heat_diffusion"] },
  { id: "mc_pi", usedIn: ["others/monte_carlo_pi"] },
];

const DECADES = 4; // bar axis: 1x .. 10^4x
const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/** 1000000 -> "10⁶", 300 -> "300". */
function fmtN(n: number): string {
  const e = Math.log10(n);
  return Number.isInteger(e) && e >= 4 ? `10${String(e).replace(/\d/g, (d) => SUP[+d])}` : n.toLocaleString("en");
}

function fmtTime(s: number): string {
  if (s >= 1) return `${s.toPrecision(2)} s`;
  const [v, unit] = s >= 1e-3 ? [s * 1e3, "ms"] : [s * 1e6, "µs"];
  return `${v >= 100 ? Math.round(v) : v.toPrecision(2)} ${unit}`;
}

function fmtX(x: number): string {
  return x >= 10 ? `×${Math.round(x).toLocaleString("en")}` : `×${x.toPrecision(2)}`;
}

const pct = (x: number) => `${(100 * Math.min(Math.max(Math.log10(Math.max(x, 1)), 0), DECADES)) / DECADES}%`;

/** A log-scale speed-up bar with the spread over passes as a thin line. */
function Bar({ value, range, color, label }: { value: number; range?: number[] | null; color: string; label: string }) {
  return (
    <div className="flex items-center gap-2" title={`${label}: ${fmtX(value)}`}>
      <div className="relative h-3.5 flex-1 rounded-sm bg-gray-100">
        <div className={`absolute inset-y-0 left-0 rounded-sm ${color}`} style={{ width: pct(value) }} />
        {range && range.length === 2 && (
          // error bar |-|: the spread over passes, with end caps
          <div
            className="absolute inset-y-0.5 border-x-[1.5px] border-gray-800"
            style={{ left: pct(range[0]), width: `calc(${pct(range[1])} - ${pct(range[0])})` }}
            data-testid="error-bar"
          >
            <div className="absolute inset-x-0 top-1/2 h-[1.5px] -translate-y-1/2 bg-gray-800" />
          </div>
        )}
      </div>
      <span className="w-16 shrink-0 text-right font-mono text-xs tabular-nums text-gray-700">{fmtX(value)}</span>
    </div>
  );
}

function SpeedupChart({ id }: { id: string }) {
  const T = usePageText();
  const rows = B.cases[id] ?? [];
  const unit = B.units[id] ?? "N";
  const gpu = rows.some((r) => r.speedup_gpu != null);
  return (
    <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50/60 p-3 text-xs">
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-gray-600">
        <span className="font-medium text-gray-800">{T("loops.speedup")}</span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-blue-600" /> {T("loops.cpu")}
        </span>
        {gpu && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-600" /> {T("loops.gpu")}
          </span>
        )}
      </div>
      <div className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-2">
        {rows.map((r) => (
          <div key={r.n} className="contents">
            <div className="whitespace-nowrap">
              <div className="font-mono text-gray-800">
                {unit} = {fmtN(r.n)}
              </div>
              <div className="text-[11px] text-gray-500">
                {T("loops.loopTime")}: {fmtTime(r.loop_s)}
              </div>
            </div>
            <div className="space-y-1">
              <Bar value={r.speedup_cpu} range={r.speedup_cpu_range} color="bg-blue-600" label={T("loops.cpu")} />
              {r.speedup_gpu != null && (
                <Bar value={r.speedup_gpu} range={r.speedup_gpu_range} color="bg-emerald-600" label={T("loops.gpu")} />
              )}
            </div>
          </div>
        ))}
        {/* axis under the bars */}
        <div />
        <div className="flex items-center gap-2">
          <div className="relative h-4 flex-1 text-[10px] text-gray-500">
            {Array.from({ length: DECADES + 1 }, (_, k) => (
              <span
                key={k}
                className="absolute top-0 -translate-x-1/2 pt-1 leading-none"
                style={{ left: `${(100 * k) / DECADES}%` }}
              >
                {k === 0 ? "1" : k < 3 ? String(10 ** k) : `10${SUP[k]}`}
              </span>
            ))}
          </div>
          <span className="w-16 shrink-0" />
        </div>
      </div>
      <p className="mt-2 text-[11px] text-gray-500">{T("loops.range")}</p>
    </div>
  );
}

/** Each comparison in maths: vector form, index (tensor) form, and the same thing spelled in PyTorch. */
const EQ: Record<string, { vector: string; index: string; code: string; note?: boolean }> = {
  dot: {
    vector: String.raw`s = \mathbf{x} \cdot \mathbf{y}`,
    index: String.raw`s = \sum_i x_i\, y_i`,
    code: `torch.einsum("i,i->", x, y)`,
  },
  pairwise: {
    vector: String.raw`D = \bigl\lVert \mathbf{x}_i - \mathbf{x}_j \bigr\rVert`,
    index: String.raw`D_{ij} = \sqrt{\textstyle\sum_k (x_{ik} - x_{jk})^2}`,
    code: `(x[:, None, :] - x[None, :, :]).norm(dim=-1)`,
    note: true,
  },
  matmul: {
    vector: String.raw`C = A B`,
    index: String.raw`C_{ij} = \sum_k A_{ik}\, B_{kj}`,
    code: `torch.einsum("ik,kj->ij", A, B)`,
  },
  pendulum: {
    vector: String.raw`\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}), \quad \mathbf{y}_{n+1} = \mathbf{y}_n + \tfrac{h}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)`,
    index: String.raw`y_{b,n+1} = y_{b,n} + \tfrac{h}{6}\left(k_{1b} + 2k_{2b} + 2k_{3b} + k_{4b}\right)`,
    code: `y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # y: (B, 2)`,
    note: true,
  },
  diffusion: {
    vector: String.raw`u^{n+1} = u^n + \alpha\, \Delta t\, \nabla^2 u^n`,
    index: String.raw`\begin{aligned} u^{n+1}_{ij} = u^n_{ij} + r\bigl(&u^n_{i+1,j} + u^n_{i-1,j} \ &+ u^n_{i,j+1} + u^n_{i,j-1} - 4u^n_{ij}\bigr) \end{aligned}`,
    code: `c + r * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * c)`,
    note: true,
  },
  mc_pi: {
    vector: String.raw`\pi \approx 4\, \Pr\!\left(\lVert \mathbf{p} \rVert \le 1\right)`,
    index: String.raw`\pi \approx \frac{4}{N} \sum_n \mathbb{1}\!\left[x_n^2 + y_n^2 \le 1\right]`,
    code: `4 * (x**2 + y**2 <= 1).float().mean()`,
  },
};

/** The operation in two notations side by side (stacked on phones), as on the Concept page, plus its PyTorch spelling. */
function EquationCard({ id }: { id: string }) {
  const T = usePageText();
  const eq = EQ[id];
  if (!eq) return null;
  const h3 = "text-xs font-semibold uppercase tracking-wide text-gray-500";
  return (
    <div className="mt-4 rounded-lg border border-gray-200 bg-white px-4 py-3" data-testid={`loops-eq-${id}`}>
      <div className="grid gap-x-6 gap-y-3 md:grid-cols-2">
        <div className="min-w-0">
          <h3 className={h3}>{T("loops.eq.vector")}</h3>
          <div className="mt-1 text-base">
            <Tex tex={eq.vector} display />
          </div>
        </div>
        <div className="min-w-0">
          <h3 className={h3}>{T("loops.eq.index")}</h3>
          <div className="mt-1 text-base">
            <Tex tex={eq.index} display />
          </div>
          {eq.note && (
            <p className="text-center text-xs text-gray-600">
              <RichText text={T(`loops.eq.${id}.note` as PageKey)} />
            </p>
          )}
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-2 gap-y-1 border-t border-gray-100 pt-2 text-xs">
        <span className="text-gray-500">{T("loops.eq.code")}</span>
        <code className="font-mono break-all text-blue-800">{eq.code}</code>
      </div>
    </div>
  );
}

/** Hardware that accelerates each operation: page-text key of the chip badge's hover text. */
const HW: Record<string, PageKey> = {
  dot: "loops.hw.dot",
  pairwise: "loops.hw.pairwise",
  matmul: "loops.hw.matmul",
  diffusion: "loops.hw.diffusion",
  mc_pi: "loops.hw.mc_pi",
};

/** Dedicated matrix-multiply units by vendor: [vendor, unit, since, precisions, reached from PyTorch via]. */
const GEMM_UNITS: string[][] = [
  ["NVIDIA", "Tensor Cores", "Volta V100 (2017); GeForce: Turing RTX 20 (2018)", "FP16, INT8; BF16, TF32 (Ampere+); FP8 (Ada/Hopper+); FP4 (Blackwell)", "cuBLAS / cuDNN"],
  ["AMD", "Matrix Cores (MFMA); WMMA", "CDNA MI100 (2020); RDNA 3 (2022)", "FP16, BF16, INT8; FP8 (MI300)", "rocBLAS / hipBLASLt (ROCm)"],
  ["Apple", "AMX (CPU); GPU", "M1 (2020)", "FP32, FP16; BF16 (M2+)", "Accelerate / MPS"],
  ["Intel", "AMX (Xeon CPU)", "Sapphire Rapids (2023)", "BF16, INT8; FP16 (Granite Rapids)", "oneDNN"],
  ["Intel", "XMX (Arc GPU)", "Arc Alchemist (2022)", "FP16, BF16, INT8", "oneDNN / XPU"],
  ["Arm", "SME / SME2", "Armv9-A; e.g. Apple M4 (2024)", "FP32, FP16, BF16, INT8", "KleidiAI"],
  ["Google", "TPU MXU (systolic array)", "TPU v1 (2016)", "INT8 (v1); BF16 (v2+), FP32 accumulate", "XLA (JAX / PyTorch/XLA)"],
];

/** The vendor table of the matrix-multiply card: scrolls sideways inside its own box on phones. */
function GemmTable() {
  const T = usePageText();
  const head = ["vendor", "unit", "since", "prec", "via"] as const;
  return (
    <details className="group mt-4 rounded-lg border border-gray-200 bg-white" data-testid="gemm-table">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2 text-sm font-medium text-gray-800 [&::-webkit-details-marker]:hidden">
        <span className="text-xs text-gray-500 group-open:hidden" aria-hidden>▸</span>
        <span className="hidden text-xs text-gray-500 group-open:inline" aria-hidden>▾</span>
        {T("loops.hw.table.title")}
      </summary>
      <div className="overflow-x-auto border-t border-gray-100">
        <table className="w-full min-w-[40rem] text-left text-xs">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              {head.map((h) => (
                <th key={h} className="px-3 py-1.5 font-semibold">
                  {T(`loops.hw.table.${h}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 text-gray-700">
            {GEMM_UNITS.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} className={`px-3 py-1.5 align-top ${j === 0 ? "font-medium text-gray-900" : ""} ${j === 4 ? "font-mono" : ""}`}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-1 border-t border-gray-100 px-4 py-2 text-xs leading-relaxed text-gray-600">
        <p>
          <RichText text={T("loops.hw.matmul.why")} />
        </p>
      </div>
    </details>
  );
}

/** Every section of the tab, in order: numbered 1..N in the table of contents and on the cards. */
const SECTIONS = ["why", ...CASES.map((c) => c.id), "fine", "bench"];
const num = (key: string) => SECTIONS.indexOf(key) + 1;

/** Card heading with the numbered blue circle, as on the Concept page. */
function Heading({ n, text, extra }: { n: number; text: string; extra?: ReactNode }) {
  return (
    <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
        {n}
      </span>
      <RichText text={text} />
      {extra}
    </h2>
  );
}

const fill = (s: string, v: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (m, k) => String(v[k] ?? m));

export default function LoopsTab({ title }: { title: (id: string) => string | undefined }) {
  const T = usePageText();
  const m = B.machine;
  const load = (m.cpu_load_percent ?? []).filter((x): x is number => x != null);
  const underLoad = load.length > 0 && Math.max(...load) >= 50;

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
      <PageToc
        scrollRoot="skills-scroll"
        items={SECTIONS.map((key, i) => ({ id: `loops-${key}`, mark: i + 1, label: T(`loops.${key}.title` as PageKey) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10">
        <header>
          <p className="text-sm font-medium text-blue-700">{T("skills.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("loops.title")}</h1>
          <p className="mt-3 text-gray-600">
            <RichText text={T("loops.intro")} />
          </p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">
            <RichText text={T("loops.howto")} />
          </p>
        </header>

        <section id="loops-why" className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6">
          <Heading n={num("why")} text={T("loops.why.title")} />
          <ul className="mt-3 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-2">
            {(["interp", "memory", "batch", "gpu"] as const).map((k) => (
              <li key={k} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`loops.why.${k}` as PageKey)} />
              </li>
            ))}
          </ul>
        </section>

        {CASES.map((c) => {
          const code = B.code[c.id];
          const links = c.usedIn.filter((id) => title(id));
          return (
            <section
              key={c.id}
              id={`loops-${c.id}`}
              className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6"
              data-testid={`loops-case-${c.id}`}
            >
              <Heading
                n={num(c.id)}
                text={T(`loops.${c.id}.title` as PageKey)}
                extra={
                  HW[c.id] && (
                    <HwChip
                      label={T("loops.hw.label")}
                      text={T(HW[c.id]) ?? ""}
                    />
                  )
                }
              />
              <p className="mt-3 text-sm leading-relaxed text-gray-700">
                <RichText text={T(`loops.${c.id}.text` as PageKey)} />
              </p>
              {c.id === "matmul" && <GemmTable />}
              <EquationCard id={c.id} />
              {code && (
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <CodeBlock code={code.loop} title={T("loops.loop")} />
                  <CodeBlock code={code.tensor} title={T("loops.tensor")} />
                </div>
              )}
              <LoopVsTensorAnim id={c.id as AnimCase} />
              <SpeedupChart id={c.id} />
              {links.length > 0 && (
                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                  <span className="text-gray-500">{T("loops.seeIt")}</span>
                  {links.map((id) => (
                    <a
                      key={id}
                      href={`#/${id}`}
                      className="rounded-full bg-blue-50 px-2.5 py-1 text-blue-700 hover:bg-blue-100"
                    >
                      <RichText text={title(id)!} />
                    </a>
                  ))}
                </div>
              )}
            </section>
          );
        })}

        <section id="loops-fine" className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6">
          <Heading n={num("fine")} text={T("loops.fine.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("loops.fine.text")} />
          </p>
          <div className="mt-4">
            <CodeBlock
              code={`y = y0                      # (B, 2): every pendulum
for t in range(n_steps):    # the loop over time stays: step t+1 needs step t
    y = rk4_step(f, y, t * dt, dt)   # the loop over B is gone`}
            />
          </div>
          <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-sm leading-relaxed text-amber-900">
            <RichText text={T("loops.fine.small")} />
          </p>
        </section>

        <section id="loops-bench" className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6 text-sm">
          <Heading n={num("bench")} text={T("loops.bench.title")} />
          <p className="mt-3 leading-relaxed text-gray-700">
            {fill(T("loops.machine"), {
              cpu: m.cpu,
              threads: m.threads,
              gpu: m.gpu ? fill(T("loops.machineGpu"), { gpu: m.gpu }) : "",
              torch: m.torch,
              date: m.date,
              passes: m.passes,
            })}
          </p>
          {underLoad && (
            <p className="mt-2 leading-relaxed text-amber-800">
              {fill(T("loops.underLoad"), { lo: Math.round(Math.min(...load)), hi: Math.round(Math.max(...load)) })}
            </p>
          )}
          {m.gpu && /GTX 10\d0/.test(m.gpu) && (
            <p className="mt-2 leading-relaxed text-gray-700">{fill(T("loops.pascalGpu"), { gpu: m.gpu })}</p>
          )}
          <p className="mt-2 leading-relaxed text-gray-700">{T("loops.caveat")}</p>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-gray-900 px-3 py-2 font-mono text-xs text-gray-100">
            pixi run -e cuda python scripts/loop_benchmark.py
          </pre>
        </section>
      </article>
    </div>
  );
}
