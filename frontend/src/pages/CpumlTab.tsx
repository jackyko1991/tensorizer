import type { ReactNode } from "react";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import bench from "../content/cpuml_bench.json";
import { usePageText, type PageKey } from "../content/pages";
import { barSpan, fmtMetric, fmtTick, fmtTime, fmtX, speedupPos, ticks } from "../cpumlChart";

/** One measured size of a case in cpuml_bench.json (written by scripts/cpuml_benchmark.py). */
interface BenchRow {
  n: number;
  sklearn_s: number | null;
  tensor_cpu_s: number | null;
  tensor_gpu_s: number | null;
  speedup_cpu: number | null;
  speedup_gpu: number | null;
  speedup_cpu_range: number[] | null;
  speedup_gpu_range: number[] | null;
}
type Metrics = Record<string, number | null>;
interface UmapRow {
  trust: number;
  cont: number;
  recall: number;
  acc: number;
  seconds: number | null;
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
    note?: string;
  };
  sklearn: { sklearn: string; numpy: string; scipy: string; blas: string; date: string; cpu_load_percent?: (number | null)[] } | null;
  settings: { pca_k: number; svm_C: number; ksvm_C: number; logreg_C: number; knn_k: number; kmeans_k: number; mnist: number[] };
  units: Record<string, string>;
  code: Record<string, { sklearn: string; tensor: string }>;
  agreement: Record<string, { sklearn?: Metrics; cpu?: Metrics; cuda?: Metrics }>;
  cases: Record<string, BenchRow[]>;
  umap: { rows: Record<string, UmapRow>; umap_learn_first_call_s: number; procrustes: number; versions: string };
}

const B = bench as unknown as Bench;

/** The six measured migrations, in page order. */
const CASES = ["pca", "linsvm", "ksvm", "logreg", "knn", "kmeans"] as const;
type Case = (typeof CASES)[number];

/** Every section of the tab, in order: numbered 1..N in the table of contents and on the cards. */
const SECTIONS = ["why", "pca", "linsvm", "ksvm", "logreg", "knn", "kmeans", "umap", "cuml", "bench"] as const;
type Section = (typeof SECTIONS)[number];
const num = (key: Section) => SECTIONS.indexOf(key) + 1;

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

function Card({ id, children }: { id: Section; children: ReactNode }) {
  return (
    <section id={`cpuml-${id}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6" data-testid={`cpuml-${id}`}>
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

/** A folded box (closed by default), as the tables on the other tabs. */
function Fold({ title, testid, children }: { title: string; testid: string; children: ReactNode }) {
  return (
    <details className="group mt-4 rounded-lg border border-gray-200 bg-white" data-testid={testid}>
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2 text-sm font-medium text-gray-800 [&::-webkit-details-marker]:hidden">
        <span className="text-xs text-gray-500 group-open:hidden" aria-hidden>▸</span>
        <span className="hidden text-xs text-gray-500 group-open:inline" aria-hidden>▾</span>
        <RichText text={title} />
      </summary>
      {children}
    </details>
  );
}

// --- the maths of each case ---------------------------------------------------------------------------------

const EQ: Record<Case, { vector: string; index: string; code: string }> = {
  pca: {
    vector: String.raw`X_c = X - \mathbf{1}\,\bar{\mathbf{x}}^\top = U \Sigma V^\top, \quad \mathrm{EVR}_k = \frac{\sigma_k^2}{\sum_j \sigma_j^2}`,
    index: String.raw`\bar x_d = \tfrac{1}{N}\sum_n x_{nd}, \quad C_{de} = \tfrac{1}{N-1}\sum_n (x_{nd} - \bar x_d)(x_{ne} - \bar x_e)`,
    code: `Xc = X - X.mean(dim=0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)`,
  },
  linsvm: {
    vector: String.raw`\min_W\ \tfrac12 \lVert W \rVert_F^2 + C \sum_n \bigl\lVert \max\bigl(0,\ \mathbf{1} - \mathbf{y}_n \circ W^\top \tilde{\mathbf{x}}_n\bigr) \bigr\rVert^2`,
    index: String.raw`J = \tfrac12 \sum_{dk} W_{dk}^2 + C \sum_{nk} \max\Bigl(0,\ 1 - Y_{nk} \textstyle\sum_d \tilde X_{nd} W_{dk}\Bigr)^2`,
    code: `hinge = (1 - Y * (Xb @ W)).clamp(min=0)  # (N, K): all K one-vs-rest SVMs`,
  },
  ksvm: {
    vector: String.raw`\max_{\boldsymbol\alpha}\ \mathbf{1}^\top \boldsymbol\alpha - \tfrac12 \boldsymbol\alpha^\top Q\, \boldsymbol\alpha, \quad 0 \le \alpha_i \le C,\ \ \mathbf{y}^\top \boldsymbol\alpha = 0`,
    index: String.raw`K_{ij} = e^{-\gamma \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2}, \quad Q_{ij} = y_i\, y_j\, K_{ij}, \quad \alpha_i \leftarrow \Pi\Bigl(\alpha_i + \tfrac{1}{L}\bigl(1 - \textstyle\sum_j Q_{ij}\alpha_j\bigr)\Bigr)`,
    code: `K = torch.exp(-gamma * (sq[:, None] + sq[None, :] - 2 * X @ X.T))`,
  },
  logreg: {
    vector: String.raw`\min_{W, \mathbf{b}}\ \tfrac12 \lVert W \rVert_F^2 + C \sum_n -\log \operatorname{softmax}\bigl(W^\top \mathbf{x}_n + \mathbf{b}\bigr)_{y_n}, \quad C = \frac{1}{\lambda}`,
    index: String.raw`\frac{\partial J}{\partial W_{dk}} = W_{dk} + C \sum_n x_{nd}\,\bigl(p_{nk} - \delta_{y_n k}\bigr), \quad W \leftarrow W - \eta\, \nabla_W J`,
    code: `loss = 0.5 * (W**2).sum() + C * F.cross_entropy(X @ W + b, y, reduction="sum")`,
  },
  knn: {
    vector: String.raw`\hat y(\mathbf{q}) = \operatorname{mode}\bigl\{\, y_n : n \in \operatorname{top}_k\bigl(-\lVert \mathbf{q} - \mathbf{x}_n \rVert\bigr) \bigr\}`,
    index: String.raw`D_{mn} = \sqrt{\textstyle\sum_d (q_{md} - x_{nd})^2}, \quad v_{mc} = \sum_{j=1}^{k} \bigl[\, y_{\mathrm{idx}_{mj}} = c \,\bigr]`,
    code: `idx = torch.cdist(Xte, Xtr).topk(k, largest=False).indices`,
  },
  kmeans: {
    vector: String.raw`\min_{\{\boldsymbol\mu_j\},\, \{c_n\}}\ \sum_n \bigl\lVert \mathbf{x}_n - \boldsymbol\mu_{c_n} \bigr\rVert^2`,
    index: String.raw`c_n = \arg\min_j \sum_d (x_{nd} - \mu_{jd})^2, \quad \mu_{jd} = \frac{\sum_n [c_n = j]\, x_{nd}}{\sum_n [c_n = j]}`,
    code: `sums = torch.zeros_like(centers).index_add_(0, labels, X)`,
  },
};

function EquationCard({ id }: { id: Case }) {
  const T = usePageText();
  const eq = EQ[id];
  const h3 = "text-xs font-semibold tracking-wide text-gray-500"; // no uppercase: it would change the TeX
  return (
    <div className="mt-4 rounded-lg border border-gray-200 bg-white px-4 py-3" data-testid={`cpuml-eq-${id}`}>
      <div className="grid gap-x-6 gap-y-3 md:grid-cols-2">
        <div className="min-w-0">
          <h3 className={h3}>{T("cpuml.eq.vector")}</h3>
          <div className="mt-1 text-base">
            <Tex tex={eq.vector} display />
          </div>
        </div>
        <div className="min-w-0">
          <h3 className={h3}>{T("cpuml.eq.index")}</h3>
          <div className="mt-1 text-base">
            <Tex tex={eq.index} display />
          </div>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-2 gap-y-1 border-t border-gray-100 pt-2 text-xs">
        <span className="text-gray-500">{T("cpuml.eq.code")}</span>
        <code className="font-mono break-all text-blue-800">{eq.code}</code>
      </div>
    </div>
  );
}

// --- agreement on MNIST -------------------------------------------------------------------------------------

type Kind = "pct" | "num" | "sci" | "int" | "time";
/** Metrics shown per case; `cmp` metrics compare with scikit-learn's result, so its own column is empty. */
const METRICS: Record<Case, { key: string; kind: Kind; cmp?: boolean }[]> = {
  pca: [
    { key: "evr_sum", kind: "num" },
    { key: "evr1", kind: "num" },
    { key: "evr_maxdiff", kind: "sci", cmp: true },
    { key: "min_cos10", kind: "num", cmp: true },
    { key: "seconds", kind: "time" },
  ],
  linsvm: [
    { key: "test_acc", kind: "pct" },
    { key: "objective", kind: "num" },
    { key: "pred_match", kind: "pct", cmp: true },
    { key: "coef_rel_diff", kind: "sci", cmp: true },
    { key: "seconds", kind: "time" },
  ],
  ksvm: [
    { key: "test_acc", kind: "pct" },
    { key: "dual", kind: "num" },
    { key: "n_sv", kind: "int" },
    { key: "pred_match", kind: "pct", cmp: true },
    { key: "seconds", kind: "time" },
  ],
  logreg: [
    { key: "test_acc", kind: "pct" },
    { key: "objective", kind: "num" },
    { key: "pred_match", kind: "pct", cmp: true },
    { key: "coef_rel_diff", kind: "sci", cmp: true },
    { key: "seconds", kind: "time" },
  ],
  knn: [
    { key: "test_acc", kind: "pct" },
    { key: "pred_match", kind: "pct", cmp: true },
    { key: "seconds", kind: "time" },
  ],
  kmeans: [
    { key: "inertia", kind: "num" },
    { key: "label_match", kind: "pct", cmp: true },
    { key: "seconds", kind: "time" },
  ],
};

function AgreementTable({ id }: { id: Case }) {
  const T = usePageText();
  const a = B.agreement[id];
  if (!a) return null;
  const cols = (["sklearn", "cpu", "cuda"] as const).filter((c) => a[c]);
  const [ntr, nte] = B.settings.mnist;
  return (
    <div className="mt-4 overflow-hidden rounded-lg border border-gray-200" data-testid={`cpuml-agree-${id}`}>
      <div className="border-b border-gray-100 bg-gray-50 px-3 py-1.5 text-xs font-medium text-gray-700">
        <RichText text={fill(T("cpuml.agree.title"), { train: ntr, test: nte })} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[30rem] text-right text-xs tabular-nums">
          <thead className="text-gray-600">
            <tr>
              <th className="px-3 py-1.5 text-left font-semibold">{T("cpuml.agree.metric")}</th>
              {cols.map((c) => (
                <th key={c} className="px-3 py-1.5 font-semibold">
                  {T(`cpuml.agree.col.${c}` as PageKey)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 text-gray-700">
            {METRICS[id].map((mt) => (
              <tr key={mt.key}>
                <td className="px-3 py-1.5 text-left text-gray-900">
                  <RichText text={T(`cpuml.metric.${mt.key}` as PageKey)} />
                </td>
                {cols.map((c) => {
                  let v = a[c]?.[mt.key];
                  if (mt.key === "min_cos10" && v != null) v = Math.min(v, 1); // float32 rounding can pass 1
                  return (
                    <td key={c} className="px-3 py-1.5 font-mono">
                      {c === "sklearn" && mt.cmp ? "–" : fmtMetric(v, mt.kind)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// --- measured speed-ups -------------------------------------------------------------------------------------

/** A speed-up bar from 1x: right when the tensor code is faster than scikit-learn, left when slower. */
function Bar({ value, range, color, label }: { value: number; range: number[] | null; color: string; label: string }) {
  const { left, width } = barSpan(value);
  return (
    <div className="flex items-center gap-2" title={`${label}: ${fmtX(value)}${range ? ` (${fmtX(range[0])}–${fmtX(range[1])})` : ""}`}>
      <div className="relative h-3.5 flex-1 rounded-sm bg-gray-100">
        <div className="absolute inset-y-[-3px] w-px bg-gray-400" style={{ left: `${speedupPos(1)}%` }} />
        <div className={`absolute inset-y-0 rounded-sm ${color}`} style={{ left: `${left}%`, width: `${width}%` }} />
        {range && range.length === 2 && (
          // error bar |-|: the spread over passes, with end caps
          <div
            className="absolute inset-y-0.5 border-x-[1.5px] border-gray-800"
            style={{ left: `${speedupPos(range[0])}%`, width: `${speedupPos(range[1]) - speedupPos(range[0])}%` }}
            data-testid="cpuml-error-bar"
          >
            <div className="absolute inset-x-0 top-1/2 h-[1.5px] -translate-y-1/2 bg-gray-800" />
          </div>
        )}
      </div>
      <span className={`w-16 shrink-0 text-right font-mono text-xs tabular-nums ${value < 1 ? "text-amber-700" : "text-gray-700"}`}>
        {fmtX(value)}
      </span>
    </div>
  );
}

function SpeedupChart({ id }: { id: Case }) {
  const T = usePageText();
  const rows = B.cases[id] ?? [];
  if (!rows.some((r) => r.speedup_cpu != null)) return null;
  const unit = B.units[id] ?? "N";
  const gpu = rows.some((r) => r.speedup_gpu != null);
  const one = speedupPos(1);
  return (
    <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50/60 p-3 text-xs" data-testid={`cpuml-chart-${id}`}>
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-gray-600">
        <span className="font-medium text-gray-800">{T("cpuml.chart.title")}</span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-blue-600" /> {T("cpuml.chart.cpu")}
        </span>
        {gpu && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-600" /> {T("cpuml.chart.gpu")}
          </span>
        )}
      </div>
      <div className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-2">
        {rows.map((r) => (
          <div key={r.n} className="contents">
            <div className="whitespace-nowrap">
              <div className="font-mono text-gray-800">
                {unit} = {r.n.toLocaleString("en")}
              </div>
              {r.sklearn_s != null && (
                <div className="text-[11px] text-gray-500">
                  {T("cpuml.chart.skTime")}: {fmtTime(r.sklearn_s)}
                </div>
              )}
            </div>
            <div className="space-y-1">
              {r.speedup_cpu != null && (
                <Bar value={r.speedup_cpu} range={r.speedup_cpu_range} color="bg-blue-600" label={T("cpuml.chart.cpu")} />
              )}
              {r.speedup_gpu != null && (
                <Bar value={r.speedup_gpu} range={r.speedup_gpu_range} color="bg-emerald-600" label={T("cpuml.chart.gpu")} />
              )}
            </div>
          </div>
        ))}
        {/* axis under the bars */}
        <div />
        <div className="flex items-center gap-2">
          <div className="relative h-4 flex-1 text-[10px] text-gray-500">
            {ticks().map((v) => (
              <span key={v} className="absolute top-0 -translate-x-1/2 pt-1 leading-none" style={{ left: `${speedupPos(v)}%` }}>
                {fmtTick(v)}
              </span>
            ))}
          </div>
          <span className="w-16 shrink-0" />
        </div>
        <div />
        <div className="flex items-center gap-2 text-[10px] text-gray-500">
          <div className="relative h-3 flex-1">
            <span className="absolute right-[calc(100%-var(--one))] pr-1.5 whitespace-nowrap" style={{ ["--one" as string]: `${one}%` }}>
              ← {T("cpuml.chart.slower")}
            </span>
            <span className="absolute pl-1.5 whitespace-nowrap" style={{ left: `${one}%` }}>
              {T("cpuml.chart.faster")} →
            </span>
          </div>
          <span className="w-16 shrink-0" />
        </div>
      </div>
      <p className="mt-2 text-[11px] text-gray-500">{T("cpuml.chart.range")}</p>
    </div>
  );
}

/** Times of every case and size, folded. */
function TimesTable() {
  const T = usePageText();
  return (
    <Fold title={T("cpuml.bench.table")} testid="cpuml-times">
      <div className="overflow-x-auto border-t border-gray-100">
        <table className="w-full min-w-[34rem] text-right text-xs tabular-nums">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              <th className="px-3 py-1.5 text-left font-semibold">{T("cpuml.bench.case")}</th>
              <th className="px-3 py-1.5 font-semibold">N</th>
              <th className="px-3 py-1.5 font-semibold">{T("cpuml.agree.col.sklearn")}</th>
              <th className="px-3 py-1.5 font-semibold">{T("cpuml.agree.col.cpu")}</th>
              <th className="px-3 py-1.5 font-semibold">{T("cpuml.agree.col.cuda")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 text-gray-700">
            {CASES.flatMap((c) =>
              (B.cases[c] ?? []).map((r, i) => (
                <tr key={`${c}${r.n}`}>
                  <td className="px-3 py-1.5 text-left text-gray-900">{i === 0 ? T(`cpuml.${c}.short` as PageKey) : ""}</td>
                  <td className="px-3 py-1.5 font-mono">{r.n.toLocaleString("en")}</td>
                  <td className="px-3 py-1.5 font-mono">{r.sklearn_s != null ? fmtTime(r.sklearn_s) : "–"}</td>
                  <td className="px-3 py-1.5 font-mono">{r.tensor_cpu_s != null ? fmtTime(r.tensor_cpu_s) : "–"}</td>
                  <td className="px-3 py-1.5 font-mono">{r.tensor_gpu_s != null ? fmtTime(r.tensor_gpu_s) : "–"}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </Fold>
  );
}

// --- UMAP and cuML ------------------------------------------------------------------------------------------

const UMAP_ROWS = ["forces", "autograd", "umap_learn", "pca"] as const;
const UMAP_COLS = ["trust", "cont", "recall", "acc", "seconds"] as const;
const UMAP_STEPS = ["knn", "sigma", "union", "forces"] as const;

const CODE_CUML = `# 1. zero code change: run an unmodified scikit-learn / umap-learn / hdbscan script on the GPU
#    shell:    python -m cuml.accel train.py
#    Jupyter:  %load_ext cuml.accel        (first cell, before the other imports)

# 2. or call cuML's estimators directly: the scikit-learn API, arrays may live on the GPU
from cuml.decomposition import PCA
from cuml.svm import SVC
from cuml.linear_model import LogisticRegression
from cuml.neighbors import KNeighborsClassifier
from cuml.cluster import KMeans
from cuml.manifold import UMAP

emb = UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)`;

/** Official documentation (verified): name and address. */
const CUML_LINKS: { key: string; name: string; url: string }[] = [
  { key: "docs", name: "cuML documentation", url: "https://docs.nvidia.com/cuml/latest/" },
  { key: "accel", name: "cuml.accel (zero code change)", url: "https://docs.nvidia.com/cuml/latest/cuml-accel/" },
  { key: "compat", name: "cuml.accel: supported estimators", url: "https://docs.nvidia.com/cuml/latest/cuml-accel/compatibility/" },
  { key: "api", name: "cuML API reference", url: "https://docs.nvidia.com/cuml/latest/api/" },
  { key: "install", name: "RAPIDS / CUDA-X installation", url: "https://docs.nvidia.com/datascience/install/" },
  { key: "github", name: "rapidsai/cuml", url: "https://github.com/rapidsai/cuml" },
  { key: "sklearn", name: "scikit-learn", url: "https://scikit-learn.org/stable/" },
];

// --- the tab ------------------------------------------------------------------------------------------------

export default function CpumlTab({ title }: { title: (id: string) => string | undefined }) {
  const T = usePageText();
  const m = B.machine;
  const sk = B.sklearn;
  const load = [...(m.cpu_load_percent ?? []), ...(sk?.cpu_load_percent ?? [])].filter((x): x is number => x != null);
  const umapTitle = title("ai/umap_tensor");
  const s = B.settings;

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
      <PageToc
        scrollRoot="skills-scroll"
        items={SECTIONS.map((key, i) => ({ id: `cpuml-${key}`, mark: i + 1, label: T(`cpuml.${key}.title` as PageKey) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10">
        <header>
          <p className="text-sm font-medium text-blue-700">{T("skills.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("cpuml.title")}</h1>
          <p className="mt-3 text-gray-600">
            <RichText text={T("cpuml.intro")} />
          </p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">
            <RichText text={T("cpuml.howto")} />
          </p>
        </header>

        <Card id="why">
          <Heading n={num("why")} text={T("cpuml.why.title")} />
          <ul className="mt-3 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-2">
            {(["sklearn", "gain", "recipe", "honest"] as const).map((k) => (
              <li key={k} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`cpuml.why.${k}`)} />
              </li>
            ))}
          </ul>
        </Card>

        {CASES.map((c) => {
          const code = B.code[c];
          return (
            <Card key={c} id={c}>
              <Heading n={num(c)} text={T(`cpuml.${c}.title` as PageKey)} />
              <p className="mt-3 text-sm leading-relaxed text-gray-700">
                <RichText
                  text={fill(T(`cpuml.${c}.text` as PageKey), {
                    k: s.pca_k,
                    C: c === "linsvm" ? s.svm_C : c === "ksvm" ? s.ksvm_C : s.logreg_C,
                    knn: s.knn_k,
                    clusters: s.kmeans_k,
                    umap: num("umap"),
                  })}
                />
              </p>
              <EquationCard id={c} />
              {code && (
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <CodeBlock code={code.sklearn} title={T("cpuml.code.sklearn")} />
                  <CodeBlock code={code.tensor} title={T("cpuml.code.tensor")} />
                </div>
              )}
              <div className="mt-4 rounded-lg bg-gray-50 px-4 py-3 text-sm leading-relaxed text-gray-700">
                <h3 className="font-semibold text-gray-900">{T("cpuml.changes")}</h3>
                <p className="mt-1">
                  <RichText text={T(`cpuml.${c}.changes` as PageKey)} />
                </p>
              </div>
              <AgreementTable id={c} />
              <SpeedupChart id={c} />
              <p className="mt-3 text-sm leading-relaxed text-gray-700">
                <RichText text={T(`cpuml.${c}.result` as PageKey)} />
              </p>
            </Card>
          );
        })}

        <Card id="umap">
          <Heading n={num("umap")} text={T("cpuml.umap.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("cpuml.umap.text")} />
          </p>
          <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-sm leading-relaxed text-gray-700">
            {UMAP_STEPS.map((k) => (
              <li key={k}>
                <RichText text={T(`cpuml.umap.step.${k}`)} />
              </li>
            ))}
          </ol>
          <div className="mt-4 overflow-hidden rounded-lg border border-gray-200" data-testid="cpuml-umap-table">
            <div className="border-b border-gray-100 bg-gray-50 px-3 py-1.5 text-xs font-medium text-gray-700">
              <RichText text={T("cpuml.umap.table")} />
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[32rem] text-right text-xs tabular-nums">
                <thead className="text-gray-600">
                  <tr>
                    <th className="px-3 py-1.5 text-left font-semibold">{T("cpuml.umap.method")}</th>
                    {UMAP_COLS.map((k) => (
                      <th key={k} className="px-3 py-1.5 font-semibold">
                        {T(`cpuml.umap.col.${k}`)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 text-gray-700">
                  {UMAP_ROWS.map((r) => {
                    const row = B.umap.rows[r];
                    return (
                      <tr key={r}>
                        <td className="px-3 py-1.5 text-left text-gray-900">{T(`cpuml.umap.row.${r}`)}</td>
                        {(["trust", "cont", "recall", "acc"] as const).map((k) => (
                          <td key={k} className="px-3 py-1.5 font-mono">
                            {k === "recall" ? row[k].toFixed(2) : row[k].toFixed(3)}
                          </td>
                        ))}
                        <td className="px-3 py-1.5 font-mono">{row.seconds != null ? `${row.seconds} s` : "–"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText
              text={fill(T("cpuml.umap.verdict"), { first: B.umap.umap_learn_first_call_s, procrustes: B.umap.procrustes, versions: B.umap.versions })}
            />
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <span className="text-gray-500">{T("cpuml.umap.open")}</span>
            <a href="#/ai/umap_tensor" className="rounded-full bg-blue-50 px-2.5 py-1 text-blue-700 hover:bg-blue-100" data-testid="cpuml-umap-link">
              <RichText text={umapTitle ?? "UMAP"} />
            </a>
          </div>
        </Card>

        <Card id="cuml">
          <Heading n={num("cuml")} text={T("cpuml.cuml.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText text={T("cpuml.cuml.text")} />
          </p>
          <div className="mt-4">
            <CodeBlock code={CODE_CUML} title={T("cpuml.cuml.codeTitle")} />
          </div>
          <p className="mt-3 rounded-lg bg-amber-50 px-4 py-2.5 text-sm leading-relaxed text-amber-950" data-testid="cpuml-cuml-req">
            <RichText text={T("cpuml.cuml.req")} />
          </p>
          <div className="mt-4 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-2">
            <div className="rounded-lg bg-gray-50 p-3">
              <h3 className="font-semibold text-gray-900">{T("cpuml.cuml.useCumlTitle")}</h3>
              <p className="mt-1">
                <RichText text={T("cpuml.cuml.useCuml")} />
              </p>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <h3 className="font-semibold text-gray-900">{T("cpuml.cuml.useTensorTitle")}</h3>
              <p className="mt-1">
                <RichText text={T("cpuml.cuml.useTensor")} />
              </p>
            </div>
          </div>
          <Fold title={T("cpuml.cuml.links")} testid="cpuml-cuml-links">
            <ul className="divide-y divide-gray-100 border-t border-gray-100 text-xs text-gray-700">
              {CUML_LINKS.map((l) => (
                <li key={l.key} className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-4 py-2">
                  <span className="font-medium">
                    <Ext href={l.url}>{l.name}</Ext>
                  </span>
                  <span className="text-gray-500">
                    <RichText text={T(`cpuml.cuml.link.${l.key}` as PageKey)} />
                  </span>
                </li>
              ))}
            </ul>
          </Fold>
        </Card>

        <Card id="bench">
          <Heading n={num("bench")} text={T("cpuml.bench.title")} />
          <p className="mt-3 text-sm leading-relaxed text-gray-700">
            <RichText
              text={fill(T("cpuml.machine"), {
                cpu: m.cpu,
                threads: m.threads,
                gpu: m.gpu ? fill(T("cpuml.machineGpu"), { gpu: m.gpu }) : "",
                torch: m.torch,
                date: m.date,
                passes: m.passes,
              })}
            />
          </p>
          {sk && (
            <p className="mt-2 text-sm leading-relaxed text-gray-700">
              <RichText text={fill(T("cpuml.skMachine"), { sklearn: sk.sklearn, numpy: sk.numpy, scipy: sk.scipy, blas: sk.blas, date: sk.date })} />
            </p>
          )}
          {load.length > 0 && Math.max(...load) >= 10 && (
            <p className="mt-2 text-sm leading-relaxed text-amber-800">
              {fill(T("cpuml.load"), { lo: Math.round(Math.min(...load)), hi: Math.round(Math.max(...load)) })}
            </p>
          )}
          <TimesTable />
          <p className="mt-4 text-sm leading-relaxed text-gray-700">
            <RichText text={T("cpuml.bench.observe")} />
          </p>
          <p className="mt-2 text-sm leading-relaxed text-gray-700">{T("cpuml.caveat")}</p>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-gray-900 px-3 py-2 font-mono text-xs text-gray-100">
            {`pixi exec --spec scikit-learn --spec numpy python scripts/cpuml_benchmark.py --sklearn-only
pixi run -e cuda python scripts/cpuml_benchmark.py`}
          </pre>
        </Card>
      </article>
    </div>
  );
}
