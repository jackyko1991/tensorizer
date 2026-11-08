import { useEffect, useRef, useState } from "react";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import bench from "../content/blas_bench.json";
import { usePageText, type PageKey } from "../content/pages";
import { attainable, bestChain, chainCost, intensity, kernelCost, orderText, ridge, type Order } from "../tensorMaths";
import MathsAnim from "../components/MathsAnim";
import TensorPiecesDiagram from "../components/TensorPiecesDiagram";
import geometry from "../content/tensorGeometry.json";
import { useLang } from "../i18n";
import { animsAfter, type MathsCase } from "../mathsAnim";

// Sections in order; the address #/concept/maths/<key> scrolls to id maths-<key>.
export const MATHS_SECTIONS = ["tensor", "einsum", "penrose", "topo", "roof", "recap"] as const;
type Section = (typeof MATHS_SECTIONS)[number];
export const FEYNMAN = ["plain", "analogy", "precise", "example"] as const;

// Equation cards: the formula, the PyTorch line that computes it (not translated) and its animation.
const EQS: Partial<Record<Section, { tex: string; code: string; anim: MathsCase }[]>> = {
  tensor: [
    { tex: String.raw`T_{i_1 i_2 \dots i_r}, \quad r = \text{rank}`, code: "x = torch.zeros(32, 480, 640, 3)   # x.ndim == 4, x.shape == (32, 480, 640, 3)", anim: "eqRank" },
    { tex: String.raw`v'_i = R_{ij} v_j, \quad A'_{ij} = R_{ik} R_{jl} A_{kl}`, code: "v2 = R @ v          # one R per index\nA2 = R @ A @ R.T    # two indices, two R's", anim: "eqRotate" },
  ],
  einsum: [
    { tex: String.raw`s = x_i y_i`, code: 'torch.einsum("i,i->", x, y)        # dot product', anim: "eqDot" },
    { tex: String.raw`y_i = A_{ij} x_j`, code: 'torch.einsum("ij,j->i", A, x)      # matrix-vector', anim: "eqMatvec" },
    { tex: String.raw`C_{ik} = A_{ij} B_{jk}`, code: 'torch.einsum("ij,jk->ik", A, B)    # matrix product, A @ B', anim: "eqMatmul" },
    { tex: String.raw`t = A_{ii}`, code: 'torch.einsum("ii->", A)            # trace', anim: "eqTrace" },
    { tex: String.raw`P_{ij} = u_i v_j`, code: 'torch.einsum("i,j->ij", u, v)      # outer product: nothing summed', anim: "eqOuter" },
    { tex: String.raw`C_{bik} = A_{bij} B_{bjk}`, code: 'torch.einsum("bij,bjk->bik", A, B)  # a batch of products, b is free', anim: "eqBatch" },
  ],
  penrose: [
    { tex: String.raw`\operatorname{tr}(AB) = A_{ij} B_{ji} = B_{ji} A_{ij} = \operatorname{tr}(BA)`, code: 'torch.einsum("ij,ji->", A, B)      # one loop through A and B', anim: "eqLoop" },
    { tex: String.raw`T_{s_1 s_2 s_3} = A_{s_1 \alpha} B_{\alpha s_2 \beta} C_{\beta s_3}`, code: 'T = torch.einsum("ia,ajb,bk->ijk", A, B, C)   # a 3-site MPS', anim: "eqMps" },
  ],
  topo: [
    { tex: String.raw`\text{cost}\big((AB)C\big) = abc + acd, \qquad \text{cost}\big(A(BC)\big) = bcd + abd`, code: 'torch.einsum("ij,jk,kl->il", A, B, C)   # opt_einsum chooses the order\ntorch.linalg.multi_dot([A, B, C])        # cheapest bracketing of a chain', anim: "eqChain" },
  ],
  roof: [
    { tex: String.raw`P_\text{att} = \min(P,\ \beta I), \qquad I = \frac{\text{FLOPs}}{\text{bytes moved}}, \qquad I^* = \frac{P}{\beta}`, code: "f = torch.compile(lambda x: torch.relu(a * x + b))   # one fused kernel instead of three", anim: "eqRoof" },
  ],
};

// External references, opened in a new tab (titles not translated).
const LINKS: Partial<Record<Section, { label: string; url?: string }[]>> = {
  tensor: [
    { label: "Wikipedia: Tensor", url: "https://en.wikipedia.org/wiki/Tensor" },
    { label: "Wikipedia: Covariance and contravariance of vectors", url: "https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors" },
    { label: "PyTorch: torch.Tensor", url: "https://docs.pytorch.org/docs/stable/tensors.html" },
  ],
  einsum: [
    { label: "Wikipedia: Einstein notation", url: "https://en.wikipedia.org/wiki/Einstein_notation" },
    { label: "PyTorch: torch.einsum", url: "https://docs.pytorch.org/docs/stable/generated/torch.einsum.html" },
    { label: "NumPy: numpy.einsum", url: "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html" },
  ],
  penrose: [
    { label: "R. Penrose (1971), Applications of negative dimensional tensors, in Combinatorial Mathematics and its Applications, Academic Press" },
    { label: "J. C. Bridgeman & C. T. Chubb (2017), Hand-waving and interpretive dance: an introductory course on tensor networks, J. Phys. A 50, 223001 (arXiv:1603.03039)", url: "https://arxiv.org/abs/1603.03039" },
    { label: "Wikipedia: Penrose graphical notation", url: "https://en.wikipedia.org/wiki/Penrose_graphical_notation" },
  ],
  topo: [
    { label: "Wikipedia: Matrix chain multiplication", url: "https://en.wikipedia.org/wiki/Matrix_chain_multiplication" },
    { label: "D. G. A. Smith & J. Gray (2018), opt_einsum, J. Open Source Softw. 3, 753", url: "https://doi.org/10.21105/joss.00753" },
    { label: "PyTorch: torch.linalg.multi_dot", url: "https://docs.pytorch.org/docs/stable/generated/torch.linalg.multi_dot.html" },
  ],
  roof: [
    { label: "S. Williams, A. Waterman & D. Patterson (2009), Roofline: an insightful visual performance model for multicore architectures, Commun. ACM 52(4), 65–76", url: "https://doi.org/10.1145/1498765.1498785" },
    { label: "Wikipedia: Roofline model", url: "https://en.wikipedia.org/wiki/Roofline_model" },
    { label: "PyTorch: torch.compile", url: "https://docs.pytorch.org/docs/stable/generated/torch.compile.html" },
  ],
};

// colours shared by the diagrams
const C = { node: "#dbeafe", edge: "#2563eb", leg: "#1f2937", idx: "#c2410c", grid: "#e5e7eb", muted: "#6b7280" };

/** The Concept page's "Tensor maths" tab: tensors, einsum, Penrose diagrams, contraction order and the roofline. */
export default function MathsConcept() {
  const [tensorView, setTensorView] = useState<"array" | "geometry">("array");
  const T = usePageText();
  const R = (key: PageKey) => <RichText text={T(key)} />;
  const k = (s: string) => `maths.${s}` as PageKey;

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
      <PageToc
        scrollRoot="concept-scroll"
        items={MATHS_SECTIONS.map((s, i) => ({ id: `maths-${s}`, mark: i + 1, label: T(k(`${s}.title`)) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10" data-testid="maths-concept">
        <header>
          <p className="text-sm font-medium text-blue-700">{T("maths.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("maths.title")}</h1>
          <p className="mt-3 text-gray-600">{R("maths.intro")}</p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">
            {R("maths.feynman")}
          </p>
        </header>

        {MATHS_SECTIONS.map((s, i) => (
          <section key={s} id={`maths-${s}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6">
            <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
                {i + 1}
              </span>
              {T(k(`${s}.title`))}
            </h2>
            <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
              {s === "tensor" && <div role="tablist" aria-label={T("maths.tensor.views")} className="flex border-b border-gray-200">
                {(["array", "geometry"] as const).map(view=><button key={view} type="button" role="tab" id={`tensor-view-${view}`} aria-selected={tensorView===view} aria-controls="tensor-view-panel" tabIndex={tensorView===view?0:-1}
                  onClick={()=>setTensorView(view)} onKeyDown={e=>{
                    if (["ArrowLeft","ArrowRight","Home","End"].includes(e.key)) {
                      e.preventDefault(); const next=e.key==="Home"?"array":e.key==="End"?"geometry":tensorView==="array"?"geometry":"array";
                      setTensorView(next); document.getElementById(`tensor-view-${next}`)?.focus();
                    }
                  }} className={`border-b-2 px-4 py-2 text-sm font-medium ${tensorView===view?"border-blue-600 text-blue-700":"border-transparent text-gray-500 hover:text-gray-800"}`}>
                  {T(view==="array"?"maths.tensor.arrayTab":"maths.tensor.geometryTab")}
                </button>)}
              </div>}
              <div id={s==="tensor"?"tensor-view-panel":undefined} role={s==="tensor"?"tabpanel":undefined} aria-labelledby={s==="tensor"?`tensor-view-${tensorView}`:undefined} className="space-y-4">
              {s === "tensor" && tensorView === "geometry" ? <TensorGeometry /> : s === "recap" ? (
                <Recap />
              ) : (
                <>
                  {FEYNMAN.map((f) => (
                    <div key={f} className="space-y-4">
                      <Feynman kind={f} text={T(k(`${s}.${f}`))} label={T(k(`label.${f}`))} />
                      {animsAfter(s, f).map((id) => (
                        <MathsAnim key={id} id={id} />
                      ))}
                      {s === "tensor" && f === "precise" && (
                        <>
                          <p>{R("maths.tensor.updown")}</p>
                          {animsAfter(s, "updown").map((id) => (
                            <MathsAnim key={id} id={id} />
                          ))}
                        </>
                      )}
                      {s === "tensor" && f === "plain" && <RankFigure />}
                      {s === "einsum" && f === "precise" && <MatmulFigure />}
                      {s === "penrose" && f === "precise" && <PenroseFigures />}
                      {s === "topo" && f === "plain" && <DagFigure />}
                      {s === "topo" && f === "example" && <ChainTable />}
                      {s === "roof" && f === "precise" && (
                        <>
                          <Roofline />
                          <p>{R("maths.roof.ops")}</p>
                        </>
                      )}
                      {s === "roof" && f === "example" && (
                        <>
                          <p>{R("maths.roof.fusion")}</p>
                          {animsAfter(s, "fusion").map((id) => (
                            <MathsAnim key={id} id={id} />
                          ))}
                        </>
                      )}
                    </div>
                  ))}
                  {EQS[s] && <EqCards items={EQS[s]!} title={T("maths.label.code")} />}
                </>
              )}
              </div>
              {LINKS[s] && <Links items={LINKS[s]!} title={T("maths.label.links")} />}
            </div>
          </section>
        ))}
      </article>
    </div>
  );
}

function TensorGeometry() {
  const { lang } = useLang();
  const T = usePageText();
  const l = lang.startsWith("zh") ? 1 : 0;
  return <div className="space-y-4" data-testid="tensor-geometric-view">
    {FEYNMAN.map(f=><div key={f} className="space-y-4">
      <Feynman kind={f} text={geometry[f][l]} label={T(`maths.label.${f}`)} />
      {f==="analogy" && <TensorPiecesDiagram/>}
    </div>)}
    <figure className="overflow-hidden rounded-lg border border-gray-200 bg-gray-50">
      <figcaption className="px-4 pt-3 text-xs text-gray-600">{geometry.legend[l]}</figcaption>
      <div className="px-3 py-2"><Tex tex={String.raw`T=\sum_{i,j=1}^{2}T^i{}_j\,(e_i\otimes\varepsilon^j),\qquad (e_i\otimes\varepsilon^j)(v)=\varepsilon^j(v)e_i`} display/></div>
    </figure>
    <details className="rounded-lg border border-gray-200 bg-gray-50/50 p-3">
      <summary className="cursor-pointer text-sm font-medium text-blue-800">{geometry.partsTitle[l]}</summary>
      <dl className="mt-3 grid gap-x-4 gap-y-2 text-xs sm:grid-cols-[auto_1fr]">{geometry.parts.map(([name,role])=><div key={name[0]} className="contents"><dt className="font-semibold">{name[l]}</dt><dd className="text-gray-600">{role[l]}</dd></div>)}</dl>
      <p className="mt-3 text-xs text-gray-600">{geometry.note[l]}</p>
    </details>
    <a className="text-xs text-blue-700 hover:underline" href="https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S2.html" target="_blank" rel="noreferrer">David Tong: Differential Geometry, tensors and dual bases ↗</a>
  </div>;
}

const FEYNMAN_STYLE: Record<(typeof FEYNMAN)[number], string> = {
  plain: "",
  analogy: "rounded-lg border-l-4 border-amber-300 bg-amber-50 px-4 py-2.5 text-amber-950",
  precise: "",
  example: "rounded-lg border-l-4 border-emerald-300 bg-emerald-50/70 px-4 py-2.5 text-emerald-950",
};

export function Feynman({ kind, text, label }: { kind: (typeof FEYNMAN)[number]; text: string; label: string }) {
  return (
    <div className={FEYNMAN_STYLE[kind]} data-feynman={kind}>
      {label && <p className="mb-1 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">{label}</p>}
      <ListOrText text={text} />
    </div>
  );
}

/** "Intro: **(1)** a; **(2)** b; **(3)** c." becomes the intro and a numbered list; other text stays one paragraph. */
export function ListOrText({ text }: { text: string }) {
  const parts = text.split(/\s*\*\*\(\d+\)\*\*\s*/);
  if (parts.length < 3) {
    return (
      <p>
        <RichText text={text} />
      </p>
    );
  }
  const [intro, ...items] = parts;
  return (
    <div className="space-y-1.5">
      {intro.trim() && (
        <p>
          <RichText text={intro.trim()} />
        </p>
      )}
      <ol className="list-decimal space-y-1 pl-6 marker:font-semibold marker:text-gray-500">
        {items.map((it, i) => (
          <li key={i}>
            <RichText text={it.trim().replace(/[;；,，、.。।]$/, "")} />
          </li>
        ))}
      </ol>
    </div>
  );
}

function EqCards({ items, title }: { items: { tex: string; code: string; anim: MathsCase }[]; title: string }) {
  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">{title}</p>
      <div className="grid gap-3 md:grid-cols-2">
        {items.map((e, i) => {
          // a card without a partner (a single one, or an odd last one) spans the whole row
          const full = items.length % 2 === 1 && i === items.length - 1;
          return (
            <div key={e.tex} className={`overflow-hidden rounded-lg border border-gray-200 bg-white ${full ? "col-span-full" : ""}`}>
              <div className="bg-gray-50 px-2 py-1 text-base">
                <Tex tex={e.tex} display />
              </div>
              <CodeBlock code={e.code} />
              <div className="p-2">
                <MathsAnim id={e.anim} compact={!full} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function Links({ items, title }: { items: { label: string; url?: string }[]; title: string }) {
  return (
    <div className="border-t border-gray-100 pt-3">
      <p className="mb-1 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">{title}</p>
      <ul className="list-disc space-y-1 pl-5 text-xs">
        {items.map((l) => (
          <li key={l.label}>
            {l.url ? (
              <a href={l.url} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
                {l.label}
              </a>
            ) : (
              <span className="text-gray-700">{l.label}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Figure({ caption, children, testid }: { caption: string; children: React.ReactNode; testid?: string }) {
  return (
    <figure className="rounded-lg border border-gray-200 bg-gray-50/60 p-2 sm:p-3" data-testid={testid}>
      {children}
      <figcaption className="mt-2 text-center text-xs text-gray-600">
        <RichText text={caption} />
      </figcaption>
    </figure>
  );
}

/* ---------- 1. rank ---------- */

function Cells({ x, y, rows, cols, s = 14, hi }: { x: number; y: number; rows: number; cols: number; s?: number; hi?: (r: number, c: number) => string | undefined }) {
  const out = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      out.push(
        <rect key={`${r}-${c}`} x={x + c * s} y={y + r * s} width={s} height={s} fill={hi?.(r, c) ?? "#fff"} stroke={C.edge} strokeWidth="1" />,
      );
  return <g>{out}</g>;
}

function Arrow({ x1, y1, x2, y2, label, lx, ly }: { x1: number; y1: number; x2: number; y2: number; label: string; lx: number; ly: number }) {
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.idx} strokeWidth="1.2" markerEnd="url(#maths-arrow)" />
      <text x={lx} y={ly} fontSize="12" fontStyle="italic" fontFamily="serif" fill={C.idx} textAnchor="middle">
        {label}
      </text>
    </g>
  );
}

function ArrowDefs() {
  return (
    <defs>
      <marker id="maths-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill={C.idx} />
      </marker>
    </defs>
  );
}

function RankFigure() {
  const T = usePageText();
  const panels: { name: PageKey; shape: string; svg: React.ReactNode }[] = [
    { name: "maths.pn.scalar", shape: "()", svg: <Cells x={53} y={38} rows={1} cols={1} s={18} /> },
    {
      name: "maths.pn.vector",
      shape: "(4,)",
      svg: (
        <>
          <Cells x={34} y={40} rows={1} cols={4} />
          <Arrow x1={34} y1={66} x2={90} y2={66} label="i" lx={62} ly={80} />
        </>
      ),
    },
    {
      name: "maths.pn.matrix",
      shape: "(3, 4)",
      svg: (
        <>
          <Cells x={40} y={22} rows={3} cols={4} />
          <Arrow x1={28} y1={22} x2={28} y2={64} label="i" lx={18} ly={47} />
          <Arrow x1={40} y1={74} x2={96} y2={74} label="j" lx={68} ly={88} />
        </>
      ),
    },
    {
      name: "maths.pn.rank3",
      shape: "(2, 3, 4)",
      svg: (
        <>
          <Cells x={50} y={14} rows={3} cols={4} s={12} />
          <Cells x={42} y={22} rows={3} cols={4} s={12} />
          <Arrow x1={30} y1={22} x2={30} y2={58} label="j" lx={21} ly={44} />
          <Arrow x1={42} y1={66} x2={90} y2={66} label="k" lx={66} ly={80} />
          <Arrow x1={100} y1={58} x2={108} y2={50} label="i" lx={114} ly={62} />
        </>
      ),
    },
  ];
  return (
    <Figure caption={T("maths.tensor.fig")} testid="maths-rank">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {panels.map((p, r) => (
          <div key={p.name} className="text-center">
            <svg viewBox="0 0 130 92" className="mx-auto w-full max-w-[150px]" role="img" aria-label={T(p.name)}>
              <ArrowDefs />
              {p.svg}
            </svg>
            <p className="text-xs font-medium text-gray-800">
              {T(p.name)} · rank {r}
            </p>
            <p className="font-mono text-[11px] text-gray-500">shape {p.shape}</p>
          </div>
        ))}
      </div>
    </Figure>
  );
}

/* ---------- 2. matrix product ---------- */

function MatmulFigure() {
  const T = usePageText();
  const s = 16;
  const ri = 1; // highlighted row of A
  const ck = 2; // highlighted column of B
  const label = (x: number, y: number, t: string) => (
    <text x={x} y={y} fontSize="14" fontFamily="serif" fontStyle="italic" fill={C.leg} textAnchor="middle">
      {t}
    </text>
  );
  return (
    <Figure caption={T("maths.einsum.fig")} testid="maths-matmul">
      <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
      <svg viewBox="0 0 295 125" className="w-full max-w-xs" role="img" aria-label="C = A B">
        <ArrowDefs />
        {/* A: 3 x 4 */}
        <Cells x={30} y={40} rows={3} cols={4} s={s} hi={(r) => (r === ri ? "#bfdbfe" : undefined)} />
        {label(62, 30, "A")}
        <Arrow x1={30} y1={98} x2={94} y2={98} label="j" lx={62} ly={114} />
        <Arrow x1={20} y1={40} x2={20} y2={88} label="i" lx={10} ly={68} />
        <text x={115} y={72} fontSize="16" fill={C.muted} textAnchor="middle">×</text>
        {/* B: 4 x 3 */}
        <Cells x={135} y={24} rows={4} cols={3} s={s} hi={(_, c) => (c === ck ? "#bfdbfe" : undefined)} />
        {label(159, 16, "B")}
        <Arrow x1={125} y1={24} x2={125} y2={88} label="j" lx={115} ly={50} />
        <Arrow x1={135} y1={98} x2={183} y2={98} label="k" lx={159} ly={114} />
        <text x={199} y={72} fontSize="16" fill={C.muted} textAnchor="middle">=</text>
        {/* C: 3 x 3 */}
        <Cells x={235} y={40} rows={3} cols={3} s={s} hi={(r, c) => (r === ri && c === ck ? "#2563eb" : undefined)} />
        {label(259, 30, "C")}
        <Arrow x1={227} y1={40} x2={227} y2={88} label="i" lx={218} ly={68} />
        <Arrow x1={235} y1={98} x2={283} y2={98} label="k" lx={259} ly={114} />
      </svg>
      <div className="text-center text-sm">
        <Tex tex={String.raw`C_{ik} = \sum_j A_{ij} B_{jk}`} />
        <p className="mt-1 font-mono text-[11px] text-gray-500">torch.einsum("ij,jk-&gt;ik", A, B)</p>
      </div>
      </div>
    </Figure>
  );
}

/* ---------- 3. Penrose diagrams ---------- */

function PNode({ x, y, label, r = 13 }: { x: number; y: number; label: string; r?: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={C.node} stroke={C.edge} strokeWidth="1.5" />
      <text x={x} y={y + 4.5} fontSize="13" fontFamily="serif" fontStyle="italic" fill={C.leg} textAnchor="middle">
        {label}
      </text>
    </g>
  );
}

function Leg({ d, label, lx, ly }: { d: string; label?: string; lx?: number; ly?: number }) {
  return (
    <g>
      <path d={d} fill="none" stroke={C.leg} strokeWidth="2" />
      {label && (
        <text x={lx} y={ly} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">
          {label}
        </text>
      )}
    </g>
  );
}

interface Panel {
  name: PageKey;
  tex?: string;
  svg: React.ReactNode;
}

function PanelGrid({ panels, caption, testid, cols }: { panels: Panel[]; caption: string; testid: string; cols: string }) {
  const T = usePageText();
  return (
    <Figure caption={caption} testid={testid}>
      <div className={`grid gap-3 ${cols}`}>
        {panels.map((p) => (
          <div key={p.name} className="text-center">
            <svg viewBox="0 0 140 80" className="mx-auto w-full max-w-[170px]" role="img" aria-label={T(p.name)}>
              {p.svg}
            </svg>
            <p className="text-xs font-medium text-gray-800">{T(p.name)}</p>
            {p.tex && (
              <p className="text-xs text-gray-600">
                <Tex tex={p.tex} />
              </p>
            )}
          </div>
        ))}
      </div>
    </Figure>
  );
}

function PenroseFigures() {
  const T = usePageText();
  const basic: Panel[] = [
    { name: "maths.pn.scalar", tex: "s", svg: <PNode x={70} y={40} label="s" /> },
    {
      name: "maths.pn.vector",
      tex: "v_i",
      svg: (
        <>
          <Leg d="M70 40 H115" label="i" lx={108} ly={33} />
          <PNode x={70} y={40} label="v" />
        </>
      ),
    },
    {
      name: "maths.pn.matrix",
      tex: "A_{ij}",
      svg: (
        <>
          <Leg d="M25 40 H115" label="i" lx={32} ly={33} />
          <text x={108} y={33} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">j</text>
          <PNode x={70} y={40} label="A" />
        </>
      ),
    },
    {
      name: "maths.pn.rank3",
      tex: "T_{ijk}",
      svg: (
        <>
          <Leg d="M25 34 H115" label="i" lx={32} ly={27} />
          <Leg d="M70 34 V75" label="k" lx={78} ly={72} />
          <text x={108} y={27} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">j</text>
          <PNode x={70} y={34} label="T" />
        </>
      ),
    },
    {
      name: "maths.pn.identity",
      tex: String.raw`\delta_{ij}`,
      svg: (
        <>
          <Leg d="M25 40 H115" label="i" lx={32} ly={33} />
          <text x={108} y={33} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">j</text>
        </>
      ),
    },
  ];
  const ops: Panel[] = [
    {
      name: "maths.pn.dot",
      tex: "u_i v_i",
      svg: (
        <>
          <Leg d="M45 40 H95" label="i" lx={70} ly={33} />
          <PNode x={45} y={40} label="u" />
          <PNode x={95} y={40} label="v" />
        </>
      ),
    },
    {
      name: "maths.pn.matvec",
      tex: "A_{ij} v_j",
      svg: (
        <>
          <Leg d="M12 40 H100" label="i" lx={20} ly={33} />
          <text x={78} y={33} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">j</text>
          <PNode x={55} y={40} label="A" />
          <PNode x={102} y={40} label="v" />
        </>
      ),
    },
    {
      name: "maths.pn.matmul",
      tex: "A_{ij} B_{jk}",
      svg: (
        <>
          <Leg d="M8 40 H132" label="i" lx={15} ly={33} />
          <text x={70} y={33} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">j</text>
          <text x={125} y={33} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">k</text>
          <PNode x={45} y={40} label="A" />
          <PNode x={95} y={40} label="B" />
        </>
      ),
    },
    {
      name: "maths.pn.outer",
      tex: "u_i v_j",
      svg: (
        <>
          <Leg d="M50 52 V10" label="i" lx={58} ly={16} />
          <Leg d="M90 52 V10" label="j" lx={98} ly={16} />
          <PNode x={50} y={52} label="u" />
          <PNode x={90} y={52} label="v" />
        </>
      ),
    },
    {
      name: "maths.pn.trace",
      tex: "A_{ii}",
      svg: (
        <>
          <Leg d="M57 52 C 20 52, 30 8, 70 8 C 110 8, 120 52, 83 52" label="i" lx={70} ly={20} />
          <PNode x={70} y={52} label="A" />
        </>
      ),
    },
  ];
  const loop = (a: string, b: string, x0: number) => (
    <g>
      <path d={`M${x0} 45 C ${x0} 5, ${x0 + 90} 5, ${x0 + 90} 45 C ${x0 + 90} 85, ${x0} 85, ${x0} 45`} fill="none" stroke={C.leg} strokeWidth="2" />
      <PNode x={x0} y={45} label={a} />
      <PNode x={x0 + 90} y={45} label={b} />
      <text x={x0 + 45} y={10} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">j</text>
      <text x={x0 + 45} y={88} fontSize="12" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">i</text>
    </g>
  );
  const sites = [0, 1, 2, 3, 4, 5];
  return (
    <div className="space-y-4">
      <PanelGrid panels={basic} caption={T("maths.penrose.fig.basic")} testid="maths-pn-basic" cols="grid-cols-2 sm:grid-cols-5" />
      <PanelGrid panels={ops} caption={T("maths.penrose.fig.ops")} testid="maths-pn-ops" cols="grid-cols-2 sm:grid-cols-5" />
      <div className="grid gap-4 md:grid-cols-2">
        <Figure caption={T("maths.penrose.fig.trace")} testid="maths-pn-trace">
          <svg viewBox="0 0 300 90" className="mx-auto w-full max-w-sm" role="img" aria-label="tr(AB) = tr(BA)">
            {loop("A", "B", 25)}
            <text x={150} y={50} fontSize="18" fill={C.muted} textAnchor="middle">=</text>
            {loop("B", "A", 185)}
          </svg>
        </Figure>
        <Figure caption={T("maths.penrose.fig.mps")} testid="maths-pn-mps">
          <svg viewBox="0 0 300 90" className="mx-auto w-full max-w-sm" role="img" aria-label="MPS">
            <Leg d={`M${25} 35 H${25 + 50 * 5}`} />
            {sites.map((n) => (
              <g key={n}>
                <Leg d={`M${25 + 50 * n} 35 V75`} label={`s${"₁₂₃₄₅₆"[n]}`} lx={25 + 50 * n + 10} ly={80} />
                {n < 5 && (
                  <text x={50 + 50 * n} y={28} fontSize="11" fontFamily="serif" fontStyle="italic" fill={C.idx} textAnchor="middle">
                    α{"₁₂₃₄₅"[n]}
                  </text>
                )}
                <PNode x={25 + 50 * n} y={35} label={"ABCDEF"[n]} r={11} />
              </g>
            ))}
          </svg>
        </Figure>
      </div>
    </div>
  );
}

/* ---------- 4. operator graph and contraction order ---------- */

const DAG = [
  { op: "@", side: "W₁ (784, 256)" },
  { op: "+", side: "b₁ (256,)" },
  { op: "ReLU" },
  { op: "@", side: "W₂ (256, 10)" },
];
const EDGES = ["x (B, 784)", "(B, 256)", "(B, 256)", "(B, 256)", "y (B, 10)"];

function DagFigure() {
  const T = usePageText();
  const node = (x: number, y: number, t: string) => (
    <g key={`${x}-${y}`}>
      <rect x={x - 24} y={y - 15} width={48} height={30} rx={8} fill={C.node} stroke={C.edge} strokeWidth="1.5" />
      <text x={x} y={y + 4} fontSize="12" fill={C.leg} textAnchor="middle" fontWeight="600">
        {t}
      </text>
    </g>
  );
  const edgeDefs = (
    <defs>
      <marker id="maths-dag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0,0 L10,5 L0,10 z" fill={C.leg} />
      </marker>
    </defs>
  );
  // wide screens: left to right; phones: top to bottom
  const xs = [0, 1, 2, 3].map((i) => 135 + i * 125);
  const ys = [0, 1, 2, 3].map((i) => 70 + i * 70);
  return (
    <Figure caption={T("maths.topo.fig")} testid="maths-dag">
      <svg viewBox="0 0 640 130" className="hidden w-full sm:block" role="img" aria-label="operator graph">
        {edgeDefs}
        {EDGES.map((e, i) => {
          const x1 = i === 0 ? 20 : xs[i - 1] + 24;
          const x2 = i === 4 ? 625 : xs[i] - 24;
          return (
            <g key={i}>
              <line x1={x1} y1={85} x2={x2} y2={85} stroke={C.leg} strokeWidth="1.5" markerEnd="url(#maths-dag)" />
              <text x={(x1 + x2) / 2} y={105} fontSize="10.5" fill={C.idx} textAnchor="middle" fontFamily="monospace">
                {e}
              </text>
            </g>
          );
        })}
        {DAG.map((d, i) => (
          <g key={i}>
            {d.side && (
              <>
                <line x1={xs[i]} y1={28} x2={xs[i]} y2={68} stroke={C.leg} strokeWidth="1.5" markerEnd="url(#maths-dag)" />
                <text x={xs[i]} y={20} fontSize="10.5" fill={C.muted} textAnchor="middle" fontFamily="monospace">
                  {d.side}
                </text>
              </>
            )}
            {node(xs[i], 85, d.op)}
          </g>
        ))}
      </svg>
      <svg viewBox="0 0 300 340" className="mx-auto block w-full max-w-xs sm:hidden" role="img" aria-label="operator graph">
        {edgeDefs}
        {EDGES.map((e, i) => {
          const y1 = i === 0 ? 8 : ys[i - 1] + 15;
          const y2 = i === 4 ? 335 : ys[i] - 15;
          return (
            <g key={i}>
              <line x1={90} y1={y1} x2={90} y2={y2} stroke={C.leg} strokeWidth="1.5" markerEnd="url(#maths-dag)" />
              <text x={84} y={(y1 + y2) / 2 + 4} fontSize="11" fill={C.idx} textAnchor="end" fontFamily="monospace">
                {e}
              </text>
            </g>
          );
        })}
        {DAG.map((d, i) => (
          <g key={i}>
            {d.side && (
              <>
                <line x1={190} y1={ys[i]} x2={116} y2={ys[i]} stroke={C.leg} strokeWidth="1.5" markerEnd="url(#maths-dag)" />
                <text x={196} y={ys[i] + 4} fontSize="11" fill={C.muted} fontFamily="monospace">
                  {d.side}
                </text>
              </>
            )}
            {node(90, ys[i], d.op)}
          </g>
        ))}
      </svg>
    </Figure>
  );
}

const CHAIN = [10, 100, 5, 50]; // A 10x100, B 100x5, C 5x50
const fmt = (n: number) => n.toLocaleString("en").replace(/,/g, " ");

function ChainTable() {
  const T = usePageText();
  const best = bestChain(CHAIN);
  const [a, b, c, d] = CHAIN;
  const rows: { order: Order; steps: string; inter: string }[] = [
    { order: [[0, 1], 2], steps: `AB: ${a}·${b}·${c}, then (AB)C: ${a}·${c}·${d}`, inter: `${a} × ${c}` },
    { order: [0, [1, 2]], steps: `BC: ${b}·${c}·${d}, then A(BC): ${a}·${b}·${d}`, inter: `${b} × ${d}` },
  ];
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200" data-testid="maths-chain">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-500">
          <tr>
            <th className="px-3 py-2 font-medium">{T("maths.topo.table.order")}</th>
            <th className="px-3 py-2 font-medium">{T("maths.topo.table.steps")}</th>
            <th className="px-3 py-2 text-right font-medium">{T("maths.topo.table.cost")}</th>
            <th className="px-3 py-2 font-medium">{T("maths.topo.table.inter")}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {rows.map((r) => {
            const cost = chainCost(CHAIN, r.order).cost;
            const isBest = cost === best.cost;
            return (
              <tr key={orderText(r.order)} className={isBest ? "bg-emerald-50/70" : ""}>
                <td className="px-3 py-2 font-mono font-semibold text-gray-800">{orderText(r.order)}</td>
                <td className="px-3 py-2 font-mono text-[11px] text-gray-700">{r.steps}</td>
                <td className="px-3 py-2 text-right font-mono text-gray-800">
                  {fmt(cost)}
                  {isBest && <span className="ml-2 font-sans text-[11px] text-emerald-700">✓ {T("maths.topo.best")}</span>}
                </td>
                <td className="px-3 py-2 font-mono text-[11px] text-gray-700">{r.inter}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- 5. roofline ---------- */

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

/* ---------- 6. recap ---------- */

function Recap() {
  const T = usePageText();
  return (
    <>
      <p className="rounded-lg border-l-4 border-amber-300 bg-amber-50 px-4 py-3 text-amber-950">
        <RichText text={T("maths.recap.kid")} />
      </p>
      <h3 className="font-semibold text-gray-900">{T("maths.recap.mis")}</h3>
      <ul className="list-disc space-y-2 pl-5">
        {[1, 2, 3, 4, 5].map((i) => (
          <li key={i}>
            <RichText text={T(`maths.recap.mis${i}` as PageKey)} />
          </li>
        ))}
      </ul>
      {animsAfter("recap", "mis").map((id) => (
        <MathsAnim key={id} id={id} />
      ))}
      <p className="rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-blue-900">
        <RichText text={T("maths.recap.test")} />
      </p>
    </>
  );
}
