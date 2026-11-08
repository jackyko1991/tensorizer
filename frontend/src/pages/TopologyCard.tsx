import { RichText } from "../components/Tex";
import MathsAnim from "../components/MathsAnim";
import AnalogyPicture from "../components/AnalogyPicture";
import { usePageText, type PageKey } from "../content/pages";
import { bestChain, chainCost, orderText, type Order } from "../tensorMaths";
import { animsAfter, type MathsCase } from "../mathsAnim";
import { FEYNMAN, Feynman, EqCards, Links, Figure } from "./MathsConcept";
const C = { node: "#dbeafe", edge: "#2563eb", leg: "#1f2937", idx: "#c2410c", muted: "#6b7280" };
const EQUATIONS: { tex: string; code: string; anim: MathsCase }[] = [
    { tex: String.raw`\text{cost}\big((AB)C\big) = abc + acd, \qquad \text{cost}\big(A(BC)\big) = bcd + abd`, code: 'torch.einsum("ij,jk,kl->il", A, B, C)   # opt_einsum chooses the order\ntorch.linalg.multi_dot([A, B, C])        # cheapest bracketing of a chain', anim: "eqChain" },
  ];
const REFERENCES = [
    { label: "Wikipedia: Matrix chain multiplication", url: "https://en.wikipedia.org/wiki/Matrix_chain_multiplication" },
    { label: "D. G. A. Smith & J. Gray (2018), opt_einsum, J. Open Source Softw. 3, 753", url: "https://doi.org/10.21105/joss.00753" },
    { label: "PyTorch: torch.linalg.multi_dot", url: "https://docs.pytorch.org/docs/stable/generated/torch.linalg.multi_dot.html" },
  ];

export default function TopologyCard({ number }: { number: number }) {
 const T=usePageText();
 return <section id="flow-topo" className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6">
  <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold"><span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{number}</span>{T("maths.topo.title")}</h2>
  <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
   {FEYNMAN.map(f=><div key={f} className="space-y-4">
    <Feynman kind={f} text={T(`maths.topo.${f}` as PageKey)} label={T(`maths.label.${f}` as PageKey)} illustration={f==="analogy"?<AnalogyPicture kind="topo"/>:undefined}/>
    {animsAfter("topo",f).map(id=><MathsAnim key={id} id={id}/>)}
    {f==="plain"&&<DagFigure/>}{f==="example"&&<ChainTable/>}
   </div>)}
   <p className="rounded-lg border-l-4 border-amber-300 bg-amber-50 px-4 py-3 text-amber-950"><RichText text={T("maths.topo.takeaway")}/></p>
   <aside className="rounded-lg border border-blue-100 bg-blue-50/50 p-4"><h3 className="mb-2 font-semibold text-gray-900">{T("maths.recap.mis")}</h3><p><RichText text={T("maths.topo.misconception")}/></p></aside>
   <EqCards items={EQUATIONS} title={T("maths.label.code")}/>
   <Links items={REFERENCES} title={T("maths.label.links")}/>
  </div>
 </section>;
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

