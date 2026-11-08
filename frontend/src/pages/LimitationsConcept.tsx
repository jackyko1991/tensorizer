import { useState, type ReactNode } from "react";
import { useLang } from "../i18n";
import { usePageText } from "../content/pages";
import content from "../content/limitations.json";
import { jointProbabilities, LIMITATION_EQUATIONS } from "../limitationsModel";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import CodeBlock from "../components/CodeBlock";
import LimitationsFigure from "../components/LimitationsFigure";
import { Links } from "./MathsConcept";

/** One question per card, with a presentation chosen for that question. */
export default function LimitationsConcept() {
  const { lang } = useLang();
  const T = usePageText();
  const l = lang.startsWith("zh") ? 1 : 0;
  const cards: Record<string, ReactNode> = {
    scope: <MeaningCard l={l} />,
    spinors: <RotationCard l={l} />,
    connections: <PathsCard l={l} />,
    topology: <LoopCard l={l} />,
    entanglement: <PairCard l={l} />,
    frontiers: <EvidenceCard l={l} />,
    bridge: <NextCard l={l} />,
  };
  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
      <PageToc scrollRoot="concept-scroll" items={content.sections.map((s, i) => ({ id: s.id, label: s.title[l], mark: i + 1 }))} />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10" data-testid="limitations-concept">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">{l ? "張量的適用範圍與限制" : "The scope and limits of tensors"}</h1>
          <p className="mt-3 text-gray-600">{l ? "張量很強大：從力、電磁場到彎曲的時空，都能用它描述。但普通時空張量也有邊界。自然界有沒有一種量子態，旋轉 360° 後振幅會變號，要到 720° 才復原？有，電子的自旋態就是例子，這時需要旋量來描述。數字相同，不一定代表相同的物理；接著看看電子、波的路徑，以及兩顆粒子如何共用一份紀錄。" : "Tensors are powerful: they describe forces, electromagnetic fields and curved spacetime. But ordinary spacetime tensors have limits. Can a quantum state’s amplitudes change sign after a 360° rotation, returning after 720°? Yes: an electron’s spin state does exactly that. We need spinors to describe this behaviour. Identical arrays need not mean identical physics. Next, explore electrons, the paths taken by waves, and the shared state of two particles."}</p>
        </header>
        {content.sections.map((s, i) => (
          <section key={s.id} id={s.id} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6" data-story={s.id}>
            <h2 className="flex items-baseline gap-3 text-lg font-semibold"><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{i + 1}</span>{s.title[l]}</h2>
            <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">{cards[s.id]}</div>
          </section>
        ))}
        <Links title={T("maths.label.links")} items={[
          { label: "David Tong: The Dirac Equation", url: "https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S4.html" },
          { label: "David Tong: Gauge Theory", url: "https://www.damtp.cam.ac.uk/user/tong/gaugetheory.html" },
          { label: "Caltech: Quantum and classical information processing with tensors", url: "https://iqim.caltech.edu/wp-content/uploads/2019/08/ACM270_complete_lecture_notes.pdf" },
        ]} />
      </article>
    </div>
  );
}

function Demo({ instruction, children }: { instruction: string; children: ReactNode }) {
  return <div className="space-y-3 rounded-lg border-l-4 border-emerald-300 bg-emerald-50/70 px-4 py-3 text-emerald-950" data-demo="true">
    <p><RichText text={instruction}/></p>{children}
  </div>;
}
function Reveal({ title, children }: { title: string; children: ReactNode }) {
  return <details className="rounded-lg border border-gray-200 bg-gray-50/50 px-4 py-3">
    <summary className="cursor-pointer font-medium text-gray-800">{title}</summary>
    <div className="mt-3 space-y-3 text-gray-600">{children}</div>
  </details>;
}

function MeaningCard({ l }: { l: number }) {
  const c = content.scope;
  return <>
    <p className="text-base font-medium text-gray-900">{c.question[l]}</p>
    <div className="grid gap-3 sm:grid-cols-2">
      {[{ title: c.arrowTitle, text: c.arrow, colour: "border-blue-200 bg-blue-50/40" }, { title: c.temperatureTitle, text: c.temperature, colour: "border-amber-200 bg-amber-50/40" }].map(item => <div key={item.title[0]} className={`rounded-lg border p-4 ${item.colour}`}>
        <p className="mb-2 font-mono text-xl text-gray-900">[1, 0]</p>
        <h3 className="font-semibold">{item.title[l]}</h3><p className="mt-1 text-sm">{item.text[l]}</p>
      </div>)}
    </div>
    <Demo instruction={c.try[l]}><LimitationsFigure kind="scope" l={l} /></Demo>
    <p>{c.insight[l]}</p>
    <Reveal title={c.more[l]}>
      <p><RichText text={c.legend[l]} /></p>
      <Tex tex={LIMITATION_EQUATIONS.scope} display />
      <CodeBlock code={'arrow = torch.tensor([1., 0.])\n# Same arrow, with the measuring axes turned 90 degrees:\narrow_in_new_axes = torch.tensor([0., -1.])'} />
    </Reveal>
  </>;
}

function RotationCard({ l }: { l: number }) {
  const c = content.spinors;
  return <>
    <p className="border-l-2 border-blue-200 pl-4">{c.invariance[l]}</p>
    <p>{c.rotationDistinction[l]}</p>
    <p>{c.question[l]}</p>
    <Demo instruction={c.try[l]}><LimitationsFigure kind="spinors" l={l} /></Demo>
    <p><RichText text={c.stateIntro[l]} /></p>
    <ol className="grid grid-cols-3 gap-3 text-center">
      {c.turnLabels.map((label, i) => <li key={i} className={`rounded-lg border px-2 py-3 ${i === 1 ? "border-violet-200 bg-violet-50" : "border-gray-200 bg-gray-50"}`}>
        <p className="text-xs text-gray-500">{label[l]} · {i * 360}°</p>
        <div className="mt-2 text-xl"><Tex tex={i === 1 ? String.raw`-\psi` : String.raw`\psi`} /></div>
      </li>)}
    </ol>
    <p>{c.insight[l]}</p>
    <Reveal title={c.reveal[l]}><p>{c.answer[l]}</p></Reveal>
  </>;
}

function PathsCard({ l }: { l: number }) {
  const c = content.connections;
  return <>
    <p className="border-l-2 border-blue-200 pl-4 text-base leading-relaxed">{c.story[l]}</p>
    <Demo instruction={c.try[l]}><LimitationsFigure kind="connections" l={l} /></Demo>
    <dl className="grid gap-4 border-y border-gray-100 py-4 sm:grid-cols-2">
      <div><dt className="font-semibold text-blue-800">{c.localTitle[l]}</dt><dd className="mt-1">{c.local[l]}</dd></div>
      <div><dt className="font-semibold text-violet-800">{c.routeTitle[l]}</dt><dd className="mt-1">{c.route[l]}</dd></div>
    </dl>
    <p>{c.name[l]}</p>
  </>;
}

function LoopCard({ l }: { l: number }) {
  const c = content.topology;
  return <>
    <p className="text-base font-medium">{c.challenge[l]}</p>
    <Demo instruction={c.try[l]}><LimitationsFigure kind="topology" l={l} /></Demo>
    <div className="flex flex-col gap-3 rounded-lg bg-gray-50 p-4 sm:flex-row sm:gap-6">
      <div className="flex-1"><h3 className="font-semibold">{c.localTitle[l]}</h3><p className="mt-1 text-gray-600">{c.local[l]}</p></div>
      <span aria-hidden className="hidden self-center text-2xl text-gray-300 sm:block">→</span>
      <div className="flex-1"><h3 className="font-semibold">{c.globalTitle[l]}</h3><p className="mt-1 text-gray-600">{c.global[l]}</p></div>
    </div>
    <p>{c.insight[l]}</p>
  </>;
}

function PairCard({ l }: { l: number }) {
  const c = content.entanglement;
  const [paired, setPaired] = useState(false);
  const probabilities = jointProbabilities(paired);
  return <>
    <p>{c.question[l]}</p>
    <div className="space-y-4 rounded-lg border-l-4 border-emerald-300 bg-emerald-50/70 p-4" data-demo="true">
      <div className="flex flex-wrap gap-2">{[false, true].map(value => <button key={String(value)} type="button" aria-pressed={paired === value} onClick={() => setPaired(value)}
        className={`rounded-full border px-3 py-1.5 text-xs ${paired === value ? "border-emerald-600 bg-emerald-700 text-white" : "border-emerald-200 bg-white text-emerald-900"}`}>{value ? c.paired[l] : c.independent[l]}</button>)}</div>
      <table className="mx-auto w-full max-w-sm border-separate border-spacing-2 text-center">
        <caption className="pb-2 text-xs text-gray-600">{c.caption[l]}</caption>
        <thead><tr><td /><th scope="col" className="text-xs font-medium">B = 0</th><th scope="col" className="text-xs font-medium">B = 1</th></tr></thead>
        <tbody>{probabilities.map((row, i) => <tr key={i}><th scope="row" className="text-xs font-medium">A = {i}</th>{row.map((p, j) => <td key={j} className={`rounded-lg border p-3 text-xl tabular-nums ${p ? "border-emerald-200 bg-white text-emerald-900" : "border-gray-100 bg-gray-50 text-gray-400"}`}>{p * 100}%</td>)}</tr>)}</tbody>
      </table>
      <p className="text-xs font-medium text-emerald-950">{c.marginal[l]}</p>
    </div>
    <p>{c.insight[l]}</p>
    <Reveal title={c.caution[l]}><p>{c.answer[l]}</p></Reveal>
  </>;
}

function EvidenceCard({ l }: { l: number }) {
  const c = content.frontiers;
  return <>
    <p>{c.lead[l]}</p>
    <div className="grid gap-3 md:grid-cols-3">{c.statuses.map((status, i) => <div key={status.title[0]} className={`rounded-lg border p-4 ${["border-blue-200 bg-blue-50/30", "border-amber-200 bg-amber-50/40", "border-gray-200 bg-gray-50"][i]}`}>
      <h3 className="mb-2 font-semibold">{status.title[l]}</h3><p className="text-sm text-gray-600">{status.body[l]}</p>
    </div>)}</div>
    <p className="text-gray-600">{c.takeaway[l]}</p>
  </>;
}

function NextCard({ l }: { l: number }) {
  const c = content.bridge;
  return <>
    <p className="text-base font-medium">{c.question[l]}</p>
    <Demo instruction={c.try[l]}><LimitationsFigure kind="bridge" l={l} /></Demo>
    <p><RichText text={c.explanation[l]} /></p>
    <div className="py-1 text-lg"><Tex tex={LIMITATION_EQUATIONS.bridge} display /></div>
    <p>{c.closing[l]}</p>
    <a className="inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white" href="https://jackyko1991.github.io/clifford/#/reading/foundations" target="_blank" rel="noreferrer">{c.link[l]} →</a>
  </>;
}
