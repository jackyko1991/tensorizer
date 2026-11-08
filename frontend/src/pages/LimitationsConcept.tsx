import { useLang } from "../i18n";
import { usePageText } from "../content/pages";
import sections from "../content/limitations.json";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import CodeBlock from "../components/CodeBlock";
import LimitationsFigure from "../components/LimitationsFigure";
import { FEYNMAN, Feynman, Links } from "./MathsConcept";

export default function LimitationsConcept() {
  const { lang } = useLang();
  const T = usePageText();
  const l = lang.startsWith("zh") ? 1 : 0;
  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
      <PageToc scrollRoot="concept-scroll" items={sections.map((s, i) => ({ id: s.id, label: s.title[l], mark: i + 1 }))} />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10" data-testid="limitations-concept">
        <header>
          <p className="text-sm font-medium text-blue-700">{l ? "概念 · 認識張量的邊界" : "Concept · understanding the limits of tensors"}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{l ? "張量的適用範圍與限制" : "The scope and limits of tensors"}</h1>
          <p className="mt-3 text-gray-600">{l ? "張量很強大：從力、電磁場到彎曲的時空，都能用它描述。但普通時空張量也有邊界。自然界有沒有一種量子態，旋轉 360° 後振幅會變號，要到 720° 才復原？有，電子的自旋態就是例子，這時需要旋量來描述。數字相同，不一定代表相同的物理；接著看看電子、波的路徑，以及兩顆粒子如何共用一份紀錄。" : "Tensors are powerful: they describe forces, electromagnetic fields and curved spacetime. But ordinary spacetime tensors have limits. Can a quantum state’s amplitudes change sign after a 360° rotation, returning after 720°? Yes: an electron’s spin state does exactly that. We need spinors to describe this behaviour. Identical arrays need not mean identical physics. Next, explore electrons, the paths taken by waves, and the shared state of two particles."}</p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">{l ? "每節先用白話和熟悉的情景建立直覺，再說清楚新名字。綠色的小例子把說明、圖解與操作放在一起，最後用公式把意思釘牢。" : "Each section builds intuition with plain words and a familiar situation, then explains the new terms. Green example boxes keep the explanation, picture and controls together; the formula follows."}</p>
        </header>
        {sections.map((s, i) => (
          <section key={s.id} id={s.id} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6">
            <h2 className="flex items-baseline gap-3 text-lg font-semibold"><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{i + 1}</span>{s.title[l]}</h2>
            <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
              {FEYNMAN.map(f => <div key={f} className="space-y-4">
                <Feynman kind={f} text={s[f][l]} label={T(`maths.label.${f}`)}>
                  {f === "example" && <LimitationsFigure kind={s.diagram} l={l} />}
                </Feynman>
              </div>)}
              {s.equation && <figure className="overflow-hidden rounded-lg border border-gray-200 bg-gray-50">
                <figcaption className="px-4 pt-3 text-sm text-gray-600"><RichText text={s.legend[l]} /></figcaption>
                <div className="overflow-x-auto px-3 py-2"><Tex tex={s.equation} display /></div>
              </figure>}
              {s.code && <CodeBlock code={s.code} />}
              {s.id === "bridge" && <a className="inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white" href="https://jackyko1991.github.io/clifford/" target="_blank" rel="noreferrer">{l ? "用圖解學克利福德代數" : "Learn Clifford algebra through pictures"} →</a>}
            </div>
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
