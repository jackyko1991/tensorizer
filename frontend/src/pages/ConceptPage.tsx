import { useState } from "react";
import { generatedUrl } from "../api";
import CodeBlock, { extractDef } from "../components/CodeBlock";
import Tex, { RichText } from "../components/Tex";
import PageToc from "../components/PageToc";
import TranslationTable from "../components/TranslationTable";
import { usePageText, type PageKey } from "../content/pages";
import { localize, useLang } from "../i18n";
import type { Manifest } from "../types";
import DataflowConcept from "./DataflowConcept";

const PENDULUM = "mechanics/simple_pendulum";
const STEPS = [0, 1, 2, 3, 4, 5, 6, 7] as const; // 0: tests first (TDD)

/**
 * "How it works", told with one problem: the simple pendulum goes from a physical picture to
 * an equation, to tensors, to code, to numbers and a picture again. Code excerpts are pulled
 * from the problem's real files in the manifest, so the page always matches the library.
 * Text comes from content/pages.<lang>.json.
 */
export default function ConceptPage({ manifest }: { manifest: Manifest }) {
  const T = usePageText();
  const { t, lang } = useLang();
  const p = manifest.problems.find((x) => x.id === PENDULUM);
  if (!p) return <p className="p-8 text-sm text-gray-500">{T("concept.missing")}</p>;
  const file = (name: string) => p.files.find((f) => f.name === name)?.code ?? "";
  const R = (key: PageKey) => <RichText text={T(key)} />;
  const [tab, setTab] = useConceptTab();

  return (
    <div id="concept-scroll" className="h-full overflow-y-auto scroll-smooth bg-gray-50/60">
      <div className="mx-auto max-w-6xl px-5 pt-6">
        <div className="flex gap-1 border-b border-gray-200" role="tablist" data-testid="concept-tabs">
          {TABS.map((k) => (
            <button
              key={k}
              role="tab"
              aria-selected={tab === k}
              onClick={() => setTab(k)}
              className={`-mb-px border-b-2 px-4 py-2 text-sm font-medium ${
                tab === k ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              {T(k === "law" ? "flow.tab.law" : "flow.tab")}
            </button>
          ))}
        </div>
      </div>
      {tab === "flow" ? (
        <div className="pt-6">
          <DataflowConcept />
        </div>
      ) : (
        <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
          <PageToc
            scrollRoot="concept-scroll"
            items={[
              ...STEPS.map((n) => ({ id: `step-${n}`, mark: n, label: T(`concept.s${n}.title` as PageKey) })),
              { id: "concept-turn", mark: "→", label: T("concept.turn.title") },
            ]}
          />
          <article className="min-w-0 max-w-4xl flex-1 space-y-10">
            <header>
              <p className="text-sm font-medium text-blue-700">{T("concept.kicker")}</p>
              <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("concept.title")}</h1>
              <p className="mt-3 text-gray-600">{R("concept.intro")}</p>
              {/* the whole pipeline on one line; scrolls sideways on narrow screens */}
              <ol className="mt-6 flex items-center gap-1 overflow-x-auto pb-1 text-[11px] whitespace-nowrap">
                {STEPS.map((n) => (
                  <li key={n} className="flex shrink-0 items-center gap-1">
                    <button
                      onClick={() =>
                        document.getElementById(`step-${n}`)?.scrollIntoView({ behavior: "smooth", block: "start" })
                      }
                      className="rounded-full border border-blue-200 bg-white px-2 py-0.5 font-medium text-blue-800 transition hover:border-blue-400 hover:bg-blue-50"
                    >
                      {n}. {T(`concept.pipeline.${n}` as PageKey)}
                    </button>
                    {n < STEPS.length - 1 && <span className="text-gray-400">→</span>}
                  </li>
                ))}
              </ol>
            </header>

            <Step n={0} title={T("concept.s0.title")} file="test.py">
              <p>{R("concept.s0.p1")}</p>
              <CodeBlock
                title="test.py"
                code={["test_small_amplitude_period", "test_energy_is_conserved"]
                  .map((name) => extractDef(file("test.py"), name))
                  .join("\n\n\n")}
              />
              <p>{R("concept.s0.p2")}</p>
            </Step>

            <Step n={1} title={T("concept.s1.title")}>
              <div className="grid items-center gap-6 sm:grid-cols-[1fr_auto]">
                <div className="space-y-3">
                  <p>{R("concept.s1.p1")}</p>
                  <p>{R("concept.s1.p2")}</p>
                </div>
                {p.diagram && (
                  <img src={generatedUrl(p.diagram)} alt={t("setup")} className="h-40 w-60 object-contain" />
                )}
              </div>
            </Step>

            <Step n={2} title={T("concept.s2.title")}>
              <p>{R("concept.s2.p1")}</p>
              <Eq tex={p.vector_form} />
              <p>{R("concept.s2.p2")}</p>
            </Step>

            <Step n={3} title={T("concept.s3.title")}>
              <p>{R("concept.s3.p1")}</p>
              <Eq tex={p.index_form} />
              <p>{R("concept.s3.p2")}</p>
              <TranslationTable rows={localize(p, lang).translation} />
            </Step>

            <Step n={4} title={T("concept.s4.title")} file="tensors.py">
              <p>{R("concept.s4.p1")}</p>
              <CodeBlock title="tensors.py" code={extractDef(file("tensors.py"), "initial_state")} />
            </Step>

            <Step n={5} title={T("concept.s5.title")} file="equation.py">
              <p>{R("concept.s5.p1")}</p>
              <CodeBlock
                title="equation.py"
                code={`${extractDef(file("equation.py"), "angular_acceleration")}\n\n\n${extractDef(file("equation.py"), "rhs")}`}
              />
              <p>{R("concept.s5.p2")}</p>
              <CodeBlock code={extractDef(file("equation.py"), "small_angle")} />
            </Step>

            <Step n={6} title={T("concept.s6.title")} file="solve.py">
              <p>{R("concept.s6.p1")}</p>
              <CodeBlock title="solve.py" code={extractDef(file("solve.py"), "solve")} />
            </Step>

            <Step n={7} title={T("concept.s7.title")} file="plot.py">
              <div className="grid items-start gap-6 md:grid-cols-2">
                <div className="space-y-3">
                  <p>{R("concept.s7.p1")}</p>
                  <p>{R("concept.s7.p2")}</p>
                </div>
                {p.output && (
                  <img
                    src={generatedUrl(p.output)}
                    alt=""
                    className="w-full rounded-lg border border-gray-200 bg-white"
                  />
                )}
              </div>
            </Step>

            <section id="concept-turn" className="scroll-mt-6 rounded-xl border border-blue-200 bg-blue-50 p-6">
              <h2 className="text-lg font-semibold text-blue-900">{T("concept.turn.title")}</h2>
              <p className="mt-2 text-sm text-blue-900">{R("concept.turn.text")}</p>
              <a
                href={`#/${PENDULUM}`}
                className="mt-4 inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
              >
                {T("concept.turn.button")}
              </a>
            </section>
          </article>
        </div>
      )}
    </div>
  );
}

const TABS = ["law", "flow"] as const;
type ConceptTab = (typeof TABS)[number];
const TAB_KEY = "tensorizer.conceptTab";

/** Which Concept tab is open, remembered per browser. */
function useConceptTab(): [ConceptTab, (t: ConceptTab) => void] {
  const [tab, setTab] = useState<ConceptTab>(() => {
    try {
      return localStorage.getItem(TAB_KEY) === "flow" ? "flow" : "law";
    } catch {
      return "law";
    }
  });
  const choose = (t: ConceptTab) => {
    setTab(t);
    try {
      localStorage.setItem(TAB_KEY, t);
    } catch {
      // private window: remembered for this page only
    }
    document.getElementById("concept-scroll")?.scrollTo({ top: 0 });
  };
  return [tab, choose];
}

function Step({ n, title, file, children }: { n: number; title: string; file?: string; children: React.ReactNode }) {
  return (
    <section id={`step-${n}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6">
      <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
          {n}
        </span>
        {title}
        {file && (
          <code className="rounded bg-gray-100 px-2 py-0.5 font-mono text-xs font-normal text-gray-600">{file}</code>
        )}
      </h2>
      <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">{children}</div>
    </section>
  );
}

function Eq({ tex }: { tex: string }) {
  return (
    <div className="rounded-lg bg-gray-50 py-2 text-lg">
      <Tex tex={tex} display />
    </div>
  );
}
