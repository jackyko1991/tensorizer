import type { ReactNode } from "react";
import { MixedAnim } from "../components/ChipsAnim";
import { GradientHistogram, LossCurves, MacGraph, MmaFigure } from "../components/MixedFigures";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { FACTS, FLAGS, FORMAT_INFO, PRECISION_DEFAULTS, setShowOriginal, useDataText, useShowOriginal } from "../content/chips/data";
import mixedJson from "../content/chips/mixed.json";
import { useLang } from "../i18n";

/** Every section of the tab, in order: numbered 1..N in the table of contents and on the cards. */
const SECTIONS = ["why", "formats", "accumulate", "recipe", "pytorch", "chips", "fp8", "science", "checklist"] as const;
const num = (key: (typeof SECTIONS)[number]) => SECTIONS.indexOf(key) + 1;
const P = "mt-3 text-sm leading-relaxed text-gray-700";

interface Point {
  id: string;
  topic: string;
  text: string;
  code?: string;
  source: string;
}
interface Row {
  family: string;
  defaultLowPrecision: string;
  autocast: string;
  lossScaling: string;
  fp8: string;
  notes: string;
  source: string;
}
const MIXED = mixedJson as unknown as { points: Point[]; table: Row[] };
const points = (topics: string[]) => MIXED.points.filter((p) => topics.includes(p.topic));

function Heading({ n, text }: { n: number; text: string }) {
  return (
    <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{n}</span>
      <RichText text={text} />
    </h2>
  );
}

/** cards whose cited text comes from the data overlay: these get the "English original" switch */
const CITED = new Set(["why", "formats", "accumulate", "pytorch", "chips", "fp8", "science"]);

/** top-right switch between the translated cited text and the English original (one setting for the whole tab) */
function OriginalSwitch() {
  const T = usePageText();
  const { lang } = useLang();
  const on = useShowOriginal();
  if (lang === "en") return null;
  return (
    <label className="absolute top-3 right-4 inline-flex cursor-pointer items-center gap-1.5 text-xs text-gray-500 select-none" data-testid="mixed-original">
      <span>{T("mixed.original")}</span>
      <button
        type="button"
        role="switch"
        aria-checked={on}
        onClick={() => setShowOriginal(!on)}
        className={`relative h-4 w-7 rounded-full transition-colors ${on ? "bg-blue-600" : "bg-gray-300"}`}
      >
        <span className={`absolute top-0.5 left-0.5 h-3 w-3 rounded-full bg-white shadow transition-transform ${on ? "translate-x-3" : ""}`} />
      </button>
    </label>
  );
}

function Card({ id, children }: { id: string; children: ReactNode }) {
  return (
    <section id={`mixed-${id}`} className="relative scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6" data-testid={`mixed-${id}`}>
      {CITED.has(id) && <OriginalSwitch />}
      {children}
    </section>
  );
}

const host = (u: string) => {
  try {
    return new URL(u).hostname.replace(/^www\./, "");
  } catch {
    return u;
  }
};

function Ext({ href }: { href: string }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-700 hover:underline">
      {host(href)}
      <span aria-hidden> ↗</span>
    </a>
  );
}

/** sourced points of the research, each a short paragraph (translated through the data overlay), code when it has some */
function Points({ topics }: { topics: string[] }) {
  const D = useDataText();
  return (
    <ul className="mt-3 space-y-3 text-sm leading-relaxed text-gray-700">
      {points(topics).map((p) => (
        <li key={p.id} className="rounded-lg bg-gray-50 p-3" data-testid="mixed-point">
          <RichText text={D(`point.${p.id}`, p.text)} /> <Ext href={p.source} />
          {p.code && (
            <div className="mt-2">
              <CodeBlock code={p.code} />
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

const sci = (v: number | null | undefined) => {
  if (v == null) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e5 || a < 1e-3 ? v.toExponential(2).replace("e+", "e") : String(Number(v.toPrecision(5)));
};

function FormatTable() {
  const T = usePageText();
  const D = useDataText();
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200" data-testid="mixed-format-table">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-600">
          <tr>
            {(["format", "bits", "layout", "max", "minNormal", "eps", "digits", "notes"] as const).map((h) => (
              <th key={h} className="px-2 py-1.5 font-semibold whitespace-nowrap">
                {T(`mixed.formats.${h}` as PageKey)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 align-top">
          {FORMAT_INFO.map((f) => (
            <tr key={f.id}>
              <td className="px-2 py-1.5 font-semibold whitespace-nowrap text-gray-900">{f.name}</td>
              <td className="px-2 py-1.5 tabular-nums">{f.bits}</td>
              <td className="px-2 py-1.5 font-mono whitespace-nowrap">
                {f.exponent || f.mantissa ? (
                  <>
                    <span className="text-red-600">{f.sign}</span>·<span className="text-blue-600">{f.exponent}</span>·<span className="text-green-700">{f.mantissa}</span>
                  </>
                ) : (
                  "–"
                )}
              </td>
              <td className="px-2 py-1.5 tabular-nums">{sci(f.maxNormal)}</td>
              <td className="px-2 py-1.5 tabular-nums">{sci(f.minNormal)}</td>
              <td className="px-2 py-1.5 tabular-nums">{sci(f.machineEpsilon)}</td>
              <td className="px-2 py-1.5 tabular-nums">{f.decimalDigits ?? "–"}</td>
              <td className="px-2 py-1.5 text-gray-700">
                {D(`format.${f.id}.notes`, f.notes)} <Ext href={f.source} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChipTable() {
  const T = usePageText();
  const D = useDataText();
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200" data-testid="mixed-chip-table">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-600">
          <tr>
            {(["family", "low", "autocast", "scaling", "fp8", "notes"] as const).map((h) => (
              <th key={h} className="px-2 py-1.5 font-semibold">
                {T(`mixed.chips.${h}` as PageKey)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 align-top">
          {MIXED.table.map((r, i) => (
            <tr key={r.family}>
              <td className="px-2 py-1.5 font-semibold text-gray-900">{r.family}</td>
              <td className="px-2 py-1.5 font-mono">{r.defaultLowPrecision}</td>
              <td className="px-2 py-1.5 font-mono text-[11px] break-words">{r.autocast}</td>
              <td className="px-2 py-1.5">{D(`mixtable.${i}.lossScaling`, r.lossScaling)}</td>
              <td className="px-2 py-1.5">{D(`mixtable.${i}.fp8`, r.fp8)}</td>
              <td className="px-2 py-1.5 text-gray-700">
                {D(`mixtable.${i}.notes`, r.notes)} <Ext href={r.source} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** every package's precision switches: grouped by package, with default and a translated line on what it does */
function FlagTable() {
  const T = usePageText();
  const D = useDataText();
  return (
    <div className="mt-5" data-testid="mixed-flags">
      <h3 className="text-sm font-semibold text-gray-900">{T("mixed.flags.title")}</h3>
      <p className="mt-1 text-xs text-gray-600">{T("mixed.flags.text")}</p>
      <div className="mt-2 divide-y divide-gray-200 rounded-lg border border-gray-200">
        {[...new Set(FLAGS.map((f) => f.package))].map((pkg) => (
          <section key={pkg} className="px-3 py-2">
            <h4 className="text-xs font-semibold text-gray-900">{pkg}</h4>
            <ul className="mt-1 space-y-1.5">
              {FLAGS.map((f, i) =>
                f.package !== pkg ? null : (
                  <li key={i} className="text-xs">
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                      <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] break-all text-gray-900">{f.flag}</code>
                      {f.default && (
                        <span className="text-[11px] text-gray-500">
                          {T("mixed.flags.default")}: <code className="font-mono">{f.default}</code>
                        </span>
                      )}
                      {f.status !== "current" && (
                        <span className={`rounded px-1 py-0.5 text-[10px] font-medium ${f.status === "deprecated" ? "bg-amber-100 text-amber-800" : "bg-blue-100 text-blue-800"}`}>
                          {T(`mixed.flags.${f.status}` as PageKey)}
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 leading-relaxed text-gray-700">
                      {D(`flag.${i}`, f.what)} <Ext href={f.source} />
                    </p>
                  </li>
                ),
              )}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}

export default function MixedTab() {
  const T = usePageText();
  const D = useDataText();
  const text = (key: string) => <RichText text={T(key as PageKey)} />;
  const facts = (ids: string[]) => FACTS.filter((f) => ids.some((id) => f.id.startsWith(id)));
  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
      <PageToc scrollRoot="skills-scroll" items={SECTIONS.map((key, i) => ({ id: `mixed-${key}`, mark: i + 1, label: T(`mixed.${key}.title` as PageKey) }))} />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">{T("mixed.title")}</h1>
          <p className="mt-3 text-gray-600">{text("mixed.intro")}</p>
        </header>

        <Card id="why">
          <Heading n={num("why")} text={T("mixed.why.title")} />
          <p className={P}>{text("mixed.why.text")}</p>
          <Points topics={["method"]} />
        </Card>

        <Card id="formats">
          <Heading n={num("formats")} text={T("mixed.formats.title")} />
          <p className={P}>{text("mixed.formats.text")}</p>
          <MixedAnim id="bits" />
          <MixedAnim id="grid" />
          <FormatTable />
          <Points topics={["bf16"]} />
        </Card>

        <Card id="accumulate">
          <Heading n={num("accumulate")} text={T("mixed.accumulate.title")} />
          <p className={P}>{text("mixed.accumulate.text")}</p>
          <div className="mt-3 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-2" data-testid="mixed-cores">
            {(["cuda", "tensor"] as const).map((c) => (
              <p key={c} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`mixed.cores.${c}` as PageKey)} />
              </p>
            ))}
          </div>
          <MmaFigure />
          <MacGraph />
          <MixedAnim id="sum" />
          <ul className="mt-3 space-y-2 text-sm leading-relaxed text-gray-700">
            {facts(["tc-accumulate", "fp8-accumulate", "kahan", "summation"]).map((f) => (
              <li key={f.id} className="rounded-lg bg-gray-50 p-3">
                <RichText text={D(`fact.${f.id}`, f.text)} /> <Ext href={f.source} />
              </li>
            ))}
          </ul>
        </Card>

        <Card id="recipe">
          <Heading n={num("recipe")} text={T("mixed.recipe.title")} />
          <p className={P}>{text("mixed.recipe.text")}</p>
          <GradientHistogram />
          <LossCurves />
          <MixedAnim id="amp" />
          <MixedAnim id="scale" />
        </Card>

        <Card id="pytorch">
          <Heading n={num("pytorch")} text={T("mixed.pytorch.title")} />
          <p className={P}>{text("mixed.pytorch.text")}</p>
          <Points topics={["pytorch"]} />
          <h3 className="mt-5 text-sm font-semibold text-gray-900">{T("mixed.pytorch.defaults")}</h3>
          <ul className="mt-2 space-y-2 text-sm leading-relaxed text-gray-700">
            {PRECISION_DEFAULTS.map((d, i) => (
              <li key={i} className="rounded-lg bg-gray-50 p-3">
                {D(`default.${i}`, d.text)} <Ext href={d.source} />
              </li>
            ))}
          </ul>
        </Card>

        <Card id="chips">
          <Heading n={num("chips")} text={T("mixed.chips.title")} />
          <p className={P}>{text("mixed.chips.text")}</p>
          <ChipTable />
          <FlagTable />
          <Points topics={["tpu", "amd", "intel", "apple", "ascend"]} />
        </Card>

        <Card id="fp8">
          <Heading n={num("fp8")} text={T("mixed.fp8.title")} />
          <p className={P}>{text("mixed.fp8.text")}</p>
          <Points topics={["fp8"]} />
        </Card>

        <Card id="science">
          <Heading n={num("science")} text={T("mixed.science.title")} />
          <p className={P}>{text("mixed.science.text")}</p>
          <Points topics={["science"]} />
        </Card>

        <Card id="checklist">
          <Heading n={num("checklist")} text={T("mixed.checklist.title")} />
          <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-gray-700">
            {(["measure", "bf16", "fp16", "keep", "check", "physics", "fp8"] as const).map((k) => (
              <li key={k}>
                <RichText text={T(`mixed.checklist.${k}` as PageKey)} />
              </li>
            ))}
          </ol>
        </Card>
      </article>
    </div>
  );
}
