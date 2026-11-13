import type { ReactNode } from "react";
import { MixedAnim } from "../components/ChipsAnim";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { FACTS, FORMAT_INFO, PRECISION_DEFAULTS, useDataText } from "../content/chips/data";
import mixedJson from "../content/chips/mixed.json";
import { formatPageRoute } from "../pageRoute";

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

function Card({ id, children }: { id: string; children: ReactNode }) {
  return (
    <section id={`mixed-${id}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6" data-testid={`mixed-${id}`}>
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
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">
            {text("mixed.howto")}{" "}
            <a className="font-medium text-blue-700 hover:underline" href={`#${formatPageRoute("skills", "chips")}`}>
              {T("mixed.toChips")} →
            </a>
          </p>
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
