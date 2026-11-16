import { Fragment, useState, type ReactNode } from "react";
import { MixedAnim } from "../components/ChipsAnim";
import { GradientHistogram, LossCurves, MacGraph, MmaFigure } from "../components/MixedFigures";
import CodeBlock from "../components/CodeBlock";
import { highlightCode } from "../components/highlight";
import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { FACTS, FLAGS, FORMAT_INFO, PRECISION_DEFAULTS, type PrecisionFlag, setShowOriginal, useDataText, useShowOriginal } from "../content/chips/data";
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

/**
 * Quoted material stays out of the reading flow: each section keeps its own explanation, figures and tables, and
 * the sourced points fold into one closed "Sources and details (N)" box. Code samples stay visible, as they are the
 * lesson itself.
 */
function Cited({ n, children }: { n: number; children: ReactNode }) {
  const T = usePageText();
  if (!n) return null;
  return (
    <details className="group mt-4 rounded-lg border border-gray-200 bg-gray-50/60" data-testid="mixed-cited">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-medium text-gray-600 select-none hover:text-gray-900">
        <span className="inline-block text-gray-400 transition-transform group-open:rotate-90">▸</span>
        {T("mixed.cited")} · {n}
      </summary>
      <ul className="space-y-2 px-4 pb-3 text-[13px] leading-relaxed text-gray-700">{children}</ul>
    </details>
  );
}

const CitedItem = ({ text, href, code }: { text: string; href: string; code?: string }) => (
  <li data-testid="mixed-point">
    <p className="border-l-2 border-gray-200 pl-3">
      <RichText text={text} /> <Ext href={href} />
    </p>
    {code && (
      <div className="mt-1.5">
        <CodeBlock code={code} />
      </div>
    )}
  </li>
);

/** the cross-chip section: one fold per platform, holding its code samples and cited points together */
const PLATFORM: Record<string, string> = { tpu: "Google TPU (JAX)", amd: "AMD (ROCm)", intel: "Intel (Gaudi, CPU)", apple: "Apple (MPS, MLX)", ascend: "Huawei Ascend (CANN)" };

function PlatformPoints({ topics }: { topics: string[] }) {
  const T = usePageText();
  const D = useDataText();
  return (
    <div className="mt-5">
    <h3 className="text-sm font-semibold text-gray-900">{T("mixed.platforms.title")}</h3>
    <p className="mt-0.5 text-xs text-gray-600">{T("mixed.platforms.text")}</p>
    <div className="mt-2 divide-y divide-gray-200 rounded-lg border border-gray-200" data-testid="mixed-platforms">
      {topics.map((t) => {
        const ps = points([t]);
        if (!ps.length) return null;
        return (
          <details key={t} className="group">
            <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-sm select-none hover:bg-gray-50">
              <span className="inline-block text-gray-400 transition-transform group-open:rotate-90">▸</span>
              <span className="font-semibold text-gray-900">{PLATFORM[t] ?? t}</span>
              <span className="text-xs text-gray-500">{ps.length}</span>
            </summary>
            <ul className="space-y-3 px-3 pt-1 pb-3 text-[13px] leading-relaxed text-gray-700">
              {ps.map((p) => (
                <CitedItem key={p.id} text={D(`point.${p.id}`, p.text)} href={p.source} code={p.code} />
              ))}
            </ul>
          </details>
        );
      })}
    </div>
    </div>
  );
}

/** a section's cited points in one fold; code samples go inside with their point, or stay visible (showCode) where
 * the code is the section's lesson */
function Points({ topics, extra, showCode }: { topics: string[]; extra?: { key: string; text: string; source: string }[]; showCode?: boolean }) {
  const D = useDataText();
  const ps = points(topics);
  const more = extra ?? [];
  return (
    <>
      {showCode &&
        ps
          .filter((p) => p.code)
          .map((p) => (
            <div key={p.id} className="mt-3">
              <CodeBlock code={p.code!} />
            </div>
          ))}
      <Cited n={ps.length + more.length}>
        {ps.map((p) => (
          <CitedItem key={p.id} text={D(`point.${p.id}`, p.text)} href={p.source} code={showCode ? undefined : p.code} />
        ))}
        {more.map((m) => (
          <CitedItem key={m.key} text={D(m.key, m.text)} href={m.source} />
        ))}
      </Cited>
    </>
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
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200" data-testid="mixed-format-table">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-600">
          <tr>
            {(["format", "bits", "layout", "max", "minNormal", "eps", "digits"] as const).map((h) => (
              <th key={h} className="px-2 py-1.5 font-semibold whitespace-nowrap">
                {T(`mixed.formats.${h}` as PageKey)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 align-top">
          {FORMAT_INFO.map((f) => (
            <Fragment key={f.id}>
            <tr className={`cursor-pointer hover:bg-gray-50 ${open === f.id ? "bg-blue-50/40" : ""}`} onClick={() => setOpen(open === f.id ? null : f.id)} aria-expanded={open === f.id}>
              <td className="px-2 py-1.5 font-semibold whitespace-nowrap text-gray-900">
                <span className={`mr-1 inline-block text-gray-400 transition-transform ${open === f.id ? "rotate-90" : ""}`}>▸</span>
                {f.name}
              </td>
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
            </tr>
            {open === f.id && (
              <tr className="bg-gray-50/70">
                <td colSpan={7} className="px-4 py-2 leading-relaxed text-gray-700">
                  {D(`format.${f.id}.notes`, f.notes)} <Ext href={f.source} />
                </td>
              </tr>
            )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChipTable() {
  const T = usePageText();
  const D = useDataText();
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200" data-testid="mixed-chip-table">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-600">
          <tr>
            {(["family", "low", "scaling", "fp8"] as const).map((h) => (
              <th key={h} className="px-2 py-1.5 font-semibold">
                {T(`mixed.chips.${h}` as PageKey)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 align-top">
          {MIXED.table.map((r, i) => (
            <Fragment key={r.family}>
              <tr className={`cursor-pointer hover:bg-gray-50 ${open === i ? "bg-blue-50/40" : ""}`} onClick={() => setOpen(open === i ? null : i)} aria-expanded={open === i}>
                <td className="px-2 py-1.5 font-semibold text-gray-900">
                  <span className={`mr-1 inline-block text-gray-400 transition-transform ${open === i ? "rotate-90" : ""}`}>▸</span>
                  {r.family}
                </td>
                <td className="px-2 py-1.5 font-mono text-[11px]">{r.defaultLowPrecision.split(/[;(]/)[0].trim()}</td>
                <td className="px-2 py-1.5">{D(`mixtable.${i}.lossScaling`, r.lossScaling)}</td>
                <td className="px-2 py-1.5">{D(`mixtable.${i}.fp8`, r.fp8)}</td>
              </tr>
              {open === i && (
                <tr className="bg-gray-50/70">
                  <td colSpan={4} className="px-4 py-2 leading-relaxed text-gray-700">
                    <p>
                      <span className="font-semibold text-gray-600">{T("mixed.chips.low")}: </span>
                      <code className="font-mono text-[11px]">{r.defaultLowPrecision}</code>
                    </p>
                    <p className="mt-1">
                      <span className="font-semibold text-gray-600">{T("mixed.chips.autocast")}: </span>
                      <code className="font-mono text-[11px] break-words">{r.autocast}</code>
                    </p>
                    <p className="mt-1">
                      {D(`mixtable.${i}.notes`, r.notes)} <Ext href={r.source} />
                    </p>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** a switch, highlighted in its own language: Python calls, shell variables (export NAME=value), C API constants */
const esc = (t: string) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const LANG_OF = { python: "Python", config: "Python", env: "shell", "c-api": "C" } as const;
function flagHtml(f: PrecisionFlag): string {
  if (f.kind === "env")
    return f.flag
      .split(/\s*\/\s*/)
      .map((part) => {
        const [name, ...value] = part.split("=");
        return `<span class="text-violet-700">export</span> <span class="text-sky-700">${esc(name)}</span>=<span class="text-emerald-700">${esc(value.join("="))}</span>`;
      })
      .join("\n");
  const html = highlightCode(esc(f.flag));
  return f.kind === "c-api" ? html.replace(/\b(CUBLAS_[A-Z0-9_]+)\b/g, '<span class="text-sky-700">$1</span>') : html;
}

/** every package's precision switches, each package folded on its own */
function FlagTable() {
  const T = usePageText();
  const D = useDataText();
  const packages = [...new Set(FLAGS.map((f) => f.package))];
  return (
    <div className="mt-5" data-testid="mixed-flags">
      <h3 className="text-sm font-semibold text-gray-900">
        {T("mixed.flags.title")} · {FLAGS.length}
      </h3>
      <p className="mt-0.5 text-xs text-gray-600">{T("mixed.flags.text")}</p>
      <div className="mt-2 divide-y divide-gray-200 rounded-lg border border-gray-200">
        {packages.map((pkg) => {
          const rows = FLAGS.map((f, i) => ({ f, i })).filter(({ f }) => f.package === pkg);
          return (
            <details key={pkg} className="group" data-testid="mixed-flag-package">
              <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-sm select-none hover:bg-gray-50">
                <span className="inline-block text-gray-400 transition-transform group-open:rotate-90">▸</span>
                <span className="font-semibold text-gray-900">{pkg}</span>
                <span className="text-xs text-gray-500">{rows.length}</span>
                <span className="ml-auto flex gap-1">
                  {[...new Set(rows.map(({ f }) => LANG_OF[f.kind]))].map((l) => (
                    <span key={l} className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[10px] text-gray-600">
                      {l}
                    </span>
                  ))}
                </span>
              </summary>
              <ul className="space-y-2.5 px-3 pt-1 pb-3">
                {rows.map(({ f, i }) => (
                  <li key={i} className="text-xs">
                    <div className="overflow-hidden rounded-md border border-gray-200">
                      <div className="flex items-center gap-2 border-b border-gray-100 bg-gray-50 px-2 py-0.5 text-[10px] text-gray-500">
                        <span className="font-mono">{LANG_OF[f.kind]}</span>
                        {f.default && (
                          <span className="ml-auto">
                            {T("mixed.flags.default")}: <code className="font-mono text-gray-700">{f.default}</code>
                          </span>
                        )}
                        {f.status !== "current" && (
                          <span className={`rounded px-1 font-medium ${f.default ? "" : "ml-auto"} ${f.status === "deprecated" ? "bg-amber-100 text-amber-800" : "bg-blue-100 text-blue-800"}`}>
                            {T(`mixed.flags.${f.status}` as PageKey)}
                          </span>
                        )}
                      </div>
                      <pre className="overflow-x-auto bg-white px-2.5 py-1.5 font-mono text-[12px] leading-relaxed whitespace-pre-wrap break-words text-gray-900" dangerouslySetInnerHTML={{ __html: flagHtml(f) }} />
                    </div>
                    <p className="mt-1 leading-relaxed text-gray-700">
                      {D(`flag.${i}`, f.what)} <Ext href={f.source} />
                    </p>
                  </li>
                ))}
              </ul>
            </details>
          );
        })}
      </div>
    </div>
  );
}

/** CUDA cores against Tensor Cores, side by side (rates from the chip table: H100 SXM, RTX 4090) */
function CoresTable() {
  const T = usePageText();
  const k = (key: string) => <RichText text={T(`mixed.cores.${key}` as PageKey)} />;
  const rows: [string, ReactNode, ReactNode][] = [
    ["op", k("cuda.op"), k("tensor.op")],
    ["in", "FP64, FP32, FP16 / BF16 ×2", "FP16, BF16, TF32, FP8, FP4, INT8"],
    ["acc", k("cuda.acc"), "FP32 · INT8 → INT32"],
    ["h100", "67 TFLOP/s FP32", "989.5 TFLOP/s BF16"],
    ["geforce", "82.6 TFLOP/s FP32", "165.2 TFLOP/s FP16 → FP32"],
  ];
  return (
    <div className="mt-4" data-testid="mixed-cores">
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-left text-xs">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              <th className="px-3 py-1.5" />
              <th className="px-3 py-1.5 font-semibold text-gray-900">CUDA core</th>
              <th className="px-3 py-1.5 font-semibold text-green-800">Tensor Core</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map(([row, a, b]) => (
              <tr key={row}>
                <th className="px-3 py-1.5 font-medium whitespace-nowrap text-gray-600">{T(`mixed.cores.row.${row}` as PageKey)}</th>
                <td className="px-3 py-1.5 text-gray-700">{a}</td>
                <td className="bg-green-50/50 px-3 py-1.5 text-gray-900">{b}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-gray-600">{T("mixed.cores.tensor.note")}</p>
    </div>
  );
}

/** the checklist: numbered, static */
function Checklist() {
  const T = usePageText();
  return (
    <ol className="mt-3 space-y-1.5 text-sm leading-relaxed" data-testid="mixed-checklist">
      {(["measure", "bf16", "fp16", "keep", "check", "physics", "fp8"] as const).map((key, i) => (
        <li key={key} className="flex gap-3 rounded-lg bg-gray-50 px-3 py-2 text-gray-800">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-600 text-[11px] font-semibold text-white">{i + 1}</span>
          <span>
            <RichText text={T(`mixed.checklist.${key}` as PageKey)} />
          </span>
        </li>
      ))}
    </ol>
  );
}

const pointText = (id: string) => MIXED.points.find((p) => p.id === id)?.text ?? "";
const pointSource = (id: string) => MIXED.points.find((p) => p.id === id)?.source ?? "";

/** the three problems of half precision, each with its solution folded underneath (the cited method points) */
const PROBLEMS: { n: 1 | 2 | 3; points: string[] }[] = [
  { n: 1, points: ["method-loss-scaling", "method-dynamic-scaling"] },
  { n: 2, points: ["method-master-weights"] },
  { n: 3, points: ["method-fp32-ops"] },
];

function ProblemCards() {
  const T = usePageText();
  const D = useDataText();
  return (
    <div className="mt-4 grid gap-3" data-testid="mixed-problems">
      {PROBLEMS.map(({ n, points: ids }) => (
        <div key={n} className="flex flex-col rounded-lg border border-red-100 bg-red-50/40 p-3 text-sm">
          <p className="font-semibold text-red-800">{T(`mixed.why.p${n}.title` as PageKey)}</p>
          <p className="mt-1 flex-1 leading-relaxed text-gray-700">
            <RichText text={T(`mixed.why.p${n}.text` as PageKey)} />
          </p>
          <details className="group mt-2 rounded-md border border-green-200 bg-white">
            <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold text-green-800 select-none">
              <span className="inline-block transition-transform group-open:rotate-90">▸</span>
              {T("mixed.why.solution")}
            </summary>
            <ul className="space-y-2 px-2.5 pb-2.5 text-[13px] leading-relaxed text-gray-700">
              {ids.map((id) => (
                <CitedItem key={id} text={D(`point.${id}`, pointText(id))} href={pointSource(id)} />
              ))}
            </ul>
          </details>
        </div>
      ))}
    </div>
  );
}

export default function MixedTab() {
  const T = usePageText();
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
          <ProblemCards />
          <Points topics={[]} extra={["method-origin", "method-gains"].map((id) => ({ key: `point.${id}`, text: pointText(id), source: pointSource(id) }))} />
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
          <CoresTable />
          <MmaFigure />
          <MacGraph />
          <MixedAnim id="sum" />
          <Points topics={[]} extra={facts(["tc-accumulate", "fp8-accumulate", "kahan", "summation"]).map((f) => ({ key: `fact.${f.id}`, text: f.text, source: f.source }))} />
        </Card>

        <Card id="recipe">
          <Heading n={num("recipe")} text={T("mixed.recipe.title")} />
          <p className={P}>{text("mixed.recipe.text")}</p>
          <div className="grid items-start gap-x-4 lg:grid-cols-2">
            <GradientHistogram />
            <LossCurves />
          </div>
          <MixedAnim id="amp" />
          <MixedAnim id="scale" />
        </Card>

        <Card id="pytorch">
          <Heading n={num("pytorch")} text={T("mixed.pytorch.title")} />
          <p className={P}>{text("mixed.pytorch.text")}</p>
          <Points topics={["pytorch"]} showCode extra={PRECISION_DEFAULTS.map((d, i) => ({ key: `default.${i}`, text: d.text, source: d.source }))} />
        </Card>

        <Card id="chips">
          <Heading n={num("chips")} text={T("mixed.chips.title")} />
          <p className={P}>{text("mixed.chips.text")}</p>
          <ChipTable />
          <FlagTable />
          <PlatformPoints topics={["tpu", "amd", "intel", "apple", "ascend"]} />
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
          <Checklist />
        </Card>
      </article>
    </div>
  );
}
