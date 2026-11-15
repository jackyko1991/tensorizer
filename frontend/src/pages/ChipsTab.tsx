import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ChipsAnim, { BudgetFigure } from "../components/ChipsAnim";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import { runProblemStream } from "../api";
import { decodeCeiling, ridgePoint } from "../chipsModel";
import { usePageText, type PageKey } from "../content/pages";
import {
  CHECKED,
  CHIPS,
  FAMILIES,
  LIBRARIES,
  PRECISION_DEFAULTS,
  STACKS,
  TIMELINE,
  UNDER,
  familyOf,
  isDerived,
  rate,
  useDataText,
  type Chip,
  type Family,
  type Level,
  type Precision,
} from "../content/chips/data";
import { formatPageRoute } from "../pageRoute";
import TimelineArt, { MAJOR_EVENTS } from "../components/TimelineArt";

/** Every section of the tab, in order: numbered 1..N in the table of contents and on the cards. */
const SECTIONS = ["why", "history", "table", "precision", "under", "libs", "china", "arm", "desktop", "measure", "choose"] as const;
const num = (key: (typeof SECTIONS)[number]) => SECTIONS.indexOf(key) + 1;
const fill = (s: string, v: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (m, k) => String(v[k] ?? m));
const P = "mt-3 text-sm leading-relaxed text-gray-700";

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
    <section id={`chips-${id}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6" data-testid={`chips-${id}`}>
      {children}
    </section>
  );
}

function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-700 hover:underline">
      {children}
      <span aria-hidden> ↗</span>
    </a>
  );
}

const host = (u: string) => {
  try {
    return new URL(u).hostname.replace(/^www\./, "");
  } catch {
    return u;
  }
};

/** the chip's name without its parenthesised qualifier ("B200 (HGX B200, per GPU)" → "B200"); the full name is the tooltip */
const short = (name: string) => name.replace(/\s*\(.*\)\s*$/, "");

const fmtNum = (v: number | number[] | null | undefined, digits = 3): string => {
  if (v == null) return "–";
  if (Array.isArray(v)) return `${fmtNum(v[0], digits)}–${fmtNum(v[1], digits)}`;
  if (v >= 1000) return Math.round(v).toLocaleString("en-US");
  return Number(v.toPrecision(digits)).toString();
};

// --- 3. the chip table ------------------------------------------------------------------------------------

const PRECISIONS: Precision[] = ["int8", "fp4", "fp8", "bf16", "fp32", "fp64"];
type SortKey = "name" | "year" | "bandwidthGBs" | "rate" | "ridge" | "memoryGB" | "powerW";

function ChipTable({ selected, onSelect, precision, setPrecision, extra }: {
  selected: string | null;
  onSelect: (id: string | null) => void;
  precision: Precision;
  setPrecision: (p: Precision) => void;
  extra: Chip | null;
}) {
  const T = usePageText();
  const D = useDataText();
  const [vendor, setVendor] = useState("all");
  const [cls, setCls] = useState("all");
  const [region, setRegion] = useState("all");
  const [device, setDevice] = useState("all");
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "bandwidthGBs", dir: -1 });
  const all = useMemo(() => (extra ? [extra, ...CHIPS] : CHIPS), [extra]);
  const vendors = [...new Set(CHIPS.map((c) => c.vendor))].sort();
  const classes = [...new Set(CHIPS.map((c) => c.class))].sort();
  const devices = [...new Set(CHIPS.map((c) => c.pytorchDevice ?? ""))].sort((a, b) => (a === "" ? 1 : b === "" ? -1 : a.localeCompare(b)));
  const value = (c: Chip, k: SortKey): number | string => {
    if (k === "name") return `${c.vendor} ${c.name}`;
    if (k === "rate") return rate(c, precision) ?? -1;
    if (k === "ridge") {
      const r = rate(c, precision);
      return r && c.bandwidthGBs ? ridgePoint(r, c.bandwidthGBs) : -1;
    }
    const v = c[k] as number | number[] | null;
    return Array.isArray(v) ? v[1] : (v ?? -1);
  };
  const rows = all
    .filter(
      (c) =>
        (vendor === "all" || c.vendor === vendor) &&
        (cls === "all" || c.class === cls) &&
        (region === "all" || c.group === region) &&
        (device === "all" || (c.pytorchDevice ?? "") === device),
    )
    .sort((a, b) => {
      const x = value(a, sort.key);
      const y = value(b, sort.key);
      return (typeof x === "string" ? x.localeCompare(String(y)) : x - (y as number)) * sort.dir;
    });
  const th = (key: SortKey, label: string) => (
    <th className="px-2 py-1.5 font-semibold whitespace-nowrap">
      <button
        className={`hover:text-blue-700 ${sort.key === key ? "text-blue-700" : ""}`}
        onClick={() => setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : key === "name" ? 1 : -1 }))}
      >
        {label}
        {sort.key === key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
      </button>
    </th>
  );
  const sel = all.find((c) => c.id === selected) ?? null;
  const select = "rounded-md border border-gray-300 bg-white px-2 py-1 text-xs";
  return (
    <div className="mt-4" data-testid="chips-table">
      <div className="flex flex-wrap items-center gap-2 text-xs text-gray-700">
        <select className={select} value={vendor} onChange={(e) => setVendor(e.target.value)} aria-label={T("chips.table.vendor")}>
          <option value="all">{T("chips.table.allVendors")}</option>
          {vendors.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
        <select className={select} value={cls} onChange={(e) => setCls(e.target.value)} aria-label={T("chips.table.class")}>
          <option value="all">{T("chips.table.allClasses")}</option>
          {classes.map((v) => (
            <option key={v} value={v}>
              {T(`chips.class.${v}` as PageKey)}
            </option>
          ))}
        </select>
        <select className={select} value={region} onChange={(e) => setRegion(e.target.value)} aria-label={T("chips.table.region")}>
          <option value="all">{T("chips.table.allRegions")}</option>
          <option value="global">{T("chips.table.global")}</option>
          <option value="china">{T("chips.table.china")}</option>
        </select>
        <select className={select} value={device} onChange={(e) => setDevice(e.target.value)} aria-label={T("chips.table.device")}>
          <option value="all">{T("chips.table.allDevices")}</option>
          {devices.map((d) => (
            <option key={d || "none"} value={d}>
              {d || T("chips.table.noDevice")}
            </option>
          ))}
        </select>
        <span className="inline-flex items-center gap-1.5">
          <span className="text-gray-500">{T("chips.table.precision")}</span>
          <span className="inline-flex rounded-md border border-gray-300 p-0.5" role="radiogroup" aria-label={T("chips.table.precision")}>
            {PRECISIONS.map((p) => (
              <button
                key={p}
                role="radio"
                aria-checked={precision === p}
                onClick={() => setPrecision(p)}
                className={`rounded px-2 py-0.5 font-mono ${precision === p ? "bg-blue-600 text-white" : "text-gray-600 hover:bg-gray-100"}`}
              >
                {p.toUpperCase()}
              </button>
            ))}
          </span>
        </span>
        <span className="ml-auto text-gray-500">{fill(T("chips.table.count"), { n: rows.length })}</span>
      </div>
      <div className="mt-2 max-h-[34rem] overflow-auto rounded-lg border border-gray-200">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 z-10 bg-gray-50 text-gray-600">
            <tr>
              {th("name", T("chips.table.chip"))}
              <th className="px-2 py-1.5 font-semibold">{T("chips.table.class")}</th>
              {th("year", T("chips.table.year"))}
              {th("memoryGB", T("chips.table.memory"))}
              {th("bandwidthGBs", T("chips.table.bandwidth"))}
              {th("rate", fill(T("chips.table.rate"), { p: precision.toUpperCase(), u: precision === "int8" ? "TOPS" : "TFLOP/s" }))}
              {th("ridge", T("chips.table.ridge"))}
              {th("powerW", T("chips.table.power"))}
              <th className="px-2 py-1.5 font-semibold">{T("chips.table.device")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map((c) => {
              const r = rate(c, precision);
              const ridge = r && c.bandwidthGBs ? ridgePoint(r, c.bandwidthGBs) : null;
              const mine = c.id === "measured";
              return (
                <tr
                  key={c.id}
                  onClick={() => onSelect(selected === c.id ? null : c.id)}
                  className={`cursor-pointer ${selected === c.id ? "bg-blue-50" : mine ? "bg-amber-50" : "hover:bg-gray-50"}`}
                  data-testid={`chip-row-${c.id}`}
                >
                  <td className="px-2 py-1.5">
                    <span className="text-gray-500">{c.vendor}</span>{" "}
                    <b className="font-semibold text-gray-900" title={c.name}>
                      {short(c.name)}
                    </b>
                    {c.nameZh && <span className="text-gray-500"> {c.nameZh}</span>}
                    {c.year != null && c.year > 2026 && <span className="ml-1 rounded bg-gray-100 px-1 text-[10px] text-gray-600">{T("chips.table.planned")}</span>}
                  </td>
                  <td className="px-2 py-1.5 text-gray-600">{T(`chips.class.${c.class}` as PageKey)}</td>
                  <td className="px-2 py-1.5 tabular-nums">{c.year ?? "–"}</td>
                  <td className="px-2 py-1.5 tabular-nums">{fmtNum(c.memoryGB)}</td>
                  <td className="px-2 py-1.5 tabular-nums">{fmtNum(c.bandwidthGBs)}</td>
                  <td className="px-2 py-1.5 font-semibold tabular-nums" title={r != null && isDerived(c, precision) ? T("chips.table.derived") : undefined}>
                    {r != null && isDerived(c, precision) ? "≈" : ""}
                    {fmtNum(r)}
                  </td>
                  <td className="px-2 py-1.5 tabular-nums">{ridge ? fmtNum(ridge) : "–"}</td>
                  <td className="px-2 py-1.5 tabular-nums">{fmtNum(c.powerW)}</td>
                  <td className="px-2 py-1.5 font-mono text-[11px]">{c.pytorchDevice ?? "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-[11px] text-gray-500">
        <RichText text={T("chips.table.legend")} />
      </p>
      {sel && (
        <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50/50 p-4 text-sm" data-testid="chip-detail">
          <div className="flex flex-wrap items-baseline gap-2">
            <b className="text-base text-gray-900">
              {sel.vendor} {sel.name}
            </b>
            {sel.nameZh && <span className="text-gray-500">{sel.nameZh}</span>}
            <button className="ml-auto text-xs text-gray-500 hover:text-gray-900" onClick={() => onSelect(null)}>
              ✕
            </button>
          </div>
          <dl className="mt-2 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
            {(
              [
                ["memoryType", sel.memoryType],
                ["matrixUnit", sel.matrixUnit],
                ["formats", sel.formats.join(", ")],
                ["blas", sel.blas],
                ["kernelLangs", sel.kernelLangs],
                ["interconnect", sel.interconnect],
                ["fp64", sel.fp64TensorTF ?? sel.fp64TF],
                ["fp32", sel.fp32TF],
                ["bf16", sel.bf16TF],
                ["fp8", sel.fp8TF],
                ["int8", sel.int8TOPS],
                ["npu", sel.npuTOPS],
              ] as [string, unknown][]
            )
              .filter(([, v]) => v !== null && v !== undefined && v !== "")
              .map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <dt className="shrink-0 text-gray-500">{T(`chips.detail.${k}` as PageKey)}</dt>
                  <dd className="text-gray-800">{typeof v === "number" ? fmtNum(v) : String(v)}</dd>
                </div>
              ))}
          </dl>
          {sel.notes && <p className="mt-2 text-xs leading-relaxed text-gray-700">{D(`chip.${sel.id}.notes`, sel.notes)}</p>}
          <p className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs">
            <span className="text-gray-500">{T("chips.sources")}</span>
            {[...new Set(Object.values(sel.sources))].map((u) => (
              <Ext key={u} href={u}>
                {host(u)}
              </Ext>
            ))}
            {sel.confidence && <span className="text-gray-500">· {sel.confidence.split(" (")[0]}</span>}
          </p>
        </div>
      )}
    </div>
  );
}

// --- 3. compute vs bandwidth map ---------------------------------------------------------------------------

const VENDOR_COL: Record<string, string> = {
  NVIDIA: "#16a34a",
  AMD: "#dc2626",
  Intel: "#2563eb",
  Apple: "#6b7280",
  Qualcomm: "#7c3aed",
  Google: "#f59e0b",
  AWS: "#ea580c",
};
const colourOf = (c: Chip) => (c.id === "measured" ? "#111827" : c.group === "china" ? "#be123c" : (VENDOR_COL[c.vendor] ?? "#0891b2"));

function RooflineMap({ precision, extra, selected, onSelect }: { precision: Precision; extra: Chip | null; selected: string | null; onSelect: (id: string) => void }) {
  const T = usePageText();
  const pts = (extra ? [extra, ...CHIPS] : CHIPS).flatMap((c) => {
    const r = rate(c, precision);
    return r && c.bandwidthGBs ? [{ c, x: c.bandwidthGBs, y: r }] : [];
  });
  const W = 560;
  const H = 320;
  const m = { l: 52, r: 16, t: 12, b: 40 };
  const [x0, x1, y0, y1] = [1, 5, -2, 4]; // log10 GB/s and TFLOP/s
  const X = (v: number) => m.l + ((Math.log10(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v: number) => H - m.b - ((Math.log10(v) - y0) / (y1 - y0)) * (H - m.t - m.b);
  // lines of constant ridge point: TFLOP/s = I × GB/s / 1000
  const ridges = [1, 10, 100, 1000];
  return (
    <figure className="mt-5 rounded-lg border border-gray-200 bg-white p-3" data-testid="chips-map">
      <figcaption className="text-xs font-semibold text-gray-700">{fill(T("chips.map.title"), { p: precision.toUpperCase() })}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 w-full" role="img" aria-label={T("chips.map.caption").replace(/[$`*]/g, "")}>
        {[1, 2, 3, 4, 5].map((e) => (
          <g key={`x${e}`}>
            <line x1={X(10 ** e)} y1={m.t} x2={X(10 ** e)} y2={H - m.b} stroke="#f3f4f6" />
            <text x={X(10 ** e)} y={H - m.b + 14} fontSize={10} textAnchor="middle" fill="#6b7280">
              {10 ** e >= 1000 ? `${10 ** e / 1000}k` : 10 ** e}
            </text>
          </g>
        ))}
        {[-2, -1, 0, 1, 2, 3, 4].map((e) => (
          <g key={`y${e}`}>
            <line x1={m.l} y1={Y(10 ** e)} x2={W - m.r} y2={Y(10 ** e)} stroke="#f3f4f6" />
            <text x={m.l - 6} y={Y(10 ** e) + 3} fontSize={10} textAnchor="end" fill="#6b7280">
              {10 ** e < 1 ? 10 ** e : 10 ** e >= 1000 ? `${10 ** e / 1000}k` : 10 ** e}
            </text>
          </g>
        ))}
        {ridges.map((I) => {
          const a = Math.max(10 ** x0, (10 ** y0 * 1000) / I);
          const b = Math.min(10 ** x1, (10 ** y1 * 1000) / I);
          if (a >= b) return null;
          return (
            <g key={I}>
              <line x1={X(a)} y1={Y((I * a) / 1000)} x2={X(b)} y2={Y((I * b) / 1000)} stroke="#cbd5e1" strokeDasharray="3 3" />
              <text x={X(b) - 4} y={Y((I * b) / 1000) + 12} fontSize={9} textAnchor="end" fill="#94a3b8">
                {fill(T("chips.map.ridge"), { i: I })}
              </text>
            </g>
          );
        })}
        {pts.map(({ c, x, y }) => (
          <g key={c.id} onClick={() => onSelect(c.id)} className="cursor-pointer">
            <circle cx={X(x)} cy={Y(y)} r={selected === c.id || c.id === "measured" ? 6 : 4} fill={colourOf(c)} fillOpacity={0.85} stroke="#fff" strokeWidth={1} />
            <title>{`${c.vendor} ${c.name}: ${fmtNum(x)} GB/s, ${fmtNum(y)} ${precision === "int8" ? "TOPS" : "TFLOP/s"}`}</title>
            {(selected === c.id || c.id === "measured") && (
              <text x={X(x) + 8} y={Y(y) + 4} fontSize={10} fontWeight={600} fill="#111827">
                {short(c.name)}
              </text>
            )}
          </g>
        ))}
        <text x={(m.l + W - m.r) / 2} y={H - 6} fontSize={10.5} textAnchor="middle" fill="#374151">
          {T("chips.map.x")}
        </text>
        <text x={12} y={(m.t + H - m.b) / 2} fontSize={10.5} textAnchor="middle" fill="#374151" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>
          {fill(T("chips.map.y"), { u: precision === "int8" ? "TOPS" : "TFLOP/s" })}
        </text>
      </svg>
      <p className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-gray-600">
        {[...Object.entries(VENDOR_COL), ["china", "#be123c"]].map(([v, col]) => (
          <span key={v} className="inline-flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: col }} />
            {v === "china" ? T("chips.table.china") : v}
          </span>
        ))}
      </p>
      <p className="mt-1 text-xs leading-relaxed text-gray-600">
        <RichText text={T("chips.map.caption")} />
      </p>
    </figure>
  );
}

// --- 2. history --------------------------------------------------------------------------------------------

const KIND_COL = { library: "#2563eb", chip: "#16a34a", standard: "#7c3aed", policy: "#be123c" } as const;

function History() {
  const T = usePageText();
  const D = useDataText();
  const [hover, setHover] = useState<string | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);
  const [width, setWidth] = useState(0);
  const wrap = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const n = TIMELINE.length;
  const pad = 10;
  const xAt = (i: number) => pad + (n === 1 ? 0 : (i / (n - 1)) * (width - 2 * pad));
  const current = hover ?? pinned;
  const isMajor = (id: string) => MAJOR_EVENTS.includes(id);
  const idx = TIMELINE.findIndex((e) => e.id === current);
  const ev = idx >= 0 && !isMajor(TIMELINE[idx].id) ? TIMELINE[idx] : null;
  const majors = TIMELINE.filter((e) => isMajor(e.id));
  const cardW = Math.min(320, width);
  const cardLeft = ev ? Math.max(0, Math.min(width - cardW, xAt(idx) - cardW / 2)) : 0;
  const decade = (y: number) => Math.floor(y / 10) * 10;
  // a decade label at the first event of each decade, skipped when it would touch the previous label
  const labelled = new Set<number>();
  let lastX = -Infinity;
  TIMELINE.forEach((e, i) => {
    if (i > 0 && decade(e.year) === decade(TIMELINE[i - 1].year)) return;
    if (xAt(i) - lastX < 40) return;
    labelled.add(i);
    lastX = xAt(i);
  });
  // wide screens: major events alternate above and below the line, each joined to its dot
  const wide = width >= 640;
  const top = majors.filter((_, i) => i % 2 === 0);
  const bottom = majors.filter((_, i) => i % 2 === 1);
  const cols = top.length;
  const inset = bottom.length < cols; // one fewer below: the bottom row sits between the top cards
  const slotX = (row: "top" | "bottom", k: number) => (row === "bottom" && inset ? (k + 1) / cols : (k + 0.5) / cols) * width;
  const dotX = (id: string) => xAt(TIMELINE.findIndex((e) => e.id === id));
  const majorCard = (e: (typeof TIMELINE)[number], extra = "") => (
    <div
      key={e.id}
      id={`chips-major-${e.id}`}
      className={`rounded-lg border bg-white p-2.5 text-sm transition-shadow ${e.id === current ? "border-blue-300 shadow-md" : "border-gray-200"} ${extra}`}
      onMouseEnter={() => setHover(e.id)}
      onMouseLeave={() => setHover(null)}
    >
            <div className="rounded-md bg-gray-50 px-2 py-1">
        <TimelineArt id={e.id} />
      </div>
      <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
        <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: KIND_COL[e.kind] }} />
        <span className="font-mono tabular-nums">{e.year}</span>
        <span>{T(`chips.history.${e.kind}` as PageKey)}</span>
      </div>
      <p className="mt-1 font-semibold text-gray-900">{D(`event.${e.id}.title`, e.title)}</p>
      <p className="mt-1 text-xs leading-relaxed text-gray-700">
        {D(`event.${e.id}.text`, e.text)} <Ext href={e.source}>{host(e.source)}</Ext>
      </p>
    </div>
  );
  const connectors = (row: "top" | "bottom") => (
    <svg width={width} height={22} className="block" aria-hidden>
      {(row === "top" ? top : bottom).map((e, k) => {
        const x1 = slotX(row, k);
        const x2 = dotX(e.id);
        const [y1, y2] = row === "top" ? [0, 22] : [22, 0];
        return <path key={e.id} d={`M${x1} ${y1} C${x1} 11, ${x2} 11, ${x2} ${y2}`} fill="none" stroke={e.id === current ? "#2563eb" : "#cbd5e1"} strokeWidth={1} />;
      })}
    </svg>
  );
  return (
    <div className="mt-4" data-testid="chips-timeline">
      {wide && (
        <>
          <div className="grid items-end gap-2" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }} data-testid="chips-major-events">
            {top.map((e) => majorCard(e))}
          </div>
          {connectors("top")}
        </>
      )}
      <div ref={wrap} className="relative h-12" onMouseLeave={() => setHover(null)}>
        <div className="absolute top-5 right-0 left-0 h-px bg-gray-300" />
        {width > 0 &&
          TIMELINE.map((e, i) => {
            const label = labelled.has(i);
            const on = e.id === current;
            const major = isMajor(e.id);
            return (
              <div key={e.id} className="absolute top-0" style={{ left: xAt(i), transform: "translateX(-50%)" }}>
                <button
                  type="button"
                  aria-label={`${e.year} ${D(`event.${e.id}.title`, e.title)}`}
                  aria-pressed={pinned === e.id}
                  data-testid="chips-event-dot"
                  className="flex h-10 w-4 items-center justify-center focus:outline-none"
                  onMouseEnter={() => setHover(e.id)}
                  onFocus={() => setHover(e.id)}
                  onBlur={() => setHover(null)}
                  onClick={() => (major ? document.getElementById(`chips-major-${e.id}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" }) : setPinned(pinned === e.id ? null : e.id))}
                >
                  <span
                    className={`block rounded-full ring-white transition-all ${on || major ? "h-3.5 w-3.5 ring-2" : "h-2.5 w-2.5"}`}
                    style={{ background: KIND_COL[e.kind], boxShadow: on ? `0 0 0 3px ${KIND_COL[e.kind]}55` : undefined }}
                  />
                </button>
                {label && <span className="absolute top-10 left-1/2 -translate-x-1/2 font-mono text-[10px] text-gray-500 tabular-nums">{decade(e.year)}s</span>}
              </div>
            );
          })}
        {ev && (
          <div
            className="absolute top-14 z-20 rounded-lg border border-gray-200 bg-white p-3 text-sm shadow-lg"
            style={{ left: cardLeft, width: cardW }}
            data-testid="chips-event-card"
          >
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: KIND_COL[ev.kind] }} />
              <span className="font-mono tabular-nums">
                {ev.year}
                {ev.month ? `-${String(ev.month).padStart(2, "0")}` : ""}
              </span>
              <span>{T(`chips.history.${ev.kind}` as PageKey)}</span>
            </div>
            <p className="mt-1 font-semibold text-gray-900">{D(`event.${ev.id}.title`, ev.title)}</p>
            <p className="mt-1 leading-relaxed text-gray-700">
              {D(`event.${ev.id}.text`, ev.text)} <Ext href={ev.source}>{host(ev.source)}</Ext>
            </p>
          </div>
        )}
      </div>
      {wide ? (
        <>
          {connectors("bottom")}
          <div className="grid items-start gap-2" style={{ gridTemplateColumns: `repeat(${bottom.length}, minmax(0, 1fr))`, paddingLeft: inset ? `${50 / cols}%` : 0, paddingRight: inset ? `${50 / cols}%` : 0 }}>
            {bottom.map((e) => majorCard(e))}
          </div>
        </>
      ) : (
        width > 0 && <div className="mt-4 grid gap-3" data-testid="chips-major-events">{majors.map((e) => majorCard(e))}</div>
      )}
      <p className="mt-3 flex flex-wrap gap-3 text-[11px] text-gray-600">
        {(Object.keys(KIND_COL) as (keyof typeof KIND_COL)[]).map((k) => (
          <span key={k} className="inline-flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: KIND_COL[k] }} />
            {T(`chips.history.${k}` as PageKey)}
          </span>
        ))}
      </p>
    </div>
  );
}

// --- 5. what runs underneath -------------------------------------------------------------------------------

function UnderTable() {
  const T = usePageText();
  const D = useDataText();
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200" data-testid="chips-under">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-600">
          <tr>
            {(["family", "blas", "langs", "best", "avoid"] as const).map((h) => (
              <th key={h} className="px-2 py-1.5 font-semibold">
                {T(`chips.under.${h}` as PageKey)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 align-top">
          {UNDER.map((u) => (
            <tr key={u.family}>
              <td className="px-2 py-1.5 font-semibold whitespace-nowrap text-gray-900">{T(`chips.fam.${u.family}` as PageKey)}</td>
              <td className="px-2 py-1.5 text-gray-700">{u.blas.join(", ")}</td>
              <td className="px-2 py-1.5 text-gray-700">
                {u.kernelLangs.join(", ")}
                <div className="text-gray-500">{u.compilers.join(", ")}</div>
              </td>
              <td className="px-2 py-1.5 text-gray-700">
                <ul className="list-disc pl-3">
                  {u.bestFor.map((b, i) => (
                    <li key={i}>{D(`under.${u.family}.best.${i}`, b)}</li>
                  ))}
                </ul>
              </td>
              <td className="px-2 py-1.5 text-gray-700">
                <ul className="list-disc pl-3">
                  {u.avoidFor.map((b, i) => (
                    <li key={i}>{D(`under.${u.family}.avoid.${i}`, b)}</li>
                  ))}
                </ul>
                <div className="mt-1 flex flex-wrap gap-x-2 break-all">
                  {u.sources.slice(0, 3).map((s) => (
                    <Ext key={s} href={s}>
                      {host(s)}
                    </Ext>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --- 6. library × hardware ---------------------------------------------------------------------------------

const LEVEL_STYLE: Record<Level, { mark: string; cls: string }> = {
  official: { mark: "✓", cls: "bg-green-50 text-green-800" },
  preview: { mark: "◐", cls: "bg-amber-50 text-amber-800" },
  community: { mark: "○", cls: "bg-sky-50 text-sky-800" },
  none: { mark: "✗", cls: "text-gray-300" },
};

function LibraryTable({ highlight }: { highlight: Family | null }) {
  const T = usePageText();
  const D = useDataText();
  const [cell, setCell] = useState<{ lib: string; fam: Family } | null>(null);
  const picked = cell ? LIBRARIES.find((l) => l.id === cell.lib)?.cells[cell.fam] : null;
  return (
    <div className="mt-4" data-testid="chips-libs">
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-center text-xs">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              <th className="sticky left-0 bg-gray-50 px-2 py-1.5 text-left font-semibold">{T("chips.libs.library")}</th>
              {FAMILIES.map((f) => (
                <th key={f} className={`px-1 py-1.5 align-bottom font-semibold ${highlight === f ? "bg-blue-100 text-blue-900" : ""}`}>
                  <span className="inline-block max-w-[5.5rem] leading-tight">{T(`chips.fam.${f}` as PageKey)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {LIBRARIES.map((lib) => (
              <tr key={lib.id}>
                <td className="sticky left-0 bg-white px-2 py-1 text-left font-semibold whitespace-nowrap text-gray-900">{lib.name}</td>
                {FAMILIES.map((f) => {
                  const c = lib.cells[f] ?? { level: "none" as Level };
                  const s = LEVEL_STYLE[c.level];
                  const on = cell?.lib === lib.id && cell.fam === f;
                  return (
                    <td key={f} className={`px-0.5 py-0.5 ${highlight === f ? "bg-blue-50/60" : ""}`}>
                      <button
                        onClick={() => setCell(on ? null : { lib: lib.id, fam: f })}
                        className={`h-6 w-full rounded font-semibold ${s.cls} ${on ? "ring-2 ring-blue-500" : ""}`}
                        title={`${lib.name} · ${T(`chips.fam.${f}` as PageKey)}: ${T(`chips.level.${c.level}` as PageKey)}`}
                        aria-label={`${lib.name}, ${T(`chips.fam.${f}` as PageKey)}: ${T(`chips.level.${c.level}` as PageKey)}`}
                      >
                        {s.mark}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1 flex flex-wrap gap-x-4 text-[11px] text-gray-600">
        {(Object.keys(LEVEL_STYLE) as Level[]).map((l) => (
          <span key={l}>
            <b className={LEVEL_STYLE[l].cls.split(" ").find((c) => c.startsWith("text-"))}>{LEVEL_STYLE[l].mark}</b> {T(`chips.level.${l}` as PageKey)}
          </span>
        ))}
        <span className="text-gray-500">{fill(T("chips.libs.checked"), { d: CHECKED })}</span>
      </p>
      {cell && picked && (
        <div className="mt-2 rounded-lg border border-blue-200 bg-blue-50/50 p-3 text-sm" data-testid="lib-detail">
          <b className="text-gray-900">
            {LIBRARIES.find((l) => l.id === cell.lib)?.name} · {T(`chips.fam.${cell.fam}` as PageKey)}
          </b>{" "}
          <span className="text-gray-600">({T(`chips.level.${picked.level}` as PageKey)})</span>
          {picked.device && (
            <p className="mt-1">
              <code className="rounded bg-white px-1 font-mono text-xs">{picked.device}</code>
            </p>
          )}
          {picked.notes && <p className="mt-1 text-xs text-gray-700">{D(`lib.${cell.lib}.${cell.fam}`, picked.notes)}</p>}
          <p className="mt-1 text-xs text-gray-500">
            {picked.version} {picked.source && <Ext href={picked.source}>{host(picked.source)}</Ext>}
          </p>
        </div>
      )}
    </div>
  );
}

// --- 9. desktop tensor computers: Mac Studio, Ryzen AI Max, DGX Spark, RTX Spark ---------------------------

/** the machines compared, with the chip each is built on; a graphics card last, for reference */
const DESKTOPS: { id: string; machine: PageKey }[] = [
  { id: "apple-m5-max", machine: "chips.desktop.macStudio" },
  { id: "apple-m5-ultra", machine: "chips.desktop.macStudio" },
  { id: "apple-m3-ultra", machine: "chips.desktop.macStudio" },
  { id: "amd-ryzen-ai-max-plus-395", machine: "chips.desktop.miniPc" },
  { id: "nvidia-dgx-spark-gb10", machine: "chips.desktop.dgxSpark" },
  { id: "nvidia-rtx-spark", machine: "chips.desktop.rtxSpark" },
  { id: "nvidia-rtx-5090", machine: "chips.desktop.card" },
];
const DESKTOP_SRC = [
  "https://www.apple.com/mac-studio/",
  "https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html",
  "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
  "https://www.nvidia.com/en-us/products/rtx-spark/",
];

/** NVIDIA's Tensor Core and its equivalents: names, where they sit, how code reaches them, first chip with one */
const MATRIX_UNITS: { vendor: string; unit: string; where: string; via: string; first: string; source: string }[] = [
  { vendor: "NVIDIA", unit: "Tensor Core", where: "NVIDIA GPUs", via: "PTX mma, wgmma (Hopper), tcgen05 (Blackwell); cuBLAS, cuDNN", first: "V100 (2017)", source: "https://www.nvidia.com/en-us/data-center/tensor-cores/" },
  { vendor: "AMD", unit: "Matrix Core", where: "AMD Instinct (CDNA)", via: "MFMA instructions; rocBLAS, hipBLASLt", first: "MI100 (2020)", source: "https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/" },
  { vendor: "AMD", unit: "AI Accelerator", where: "AMD Radeon (RDNA 3, RDNA 4)", via: "WMMA instructions", first: "RX 7900 (2022)", source: "https://gpuopen.com/learn/wmma_on_rdna3/" },
  { vendor: "Intel", unit: "XMX", where: "Intel Arc and Data Center GPUs", via: "DPAS instruction; oneDNN, oneMKL", first: "Arc A-series (2022)", source: "https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html" },
  { vendor: "Intel", unit: "AMX (TMUL)", where: "Intel Xeon CPUs", via: "AMX tile instructions; oneDNN", first: "Xeon Sapphire Rapids (2023)", source: "https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html" },
  { vendor: "Arm", unit: "SME / SME2", where: "Arm CPUs (Apple M4 and later)", via: "outer-product instructions into the ZA tile; Accelerate, KleidiAI", first: "Apple M4 (2024)", source: "https://developer.arm.com/documentation/109246/latest/" },
  { vendor: "Apple", unit: "Neural Accelerator", where: "Apple GPU cores (M5 and later)", via: "Metal 4 tensor APIs; MLX", first: "Apple M5 (2025)", source: "https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/" },
  { vendor: "Apple", unit: "Neural Engine", where: "Apple NPU", via: "Core ML", first: "A11 Bionic (2017)", source: "https://developer.apple.com/machine-learning/core-ml/" },
  { vendor: "Google", unit: "MXU (systolic array)", where: "Google TPU TensorCores", via: "XLA (JAX, PyTorch/XLA)", first: "TPU v1 (2015)", source: "https://cloud.google.com/tpu/docs/system-architecture-tpu-vm" },
  { vendor: "AWS", unit: "Tensor Engine (systolic)", where: "AWS NeuronCores (Trainium, Inferentia)", via: "Neuron compiler; NKI", first: "Inferentia (2019)", source: "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html" },
  { vendor: "Huawei", unit: "Cube Unit", where: "Huawei Ascend (Da Vinci cores)", via: "CANN, Ascend C", first: "Ascend 910 (2019)", source: "https://www.huawei.com/us/press-events/news/2019/8/huawei-ascend-910-most-powerful-ai-processor" },
  { vendor: "Qualcomm", unit: "Hexagon tensor accelerator", where: "Qualcomm Hexagon NPU", via: "Qualcomm AI Engine Direct (QNN)", first: "Snapdragon", source: "https://www.qualcomm.com/products/technology/processors/hexagon" },
];

/** the table in order of the first chip's year (rows without a year last) */
const yearOf = (first: string) => Number(first.match(/\((\d{4})\)/)?.[1] ?? 9999);
const BY_YEAR = [...MATRIX_UNITS].sort((a, b) => yearOf(a.first) - yearOf(b.first));

function MatrixUnits() {
  const T = usePageText();
  return (
    <details className="group mt-4 rounded-lg border border-gray-200" data-testid="chips-units">
      <summary className="cursor-pointer list-none px-3 py-2 text-sm font-semibold text-gray-900 select-none">
        <span className="mr-1.5 inline-block text-gray-400 transition-transform group-open:rotate-90">▸</span>
        {T("chips.units.title")}
      </summary>
      <div className="px-3 pb-3">
      <div className="mt-2 overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-left text-xs">
          <thead className="bg-gray-50 text-gray-600">
            <tr>
              {(["vendor", "unit", "where", "via", "first"] as const).map((h) => (
                <th key={h} className="px-2 py-1.5 font-semibold whitespace-nowrap">
                  {T(`chips.units.${h}` as PageKey)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 align-top">
            {BY_YEAR.map((u) => (
              <tr key={u.unit}>
                <td className="px-2 py-1.5 whitespace-nowrap text-gray-700">{u.vendor}</td>
                <td className="px-2 py-1.5 font-semibold whitespace-nowrap text-gray-900">{u.unit}</td>
                <td className="px-2 py-1.5">{u.where}</td>
                <td className="px-2 py-1.5 text-gray-700">{u.via}</td>
                <td className="px-2 py-1.5 whitespace-nowrap">
                  {u.first} <Ext href={u.source}>↗</Ext>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-gray-600">
        <RichText text={T("chips.units.note")} />
      </p>
      </div>
    </details>
  );
}

function DesktopTable({ onSelect }: { onSelect: (id: string) => void }) {
  const T = usePageText();
  const rows = DESKTOPS.flatMap((d) => {
    const c = CHIPS.find((x) => x.id === d.id);
    return c ? [{ c, machine: d.machine }] : [];
  });
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200" data-testid="chips-desktop-table">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-600">
          <tr>
            {(["chip", "memory", "bandwidth", "fits", "decode", "power"] as const).map((h) => (
              <th key={h} className="px-2 py-1.5 font-semibold">
                {T(`chips.desktop.${h}` as PageKey)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {rows.map(({ c, machine }) => {
            const gb = Array.isArray(c.memoryGB) ? c.memoryGB[1] : (c.memoryGB ?? 0);
            // largest model at 4 bits that fits, leaving 20% for the cache, activations and the system
            const fitB = Math.floor((gb * 0.8) / 0.5);
            return (
              <tr key={c.id} className={c.id === "nvidia-rtx-5090" ? "bg-gray-50 text-gray-600" : ""}>
                <td className="px-2 py-1.5">
                  <span className="text-gray-500">{T(machine)}</span>
                  <span className="text-gray-400"> · </span>
                  <button className="font-semibold text-gray-900 hover:text-blue-700" onClick={() => onSelect(c.id)}>
                    {short(c.name)}
                  </button>
                </td>
                <td className="px-2 py-1.5 tabular-nums">{fmtNum(c.memoryGB)}</td>
                <td className="px-2 py-1.5 tabular-nums">{fmtNum(c.bandwidthGBs)}</td>
                <td className="px-2 py-1.5 tabular-nums">{gb ? fill(T("chips.desktop.fitsValue"), { b: fitB }) : "–"}</td>
                <td className="px-2 py-1.5 tabular-nums">{c.bandwidthGBs ? fmtNum(decodeCeiling(c.bandwidthGBs, 70, 0.5), 2) : "–"}</td>
                <td className="px-2 py-1.5 tabular-nums">{fmtNum(c.powerW)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// --- 10. measure your chip ----------------------------------------------------------------------------------

interface Measured {
  device: string;
  deviceType: string;
  bandwidth: { gbs: number };
  matmul: { precision: string; tflops?: number; skipped?: string; slow?: string; n?: number }[];
}

function toChip(m: Measured): Chip {
  const r = (p: string) => m.matmul.find((x) => x.precision === p && x.tflops && !x.slow)?.tflops ?? null;
  return {
    id: "measured",
    vendor: "",
    name: m.device,
    group: "global",
    class: m.deviceType === "cpu" ? "cpu" : "dgpu",
    year: null,
    memoryType: null,
    memoryGB: null,
    bandwidthGBs: m.bandwidth.gbs,
    fp64TF: r("fp64"),
    fp32TF: r("fp32"),
    tf32TF: r("tf32"),
    bf16TF: r("bf16"),
    fp16TF: r("fp16"),
    fp8TF: null,
    int8TOPS: null,
    npuTOPS: null,
    powerW: null,
    matrixUnit: null,
    formats: [],
    pytorchDevice: m.deviceType,
    blas: "",
    kernelLangs: "",
    notes: "",
    confidence: "",
    sources: {},
  };
}

function Measure({ onResult }: { onResult: (c: Chip) => void }) {
  const T = usePageText();
  const [state, setState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [lines, setLines] = useState<string[]>([]);
  const [error, setError] = useState("");
  const run = async () => {
    setState("running");
    setLines([]);
    setError("");
    let result: Measured | null = null;
    try {
      await runProblemStream("others/chip_roofline", { matrix_size: 4096, copy_mb: 512, repeats: 5 }, undefined, false, ({ text }) => {
        if (text.startsWith("CHIP_RESULT ")) result = JSON.parse(text.slice("CHIP_RESULT ".length)) as Measured;
        else setLines((l) => [...l, text]);
      });
      if (!result) throw new Error(T("chips.measure.noResult"));
      onResult(toChip(result));
      setState("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState("error");
    }
  };
  return (
    <div className="mt-4 rounded-lg border border-gray-200 bg-gray-50/60 p-4" data-testid="chips-measure-card">
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => void run()}
          disabled={state === "running"}
          className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          data-testid="chips-measure"
        >
          {state === "running" ? T("chips.measure.running") : T("chips.measure.run")}
        </button>
        <a className="text-sm text-blue-700 hover:underline" href="#/problem/others/chip_roofline">
          {T("chips.measure.open")}
        </a>
      </div>
      {lines.length > 0 && <pre className="mt-3 max-h-48 overflow-auto rounded bg-white p-2 font-mono text-xs text-gray-800">{lines.join("\n")}</pre>}
      {state === "done" && <p className="mt-2 text-sm text-green-800">{T("chips.measure.done")}</p>}
      {state === "error" && (
        <p className="mt-2 text-sm text-red-700">
          {T("chips.measure.error")} {error}
        </p>
      )}
    </div>
  );
}

// --- the tab -----------------------------------------------------------------------------------------------

export default function ChipsTab() {
  const T = usePageText();
  const D = useDataText();
  const [selected, setSelected] = useState<string | null>(null);
  const [precision, setPrecision] = useState<Precision>("bf16");
  const [measured, setMeasured] = useState<Chip | null>(null);
  const selChip = (measured && selected === "measured" ? measured : CHIPS.find((c) => c.id === selected)) ?? null;
  const family = selChip ? familyOf(selChip) : null;
  const china = CHIPS.filter((c) => c.group === "china");
  const boards = CHIPS.filter((c) => c.class === "board" || c.class === "npu" || /raspberry|rk3588|coral|hailo|jetson/i.test(`${c.vendor} ${c.name}`));
  const text = (key: string) => <RichText text={T(key as PageKey)} />;
  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
      <PageToc scrollRoot="skills-scroll" items={SECTIONS.map((key, i) => ({ id: `chips-${key}`, mark: i + 1, label: T(`chips.${key}.title` as PageKey) }))} />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">{T("chips.title")}</h1>
          <p className="mt-3 text-gray-600">{text("chips.intro")}</p>
        </header>

        <Card id="why">
          <Heading n={num("why")} text={T("chips.why.title")} />
          <p className={P}>{text("chips.why.text")}</p>
          <ul className="mt-3 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-2">
            {(["power", "bandwidth", "units", "ecosystem"] as const).map((k) => (
              <li key={k} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`chips.why.${k}` as PageKey)} />
              </li>
            ))}
          </ul>
          <BudgetFigure />
          <MatrixUnits />
          <ChipsAnim id="feed" />
          <p className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm" data-testid="chips-roofline-links">
            <a className="text-blue-700 hover:underline" href={`#${formatPageRoute("concept", "dataflow", "roof")}`}>
              → {T("maths.roof.title")}
            </a>
            <a className="text-blue-700 hover:underline" href="#/problem/others/chip_roofline">
              → {T("chips.measure.title")}
            </a>
          </p>
        </Card>

        <Card id="history">
          <Heading n={num("history")} text={T("chips.history.title")} />
          <p className={P}>
            {T("chips.history.text")
              .split(/\[\[(.+?)\]\]/)
              .map((part, i) =>
                i % 2 ? (
                  <a key={i} className="text-blue-700 hover:underline" href={`#${formatPageRoute("concept", "history")}`}>
                    {part}
                  </a>
                ) : (
                  <RichText key={i} text={part} />
                ),
              )}
          </p>
          <History />
        </Card>

        <Card id="table">
          <Heading n={num("table")} text={T("chips.table.title")} />
          <p className={P}>
            {T("chips.table.text")
              .split(/\[\[(.+?)\]\]/)
              .map((part, i) =>
                i % 2 ? (
                  <span key={i} tabIndex={0} className="group relative cursor-help border-b border-dotted border-gray-500 focus:outline-none" data-testid="chips-sparsity">
                    {part}
                    <span role="tooltip" className="pointer-events-none invisible absolute top-full left-1/2 z-30 mt-1.5 w-72 -translate-x-1/2 rounded-lg border border-gray-200 bg-white p-3 text-xs leading-relaxed font-normal text-gray-700 opacity-0 shadow-lg transition-opacity group-hover:visible group-hover:opacity-100 group-focus:visible group-focus:opacity-100">
                      {T("chips.sparsity.tip")}
                    </span>
                  </span>
                ) : (
                  <RichText key={i} text={part} />
                ),
              )}
          </p>
          <ChipTable selected={selected} onSelect={setSelected} precision={precision} setPrecision={setPrecision} extra={measured} />
          <RooflineMap precision={precision} extra={measured} selected={selected} onSelect={setSelected} />
          <ChipsAnim id="mma" />
        </Card>

        <Card id="precision">
          <Heading n={num("precision")} text={T("chips.precision.title")} />
          <p className={P}>{text("chips.precision.text")}</p>
          <ul className="mt-3 space-y-2 text-sm leading-relaxed text-gray-700">
            {PRECISION_DEFAULTS.slice(0, 6).map((d, i) => (
              <li key={i} className="rounded-lg bg-gray-50 p-3">
                {D(`default.${i}`, d.text)} <Ext href={d.source}>{host(d.source)}</Ext>
              </li>
            ))}
          </ul>
          <CodeBlock code={DEVICE_CODE} />
          <p className={P}>
            <a className="font-medium text-blue-700 hover:underline" href={`#${formatPageRoute("skills", "mixed")}`}>
              {T("chips.precision.more")} →
            </a>
          </p>
        </Card>

        <Card id="under">
          <Heading n={num("under")} text={T("chips.under.title")} />
          <p className={P}>{text("chips.under.text")}</p>
          <UnderTable />
        </Card>

        <Card id="libs">
          <Heading n={num("libs")} text={T("chips.libs.title")} />
          <p className={P}>{text("chips.libs.text")}</p>
          {selChip && family && <p className="mt-2 text-xs text-blue-800">{fill(T("chips.libs.highlight"), { chip: selChip.name, fam: T(`chips.fam.${family}` as PageKey) })}</p>}
          <LibraryTable highlight={family} />
        </Card>

        <Card id="china">
          <Heading n={num("china")} text={T("chips.china.title")} />
          <p className={P}>{text("chips.china.text")}</p>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            {STACKS.map((s) => (
              <div key={s.vendor} className="rounded-lg bg-gray-50 p-3 text-sm" data-testid="chips-stack">
                <b className="text-gray-900">{s.vendor}</b>
                <dl className="mt-1 space-y-0.5 text-xs text-gray-700">
                  {(
                    [
                      ["sdk", s.sdk],
                      ["kernelLanguage", s.kernelLanguage],
                      ["mathLibraries", s.mathLibraries],
                      ["collectives", s.collectives],
                      ["frameworks", s.frameworks],
                    ] as [string, string | null][]
                  )
                    .filter(([, v]) => v)
                    .map(([k, v]) => (
                      <div key={k}>
                        <dt className="inline text-gray-500">{T(`chips.stack.${k}` as PageKey)}: </dt>
                        <dd className="inline">{D(`stack.${s.vendor}.${k}`, v!)}</dd>
                      </div>
                    ))}
                  {s.pytorch && (
                    <div>
                      <dt className="inline text-gray-500">PyTorch: </dt>
                      <dd className="inline">
                        {D(`stack.${s.vendor}.pytorch`, s.pytorch)} {s.pytorchRepo && <Ext href={s.pytorchRepo}>{host(s.pytorchRepo)}</Ext>}
                      </dd>
                    </div>
                  )}
                  {s.opensource && (
                    <div>
                      <dt className="inline text-gray-500">{T("chips.stack.opensource")}: </dt>
                      <dd className="inline">
                        {D(`stack.${s.vendor}.opensource`, s.opensource)} {s.opensourceLink && <Ext href={s.opensourceLink}>{host(s.opensourceLink)}</Ext>}
                      </dd>
                    </div>
                  )}
                </dl>
                {s.source && (
                  <p className="mt-1 text-xs">
                    <Ext href={s.source}>{host(s.source)}</Ext>
                  </p>
                )}
              </div>
            ))}
          </div>
          <p className={P}>{fill(T("chips.china.inTable"), { n: china.length })}</p>
        </Card>

        <Card id="arm">
          <Heading n={num("arm")} text={T("chips.arm.title")} />
          <p className={P}>{text("chips.arm.text")}</p>
          <ul className="mt-3 grid gap-3 text-sm md:grid-cols-2">
            {boards.map((c) => (
              <li key={c.id} className="rounded-lg bg-gray-50 p-3">
                <button className="text-left font-semibold text-gray-900 hover:text-blue-700" onClick={() => (setSelected(c.id), document.getElementById("chips-table")?.scrollIntoView({ behavior: "smooth" }))}>
                  {c.vendor} {c.name}
                </button>
                <p className="mt-1 text-xs text-gray-700">{D(`chip.${c.id}.notes`, c.notes)}</p>
              </li>
            ))}
          </ul>
          <p className={P}>{text("chips.arm.tensorizer")}</p>
        </Card>

        <Card id="desktop">
          <Heading n={num("desktop")} text={T("chips.desktop.title")} />
          <p className={P}>{text("chips.desktop.text")}</p>
          <ul className="mt-3 grid gap-3 text-sm leading-relaxed text-gray-700 md:grid-cols-2">
            {(["mac", "amd", "dgx", "rtx"] as const).map((k) => (
              <li key={k} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`chips.desktop.${k}` as PageKey)} />
              </li>
            ))}
          </ul>
          <DesktopTable onSelect={(id) => (setSelected(id), document.getElementById("chips-table")?.scrollIntoView({ behavior: "smooth" }))} />
          <p className="mt-1 text-[11px] text-gray-500">
            <RichText text={T("chips.desktop.legend")} />
          </p>
          <p className={P}>{text("chips.desktop.verdict")}</p>
          <p className="mt-3 flex flex-wrap gap-x-3 text-xs">
            <span className="text-gray-500">{T("chips.sources")}</span>
            {DESKTOP_SRC.map((u) => (
              <Ext key={u} href={u}>
                {host(u)}
              </Ext>
            ))}
          </p>
        </Card>

        <Card id="measure">
          <Heading n={num("measure")} text={T("chips.measure.title")} />
          <p className={P}>{text("chips.measure.text")}</p>
          <Measure onResult={(c) => (setMeasured(c), setSelected("measured"))} />
        </Card>

        <Card id="choose">
          <Heading n={num("choose")} text={T("chips.choose.title")} />
          <ul className="mt-3 space-y-2 text-sm leading-relaxed text-gray-700">
            {(["laptop", "mac", "board", "nvidia", "cloud", "china", "physics"] as const).map((k) => (
              <li key={k} className="rounded-lg bg-gray-50 p-3">
                <RichText text={T(`chips.choose.${k}` as PageKey)} />
              </li>
            ))}
          </ul>
        </Card>
      </article>
    </div>
  );
}

/** picking the device in PyTorch, and what each backend offers */
const DEVICE_CODE = `import torch

if torch.cuda.is_available():                 # NVIDIA (CUDA), and AMD (ROCm builds use the same name)
    device = torch.device("cuda")
elif torch.backends.mps.is_available():       # Apple GPU: float32/16/bfloat16, no float64
    device = torch.device("mps")
elif hasattr(torch, "xpu") and torch.xpu.is_available():   # Intel GPU
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

dtype = torch.float64 if device.type != "mps" else torch.float32   # physics that needs FP64: check first
x = torch.randn(4096, 4096, device=device, dtype=dtype)`;
