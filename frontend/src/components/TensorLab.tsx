import { useEffect, useMemo, useState } from "react";
import { usePageText, type PageKey } from "../content/pages";
import {memoryPercent,MEMORY_TICKS,type MemoryScale} from "../pairMemory";

/** One step of the lab: a view or a new tensor, described by what PyTorch actually stores. */
interface Step {
  code: string;
  shape: number[];
  strides: number[];
  buffer: number[]; // the storage this tensor reads from
  shares: boolean; // the same storage as x (a view)
  newBytes: number; // memory newly allocated by this step (float32 = 4 bytes)
  contiguous: boolean;
}

const X = [0, 1, 2, 3, 4, 5];
const STEPS: Step[] = [
  {
    code: "x = torch.arange(6.).reshape(2, 3)",
    shape: [2, 3],
    strides: [3, 1],
    buffer: X,
    shares: false,
    newBytes: 24,
    contiguous: true,
  },
  { code: "x.T", shape: [3, 2], strides: [1, 3], buffer: X, shares: true, newBytes: 0, contiguous: false },
  { code: "x[:, ::2]", shape: [2, 2], strides: [3, 2], buffer: X, shares: true, newBytes: 0, contiguous: false },
  {
    code: "x[:, None, :]",
    shape: [2, 1, 3],
    strides: [3, 3, 1],
    buffer: X,
    shares: true,
    newBytes: 0,
    contiguous: true,
  },
  {
    code: "x[:, None, :].expand(2, 4, 3)",
    shape: [2, 4, 3],
    strides: [3, 0, 1],
    buffer: X,
    shares: true,
    newBytes: 0,
    contiguous: false,
  },
  {
    code: "x.T.contiguous()",
    shape: [3, 2],
    strides: [2, 1],
    buffer: [0, 3, 1, 4, 2, 5],
    shares: false,
    newBytes: 24,
    contiguous: true,
  },
  {
    code: 'torch.einsum("ij->i", x)',
    shape: [2],
    strides: [1],
    buffer: [3, 12],
    shares: false,
    newBytes: 8,
    contiguous: true,
  },
  {
    code: 'torch.einsum("ij,kj->ik", x, x)',
    shape: [2, 2],
    strides: [2, 1],
    buffer: [5, 14, 14, 50],
    shares: false,
    newBytes: 16,
    contiguous: true,
  },
];

/** Every element in logical (row-major) order, with the storage index its strides point to. */
function elements(step: Step): { idx: number[]; at: number }[] {
  const out: { idx: number[]; at: number }[] = [];
  const total = step.shape.reduce((a, b) => a * b, 1);
  for (let flat = 0; flat < total; flat++) {
    let rest = flat;
    const idx = step.shape.map((_, k) => {
      const below = step.shape.slice(k + 1).reduce((a, b) => a * b, 1);
      const i = Math.floor(rest / below);
      rest %= below;
      return i;
    });
    out.push({ idx, at: idx.reduce((s, i, k) => s + i * step.strides[k], 0) });
  }
  return out;
}

const TICK_MS = 550;

/**
 * Step through views and new tensors of a (2, 3) tensor. A cursor walks the elements in logical order and lights
 * the storage cell each one reads: smooth for contiguous tensors, jumping for a transpose, repeating for stride 0.
 */
export default function TensorLab() {
  const T = usePageText();
  const L = (k: string) => T(`lab.${k}` as PageKey);
  const [n, setN] = useState(0);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(true);
  const step = STEPS[n];
  const els = useMemo(() => elements(step), [step]);

  // while playing, the read cursor walks the elements and starts over on the same step (the user picks the next)
  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(
      () => setCursor(cursor + 1 < els.length ? cursor + 1 : 0),
      cursor + 1 < els.length ? TICK_MS : 1400,
    );
    return () => window.clearTimeout(timer);
  }, [cursor, els.length, playing]);

  const go = (k: number) => {
    setN((k + STEPS.length) % STEPS.length);
    setCursor(0);
  };
  const reads = els.slice(0, cursor + 1).map((e) => e.at);
  const [rows, cols] =
    step.shape.length === 1
      ? [1, step.shape[0]]
      : [step.shape.slice(0, -1).reduce((a, b) => a * b, 1), step.shape.at(-1)!];
  const block = step.shape.length === 3 ? step.shape[1] : 0; // rows per outer index, for a separator

  return (
    <figure className="rounded-lg border border-gray-200 bg-gray-50 p-4" data-testid="tensor-lab">
      <div className="flex flex-wrap justify-center gap-1.5">
        {STEPS.map((s, k) => (
          <button
            key={s.code}
            onClick={() => go(k)}
            className={`rounded-md px-2 py-1 font-mono text-[11px] ${
              k === n ? "bg-blue-600 text-white" : "bg-white text-gray-600 ring-1 ring-gray-200 hover:ring-blue-300"
            }`}
          >
            {s.code.replace("x = torch.arange(6.).reshape(2, 3)", "x")}
          </button>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-start justify-center gap-x-10 gap-y-5">
        {/* logical view */}
        <div>
          <p className="mb-1.5 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">
            {L("logical")} {`(${step.shape.join(", ")}${step.shape.length === 1 ? "," : ""})`}
          </p>
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, 2.25rem)` }}>
            {els.map((e, k) => {
              const row = Math.floor(k / cols);
              return (
                <div
                  key={k}
                  className={`flex h-8 items-center justify-center rounded border text-sm transition-all duration-300 ${
                    k === cursor
                      ? "scale-110 border-amber-500 bg-amber-100 font-semibold text-amber-900 shadow"
                      : "border-blue-300 bg-blue-50 text-blue-900"
                  } ${block && row % block === 0 && row > 0 ? "mt-1.5" : ""}`}
                  title={`[${e.idx.join(", ")}] → storage ${e.at}`}
                >
                  {step.buffer[e.at]}
                </div>
              );
            })}
          </div>
          <p className="mt-1 text-[10px] text-gray-400">
            {rows} × {cols}
            {step.shape.length === 3 ? ` (${step.shape[0]} blocks)` : ""}
          </p>
        </div>

        {/* storage */}
        <div className="min-w-0 max-w-sm">
          <p className="mb-1.5 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">
            {step.shares ? L("storageShared") : L("storageNew")} ({step.buffer.length} × 4 B)
          </p>
          <div className="flex flex-wrap gap-1">
            {step.buffer.map((v, k) => {
              const lit = els[cursor]?.at === k;
              const times = reads.filter((r) => r === k).length;
              return (
                <div
                  key={k}
                  className={`relative flex h-8 w-9 items-center justify-center rounded border text-sm transition-all duration-300 ${
                    lit
                      ? "-translate-y-1 border-amber-500 bg-amber-100 font-semibold text-amber-900 shadow"
                      : step.shares
                        ? "border-indigo-300 bg-indigo-50 text-indigo-900"
                        : "border-emerald-300 bg-emerald-50 text-emerald-900"
                  }`}
                >
                  {v}
                  {times > 1 && (
                    <span className="absolute -top-2 -right-1.5 rounded-full bg-amber-500 px-1 text-[9px] text-white">
                      ×{times}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          <p className="mt-2 font-mono text-[11px] text-gray-600" data-testid="read-order">
            {L("reads")}: {reads.join(" → ")}
          </p>

          <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            <dt className="text-gray-500">{L("shape")}</dt>
            <dd className="font-mono">
              ({step.shape.join(", ")}
              {step.shape.length === 1 ? "," : ""})
            </dd>
            <dt className="text-gray-500">{L("strides")}</dt>
            <dd className="font-mono">
              ({step.strides.join(", ")}
              {step.strides.length === 1 ? "," : ""})
            </dd>
            <dt className="text-gray-500">{L("shares")}</dt>
            <dd className={step.shares ? "text-indigo-700" : "text-gray-700"}>{step.shares ? L("yes") : L("no")}</dd>
            <dt className="text-gray-500">{L("newMemory")}</dt>
            <dd className={step.newBytes ? "text-emerald-700" : "text-gray-700"}>{step.newBytes} B</dd>
            <dt className="text-gray-500">{L("contiguous")}</dt>
            <dd>{step.contiguous ? L("yes") : L("no")}</dd>
          </dl>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-center gap-2 text-xs">
        <button
          onClick={() => go(n - 1)}
          className="rounded-md px-2 py-1 ring-1 ring-gray-300 hover:bg-white"
          aria-label="previous"
        >
          ◀
        </button>
        <button
          onClick={() => setPlaying((p) => !p)}
          className="rounded-md bg-blue-600 px-3 py-1 font-medium text-white hover:bg-blue-700"
          data-testid="lab-play"
        >
          {playing ? L("pause") : L("play")}
        </button>
        <button
          onClick={() => go(n + 1)}
          className="rounded-md px-2 py-1 ring-1 ring-gray-300 hover:bg-white"
          aria-label="next"
        >
          ▶
        </button>
        <span className="ml-2 font-mono text-gray-500">{step.code}</span>
      </div>
      <figcaption className="mt-2 text-center text-[11px] text-gray-500">{L(`note${n}`)}</figcaption>
    </figure>
  );
}

const GB = 1e9;
const GPUS = [
  { label: "8 GB", bytes: 8 * GB },
  { label: "24 GB", bytes: 24 * GB },
  { label: "80 GB", bytes: 80 * GB },
];

function size(bytes: number): string {
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(1)} TB`;
  if (bytes >= GB) return `${(bytes / GB).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} kB`;
  return `${bytes} B`;
}

/** How big the pairwise tensor D (N, N, 3) gets: slide N and watch it pass typical GPU memory sizes. */
export function PairMemory() {
  const T = usePageText();
  const L = (k: string) => T(`lab.${k}` as PageKey);
  const [n, setN] = useState(6);
  const [scale,setScale]=useState<MemoryScale>("linear");
  const f32 = 12 * n * n;
  const f64 = 24 * n * n;
  const pos=(bytes:number)=>`${memoryPercent(bytes,scale)}%`;
  const ticks=MEMORY_TICKS[scale];

  return (
    <figure className="rounded-lg border border-gray-200 bg-gray-50 p-4" data-testid="pair-memory">
      <label className="flex items-center gap-3 text-xs text-gray-600">
        <span className="font-mono">N = {n.toLocaleString()}</span>
        <input
          type="range"
          min={Math.log10(2)}
          max={5}
          step={0.01}
          value={Math.log10(n)}
          onChange={(e) => setN(Math.round(10 ** Number(e.target.value)))}
          className="min-w-0 flex-1 accent-blue-600"
          aria-label={L("particles")}
        />
      </label>
      <div className="mt-2 flex flex-wrap gap-2">
        {[6,12,1000,10000,100000].map(count=><button key={count} type="button" aria-pressed={n===count} onClick={()=>setN(count)} className={`rounded border px-2 py-1 text-xs ${n===count?"border-blue-500 bg-blue-50 text-blue-700":"border-gray-200 bg-white text-gray-600"}`}>{count.toLocaleString()}</button>)}
      </div>
      <PairTensorDots n={n}/>
      <div className="mt-3 flex flex-wrap items-center gap-2" role="group" aria-label={L("memoryScale")}>
        <span className="text-xs text-gray-600">{L("memoryScale")}</span>
        {(["linear","log"] as const).map(value=><button key={value} type="button" data-memory-scale={value} aria-pressed={scale===value} onClick={()=>setScale(value)} className={`rounded border px-2 py-1 text-xs ${scale===value?"border-blue-500 bg-blue-50 text-blue-700":"border-gray-200 bg-white text-gray-600"}`}>{L(value==="linear"?"scaleLinear":"scaleLog")}</button>)}
      </div>
      <p className="mt-2 text-[11px] text-gray-600" data-testid="memory-scale-note">{L(scale==="linear"?"scaleLinearNote":"scaleLogNote")}</p>
      <div className="relative mt-14 h-6 rounded bg-white ring-1 ring-gray-200" data-testid="memory-capacity-plot" data-scale={scale}>
        <div data-memory-dtype="float32" className="absolute inset-y-0 left-0 rounded bg-blue-500/80 transition-all duration-200" style={{width:pos(f32)}} title={`float32: ${size(f32)}`}/>
        <div data-memory-dtype="float64" className="absolute inset-y-0 border-l-2 border-dashed border-violet-500 transition-all duration-200" style={{left:pos(f64)}} title={`float64: ${size(f64)}`}/>
        {GPUS.map((g,i)=><div key={g.label} data-gpu-bytes={g.bytes} className="absolute border-l border-red-400" style={{left:pos(g.bytes),top:-18-i*15,height:42+i*15}}>
          <span className="absolute -top-1 z-10 whitespace-nowrap bg-gray-50 px-1 text-[10px] leading-none text-red-600" style={memoryPercent(g.bytes,scale)>70?{right:1}:{left:1}}>{g.label}</span>
        </div>)}
      </div>
      <div className="relative mt-1 h-6 font-mono text-[10px] text-gray-500" data-testid="memory-axis">
        {ticks.map((bytes,i)=><span key={bytes} data-memory-tick={bytes} className={`absolute top-0 whitespace-nowrap ${scale==="linear"&&i===3?"hidden sm:inline":""}`} style={{left:pos(bytes),transform:i===0?"none":i===ticks.length-1?"translateX(-100%)":"translateX(-50%)"}}><span className="absolute -top-1 h-1 border-l border-gray-300" style={{left:i===0?0:i===ticks.length-1?"100%":"50%"}}/>{bytes===0?"0 B":size(bytes).replace(".0 "," ")}</span>)}
      </div>
      <p className="mt-2 font-mono text-xs text-gray-700" data-testid="pair-bytes">
        D ({n.toLocaleString()}, {n.toLocaleString()}, 3): float32 {size(f32)} / float64 {size(f64)}
      </p>
      <figcaption className="mt-1 text-[11px] text-gray-500">
        {L("memCaption").split(/([,，、.。।])/u).map((part,i)=><span key={i} className={i===0?"text-blue-600":i===2?"text-violet-600":i===4?"text-red-600":undefined}>{part}</span>)}
      </figcaption>
    </figure>
  );
}

/** Each visible dot is one stored scalar, never a scaled area or a whole vector. */
function PairTensorDots({n}:{n:number}) {
  const T=usePageText(),shown=Math.min(n,12),colors=["#2563eb","#d97706","#059669"];
  return <div className="my-4" data-testid="pair-tensor-dots" data-n={n}>
    <div className="grid gap-3 sm:grid-cols-[1fr_2fr]">
      <div className="rounded-lg border border-blue-100 bg-white p-3 text-center">
        <p className="text-xs font-semibold text-blue-700">{T("lab.dotInput")} · X ({n.toLocaleString()}, 3)</p>
        <svg viewBox={`0 0 70 ${shown*10+(n>12?30:14)}`} className="mx-auto my-2 h-36 max-w-full" role="img" aria-label={`X: ${3*n} ${T("lab.dotScalars")}`}>
          {Array.from({length:shown*3},(_,q)=><circle key={q} cx={25+q%3*10} cy={12+Math.floor(q/3)*10} r="2.5" fill="#2563eb"/>)}
          {n>12&&<text x="35" y={shown*10+18} textAnchor="middle" fontSize="15" fill="#2563eb">⋮</text>}
        </svg>
        <p className="font-mono text-xs text-blue-700">3N = {(3*n).toLocaleString()}</p>
      </div>
      <div className="min-w-0 rounded-lg border border-gray-200 bg-white p-3 text-center">
        <p className="text-xs font-semibold text-gray-700">{T("lab.dotOutput")} · D ({n.toLocaleString()}, {n.toLocaleString()}, 3)</p>
        <div className="my-2 grid grid-cols-3 gap-2">
          {colors.map((color,k)=><div key={color} className="min-w-0"><p className="font-mono text-[11px]" style={{color}}>{["x","y","z"][k]}</p><p className="mb-1 flex flex-wrap justify-center gap-x-0.5 font-mono text-[10px]" style={{color}}><span>{n.toLocaleString()}</span><span>×</span><span>{n.toLocaleString()}</span></p><svg viewBox={`0 0 ${shown*10+(n>12?30:10)} ${shown*10+(n>12?30:10)}`} className="mx-auto aspect-square w-full max-w-36" role="img" aria-label={`D[..., ${k}]: ${n*n} ${T("lab.dotScalars")}`}>
            {Array.from({length:shown*shown},(_,q)=><circle key={q} cx={10+q%shown*10} cy={10+Math.floor(q/shown)*10} r="2.5" fill={color}/>)}
            {n>12&&<g fill={color} fontSize="15" textAnchor="middle"><text x={shown*10+15} y={shown*5+10}>⋯</text><text x={shown*5+5} y={shown*10+20}>⋮</text><text x={shown*10+15} y={shown*10+20}>⋱</text></g>}
          </svg></div>)}
        </div>
        <p className="font-mono text-xs text-gray-700">3N² = {(3*n*n).toLocaleString()}</p>
      </div>
    </div>
    <p className="mt-2 text-xs text-gray-600">{T("lab.dotLegend")}</p>
    {n>12&&<p className="mt-1 text-xs text-amber-700">{T("lab.dotSample")}</p>}
    <p className="mt-2 rounded bg-blue-50 px-3 py-2 text-xs text-blue-900" aria-live="polite">{T("lab.dotWork")}: {(3*n*n).toLocaleString()}</p>
  </div>;
}
