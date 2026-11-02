import { useEffect, useMemo, useState } from "react";
import { usePageText, type PageKey } from "../content/pages";

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

  // the read cursor walks the elements; when playing, the lab moves to the next step after one full pass
  useEffect(() => {
    const timer = window.setTimeout(
      () => {
        if (cursor + 1 < els.length) setCursor(cursor + 1);
        else if (playing) {
          setN((n + 1) % STEPS.length);
          setCursor(0);
        } else setCursor(0);
      },
      cursor + 1 < els.length ? TICK_MS : 1400,
    );
    return () => window.clearTimeout(timer);
  }, [cursor, els.length, playing, n]);

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
      <div className="flex flex-wrap gap-1.5">
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

      <div className="mt-4 grid gap-5 md:grid-cols-[auto_1fr]">
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
        <div className="min-w-0">
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

      <div className="mt-4 flex items-center gap-2 text-xs">
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
      <figcaption className="mt-2 text-[11px] text-gray-500">{L(`note${n}`)}</figcaption>
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
  return `${(bytes / 1e3).toFixed(1)} kB`;
}

/** How big the pairwise tensor D (N, N, 3) gets: slide N and watch it pass typical GPU memory sizes. */
export function PairMemory() {
  const T = usePageText();
  const L = (k: string) => T(`lab.${k}` as PageKey);
  const [exp, setExp] = useState(3); // N = 10^exp
  const n = Math.round(10 ** exp);
  const f32 = 12 * n * n;
  const f64 = 24 * n * n;
  // bar on a log scale from 1 MB to 1 TB
  const pos = (b: number) => `${Math.min(100, Math.max(0, ((Math.log10(b) - 6) / 6) * 100))}%`;

  return (
    <figure className="rounded-lg border border-gray-200 bg-gray-50 p-4" data-testid="pair-memory">
      <label className="flex items-center gap-3 text-xs text-gray-600">
        <span className="font-mono">N = {n.toLocaleString()}</span>
        <input
          type="range"
          min={2}
          max={5}
          step={0.05}
          value={exp}
          onChange={(e) => setExp(Number(e.target.value))}
          className="flex-1 accent-blue-600"
          aria-label={L("particles")}
        />
      </label>
      <div className="relative mt-5 h-6 rounded bg-white ring-1 ring-gray-200">
        <div
          className="absolute inset-y-0 left-0 rounded bg-blue-500/80 transition-all duration-200"
          style={{ width: pos(f32) }}
        />
        <div
          className="absolute inset-y-0 left-0 rounded border-r-2 border-dashed border-violet-500 transition-all duration-200"
          style={{ width: pos(f64) }}
        />
        {GPUS.map((g) => (
          <div key={g.label} className="absolute -top-4 h-10 border-l border-red-400" style={{ left: pos(g.bytes) }}>
            <span className="absolute -top-0.5 left-1 text-[10px] whitespace-nowrap text-red-600">{g.label}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 font-mono text-xs text-gray-700" data-testid="pair-bytes">
        D ({n.toLocaleString()}, {n.toLocaleString()}, 3): float32 {size(f32)} / float64 {size(f64)}
      </p>
      <figcaption className="mt-1 text-[11px] text-gray-500">{L("memCaption")}</figcaption>
    </figure>
  );
}
