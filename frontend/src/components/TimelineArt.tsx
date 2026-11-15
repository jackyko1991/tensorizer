/**
 * Small drawings for the major events of the Acceleration chipsets timeline (Coding Skills ▸ Acceleration chipsets
 * ▸ How we got here). Schematic pictures drawn for the site, no photographs: each shows the idea the event brought.
 */
import type { ReactNode } from "react";

const BLUE = "#2563eb";
const GREEN = "#16a34a";
const VIOLET = "#7c3aed";
const AMBER = "#f59e0b";
const RED = "#dc2626";
const GREY = "#94a3b8";
const INK = "#374151";

const T = ({ x, y, children, size = 7, color = INK, anchor = "middle", weight }: { x: number; y: number; children: ReactNode; size?: number; color?: string; anchor?: "start" | "middle" | "end"; weight?: number }) => (
  <text x={x} y={y} fontSize={size} fill={color} textAnchor={anchor} fontWeight={weight} fontFamily="ui-sans-serif, system-ui, sans-serif">
    {children}
  </text>
);

/** a strip of bits: sign, exponent, mantissa */
function Bits({ x, y, e, m, w = 6 }: { x: number; y: number; e: number; m: number; w?: number }) {
  const cells = [RED, ...Array(e).fill(BLUE), ...Array(m).fill(GREEN)];
  return (
    <g>
      {cells.map((c, i) => (
        <rect key={i} x={x + i * w} y={y} width={w - 1} height={w + 3} rx={0.8} fill={c} opacity={0.85} />
      ))}
    </g>
  );
}

const ART: Record<string, () => ReactNode> = {
  // y ← αx + y: three column vectors
  blas1: () => (
    <g>
      {[22, 52, 92].map((x, j) => (
        <g key={x}>
          {[0, 1, 2, 3, 4].map((i) => (
            <rect key={i} x={x} y={8 + i * 8} width={10} height={7} rx={1} fill={j === 0 ? BLUE : j === 1 ? "#93c5fd" : "#bfdbfe"} />
          ))}
        </g>
      ))}
      <T x={42} y={30} size={9}>
        ←
      </T>
      <T x={77} y={30} size={8}>
        αx +
      </T>
      <T x={60} y={58} size={6.5} color={GREY}>
        y ← αx + y
      </T>
    </g>
  ),
  // the IEEE 754 single-precision layout
  "ieee754-1985": () => (
    <g>
      <Bits x={12} y={18} e={8} m={8} w={6} />
      <T x={15} y={44} size={6} color={RED}>
        s
      </T>
      <T x={39} y={44} size={6} color={BLUE}>
        exponent
      </T>
      <T x={87} y={44} size={6} color={GREEN}>
        mantissa …
      </T>
      <T x={60} y={58} size={6.5} color={GREY}>
        (−1)ˢ · 2ᵉ · 1.m
      </T>
    </g>
  ),
  // a grid of thread blocks
  cuda: () => (
    <g>
      {[0, 1, 2].map((bx) =>
        [0, 1].map((by) => (
          <g key={`${bx}${by}`}>
            <rect x={14 + bx * 32} y={6 + by * 24} width={28} height={20} rx={2} fill="#dcfce7" stroke={GREEN} strokeWidth={0.6} />
            {Array.from({ length: 12 }, (_, i) => (
              <path key={i} d={`M${18 + bx * 32 + (i % 4) * 6} ${10 + by * 24 + Math.floor(i / 4) * 5} q1.5 3 0 4`} stroke={GREEN} strokeWidth={0.8} fill="none" />
            ))}
          </g>
        )),
      )}
      <T x={60} y={60} size={6.5} color={GREY}>
        grid · blocks · threads
      </T>
    </g>
  ),
  // stacked convolution layers on two GPUs
  alexnet: () => (
    <g>
      {[0, 1, 2, 3, 4].map((i) => {
        const h = 40 - i * 6;
        const x = 10 + i * 20;
        return (
          <g key={i}>
            <path d={`M${x} ${28 - h / 2} l6 -4 v${h} l-6 4 z`} fill="#bfdbfe" stroke={BLUE} strokeWidth={0.5} />
            <rect x={x - 8} y={28 - h / 2} width={8} height={h} fill="#dbeafe" stroke={BLUE} strokeWidth={0.5} />
          </g>
        );
      })}
      <rect x={100} y={12} width={14} height={14} rx={2} fill={GREEN} />
      <rect x={100} y={30} width={14} height={14} rx={2} fill={GREEN} />
      <T x={107} y={22} size={5} color="#fff" weight={700}>
        GPU
      </T>
      <T x={107} y={40} size={5} color="#fff" weight={700}>
        GPU
      </T>
      <T x={60} y={60} size={6.5} color={GREY}>
        ImageNet, 2 × GTX 580
      </T>
    </g>
  ),
  // a systolic array: data flows right and down
  tpu1: () => (
    <g>
      {Array.from({ length: 25 }, (_, i) => (
        <rect key={i} x={36 + (i % 5) * 10} y={4 + Math.floor(i / 5) * 10} width={8} height={8} rx={1} fill="#ede9fe" stroke={VIOLET} strokeWidth={0.5} />
      ))}
      {[0, 1, 2, 3, 4].map((i) => (
        <g key={i}>
          <path d={`M22 ${8 + i * 10} h11`} stroke={BLUE} strokeWidth={0.8} markerEnd="url(#ta-arrow)" />
          <path d={`M${40 + i * 10} 56 v0`} />
        </g>
      ))}
      {[0, 1, 2, 3, 4].map((i) => (
        <path key={i} d={`M${40 + i * 10} 54 v4`} stroke={AMBER} strokeWidth={0.8} />
      ))}
      <T x={14} y={32} size={5.5} color={BLUE}>
        x
      </T>
      <T x={102} y={32} size={6} color={GREY} anchor="start">
        MXU
      </T>
    </g>
  ),
  // D = A·B + C as a cube
  volta: () => (
    <g>
      <path d="M40 22 l20 -10 l20 10 l-20 10 z" fill="#ede9fe" stroke={VIOLET} strokeWidth={0.6} />
      <path d="M40 22 v20 l20 10 v-20 z" fill="#ccfbf1" stroke="#0f766e" strokeWidth={0.6} />
      <path d="M80 22 v20 l-20 10 v-20 z" fill="#dcfce7" stroke={GREEN} strokeWidth={0.6} />
      {[1, 2, 3].map((i) => (
        <g key={i} stroke="#94a3b8" strokeWidth={0.3}>
          <path d={`M40 ${22 + i * 5} l20 10`} />
          <path d={`M${40 + i * 5} ${22 + i * 2.5} v20`} />
          <path d={`M80 ${22 + i * 5} l-20 10`} />
        </g>
      ))}
      <T x={60} y={60} size={6.5} color={GREY}>
        D = A·B + C, 4×4×4
      </T>
    </g>
  ),
  // one package: CPU, GPU, Neural Engine around one memory
  m1: () => (
    <g>
      <rect x={24} y={4} width={72} height={46} rx={4} fill="#f8fafc" stroke={INK} strokeWidth={0.7} />
      <rect x={30} y={10} width={26} height={16} rx={1.5} fill="#dbeafe" stroke={BLUE} strokeWidth={0.5} />
      <rect x={60} y={10} width={30} height={16} rx={1.5} fill="#dcfce7" stroke={GREEN} strokeWidth={0.5} />
      <rect x={30} y={30} width={18} height={14} rx={1.5} fill="#ede9fe" stroke={VIOLET} strokeWidth={0.5} />
      <rect x={52} y={30} width={38} height={14} rx={1.5} fill="#fde68a" stroke={AMBER} strokeWidth={0.5} />
      <T x={43} y={20} size={5.5} color={BLUE}>
        CPU
      </T>
      <T x={75} y={20} size={5.5} color={GREEN}>
        GPU
      </T>
      <T x={39} y={39} size={5} color={VIOLET}>
        NPU
      </T>
      <T x={71} y={39} size={5.5} color="#92400e">
        memory
      </T>
      <T x={60} y={60} size={6.5} color={GREY}>
        unified memory
      </T>
    </g>
  ),
  // the two FP8 layouts
  "h100-fp8": () => (
    <g>
      <Bits x={34} y={10} e={4} m={3} w={7} />
      <Bits x={34} y={28} e={5} m={2} w={7} />
      <T x={30} y={17} size={6} anchor="end">
        E4M3
      </T>
      <T x={30} y={35} size={6} anchor="end">
        E5M2
      </T>
      <T x={60} y={60} size={6.5} color={GREY}>
        FP8 + Transformer Engine
      </T>
    </g>
  ),
  // two dies as one GPU, FP4 values
  blackwell: () => (
    <g>
      <rect x={18} y={8} width={38} height={36} rx={2} fill="#dcfce7" stroke={GREEN} strokeWidth={0.7} />
      <rect x={64} y={8} width={38} height={36} rx={2} fill="#dcfce7" stroke={GREEN} strokeWidth={0.7} />
      <path d="M56 20 h8 M56 26 h8 M56 32 h8" stroke={INK} strokeWidth={0.8} />
      <Bits x={25} y={22} e={2} m={1} w={7} />
      <Bits x={71} y={22} e={2} m={1} w={7} />
      <T x={60} y={60} size={6.5} color={GREY}>
        2 dies · FP4 (E2M1)
      </T>
    </g>
  ),
};

/** ids of the events that always show a card with a drawing */
export const MAJOR_EVENTS = Object.keys(ART);

export default function TimelineArt({ id }: { id: string }) {
  const draw = ART[id];
  if (!draw) return null;
  return (
    <svg viewBox="0 0 120 64" className="h-auto w-full" aria-hidden>
      <defs>
        <marker id="ta-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" fill={BLUE} />
        </marker>
      </defs>
      {draw()}
    </svg>
  );
}
