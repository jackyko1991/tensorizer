/**
 * Animations of the NVLink & NVSwitch tab (one or more per card), on the step frame shared with the BLAS tab.
 * Bandwidths are the H100 / NVLink numbers of the tab's own table (NVIDIA documents cited there); times are
 * ideal (bandwidth only). Plans and numbers: ../blasNvlinkAnim.ts.
 */
import { type ReactNode } from "react";
import { type PageKey } from "../content/pages";
import {
  GIB,
  H100_LINKS,
  NV_GENS,
  NV_PLAN,
  PAIRS,
  PCIE_GENS,
  genTotal,
  moveSeconds,
  shortestPath,
  stepMs,
  treeDepth,
  type NvCase,
} from "../blasNvlinkAnim";
import { MESH_EDGES, fmtSeconds, ringAllReduceSeconds, ringLatencySteps, treeLatencySteps } from "../nvlinkModel";
import { COL, Cell, Txt, blueScale } from "./animKit";
import { Arrow, Box, StepAnim, W, along, fill, type DrawProps } from "./BlasAnim";

const GPU_COL = [COL.blue, COL.cur, COL.green, COL.violet];
const GPU_LIGHT = [COL.blueLight, COL.curLight, COL.greenLight, COL.violetLight];

function Gpu({ x, y, w = 64, h = 30, i, k, hot }: { x: number; y: number; w?: number; h?: number; i: number; k: DrawProps["k"]; hot?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={hot ? COL.curLight : "#fff"} stroke="#334155" strokeWidth={1.1} />
      <Txt x={x + w / 2} y={y + h / 2 + 3} size={8} anchor="middle" weight={600}>
        {fill(k("gpu"), { i })}
      </Txt>
    </g>
  );
}

/** Small squares travelling along a path; `every` = one packet per that many slots (fewer = more bandwidth). */
function Packets({ pts, t, every, color = COL.cur, slots = 10 }: { pts: [number, number][]; t: number; every: number; color?: string; slots?: number }) {
  return (
    <g>
      {Array.from({ length: slots }, (_, j) => {
        if ((j + slots * 8 - t) % every !== 0) return null;
        const [x, y] = along(pts, (j + 0.5) / slots);
        return <rect key={j} x={x - 3} y={y - 3} width={6} height={6} rx={1} fill={color} />;
      })}
    </g>
  );
}

// --- 1. the PCIe path and the NVLink path ----------------------------------------------------------------

const PCIE_MS = moveSeconds(1e9, H100_LINKS.pcie5) * 1e3; // 15.6 ms
const NVL_MS = moveSeconds(1e9, H100_LINKS.nvlink4) * 1e3; // 2.2 ms

function PathPic({ step, t, f, k }: DrawProps) {
  const g0 = { x: 12, y: 74 };
  const g1 = { x: 204, y: 74 };
  const pciePts: [number, number][] = [
    [g0.x + 32, g0.y],
    [g0.x + 32, 52],
    [140, 52],
    [g1.x + 32, 52],
    [g1.x + 32, g1.y],
  ];
  const nvlPts: [number, number][] = [
    [g0.x + 64, g0.y + 15],
    [g1.x, g1.y + 15],
  ];
  const X = (ms: number) => 70 + (ms / 16) * 196;
  const now = f * 16;
  const bars = [
    { label: k("pcie"), ms: PCIE_MS, col: "#9ca3af", on: step !== 1 },
    { label: k("nvl"), ms: NVL_MS, col: COL.blue, on: step !== 0 },
  ];
  return (
    <g>
      <Box x={92} y={6} w={96} h={18} label={k("host")} size={6.8} fill={COL.idle} />
      <line x1={140} y1={24} x2={140} y2={44} stroke="#9ca3af" strokeWidth={2} />
      <polyline points={pciePts.map((p) => p.join(",")).join(" ")} fill="none" stroke="#9ca3af" strokeWidth={step === 1 ? 1.5 : 2.5} />
      <line x1={nvlPts[0][0]} y1={nvlPts[0][1]} x2={nvlPts[1][0]} y2={nvlPts[1][1]} stroke={COL.blue} strokeWidth={step === 0 ? 2 : 5} opacity={step === 0 ? 0.35 : 1} />
      <Gpu x={g0.x} y={g0.y} i={0} k={k} />
      <Gpu x={g1.x} y={g1.y} i={1} k={k} />
      {step !== 1 && <Packets pts={pciePts} t={t} every={7} color="#6b7280" slots={14} />}
      {step !== 0 && <Packets pts={nvlPts} t={t} every={1} color={COL.cur} slots={14} />}
      {/* drawn after the packets: they pass through (behind) the switch, never over its label */}
      <Box x={108} y={44} w={64} h={16} label={k("sw")} size={6.5} fill="#fff" />
      {bars.map((b, i) => {
        const y = 116 + i * 14;
        const frac = b.on ? Math.min(1, now / b.ms) : 0;
        return (
          <g key={i} opacity={b.on ? 1 : 0.3}>
            <Txt x={66} y={y + 8} size={6.3} anchor="end">
              {b.label}
            </Txt>
            <rect x={X(0)} y={y} width={X(16) - X(0)} height={10} fill={COL.idle} />
            <rect x={X(0)} y={y} width={(X(b.ms) - X(0)) * frac} height={10} fill={b.col} />
            {frac >= 1 && (
              <Txt x={X(b.ms) + 3} y={y + 8} size={6.5} kind="mono" weight={600}>
                {`${b.ms.toFixed(1)} ms`}
              </Txt>
            )}
          </g>
        );
      })}
      {[0, 4, 8, 12, 16].map((v) => (
        <Txt key={v} x={X(v)} y={152} size={6} anchor="middle" color={COL.muted}>
          {String(v)}
        </Txt>
      ))}
      <Txt x={168} y={162} size={6.5} anchor="middle">
        {k("axis")}
      </Txt>
    </g>
  );
}

// --- 1. data, tensor and pipeline parallel ---------------------------------------------------------------

function Grid({ x, y, rows, cols, s = 7, color, mask, label }: {
  x: number;
  y: number;
  rows: number;
  cols: number;
  s?: number;
  color: (r: number, c: number) => string;
  mask?: (r: number, c: number) => boolean;
  label?: string;
}) {
  return (
    <g>
      {label && (
        <Txt x={x + (cols * s) / 2} y={y - 3} size={7.5} anchor="middle" kind="math">
          {label}
        </Txt>
      )}
      {Array.from({ length: rows * cols }, (_, q) => {
        const r = Math.floor(q / cols);
        const c = q % cols;
        const on = mask ? mask(r, c) : true;
        return <rect key={q} x={x + c * s} y={y + r * s} width={s - 0.6} height={s - 0.6} rx={0.8} fill={on ? color(r, c) : "#fff"} stroke={on ? "none" : COL.idleLine} strokeWidth={0.5} strokeDasharray={on ? undefined : "1.5 1"} />;
      })}
    </g>
  );
}

function Parallel({ step, f, k, T }: DrawProps) {
  const colX = [8, 146];
  const top = 40;
  const title = k(["dp", "tp", "pp"][step]);
  const s = 7;
  const phase2 = f > 0.45;
  const phase3 = f > 0.7;
  const gpuBox = (g: number) => (
    <g key={`g${g}`}>
      <rect x={colX[g]} y={18} width={126} height={118} rx={5} fill="#fff" stroke="#cbd5e1" />
      <Txt x={colX[g] + 63} y={28} size={7.5} anchor="middle" weight={600}>
        {fill(k("gpu"), { i: g })}
      </Txt>
    </g>
  );
  const body: ReactNode[] = [];
  if (step === 0) {
    for (const g of [0, 1]) {
      const x = colX[g] + 10;
      body.push(
        <g key={`dp${g}`}>
          <Grid x={x} y={top} rows={2} cols={4} s={s} color={() => GPU_LIGHT[g]} label="X" />
          <Txt x={x + 14} y={top + 24} size={6} anchor="middle" color={COL.muted}>
            {k("batch")}
          </Txt>
          <Grid x={x + 40} y={top} rows={4} cols={4} s={s} color={() => blueScale(0.35)} label="W" />
          <Grid x={x + 80} y={top} rows={2} cols={4} s={s} color={() => GPU_LIGHT[g]} label="Y" />
          {phase2 && (
            <g>
              <Grid x={x + 40} y={top + 54} rows={4} cols={4} s={s} color={() => (phase3 ? blueScale(0.7) : GPU_COL[g])} label="∂W" />
              <Txt x={x + 54} y={top + 92} size={6} anchor="middle" color={COL.muted}>
                {k("grad")}
              </Txt>
            </g>
          )}
        </g>,
      );
    }
    if (phase2)
      body.push(
        <g key="ar">
          <Arrow x1={colX[0] + 84} y1={top + 66} x2={colX[1] + 46} y2={top + 66} color={phase3 ? COL.green : COL.cur} w={1.6} />
          <Arrow x1={colX[1] + 46} y1={top + 74} x2={colX[0] + 84} y2={top + 74} color={phase3 ? COL.green : COL.cur} w={1.6} />
          <rect x={104} y={top + 79} width={72} height={10} rx={2} fill="#fff" />
          <Txt x={140} y={top + 86} size={6.5} anchor="middle" color={COL.text}>
            {T("nvlink.coll.allreduce.name")}
          </Txt>
        </g>,
      );
  } else if (step === 1) {
    for (const g of [0, 1]) {
      const x = colX[g] + 10;
      const mine = (_r: number, c: number) => (g === 0 ? c < 2 : c >= 2);
      body.push(
        <g key={`tp${g}`}>
          <Grid x={x} y={top} rows={4} cols={4} s={s} color={() => COL.blueLight} label="X" />
          <Grid x={x + 40} y={top} rows={4} cols={4} s={s} color={() => GPU_COL[g]} mask={mine} label="W" />
          <Grid x={x + 80} y={top} rows={4} cols={4} s={s} color={() => GPU_LIGHT[g]} mask={phase2 ? mine : () => false} label="Y" />
          {phase3 && <Grid x={x + 40} y={top + 60} rows={4} cols={4} s={s} color={(_r, c) => GPU_LIGHT[c < 2 ? 0 : 1]} label="Y" />}
        </g>,
      );
    }
    if (phase3)
      body.push(
        <g key="ag">
          <Arrow x1={colX[0] + 84} y1={top + 70} x2={colX[1] + 46} y2={top + 70} color={COL.cur} w={1.6} />
          <Arrow x1={colX[1] + 46} y1={top + 78} x2={colX[0] + 84} y2={top + 78} color={COL.cur} w={1.6} />
          <rect x={104} y={top + 83} width={72} height={10} rx={2} fill="#fff" />
          <Txt x={140} y={top + 90} size={6.5} anchor="middle" color={COL.text}>
            {T("nvlink.coll.allgather.name")}
          </Txt>
        </g>,
      );
  } else {
    const x0 = colX[0] + 10;
    const x1 = colX[1] + 10;
    body.push(
      <g key="pp">
        <Grid x={x0} y={top} rows={4} cols={4} s={s} color={() => COL.blueLight} label="X" />
        <Grid x={x0 + 40} y={top} rows={4} cols={4} s={s} color={() => GPU_COL[0]} label="W₁" />
        <Txt x={x0 + 54} y={top + 40} size={6.5} anchor="middle" color={COL.muted}>
          {fill(k("layer"), { i: 1 })}
        </Txt>
        {phase2 && <Grid x={x0 + 80} y={top} rows={4} cols={4} s={s} color={() => COL.curLight} label="H" />}
        <Grid x={x1 + 40} y={top} rows={4} cols={4} s={s} color={() => GPU_COL[1]} label="W₂" />
        <Txt x={x1 + 54} y={top + 40} size={6.5} anchor="middle" color={COL.muted}>
          {fill(k("layer"), { i: 2 })}
        </Txt>
        {phase3 && (
          <g>
            <Grid x={x1} y={top} rows={4} cols={4} s={s} color={() => COL.curLight} label="H" />
            <Grid x={x1 + 80} y={top} rows={4} cols={4} s={s} color={() => GPU_LIGHT[1]} label="Y" />
          </g>
        )}
        {phase2 && (
          <g>
            <Arrow x1={x0 + 94} y1={top + 50} x2={x1 + 14} y2={top + 50} color={COL.cur} w={1.6} />
            <rect x={104} y={top + 55} width={72} height={10} rx={2} fill="#fff" />
            <Txt x={140} y={top + 62} size={6.5} anchor="middle">
              {k("act")}
            </Txt>
          </g>
        )}
      </g>,
    );
  }
  return (
    <g>
      <Txt x={W / 2} y={11} size={8.5} anchor="middle" weight={600} color={COL.blue}>
        {title}
      </Txt>
      {[0, 1].map(gpuBox)}
      {body}
    </g>
  );
}

// --- 2. generations: links x rate per link ---------------------------------------------------------------

function Gens({ step, f, k }: DrawProps) {
  const X = (v: number) => 74 + (v / 2000) * 190;
  const rowY = (i: number) => 22 + i * 12.5;
  return (
    <g>
      <Txt x={8} y={10} size={6.5} color={COL.muted}>
        {k("link")}
      </Txt>
      {[0, 500, 1000, 1500, 2000].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={18} y2={rowY(9)} stroke="#e5e7eb" />
          <Txt x={X(v)} y={rowY(9) + 7} size={6.3} anchor="middle" color={COL.muted}>
            {v.toLocaleString("en")}
          </Txt>
        </g>
      ))}
      {NV_GENS.map((g, i) => {
        if (i > step) return null;
        const seg = 2 * g.perDir;
        const shownLinks = i === step && step < 5 ? Math.ceil(f * g.links) : g.links;
        const y = rowY(i);
        return (
          <g key={g.gen}>
            <Txt x={70} y={y + 7} size={6.3} anchor="end">
              {`${g.gen} (${g.gpu})`}
            </Txt>
            {Array.from({ length: shownLinks }, (_, l) => (
              <rect key={l} x={X(l * seg)} y={y} width={Math.max(0.6, X(seg) - X(0) - 0.7)} height={9} fill={i === step && step < 5 ? COL.blue : blueScale(0.55)} />
            ))}
            {shownLinks === g.links && (
              <Txt x={X(genTotal(g)) + 3} y={y + 7} size={6.3} kind="mono" weight={600}>
                {`${g.links}×${seg} = ${genTotal(g).toLocaleString("en")}`}
              </Txt>
            )}
          </g>
        );
      })}
      {step === 5 &&
        PCIE_GENS.map((p, i) => {
          const y = rowY(5 + i);
          return (
            <g key={p.gen}>
              <Txt x={70} y={y + 7} size={6.3} anchor="end">
                {`${p.gen} x16`}
              </Txt>
              <rect x={X(0)} y={y} width={(X(p.total) - X(0)) * Math.min(1, f * 1.5)} height={9} fill="#9ca3af" />
              <Txt x={X(p.total) + 3} y={y + 7} size={6.3} kind="mono">
                {String(p.total)}
              </Txt>
            </g>
          );
        })}
      <Txt x={169} y={rowY(9) + 18} size={6.8} anchor="middle">
        {k("axis")}
      </Txt>
    </g>
  );
}

// --- 3. hops: cube-mesh vs NVSwitch ----------------------------------------------------------------------

const FRONT: [number, number][] = [
  [20, 64],
  [76, 64],
  [76, 120],
  [20, 120],
];
const MESH_POS: [number, number][] = [...FRONT, ...FRONT.map(([x, y]) => [x + 36, y - 30] as [number, number])];
const SW_X = (i: number) => 160 + (i % 4) * 34;
const SW_Y = (i: number) => (i < 4 ? 46 : 132);

function Hops({ step, t, k }: DrawProps) {
  const pair: [number, number] = step === 0 ? [0, 1] : step === 1 ? [0, 5] : PAIRS[Math.min(t, PAIRS.length - 1)];
  const path = step === 1 ? [0, 4, 5] : shortestPath(MESH_EDGES, pair[0], pair[1]);
  const onPath = new Set(path.slice(1).map((v, i) => [path[i], v].sort().join("-")));
  const nh = path.length - 1;
  const hot = nh === 2 ? COL.red : COL.cur;
  const tally = [0, 0];
  if (step === 2) for (const [a, b] of PAIRS.slice(0, Math.min(t, PAIRS.length - 1) + 1)) tally[shortestPath(MESH_EDGES, a, b).length - 2]++;
  const r = 9;
  return (
    <g>
      <Txt x={66} y={12} size={8} anchor="middle" weight={600}>
        {k("mesh")}
      </Txt>
      <Txt x={211} y={12} size={8} anchor="middle" weight={600}>
        {k("switch")}
      </Txt>
      <line x1={140} x2={140} y1={18} y2={150} stroke="#e5e7eb" />
      {MESH_EDGES.map(([a, b]) => {
        const on = onPath.has([a, b].sort().join("-"));
        return <line key={`${a}-${b}`} x1={MESH_POS[a][0]} y1={MESH_POS[a][1]} x2={MESH_POS[b][0]} y2={MESH_POS[b][1]} stroke={on ? hot : "#bfdbfe"} strokeWidth={on ? 3 : 1.3} />;
      })}
      {MESH_POS.map(([x, y], i) => (
        <g key={i}>
          <rect x={x - r} y={y - r} width={2 * r} height={2 * r} rx={3} fill={pair.includes(i) ? COL.curLight : path.includes(i) ? COL.redLight : "#fff"} stroke="#334155" strokeWidth={1.1} />
          <Txt x={x} y={y + 3} size={7.5} anchor="middle">
            {String(i)}
          </Txt>
        </g>
      ))}
      {Array.from({ length: 8 }, (_, i) => (
        <g key={`s${i}`}>
          <line x1={SW_X(i)} x2={SW_X(i)} y1={SW_Y(i)} y2={89} stroke={pair.includes(i) ? COL.cur : "#bfdbfe"} strokeWidth={pair.includes(i) ? 3 : 1.3} />
          <rect x={SW_X(i) - r} y={SW_Y(i) - r} width={2 * r} height={2 * r} rx={3} fill={pair.includes(i) ? COL.curLight : "#fff"} stroke="#334155" strokeWidth={1.1} />
          <Txt x={SW_X(i)} y={SW_Y(i) + 3} size={7.5} anchor="middle">
            {String(i)}
          </Txt>
        </g>
      ))}
      <rect x={150} y={81} width={124} height={16} rx={4} fill="#334155" />
      <Txt x={212} y={92} size={7.5} anchor="middle" color="#fff" weight={600}>
        NVSwitch
      </Txt>
      <Txt x={66} y={146} size={7.5} anchor="middle" kind="mono" weight={600} color={nh === 2 ? COL.red : COL.text}>
        {`${pair[0]} → ${pair[1]}: ${k(nh === 2 ? "two" : "one")}`}
      </Txt>
      <Txt x={211} y={160} size={7.5} anchor="middle" kind="mono" weight={600}>
        {`${pair[0]} → ${pair[1]}: ${k("one")}`}
      </Txt>
      {step === 2 && (
        <g>
          <Txt x={66} y={160} size={7} anchor="middle" color={COL.muted}>
            {`${k("one")}: ${tally[0]} · ${k("two")}: ${tally[1]}`}
          </Txt>
        </g>
      )}
    </g>
  );
}

// --- 4. the four collectives -----------------------------------------------------------------------------

type Chunk = { src: number; sum?: boolean } | null;
const COLL_IDS = ["allreduce", "allgather", "reducescatter", "broadcast"] as const;
function collState(op: (typeof COLL_IDS)[number], after: boolean, g: number, c: number): Chunk {
  if (op === "allreduce") return after ? { src: -1, sum: true } : { src: g };
  if (op === "allgather") return after ? { src: c } : c === g ? { src: g } : null;
  if (op === "reducescatter") return after ? (c === g ? { src: -1, sum: true } : null) : { src: g };
  return after || g === 0 ? { src: 0 } : null;
}

function Colls({ step, f, k, T }: DrawProps) {
  const op = COLL_IDS[step];
  const showAfter = f > 0.6;
  const moving = f > 0.3 && f <= 0.6;
  const group = (x0: number, after: boolean) => (
    <g opacity={after && !showAfter ? 0.15 : 1}>
      <Txt x={x0 + 66} y={30} size={7} anchor="middle" color={COL.muted}>
        {k(after ? "after" : "before")}
      </Txt>
      {[0, 1, 2, 3].map((g) => (
        <g key={g}>
          <Txt x={x0 + 26} y={47 + g * 26} size={6.8} anchor="end">
            {fill(k("gpu"), { i: g })}
          </Txt>
          {[0, 1, 2, 3].map((c) => {
            const ch = collState(op, after, g, c);
            const x = x0 + 30 + c * 22;
            const y = 36 + g * 26;
            if (!ch) return <rect key={c} x={x} y={y} width={20} height={16} rx={2} fill="#fff" stroke={COL.idleLine} strokeDasharray="2 1.5" />;
            const label = ch.sum ? `Σ${c}` : `${"abcd"[ch.src]}${c}`;
            return <Cell key={c} x={x} y={y} s={20} fill={ch.sum ? "#1e3a8a" : GPU_LIGHT[ch.src]} stroke={ch.sum ? "#1e3a8a" : GPU_COL[ch.src]} label={label} labelColor={ch.sum ? "#fff" : COL.text} size={7} />;
          })}
        </g>
      ))}
    </g>
  );
  return (
    <g>
      <Txt x={W / 2} y={12} size={8.5} anchor="middle" weight={600} color={COL.blue}>
        {T(`nvlink.coll.${op}.name` as PageKey)}
      </Txt>
      {group(0, false)}
      {group(146, true)}
      <Arrow x1={124} y1={86} x2={150} y2={86} color={moving ? COL.cur : COL.idleLine} w={moving ? 2.4 : 1.4} head={6} />
    </g>
  );
}

// --- 4. ring vs tree: number of steps --------------------------------------------------------------------

function Tree({ step, t, k }: DrawProps) {
  const N = 8;
  const cx = 140;
  const cy = 80;
  if (step === 0) {
    const total = ringLatencySteps(N);
    const done = Math.min(t, total);
    const pos = (i: number): [number, number] => [cx + 52 * Math.sin((2 * Math.PI * i) / N), cy - 52 * Math.cos((2 * Math.PI * i) / N)];
    return (
      <g>
        <Txt x={W / 2} y={12} size={8.5} anchor="middle" weight={600} color={COL.blue}>
          {k("ring")}
        </Txt>
        {Array.from({ length: N }, (_, i) => {
          const [x1, y1] = pos(i);
          const [x2, y2] = pos((i + 1) % N);
          const a = Math.atan2(y2 - y1, x2 - x1);
          const active = t >= 1 && t <= total;
          return <Arrow key={i} x1={x1 + 11 * Math.cos(a)} y1={y1 + 11 * Math.sin(a)} x2={x2 - 11 * Math.cos(a)} y2={y2 - 11 * Math.sin(a)} color={active ? (done <= N - 1 ? COL.cur : COL.green) : COL.idleLine} w={active ? 1.8 : 1} />;
        })}
        {Array.from({ length: N }, (_, i) => {
          const [x, y] = pos(i);
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={9} fill={blueScale(Math.min(done, N - 1) / (N - 1) * 0.6 + 0.05)} stroke="#334155" strokeWidth={1} />
              <Txt x={x} y={y + 3} size={7.5} anchor="middle">
                {String(i)}
              </Txt>
            </g>
          );
        })}
        <Txt x={W / 2} y={150} size={8} anchor="middle" kind="mono" weight={600}>
          {`${fill(k("steps"), { k: done })} / 2(N − 1) = ${total}`}
        </Txt>
      </g>
    );
  }
  const total = treeLatencySteps(N);
  const done = Math.min(t, total);
  const pos = (i: number): [number, number] => {
    const d = treeDepth(i);
    const first = 2 ** d - 1;
    const width = d === 3 ? 4 : 2 ** d; // node 7 sits under node 3
    const slot = d === 3 ? 0 : i - first;
    return [20 + ((slot + 0.5) * 240) / width, 26 + d * 34];
  };
  const maxD = treeDepth(N - 1);
  return (
    <g>
      <Txt x={W / 2} y={12} size={8.5} anchor="middle" weight={600} color={COL.blue}>
        {k("tree")}
      </Txt>
      {Array.from({ length: N - 1 }, (_, j) => {
        const i = j + 1;
        const p = (i - 1) >> 1;
        const d = treeDepth(i);
        // reduce: step s (1..maxD) moves depth maxD - s + 1 up; broadcast: step maxD + s moves depth s down
        const up = done >= 1 && done <= maxD && d === maxD - done + 1;
        const down = done > maxD && d === done - maxD;
        const [x1, y1] = pos(i);
        const [x2, y2] = pos(p);
        const color = up ? COL.cur : down ? COL.green : COL.idleLine;
        return up ? (
          <Arrow key={i} x1={x1} y1={y1 - 10} x2={x2} y2={y2 + 10} color={color} w={1.8} />
        ) : down ? (
          <Arrow key={i} x1={x2} y1={y2 + 10} x2={x1} y2={y1 - 10} color={color} w={1.8} />
        ) : (
          <line key={i} x1={x1} y1={y1 - 10} x2={x2} y2={y2 + 10} stroke={color} strokeWidth={1} />
        );
      })}
      {Array.from({ length: N }, (_, i) => {
        const [x, y] = pos(i);
        const d = treeDepth(i);
        const full = done > maxD ? d <= done - maxD : i === 0 && done >= maxD;
        return (
          <g key={i}>
            <circle cx={x} cy={y} r={9} fill={full ? blueScale(0.65) : d >= maxD - done + 1 && done <= maxD ? COL.curLight : "#fff"} stroke="#334155" strokeWidth={1} />
            <Txt x={x} y={y + 3} size={7.5} anchor="middle" color={full ? "#fff" : COL.text}>
              {String(i)}
            </Txt>
          </g>
        );
      })}
      <Txt x={W / 2} y={150} size={8} anchor="middle" kind="mono" weight={600}>
        {`${fill(k("steps"), { k: done })} / 2·log₂N = ${total}`}
      </Txt>
    </g>
  );
}

// --- 5. DDP: forward, backward with bucketed all-reduce, step --------------------------------------------

function Ddp({ step, f, k }: DrawProps) {
  const rowY = [30, 98];
  const xs = { batch: 6, l: [52, 106, 160], loss: 222 };
  const lw = 44;
  const h = 26;
  const fwdPos = step === 1 ? Math.min(3, Math.floor(f * 4)) : step > 1 ? 3 : -1; // layers reached
  const bwdDone = step === 2 ? Math.min(3, Math.floor(f * 4.2)) : step > 2 ? 3 : 0; // layers with gradients, from the top
  const reduced = step === 2 ? Math.max(0, bwdDone - 1 + (f > 0.95 ? 1 : 0)) : step > 2 ? 3 : 0; // buckets done
  return (
    <g>
      {[0, 1].map((g) => {
        const y = rowY[g];
        return (
          <g key={g}>
            <Txt x={6} y={y - 6} size={7.5} weight={600}>
              {fill(k("gpu"), { i: g })}
            </Txt>
            <Box x={xs.batch} y={y} w={38} h={h} fill={GPU_LIGHT[g]} stroke={GPU_COL[g]} label={k("batch")} size={6} />
            {xs.l.map((x, li) => {
              const hasGrad = bwdDone >= 3 - li;
              const upd = step === 3;
              return (
                <g key={li}>
                  <line x1={li === 0 ? xs.batch + 38 : xs.l[li - 1] + lw} x2={x} y1={y + h / 2} y2={y + h / 2} stroke={fwdPos >= li ? COL.blue : COL.idleLine} strokeWidth={1.4} />
                  <Box x={x} y={y} w={lw} h={h} fill={upd ? COL.greenLight : hasGrad ? COL.curLight : fwdPos >= li ? COL.blueLight : "#fff"} stroke={upd ? COL.green : hasGrad ? COL.cur : "#94a3b8"} label={`L${li + 1}`} kind="mono" size={7} />
                </g>
              );
            })}
            <line x1={xs.l[2] + lw} x2={xs.loss} y1={y + h / 2} y2={y + h / 2} stroke={fwdPos >= 3 ? COL.blue : COL.idleLine} strokeWidth={1.4} />
            <Box x={xs.loss} y={y} w={50} h={h} fill={fwdPos >= 3 ? COL.blueLight : "#fff"} label="loss" kind="mono" size={7} />
          </g>
        );
      })}
      {xs.l.map((x, li) => {
        const on = reduced >= 3 - li;
        const busy = step === 2 && bwdDone >= 3 - li && !on;
        return on || busy ? (
          <g key={`ar${li}`}>
            <Arrow x1={x + lw / 2 - 4} y1={rowY[0] + h + 2} x2={x + lw / 2 - 4} y2={rowY[1] - 2} color={on ? COL.green : COL.cur} w={1.6} />
            <Arrow x1={x + lw / 2 + 4} y1={rowY[1] - 2} x2={x + lw / 2 + 4} y2={rowY[0] + h + 2} color={on ? COL.green : COL.cur} w={1.6} />
          </g>
        ) : null;
      })}
      {step === 2 && (
        <Txt x={W / 2} y={(rowY[0] + h + rowY[1]) / 2 + 3} size={6.5} anchor="middle" color={COL.muted}>
          {k("bucket")}
        </Txt>
      )}
      <Txt x={W / 2} y={146} size={7.5} anchor="middle" weight={600} color={step === 3 ? COL.green : step === 2 ? COL.cur : COL.blue}>
        {k(["batch", "fwd", "bwd", "step"][step])}
      </Txt>
    </g>
  );
}

// --- 5. peer-to-peer copy routes -------------------------------------------------------------------------

function P2p({ step, t, f, k }: DrawProps) {
  const g0 = { x: 12, y: 96 };
  const g1 = { x: 204, y: 96 };
  const nvl: [number, number][] = [
    [g0.x + 64, g0.y + 15],
    [g1.x, g1.y + 15],
  ];
  const pcie: [number, number][] = [
    [g0.x + 32, g0.y],
    [g0.x + 32, 66],
    [140, 66],
    [g1.x + 32, 66],
    [g1.x + 32, g1.y],
  ];
  const up: [number, number][] = [
    [g0.x + 32, g0.y],
    [g0.x + 32, 66],
    [140, 66],
    [140, 30],
  ];
  const down: [number, number][] = [
    [140, 30],
    [140, 66],
    [g1.x + 32, 66],
    [g1.x + 32, g1.y],
  ];
  const ram = step === 2 ? (f < 0.5 ? f * 2 : 2 - f * 2) : 0;
  const ideal = step === 0 ? moveSeconds(GIB, H100_LINKS.nvlink4) : step === 1 ? moveSeconds(GIB, H100_LINKS.pcie5) : null;
  return (
    <g>
      <rect x={84} y={6} width={112} height={24} rx={3} fill={COL.idle} stroke="#94a3b8" />
      <Txt x={140} y={16} size={6.8} anchor="middle">
        {k("host")}
      </Txt>
      <rect x={104} y={20} width={72} height={6} fill="#fff" stroke={COL.idleLine} strokeWidth={0.5} />
      <rect x={104} y={20} width={72 * ram} height={6} fill={COL.cur} />
      <line x1={140} y1={30} x2={140} y2={58} stroke="#9ca3af" strokeWidth={2} />
      <polyline points={pcie.map((p) => p.join(",")).join(" ")} fill="none" stroke="#9ca3af" strokeWidth={2} />
      <line x1={nvl[0][0]} y1={nvl[0][1]} x2={nvl[1][0]} y2={nvl[1][1]} stroke={COL.blue} strokeWidth={4} opacity={step === 0 ? 1 : 0.3} />
      <Gpu x={g0.x} y={g0.y} i={0} k={k} />
      <Gpu x={g1.x} y={g1.y} i={1} k={k} />
      <Txt x={g0.x + 32} y={g0.y + 42} size={6.5} anchor="middle" kind="mono" color={COL.muted}>
        cuda:0
      </Txt>
      <Txt x={g1.x + 32} y={g1.y + 42} size={6.5} anchor="middle" kind="mono" color={COL.muted}>
        cuda:1
      </Txt>
      {step === 0 && <Packets pts={nvl} t={t} every={1} slots={12} />}
      {step === 1 && <Packets pts={pcie} t={t} every={3} slots={16} />}
      {step === 2 && (f < 0.5 ? <Packets pts={up} t={t} every={3} slots={12} /> : <Packets pts={down} t={t} every={3} slots={12} />)}
      {/* after the packets, opaque: they pass behind the switch label */}
      <Box x={108} y={58} w={64} h={16} label={k("sw")} size={6.5} fill="#fff" />
      <Txt x={W / 2} y={150} size={7.5} anchor="middle" weight={600} color={step === 2 ? COL.red : COL.text}>
        {ideal != null ? `${k("ideal")}: ${fmtSeconds(ideal)}` : k("staged")}
      </Txt>
    </g>
  );
}

// --- 6. the worked example as a time line ----------------------------------------------------------------

const COMPUTE_MS = 50;
const COMM = {
  pcie: ringAllReduceSeconds(1e9, 8, 32e9) * 1e3,
  nvl: ringAllReduceSeconds(1e9, 8, 450e9) * 1e3,
};

function Overlap({ step, f, k }: DrawProps) {
  const X = (ms: number) => 78 + (ms / 120) * 190;
  const now = f * 120;
  const overlap = step === 2;
  const rows: { id: "pcie" | "nvl"; y: number; on: boolean }[] = [
    { id: "pcie", y: 22, on: true },
    { id: "nvl", y: 70, on: step >= 1 },
  ];
  const bar = (a: number, b: number, y: number, col: string, label?: string) =>
    a < now ? (
      <g>
        <rect x={X(a)} y={y} width={Math.max(0, X(Math.min(b, now)) - X(a))} height={12} rx={2} fill={col} />
        {label && now > b && X(b) - X(a) > 30 && (
          <Txt x={(X(a) + X(b)) / 2} y={y + 8.5} size={6.2} anchor="middle" color="#fff">
            {label}
          </Txt>
        )}
      </g>
    ) : null;
  return (
    <g>
      {[0, 20, 40, 60, 80, 100, 120].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={16} y2={112} stroke="#eef0f3" />
          <Txt x={X(v)} y={121} size={6.3} anchor="middle" color={COL.muted}>
            {String(v)}
          </Txt>
        </g>
      ))}
      {rows.map((r) => {
        if (!r.on) return null;
        const comm = COMM[r.id];
        const fwd = COMPUTE_MS / 3;
        const commStart = overlap ? fwd : COMPUTE_MS;
        const end = stepMs(COMPUTE_MS, comm, overlap);
        return (
          <g key={r.id}>
            <Txt x={74} y={r.y + 9} size={6.5} anchor="end" weight={600}>
              {k(r.id)}
            </Txt>
            {overlap ? (
              <>
                {bar(0, fwd, r.y, blueScale(0.45), k("fwd"))}
                {bar(fwd, COMPUTE_MS, r.y, COL.blue, k("bwd"))}
              </>
            ) : (
              bar(0, COMPUTE_MS, r.y, COL.blue, k("compute"))
            )}
            {bar(commStart, commStart + comm, r.y + 15, COL.cur, X(commStart + comm) - X(commStart) > 30 ? k("comm") : undefined)}
            {now >= end && (
              <g>
                <line x1={X(end)} x2={X(end)} y1={r.y - 3} y2={r.y + 31} stroke={COL.red} strokeWidth={1.2} />
                <Txt x={X(end) + 3} y={r.y + 38} size={6.8} kind="mono" weight={600} color={COL.red}>
                  {`${end.toFixed(1)} ms`}
                </Txt>
              </g>
            )}
          </g>
        );
      })}
      <Txt x={173} y={134} size={6.8} anchor="middle">
        {k("axis")}
      </Txt>
    </g>
  );
}

const PICTURES: Record<NvCase, { draw: (p: DrawProps) => ReactNode; h: number }> = {
  path: { draw: (p) => <PathPic {...p} />, h: 166 },
  parallel: { draw: (p) => <Parallel {...p} />, h: 140 },
  gens: { draw: (p) => <Gens {...p} />, h: 156 },
  hops: { draw: (p) => <Hops {...p} />, h: 166 },
  colls: { draw: (p) => <Colls {...p} />, h: 140 },
  tree: { draw: (p) => <Tree {...p} />, h: 158 },
  ddp: { draw: (p) => <Ddp {...p} />, h: 152 },
  p2p: { draw: (p) => <P2p {...p} />, h: 156 },
  overlap: { draw: (p) => <Overlap {...p} />, h: 140 },
};

/** One animation of the NVLink & NVSwitch tab. */
export default function NvlinkAnim({ id }: { id: NvCase }) {
  const pic = PICTURES[id];
  return <StepAnim prefix="nvlink" id={id} plan={NV_PLAN[id]} h={pic.h} draw={pic.draw} />;
}
