import { type ReactNode } from "react";
import { usePageText, type PageKey } from "../content/pages";
import {
  AG5,
  CON,
  FLOAT_BITS,
  FUSE_OPS,
  PAIR_N,
  PLAN,
  SCENARIOS,
  VIEW_STRIDES,
  agGraph,
  asMatrix,
  checks,
  conK,
  conM,
  digitsOf,
  einsumGemm,
  memTrips,
  pairAB,
  stageAt,
  stepCount,
  totalFrames,
  type FlowCase,
} from "../dataflowAnim";
import { AnimControls, COL, Cell, FAMILY, Txt, useAnimClock } from "./animKit";
import { RichText } from "./Tex";

const W = 280; // SVG user units, scaled to the column width
const HOLD = 8; // ticks the finished picture stays before the cycle restarts
const P_COL = [COL.blue, COL.cur, COL.green, COL.violet]; // one colour per particle
const P_LIGHT = [COL.blueLight, COL.curLight, COL.greenLight, COL.violetLight];

type Stage = ReturnType<typeof stageAt>;
interface PicProps {
  s: Stage;
  T: (k: string) => string;
}

const fmt = (v: number, d = 3) => {
  const r = Number(v.toFixed(d));
  return (Object.is(r, -0) ? 0 : r).toString().replace("-", "−");
};

function Arrow({ x1, y1, x2, y2, color, w = 1.2, dash, head = 3.5 }: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  w?: number;
  dash?: string;
  head?: number;
}) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  if (Math.hypot(x2 - x1, y2 - y1) < 0.5) return null;
  const hx = (k: number) => x2 - head * Math.cos(a + k * 0.45);
  const hy = (k: number) => y2 - head * Math.sin(a + k * 0.45);
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2 - 0.6 * head * Math.cos(a)} y2={y2 - 0.6 * head * Math.sin(a)} stroke={color} strokeWidth={w} strokeDasharray={dash} />
      <path d={`M ${x2} ${y2} L ${hx(1)} ${hy(1)} L ${hx(-1)} ${hy(-1)} Z`} fill={color} />
    </g>
  );
}

/** A labelled rounded box. */
function Box({ x, y, w, h, fill = "white", stroke = COL.idleLine, sw = 0.8, children }: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill?: string;
  stroke?: string;
  sw?: number;
  children?: ReactNode;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={3} fill={fill} stroke={stroke} strokeWidth={sw} />
      {children}
    </g>
  );
}

// ---- s2: Python loop vs one broadcast kernel ------------------------------------------------------------

function Vectorise({ s, T }: PicProps) {
  const n = PAIR_N;
  const NN = n * n;
  const k = s.step === 0 ? s.t : NN; // pass being made (all done after the loop stage)
  const [ca, cb] = s.step === 0 ? pairAB(k) : [-1, -1];
  const trips = s.step === 0 ? k + 1 : NN;
  // top: the loop
  const gs = 15;
  const gx = 128;
  const gy = 26;
  // bottom: the broadcast
  const by = 130;
  const vs = 11;
  const v1 = 58;
  const v2 = 124;
  const rx = 192;
  const rs = 12;
  const ghosts = s.step === 0 ? 0 : s.step === 1 ? Math.min(s.t + 1, n - 1) : n - 1;
  const filled = s.step === 2;
  const flash = filled && s.t === 0;
  const viewCell = (x0: number, a: number, b: number, which: "rows" | "cols") => {
    const p = which === "rows" ? b : a; // particle read: x[b] along the rows view, x[a] along the cols view
    const stored = which === "rows" ? a === 0 : b === 0;
    const shown = stored || (which === "rows" ? a : b) <= ghosts;
    if (!shown) return <Cell key={`${which}${a}${b}`} x={x0 + b * vs} y={by + a * vs} s={vs - 1} fill="white" stroke={COL.idleLine} sw={0.4} />;
    return (
      <Cell
        key={`${which}${a}${b}`}
        x={x0 + b * vs}
        y={by + a * vs}
        s={vs - 1}
        fill={P_LIGHT[p]}
        stroke={P_COL[p]}
        sw={stored ? 0.9 : 0.6}
        dash={stored ? undefined : "1.5 1"}
        opacity={stored ? 1 : 0.6}
      />
    );
  };
  return (
    <>
      <Txt x={6} y={11} size={8} weight={600} color={COL.cur}>
        {T("flow.anim.s2.loop")}
      </Txt>
      {/* interpreter <-> kernel round trip */}
      <Box x={8} y={22} w={96} h={17} fill={s.step === 0 ? "#fffbeb" : "white"} stroke={s.step === 0 ? COL.cur : COL.idleLine}>
        <Txt x={56} y={33} size={7} anchor="middle">{T("flow.anim.interp")}</Txt>
      </Box>
      <Box x={8} y={62} w={96} h={17} fill={s.step === 0 ? "#fffbeb" : "white"} stroke={s.step === 0 ? COL.cur : COL.idleLine}>
        <Txt x={56} y={73} size={7} anchor="middle">{T("flow.anim.kernel")}</Txt>
      </Box>
      <Arrow x1={70} y1={40} x2={70} y2={61} color={s.step === 0 ? COL.cur : COL.idleLine} />
      <Arrow x1={42} y1={61} x2={42} y2={40} color={s.step === 0 ? COL.cur : COL.idleLine} />
      <Txt x={8} y={93} size={6.5} kind="mono" color={COL.muted}>
        r[a, b] = x[b] - x[a]
      </Txt>
      <Txt x={gx + (n * gs) / 2} y={gy - 12} anchor="middle" kind="math" color={COL.muted}>b</Txt>
      <Txt x={gx - 14} y={gy + (n * gs) / 2 + 3} anchor="middle" kind="math" color={COL.muted}>a</Txt>
      {Array.from({ length: n }, (_, i) => (
        <g key={i}>
          <Txt x={gx + i * gs + gs / 2} y={gy - 3} size={6} anchor="middle" kind="mono" color={COL.muted}>{i}</Txt>
          <Txt x={gx - 5} y={gy + i * gs + gs / 2 + 2} size={6} anchor="middle" kind="mono" color={COL.muted}>{i}</Txt>
        </g>
      ))}
      {Array.from({ length: NN }, (_, p) => {
        const [a, b] = pairAB(p);
        const cur = p === k && s.step === 0;
        const done = p < k;
        return (
          <Cell
            key={p}
            x={gx + b * gs}
            y={gy + a * gs}
            s={gs - 1}
            fill={cur ? COL.curLight : done ? COL.done : "white"}
            stroke={cur ? COL.cur : done ? COL.doneLine : COL.idleLine}
            sw={cur ? 1.4 : 0.6}
          />
        );
      })}
      <Txt x={200} y={36} size={6.5} kind="mono" color={COL.cur}>
        {s.step === 0 ? `a, b = ${ca}, ${cb}` : ""}
      </Txt>
      <Txt x={200} y={54} size={7} kind="math" color={COL.muted}>{"r_{ab} = x_b − x_a"}</Txt>
      <Txt x={200} y={72} size={7} color={COL.cur} weight={600}>
        {T("flow.anim.s2.trips").replace("{n}", String(trips))}
      </Txt>

      <line x1={4} y1={102} x2={W - 4} y2={102} stroke={COL.idleLine} strokeWidth={0.6} strokeDasharray="3 2" />

      <Txt x={6} y={114} size={8} weight={600} color={COL.blue}>
        {T("flow.anim.s2.bcast")}
      </Txt>
      {/* x: (N, 3), one row per particle */}
      <Txt x={23} y={by - 4} anchor="middle" kind="mono" size={7}>x</Txt>
      {Array.from({ length: n }, (_, a) =>
        [0, 1, 2].map((i) => <Cell key={`x${a}${i}`} x={8 + i * 10} y={by + a * 11} s={9} fill={P_LIGHT[a]} stroke={P_COL[a]} />),
      )}
      <Txt x={23} y={by + n * 11 + 9} anchor="middle" kind="mono" size={6} color={COL.muted}>(N, 3)</Txt>
      {/* the two stride-0 views */}
      {Array.from({ length: NN }, (_, p) => {
        const [a, b] = pairAB(p);
        return (
          <g key={p}>
            {viewCell(v1, a, b, "rows")}
            {viewCell(v2, a, b, "cols")}
          </g>
        );
      })}
      {s.step >= 1 && (
        <g>
          <Arrow x1={v1 - 4} y1={by} x2={v1 - 4} y2={by + n * vs} color={COL.violet} w={0.9} />
          <Txt x={v1 - 7} y={by + (n * vs) / 2 + 2} size={6} anchor="end" kind="mono" color={COL.violet}>0</Txt>
          <Arrow x1={v2} y1={by - 4} x2={v2 + n * vs} y2={by - 4} color={COL.violet} w={0.9} />
          <Txt x={v2 + (n * vs) / 2} y={by - 7} size={6} anchor="middle" kind="mono" color={COL.violet}>0</Txt>
        </g>
      )}
      <Txt x={(v1 + n * vs + v2) / 2} y={by + (n * vs) / 2 + 4} size={10} anchor="middle" kind="math">−</Txt>
      <Txt x={(v2 + n * vs + rx) / 2} y={by + (n * vs) / 2 + 4} size={10} anchor="middle" kind="math">=</Txt>
      {/* the result, all pairs written by one kernel */}
      {Array.from({ length: NN }, (_, p) => {
        const [a, b] = pairAB(p);
        const diag = a === b;
        return (
          <Cell
            key={`r${p}`}
            x={rx + b * rs}
            y={by - 2 + a * rs}
            s={rs - 1}
            fill={!filled ? "white" : flash ? COL.blue : diag ? COL.blueLight : COL.done}
            stroke={filled ? COL.blue : COL.idleLine}
            sw={filled ? 0.8 : 0.5}
            label={filled && diag ? 0 : undefined}
            size={6}
            labelColor={flash ? "white" : COL.text}
          />
        );
      })}
      <Txt x={v1 + (n * vs) / 2} y={by + n * vs + 10} size={6} anchor="middle" kind="mono">x[None, :, :]</Txt>
      <Txt x={v2 + (n * vs) / 2} y={by + n * vs + 10} size={6} anchor="middle" kind="mono">x[:, None, :]</Txt>
      <Txt x={rx + (n * rs) / 2} y={by + n * vs + 10} size={6} anchor="middle" kind="mono">(N, N, 3)</Txt>
      <Txt x={v1 + (n * vs) / 2} y={by + n * vs + 20} size={6} anchor="middle" kind="mono" color={s.step >= 1 ? COL.violet : COL.muted}>
        {`(${VIEW_STRIDES.rows.join(", ")})`}
      </Txt>
      <Txt x={v2 + (n * vs) / 2} y={by + n * vs + 20} size={6} anchor="middle" kind="mono" color={s.step >= 1 ? COL.violet : COL.muted}>
        {`(${VIEW_STRIDES.cols.join(", ")})`}
      </Txt>
      <Txt x={6} y={by + n * vs + 36} size={6.5} color={COL.violet} opacity={s.step >= 1 ? 1 : 0}>
        {T("flow.anim.s2.same")}
      </Txt>
      <Txt x={W - 6} y={by + n * vs + 36} size={7} anchor="end" weight={600} color={COL.blue}>
        {T("flow.anim.s2.launches").replace("{n}", filled ? "1" : "0")}
      </Txt>
    </>
  );
}

// ---- s3: host and device --------------------------------------------------------------------------------

function GpuCpu({ s, T }: PicProps) {
  const { step, t } = s;
  const memCells = 8;
  const hostX = 12;
  const devX = 188;
  const memY = 62;
  // what is on the device / host
  const xCells = step === 0 ? Math.min(t + 2, 4) : 4;
  const yOnDevice = step >= 3 || (step === 2 && t >= 3);
  const copied = step === 3 ? Math.min(Math.max(t - 1, 0), 4) : 0; // cells of y arrived on the host
  const packetU = step === 3 && t < 5 ? t / 5 : -1;
  const gpuBusy = (step === 1 && t >= 2) || (step === 2 && t < 3);
  const kernelAt = step === 1 ? (t === 0 ? "cpu" : t === 1 ? "queue" : "gpu") : "none";
  // timeline (time in a.u., fixed 0 .. 10)
  const x0 = 36;
  const x1 = 272;
  const X = (v: number) => x0 + (v / 10) * (x1 - x0);
  const gpuEnd = step < 1 ? 0 : step === 1 ? 2.6 + ((t + 1) / 6) * 2.4 : step === 2 ? 5 + (Math.min(t + 1, 3) / 3) * 2 : 7;
  const bars: { lane: number; a: number; b: number; label: string; fill: string; stroke: string; mono?: boolean; hatch?: boolean }[] = [];
  bars.push({ lane: 0, a: 0, b: step === 0 ? Math.min((t + 1) / 2, 1) * 0.8 : 0.8, label: T("flow.anim.s3.alloc"), fill: COL.blueLight, stroke: COL.blue });
  if (step > 0 || t >= 1) bars.push({ lane: 1, a: 0.8, b: step === 0 ? 0.8 + Math.min(t / 2, 1) * 0.8 : 1.6, label: "randn", fill: COL.done, stroke: COL.doneLine, mono: true });
  if (step >= 1) bars.push({ lane: 0, a: 1.6, b: 2.6, label: T("flow.anim.s3.launch"), fill: COL.curLight, stroke: COL.cur });
  if (step >= 1 && (step > 1 || t >= 2)) bars.push({ lane: 1, a: 2.6, b: gpuEnd, label: "x @ x", fill: COL.done, stroke: COL.doneLine, mono: true });
  if (step >= 2) bars.push({ lane: 0, a: 2.6, b: gpuEnd, label: T("flow.anim.s3.wait"), fill: "white", stroke: COL.muted, hatch: true });
  if (step === 3) bars.push({ lane: 2, a: 7, b: 7 + ((t + 1) / 6) * 2.6, label: T("flow.anim.s3.copy"), fill: COL.redLight, stroke: COL.red });
  const laneY = (l: number) => 108 + l * 15;
  const code = ["x = torch.randn(4096, 4096, device=dev)", "y = x @ x", "torch.cuda.synchronize()", "y_cpu = y.cpu()"][step];
  return (
    <>
      <defs>
        <pattern id="flow-hatch" width="3" height="3" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="3" stroke={COL.idleLine} strokeWidth="1.2" />
        </pattern>
      </defs>
      {/* host */}
      <Box x={6} y={6} w={94} h={74}>
        <Txt x={12} y={17} size={8} weight={600}>CPU</Txt>
        {[0, 1, 2, 3].map((c) => (
          <rect key={c} x={12 + c * 20} y={23} width={16} height={16} rx={2} fill={COL.idle} stroke={COL.idleLine} strokeWidth={0.6} />
        ))}
        <Txt x={12} y={57} size={6.5} color={COL.muted}>{T("flow.anim.s3.host")}</Txt>
        {Array.from({ length: memCells }, (_, c) => (
          <Cell key={c} x={hostX + c * 10} y={memY} s={9} fill={c < copied ? COL.blue : "white"} stroke={c < copied ? COL.blue : COL.idleLine} sw={0.6} />
        ))}
        {copied > 0 && <Txt x={hostX + 20} y={memY + 16} size={6} anchor="middle" kind="mono" color={COL.muted}>y_cpu</Txt>}
      </Box>
      {/* device */}
      <Box x={182} y={6} w={92} h={74} stroke={gpuBusy ? COL.blue : COL.idleLine} sw={gpuBusy ? 1.2 : 0.8}>
        <Txt x={188} y={17} size={8} weight={600}>GPU</Txt>
        <Txt x={268} y={17} size={6} anchor="end" color={COL.muted}>{T("flow.anim.s3.threads")}</Txt>
        {Array.from({ length: 36 }, (_, c) => {
          const on = gpuBusy && (c + t) % 3 !== 0;
          return <circle key={c} cx={191 + (c % 12) * 6.8} cy={26 + Math.floor(c / 12) * 6} r={2} fill={on ? COL.blue : COL.idle} stroke={COL.idleLine} strokeWidth={0.3} />;
        })}
        <Txt x={188} y={57} size={6.5} color={COL.muted}>{T("flow.anim.s3.device")}</Txt>
        {Array.from({ length: memCells }, (_, c) => {
          const isX = c < 4 && c < xCells;
          const isY = c >= 4 && yOnDevice;
          return <Cell key={c} x={devX + c * 10} y={memY} s={9} fill={isX ? COL.done : isY ? COL.blue : "white"} stroke={isX || isY ? COL.blue : COL.idleLine} sw={0.6} />;
        })}
        <Txt x={devX + 20} y={memY + 16} size={6} anchor="middle" kind="mono" color={COL.muted}>x</Txt>
        <Txt x={devX + 60} y={memY + 16} size={6} anchor="middle" kind="mono" color={COL.muted}>y</Txt>
      </Box>
      {/* queue between them */}
      <Txt x={140} y={17} size={6.5} anchor="middle" color={COL.muted}>{T("flow.anim.s3.queue")}</Txt>
      <rect x={110} y={21} width={60} height={16} rx={2} fill="white" stroke={COL.idleLine} strokeWidth={0.7} />
      <Arrow x1={101} y1={29} x2={109} y2={29} color={COL.idleLine} w={0.8} />
      <Arrow x1={171} y1={29} x2={181} y2={29} color={COL.idleLine} w={0.8} />
      {kernelAt !== "none" && kernelAt !== "gpu" && (
        <g>
          <rect x={kernelAt === "cpu" ? 64 : 120} y={23} width={30} height={12} rx={2} fill={COL.curLight} stroke={COL.cur} />
          <Txt x={kernelAt === "cpu" ? 79 : 135} y={31.5} size={6} anchor="middle" kind="mono">x @ x</Txt>
        </g>
      )}
      {/* the bus */}
      <line x1={100} y1={memY + 4.5} x2={182} y2={memY + 4.5} stroke={step === 3 ? COL.red : COL.idleLine} strokeWidth={3} strokeLinecap="round" opacity={0.6} />
      <Txt x={141} y={memY - 2} size={6.5} anchor="middle" color={COL.muted}>{`${T("flow.anim.s3.bus")} (PCIe)`}</Txt>
      {packetU >= 0 && <rect x={176 - packetU * 74 - 6} y={memY} width={12} height={9} rx={1.5} fill={COL.red} />}

      {/* timeline */}
      {["CPU", "GPU", T("flow.anim.s3.bus")].map((l, i) => (
        <Txt key={i} x={6} y={laneY(i) + 7} size={6.5} color={COL.muted}>{l}</Txt>
      ))}
      {[0, 1, 2].map((i) => (
        <line key={i} x1={x0} y1={laneY(i) + 5} x2={x1} y2={laneY(i) + 5} stroke={COL.idle} strokeWidth={0.6} />
      ))}
      {bars.map((b, i) => {
        const w = Math.max(X(b.b) - X(b.a), 0.5);
        return (
          <g key={i}>
            <rect x={X(b.a)} y={laneY(b.lane)} width={w} height={10} rx={1.5} fill={b.hatch ? "url(#flow-hatch)" : b.fill} stroke={b.stroke} strokeWidth={0.6} />
            {w > 15 && (
              <Txt x={X(b.a) + w / 2} y={laneY(b.lane) + 7.2} size={5.5} anchor="middle" kind={b.mono ? "mono" : "sans"}>
                {b.label}
              </Txt>
            )}
          </g>
        );
      })}
      <line x1={x0} y1={152} x2={x1} y2={152} stroke={COL.muted} strokeWidth={0.6} />
      {[0, 2, 4, 6, 8, 10].map((v) => (
        <g key={v}>
          <line x1={X(v)} y1={152} x2={X(v)} y2={154.5} stroke={COL.muted} strokeWidth={0.6} />
          <Txt x={X(v)} y={161} size={6} anchor="middle" kind="math" color={COL.muted}>{v}</Txt>
        </g>
      ))}
      <Txt x={x1} y={170} size={6.5} anchor="end" color={COL.muted}>{T("flow.anim.time")}</Txt>
      <rect x={4} y={176} width={W - 8} height={15} rx={2} fill={COL.idle} />
      <Txt x={9} y={186.5} size={7} kind="mono" color={COL.blue}>{code}</Txt>
    </>
  );
}

// ---- s4: a contraction as a matrix product; float32 vs float64 ----------------------------------------

function Contraction({ s, T }: PicProps) {
  if (s.step === 3) return <Precision s={s} T={T} />;
  const cs = 14;
  const sy = 36; // slices' top
  const slX = [24, 96]; // slice i = 0, 1 (stage 0)
  const mx = 30; // matrix (stage 1+)
  const my = 26;
  const u = s.step === 0 ? 0 : s.step === 1 ? Math.min(s.t / 3, 1) : 1;
  const M = asMatrix(conK);
  const F = einsumGemm(conK, conM);
  const yv = F.flat();
  const mX = s.step === 0 ? 166 : mx + CON.B * cs + 14;
  const mY = s.step === 0 ? sy - 6 : my + cs;
  const gemm = s.step === 2;
  const yx = mX + cs + 22;
  const showF = gemm && s.t >= 2;
  const spec = 'einsum("abi,b->ai", K, m)';
  const tone = (ch: string) => (ch === "b" ? COL.cur : ch === "a" || ch === "i" ? COL.blue : COL.text);
  return (
    <>
      <text x={6} y={12} fontSize={7.5} style={{ fontFamily: FAMILY.mono }} xmlSpace="preserve">
        {[...spec].map((ch, k) => {
          const inSpec = k >= 8 && k <= 16;
          return (
            <tspan key={k} fill={inSpec ? tone(ch) : COL.text} fontWeight={inSpec && /[abi]/.test(ch) ? 700 : undefined}>
              {ch}
            </tspan>
          );
        })}
      </text>
      <Txt x={W - 6} y={12} size={6.5} anchor="end" color={COL.cur}>{`b: ${T("flow.anim.s4.summed")}`}</Txt>
      <Txt x={W - 6} y={21} size={6.5} anchor="end" color={COL.blue}>{`a, i: ${T("flow.anim.s4.kept")}`}</Txt>
      {s.step === 0 &&
        slX.map((x, i) => (
          <g key={i}>
            <Txt x={x + (CON.B * cs) / 2} y={sy - 4} size={7} anchor="middle" kind="math">{`i = ${i}`}</Txt>
            <Txt x={x - 6} y={sy + (CON.A * cs) / 2 + 3} size={7} anchor="middle" kind="math" color={COL.muted}>a</Txt>
            <Txt x={x + (CON.B * cs) / 2} y={sy + CON.A * cs + 9} size={7} anchor="middle" kind="math" color={COL.cur}>b</Txt>
          </g>
        ))}
      {/* the cells of K, moving from the two slices into the (a·i, b) matrix */}
      {conK.map((Ka, a) =>
        Ka.map((Kab, b) =>
          Kab.map((v, i) => {
            const fx = slX[i] + b * cs;
            const fy = sy + a * cs;
            const tx = mx + b * cs;
            const ty = my + (a * CON.I + i) * cs;
            const px = fx + (tx - fx) * u;
            const py = fy + (ty - fy) * u;
            return (
              <g key={`${a}${b}${i}`} style={{ transform: `translate(${px}px, ${py}px)`, transition: "transform 0.6s ease" }}>
                <Cell
                  x={0}
                  y={0}
                  s={cs - 1}
                  fill={i === 0 ? COL.blueLight : COL.violetLight}
                  stroke={s.step === 0 ? COL.cur : i === 0 ? COL.blue : COL.violet}
                  sw={s.step === 0 ? 0.9 : 0.6}
                  label={fmt(v)}
                  size={7}
                />
              </g>
            );
          }),
        ),
      )}
      {s.step >= 1 && u >= 1 && (
        <g>
          {M.map((_, r) => (
            <Txt key={r} x={mx - 3} y={my + r * cs + 8} size={5.5} anchor="end" kind="mono" color={COL.muted}>
              {`${Math.floor(r / CON.I)},${r % CON.I}`}
            </Txt>
          ))}
          <Txt x={mx - 3} y={my - 3} size={5.5} anchor="end" kind="mono" color={COL.muted}>a,i</Txt>
        </g>
      )}
      {/* m (indexed by b) */}
      <Txt x={mX + cs / 2} y={mY - 4} size={7} anchor="middle" kind="math">m</Txt>
      {conM.map((v, b) => (
        <Cell key={b} x={mX} y={mY + b * cs} s={cs - 1} fill={COL.curLight} stroke={COL.cur} sw={0.8} label={v} size={6.5} />
      ))}
      {s.step >= 1 && (
        <Txt x={mX + cs + 10} y={my + (M.length * cs) / 2 + 3} size={10} anchor="middle" kind="math">=</Txt>
      )}
      {/* GEMM result, then reshaped back to F_ai */}
      {s.step >= 1 &&
        yv.map((v, r) => (
          <Cell
            key={r}
            x={yx}
            y={my + r * cs}
            s={cs - 1}
            fill={!gemm ? "white" : s.t === 0 ? COL.blue : COL.done}
            stroke={gemm ? COL.blue : COL.idleLine}
            label={gemm ? fmt(v) : undefined}
            labelColor={gemm && s.t === 0 ? "white" : COL.text}
            size={6.5}
          />
        ))}
      {showF && (
        <g>
          <Arrow x1={yx + cs + 4} y1={my + 3 * cs} x2={yx + cs + 22} y2={my + 3 * cs} color={COL.muted} w={0.9} />
          <Txt x={yx + cs + 50} y={my + cs - 2} size={7} anchor="middle" kind="math">{"F_{ai}"}</Txt>
          {F.map((row, a) =>
            row.map((v, i) => (
              <Cell key={`${a}${i}`} x={yx + cs + 38 + i * cs} y={my + cs + 4 + a * cs} s={cs - 1} fill={COL.done} stroke={COL.blue} label={fmt(v)} size={6.5} />
            )),
          )}
        </g>
      )}
      <Txt x={6} y={126} size={6.5} kind="mono" color={COL.muted}>
        {s.step === 0 ? "K: (a, b, i) = (3, 4, 2)" : "K → (a·i, b) = (6, 4)"}
      </Txt>
      <Txt x={W - 6} y={126} size={6.5} anchor="end" kind="mono" color={gemm ? COL.blue : COL.muted}>
        {gemm ? "(6, 4) @ (4,) → (6,) → (3, 2)" : ""}
      </Txt>
    </>
  );
}

function Precision({ s, T }: PicProps) {
  const bx = 52;
  const per = 186 / 64; // user units per bit
  const segCol = [COL.red, COL.cur, COL.blue];
  const segLight = [COL.redLight, COL.curLight, COL.blueLight];
  const row = (dtype: "float32" | "float64", y: number, showBar: boolean, showDigits: boolean, eps: string, nd: number) => {
    const bits = FLOAT_BITS[dtype];
    let x = bx;
    const d = digitsOf(dtype);
    return (
      <g>
        <Txt x={6} y={y + 8} size={7.5} kind="mono" weight={600}>{dtype}</Txt>
        {showBar &&
          bits.map((nb, k) => {
            const w = nb * per;
            const g = (
              <g key={k}>
                <rect x={x} y={y} width={w} height={11} fill={segLight[k]} stroke={segCol[k]} strokeWidth={0.6} />
                {w > 10 && (
                  <Txt x={x + w / 2} y={y + 8} size={6} anchor="middle" kind="mono">{nb}</Txt>
                )}
              </g>
            );
            x += w;
            return g;
          })}
        {showBar && (
          <Txt x={W - 6} y={y + 8} size={6} anchor="end" color={COL.muted}>{T("flow.anim.s4.bits").replace("{n}", String(bits[0] + bits[1] + bits[2]))}</Txt>
        )}
        {showDigits && (
          <g>
            <text x={bx} y={y + 25} fontSize={7.5} style={{ fontFamily: FAMILY.mono }}>
              <tspan fill={COL.text}>0.1 → </tspan>
              <tspan fill={COL.green} fontWeight={700}>{d.ok}</tspan>
              <tspan fill={COL.red}>{d.bad}</tspan>
            </text>
            <Txt x={bx} y={y + 36} size={6} color={COL.muted}>
              {`${T("flow.anim.s4.digits").replace("{n}", String(nd))}, eps = ${eps}`}
            </Txt>
          </g>
        )}
      </g>
    );
  };
  const t = s.t;
  return (
    <>
      {row("float32", 10, true, t >= 1, "1.2e-07", 7)}
      {row("float64", 60, t >= 2, t >= 3, "2.2e-16", 16)}
      {[T("flow.anim.s4.sign"), T("flow.anim.s4.exponent"), T("flow.anim.s4.mantissa")].map((l, k) => (
        <g key={k}>
          <rect x={bx + k * 70} y={110} width={7} height={7} fill={segLight[k]} stroke={segCol[k]} strokeWidth={0.6} />
          <Txt x={bx + k * 70 + 10} y={116} size={6.5} color={COL.muted}>{l}</Txt>
        </g>
      ))}
      <Txt x={6} y={127} size={6} color={COL.green}>{T("flow.anim.s4.correct")}</Txt>
      <Txt x={W - 6} y={127} size={6} anchor="end" color={COL.red}>{T("flow.anim.s4.error")}</Txt>
    </>
  );
}

// ---- s5: the dataflow graph, backward pass and fusion ---------------------------------------------------

function Graph({ s, T }: PicProps) {
  const { step, t } = s;
  const { r, phi, dphidr, grad } = agGraph();
  const fw = step > 0 ? 3 : t; // forward progress: 0 = only x, 3 = all recorded
  const bw = step === 1 ? t : step > 1 ? 3 : -1;
  const ny = 30; // node row top
  const nh = 24;
  const cy = ny + nh / 2;
  const node = (x: number, w: number, label: string, value: string, on: boolean, hot: boolean) => (
    <g opacity={on ? 1 : 0.25}>
      <rect x={x} y={ny} width={w} height={nh} rx={3} fill={hot ? COL.redLight : COL.blueLight} stroke={hot ? COL.red : COL.blue} strokeWidth={0.9} />
      <Txt x={x + w / 2} y={ny + 10} size={8} anchor="middle" kind="math">{label}</Txt>
      <Txt x={x + w / 2} y={ny + 20} size={5.5} anchor="middle" kind="mono" color={COL.muted}>{value}</Txt>
    </g>
  );
  const op = (x: number, w: number, label: string, on: boolean) => (
    <g opacity={on ? 1 : 0.25}>
      <rect x={x} y={cy - 7} width={w} height={14} rx={7} fill="white" stroke={COL.muted} strokeWidth={0.8} />
      <Txt x={x + w / 2} y={cy + 2.5} size={6.5} anchor="middle" kind="mono">{label}</Txt>
    </g>
  );
  // eager / fused memory traffic
  const eagerT = step === 2 ? t + 1 : step > 2 ? 2 * FUSE_OPS : 0; // arrows drawn so far
  const fusedT = step === 3 ? t : -1;
  const memY = 168;
  const chipY = 128;
  const ex = (k: number) => 10 + k * 32;
  const fx = (k: number) => 156 + k * 28;
  const opsLbl = ["mul", "add", "mul", "add"];
  return (
    <>
      <Txt x={6} y={12} size={7} weight={600} color={COL.blue}>{`${T("flow.anim.s5.forward")} →`}</Txt>
      <Txt x={6} y={94} size={7} weight={600} color={COL.red} opacity={bw >= 0 ? 1 : 0.3}>{`← ${T("flow.anim.s5.backward")}`}</Txt>
      {node(6, 44, "x", "(1, 0.5, 0)", true, bw >= 3)}
      {op(60, 30, "norm", fw >= 1)}
      {node(100, 36, "r", fmt(r), fw >= 1, bw >= 2)}
      {op(146, 36, "-GM/r", fw >= 2)}
      {node(192, 40, "ϕ", fmt(phi), fw >= 2, bw >= 0)}
      {fw >= 1 && <Arrow x1={50} y1={cy} x2={60} y2={cy} color={COL.blue} />}
      {fw >= 1 && <Arrow x1={90} y1={cy} x2={100} y2={cy} color={COL.blue} />}
      {fw >= 2 && <Arrow x1={136} y1={cy} x2={146} y2={cy} color={COL.blue} />}
      {fw >= 2 && <Arrow x1={182} y1={cy} x2={192} y2={cy} color={COL.blue} />}
      {fw >= 3 && step === 0 && (
        <Txt x={W - 4} y={12} size={6} anchor="end" kind="mono" color={COL.muted}>
          {"requires_grad=True"}
        </Txt>
      )}
      {/* backward: local derivatives, multiplied from phi back to x */}
      {bw >= 0 && <Txt x={240} y={cy + 3} size={7} kind="math" color={COL.red}>{"∂ϕ/∂ϕ = 1"}</Txt>}
      {bw >= 1 && (
        <g>
          <path d={`M 212 ${ny + nh} Q 175 ${ny + nh + 20} 118 ${ny + nh + 2}`} fill="none" stroke={COL.red} strokeWidth={1.1} />
          <Arrow x1={121} y1={ny + nh + 5} x2={118} y2={ny + nh + 1} color={COL.red} />
          <Txt x={165} y={ny + nh + 26} size={6.5} anchor="middle" kind="math" color={COL.red}>{`∂ϕ/∂r = GM/r^2 = ${fmt(dphidr)}`}</Txt>
        </g>
      )}
      {bw >= 2 && (
        <g>
          <path d={`M 118 ${ny + nh} Q 75 ${ny + nh + 20} 28 ${ny + nh + 2}`} fill="none" stroke={COL.red} strokeWidth={1.1} />
          <Arrow x1={31} y1={ny + nh + 5} x2={28} y2={ny + nh + 1} color={COL.red} />
          <Txt x={72} y={ny + nh + 26} size={6.5} anchor="middle" kind="math" color={COL.red}>{"∂r/∂x = x/r"}</Txt>
        </g>
      )}
      {bw >= 3 && (
        <Txt x={W - 4} y={94} size={6.5} anchor="end" kind="math" color={COL.red}>
          {`∇ϕ = (${grad.map((v) => fmt(v)).join(", ")})`}
        </Txt>
      )}
      <Txt x={W - 4} y={22} size={5.5} anchor="end" kind="math" color={COL.muted}>{`GM = ${AG5.GM}`}</Txt>

      <line x1={4} y1={102} x2={W - 4} y2={102} stroke={COL.idleLine} strokeWidth={0.6} strokeDasharray="3 2" />

      {/* eager: one kernel per op, each a memory round trip */}
      <Txt x={6} y={115} size={7} weight={600} color={step === 2 ? COL.cur : COL.text}>{T("flow.anim.s5.eager")}</Txt>
      <Txt x={146} y={115} size={7} weight={600} color={step === 3 ? COL.blue : COL.text}>{T("flow.anim.s5.fused")}</Txt>
      {opsLbl.map((l, k) => {
        const cur = step === 2 && Math.floor(t / 2) === k;
        const done = eagerT > 2 * k + 1;
        return (
          <g key={k}>
            <rect x={ex(k)} y={chipY} width={26} height={14} rx={2} fill={cur ? COL.curLight : done ? COL.done : COL.idle} stroke={cur ? COL.cur : COL.idleLine} strokeWidth={cur ? 1.2 : 0.6} />
            <Txt x={ex(k) + 13} y={chipY + 9.5} size={6.5} anchor="middle" kind="mono">{l}</Txt>
            {eagerT > 2 * k && <Arrow x1={ex(k) + 7} y1={memY} x2={ex(k) + 7} y2={chipY + 15} color={cur && t % 2 === 0 ? COL.cur : COL.idleLine} w={1} />}
            {eagerT > 2 * k + 1 && <Arrow x1={ex(k) + 19} y1={chipY + 14} x2={ex(k) + 19} y2={memY - 1} color={cur && t % 2 === 1 ? COL.cur : COL.idleLine} w={1} />}
          </g>
        );
      })}
      {/* compiled: the same ops fused, values in registers */}
      <rect x={fx(0) - 6} y={chipY - 5} width={4 * 28 + 8} height={24} rx={3} fill={fusedT >= 1 ? COL.blueLight : "white"} stroke={fusedT >= 0 ? COL.blue : COL.idleLine} strokeWidth={0.9} />
      {opsLbl.map((l, k) => (
        <g key={k}>
          <rect x={fx(k)} y={chipY} width={22} height={14} rx={2} fill={fusedT >= 1 ? COL.done : COL.idle} stroke={COL.idleLine} strokeWidth={0.6} />
          <Txt x={fx(k) + 11} y={chipY + 9.5} size={6.5} anchor="middle" kind="mono">{l}</Txt>
          {k < 3 && fusedT >= 1 && <Arrow x1={fx(k) + 22} y1={chipY + 7} x2={fx(k) + 28} y2={chipY + 7} color={COL.blue} w={0.8} head={2.5} />}
        </g>
      ))}
      <Txt x={fx(0) + 53} y={chipY + 28} size={6} anchor="middle" color={COL.blue} opacity={fusedT >= 1 ? 1 : 0.4}>{T("flow.anim.s5.registers")}</Txt>
      {fusedT >= 0 && <Arrow x1={fx(0) + 4} y1={memY} x2={fx(0) + 4} y2={chipY + 20} color={fusedT === 0 ? COL.blue : COL.idleLine} w={1} />}
      {fusedT >= 2 && <Arrow x1={fx(3) + 16} y1={chipY + 20} x2={fx(3) + 16} y2={memY - 1} color={fusedT === 2 ? COL.blue : COL.idleLine} w={1} />}
      {/* memory */}
      {[6, 146].map((x) => (
        <g key={x}>
          <rect x={x} y={memY} width={128} height={11} rx={2} fill={COL.idle} stroke={COL.idleLine} strokeWidth={0.6} />
          <Txt x={x + 64} y={memY + 8} size={6.5} anchor="middle" color={COL.muted}>{T("flow.anim.s5.memory")}</Txt>
        </g>
      ))}
      <Txt x={6} y={193} size={7} color={COL.cur} weight={600}>{T("flow.anim.s5.trips").replace("{n}", String(eagerT))}</Txt>
      <Txt x={146} y={193} size={7} color={COL.blue} weight={600}>
        {T("flow.anim.s5.trips").replace("{n}", String(fusedT < 0 ? 0 : fusedT >= 2 ? memTrips(true) : 1))}
      </Txt>
    </>
  );
}

// ---- s7: which device a problem runs on -----------------------------------------------------------------

type FlowNode = "start" | "d1" | "d2" | "mps" | "cuda" | "cpu";

function DeviceFlow({ s, T }: PicProps) {
  const sc = SCENARIOS[s.step];
  const [m, c] = checks(sc.choice, sc.has);
  const path: FlowNode[] = ["start", "d1", m ? "mps" : "d2", m ? "mps" : c ? "cuda" : "cpu"];
  const at = path[Math.min(s.t, 3)];
  const visited = new Set(path.slice(0, Math.min(s.t, 3) + 1));
  const result = s.t >= 3 ? path[3] : null;
  const POS: Record<FlowNode, [number, number]> = {
    start: [40, 64],
    d1: [138, 62],
    d2: [138, 114],
    mps: [238, 62],
    cuda: [238, 114],
    cpu: [238, 156],
  };
  const edge = (a: FlowNode, b: FlowNode) => visited.has(a) && visited.has(b) && path.indexOf(b) === path.indexOf(a) + 1;
  const ec = (a: FlowNode, b: FlowNode) => (edge(a, b) ? COL.cur : COL.idleLine);
  const ew = (a: FlowNode, b: FlowNode) => (edge(a, b) ? 1.5 : 0.9);
  const hex = (cx: number, cy: number, w: number, h: number) =>
    `${cx - w / 2},${cy} ${cx - w / 2 + 8},${cy - h / 2} ${cx + w / 2 - 8},${cy - h / 2} ${cx + w / 2},${cy} ${cx + w / 2 - 8},${cy + h / 2} ${cx - w / 2 + 8},${cy + h / 2}`;
  const decision = (n: "d1" | "d2", code: string, q: string) => {
    const [x, y] = POS[n];
    const on = at === n;
    return (
      <g>
        <polygon points={hex(x, y, 104, 34)} fill={on ? COL.curLight : visited.has(n) ? "#fffbeb" : "white"} stroke={on ? COL.cur : COL.idleLine} strokeWidth={on ? 1.4 : 0.8} />
        <Txt x={x} y={y - 3} size={6.5} anchor="middle" kind="mono">{code}</Txt>
        <Txt x={x} y={y + 8} size={6.5} anchor="middle">{q}</Txt>
      </g>
    );
  };
  const res = (n: "mps" | "cuda" | "cpu", sub: string) => {
    const [x, y] = POS[n];
    const win = result === n;
    return (
      <g>
        <rect x={x - 32} y={y - 13} width={64} height={26} rx={4} fill={win ? COL.greenLight : "white"} stroke={win ? COL.green : COL.idleLine} strokeWidth={win ? 1.5 : 0.8} />
        <Txt x={x} y={y - 1} size={8} anchor="middle" kind="mono" weight={win ? 700 : undefined} color={win ? COL.green : COL.text}>{n}</Txt>
        <Txt x={x} y={y + 9} size={6} anchor="middle" color={COL.muted}>{sub}</Txt>
      </g>
    );
  };
  const yes = T("flow.anim.yes");
  const no = T("flow.anim.no");
  const [tx, ty] = POS[at];
  return (
    <>
      <rect x={4} y={4} width={W - 8} height={20} rx={3} fill={COL.idle} />
      <Txt x={10} y={17} size={7.5} weight={600}>{T(`flow.anim.s7.m${sc.machine}`)}</Txt>
      <Txt x={W - 10} y={17} size={6.5} anchor="end" kind="mono" color={COL.blue}>{`TENSORIZER_DEVICE="${sc.choice}"`}</Txt>
      {/* edges */}
      <Arrow x1={72} y1={64} x2={86} y2={62} color={ec("start", "d1")} w={ew("start", "d1")} />
      <Arrow x1={190} y1={62} x2={205} y2={62} color={ec("d1", "mps")} w={ew("d1", "mps")} />
      <Txt x={197} y={57} size={6} anchor="middle" color={COL.green}>{yes}</Txt>
      <Arrow x1={138} y1={79} x2={138} y2={96} color={ec("d1", "d2")} w={ew("d1", "d2")} />
      <Txt x={142} y={90} size={6} color={COL.red}>{no}</Txt>
      <Arrow x1={190} y1={114} x2={205} y2={114} color={ec("d2", "cuda")} w={ew("d2", "cuda")} />
      <Txt x={197} y={109} size={6} anchor="middle" color={COL.green}>{yes}</Txt>
      <polyline points="138,131 138,156 200,156" fill="none" stroke={ec("d2", "cpu")} strokeWidth={ew("d2", "cpu")} />
      <Arrow x1={199} y1={156} x2={205} y2={156} color={ec("d2", "cpu")} w={ew("d2", "cpu")} />
      <Txt x={142} y={146} size={6} color={COL.red}>{no}</Txt>
      {/* nodes */}
      <rect x={8} y={50} width={64} height={28} rx={4} fill={at === "start" ? COL.curLight : COL.blueLight} stroke={at === "start" ? COL.cur : COL.blue} strokeWidth={0.9} />
      <Txt x={40} y={61} size={6} anchor="middle">{T("flow.anim.s7.panel")}</Txt>
      <Txt x={40} y={72} size={7} anchor="middle" kind="mono" color={COL.blue}>{`"${sc.choice}"`}</Txt>
      {decision("d1", 'choice == "mps"', T("flow.anim.s7.mps"))}
      {decision("d2", "choice in (auto, cuda)", T("flow.anim.s7.cuda"))}
      {res("mps", T("flow.anim.s7.appleGpu"))}
      {res("cuda", T("flow.anim.s7.nvidiaGpu"))}
      {res("cpu", "CPU")}
      {/* the token */}
      {result == null && <circle cx={at === "start" ? tx + 30 : tx - 50} cy={ty} r={4} fill={COL.cur} stroke="#b45309" strokeWidth={1} />}
      {result && (
        <Txt x={8} y={100} size={6.5} kind="mono" color={COL.green}>
          {`device() → ${result}`}
        </Txt>
      )}
    </>
  );
}

const PICTURES: Record<FlowCase, { draw: (p: PicProps) => ReactNode; h: number }> = {
  s2: { draw: (p) => <Vectorise {...p} />, h: 222 },
  s3: { draw: (p) => <GpuCpu {...p} />, h: 194 },
  s4: { draw: (p) => <Contraction {...p} />, h: 134 },
  s5: { draw: (p) => <Graph {...p} />, h: 198 },
  s7: { draw: (p) => <DeviceFlow {...p} />, h: 172 },
};

/**
 * One animation per card of the Tensor dataflow tab: the picture, the numbered steps (the current one
 * highlighted) and a caption. Loops; starts paused under prefers-reduced-motion; pauses while off-screen.
 */
export default function DataflowAnim({ id }: { id: FlowCase }) {
  const T0 = usePageText();
  const T = (k: string) => T0(k as PageKey);
  const cycle = totalFrames(id) + HOLD;
  const clock = useAnimClock(cycle, PLAN[id].tick);
  const { root, g } = clock;
  const s = stageAt(id, g);
  const n = stepCount(id);
  const pic = PICTURES[id];
  const caption = T(`flow.anim.${id}.caption`);
  return (
    <div ref={root} className="mt-2" data-testid={`flow-anim-${id}`}>
      <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <figcaption className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">
          <span>{T("flow.anim.title")}</span>
          <span className="font-mono tabular-nums text-blue-700" data-testid={`flow-anim-step-${id}`}>
            {s.step + 1}/{n}
          </span>
        </figcaption>
        <div className="grid md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <svg
            viewBox={`0 0 ${W} ${pic.h}`}
            className="mx-auto block h-auto w-full max-w-[540px] p-1"
            role="img"
            aria-label={caption.replace(/[$`\\]/g, "")}
          >
            {pic.draw({ s, T })}
          </svg>
          <ol className="space-y-1 border-t border-gray-100 p-3 text-xs leading-snug md:border-t-0 md:border-l">
            {Array.from({ length: n }, (_, k) => (
              <li
                key={k}
                className={`flex gap-2 rounded px-1.5 py-1 ${k === s.step ? "bg-blue-50 text-gray-900" : "text-gray-500"}`}
                aria-current={k === s.step ? "step" : undefined}
              >
                <span
                  className={`mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${
                    k === s.step ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-600"
                  }`}
                >
                  {k + 1}
                </span>
                <span>
                  <RichText text={T(`flow.anim.${id}.s${k}`)} />
                </span>
              </li>
            ))}
          </ol>
        </div>
        <p className="border-t border-gray-100 px-3 py-2 text-xs leading-snug text-gray-600">
          <RichText text={caption} />
        </p>
      </figure>
      <AnimControls clock={clock} />
    </div>
  );
}
