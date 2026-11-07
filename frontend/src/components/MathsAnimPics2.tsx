/**
 * Pictures of the Tensor maths animations, part 2: Penrose diagrams, contraction order, the roofline and the
 * recap. Same conventions as MathsAnimPics.tsx.
 */
import {
  COUNT,
  FUSION,
  LOOP,
  MACH,
  MPS,
  RANK1,
  VAN_I,
  att,
  busy,
  chainSteps,
  countAt,
  intensityOf,
  loopTerms,
  mpsTerms,
  rank1Matrix,
  sweepI,
} from "../mathsAnim";
import { COL, Txt, mathSpans } from "./animKit";
import { Arrow, MU, Mat, W, fint, fmt, type PicProps } from "./MathsAnimPics";

// ---- 3. Penrose ---------------------------------------------------------------------------------------

function Node({ x, y, label, r = 12, hot }: { x: number; y: number; label: string; r?: number; hot?: boolean }) {
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={hot ? COL.curLight : COL.blueLight} stroke={hot ? COL.cur : COL.blue} strokeWidth={1.4} />
      <Txt x={x} y={y + 3.5} size={10} anchor="middle" kind="math">
        {label}
      </Txt>
    </g>
  );
}

/** Index label just above a horizontal leg (never on the line). */
const LegLbl = ({ x, y, t, color = COL.cur }: { x: number; y: number; t: string; color?: string }) => (
  <Txt x={x} y={y - 5} size={8} anchor="middle" kind="math" color={color}>
    {t}
  </Txt>
);

export function Wire({ s, T }: PicProps) {
  const y = 46;
  const ax = 150;
  const vx = s.step === 0 ? 250 : s.step === 1 ? [250, 228, 210][s.t] : 210;
  const ux = s.step < 2 ? -40 : [40, 62, 90][s.t];
  const jJoined = (s.step === 1 && s.t === 2) || s.step === 2;
  const iJoined = s.step === 2 && s.t === 2;
  const free = s.step === 0 || (s.step === 1 && s.t < 2) ? 3 : s.step === 1 || !iJoined ? 1 : 0;
  return (
    <>
      {/* leg i of A */}
      <line x1={iJoined ? ux : ax - 40} y1={y} x2={ax} y2={y} stroke={iJoined ? COL.cur : COL.text} strokeWidth={2} />
      <LegLbl x={ax - 26} y={y} t="i" color={iJoined ? COL.cur : COL.blue} />
      {/* leg j of A and of v */}
      {jJoined ? (
        <line x1={ax} y1={y} x2={vx} y2={y} stroke={COL.cur} strokeWidth={2} />
      ) : (
        <>
          <line x1={ax} y1={y} x2={ax + 32} y2={y} stroke={COL.text} strokeWidth={2} />
          <line x1={vx - 26} y1={y} x2={vx} y2={y} stroke={COL.text} strokeWidth={2} />
          <LegLbl x={vx - 18} y={y} t="j" color={COL.blue} />
        </>
      )}
      <LegLbl x={ax + 24} y={y} t="j" color={jJoined ? COL.cur : COL.blue} />
      {s.step === 2 && !iJoined && <line x1={ux} y1={y} x2={ux + 24} y2={y} stroke={COL.text} strokeWidth={2} />}
      {s.step === 2 && <Node x={ux} y={y} label="u" />}
      <Node x={ax} y={y} label="A" />
      <Node x={vx} y={y} label="v" />
      <Txt x={10} y={86} size={7.5} color={COL.text}>
        {`${T("maths.anim.wire.free")}: ${free}`}
      </Txt>
      <Txt x={W - 10} y={86} size={7.5} anchor="end" kind="math" color={free === 1 ? COL.blue : free === 0 ? COL.green : COL.muted} weight={600}>
        {free === 1 ? "w_i = A_{ij} v_j" : free === 0 ? "u_i A_{ij} v_j" : ""}
      </Txt>
      <Txt x={W - 10} y={98} size={6.5} anchor="end" color={free === 1 ? COL.blue : COL.green}>
        {free === 1 ? T("maths.anim.wire.vector") : free === 0 ? T("maths.anim.wire.scalar") : ""}
      </Txt>
      <Txt x={10} y={112} size={7} kind="mono" color={COL.muted}>
        {free === 3 ? "A: (3, 4)   v: (4,)" : free === 1 ? "A @ v: (3, 4), (4,) -> (3,)" : "u @ A @ v: -> ()"}
      </Txt>
    </>
  );
}

export function EqLoop({ s }: PicProps) {
  const terms = loopTerms();
  const n = s.step === 0 ? s.t + 1 : 4;
  const cur = s.step === 0 ? terms[s.t] : null;
  const sum = terms.slice(0, n).reduce((a, t) => a + t.a * t.b, 0);
  const A = { x: 40, y: 62 };
  const B = { x: 122, y: 62 };
  const fromB = s.step === 2;
  return (
    <>
      <path d={`M ${A.x} ${A.y} C ${A.x} 18, ${B.x} 18, ${B.x} ${B.y}`} fill="none" stroke={cur ? COL.cur : COL.text} strokeWidth={2} />
      <path d={`M ${A.x} ${A.y} C ${A.x} 106, ${B.x} 106, ${B.x} ${B.y}`} fill="none" stroke={cur ? COL.cur : COL.text} strokeWidth={2} />
      {/* labels sit outside the arcs: j above the top one, i below the bottom one */}
      <Txt x={81} y={22} size={8} anchor="middle" kind="math" color={COL.cur}>
        {cur ? `j = ${cur.j + 1}` : "j"}
      </Txt>
      <Txt x={81} y={106} size={8} anchor="middle" kind="math" color={COL.cur}>
        {cur ? `i = ${cur.i + 1}` : "i"}
      </Txt>
      <Node x={A.x} y={A.y} label="A" hot={!fromB && s.step === 0} />
      <Node x={B.x} y={B.y} label="B" hot={fromB} />
      <Txt x={160} y={16} size={8} kind="math">A</Txt>
      <Mat x={156} y={20} m={LOOP.A} cs={20} fill={(r, c) => (cur && r === cur.i && c === cur.j ? COL.curLight : undefined)} stroke={(r, c) => (cur && r === cur.i && c === cur.j ? COL.cur : undefined)} />
      <Txt x={222} y={16} size={8} kind="math">B</Txt>
      <Mat x={218} y={20} m={LOOP.B} cs={20} fill={(r, c) => (cur && r === cur.j && c === cur.i ? COL.curLight : undefined)} stroke={(r, c) => (cur && r === cur.j && c === cur.i ? COL.cur : undefined)} />
      {cur && (
        <Txt x={156} y={76} size={7.5} kind="math" color={COL.cur}>
          {`A_{${cur.i + 1}${cur.j + 1}} B_{${cur.j + 1}${cur.i + 1}} = ${cur.a}·${cur.b}`}
        </Txt>
      )}
      <Txt x={156} y={92} size={7.5} kind="math" color={COL.blue} weight={600}>
        {`${terms.slice(0, n).map((t) => `${t.a}·${t.b}`).join(" + ")}${n === 4 ? ` = ${sum}` : ""}`}
      </Txt>
      {s.step >= 1 && (
        <Txt x={156} y={106} size={8} kind="math" color={COL.green} weight={600}>
          {fromB ? "tr(BA) = 13 = tr(AB)" : "tr(AB) = 13"}
        </Txt>
      )}
      <Txt x={10} y={124} size={7} kind="mono" color={COL.blue}>
        {'einsum("ij,ji->", A, B)'}
      </Txt>
    </>
  );
}

export function EqMps({ s }: PicProps) {
  const xs = [60, 140, 220];
  const y = 36;
  const terms = mpsTerms();
  const n = s.step === 3 ? s.t + 1 : 0;
  return (
    <>
      {s.step === 1 && <rect x={42} y={18} width={116} height={64} rx={14} fill={COL.curLight} opacity={0.55} />}
      {s.step >= 2 && <rect x={42} y={18} width={196} height={64} rx={14} fill={COL.curLight} opacity={0.55} />}
      <line x1={xs[0]} y1={y} x2={xs[2]} y2={y} stroke={COL.text} strokeWidth={2} />
      {xs.map((x, k) => (
        <g key={k}>
          <line x1={x} y1={y} x2={x} y2={72} stroke={COL.text} strokeWidth={2} />
          <Txt x={x + 8} y={72} size={8} kind="math" color={COL.blue}>
            {"ijk"[k]}
          </Txt>
        </g>
      ))}
      <Txt x={100} y={y - 6} size={8} anchor="middle" kind="math" color={s.step >= 1 ? COL.cur : COL.muted}>
        α
      </Txt>
      <Txt x={180} y={y - 6} size={8} anchor="middle" kind="math" color={s.step >= 2 ? COL.cur : COL.muted}>
        β
      </Txt>
      {xs.map((x, k) => (
        <Node key={k} x={x} y={y} label={"ABC"[k]} r={11} />
      ))}
      <Txt x={10} y={98} size={7} kind="mono" color={COL.text}>
        {["A: (2, 2)  B: (2, 2, 2)  C: (2, 2)", "AB: (2, 2, 2), legs i, j, β", "T: (2, 2, 2), legs i, j, k", "T: (2, 2, 2), legs i, j, k"][s.step]}
      </Txt>
      {s.step === 3 && (
        <>
          <Txt x={10} y={114} size={7.5} kind="math">{"T_{111} = A_{1α} B_{α1β} C_{β1}"}</Txt>
          <Txt x={10} y={128} size={7.5} kind="math" color={COL.cur} weight={600}>
            {`= ${terms.slice(0, n).map((t) => `${MPS.A[0][t.a]}·${MPS.B[t.a][0][t.b]}·${MPS.C[t.b][0]}`).join(" + ")}${n === 4 ? ` = ${terms.reduce((a, t) => a + t.v, 0)}` : ""}`}
          </Txt>
        </>
      )}
      <Txt x={W - 10} y={114} size={6.5} anchor="end" kind="mono" color={COL.blue}>
        {'"ia,ajb,bk->ijk"'}
      </Txt>
    </>
  );
}

// ---- 4. contraction order -----------------------------------------------------------------------------

interface Blk {
  name: string;
  rows: number;
  cols: number;
  prod?: boolean;
}

function Lane({ y, blocks, label }: { y: number; blocks: Blk[]; label: string }) {
  const sc = 0.5;
  let x = 52;
  return (
    <g>
      <Txt x={4} y={y + 3} size={7.5} kind="math" weight={600}>
        {label}
      </Txt>
      {blocks.map((b) => {
        const w = Math.max(b.cols * sc, 2);
        const h = Math.max(b.rows * sc, 2);
        const bx = x;
        x += w + 26;
        return (
          <g key={b.name}>
            <rect x={bx} y={y - h / 2} width={w} height={h} fill={b.prod ? COL.curLight : COL.blueLight} stroke={b.prod ? COL.cur : COL.blue} strokeWidth={0.8} />
            <Txt x={bx + w / 2} y={y - h / 2 - 3} size={7} anchor="middle" kind="math" color={b.prod ? COL.cur : COL.text}>
              {b.name}
            </Txt>
            <Txt x={bx + w / 2} y={y + h / 2 + 8} size={5.5} anchor="middle" kind="mono" color={COL.muted}>
              {`${b.rows}×${b.cols}`}
            </Txt>
          </g>
        );
      })}
    </g>
  );
}

export function Chain({ s, T }: PicProps) {
  const st = chainSteps();
  const A: Blk = { name: "A", rows: 10, cols: 100 };
  const B: Blk = { name: "B", rows: 100, cols: 5 };
  const C: Blk = { name: "C", rows: 5, cols: 50 };
  const p1 = s.step < 1 ? 0 : s.step === 1 ? (s.t < 2 ? 1 : 2) : 2;
  const p2 = s.step < 2 ? 0 : s.step === 2 ? (s.t < 2 ? 1 : 2) : 2;
  // the last step shows the two intermediates side by side: small AB against big BC
  const d1 = s.step === 3 ? 1 : p1;
  const d2 = s.step === 3 ? 1 : p2;
  const lane1 = [[A, B, C], [{ name: "AB", rows: 10, cols: 5, prod: true }, C], [{ name: "ABC", rows: 10, cols: 50, prod: true }]][d1];
  const lane2 = [[A, B, C], [A, { name: "BC", rows: 100, cols: 50, prod: true }], [{ name: "ABC", rows: 10, cols: 50, prod: true }]][d2];
  const cost1 = (p1 >= 1 ? st.left[0].cost : 0) + (p1 >= 2 ? st.left[1].cost : 0);
  const cost2 = (p2 >= 1 ? st.right[0].cost : 0) + (p2 >= 2 ? st.right[1].cost : 0);
  const X = (v: number) => 52 + (v / 80000) * 214;
  return (
    <>
      <Lane y={44} blocks={lane1} label="(AB)C" />
      <Lane y={124} blocks={lane2} label="A(BC)" />
      <Txt x={W - 6} y={46} size={7.5} anchor="end" kind="mono" color={COL.blue} weight={600}>
        {fint(cost1)}
      </Txt>
      <Txt x={W - 6} y={126} size={7.5} anchor="end" kind="mono" color={COL.cur} weight={600}>
        {fint(cost2)}
      </Txt>
      <rect x={X(0)} y={172} width={X(cost1) - X(0)} height={8} fill={COL.blue} />
      <rect x={X(0)} y={184} width={X(cost2) - X(0)} height={8} fill={COL.cur} />
      <Txt x={48} y={179} size={6.5} anchor="end" kind="math">{"(AB)C"}</Txt>
      <Txt x={48} y={191} size={6.5} anchor="end" kind="math">{"A(BC)"}</Txt>
      {s.step === 3 && (
        <Txt x={X(cost1) + 4} y={179} size={6.5} color={COL.green} weight={700}>
          ✓
        </Txt>
      )}
      <line x1={X(0)} y1={194} x2={X(80000)} y2={194} stroke={COL.text} strokeWidth={0.6} />
      {[0, 20000, 40000, 60000, 80000].map((v) => (
        <g key={v}>
          <line x1={X(v)} y1={194} x2={X(v)} y2={197} stroke={COL.text} strokeWidth={0.6} />
          <Txt x={X(v)} y={204} size={5.5} anchor="middle" color={COL.muted} kind="math">
            {fint(v)}
          </Txt>
        </g>
      ))}
      <Txt x={(X(0) + X(80000)) / 2} y={215} size={6.5} anchor="middle" color={COL.muted}>
        {T("maths.anim.chain.axis")}
      </Txt>
    </>
  );
}

export function EqChain({ s, T }: PicProps) {
  const k = s.step === 0 ? s.t : COUNT.a * COUNT.b * COUNT.c;
  const p = s.step === 0 ? countAt(s.t) : null;
  const doneOut = (r: number, c: number) => (s.step > 0 ? true : r * COUNT.c + c < Math.floor(s.t / COUNT.b));
  const A = Array.from({ length: COUNT.a }, () => Array(COUNT.b).fill(""));
  const B = Array.from({ length: COUNT.b }, () => Array(COUNT.c).fill(""));
  const C = Array.from({ length: COUNT.a }, () => Array(COUNT.c).fill(""));
  const lines = [
    "(AB)C:  abc + acd = 5 000 + 2 500 = 7 500",
    "A(BC):  bcd + abd = 25 000 + 50 000 = 75 000",
  ];
  return (
    <>
      <Mat x={12} y={22} m={A} cs={14} fill={(r, c) => (p && r === p.i ? (c === p.j ? COL.cur : COL.blueLight) : undefined)} />
      <Txt x={56} y={36} size={9} anchor="middle" color={COL.muted}>·</Txt>
      <Mat x={64} y={15} m={B} cs={14} fill={(r, c) => (p && c === p.k2 ? (r === p.j ? COL.cur : COL.blueLight) : undefined)} />
      <Txt x={100} y={36} size={10} anchor="middle" color={COL.muted}>=</Txt>
      <Mat x={108} y={22} m={C} cs={14} fill={(r, c) => (p && r === p.i && c === p.k2 ? COL.curLight : doneOut(r, c) ? COL.done : undefined)} />
      <Txt x={12} y={14} size={7} kind="math" color={COL.muted}>{"a × b"}</Txt>
      <Txt x={64} y={9} size={7} kind="math" color={COL.muted}>{"b × c"}</Txt>
      <Txt x={150} y={22} size={7.5} kind="math">{"a = 2,  b = 3,  c = 2"}</Txt>
      <Txt x={150} y={38} size={8} color={COL.cur} weight={600}>
        {`${T("maths.anim.eqChain.count")}: ${Math.min(k + (s.step === 0 ? 1 : 0), 12)}`}
      </Txt>
      {s.step >= 1 && (
        <Txt x={150} y={54} size={8.5} kind="math" color={COL.blue} weight={600}>
          {"abc = 2·3·2 = 12"}
        </Txt>
      )}
      <g opacity={s.step === 2 ? 1 : 0.3}>
        <Txt x={10} y={78} size={7} kind="math" color={COL.muted}>{"a, b, c, d = 10, 100, 5, 50"}</Txt>
        {lines.map((l, n) => (
          <Txt key={n} x={10} y={93 + n * 14} size={7} kind="math" color={n === 0 && s.step === 2 && s.t >= 2 ? COL.green : COL.text} weight={n === 0 && s.t >= 2 ? 600 : undefined}>
            {s.step === 2 && s.t < n ? "" : l}
          </Txt>
        ))}
        <Txt x={10} y={132} size={7} kind="mono" color={COL.blue}>
          {s.step === 2 && s.t >= 2 ? "multi_dot([A, B, C])  # runs (AB)C" : "multi_dot([A, B, C])"}
        </Txt>
      </g>
    </>
  );
}

// ---- 5. roofline --------------------------------------------------------------------------------------

export function Van({ s, g, T }: PicProps) {
  const meal = s.step === 1;
  const I = VAN_I[s.step];
  const u = busy(I);
  const packets: number[] = meal ? (s.t % 6 < 2 ? [70 + (s.t % 6) * 70] : []) : [0, 1, 2, 3].map((k) => 70 + ((g * 12 + k * 36) % 140));
  const X = (v: number) => 78 + v * 172;
  return (
    <>
      <Txt x={140} y={12} size={7.5} anchor="middle" weight={600}>
        {T(meal ? "maths.anim.van.meal" : "maths.anim.van.salad")}
      </Txt>
      <rect x={6} y={30} width={60} height={50} rx={4} fill={COL.blueLight} stroke={COL.blue} />
      {[0, 1, 2, 3].map((r) => (
        <rect key={r} x={14} y={38 + r * 10} width={44} height={6} fill="white" stroke={COL.doneLine} strokeWidth={0.6} />
      ))}
      <Txt x={6} y={25} size={6.5} color={COL.blue}>
        {T("maths.anim.van.mem")}
      </Txt>
      <line x1={66} y1={55} x2={214} y2={55} stroke={COL.idleLine} strokeWidth={6} strokeLinecap="round" />
      <line x1={66} y1={55} x2={214} y2={55} stroke="white" strokeWidth={0.8} strokeDasharray="4 4" />
      {packets.map((x, k) => (
        <rect key={k} x={x} y={51} width={8} height={8} rx={1} fill={COL.blue} />
      ))}
      <rect x={214} y={30} width={60} height={50} rx={4} fill={COL.cur} fillOpacity={0.15 + 0.6 * u.alu * (g % 2 ? 1 : 0.8)} stroke={COL.cur} />
      <Txt x={244} y={62} size={16} anchor="middle" color={COL.text} kind="math">
        {"×+"}
      </Txt>
      <Txt x={274} y={25} size={6.5} anchor="end" color={COL.cur}>
        {T("maths.anim.van.alu")}
      </Txt>
      <MU x={140} y={74} m={`I = ${fmt(I)}`} u=" FLOP/byte" size={7.5} anchor="middle" />
      <Txt x={74} y={102} size={6.5} anchor="end" color={COL.blue}>{T("maths.anim.van.memBar")}</Txt>
      <rect x={X(0)} y={96} width={X(u.mem) - X(0)} height={8} fill={COL.blue} />
      <Txt x={X(u.mem) + 3} y={102.5} size={6.5} kind="math">{`${Math.round(u.mem * 100)}`}</Txt>
      <Txt x={74} y={116} size={6.5} anchor="end" color={COL.cur}>{T("maths.anim.van.aluBar")}</Txt>
      <rect x={X(0)} y={110} width={X(u.alu) - X(0)} height={8} fill={COL.cur} />
      <Txt x={X(u.alu) + 3} y={116.5} size={6.5} kind="math">{`${Math.round(u.alu * 100)}`}</Txt>
      <line x1={X(0)} y1={122} x2={X(1)} y2={122} stroke={COL.text} strokeWidth={0.6} />
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={X(v)} y1={122} x2={X(v)} y2={125} stroke={COL.text} strokeWidth={0.6} />
          <Txt x={X(v)} y={133} size={6} anchor="middle" kind="math" color={COL.muted}>{String(v * 100)}</Txt>
        </g>
      ))}
      <Txt x={X(0.5)} y={145} size={6.5} anchor="middle" color={COL.muted}>
        {T("maths.anim.van.axis")}
      </Txt>
      <Txt x={W - 4} y={145} size={6} anchor="end" color={COL.muted} kind="math">
        {`P = ${MACH.P}, β = ${MACH.beta}`}
      </Txt>
    </>
  );
}

export function Fusion({ s, T }: PicProps) {
  const ax = [30, 90, 150, 210];
  const names = ["x", "t_1", "t_2", "y"];
  const fused = s.step >= 1;
  const k = s.step === 0 ? Math.floor(s.t / 3) : -1;
  const ph = s.step === 0 ? s.t % 3 : s.step === 1 ? s.t : 2;
  const movedUnfused = s.step === 0 ? k * 2 + 1 + (ph >= 2 ? 1 : 0) : FUSION.unfused.moved;
  const movedFused = s.step === 1 ? 1 + (ph >= 2 ? 1 : 0) : FUSION.fused.moved;
  const kernels = ["a * x", "+ b", "relu"];
  const kx = (i: number) => (ax[i] + ax[i + 1]) / 2 + 20;
  return (
    <>
      <Txt x={4} y={12} size={6.5} color={COL.muted}>{T("maths.anim.fusion.mem")}</Txt>
      {ax.map((x, i) => {
        const ghost = fused && (i === 1 || i === 2);
        return (
          <g key={i} opacity={ghost ? 0.3 : 1}>
            <rect x={x} y={18} width={40} height={18} rx={3} fill={ghost ? "white" : COL.blueLight} stroke={COL.blue} strokeDasharray={ghost ? "2 2" : undefined} />
            <Txt x={x + 20} y={30.5} size={8} anchor="middle" kind="math">{names[i]}</Txt>
          </g>
        );
      })}
      {!fused &&
        kernels.map((kn, i) => {
          const on = i === k;
          const past = i < k;
          return (
            <g key={i} opacity={on ? 1 : past ? 0.55 : 0.3}>
              <rect x={kx(i) - 25} y={90} width={50} height={22} rx={4} fill={on && ph === 1 ? COL.cur : COL.curLight} stroke={COL.cur} />
              <Txt x={kx(i)} y={104} size={7} anchor="middle" kind="mono" color={on && ph === 1 ? "white" : COL.text}>{kn}</Txt>
              {(past || (on && ph >= 0)) && <Arrow x1={ax[i] + 20} y1={37} x2={kx(i) - 6} y2={89} color={COL.blue} w={1.1} />}
              {(past || (on && ph >= 2)) && <Arrow x1={kx(i) + 6} y1={89} x2={ax[i + 1] + 20} y2={37} color={COL.red} w={1.1} />}
            </g>
          );
        })}
      {fused && (
        <g opacity={s.step === 1 ? 1 : 0.7}>
          <rect x={60} y={90} width={170} height={22} rx={4} fill={ph === 1 && s.step === 1 ? COL.cur : COL.curLight} stroke={COL.cur} />
          <Txt x={145} y={104} size={7} anchor="middle" kind="mono" color={ph === 1 && s.step === 1 ? "white" : COL.text}>relu(a * x + b)</Txt>
          <Txt x={145} y={124} size={6.5} anchor="middle" color={COL.cur}>{T("maths.anim.fusion.fused")}</Txt>
          <Arrow x1={50} y1={37} x2={100} y2={89} color={COL.blue} w={1.1} />
          {(ph >= 2 || s.step === 2) && <Arrow x1={190} y1={89} x2={230} y2={37} color={COL.red} w={1.1} />}
        </g>
      )}
      <Txt x={10} y={142} size={7} color={COL.text}>{T("maths.anim.fusion.moved")}</Txt>
      <Txt x={W - 10} y={142} size={7.5} anchor="end" kind="math" weight={600} color={fused ? COL.green : COL.red}>
        {s.step === 2 ? `6 → 2` : fused ? String(movedFused) : String(movedUnfused)}
      </Txt>
      <MU
        x={10}
        y={157}
        m={s.step === 2 ? `I = 1/8 → 3/8` : fused ? `I = 3/(2·4) = ${fmt(intensityOf(FUSION.fused), 3)}` : `I = 3/(6·4) = ${fmt(intensityOf(FUSION.unfused), 3)}`}
        u=" FLOP/byte"
        size={7.5}
        color={COL.blue}
      />
    </>
  );
}

export function EqRoof({ s, T }: PicProps) {
  const x0 = 46;
  const x1 = 272;
  const y0 = 18;
  const y1 = 126;
  const X = (I: number) => x0 + ((Math.log10(I) + 2) / 4) * (x1 - x0);
  const Y = (P: number) => y1 - ((Math.log10(P) + 1) / 4) * (y1 - y0);
  const ridge = MACH.P / MACH.beta;
  const roof = `M ${X(0.01)} ${Y(att(0.01))} L ${X(ridge)} ${Y(MACH.P)} L ${X(100)} ${Y(MACH.P)}`;
  const I = sweepI(s.step === 0 ? s.t : 12);
  const sweep = s.step === 0;
  const fa = 1 / 8;
  const fb = 3 / 8;
  return (
    <>
      <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke={COL.idleLine} strokeWidth={0.6} />
      {[0.01, 0.1, 1, 10, 100].map((v) => (
        <g key={`x${v}`}>
          <line x1={X(v)} y1={y0} x2={X(v)} y2={y1} stroke={COL.idle} strokeWidth={0.6} />
          <Txt x={X(v)} y={y1 + 9} size={6} anchor="middle" color={COL.muted} kind="math">{fmt(v)}</Txt>
        </g>
      ))}
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={`y${v}`}>
          <line x1={x0} y1={Y(v)} x2={x1} y2={Y(v)} stroke={COL.idle} strokeWidth={0.6} />
          <Txt x={x0 - 3} y={Y(v) + 2} size={6} anchor="end" color={COL.muted} kind="math">{fmt(v)}</Txt>
        </g>
      ))}
      <text x={x0} y={11} fontSize={7} fill={COL.muted}>
        {mathSpans("P")}
        <tspan fontSize="72%" dy="0.3em">att</tspan>
        <tspan dy="-0.25em"> (GFLOP/s)</tspan>
      </text>
      <MU x={(x0 + x1) / 2} y={y1 + 20} m="I" u=" (FLOP/byte)" size={7} color={COL.muted} anchor="middle" />
      <line x1={X(ridge)} y1={Y(MACH.P)} x2={X(ridge)} y2={y1} stroke={COL.muted} strokeDasharray="2 2" strokeWidth={0.7} />
      <Txt x={X(ridge) + 3} y={y1 - 4} size={6.5} kind="math" color={COL.muted}>{`I^* = ${ridge}`}</Txt>
      <path d={roof} fill="none" stroke={COL.blue} strokeWidth={2} />
      <Txt x={X(0.03)} y={Y(400)} size={6.5} color={COL.muted}>{T("maths.anim.eqRoof.mem")}</Txt>
      <Txt x={X(60)} y={Y(60)} size={6.5} anchor="end" color={COL.muted}>{T("maths.anim.eqRoof.comp")}</Txt>
      {sweep && (
        <>
          <line x1={X(I)} y1={y1} x2={X(I)} y2={Y(att(I))} stroke={COL.cur} strokeDasharray="2 1.5" strokeWidth={0.8} />
          <circle cx={X(I)} cy={Y(att(I))} r={3.5} fill={COL.cur} />
        </>
      )}
      {!sweep && (
        <>
          <circle cx={X(fa)} cy={Y(att(fa))} r={3} fill="white" stroke={COL.red} strokeWidth={1.3} />
          <circle cx={X(fb)} cy={Y(att(fb))} r={3.5} fill={COL.green} />
          {s.t >= 1 && <Arrow x1={X(fa) + 4} y1={Y(att(fa)) + 6} x2={X(fb) - 2} y2={Y(att(fb)) + 6} color={COL.green} w={1.2} />}
          <Txt x={X(fb) + 6} y={Y(att(fb)) + 12} size={7} color={COL.green} weight={700} kind="math">×3</Txt>
        </>
      )}
      <Txt x={10} y={162} size={7.5} kind="math" color={sweep ? COL.cur : COL.green} weight={600}>
        {sweep
          ? `I = ${fmt(I, 2)}:  βI = ${fmt(MACH.beta * I, 1)},  P = ${MACH.P}  ⇒  P_{att} = ${fmt(att(I), 1)}`
          : `I: 1/8 → 3/8,  P_{att}: ${fmt(att(fa), 2)} → ${fmt(att(fb), 2)}`}
      </Txt>
      {!sweep && (
        <Txt x={W - 6} y={174} size={6.5} anchor="end" kind="mono" color={COL.blue}>
          torch.compile(relu(a*x+b))
        </Txt>
      )}
    </>
  );
}

// ---- 6. recap -----------------------------------------------------------------------------------------

export function Ranks({ s, T }: PicProps) {
  const M = rank1Matrix();
  const row = s.step === 1 ? s.t : -1;
  return (
    <>
      <Txt x={22} y={30} size={8} kind="math" anchor="middle">u</Txt>
      <Mat x={14} y={36} m={RANK1.u.map((x) => [x])} cs={18} fill={(r) => (r === row ? COL.curLight : undefined)} />
      <Txt x={110} y={9} size={8} kind="math" anchor="middle">v</Txt>
      <Mat x={42} y={14} m={[RANK1.v]} cs={18} fill={() => (s.step >= 1 ? COL.blueLight : undefined)} />
      <Mat
        x={42}
        y={36}
        m={M}
        cs={18}
        fill={(r) => (r === row ? COL.curLight : s.step === 2 || r < row ? COL.done : undefined)}
        stroke={(r) => (r === row ? COL.cur : undefined)}
      />
      {s.step === 0 && (
        <>
          <Arrow x1={8} y1={36} x2={8} y2={106} color={COL.blue} />
          <Txt x={4} y={118} size={8} kind="math" color={COL.blue}>i</Txt>
          <Arrow x1={42} y1={112} x2={112} y2={112} color={COL.blue} />
          <Txt x={118} y={115} size={8} kind="math" color={COL.blue}>j</Txt>
        </>
      )}
      <Txt x={130} y={30} size={8} kind="math">{"M_{ij} = u_i v_j"}</Txt>
      {row >= 0 && (
        <Txt x={130} y={50} size={8} kind="math" color={COL.cur} weight={600}>
          {`M_{${row + 1}j} = ${RANK1.u[row]} · v_j`}
        </Txt>
      )}
      <Txt x={130} y={74} size={7.5} color={COL.blue} weight={600}>{`${T("maths.anim.ranks.tensor")} = 2`}</Txt>
      <Txt x={130} y={86} size={6.5} kind="mono" color={COL.blue}>M.ndim == 2</Txt>
      <g opacity={s.step === 2 ? 1 : 0.3}>
        <Txt x={130} y={104} size={7.5} color={COL.green} weight={600}>{`${T("maths.anim.ranks.matrix")} = 1`}</Txt>
        <Txt x={130} y={116} size={6.5} kind="mono" color={COL.green}>matrix_rank(M) == 1</Txt>
      </g>
    </>
  );
}
