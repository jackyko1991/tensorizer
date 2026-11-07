import { type ReactNode } from "react";
import { usePageText, type PageKey } from "../content/pages";
import {
  KM,
  KNN,
  KNN_K,
  KNN_TE,
  KNN_TR,
  KNN_YTR,
  KM_X,
  KS,
  KS_C,
  KS_GAMMA,
  KS_X,
  KS_Y,
  LIN_X,
  LIN_Y,
  LR_B,
  LR_W0,
  LR_W1,
  PCA,
  PCA_X,
  PLAN,
  SVM_W0,
  SVM_WSTAR,
  lrParts,
  stageAt,
  stepCount,
  svmParts,
  totalFrames,
  type CpumlCase,
  type M,
} from "../cpumlAnim";
import { AnimControls, COL, Txt, blueScale, useAnimClock } from "./animKit";
import { RichText } from "./Tex";

const W = 280; // SVG user units, scaled to the column width
const HOLD = 8; // ticks the finished picture stays before the cycle restarts
const CLS = [COL.blue, COL.cur, COL.green];
const CLS_LIGHT = [COL.blueLight, COL.curLight, COL.greenLight];

type Stage = ReturnType<typeof stageAt>;
interface PicProps {
  s: Stage;
}

const fmt = (v: number, d = 1) => {
  const r = Number(v.toFixed(d));
  return (Object.is(r, -0) ? 0 : r).toFixed(d).replace("-", "−");
};

/** Diverging fill: blue for positive, amber for negative values, |v| / max in [0, 1]. */
function divScale(v: number, max: number): string {
  const t = Math.min(Math.abs(v) / max, 1);
  if (v >= 0) return blueScale(t * 0.7);
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(255, 251)},${lerp(251, 191)},${lerp(235, 36)})`; // amber-50 -> amber-400
}
const absMax = (A: M) => Math.max(...A.flat().map(Math.abs), 1e-9);

function Arrow({ x1, y1, x2, y2, color, w = 1.3, head = 4 }: { x1: number; y1: number; x2: number; y2: number; color: string; w?: number; head?: number }) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  if (Math.hypot(x2 - x1, y2 - y1) < 0.5) return null;
  const hx = (s: number) => x2 - head * Math.cos(a + s * 0.45);
  const hy = (s: number) => y2 - head * Math.sin(a + s * 0.45);
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2 - 0.6 * head * Math.cos(a)} y2={y2 - 0.6 * head * Math.sin(a)} stroke={color} strokeWidth={w} />
      <path d={`M ${x2} ${y2} L ${hx(1)} ${hy(1)} L ${hx(-1)} ${hy(-1)} Z`} fill={color} />
    </g>
  );
}

/** Plot frame with fixed limits: the data -> SVG maps. */
function frame(x0: number, y0: number, w: number, h: number, xl: [number, number], yl: [number, number]) {
  return {
    X: (v: number) => x0 + ((v - xl[0]) / (xl[1] - xl[0])) * w,
    Y: (v: number) => y0 + h - ((v - yl[0]) / (yl[1] - yl[0])) * h,
  };
}

/** Axes with fixed limits; the labels carry units (a.u. when the quantity has none). */
function Axes({ x0, y0, w, h, xl, yl, xlab, ylab, xt, yt, d = 0 }: {
  x0: number;
  y0: number;
  w: number;
  h: number;
  xl: [number, number];
  yl: [number, number];
  xlab: string;
  ylab: string;
  xt?: number[];
  yt?: number[];
  d?: number;
}) {
  const { X, Y } = frame(x0, y0, w, h, xl, yl);
  return (
    <g>
      <rect x={x0} y={y0} width={w} height={h} fill="white" stroke={COL.idleLine} strokeWidth={0.6} />
      {(xt ?? xl).map((v) => (
        <Txt key={`x${v}`} x={X(v)} y={y0 + h + 7} size={6} anchor="middle" color={COL.muted} kind="math">
          {fmt(v, d)}
        </Txt>
      ))}
      {(yt ?? yl).map((v) => (
        <Txt key={`y${v}`} x={x0 - 2} y={Y(v) + 2.1} size={6} anchor="end" color={COL.muted} kind="math">
          {fmt(v, d)}
        </Txt>
      ))}
      <Txt x={x0 + w} y={y0 + h + 15} size={6.5} anchor="end" color={COL.muted} kind="math">
        {xlab}
      </Txt>
      <Txt x={x0 + 1} y={y0 - 2.5} size={6.5} color={COL.muted} kind="math">
        {ylab}
      </Txt>
    </g>
  );
}

/**
 * A matrix of cells with an optional value in each, its name and shape above it (as code). `fill`, `text`
 * and `hot` (amber outline) are per cell; `show` hides the values until the matrix is computed.
 */
function Mat({ x, y, A, cw = 16, ch = 10, title, fill, text, hot, show = true, d = 1, size = 6, titleColor = COL.muted }: {
  x: number;
  y: number;
  A: M;
  cw?: number;
  ch?: number;
  title?: string;
  fill?: (i: number, j: number, v: number) => string;
  text?: (i: number, j: number, v: number) => string | undefined;
  hot?: (i: number, j: number) => boolean;
  show?: boolean;
  d?: number;
  size?: number;
  titleColor?: string;
}) {
  return (
    <g>
      {title && (
        <Txt x={x} y={y - 2.5} size={6.3} kind="mono" color={titleColor}>
          {title}
        </Txt>
      )}
      {A.map((row, i) =>
        row.map((v, j) => {
          const h = hot?.(i, j);
          return (
            <g key={`${i}-${j}`}>
              <rect
                x={x + j * cw}
                y={y + i * ch}
                width={cw}
                height={ch}
                fill={show ? (fill?.(i, j, v) ?? "white") : COL.idle}
                stroke={h ? COL.cur : COL.idleLine}
                strokeWidth={h ? 1.2 : 0.5}
              />
              {show && (
                <Txt x={x + j * cw + cw / 2} y={y + i * ch + ch / 2 + size * 0.36} size={size} anchor="middle">
                  {text ? (text(i, j, v) ?? "") : fmt(v, d)}
                </Txt>
              )}
            </g>
          );
        }),
      )}
    </g>
  );
}

/** The current line of the card's tensor code, under the picture. */
function Code({ y, children }: { y: number; children: string }) {
  return (
    <Txt x={4} y={y} size={6.3} kind="mono" color={COL.blue} pre>
      {children}
    </Txt>
  );
}

const Op = ({ x, y, children }: { x: number; y: number; children: string }) => (
  <Txt x={x} y={y} size={8} anchor="middle" color={COL.muted} kind="math">
    {children}
  </Txt>
);

const col = (v: number[]): M => v.map((x) => [x]);

// ---- 1. PCA ---------------------------------------------------------------------------------------------

function Pca({ s }: PicProps) {
  const { X, Y } = frame(24, 10, 118, 118, [-3, 6], [-3, 6]);
  const f = s.step === 0 ? 0 : s.step === 1 ? Math.min((s.t + 1) / 4, 1) : 1; // how far the centring has moved
  const pts = PCA_X.map(([a, b]) => [a - f * PCA.mean[0], b - f * PCA.mean[1]]);
  const centred = s.step >= 2 || (s.step === 1 && s.t >= 3);
  const code = [
    "X.mean(dim=0)            # (8, 2) -> (2,)",
    "Xc = X - X.mean(dim=0)   # (8, 2) - (2,) broadcasts",
    "U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)",
    "var = S**2 / (N - 1);  var[:k] / var.sum()",
  ][s.step];
  const sd = PCA.variance.map(Math.sqrt);
  const u = PCA.Vh[0];
  return (
    <>
      <Axes x0={24} y0={10} w={118} h={118} xl={[-3, 6]} yl={[-3, 6]} xlab="x_1 (a.u.)" ylab="x_2 (a.u.)" xt={[-3, 0, 3, 6]} yt={[-3, 0, 3, 6]} />
      <line x1={X(0)} y1={10} x2={X(0)} y2={128} stroke={COL.idleLine} strokeWidth={0.5} strokeDasharray="2 2" />
      <line x1={24} y1={Y(0)} x2={142} y2={Y(0)} stroke={COL.idleLine} strokeWidth={0.5} strokeDasharray="2 2" />
      {s.step === 3 && (
        <line x1={X(-3.4 * u[0])} y1={Y(-3.4 * u[1])} x2={X(3.4 * u[0])} y2={Y(3.4 * u[1])} stroke={COL.blue} strokeWidth={0.6} strokeDasharray="3 2" />
      )}
      {pts.map(([a, b], i) => {
        const p = PCA.scores[i];
        return (
          <g key={i}>
            {s.step === 3 && s.t >= 1 && (
              <>
                <line x1={X(a)} y1={Y(b)} x2={X(p * u[0])} y2={Y(p * u[1])} stroke={COL.muted} strokeWidth={0.5} />
                <circle cx={X(p * u[0])} cy={Y(p * u[1])} r={1.6} fill={COL.blue} />
              </>
            )}
            <circle cx={X(a)} cy={Y(b)} r={2.4} fill={COL.blueLight} stroke={COL.blue} strokeWidth={0.9} />
          </g>
        );
      })}
      {/* the mean: the point every row moves by */}
      {(s.step > 0 || s.t >= 1) && (
        <g stroke={COL.cur} strokeWidth={1.4}>
          {(() => {
            const mx = PCA.mean[0] * (1 - f);
            const my = PCA.mean[1] * (1 - f);
            return (
              <>
                <line x1={X(mx) - 3.5} y1={Y(my) - 3.5} x2={X(mx) + 3.5} y2={Y(my) + 3.5} />
                <line x1={X(mx) - 3.5} y1={Y(my) + 3.5} x2={X(mx) + 3.5} y2={Y(my) - 3.5} />
              </>
            );
          })()}
        </g>
      )}
      {s.step >= 2 &&
        [0, 1].map((k) => (
          <Arrow
            key={k}
            x1={X(0)}
            y1={Y(0)}
            x2={X(2 * sd[k] * PCA.Vh[k][0])}
            y2={Y(2 * sd[k] * PCA.Vh[k][1])}
            color={k === 0 ? COL.blue : COL.violet}
            w={1.5}
          />
        ))}
      {/* the matrices */}
      <Mat
        x={160}
        y={18}
        A={centred ? PCA.Xc : PCA_X}
        cw={20}
        ch={9}
        title={centred ? "Xc (8, 2)" : "X (8, 2)"}
        fill={() => (s.step === 1 && s.t < 3 ? COL.curLight : centred ? COL.blueLight : "white")}
        titleColor={centred ? COL.blue : COL.muted}
      />
      <Mat
        x={160}
        y={104}
        A={[PCA.mean]}
        cw={20}
        ch={9}
        title="mean (2,)"
        show={s.step > 0 || s.t >= 1}
        d={2}
        fill={() => COL.curLight}
      />
      {s.step >= 2 && (
        <>
          <Mat x={214} y={18} A={col(PCA.S)} cw={26} ch={9} title="S (2,)" d={2} fill={(i) => (i === 0 ? COL.blueLight : COL.violetLight)} />
          <Mat x={214} y={50} A={PCA.Vh} cw={26} ch={9} title="Vh (2, 2)" d={2} fill={(i) => (i === 0 ? COL.blueLight : COL.violetLight)} />
        </>
      )}
      {s.step === 3 && (
        <g>
          <Txt x={214} y={81.5} size={6} kind="mono" color={COL.muted}>
            evr (2,)
          </Txt>
          {PCA.evr.map((e, k) => (
            <g key={k}>
              <rect x={214} y={84 + k * 10} width={52} height={9} fill="white" stroke={COL.idleLine} strokeWidth={0.5} />
              <rect x={214} y={84 + k * 10} width={52 * e} height={9} fill={k === 0 ? COL.blueLight : COL.violetLight} />
              <Txt x={240} y={84 + k * 10 + 6.6} size={5.6} anchor="middle">
                {fmt(e, 2)}
              </Txt>
            </g>
          ))}
        </g>
      )}
      <Code y={154}>{code}</Code>
    </>
  );
}

// ---- 2. linear SVM ------------------------------------------------------------------------------------------

const SVM0 = svmParts(SVM_W0);
const SVM1 = svmParts(SVM_WSTAR);

function LinSvm({ s }: PicProps) {
  const done = s.step === 5;
  const P = done ? SVM1 : SVM0;
  const Wcur = done ? SVM_WSTAR : SVM_W0;
  const sRows = s.step < 2 ? 0 : s.step === 2 ? Math.min(s.t + 1, 4) : 4;
  const code = [
    "Xb = torch.cat([X, torch.ones_like(X[:, :1])], dim=1)",
    "Y = 2.0 * (y[:, None] == torch.arange(K)) - 1",
    "Xb @ W                # (4, 3) @ (3, 3) -> (4, 3)",
    "hinge = (1 - Y * (Xb @ W)).clamp(min=0)",
    "loss = 0.5*(W**2).sum() + C*(hinge**2).sum(); loss.backward()",
    "opt.step(objective)   # L-BFGS: all K columns at once",
  ][s.step];
  const wMax = Math.max(absMax(SVM_W0), absMax(SVM_WSTAR));
  const sMax = Math.max(absMax(SVM0.S), absMax(SVM1.S));
  const hMax = absMax(SVM0.H);
  const gMax = absMax(SVM0.G);
  const hotRow = s.step === 2 ? sRows - 1 : -1;
  return (
    <>
      <Mat
        x={4}
        y={18}
        A={P.Xb}
        cw={18}
        ch={11}
        title="Xb (4, 3)"
        fill={(i, j) => (j === 2 ? (s.step === 0 && s.t >= 1 ? COL.curLight : COL.idle) : i === hotRow ? COL.curLight : "white")}
        text={(_, j, v) => (j === 2 && s.step === 0 && s.t < 1 ? "" : fmt(v))}
      />
      <Op x={65} y={42}>@</Op>
      <Mat x={72} y={18} A={Wcur} cw={18} ch={11} title={done ? "W* (3, 3)" : "W (3, 3)"} fill={(_, __, v) => divScale(v, wMax)} titleColor={done ? COL.blue : COL.muted} />
      <Op x={133} y={42}>=</Op>
      <Mat
        x={140}
        y={18}
        A={P.S}
        cw={18}
        ch={11}
        title="Xb @ W (4, 3)"
        show={sRows > 0}
        fill={(i, _, v) => (i < sRows ? divScale(v, sMax) : COL.idle)}
        text={(i, _, v) => (i < sRows ? fmt(v) : "")}
        hot={(i) => i === hotRow}
      />
      <Mat x={206} y={18} A={col(LIN_Y)} cw={10} ch={11} title="y" show={s.step >= 1} text={(_, __, v) => String(v)} fill={(i) => CLS_LIGHT[LIN_Y[i]]} />
      <Mat
        x={220}
        y={18}
        A={P.Y}
        cw={18}
        ch={11}
        title="Y (4, 3)"
        show={s.step >= 1}
        text={(_, __, v) => (v > 0 ? "+1" : "−1")}
        fill={(_, __, v) => (v > 0 ? COL.blueLight : COL.curLight)}
      />
      <Mat
        x={4}
        y={86}
        A={P.H}
        cw={18}
        ch={11}
        title={done ? "hinge at W* (4, 3)" : "hinge (4, 3)"}
        show={s.step >= 3}
        d={2}
        size={5.7}
        fill={(_, __, v) => (v === 0 ? COL.greenLight : divScale(-v, hMax))}
      />
      <Mat
        x={80}
        y={86}
        A={done ? SVM1.G : SVM0.G}
        cw={18}
        ch={11}
        title={done ? "W.grad at W* (3, 3)" : "W.grad (3, 3)"}
        show={s.step >= 4}
        fill={(_, __, v) => divScale(v, gMax)}
      />
      {s.step >= 4 && (
        <>
          <Txt x={160} y={98} size={7.5} kind="math">{`J(W) = ${fmt(SVM0.loss, 2)}`}</Txt>
          {done && (
            <Txt x={160} y={112} size={7.5} kind="math" color={COL.blue}>{`J(W*) = ${fmt(SVM1.loss, 2)}`}</Txt>
          )}
          <Txt x={160} y={124} size={6.5} kind="math" color={COL.muted}>
            C = 1
          </Txt>
        </>
      )}
      <Code y={146}>{code}</Code>
    </>
  );
}

// ---- 3. kernel SVM ----------------------------------------------------------------------------------------

function KSvm({ s }: PicProps) {
  const snap = KS.snaps[s.step === 4 ? s.round : s.step === 5 ? KS.snaps.length - 1 : 0];
  const a = snap.a;
  const slot: { A: M; title: string; signed?: boolean; d: number } =
    s.step === 0
      ? { A: KS.gram, title: "X @ X.T (5, 5)", d: 1 }
      : s.step === 1
        ? { A: KS.D2, title: "D2 (5, 5)", d: 1 }
        : s.step === 2
          ? { A: KS.K, title: "K (5, 5)", d: 1 }
          : { A: KS.Q, title: "Q (5, 5)", signed: true, d: 1 };
  const mx = absMax(slot.A);
  const code = [
    "sq = (X**2).sum(dim=1);  X @ X.T",
    "sq[:, None] + sq[None, :] - 2 * X @ X.T",
    "K = torch.exp(-gamma * D2)",
    "Q = K * (y[:, None] * y[None, :])",
    "a = project(z + (1 - Q @ z) / L, y, C)",
    "f = K @ (a * y);  b = (y - f)[free].mean()",
  ][s.step];
  const sc = frame(122, 14, 62, 62, [0, 3], [0, 3]);
  const al = frame(206, 14, 68, 62, [-0.5, 4.5], [0, 1]);
  const sv = s.step === 5;
  return (
    <>
      {/* sq as a column and (when broadcasting) as a row */}
      <Mat x={4} y={30} A={col(KS.sq)} cw={14} ch={13} title="sq" d={1} size={5.7} fill={() => (s.step === 1 ? COL.curLight : "white")} />
      {s.step === 1 && (
        <Mat x={22} y={17} A={[KS.sq]} cw={13} ch={8} d={1} size={5} fill={() => COL.curLight} />
      )}
      <Mat
        x={22}
        y={30}
        A={slot.A}
        cw={13}
        ch={13}
        title={s.step === 1 ? undefined : slot.title}
        d={slot.d}
        size={5.3}
        fill={(_, __, v) => (slot.signed ? divScale(v, mx) : blueScale((0.75 * v) / mx))}
        text={(_, __, v) => (Math.abs(v) < 0.05 ? "0" : fmt(v, slot.d))}
        titleColor={COL.blue}
      />
      {s.step === 1 && (
        <Txt x={22} y={100} size={6} kind="mono" color={COL.blue}>
          D2 (5, 5)
        </Txt>
      )}
      {s.step >= 4 && (
        <Mat x={92} y={30} A={col(a)} cw={18} ch={13} title="a" d={2} size={5} fill={(_, __, v) => blueScale(0.6 * v)} titleColor={COL.blue} />
      )}
      {/* the points, y = +1 and -1 */}
      <Axes x0={122} y0={14} w={62} h={62} xl={[0, 3]} yl={[0, 3]} xlab="x_1 (a.u.)" ylab="x_2 (a.u.)" xt={[0, 3]} yt={[0, 3]} />
      {KS_X.map(([x, y], i) => {
        const c = KS_Y[i] > 0 ? COL.blue : COL.cur;
        const cx = sc.X(x);
        const cy = sc.Y(y);
        return (
          <g key={i}>
            {sv && KS.alpha[i] > 1e-3 && <circle cx={cx} cy={cy} r={5} fill="none" stroke={COL.green} strokeWidth={1.2} />}
            {KS_Y[i] > 0 ? (
              <circle cx={cx} cy={cy} r={2.6} fill={c} />
            ) : (
              <rect x={cx - 2.4} y={cy - 2.4} width={4.8} height={4.8} fill={c} />
            )}
            <Txt x={cx + 4} y={cy - 3} size={5} color={COL.muted} kind="math">
              {String(i)}
            </Txt>
          </g>
        );
      })}
      {/* the multipliers: 0 <= a_i <= C */}
      <Axes x0={206} y0={14} w={68} h={62} xl={[-0.5, 4.5]} yl={[0, 1]} xlab="i" ylab="α_i (a.u.)" xt={[0, 1, 2, 3, 4]} yt={[0, 1]} />
      <line x1={206} y1={al.Y(KS_C)} x2={274} y2={al.Y(KS_C)} stroke={COL.red} strokeWidth={0.7} strokeDasharray="2 2" />
      <Txt x={275.5} y={al.Y(KS_C) + 2} size={6} color={COL.red} kind="math">
        C
      </Txt>
      {a.map((v, i) => (
        <rect key={i} x={al.X(i) - 5} y={al.Y(v)} width={10} height={al.Y(0) - al.Y(v)} fill={KS_Y[i] > 0 ? COL.blue : COL.cur} opacity={0.85} />
      ))}
      {/* numbers */}
      <Txt x={4} y={114} size={7} kind="math" color={COL.muted}>{`γ = ${fmt(KS_GAMMA, 2)}`}</Txt>
      {s.step >= 4 && (
        <>
          <Txt x={60} y={114} size={7} kind="math" color={COL.muted}>{`L = λ_{max}(Q) = ${fmt(KS.L, 2)}`}</Txt>
          <Txt x={160} y={114} size={7} kind="mono" color={COL.blue}>{`it = ${snap.it}`}</Txt>
          <Txt x={206} y={114} size={7} kind="mono" color={COL.blue}>{`dual = ${fmt(snap.dual, 3)}`}</Txt>
        </>
      )}
      {sv && (
        <Txt x={4} y={128} size={7} kind="math">{`b = ${fmt(KS.b, 2)},   Σ_i y_i α_i = ${fmt(KS.alpha.reduce((t, v, i) => t + v * KS_Y[i], 0), 2)}`}</Txt>
      )}
      <Code y={144}>{code}</Code>
    </>
  );
}

// ---- 4. logistic regression -------------------------------------------------------------------------------

const LR0 = lrParts(LR_W0);
const LR1 = lrParts(LR_W1);

function LogReg({ s }: PicProps) {
  const zRows = s.step === 0 ? Math.min(s.t + 1, 4) : 4;
  const withB = s.step >= 1;
  const Z = withB ? LR0.Z : LR0.XW;
  const code = [
    "X @ W                # (4, 2) @ (2, 3) -> (4, 3)",
    "Z = X @ W + b        # (4, 3) + (3,) broadcasts",
    "P = softmax(Z, dim=1)   # inside F.cross_entropy",
    "P - one_hot(y, K)    # (4, 3)",
    "loss.backward()  # W.grad = X.T @ (P - Y) + W / C",
    "W = W - lr * W.grad  # L-BFGS: better steps, same gradient",
  ][s.step];
  const wMax = Math.max(absMax(LR_W0), absMax(LR_W1));
  const hotRow = s.step === 0 ? zRows - 1 : -1;
  return (
    <>
      <Mat x={4} y={18} A={LIN_X} cw={18} ch={11} title="X (4, 2)" fill={(i) => (i === hotRow ? COL.curLight : "white")} />
      <Op x={46} y={36}>@</Op>
      <Mat x={52} y={18} A={LR_W0} cw={18} ch={11} title="W (2, 3)" d={2} size={5.7} fill={(_, __, v) => divScale(v, wMax)} />
      <Mat
        x={52}
        y={50}
        A={[LR_B]}
        cw={18}
        ch={11}
        title="+ b (3,)"
        show={withB}
        fill={() => (s.step === 1 ? COL.curLight : "white")}
      />
      <Op x={115} y={36}>=</Op>
      <Mat
        x={122}
        y={18}
        A={Z}
        cw={18}
        ch={11}
        title={withB ? "Z (4, 3)" : "X @ W (4, 3)"}
        d={2}
        size={5.7}
        fill={(i, _, v) => (i < zRows ? (s.step === 1 ? COL.curLight : divScale(v, 1.5)) : COL.idle)}
        text={(i, _, v) => (i < zRows ? fmt(v, 2) : "")}
        hot={(i) => i === hotRow}
      />
      <Arrow x1={180} y1={40} x2={196} y2={40} color={s.step === 2 ? COL.cur : COL.muted} />
      <Mat
        x={200}
        y={18}
        A={LR0.P}
        cw={18}
        ch={11}
        title="P (4, 3)"
        show={s.step >= 2}
        d={2}
        size={5.7}
        fill={(_, __, v) => blueScale(0.8 * v)}
        text={(_, __, v) => fmt(v, 2)}
      />
      <Mat x={4} y={90} A={LR0.Y} cw={18} ch={11} title="Y = one_hot(y)" show={s.step >= 3} text={(_, __, v) => String(v)} fill={(i, j) => (LIN_Y[i] === j ? CLS_LIGHT[j] : "white")} />
      <Mat x={72} y={90} A={LR0.PY} cw={18} ch={11} title="P − Y (4, 3)" show={s.step >= 3} d={2} size={5.7} fill={(_, __, v) => divScale(v, 1)} />
      <Mat x={146} y={90} A={LR0.G} cw={18} ch={11} title="W.grad (2, 3)" show={s.step >= 4} d={2} size={5.7} fill={(_, __, v) => divScale(v, absMax(LR0.G))} />
      <Mat
        x={220}
        y={90}
        A={LR_W1}
        cw={18}
        ch={11}
        title="new W (2, 3)"
        show={s.step >= 5}
        d={2}
        size={5.7}
        fill={(_, __, v) => divScale(v, wMax)}
        titleColor={s.step >= 5 ? COL.blue : COL.muted}
      />
      {s.step >= 4 && (
        <Txt x={146} y={148} size={7.5} kind="math">{`J/C = ${fmt(LR0.loss, 2)}`}</Txt>
      )}
      {s.step >= 5 && (
        <Txt x={204} y={148} size={7.5} kind="math" color={COL.blue}>{`→ ${fmt(LR1.loss, 2)}`}</Txt>
      )}
      <Txt x={4} y={148} size={6.5} kind="math" color={COL.muted}>
        C = 1, lr = 0.5
      </Txt>
      <Code y={164}>{code}</Code>
    </>
  );
}

// ---- 5. kNN -------------------------------------------------------------------------------------------------

function Knn({ s }: PicProps) {
  const { X, Y } = frame(22, 10, 112, 112, [0, 5], [0, 5]);
  const dRows = s.step === 0 ? Math.min(s.t + 1, 3) : 3;
  const kRows = s.step === 1 ? Math.min(s.t + 1, 3) : s.step > 1 ? 3 : 0;
  const code = [
    "d = torch.cdist(Xte, Xtr)    # (3, 2), (8, 2) -> (3, 8)",
    "idx = d.topk(k, largest=False).indices   # (3, 3)",
    "votes = one_hot(ytr[idx], K).sum(dim=1)  # (3, 3)",
    "pred = votes.argmax(dim=1)   # (3,)",
  ][s.step];
  const dMax = Math.max(...KNN.D.flat());
  const nearest = (m: number, n: number) => m < kRows && KNN.idx[m].includes(n);
  return (
    <>
      <Axes x0={22} y0={10} w={112} h={112} xl={[0, 5]} yl={[0, 5]} xlab="x_1 (a.u.)" ylab="x_2 (a.u.)" xt={[0, 5]} yt={[0, 5]} />
      {/* distance lines from the current query */}
      {s.step === 0 &&
        KNN_TR.map(([a, b], n) => (
          <line key={n} x1={X(KNN_TE[dRows - 1][0])} y1={Y(KNN_TE[dRows - 1][1])} x2={X(a)} y2={Y(b)} stroke={COL.idleLine} strokeWidth={0.6} />
        ))}
      {s.step >= 1 &&
        KNN_TE.slice(0, kRows).flatMap(([qa, qb], m) =>
          KNN.idx[m].map((n) => <line key={`${m}-${n}`} x1={X(qa)} y1={Y(qb)} x2={X(KNN_TR[n][0])} y2={Y(KNN_TR[n][1])} stroke={COL.cur} strokeWidth={0.9} />),
        )}
      {KNN_TR.map(([a, b], n) => (
        <g key={n}>
          <circle cx={X(a)} cy={Y(b)} r={2.8} fill={CLS[KNN_YTR[n]]} />
          <Txt x={X(a) + 3.5} y={Y(b) + 7} size={5} color={COL.muted} kind="math">
            {String(n)}
          </Txt>
        </g>
      ))}
      {KNN_TE.map(([a, b], m) => {
        const c = s.step === 3 ? CLS[KNN.pred[m]] : "white";
        return (
          <g key={m}>
            <path d={`M ${X(a)} ${Y(b) - 4} L ${X(a) + 4} ${Y(b)} L ${X(a)} ${Y(b) + 4} L ${X(a) - 4} ${Y(b)} Z`} fill={c} stroke={COL.text} strokeWidth={1} />
            <Txt x={X(a) + 5} y={Y(b) - 3} size={6} kind="math">{`q_${m}`}</Txt>
          </g>
        );
      })}
      <Mat
        x={150}
        y={18}
        A={KNN.D}
        cw={15}
        ch={11}
        title="d (3, 8)"
        fill={(i, j, v) => (i < dRows ? (nearest(i, j) ? COL.curLight : blueScale((0.55 * v) / dMax)) : COL.idle)}
        text={(i, _, v) => (i < dRows ? fmt(v) : "")}
        hot={nearest}
        size={5.7}
      />
      <Mat
        x={150}
        y={74}
        A={KNN.idx}
        cw={12}
        ch={11}
        title="idx"
        show={s.step >= 1}
        text={(i, _, v) => (i < kRows ? String(v) : "")}
        fill={(i) => (i < kRows ? COL.curLight : COL.idle)}
      />
      <Mat x={192} y={74} A={KNN.lab} cw={12} ch={11} title="ytr[idx]" show={s.step >= 2} text={(_, __, v) => String(v)} fill={(_, __, v) => CLS_LIGHT[v]} />
      <Mat x={232} y={74} A={KNN.votes} cw={12} ch={11} title="votes" show={s.step >= 2} text={(_, __, v) => String(v)} fill={(_, j, v) => (v ? CLS_LIGHT[j] : "white")} hot={(i, j) => s.step === 3 && KNN.pred[i] === j} />
      <Mat x={232} y={122} A={[KNN.pred]} cw={12} ch={11} title="pred (3,)" show={s.step >= 3} text={(_, __, v) => String(v)} fill={(_, __, v) => CLS_LIGHT[v]} />
      <Txt x={150} y={140} size={6.5} kind="math" color={COL.muted}>{`k = ${KNN_K}`}</Txt>
      <Code y={154}>{code}</Code>
    </>
  );
}

// ---- 6. k-means ---------------------------------------------------------------------------------------------

function KMeans({ s }: PicProps) {
  const { X, Y } = frame(22, 10, 118, 118, [0, 6], [0, 6]);
  const r = KM[s.round];
  const assigned = s.step >= 1;
  const labels = assigned ? r.labels : s.round > 0 ? KM[s.round - 1].labels : null;
  const move = s.step === 3 ? Math.min((s.t + 1) / 3, 1) : 0;
  const centers = r.centers.map((c, j) => c.map((v, d) => v + move * (r.next[j][d] - v)));
  const code = [
    "torch.cdist(X, centers)     # (9, 2), (3, 2) -> (9, 3)",
    "labels = d.argmin(dim=1)    # (9,)",
    "sums = zeros.index_add_(0, labels, X); bincount(labels)",
    "centers = sums / counts[:, None]   # (3, 2) / (3, 1)",
    "torch.equal(new, labels)    # True: converged",
  ][s.step];
  const dMax = Math.max(...KM.flatMap((k) => k.D.flat()));
  return (
    <>
      <Axes x0={22} y0={10} w={118} h={118} xl={[0, 6]} yl={[0, 6]} xlab="x_1 (a.u.)" ylab="x_2 (a.u.)" xt={[0, 3, 6]} yt={[0, 3, 6]} />
      {s.step === 0 &&
        KM_X.flatMap(([a, b], n) =>
          centers.map((c, j) => <line key={`${n}-${j}`} x1={X(a)} y1={Y(b)} x2={X(c[0])} y2={Y(c[1])} stroke={COL.idleLine} strokeWidth={0.4} />),
        )}
      {assigned &&
        KM_X.map(([a, b], n) => (
          <line key={n} x1={X(a)} y1={Y(b)} x2={X(centers[r.labels[n]][0])} y2={Y(centers[r.labels[n]][1])} stroke={CLS[r.labels[n]]} strokeWidth={0.6} opacity={0.6} />
        ))}
      {KM_X.map(([a, b], n) => (
        <circle key={n} cx={X(a)} cy={Y(b)} r={2.8} fill={labels ? CLS_LIGHT[labels[n]] : "white"} stroke={labels ? CLS[labels[n]] : COL.muted} strokeWidth={1} />
      ))}
      {centers.map(([a, b], j) => (
        <g key={j} stroke={CLS[j]} strokeWidth={2}>
          <line x1={X(a) - 4} y1={Y(b) - 4} x2={X(a) + 4} y2={Y(b) + 4} />
          <line x1={X(a) - 4} y1={Y(b) + 4} x2={X(a) + 4} y2={Y(b) - 4} />
        </g>
      ))}
      <Mat
        x={148}
        y={18}
        A={r.D}
        cw={15}
        ch={9}
        title="d (9, 3)"
        size={5}
        fill={(i, j, v) => (assigned && r.labels[i] === j ? CLS_LIGHT[j] : blueScale((0.5 * v) / dMax))}
        hot={(i, j) => assigned && r.labels[i] === j}
      />
      <Mat
        x={196}
        y={18}
        A={col(r.labels)}
        cw={12}
        ch={9}
        title="labels"
        show={assigned}
        size={5}
        text={(_, __, v) => String(v)}
        fill={(_, __, v) => CLS_LIGHT[v]}
      />
      <Mat x={224} y={18} A={r.sums} cw={20} ch={9} title="sums (3, 2)" show={s.step >= 2 && s.step < 4} size={5} fill={(i) => CLS_LIGHT[i]} />
      <Mat x={224} y={56} A={col(r.counts)} cw={20} ch={9} title="counts (3,)" show={s.step >= 2 && s.step < 4} size={5} text={(_, __, v) => String(v)} fill={(i) => CLS_LIGHT[i]} />
      <Mat
        x={224}
        y={94}
        A={centers}
        cw={20}
        ch={9}
        title="centers (3, 2)"
        d={2}
        size={5}
        fill={(i) => (s.step === 3 ? CLS_LIGHT[i] : "white")}
        titleColor={s.step === 3 ? COL.blue : COL.muted}
      />
      <Txt x={148} y={124} size={7} kind="mono" color={COL.blue}>{`n_iter = ${s.round}`}</Txt>
      <Code y={154}>{code}</Code>
    </>
  );
}

const PICTURES: Record<CpumlCase, { draw: (p: PicProps) => ReactNode; h: number }> = {
  pca: { draw: (p) => <Pca {...p} />, h: 160 },
  linsvm: { draw: (p) => <LinSvm {...p} />, h: 152 },
  ksvm: { draw: (p) => <KSvm {...p} />, h: 150 },
  logreg: { draw: (p) => <LogReg {...p} />, h: 170 },
  knn: { draw: (p) => <Knn {...p} />, h: 160 },
  kmeans: { draw: (p) => <KMeans {...p} />, h: 160 },
};

/**
 * The animation of one "CPU-only ML to tensors" card: the algorithm run as tensor operations on toy data
 * (shapes on every matrix), the list of steps with the current one highlighted, a caption and the controls.
 */
export default function CpumlAnim({ id }: { id: CpumlCase }) {
  const T = usePageText();
  const cycle = totalFrames(id) + HOLD;
  const clock = useAnimClock(cycle, PLAN[id].tick);
  const { root, g } = clock;
  const s = stageAt(id, g);
  const n = stepCount(id);
  const pic = PICTURES[id];
  const caption = T(`cpuml.anim.${id}.caption` as PageKey);
  return (
    <div ref={root} className="mt-4" data-testid={`cpuml-anim-${id}`}>
      <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <figcaption className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">
          <span>{T("cpuml.anim.title")}</span>
          <span className="font-mono tabular-nums text-blue-700" data-testid={`cpuml-anim-step-${id}`}>
            {s.step + 1}/{n}
          </span>
        </figcaption>
        <div className="grid md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <svg viewBox={`0 0 ${W} ${pic.h}`} className="mx-auto block h-auto w-full max-w-[560px] p-1" role="img" aria-label={caption.replace(/[$`\\]/g, "")}>
            {pic.draw({ s })}
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
                  <RichText text={T(`cpuml.anim.${id}.s${k}` as PageKey)} />
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

/** The card's tensor code explained line by line (cpuml.anim.<id>.code0 ..). */
export function CpumlCodeNotes({ id, lines }: { id: CpumlCase; lines: number }) {
  const T = usePageText();
  return (
    <div className="mt-4 rounded-lg border border-gray-200 bg-white px-4 py-3 text-sm leading-relaxed text-gray-700" data-testid={`cpuml-code-notes-${id}`}>
      <h3 className="font-semibold text-gray-900">{T("cpuml.anim.codeTitle")}</h3>
      <ul className="mt-1.5 list-disc space-y-1 pl-5">
        {Array.from({ length: lines }, (_, k) => (
          <li key={k}>
            <RichText text={T(`cpuml.anim.${id}.code${k}` as PageKey)} />
          </li>
        ))}
      </ul>
    </div>
  );
}
