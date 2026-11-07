// Drawings of the Concept tab "The Feynman technique" (static inline SVG). Every visible word comes from a
// feynman.fig.* page key; letters, numbers and units inside the drawings are not translated.
import { RichText } from "./Tex";
import { usePageText, type PageKey } from "../content/pages";

const C = {
  blue: "#2563eb",
  blueSoft: "#dbeafe",
  ink: "#1f2937",
  muted: "#6b7280",
  grid: "#e5e7eb",
  red: "#dc2626",
  redSoft: "#fee2e2",
  amber: "#d97706",
  green: "#059669",
  water: "#bfdbfe",
  ice: "#f0f9ff",
  rubber: "#374151",
};

function Figure({ caption, children, testid }: { caption: string; children: React.ReactNode; testid?: string }) {
  return (
    <figure className="rounded-lg border border-gray-200 bg-gray-50/60 p-2 sm:p-3" data-testid={testid}>
      {children}
      <figcaption className="mt-2 text-center text-xs text-gray-600">
        <RichText text={caption} />
      </figcaption>
    </figure>
  );
}

/** Numbered marker used in drawings and in the legend under them. */
function Mark({ n, x, y, color = C.blue }: { n: number; x: number; y: number; color?: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={8} fill={color} />
      <text x={x} y={y + 3.5} textAnchor="middle" fontSize={10} fontWeight={700} fill="#fff">
        {n}
      </text>
    </g>
  );
}

function Legend({ items }: { items: { n: number; key: PageKey; color?: string }[] }) {
  const T = usePageText();
  return (
    <ul className="mt-2 grid gap-1.5 text-xs text-gray-700 sm:grid-cols-2">
      {items.map((it) => (
        <li key={it.n} className="flex items-start gap-2">
          <span
            className="mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white"
            style={{ background: it.color ?? C.blue }}
          >
            {it.n}
          </span>
          <span>
            <RichText text={T(it.key)} />
          </span>
        </li>
      ))}
    </ul>
  );
}

/** Small pictogram for each of the four steps: a notebook, a talk to a child, a magnifier on a gap, a loop. */
export function StepIcon({ n }: { n: 1 | 2 | 3 | 4 }) {
  const s = { fill: "none", stroke: C.blue, strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" } as const;
  return (
    <svg viewBox="0 0 64 48" className="mx-auto block h-12 w-16" aria-hidden data-testid={`feynman-step-icon-${n}`}>
      {n === 1 && (
        <>
          <rect x={14} y={6} width={30} height={36} rx={3} fill={C.blueSoft} stroke={C.blue} strokeWidth={2} />
          {[14, 20, 26, 32].map((y) => (
            <line key={y} x1={20} y1={y} x2={38} y2={y} stroke={C.blue} strokeWidth={1.5} opacity={0.6} />
          ))}
          <path d="M50 10 L56 16 L40 32 L33 34 L35 27 Z" {...s} fill="#fff" />
        </>
      )}
      {n === 2 && (
        <>
          {/* the explainer (left) talks; the bubble's tail points at his head; a smaller listener on the right */}
          <circle cx={14} cy={24} r={6} {...s} />
          <path d="M4 46 Q14 30 24 46" {...s} />
          <circle cx={51} cy={30} r={4.5} {...s} />
          <path d="M44 46 Q51 36 58 46" {...s} />
          <path d="M28 3 H48 a4 4 0 0 1 4 4 V13 a4 4 0 0 1 -4 4 H33 L22 22 L28 17 a4 4 0 0 1 -4 -4 V7 a4 4 0 0 1 4 -4 Z" fill={C.blueSoft} stroke={C.blue} strokeWidth={1.5} strokeLinejoin="round" />
          {[31, 38, 45].map((x) => (
            <circle key={x} cx={x} cy={10} r={1.4} fill={C.blue} />
          ))}
        </>
      )}
      {n === 3 && (
        <>
          {/* a magnifier with the question it finds */}
          <circle cx={27} cy={20} r={13} {...s} fill="#fff" />
          <path d="M36.5 29.5 L48 41" {...s} strokeWidth={3.5} />
          <text x={27} y={26} fontSize={17} fontWeight={700} fill={C.red} textAnchor="middle">?</text>
        </>
      )}
      {n === 4 && (
        <>
          {/* a long text becomes a short one */}
          {[10, 16, 22, 28, 34].map((y, i) => (
            <line key={y} x1={6} y1={y} x2={i === 4 ? 17 : 24} y2={y} stroke={C.muted} strokeWidth={2} strokeLinecap="round" />
          ))}
          <path d="M29 22 H37 M33.5 18.5 L37 22 L33.5 25.5" {...s} />
          <path d="M43 18 H57 M43 26 H52" stroke={C.green} strokeWidth={2.5} strokeLinecap="round" />
        </>
      )}
    </svg>
  );
}

/** Reading looks at the bridge from the bank; explaining walks it, and the missing plank shows. */
export function BridgeFigure() {
  const T = usePageText();
  const planks = Array.from({ length: 10 }, (_, i) => 70 + i * 18);
  return (
    <Figure caption={T("feynman.fig.bridge.caption")} testid="feynman-fig-bridge">
      <svg viewBox="0 0 320 120" className="mx-auto w-full max-w-[460px]" role="img" aria-label={T("feynman.fig.bridge.caption")}>
        <path d="M0 70 H62 L62 120 H0 Z" fill="#d6d3d1" />
        <path d="M258 70 H320 V120 H258 Z" fill="#d6d3d1" />
        <path d="M62 120 Q160 92 258 120 Z" fill={C.water} />
        <line x1={62} y1={66} x2={258} y2={66} stroke={C.ink} strokeWidth={1.5} />
        <line x1={62} y1={44} x2={258} y2={44} stroke={C.muted} strokeWidth={1} />
        {planks.map((x, i) =>
          i === 6 ? (
            <rect key={x} x={x} y={62} width={14} height={7} fill={C.redSoft} stroke={C.red} strokeDasharray="2 2" />
          ) : (
            <rect key={x} x={x} y={62} width={14} height={7} fill="#a16207" />
          ),
        )}
        {planks.map((x) => (
          <line key={x} x1={x + 7} y1={44} x2={x + 7} y2={62} stroke={C.muted} strokeWidth={0.6} />
        ))}
        {/* the reader on the bank: a person standing and looking across */}
        <g stroke={C.ink} strokeWidth={1.8} strokeLinecap="round" fill="none">
          <circle cx={30} cy={38} r={5} fill="#fff" />
          <line x1={30} y1={43} x2={30} y2={56} />
          <line x1={30} y1={47} x2={23} y2={53} />
          <line x1={30} y1={47} x2={37} y2={53} />
          <line x1={30} y1={56} x2={25} y2={69} />
          <line x1={30} y1={56} x2={35} y2={69} />
        </g>
        <circle cx={32} cy={37} r={0.9} fill={C.ink} />
        <path d="M38 37 L66 42" stroke={C.muted} strokeDasharray="3 3" />
        <Mark n={1} x={30} y={20} />
        {/* the explainer walking: a stick figure one plank before the gap */}
        <g stroke={C.ink} strokeWidth={1.8} strokeLinecap="round" fill="none">
          <circle cx={174} cy={30} r={5} fill="#fff" />
          <line x1={174} y1={35} x2={174} y2={48} />
          <line x1={174} y1={40} x2={166} y2={45} />
          <line x1={174} y1={40} x2={182} y2={44} />
          <line x1={174} y1={48} x2={168} y2={61} />
          <line x1={174} y1={48} x2={181} y2={61} />
        </g>
        <text x={185} y={94} fontSize={18} fontWeight={700} fill={C.red} textAnchor="middle">!</text>
        <Mark n={2} x={190} y={22} color={C.red} />
      </svg>
      <Legend
        items={[
          { n: 1, key: "feynman.fig.bridge.bank" },
          { n: 2, key: "feynman.fig.bridge.walk", color: C.red },
        ]}
      />
    </Figure>
  );
}

/** Why the sky is blue: short wavelengths are scattered far more (Rayleigh, 1/λ⁴), with the curve on fixed axes. */
export function SkyFigure() {
  const T = usePageText();
  // plot frame: λ from 400 to 700 nm, relative scattering (700/λ)^4 from 0 to 10 (a.u., 700 nm = 1)
  const X0 = 46, X1 = 230, Y0 = 130, Y1 = 14;
  const px = (l: number) => X0 + ((l - 400) / 300) * (X1 - X0);
  const py = (v: number) => Y0 - (v / 10) * (Y0 - Y1);
  const curve = Array.from({ length: 61 }, (_, i) => 400 + i * 5)
    .map((l, i) => `${i ? "L" : "M"}${px(l).toFixed(1)} ${py((700 / l) ** 4).toFixed(1)}`)
    .join(" ");
  // a wavy ray from (x, y) at angle ang, starting r0 away from the centre so the molecule stays visible
  const wave = (x: number, y: number, ang: number, len: number, k: number, color: string, r0 = 0) => {
    const pts = Array.from({ length: 25 }, (_, i) => {
      const t = r0 + (i / 24) * (len - r0);
      return `${i ? "L" : "M"}${t.toFixed(1)} ${(3.2 * Math.sin((t / len) * k * 2 * Math.PI)).toFixed(1)}`;
    }).join(" ");
    return <path d={pts} transform={`translate(${x} ${y}) rotate(${ang})`} fill="none" stroke={color} strokeWidth={1.8} />;
  };
  return (
    <Figure caption={T("feynman.fig.sky.caption")} testid="feynman-fig-sky">
      <div className="grid items-center gap-3 sm:grid-cols-2">
        <div>
          <svg viewBox="0 0 240 150" className="mx-auto w-full max-w-[340px]" role="img" aria-label={T("feynman.fig.sky.caption")}>
            <circle cx={26} cy={75} r={16} fill="#fde047" stroke={C.amber} strokeWidth={1.5} />
            {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
              <line key={a} x1={26 + 20 * Math.cos((a * Math.PI) / 180)} y1={75 + 20 * Math.sin((a * Math.PI) / 180)} x2={26 + 25 * Math.cos((a * Math.PI) / 180)} y2={75 + 25 * Math.sin((a * Math.PI) / 180)} stroke={C.amber} strokeWidth={1.5} />
            ))}
            <line x1={46} y1={75} x2={112} y2={75} stroke="#facc15" strokeWidth={5} opacity={0.7} />
            {/* an air molecule */}
            <circle cx={120} cy={75} r={5} fill={C.muted} />
            <circle cx={127} cy={71} r={4} fill={C.muted} />
            {/* red: mostly straight on */}
            {wave(134, 76, 0, 96, 3, C.red)}
            {/* blue: thrown in every direction */}
            {[-150, -105, -60, -25, 30, 70, 115, 160].map((a) => (
              <g key={a}>{wave(123, 73, a, 50, 5, C.blue, 12)}</g>
            ))}
            <Mark n={1} x={26} y={36} color={C.amber} />
            {/* marker 2 sits below the molecule, its leader running down the gap between two blue rays */}
            <path d="M121 124 L121 82" stroke={C.muted} strokeWidth={1} />
            <Mark n={2} x={121} y={132} color={C.muted} />
            <Mark n={3} x={200} y={60} color={C.red} />
            <Mark n={4} x={170} y={28} />
          </svg>
          <Legend
            items={[
              { n: 1, key: "feynman.fig.sky.sun", color: C.amber },
              { n: 2, key: "feynman.fig.sky.molecule", color: C.muted },
              { n: 3, key: "feynman.fig.sky.red", color: C.red },
              { n: 4, key: "feynman.fig.sky.blue" },
            ]}
          />
        </div>
        <svg viewBox="0 0 250 168" className="mx-auto w-full max-w-[340px]" role="img" aria-label={T("feynman.fig.sky.ylabel")}>
          {[0, 2, 4, 6, 8, 10].map((v) => (
            <g key={v}>
              <line x1={X0} x2={X1} y1={py(v)} y2={py(v)} stroke={C.grid} />
              <text x={X0 - 5} y={py(v) + 3.5} fontSize={10} textAnchor="end" fill={C.muted}>{v}</text>
            </g>
          ))}
          {[400, 500, 600, 700].map((l) => (
            <text key={l} x={px(l)} y={Y0 + 13} fontSize={10} textAnchor="middle" fill={C.muted}>{l}</text>
          ))}
          <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke={C.ink} />
          <line x1={X0} y1={Y0} x2={X0} y2={Y1} stroke={C.ink} />
          <path d={curve} fill="none" stroke={C.ink} strokeWidth={1.8} />
          <circle cx={px(450)} cy={py((700 / 450) ** 4)} r={4.5} fill={C.blue} />
          <text x={px(450) + 8} y={py((700 / 450) ** 4) - 4} fontSize={10.5} fill={C.blue}>450 nm: ×5.9</text>
          <circle cx={px(700)} cy={py(1)} r={4.5} fill={C.red} />
          <line x1={px(700)} y1={py(1) - 5} x2={px(700)} y2={py(3.4)} stroke={C.red} strokeWidth={1} />
          <text x={px(700) - 5} y={py(3.4) + 4} fontSize={10.5} textAnchor="end" fill={C.red}>700 nm: ×1</text>
          <text x={(X0 + X1) / 2} y={Y0 + 30} fontSize={11} textAnchor="middle" fill={C.ink}>{T("feynman.fig.sky.xlabel")}</text>
          <text x={12} y={(Y0 + Y1) / 2} fontSize={11} textAnchor="middle" fill={C.ink} transform={`rotate(-90 12 ${(Y0 + Y1) / 2})`}>
            {T("feynman.fig.sky.ylabel")}
          </text>
        </svg>
      </div>
    </Figure>
  );
}

/** A Feynman diagram drawn for this page: two electrons exchange one photon (time runs upwards). */
export function DiagramFigure() {
  const T = usePageText();
  const photon = Array.from({ length: 41 }, (_, i) => {
    const x = 80 + (i / 40) * 80;
    return `${i ? "L" : "M"}${x.toFixed(1)} ${(80 + 6 * Math.sin((i / 40) * 7 * 2 * Math.PI)).toFixed(1)}`;
  }).join(" ");
  const arrowHead = (x: number, y: number, ang: number) => (
    <path d="M-5 -4 L4 0 L-5 4 Z" fill={C.ink} transform={`translate(${x} ${y}) rotate(${ang})`} />
  );
  return (
    <Figure caption={T("feynman.fig.diagram.caption")} testid="feynman-fig-diagram">
      <svg viewBox="0 0 260 150" className="mx-auto w-full max-w-[320px]" role="img" aria-label={T("feynman.fig.diagram.caption")}>
        <defs>
          <marker id="fy-ax" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0 0 L10 5 L0 10 Z" fill={C.muted} />
          </marker>
        </defs>
        <line x1={24} y1={138} x2={24} y2={18} stroke={C.muted} markerEnd="url(#fy-ax)" />
        <line x1={24} y1={138} x2={130} y2={138} stroke={C.muted} markerEnd="url(#fy-ax)" />
        <text x={16} y={78} fontSize={11} fill={C.muted} textAnchor="middle" transform="rotate(-90 16 78)">{T("feynman.fig.diagram.time")}</text>
        <text x={136} y={142} fontSize={11} fill={C.muted}>{T("feynman.fig.diagram.space")}</text>
        <path d="M60 128 L80 80 L60 22" fill="none" stroke={C.ink} strokeWidth={2} />
        <path d="M200 128 L160 80 L200 22" fill="none" stroke={C.ink} strokeWidth={2} />
        {arrowHead(70, 104, -67)}
        {arrowHead(70, 51, -109)}
        {arrowHead(180, 104, -130)}
        {arrowHead(180, 51, -50)}
        <path d={photon} fill="none" stroke={C.blue} strokeWidth={2} />
        <circle cx={80} cy={80} r={3.5} fill={C.ink} />
        <circle cx={160} cy={80} r={3.5} fill={C.ink} />
        <text x={50} y={132} fontSize={12} fill={C.ink}>e⁻</text>
        <text x={204} y={132} fontSize={12} fill={C.ink}>e⁻</text>
        <text x={50} y={24} fontSize={12} fill={C.ink}>e⁻</text>
        <text x={204} y={24} fontSize={12} fill={C.ink}>e⁻</text>
        <text x={120} y={68} fontSize={13} fill={C.blue} textAnchor="middle">γ</text>
      </svg>
    </Figure>
  );
}

/** Feynman's demonstration: the same squeezed rubber springs back when warm and stays flat in ice water. */
export function ORingFigure() {
  const T = usePageText();
  const panel = (cold: boolean) => (
    <svg viewBox="0 0 150 150" className="mx-auto w-full max-w-[190px]" role="img" aria-label={T(cold ? "feynman.fig.oring.cold" : "feynman.fig.oring.warm")}>
      {/* glass with water (and ice) */}
      <path d="M30 40 L38 140 H112 L120 40" fill="none" stroke={C.muted} strokeWidth={2} />
      <path d="M33 66 L38 138 H112 L117 66 Z" fill={cold ? C.water : "#e0f2fe"} opacity={0.8} />
      {cold &&
        [
          [44, 70],
          [80, 76],
          [60, 92],
        ].map(([x, y]) => <rect key={x} x={x} y={y} width={18} height={15} rx={3} fill={C.ice} stroke="#93c5fd" transform={`rotate(${x % 20 - 10} ${x + 9} ${y + 7})`} />)}
      {/* the rubber cross-section after the clamp is taken off: round again, or still squashed */}
      {cold ? (
        <ellipse cx={75} cy={118} rx={22} ry={8} fill={C.rubber} />
      ) : (
        <circle cx={75} cy={114} r={14} fill={C.rubber} />
      )}
      {/* thermometer */}
      <rect x={128} y={34} width={8} height={90} rx={4} fill="#fff" stroke={C.muted} />
      <circle cx={132} cy={128} r={7} fill={cold ? C.blue : C.red} />
      <rect x={130} y={cold ? 104 : 70} width={4} height={cold ? 22 : 56} fill={cold ? C.blue : C.red} />
      <text x={132} y={26} fontSize={11} textAnchor="middle" fill={C.ink}>{cold ? "0 °C" : "20 °C"}</text>
    </svg>
  );
  return (
    <Figure caption={T("feynman.fig.oring.caption")} testid="feynman-fig-oring">
      <div className="grid grid-cols-2 gap-2">
        {[false, true].map((cold) => (
          <div key={String(cold)} className="text-center">
            {panel(cold)}
            <p className="mt-1 text-xs text-gray-700">
              <RichText text={T(cold ? "feynman.fig.oring.cold" : "feynman.fig.oring.warm")} />
            </p>
          </div>
        ))}
      </div>
    </Figure>
  );
}

/** The repeated index: Q (sides i, j) and p (side j) are zipped along j; only i is left on the result. */
export function ZipFigure() {
  const T = usePageText();
  const Q = [
    [2, 1],
    [0, 3],
  ];
  const p = [3, 5];
  const cell = (x: number, y: number, v: number | string, fill: string, key?: string) => (
    <g key={key}>
      <rect x={x} y={y} width={30} height={30} fill={fill} stroke={C.blue} />
      <text x={x + 15} y={y + 20} fontSize={14} textAnchor="middle" fill={C.ink}>{v}</text>
    </g>
  );
  return (
    <Figure caption={T("feynman.fig.zip.caption")} testid="feynman-fig-zip">
      <svg viewBox="0 0 330 150" className="mx-auto w-full max-w-[460px]" role="img" aria-label={T("feynman.fig.zip.caption")}>
        {/* Q: rows are customers (i), columns are products (j) */}
        {Q.map((row, i) => row.map((v, j) => cell(30 + j * 30, 40 + i * 30, v, C.blueSoft, `q${i}${j}`)))}
        <text x={60} y={30} fontSize={13} textAnchor="middle" fill={C.amber} fontWeight={700}>j</text>
        <text x={18} y={74} fontSize={13} textAnchor="middle" fill={C.green} fontWeight={700}>i</text>
        <text x={60} y={124} fontSize={13} textAnchor="middle" fill={C.ink} fontStyle="italic">Q</text>
        {/* p: one price per product (j), a column vector: Q's row meets p's column along j */}
        {p.map((v, j) => cell(142, 40 + j * 30, v, "#fef3c7", `p${j}`))}
        <text x={132} y={74} fontSize={13} textAnchor="middle" fill={C.amber} fontWeight={700}>j</text>
        <text x={157} y={124} fontSize={13} textAnchor="middle" fill={C.ink} fontStyle="italic">p</text>
        {/* zip: the j sides meet and disappear */}
        <text x={108} y={75} fontSize={14} textAnchor="middle" fill={C.amber}>×</text>
        {/* the same index j: Q's columns and p's rows */}
        <path d="M69 26 Q122 14 131 60" fill="none" stroke={C.amber} strokeWidth={1.5} markerEnd="url(#fz-arrow-j)" />
        <defs>
          <marker id="fz-arrow-j" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0 L10 5 L0 10 Z" fill={C.amber} />
          </marker>
        </defs>
        <path d="M182 70 H234" stroke={C.ink} strokeWidth={1.5} markerEnd="url(#fz-arrow)" />
        <text x={208} y={62} fontSize={13} textAnchor="middle" fill={C.amber}>Σⱼ</text>
        <defs>
          <marker id="fz-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0 0 L10 5 L0 10 Z" fill={C.ink} />
          </marker>
        </defs>
        {/* result: one total per customer, only i is left */}
        {[11, 15].map((v, i) => cell(250, 40 + i * 30, v, "#d1fae5", `r${i}`))}
        <text x={300} y={74} fontSize={13} textAnchor="middle" fill={C.green} fontWeight={700}>i</text>
        <text x={265} y={124} fontSize={12} textAnchor="middle" fill={C.muted}>2·3 + 1·5</text>
        <text x={265} y={141} fontSize={12} textAnchor="middle" fill={C.muted}>0·3 + 3·5</text>
      </svg>
      <Legend
        items={[
          { n: 1, key: "feynman.fig.zip.j", color: C.amber },
          { n: 2, key: "feynman.fig.zip.i", color: C.green },
        ]}
      />
    </Figure>
  );
}
