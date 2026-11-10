import { useState, type ReactNode } from "react";
import { usePageText, type PageKey } from "../content/pages";
import { jointProbabilities, LIMITATION_EQUATIONS } from "../limitationsModel";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import CodeBlock from "../components/CodeBlock";
import LimitationsFigure from "../components/LimitationsFigure";
import { Feynman, Links } from "./MathsConcept";

// Sections in order; the address #/concept/limitations/<key> scrolls to id limits-<key>.
export const LIMITATION_SECTIONS = ["scope", "spinors", "gravity", "connections", "topology", "entanglement", "frontiers", "recap", "bridge"] as const;
type Section = (typeof LIMITATION_SECTIONS)[number];
type EqSection = keyof typeof LIMITATION_EQUATIONS;

// The PyTorch line(s) under each equation (not translated); the same order as LIMITATION_EQUATIONS[section].
const CODE: Partial<Record<EqSection, (string | null)[]>> = {
  scope: [`arrow = torch.tensor([1., 0.])
theta = torch.tensor(torch.pi / 2)                    # turn the measuring axes by 90 degrees
R = torch.stack([torch.stack([theta.cos(), theta.sin()]),
                 torch.stack([-theta.sin(), theta.cos()])])
R @ arrow                                             # tensor([0., -1.]): same arrow, new numbers`],
  spinors: [`sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
U = lambda t: torch.linalg.matrix_exp(-0.5j * t * sz)  # spin-1/2 rotation about z
U(2 * torch.pi)   # [[-1, 0], [0, -1]]: one full turn gives -1
U(4 * torch.pi)   # the identity: two full turns`,
  `g0 = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1],
                   [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.complex64)  # gamma^0, Weyl basis
psi = torch.randn(4, dtype=torch.complex64)   # a Dirac spinor: 4 complex numbers
J0 = psi.conj() @ g0 @ g0 @ psi               # psi-bar gamma^0 psi = |psi|^2, the charge density`],
  gravity: [`# A flat plane. Cartesian coordinates: every Gamma is 0.
# Polar coordinates (r, phi) of the same plane: some are not.
r = torch.tensor(2.0)
Gamma_r_phiphi = -r        # Gamma^r_{phi phi}
Gamma_phi_rphi = 1 / r     # Gamma^phi_{r phi}
# Zero in one system, non-zero in another: a tensor could never do that.`,
  `GM, r = 3.986e14, 6.371e6        # Earth: m^3/s^2, m
k = GM / r**3                     # 1.54e-6 s^-2: the tidal gradient
xi, t = 1.0, 2.0                  # side apple 1 m from the centre, 2 s of fall
squeeze = 0.5 * k * xi * t**2     # 3.1e-6 m towards the centre, in every frame`],
  connections: [`n = 64; d = 2 / (n - 1)
x, y = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
F = lambda Ax, Ay: torch.gradient(Ay, spacing=d, dim=0)[0] - torch.gradient(Ax, spacing=d, dim=1)[0]
Ax, Ay = -y / 2, x / 2                      # uniform field B = 1
ga, gb = torch.gradient(torch.sin(3 * x) * y, spacing=d)   # gradient of a gauge function
F(Ax, Ay), F(Ax + ga, Ay + gb)              # the same field (up to grid error)`,
  `h, e = 6.62607015e-34, 1.602176634e-19
Phi = 0.25 * h / e                          # the enclosed flux in the animation
dphi = -e * Phi / (h / (2 * torch.pi))      # q = -e: -pi/2, i.e. -90 degrees per loop`],
  topology: [`sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

def chern(m, n=40):     # a two-band model; lattice (Fukui-Hatsugai) Chern number
    k = torch.linspace(0, 2 * torch.pi, n + 1, dtype=torch.float64)[:-1]
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    d = torch.stack([kx.sin(), ky.sin(), m + kx.cos() + ky.cos()], -1).to(sx.dtype)
    u = torch.linalg.eigh(torch.einsum("xyi,iab->xyab", d, torch.stack([sx, sy, sz]))).eigenvectors[..., 0]
    link = lambda a, b: (a.conj() * b).sum(-1)
    u1, u2, u12 = u.roll(-1, 0), u.roll(-1, 1), u.roll((-1, -1), (0, 1))
    loop = link(u, u1) * link(u1, u12) * link(u12, u2) * link(u2, u)
    return round(loop.angle().sum().item() / (2 * torch.pi))

chern(1.0), chern(1.5), chern(3.0)   # (1, 1, 0): whole numbers, a jump only when the gap closes`,
  `h, e = 6.62607015e-34, 1.602176634e-19
R_K = h / e**2                         # 25812.807 ohm, the von Klitzing constant
[R_K / nu for nu in (1, 2, 3, 4)]      # the Hall resistance plateaus`],
  entanglement: [`bell = torch.tensor([[1., 0.], [0., 1.]]) / 2**0.5      # amplitudes psi[A, B]
indep = torch.outer(torch.tensor([1., 1.]), torch.tensor([1., 1.])) / 2
torch.linalg.matrix_rank(indep), torch.linalg.matrix_rank(bell)   # 1 (product), 2 (entangled)
bell**2, indep**2                                         # the two tables above`,
  `E = lambda a, b: torch.cos(torch.tensor(2 * (a - b)))   # Bell pair, polariser angles
a, a2, b, b2 = 0.0, torch.pi / 4, torch.pi / 8, 3 * torch.pi / 8
S = E(a, b) - E(a, b2) + E(a2, b) + E(a2, b2)           # 2.83 = 2*sqrt(2) > 2`],
  frontiers: [`x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)     # coordinates as matrices
y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
x @ y - y @ x                                                   # 2i diag(1, -1): the order matters`,
  `T = [(1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7), (5, 6, 1), (6, 7, 2), (7, 1, 3)]
def mul(i, j):                          # e_i e_j of two different imaginary octonion units
    for a, b, c in T:
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            if (i, j) == (p, q): return 1, r
            if (i, j) == (q, p): return -1, r
s1, k = mul(1, 2); s2, m = mul(k, 3)    # (e1 e2) e3
s3, k = mul(2, 3); s4, n = mul(1, k)    # e1 (e2 e3)
(s1 * s2, m), (s3 * s4, n)              # (-1, 6) vs (1, 6): grouping changes the answer`],
};

// External references per section, opened in a new tab (titles not translated).
const LINKS: Partial<Record<Section, { label: string; url?: string }[]>> = {
  scope: [
    { label: "Wikipedia: Tensor", url: "https://en.wikipedia.org/wiki/Tensor" },
    { label: "Wikipedia: Covariance and contravariance of vectors", url: "https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors" },
  ],
  spinors: [
    { label: "David Tong: The Dirac Equation", url: "https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S4.html" },
    { label: "Wikipedia: Spin group", url: "https://en.wikipedia.org/wiki/Spin_group" },
    { label: "H. Rauch et al. (1975), Verification of coherent spinor rotation of fermions, Phys. Lett. A 54, 425" },
  ],
  gravity: [
    { label: "David Tong: General Relativity", url: "https://www.damtp.cam.ac.uk/user/tong/gr.html" },
    { label: "Wikipedia: Christoffel symbols", url: "https://en.wikipedia.org/wiki/Christoffel_symbols" },
    { label: "Wikipedia: Geodesic deviation", url: "https://en.wikipedia.org/wiki/Geodesic_deviation" },
  ],
  connections: [
    { label: "David Tong: Gauge Theory", url: "https://www.damtp.cam.ac.uk/user/tong/gaugetheory.html" },
    { label: "Wikipedia: Aharonov–Bohm effect", url: "https://en.wikipedia.org/wiki/Aharonov%E2%80%93Bohm_effect" },
    { label: "Wikipedia: Wilson loop", url: "https://en.wikipedia.org/wiki/Wilson_loop" },
  ],
  topology: [
    { label: "David Tong: The Quantum Hall Effect", url: "https://www.damtp.cam.ac.uk/user/tong/qhe.html" },
    { label: "D. J. Thouless, M. Kohmoto, M. P. Nightingale & M. den Nijs (1982), Quantized Hall conductance in a two-dimensional periodic potential, Phys. Rev. Lett. 49, 405", url: "https://doi.org/10.1103/PhysRevLett.49.405" },
  ],
  entanglement: [
    { label: "Caltech: Quantum and classical information processing with tensors", url: "https://iqim.caltech.edu/wp-content/uploads/2019/08/ACM270_complete_lecture_notes.pdf" },
    { label: "Wikipedia: CHSH inequality", url: "https://en.wikipedia.org/wiki/CHSH_inequality" },
  ],
  frontiers: [
    { label: "Wikipedia: Noncommutative geometry", url: "https://en.wikipedia.org/wiki/Noncommutative_geometry" },
    { label: "R. Blumenhagen & E. Plauschinn (2011), Nonassociative gravity in string theory?, J. Phys. A 44, 015401 (arXiv:1010.1263)", url: "https://arxiv.org/abs/1010.1263" },
    { label: "Wikipedia: Octonion", url: "https://en.wikipedia.org/wiki/Octonion" },
  ],
  bridge: [{ label: "Wikipedia: Rotor (mathematics)", url: "https://en.wikipedia.org/wiki/Rotor_(mathematics)" }],
};

type Text = (key: string) => string;

/** The Concept page's "Limits of tensors" tab: one question per card, laid out like the other concept tabs. */
export default function LimitationsConcept() {
  const T = usePageText();
  const t: Text = (key) => T(`limitations.${key}` as PageKey);
  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
      <PageToc scrollRoot="concept-scroll" items={LIMITATION_SECTIONS.map((s, i) => ({ id: `limits-${s}`, label: t(`${s}.title`), mark: i + 1 }))} />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10" data-testid="limitations-concept">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">{t("title")}</h1>
          <p className="mt-3 text-gray-600"><RichText text={t("intro")} /></p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900"><RichText text={t("callout")} /></p>
        </header>
        {LIMITATION_SECTIONS.map((s, i) => (
          <section key={s} id={`limits-${s}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6" data-story={s}>
            <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{i + 1}</span>
              {t(`${s}.title`)}
            </h2>
            <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
              <SectionBody s={s} t={t} />
              {s in LIMITATION_EQUATIONS && s !== "bridge" && <EqTabs s={s as EqSection} t={t} title={T("maths.label.code")} />}
              {LINKS[s] && <Links items={LINKS[s]!} title={T("maths.label.links")} />}
            </div>
          </section>
        ))}
      </article>
    </div>
  );
}

function SectionBody({ s, t }: { s: Section; t: Text }) {
  const T = usePageText();
  const label = (kind: "plain" | "analogy" | "precise" | "example") => T(`maths.label.${kind}`);
  const block = (kind: "plain" | "analogy" | "precise" | "example", children?: ReactNode) =>
    <Feynman kind={kind} text={t(`${s}.${kind}`)} label={label(kind)}>{children}</Feynman>;
  const p = (key: string) => <p><RichText text={t(`${s}.${key}`)} /></p>;

  switch (s) {
    case "scope": return <>
      {block("plain")}
      <div className="grid gap-3 sm:grid-cols-2">
        {["arrow", "temperature"].map(k => <div key={k} className="rounded-lg border border-gray-200 bg-gray-50/60 p-4">
          <p className="mb-2 font-mono text-xl text-gray-900">[1, 0]</p>
          <h3 className="font-semibold text-gray-900">{t(`scope.${k}Title`)}</h3><p className="mt-1">{t(`scope.${k}`)}</p>
        </div>)}
      </div>
      {block("example", <LimitationsFigure kind="scope" />)}
      {p("insight")}
      {block("precise")}
    </>;
    case "spinors": return <>
      {block("plain")}
      {block("example", <LimitationsFigure kind="spinors" />)}
      {p("stateIntro")}
      <ol className="grid grid-cols-3 gap-3 text-center">
        {[0, 1, 2].map(i => <li key={i} className={`rounded-lg border px-2 py-3 ${i === 1 ? "border-violet-200 bg-violet-50" : "border-gray-200 bg-gray-50"}`}>
          <p className="text-xs text-gray-500">{t(`spinors.turn${i}`)} · {i * 360}°</p>
          <div className="mt-2 text-xl"><Tex tex={i === 1 ? String.raw`-\psi` : String.raw`\psi`} /></div>
        </li>)}
      </ol>
      {p("insight")}
      <SpinTable t={t} />
      {block("precise")}
      <QA q={t("spinors.q")} a={t("spinors.answer")} />
    </>;
    case "gravity": return <>
      {block("plain")}
      {block("analogy")}
      {block("example", <LimitationsFigure kind="gravity" />)}
      {p("insight")}
      {block("precise")}
    </>;
    case "connections": return <>
      {block("plain")}
      {block("example", <LimitationsFigure kind="connections" />)}
      <dl className="grid gap-4 border-y border-gray-100 py-4 sm:grid-cols-2">
        <div><dt className="font-semibold text-gray-900">{t("connections.localTitle")}</dt><dd className="mt-1">{t("connections.local")}</dd></div>
        <div><dt className="font-semibold text-gray-900">{t("connections.routeTitle")}</dt><dd className="mt-1">{t("connections.route")}</dd></div>
      </dl>
      {p("insight")}
      {block("precise")}
    </>;
    case "topology": return <>
      {block("plain")}
      {block("example", <LimitationsFigure kind="topology" />)}
      <div className="flex flex-col gap-3 rounded-lg bg-gray-50 p-4 sm:flex-row sm:gap-6">
        <div className="flex-1"><h3 className="font-semibold text-gray-900">{t("topology.localTitle")}</h3><p className="mt-1">{t("topology.local")}</p></div>
        <span aria-hidden className="hidden self-center text-2xl text-gray-300 sm:block">→</span>
        <div className="flex-1"><h3 className="font-semibold text-gray-900">{t("topology.globalTitle")}</h3><p className="mt-1">{t("topology.global")}</p></div>
      </div>
      {p("insight")}
      {block("precise")}
    </>;
    case "entanglement": return <>
      {block("plain")}
      {block("example", <PairTable t={t} />)}
      {p("insight")}
      <QA q={t("entanglement.q")} a={t("entanglement.answer")} />
      {block("precise")}
    </>;
    case "frontiers": return <>
      {block("plain")}
      <ol className="grid gap-3 md:grid-cols-3">
        {[1, 2, 3].map(i => <li key={i} className="rounded-lg border border-gray-200 bg-gray-50/60 p-4">
          <h3 className="mb-2 font-semibold text-gray-900">{t(`frontiers.s${i}t`)}</h3><p>{t(`frontiers.s${i}b`)}</p>
        </li>)}
      </ol>
      {p("insight")}
      {block("precise")}
    </>;
    case "recap": return <Recap t={t} />;
    case "bridge": return <>
      {block("plain")}
      {block("example", <LimitationsFigure kind="bridge" />)}
      {p("explanation")}
      <div className="py-1 text-lg"><Tex tex={LIMITATION_EQUATIONS.bridge[0]} display /></div>
      {block("precise")}
      <div className="py-1"><Tex tex={LIMITATION_EQUATIONS.bridge[1]} display /></div>
      {p("closing")}
      <a className="inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700" href="https://jackyko1991.github.io/clifford/" target="_blank" rel="noreferrer">{t("bridge.link")} →</a>
    </>;
  }
}

/** A question the reader is likely to ask, answered in place. */
function QA({ q, a }: { q: string; a: string }) {
  return <p className="rounded-lg bg-gray-50 px-4 py-3"><strong className="text-gray-900">{q}</strong> <RichText text={a} /></p>;
}

/** Rows are "object|after 360°|examples" in each language; the spin column is not translated. */
function SpinTable({ t }: { t: Text }) {
  const spins = ["0", "½", "1", "2"];
  return <div className="space-y-2">
    <div className="overflow-x-auto rounded-lg border border-gray-200" data-testid="spin-table">
      <table className="w-full text-left text-xs leading-relaxed">
        <thead className="bg-gray-50 text-gray-500"><tr>
          {["colSpin", "colObject", "colTurn", "colExample"].map(c => <th key={c} scope="col" className="px-3 py-2 font-medium">{t(`spinors.${c}`)}</th>)}
        </tr></thead>
        <tbody className="divide-y divide-gray-100">{spins.map((spin, i) => {
          const [object, turn, example] = t(`spinors.row${i}`).split("|");
          return <tr key={spin} className={i === 1 ? "bg-violet-50/60" : undefined}>
            <th scope="row" className="px-3 py-2 font-medium text-gray-900">{spin}</th>
            <td className="px-3 py-2">{object}</td><td className="px-3 py-2">{turn}</td><td className="px-3 py-2 text-gray-600">{example}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <p className="text-xs text-gray-600">{t("spinors.tableNote")}</p>
  </div>;
}

function PairTable({ t }: { t: Text }) {
  const [paired, setPaired] = useState(false);
  const probabilities = jointProbabilities(paired);
  return <div className="space-y-4 rounded-lg border border-gray-200 bg-white p-3">
    <div className="flex flex-wrap gap-2">{[false, true].map(value => <button key={String(value)} type="button" aria-pressed={paired === value} onClick={() => setPaired(value)}
      className={`rounded-full border px-3 py-1.5 text-xs ${paired === value ? "border-blue-600 bg-blue-600 text-white" : "border-gray-200 bg-white text-gray-700 hover:border-blue-300"}`}>{t(value ? "entanglement.paired" : "entanglement.independent")}</button>)}</div>
    <table className="mx-auto w-full max-w-sm border-separate border-spacing-2 text-center">
      <caption className="pb-2 text-xs text-gray-600">{t("entanglement.caption")}</caption>
      <thead><tr><td /><th scope="col" className="text-xs font-medium">B = 0</th><th scope="col" className="text-xs font-medium">B = 1</th></tr></thead>
      <tbody>{probabilities.map((row, i) => <tr key={i}><th scope="row" className="text-xs font-medium">A = {i}</th>{row.map((v, j) => <td key={j} className={`rounded-lg border p-3 text-xl tabular-nums ${v ? "border-blue-200 bg-blue-50 text-blue-900" : "border-gray-100 bg-gray-50 text-gray-400"}`}>{v * 100}%</td>)}</tr>)}</tbody>
    </table>
    <p className="text-center text-xs text-gray-600">{t("entanglement.marginal")}</p>
  </div>;
}

/** Equation tabs in the style of the Tensor maths tab: each tab is an equation with the PyTorch that computes it. */
function EqTabs({ s, t, title }: { s: EqSection; t: Text; title: string }) {
  const [selected, setSelected] = useState(0);
  const items = LIMITATION_EQUATIONS[s].map((tex, i) => ({ tex, label: t(`${s}.eq${i + 1}`), code: CODE[s]?.[i] ?? null }));
  const item = items[selected];
  return <div data-testid={`limitations-eq-${s}`}>
    <p className="mb-2 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">{title}</p>
    <div className="flex flex-wrap gap-1 border-b border-gray-200" role="tablist" aria-label={title}>
      {items.map((e, i) => <button key={i} type="button" role="tab" id={`limits-${s}-eq-${i}`} aria-selected={selected === i} aria-controls={`limits-${s}-eq-panel`}
        onClick={() => setSelected(i)}
        className={`min-w-0 max-w-full rounded-t-md border-b-2 px-3 py-2 text-left text-xs transition-colors sm:text-sm ${selected === i ? "border-blue-600 bg-blue-50 text-blue-800" : "border-transparent text-gray-600 hover:bg-gray-50 hover:text-blue-700"}`}>
        {e.label}
      </button>)}
    </div>
    <div id={`limits-${s}-eq-panel`} role="tabpanel" aria-labelledby={`limits-${s}-eq-${selected}`} className="overflow-hidden rounded-b-lg border border-t-0 border-gray-200 bg-white">
      <div className="overflow-x-auto bg-gray-50 px-3 py-2"><Tex tex={item.tex} display /></div>
      {item.code && <div className="p-2"><CodeBlock code={item.code} /></div>}
    </div>
  </div>;
}

function Recap({ t }: { t: Text }) {
  return <div className="space-y-4" data-testid="limitations-recap">
    <p><RichText text={t("recap.plain")} /></p>
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="w-full text-left text-xs leading-relaxed">
        <thead className="bg-gray-50 text-gray-500"><tr>
          {["colMissing", "colTool", "colWhere"].map(c => <th key={c} scope="col" className="px-3 py-2 font-medium">{t(`recap.${c}`)}</th>)}
        </tr></thead>
        <tbody className="divide-y divide-gray-100">{[1, 2, 3, 4, 5, 6].map(i => {
          const [missing, tool, where] = t(`recap.row${i}`).split("|");
          return <tr key={i} className="align-top">
            <th scope="row" className="px-3 py-2 font-medium text-gray-900">{missing}</th>
            <td className="px-3 py-2"><RichText text={tool} /></td>
            <td className="px-3 py-2 text-gray-600">{where}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <div className="rounded-lg border-l-4 border-amber-300 bg-amber-50 px-4 py-3 text-amber-950">
      <h3 className="mb-2 font-semibold">{t("recap.misTitle")}</h3>
      <ul className="list-disc space-y-2 pl-5">{[1, 2, 3, 4, 5, 6].map(i => <li key={i}><RichText text={t(`recap.mis${i}`)} /></li>)}</ul>
    </div>
    <p className="rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-blue-900">{t("recap.summary")}</p>
  </div>;
}
