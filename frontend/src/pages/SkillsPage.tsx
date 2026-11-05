import { useEffect } from "react";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { useLang } from "../i18n";
import { PAGE_TABS, type SkillsTab } from "../pageRoute";
import type { Manifest } from "../types";
import LoopsTab from "./LoopsTab";

// The tab comes from the address (#/skills/<tab>[/<section>], parsed in App); App remembers the last one.
const TABS = PAGE_TABS.skills;
type Tab = SkillsTab;

interface Skill {
  /** key of skills.<id>.title / skills.<id>.text in content/pages.<lang>.json */
  id: string;
  code: string;
  usedIn: string[];
}

/** The PyTorch techniques the problem library is built from, each with a minimal example and where to see it. */
const SKILLS: Skill[] = [
  {
    id: "tdd",
    code: `# test.py: step 0, written before the solver
def test_energy_is_conserved():
    _, xs, vs = solve.solve(m, x0, v0, G, eps=0.0, t_max=6.0, dt=0.002)
    e0, e1 = equation.energy(xs[0], vs[0], m, G), equation.energy(xs[-1], vs[-1], m, G)
    assert abs((e1 - e0) / e0) < 1e-5     # no external force: E is constant

def test_period_matches_kepler():
    assert measured == pytest.approx(2 * math.pi * math.sqrt(a**3 / (G * M)), rel=1e-3)`,
    usedIn: ["mechanics/n_body_gravity", "mechanics/simple_pendulum", "thermodynamics/carnot_cycle"],
  },
  {
    id: "tensors",
    code: `import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
theta0 = torch.deg2rad(torch.linspace(10, 150, 4)).to(device)   # (B,)
y0 = torch.stack([theta0, torch.zeros_like(theta0)], dim=1)      # (B, 2)`,
    usedIn: ["mechanics/simple_pendulum", "mechanics/projectile"],
  },
  {
    id: "broadcasting",
    code: `T = torch.tensor([3000., 6000.])[:, None]        # (A, 1) temperatures
lam = torch.linspace(1e-7, 3e-6, 500)[None, :]    # (1, L) wavelengths
u = planck(lam, T)                                # (A, L) every spectrum at once
r = x[None, :, :] - x[:, None, :]                 # (N, N, 3) all pair separations`,
    usedIn: ["thermodynamics/blackbody_radiation", "mechanics/n_body_gravity", "thermodynamics/kinetic_gas"],
  },
  {
    id: "einsum",
    code: `speed2 = torch.einsum("bj,bj->b", v, v)              # |v|² per particle
y_out  = torch.einsum("ij,rj->ri", M, y_in)         # M applied to every ray
force  = torch.einsum("ijk,bj,k->bi", eps, v, B)    # q v × B for every particle
F_new  = torch.einsum("ma,nb,ab->mn", L, L, F)      # Lorentz boost of F^{μν}`,
    usedIn: ["mechanics/projectile", "optics/lens_system", "em/lorentz_force", "em/field_tensor_boost"],
  },
  {
    id: "batching",
    code: `angles = torch.deg2rad(torch.linspace(15, 75, 6))           # (B,)
v0 = 30 * torch.stack([angles.cos(), angles.sin()], dim=-1)   # (B, 2)
# one RK4 step now advances all B projectiles`,
    usedIn: ["mechanics/projectile", "nuclear/radioactive_decay", "biophysics/receptor_binding"],
  },
  {
    id: "rk4",
    code: `def rk4_step(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)`,
    usedIn: ["mechanics/double_pendulum", "em/lorentz_force", "astro/kepler_orbits"],
  },
  {
    id: "linalg",
    code: `phi = torch.linalg.solve(L, -rho)          # Poisson equation on a grid
E, psi = torch.linalg.eigh(H)              # energies and eigenstates
omega = torch.sqrt(torch.linalg.eigvalsh(K / m))   # normal-mode frequencies`,
    usedIn: ["em/electrostatics", "quantum/infinite_well", "mechanics/coupled_oscillators", "nuclear/shell_model"],
  },
  {
    id: "stencils",
    code: `kernel = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]])[None, None] / h**2
lap = torch.nn.functional.conv2d(u[None, None], kernel, padding=1)[0, 0]
u_next = 2 * u - u_prev + (c * dt) ** 2 * lap      # wave equation, leapfrog`,
    usedIn: ["thermodynamics/heat_diffusion", "optics/double_slit", "waves/ripple_tank", "thermodynamics/ising_model"],
  },
  {
    id: "fft",
    code: `k = 2 * torch.pi * torch.fft.fftfreq(n, d=dx)
psi_k = torch.fft.fft(psi)
psi = torch.fft.ifft(torch.exp(-0.5j * k**2 * dt) * psi_k)   # kinetic half of split-step`,
    usedIn: ["quantum/tunneling", "waves/kdv_solitons", "waves/dispersive_waves", "optics/fourier_optics"],
  },
  {
    id: "autograd",
    code: `x = torch.tensor([1.0, 0.5, 0.0], requires_grad=True)
phi = -G * M / x.norm()
force = -torch.autograd.grad(phi, x, create_graph=True)[0]
tidal = -torch.autograd.functional.hessian(lambda r: -G * M / r.norm(), x)`,
    usedIn: ["mechanics/lagrangian_autograd", "astro/tidal_tensor", "astro/black_hole_geodesics"],
  },
  {
    id: "complex",
    code: `psi_t = torch.einsum("xn,n->x", phi, c * torch.exp(-1j * E * t))
prob = (psi_t.conj() * psi_t).real
bell = torch.einsum("abcd,cd->ab", CNOT, torch.einsum("ac,cb->ab", H1, psi0))`,
    usedIn: [
      "quantum/harmonic_oscillator",
      "quantum/spin_precession",
      "quantum/two_qubit_entanglement",
      "nuclear/neutrino_oscillations",
    ],
  },
];

export default function SkillsPage({
  manifest,
  tab,
  section,
  onTab,
}: {
  manifest: Manifest;
  tab: Tab;
  /** optional anchor inside the tab: #/skills/techniques/einsum → card skill-einsum, #/skills/loops/matmul → loops-matmul */
  section: string | null;
  onTab: (t: Tab) => void;
}) {
  const T = usePageText();
  const { lang } = useLang();
  const title = (id: string) => {
    const p = manifest.problems.find((x) => x.id === id);
    return p && (p.i18n?.[lang]?.title ?? p.title);
  };

  const choose = (t: Tab) => {
    if (t === tab) return;
    onTab(t);
  };
  // New tab: start at the top, or at the section the address names.
  useEffect(() => {
    const el = section && document.getElementById(`${tab === "loops" ? "loops" : "skill"}-${section}`);
    if (el) el.scrollIntoView({ block: "start", behavior: "instant" });
    else document.getElementById("skills-scroll")?.scrollTo({ top: 0 });
  }, [tab, section]);

  return (
    <div id="skills-scroll" className="h-full overflow-y-auto scroll-smooth bg-gray-50/60">
      <div className="mx-auto max-w-6xl px-5 pt-6">
        <div
          className="flex gap-1 overflow-x-auto overflow-y-hidden border-b border-gray-200 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          role="tablist"
          data-testid="skills-tabs"
        >
          {TABS.map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => choose(t)}
              data-testid={`skills-tab-${t}`}
              className={`-mb-px shrink-0 whitespace-nowrap border-b-2 px-4 py-2 text-sm font-medium ${
                tab === t ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              {T(t === "techniques" ? "skills.tab.techniques" : "skills.tab.loops")}
            </button>
          ))}
        </div>
      </div>
      {tab === "loops" ? (
        <LoopsTab title={title} />
      ) : (
      <div className="mx-auto flex max-w-6xl gap-8 px-5 pt-6 pb-10">
        <PageToc
          scrollRoot="skills-scroll"
          items={SKILLS.map((s, i) => ({
            id: `skill-${s.id}`,
            mark: i + 1,
            label: T(`skills.${s.id}.title` as PageKey),
          }))}
        />
        <article className="min-w-0 max-w-4xl flex-1 space-y-8">
          <header>
            <p className="text-sm font-medium text-blue-700">{T("skills.kicker")}</p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("skills.title")}</h1>
            <p className="mt-3 text-gray-600">{T("skills.intro")}</p>
          </header>

          {SKILLS.map((s, i) => (
            <section
              key={s.id}
              id={`skill-${s.id}`}
              className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6"
            >
              <h2 className="flex items-baseline gap-3 text-lg font-semibold">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
                  {i + 1}
                </span>
                {T(`skills.${s.id}.title` as PageKey)}
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-gray-700">
                <RichText text={T(`skills.${s.id}.text` as PageKey)} />
              </p>
              <div className="mt-4">
                <CodeBlock code={s.code} />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-gray-500">{T("skills.usedIn")}</span>
                {s.usedIn
                  .filter((id) => title(id))
                  .map((id) => (
                    <a
                      key={id}
                      href={`#/${id}`}
                      className="rounded-full bg-blue-50 px-2.5 py-1 text-blue-700 hover:bg-blue-100"
                    >
                      <RichText text={title(id)!} />
                    </a>
                  ))}
              </div>
            </section>
          ))}
        </article>
      </div>
      )}
    </div>
  );
}
