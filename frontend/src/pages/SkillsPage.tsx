import CodeBlock from "../components/CodeBlock";
import { RichText } from "../components/Tex";
import type { Manifest } from "../types";

interface Skill {
  title: string;
  text: string;
  code: string;
  usedIn: string[];
}

/** The PyTorch techniques the problem library is built from, each with a minimal example and where to see it. */
const SKILLS: Skill[] = [
  {
    title: "Tensors, shapes and devices",
    text: "Every physical quantity is a tensor with a shape you can say out loud: $B$ pendulums × 2 state variables is `(B, 2)`. Keep shapes in comments like units, and create tensors on one `device` so the same code runs on CPU or GPU.",
    code: `import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
theta0 = torch.deg2rad(torch.linspace(10, 150, 4)).to(device)   # (B,)
y0 = torch.stack([theta0, torch.zeros_like(theta0)], dim=1)      # (B, 2)`,
    usedIn: ["mechanics/simple_pendulum", "mechanics/projectile"],
  },
  {
    title: "Broadcasting",
    text: "Combine tensors of different shapes without loops: a column `(A, 1)` against a row `(1, L)` fills an `(A, L)` table. Pairwise separations $r_{ab} = x_b - x_a$ of $N$ bodies are one subtraction.",
    code: `T = torch.tensor([3000., 6000.])[:, None]        # (A, 1) temperatures
lam = torch.linspace(1e-7, 3e-6, 500)[None, :]    # (1, L) wavelengths
u = planck(lam, T)                                # (A, L) every spectrum at once
r = x[None, :, :] - x[:, None, :]                 # (N, N, 3) all pair separations`,
    usedIn: ["thermodynamics/blackbody_radiation", "mechanics/n_body_gravity", "thermodynamics/kinetic_gas"],
  },
  {
    title: "einsum = index notation",
    text: "`torch.einsum` is the Einstein convention typed out: name each index, repeated indices are summed. $v_j v_j$ is `'bj,bj->b'`, a matrix acting on vectors $M_{ij}y_j$ is `'ij,rj->ri'`, a cross product $\\varepsilon_{ijk}v_jB_k$ uses the Levi-Civita tensor.",
    code: `speed2 = torch.einsum("bj,bj->b", v, v)              # |v|² per particle
y_out  = torch.einsum("ij,rj->ri", M, y_in)         # M applied to every ray
force  = torch.einsum("ijk,bj,k->bi", eps, v, B)    # q v × B for every particle
F_new  = torch.einsum("ma,nb,ab->mn", L, L, F)      # Lorentz boost of F^{μν}`,
    usedIn: ["mechanics/projectile", "optics/lens_system", "em/lorentz_force", "em/field_tensor_boost"],
  },
  {
    title: "Batching many simulations",
    text: "Add a batch index instead of a Python loop: launch angles, amplitudes, isotopes or receptor variants become one more axis, and every operation handles them together. On a GPU this is where the speed comes from.",
    code: `angles = torch.deg2rad(torch.linspace(15, 75, 6))           # (B,)
v0 = 30 * torch.stack([angles.cos(), angles.sin()], dim=-1)   # (B, 2)
# one RK4 step now advances all B projectiles`,
    usedIn: ["mechanics/projectile", "nuclear/radioactive_decay", "others/receptor_binding"],
  },
  {
    title: "Time stepping (RK4)",
    text: "Write the law as $\\dot y = f(y, t)$ and march forward. Classical Runge–Kutta samples $f$ four times per step with error $\\propto \\Delta t^4$; because $y$ is a tensor, one step updates the whole system.",
    code: `def rk4_step(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)`,
    usedIn: ["mechanics/double_pendulum", "em/lorentz_force", "astro/kepler_orbits"],
  },
  {
    title: "Linear algebra: solve and eigh",
    text: "Many problems are a matrix equation once discretised: $L_{ij}\\phi_j = -\\rho_i$ is `torch.linalg.solve`, and normal modes or quantum levels $H_{ij}\\psi_j = E\\,\\psi_i$ come from `torch.linalg.eigh`.",
    code: `phi = torch.linalg.solve(L, -rho)          # Poisson equation on a grid
E, psi = torch.linalg.eigh(H)              # energies and eigenstates
omega = torch.sqrt(torch.linalg.eigvalsh(K / m))   # normal-mode frequencies`,
    usedIn: ["em/electrostatics", "quantum/infinite_well", "mechanics/coupled_oscillators", "nuclear/shell_model"],
  },
  {
    title: "Finite differences, stencils and conv2d",
    text: "A derivative on a grid is a small stencil; the Laplacian $\\partial_i\\partial_i u$ is the kernel $[[0,1,0],[1,-4,1],[0,1,0]]/h^2$. `conv2d` applies it everywhere at once, and the time step must respect the CFL condition.",
    code: `kernel = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]])[None, None] / h**2
lap = torch.nn.functional.conv2d(u[None, None], kernel, padding=1)[0, 0]
u_next = 2 * u - u_prev + (c * dt) ** 2 * lap      # wave equation, leapfrog`,
    usedIn: ["thermodynamics/heat_diffusion", "optics/double_slit", "waves/ripple_tank", "thermodynamics/ising_model"],
  },
  {
    title: "Spectral methods with FFT",
    text: "In Fourier space a derivative is a multiplication by $ik$. Free evolution, dispersion and the kinetic step of the Schrödinger equation become element-wise products between two FFTs.",
    code: `k = 2 * torch.pi * torch.fft.fftfreq(n, d=dx)
psi_k = torch.fft.fft(psi)
psi = torch.fft.ifft(torch.exp(-0.5j * k**2 * dt) * psi_k)   # kinetic half of split-step`,
    usedIn: ["quantum/tunneling", "waves/kdv_solitons", "waves/dispersive_waves", "optics/fourier_optics"],
  },
  {
    title: "Autograd for physics",
    text: "PyTorch can differentiate any code: forces from a potential, the mass matrix of a Lagrangian, a tidal tensor $-\\partial_i\\partial_j\\Phi$, or Christoffel symbols from a metric $g_{\\mu\\nu}$ — no derivatives by hand.",
    code: `x = torch.tensor([1.0, 0.5, 0.0], requires_grad=True)
phi = -G * M / x.norm()
force = -torch.autograd.grad(phi, x, create_graph=True)[0]
tidal = -torch.autograd.functional.hessian(lambda r: -G * M / r.norm(), x)`,
    usedIn: ["mechanics/lagrangian_autograd", "astro/tidal_tensor", "astro/black_hole_geodesics"],
  },
  {
    title: "Complex tensors and quantum states",
    text: "Wavefunctions and fields are complex: use `torch.complex128`, `.conj()` and phases $e^{-iE_nt}$. Multi-qubit states are tensors $\\psi_{ab}$, and gates act on them with einsum.",
    code: `psi_t = torch.einsum("xn,n->x", phi, c * torch.exp(-1j * E * t))
prob = (psi_t.conj() * psi_t).real
bell = torch.einsum("abcd,cd->ab", CNOT, torch.einsum("ac,cb->ab", H1, psi0))`,
    usedIn: ["quantum/harmonic_oscillator", "quantum/spin_precession", "quantum/two_qubit_entanglement", "nuclear/neutrino_oscillations"],
  },
];

export default function SkillsPage({ manifest }: { manifest: Manifest }) {
  const title = (id: string) => manifest.problems.find((p) => p.id === id)?.title;

  return (
    <div className="h-full overflow-y-auto bg-gray-50/60">
      <article className="mx-auto max-w-4xl space-y-8 px-5 py-10">
        <header>
          <p className="text-sm font-medium text-blue-700">Coding skills</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">The PyTorch toolbox behind every problem</h1>
          <p className="mt-3 text-gray-600">
            The whole library is built from a handful of techniques. Learn them once and every problem becomes readable.
            Each card links to problems where you can see the technique in context and run it.
          </p>
        </header>

        {SKILLS.map((s, i) => (
          <section key={s.title} className="rounded-xl border border-gray-200 bg-white p-6">
            <h2 className="flex items-baseline gap-3 text-lg font-semibold">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
                {i + 1}
              </span>
              {s.title}
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-gray-700">
              <RichText text={s.text} />
            </p>
            <div className="mt-4">
              <CodeBlock code={s.code} />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-gray-500">Used in</span>
              {s.usedIn
                .filter((id) => title(id))
                .map((id) => (
                  <a key={id} href={`#/${id}`} className="rounded-full bg-blue-50 px-2.5 py-1 text-blue-700 hover:bg-blue-100">
                    <RichText text={title(id)!} />
                  </a>
                ))}
            </div>
          </section>
        ))}
      </article>
    </div>
  );
}
