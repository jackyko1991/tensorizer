# Tensorizer

**An education tool to study physics equation conversion to tensor form, PyTorch implementation, solving and visualization.**

Tensorizer standardizes the conversion of physics equations into GPU-accelerated tensors through the following workflow:

```mermaid
flowchart LR
    A["Vector form"] --> B["Index (tensor) notation"]
    B --> C["PyTorch tensors / einsum"]
    C --> D["Solver"]
    D --> E["Animated result"]
```

For example, the projectile with air drag:

```math
m\,\ddot{\mathbf r} = m\,\mathbf g - c\,\lVert\dot{\mathbf r}\rVert\,\dot{\mathbf r}
```

first convert to the tensor notation form:
```math
\ddot x_{bi} = g_i - k\,\sqrt{v_{bj}v_{bj}}\;v_{bi}
```

Here $b$ indexes a batch of balls and $i, j$ are spatial components. The repeated index $j$ is summed, which is exactly `torch.einsum('bj,bj->b', v, v)`. The trajectory solution is integrated via RK4 with animated visualization.

## Problem library

<!-- library:start -->
82 problems in 10 topics, three per difficulty level. Click a topic to expand it.

<details>
<summary><b>Mechanics</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Simple pendulum beyond small angles](problems/mechanics/simple_pendulum) | $\ddot\theta_b = -\frac{g}{L}\,\sin\theta_b$ |
| [Projectile with air drag](problems/mechanics/projectile) | $\ddot x_{bi} = g_i - \tfrac{c}{m}\,\sqrt{v_{bj}v_{bj}}\;v_{bi}$ |
| [Block on an inclined plane with friction](problems/mechanics/inclined_plane) | $a_{bi} = g_i + \tfrac{1}{m}N_{bi} + \tfrac{1}{m}f_{bi}$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Coupled oscillators and normal modes](problems/mechanics/coupled_oscillators) | $m\,\ddot x_i = -K_{ij}\,x_j$ |
| [Double pendulum and chaos](problems/mechanics/double_pendulum) | $M_{bij}(q)\,\ddot q_{bj} = f_{bi}(q, \dot q)$ |
| [Free rigid-body rotation (tennis-racket effect)](problems/mechanics/rigid_body_rotation) | $I_{ij}\,\dot\omega_j + \varepsilon_{ijk}\,\omega_j\,I_{kl}\,\omega_l = 0$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [N-body gravity in 3D](problems/mechanics/n_body_gravity) | $\ddot x_{ai} = G\,m_b\,(r_{abj}r_{abj} + \varepsilon^2)^{-3/2}\,r_{abi}$ |
| [Equations of motion from a Lagrangian by autograd](problems/mechanics/lagrangian_autograd) | $\ddot q_i = H^{-1}_{ij}\Big(\frac{\partial L}{\partial q_j} - C_{jk}\,\dot q_k\Big)$ |
| [Mass–spring cloth](problems/mechanics/mass_spring_cloth) | $m\,\ddot x_{ai} = -k\,D_{ea}\,(\lVert d_e\rVert - \ell_0)\,\hat d_{ei} + m\,g_i$ |

</details>

</details>

<details>
<summary><b>Optics</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Snell's law in layered media](problems/optics/snell_refraction) | $t_{ri} = \eta_r\,d_{ri} + \left(\eta_r c_r - \sqrt{1-\eta_r^2(1-c_r^2)}\right) n_{ri}$ |
| [Reflection from plane and curved mirrors](problems/optics/mirror_reflection) | $r_{ri} = d_{ri} - 2\,(d_{rj}\,n_{rj})\,n_{ri}$ |
| [Image formed by a thin lens](problems/optics/thin_lens_image) | $s'_k = \left(\frac{1}{f} - \frac{1}{s_k}\right)^{-1}$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Two-lens system with ray-transfer matrices](problems/optics/lens_system) | $y'_{ri} = M_{ij}\,y_{rj}$ |
| [Diffraction grating as a sum of phasors](problems/optics/diffraction_grating) | $E_t = A_n\,P_{nt}$ |
| [Polarization with Jones matrices](problems/optics/polarization_jones) | $E^{\text{out}}_i = P_{ij}\,W_{jk}\,E^{\text{in}}_k$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [Double-slit interference](problems/optics/double_slit) | $\partial_t^2 u = c^2\,\partial_i\partial_i u$ |
| [Fraunhofer diffraction as a 2D Fourier transform](problems/optics/fourier_optics) | $U_{kl} = F_{ka}\,A_{ab}\,F_{lb}$ |
| [Road mirage from the ray equation](problems/optics/grin_mirage) | $\frac{d}{ds}\Big(n\,\frac{dr_i}{ds}\Big) = \partial_i n$ |

</details>

</details>

<details>
<summary><b>Waves</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Standing waves on a string](problems/waves/standing_waves) | $\ddot u_{bi} = c^2\,D_{ij}\,u_{bj}$ |
| [Beats and group velocity](problems/waves/beats_superposition) | $y_{nj} = A_s\cos\theta_{snj}$ |
| [Doppler effect and Mach cone](problems/waves/doppler_effect) | $t_{oe} = t_e + \tfrac{1}{c}\sqrt{(o_{oi} - s_{ei})(o_{oi} - s_{ei})}$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Pulse at a string junction](problems/waves/string_junction) | $\mu_i\,\ddot u_i = T\,D_{ij}\,u_j$ |
| [Dispersive water waves](problems/waves/dispersive_waves) | $\hat a_{nk} = \hat a_k\,e^{-i\omega_k t_n}$ |
| [Vibrating drum modes](problems/waves/membrane_modes) | $\Delta_{(ab)(cd)} = D_{ac}\,\delta_{bd} + \delta_{ac}\,D_{bd}$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [KdV soliton collision](problems/waves/kdv_solitons) | $\partial_t \hat u_k = i k^3\,\hat u_k - 3ik\,F_{kj}\,(u_j u_j)$ |
| [Rogue waves (nonlinear Schrödinger)](problems/waves/nonlinear_schrodinger) | $\psi_j \leftarrow F^{-1}_{jk}\,e^{-ik^2\Delta t/4}\,F_{kl}\,\psi_l$ |
| [Ripple tank interference](problems/waves/ripple_tank) | $(\nabla^2 u)_{ab} = \frac{1}{h^2}\,K_{\alpha\beta}\,u_{a+\alpha,\,b+\beta}$ |

</details>

</details>

<details>
<summary><b>Thermodynamics</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Carnot cycle of an ideal gas](problems/thermodynamics/carnot_cycle) | $W = \oint p\,dV,\;\; \eta = 1 - T_c/T_h$ |
| [Newton's law of cooling](problems/thermodynamics/newton_cooling) | $\dot T_b = -k_b\,(T_b - T_{\text{env}})$ |
| [Four ideal-gas processes](problems/thermodynamics/ideal_gas_processes) | $W = \int p\,dV,\;\; Q = \Delta U + W$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Heat diffusion on a plate](problems/thermodynamics/heat_diffusion) | $\partial_t T = \alpha\,\partial_i\partial_i T$ |
| [Random walk and diffusion](problems/thermodynamics/random_walk_diffusion) | $\langle x_i x_i\rangle = 2dDt$ |
| [Black-body radiation](problems/thermodynamics/blackbody_radiation) | $u_{ab} = \frac{8\pi h c}{\lambda_b^5}\,\frac{1}{\exp\!\big(hc / \lambda_b k T_a\big) - 1}$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [Kinetic theory of an ideal gas](problems/thermodynamics/kinetic_gas) | $v'_{ai} = v_{ai} - (\Delta v_{abj}\,n_{abj})\,n_{abi}$ |
| [2D Ising model](problems/thermodynamics/ising_model) | $h_{ij} = K_{kl}\,s_{i+k,\,j+l}$ |
| [Van der Waals gas and Maxwell's construction](problems/thermodynamics/van_der_waals) | $p_{tv} = \frac{nRT_t}{V_v - nb} - \frac{an^2}{V_v^2}$ |

</details>

</details>

<details>
<summary><b>Electromagnetism</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Charged particles in E and B fields](problems/em/lorentz_force) | $m_b\,\dot v_{bi} = q_b\,\big(E_i + \varepsilon_{ijk}\,v_{bj}\,B_k\big)$ |
| [Electric field of point charges](problems/em/coulomb_field) | $E_{gi} = k\,Q_q\,\frac{r_{gqi}}{\big(r_{gqj}\,r_{gqj}\big)^{3/2}}$ |
| [Charging a capacitor (RC circuit)](problems/em/rc_circuit) | $\dot V_b = \frac{V_s - V_b}{R_b\,C}$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Electrostatic potential (Poisson equation)](problems/em/electrostatics) | $\partial_i\partial_i\,\phi = -\rho/\varepsilon_0 \;\;\longrightarrow\;\; L_{nm}\,\phi_m = -\rho_n/\varepsilon_0$ |
| [Magnetic field of current loops (Biot–Savart)](problems/em/biot_savart) | $B_{gi} = \frac{\mu_0 I}{4\pi}\,\varepsilon_{ijk}\,dl_{sj}\,\frac{r_{gsk}}{\big(r_{gsl}\,r_{gsl}\big)^{3/2}}$ |
| [Electromagnetic waves (FDTD)](problems/em/em_wave_fdtd) | $\partial_t H_i = -\varepsilon_{ijk}\,\partial_j E_k$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [Lorentz boost of the field tensor](problems/em/field_tensor_boost) | $F'^{\mu\nu} = \Lambda^{\mu}{}_{\alpha}\,\Lambda^{\nu}{}_{\beta}\,F^{\alpha\beta}$ |
| [Relativistic charged particle](problems/em/relativistic_particle) | $\frac{dU^\mu}{d\tau} = \frac{q}{m}\,F^{\mu}{}_{\nu}\,U^\nu$ |
| [Rectangular waveguide modes](problems/em/waveguide_modes) | $\partial_i\partial_i\,\psi + k_c^2\,\psi = 0 \;\;\longrightarrow\;\; -L_{nm}\,\psi_m = k_c^2\,\psi_n$ |

</details>

</details>

<details>
<summary><b>Fluids & Continuum</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Hydrostatic pressure and buoyancy](problems/fluids/hydrostatics_buoyancy) | $p = p_0 + \rho_w\,g_i x_i$ |
| [Bernoulli and the Venturi tube](problems/fluids/bernoulli_venturi) | $A_k\,u_k = Q$ |
| [Draining a tank (Torricelli)](problems/fluids/torricelli_draining) | $\dot h_b = -\frac{a_b}{A}\,\sqrt{2 g\,h_b}$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Burgers equation and shock formation](problems/fluids/burgers_shock) | $\dot u_i = -\tfrac12\,D_{ij}\,u_j^2 + \nu\,L_{ij}\,u_j$ |
| [Navier–Stokes pipe flow](problems/fluids/navier_stokes_pipe_flow) | $\rho\,(\partial_t u_i + u_j\,\partial_j u_i) = -\partial_i p + \mu\,\partial_j\partial_j u_i$ |
| [Stress, strain and Mohr's circle](problems/fluids/stress_strain) | $\varepsilon_{ij} = \tfrac12(\partial_i u_j + \partial_j u_i)$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [Lid-driven cavity (Navier–Stokes)](problems/fluids/lid_driven_cavity) | $\partial_t\omega + u_j\,\partial_j\omega = \nu\,\partial_j\partial_j\omega$ |
| [Flow past a cylinder: drag and vortex shedding](problems/fluids/cylinder_drag) | $f_i(\mathbf x + \mathbf e_i, t + 1) = f_i - \tfrac1\tau\big(f_i - f_i^\text{eq}\big)$ |
| [Elastic P and S waves](problems/fluids/elastic_waves) | $\rho\,\ddot u_i = (\lambda + \mu)\,\partial_i\partial_j u_j + \mu\,\partial_j\partial_j u_i + f_i$ |

</details>

</details>

<details>
<summary><b>Quantum</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Particle in a box](problems/quantum/infinite_well) | $H_{xy}\,\varphi_{yn} = E_n\,\varphi_{xn}$ |
| [Quantum harmonic oscillator](problems/quantum/harmonic_oscillator) | $H_{xy}\,\varphi_{yn} = E_n\,\varphi_{xn},\;\; V_x = \tfrac12\omega^2 x^2$ |
| [Free wave packet](problems/quantum/free_wave_packet) | $\tilde\psi_{tk} = e^{-ik_k^2 t_t/2}\,\tilde\psi_k$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Spin precession and Rabi flopping](problems/quantum/spin_precession) | $H_{ab} = -\tfrac{\gamma}{2}\,B_k\,\sigma_{kab}$ |
| [Hydrogen atom orbitals](problems/quantum/hydrogen_radial) | $H_{lij}\,u_{ljn} = E_{ln}\,u_{lin}$ |
| [Tunnelling in a double well](problems/quantum/double_well) | $H_{xy}\,\varphi_{yn} = E_n\,\varphi_{xn}$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [Quantum tunnelling](problems/quantum/tunneling) | $\psi_x \leftarrow F^{-1}_{xk}\,K_k\,F_{ky}\,P_y\,\psi_y$ |
| [2D wave-packet scattering](problems/quantum/scattering_2d) | $\psi \leftarrow P\,\mathcal F^{-1} K\,\mathcal F\,P\,\psi$ |
| [Two-qubit entanglement](problems/quantum/two_qubit_entanglement) | $\psi_{ab} \leftarrow G_{abcd}\,\psi_{cd}$ |

</details>

</details>

<details>
<summary><b>Astrophysics & Relativity</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Kepler's laws of planetary motion](problems/astro/kepler_orbits) | $\ddot x_{bi} = -GM\,(x_{bj}x_{bj})^{-3/2}\,x_{bi}$ |
| [Escape velocity](problems/astro/escape_velocity) | $\ddot x_{bi} = -GM\,(x_{bj}x_{bj})^{-3/2}\,x_{bi}$ |
| [Hubble expansion](problems/astro/hubble_expansion) | $x_{tgi} = a_t\,q_{gi}$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Special relativity: Minkowski diagrams and Lorentz boosts](problems/astro/minkowski_diagram) | $x'^\mu = \Lambda^\mu{}_\nu\,x^\nu$ |
| [Lagrange points of the restricted three-body problem](problems/astro/lagrange_points) | $\ddot x_i = -m_s\,\frac{(x - x_s)_i}{\lvert x - x_s\rvert^3} - 2\,\varepsilon_{ijk}\Omega_j\dot x_k - \varepsilon_{ijk}\Omega_j\,\varepsilon_{klm}\Omega_l x_m$ |
| [Spaghettification near a black hole](problems/astro/tidal_tensor) | $\ddot\xi_{bi} = T_{ij}\,\xi_{bj}$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [General relativity: orbits and infall around a black hole](problems/astro/black_hole_geodesics) | $\frac{d^2x^\mu}{d\tau^2} = -\Gamma^\mu{}_{\alpha\beta}\,\dot x^\alpha\dot x^\beta$ |
| [Hawking radiation and black-hole evaporation](problems/astro/hawking_radiation) | $T_m = \frac{\hbar c^3}{8\pi G k_B}\,M_m^{-1}$ |
| [Gravitational lensing by a black hole](problems/astro/gravitational_lensing) | $u_b'' = -u_b + 3M\,u_b^2$ |

</details>

</details>

<details>
<summary><b>Nuclear & Particle</b> (9 problems)</summary>

<details>
<summary>🟢 Beginner (3)</summary>

| Problem | Key equation |
|---|---|
| [Radioactive decay](problems/nuclear/radioactive_decay) | $s_n(t+\Delta t) = s_n(t)\,\big[u_n \ge p\big]$ |
| [Decay chain (Bateman equations)](problems/nuclear/decay_chain) | $\dot N_i = A_{ij}\,N_j$ |
| [Nuclear binding energy](problems/nuclear/binding_energy) | $B_{zn} = a_V A_{zn} - a_S A_{zn}^{2/3} - a_C\,\frac{Z_z(Z_z-1)}{A_{zn}^{1/3}} - a_A\,\frac{(N_n-Z_z)^2}{A_{zn}} + \delta_{zn}$ |

</details>

<details>
<summary>🟠 Intermediate (3)</summary>

| Problem | Key equation |
|---|---|
| [Rutherford scattering](problems/nuclear/rutherford_scattering) | $\ddot x_{bi} = \frac{k}{m}\,\frac{x_{bi}}{(x_{bj}x_{bj})^{3/2}}$ |
| [Neutrino oscillations](problems/nuclear/neutrino_oscillations) | $A_{k\beta\alpha} = U_{\beta i}\,U^*_{\alpha i}\,e^{-i\,m_i^2 L_k/2E}$ |
| [Four-momentum and two-body decay](problems/nuclear/four_momentum) | $p_\mu p^\mu = \eta_{\mu\nu}\,p^\mu p^\nu = m^2$ |

</details>

<details>
<summary>🔴 Advanced (3)</summary>

| Problem | Key equation |
|---|---|
| [Nuclear shell model](problems/nuclear/shell_model) | $H_{kij} = -\frac{\hbar^2}{2m}D_{ij} + \Big[V_k(r_i) + \frac{\hbar^2 l_k(l_k+1)}{2m r_i^2}\Big]\delta_{ij}$ |
| [Reactor criticality](problems/nuclear/reactor_criticality) | $M_{ij}\,\phi_j = \frac1k\,F_{ij}\,\phi_j$ |
| [Electromagnetic shower](problems/nuclear/em_shower) | $(E_p, s_p) \to \big(\tfrac{E_p}{2}, 1\big),\ \big(\tfrac{E_p}{2}, 1 - s_p\big)\ \text{ if } \tfrac{E_p}{2} \ge E_c$ |

</details>

</details>

<details>
<summary><b>Others</b> (1 problems)</summary>

<details>
<summary>🟢 Beginner (1)</summary>

| Problem | Key equation |
|---|---|
| [Receptor–ligand binding (k_on, k_off)](problems/others/receptor_binding) | $J_{bc} = k^{\mathrm{on}}_b\,R_{bc}\,L_{bc} - k^{\mathrm{off}}_b\,C_{bc}$ |

</details>

</details>
<!-- library:end -->

Every problem page shows:

1. **Equations**: vector form, index form, and a translation table (symbol → code → tensor shape → meaning).
2. **Code**: five syntax-highlighted files:
   | File | Role |
   |---|---|
   | `main.py` | main loop: parameters → tensors → solve → plot |
   | `equation.py` | physics → tensor translation; the law written with `einsum`, one-to-one with the index form |
   | `tensors.py` | builds the concrete PyTorch tensors (grids, operators, initial states) |
   | `solve.py` | the solver (time stepping, `linalg.solve`, `eigh`, FFT, …) |
   | `plot.py` | 2D/3D matplotlib plot rendered to a GIF (or PNG for static problems) |
3. **Result**: an animated GIF, parameter sliders, a Run button, and the console output.

## Quick start

Requires [pixi](https://pixi.sh). It installs Python, PyTorch, Node.js and everything else into `.pixi/`, so nothing else is needed. It works on Windows, macOS (Apple silicon) and Linux.

```bash
pixi install              # create the environment
pixi run serve            # render all problems, build the site, serve at http://localhost:8000
```

The first `serve` renders every problem, which takes about 6 minutes on a CPU. Later runs re-render only the problems whose code changed. Then open <http://localhost:8000>.

### Development (hot reload)

```bash
pixi run dev              # API + Vite together → http://localhost:5173, Ctrl+C stops both
```

| You edit | What happens |
|---|---|
| `frontend/src/**` | the page updates instantly (React Fast Refresh) |
| `backend/**`, `problems/_common/**` | the API restarts automatically |
| `problems/<category>/<slug>/*.py` or `meta.yaml` | about 1 s later the page reloads with the new code and equations; the GIF re-renders in the background (about 20 s) and the page reloads again |

`pixi run dev` needs port 8000 free, so stop `pixi run serve` first. To run the two servers in separate terminals instead, use `pixi run api` and `pixi run web`.

Other tasks:

```bash
pixi run build-assets     # re-render changed problems → frontend/public/generated/ (add -- --force for all)
pixi run test             # physics checks + API tests
pixi run lint             # ruff
pixi run format
python -m problems.mechanics.projectile.main --out out --params '{"v0": 40}'   # run one problem (inside `pixi shell`)
```

## How it works

```mermaid
flowchart LR
    subgraph P["problems/&lt;category&gt;/&lt;slug&gt;/"]
        meta["meta.yaml<br/>LaTeX, translation table, params"]
        code["main.py · equation.py · tensors.py<br/>solve.py · plot.py"]
    end
    common["problems/_common/<br/>device, params, RK4, GIF writer"] --> code
    P --> build["scripts/build_assets.py<br/>default params"]
    build --> gen["frontend/public/generated/<br/>GIFs + manifest.json"]
    gen --> web["frontend/<br/>React + KaTeX + CodeMirror"]
    web -- "POST /api/run/{category}/{slug}<br/>custom params or edited code" --> api["backend/app/<br/>FastAPI"]
    api -- "subprocess + timeout + cache" --> P
    api -- "GIF / PNG + console" --> web
    tests["tests/<br/>physics checks + API tests"] -.-> P
```

| Path | Contents |
|---|---|
| `problems/<category>/<slug>/` | one self-contained package per problem: `meta.yaml` + the five code files |
| `problems/_common/` | device choice, parameter handling, RK4, white-theme GIF writer |
| `scripts/build_assets.py` | renders problems with default params → GIFs + `manifest.json` |
| `backend/app/` | FastAPI: `POST /api/run/{category}/{slug}` re-renders with custom params |
| `frontend/` | React + Vite + TypeScript, KaTeX (equations), CodeMirror (code) |
| `tests/` | per-problem physics checks + API tests |

- **Prebuilt results.** `build_assets.py` renders every problem ahead of time. The site therefore shows results instantly, even with no backend running (static hosting).
- **Re-runs.** The backend runs a problem in a subprocess with a timeout, one render at a time, and caches results by a hash of the parameters (and code).
- **Edited code.** Running your own edits is controlled by `ALLOW_CODE_EXEC=1`. The pixi `api` and `serve` tasks turn it on, because running edited code executes arbitrary Python. Only enable it on your own machine, never on a public server.
- **Physics tests.** Each problem is checked against known results. Examples:

  | Check | Expected |
  |---|---|
  | Projectile range without drag | $R = v_0^2 \sin 2\theta / g$ |
  | Normal-mode frequencies | $\omega_n = 2\sqrt{k/m}\,\sin\frac{n\pi}{2(N+1)}$ |
  | Snell's law | $n_1 \sin\theta_1 = n_2 \sin\theta_2$ |
  | Carnot efficiency | $\eta = 1 - T_c/T_h$ |
  | Infinite-well eigenvalues | $E_n = \dfrac{n^2\pi^2}{2L^2}$ |
  | Larmor frequency | $\omega = \gamma B$ |
  | Field-tensor invariant | $F_{\mu\nu}F^{\mu\nu} = 2(B^2 - E^2)$ |

  Energy and momentum conservation are checked too.

## Adding a problem

1. Copy `problems/mechanics/projectile/` to `problems/<category>/<new_slug>/`.
2. Edit `meta.yaml`: the LaTeX, translation table and parameters. In `summary`, `notes`, the translation `meaning` and parameter `label`, write math as inline LaTeX between `$…$` (for example `$2\times 2$ matrices`); the page typesets it with KaTeX. Use single-quoted or block YAML strings so backslashes survive. Then rewrite the five files, keeping `run(params, out_dir) -> Path` in `main.py`, and add a `diagram.svg` sketch of the setup (viewBox `0 0 240 160`, same style as `simple_pendulum`).
3. Add physics checks to `tests/test_physics_<category>.py`. `tests/test_smoke.py` already checks that every problem follows the convention and renders.
4. Run `pixi run test`, `pixi run build-assets` and `pixi run readme` (refreshes the problem library above).

A new category also needs an entry in `CATEGORIES` in `scripts/build_assets.py`.

## Roadmap

- More problems in every category, then fluid dynamics (Burgers, Navier–Stokes) and continuum mechanics (stress/strain tensors).
- Real datasets: compare simulations with measurements, and fit physical parameters with autograd (inverse problems, PINNs).
- Free-form input: type LaTeX, handwrite, or photograph an equation → translated to tensor form.
- Hosting: GitHub Pages for the static site, plus an optional backend for re-runs.
