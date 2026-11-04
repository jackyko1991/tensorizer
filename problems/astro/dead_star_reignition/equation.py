r"""Physics → tensor translation (cgs units throughout).

Can a dead star be re-lit by feeding it gas? It depends on whether it has a surface, and on degeneracy.

1. Polytropes (Lane–Emden), batched over the index n:   θ'' + (2/ξ) θ' + θ^n = 0,  θ(0) = 1, θ'(0) = 0.
   n = 3 with relativistic degenerate electrons gives the Chandrasekhar mass
       M_Ch = \frac{\sqrt{3π}}{2}\, ω_3 \left(\frac{ħc}{G}\right)^{3/2} \frac{1}{(μ_e m_u)^2},   ω_3 = -ξ_1^2 θ'(ξ_1)
2. Brown dwarfs: an n = 3/2 polytrope of mass M and radius R has P_c = W GM²/R⁴, ρ_c = (ρ_c/ρ̄)·3M/(4πR³).
   With ions ideal and electrons partly degenerate,
       P_c = n_i k T_c + \sqrt{(n_e k T_c)^2 + P_{deg}(n_e)^2},
   T_c(R) rises as the body contracts, peaks, then falls once degeneracy pressure takes over. Its maximum
   T_max(M) ∝ M^{4/3} decides whether deuterium (≈ 3×10⁵ K over Gyr) or hydrogen (≈ 3×10⁶ K) can ever burn.
3. Accreted shells on white dwarfs (novae) and neutron stars (type I X-ray bursts): a one-zone model of the base
   of the accreted layer at column depth y (g/cm²), pressure P = g y, temperature from radiative diffusion
       T^4 = T_{floor}^4 + \frac{3κ\,y\,F}{a c},\qquad F = \dot m\,(Q_b + f_{comp}\, c_p T) + f_H\,\tfrac12 ε_{HCNO}\, y
   ignites where the burning becomes thermally unstable (Fujimoto et al. 1981; Bildsten 1998):
       \frac{∂ε_{nuc}}{∂T} > \frac{∂ε_{cool}}{∂T},\qquad ε_{cool} = \frac{a c T^4}{3 κ y^2}
   (derivatives by autograd at fixed ρ: the thin layer cannot expand in time). Critical mass ΔM = 4πR² y_ign.
4. Black holes have no surface: nothing piles up, nothing ignites. They only shine by accretion,
   L = η \dot M c^2 with η = 1 − E_{ISCO}, capped near L_Edd = 4πGMc m_p/σ_T.

Everything except items 1 and 4 is an order-of-magnitude model: real numbers come from evolution codes.
"""

import math

import torch

# --- constants (cgs) ---
G = 6.674e-8
C = 2.998e10
HBAR = 1.0546e-27
K_B = 1.3807e-16
M_U = 1.6605e-24  # atomic mass unit
M_P = 1.6726e-24
M_E = 9.109e-28
SIGMA_T = 6.6524e-25
A_RAD = 7.5657e-15
M_SUN = 1.989e33
R_SUN = 6.957e10
M_JUP = 1.898e30
YEAR = 3.156e7
MEV = 1.6022e-6


# ---------- 1. polytropes ----------
def lane_emden_rhs(y: torch.Tensor, xi: float, n: torch.Tensor) -> torch.Tensor:
    """y = (θ, θ'): (B, 2) → (B, 2). θ is clipped at 0 past the surface so θ^n stays real."""
    theta, dtheta = y[:, 0], y[:, 1]
    return torch.stack([dtheta, -(theta.clamp(min=0) ** n) - 2 * dtheta / xi], dim=1)


def chandrasekhar_mass(omega3: torch.Tensor, mu_e: float) -> torch.Tensor:
    """M_Ch = (√(3π)/2) ω₃ (ħc/G)^{3/2} / (μ_e m_u)²  (grams)."""
    return math.sqrt(3 * math.pi) / 2 * omega3 * (HBAR * C / G) ** 1.5 / (mu_e * M_U) ** 2


def white_dwarf_radius(M: torch.Tensor, m_ch: float) -> torch.Tensor:
    """Nauenberg (1972) fit R = 0.0112 R☉ √((M/M_Ch)^{-2/3} − (M/M_Ch)^{2/3}): → 0 at the Chandrasekhar mass."""
    x = M / m_ch
    return 0.0112 * R_SUN * torch.sqrt(x ** (-2 / 3) - x ** (2 / 3))


# ---------- equation of state ----------
def degenerate_pressure(n_e: torch.Tensor) -> torch.Tensor:
    """Cold electron pressure, joining the non-relativistic ∝ n^{5/3} and relativistic ∝ n^{4/3} limits."""
    p_nr = (3 * math.pi**2) ** (2 / 3) * HBAR**2 / (5 * M_E) * n_e ** (5 / 3)
    p_r = (3 * math.pi**2) ** (1 / 3) * HBAR * C / 4 * n_e ** (4 / 3)
    return (p_nr**-2 + p_r**-2) ** -0.5


def number_densities(rho, X, Y, Z):
    """Ions and electrons per cm³ for mass fractions X (H), Y (He), Z (metals, A ≈ 14, A/Z ≈ 2)."""
    n_i = rho * (X + Y / 4 + Z / 14) / M_U
    n_e = rho * (X + Y / 2 + Z / 2) / M_U
    return n_i, n_e


def pressure(rho, T, X, Y, Z):
    """Ideal ions + electrons interpolated between ideal and degenerate: n_i kT + √((n_e kT)² + P_deg²)."""
    n_i, n_e = number_densities(rho, X, Y, Z)
    return n_i * K_B * T + torch.sqrt((n_e * K_B * T) ** 2 + degenerate_pressure(n_e) ** 2)


def density(P, T, X, Y, Z, iterations: int = 48):
    """Invert P(ρ, T) for ρ by bisection in log₁₀ ρ ∈ [−8, 12], all batch members at once."""
    lo, hi = torch.full_like(P, -8.0), torch.full_like(P, 12.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        too_dense = pressure(10**mid, T, X, Y, Z) > P
        hi, lo = torch.where(too_dense, mid, hi), torch.where(too_dense, lo, mid)
    return 10 ** (0.5 * (lo + hi))


# ---------- 2. brown dwarfs ----------
def central_temperature(M, R, W, rho_ratio, X, Y, Z):
    """T_c of an n = 3/2 polytrope: solve P_c = A T + √((B T)² + D²) for T (a quadratic). NaN if P_c < P_deg."""
    rho = rho_ratio * 3 * M / (4 * math.pi * R**3)
    P = W * G * M**2 / R**4
    n_i, n_e = number_densities(rho, X, Y, Z)
    A, B, D = n_i * K_B, n_e * K_B, degenerate_pressure(n_e)
    a2 = A**2 - B**2  # < 0 (more electrons than ions)
    T = (P * A - torch.sqrt((P * A) ** 2 - a2 * (P**2 - D**2))) / a2
    return torch.where(P > D, T, torch.full_like(T, math.nan))


# ---------- 3. accreted shells ----------
def opacity(rho, T, X, Z):
    """Electron scattering + Kramers free–free (conduction neglected)."""
    return 0.2 * (1 + X) + 3.68e22 * (1 - Z) * (1 + X) * rho * T**-3.5


def eps_hot_cno(Z):
    """β-decay-limited ('hot') CNO cycle: temperature independent, so it never runs away by itself."""
    return 5.8e15 * Z


def eps_nuclear(rho, T, X, Y, Z):
    """Hydrogen (CNO, saturating at the hot-CNO rate) + helium (triple-α, Bildsten 1998), erg g⁻¹ s⁻¹."""
    T9, T8 = T / 1e9, T / 1e8
    cno = 8.67e27 * rho * X * Z * T9 ** (-2 / 3) * torch.exp(-15.228 * T9 ** (-1 / 3))
    hot = eps_hot_cno(Z)
    hydrogen = cno * hot / (cno + hot)  # harmonic join: cold CNO at low T, hot CNO at high T
    triple_alpha = 5.3e21 * (rho / 1e5) ** 2 * Y**3 / T8**3 * torch.exp(-44 / T8)
    return hydrogen + triple_alpha


def eps_cooling(T, y, kappa):
    """Radiative loss of the layer per gram: a c T⁴ / (3 κ y²)."""
    return A_RAD * C * T**4 / (3 * kappa * y**2)


def instability(rho, T, y, X, Y, Z) -> torch.Tensor:
    """∂ε_nuc/∂T − ∂ε_cool/∂T at fixed ρ and y, by autograd; > 0 ⇒ thermonuclear runaway."""
    T = T.detach().requires_grad_(True)
    net = eps_nuclear(rho, T, X, Y, Z) - eps_cooling(T, y, opacity(rho, T, X, Z))
    return torch.autograd.grad(net.sum(), T)[0]  # elementwise: each entry depends on its own T only


def burst_alpha(M, R, X):
    """α = gravitational energy released per gram / nuclear energy per gram (Q_nuc ≈ 1.6 + 4X MeV per nucleon)."""
    return (G * M / R) / ((1.6 + 4.0 * X) * MEV / M_U)


def spin_up_mass(P, M, inertia: float = 1e45):
    """Accreted mass that carries the angular momentum I·2π/P at the corotation radius: ΔM = IΩ / √(G M r_co)."""
    omega = 2 * math.pi / P
    r_co = (G * M / omega**2) ** (1 / 3)
    return inertia * omega / torch.sqrt(G * M * r_co)


# ---------- 4. black holes ----------
def eddington_luminosity(M):
    """L_Edd = 4πGMc m_p / σ_T: radiation pressure on electrons balances gravity on protons (≈ 1.26e38 erg/s/M☉)."""
    return 4 * math.pi * G * M * C * M_P / SIGMA_T


def isco_radius(spin):
    """Innermost stable circular orbit of a prograde disc, in GM/c² (Bardeen, Press & Teukolsky 1972): 6 for a = 0,
    1 for a → 1."""
    z1 = 1 + (1 - spin**2) ** (1 / 3) * ((1 + spin) ** (1 / 3) + (1 - spin).clamp(min=0) ** (1 / 3))
    z2 = torch.sqrt(3 * spin**2 + z1**2)
    return 3 + z2 - torch.sqrt((3 - z1) * (3 + z1 + 2 * z2))  # in GM/c²


def horizon_radius(spin):
    """Outer horizon r₊ = 1 + √(1 − a²), in GM/c²."""
    return 1 + torch.sqrt(1 - spin**2)


def accretion_efficiency(spin):
    """η = 1 − E_ISCO for a prograde thin disc, E = √(1 − 2M/(3 r_ISCO)) (Bardeen, Press & Teukolsky 1972)."""
    return 1 - torch.sqrt(1 - 2 / (3 * isco_radius(spin)))


def disc_temperature(r, r_in):
    """Thin-disc (Shakura–Sunyaev) temperature profile T(r) ∝ r^(−3/4) (1 − √(r_in/r))^(1/4), zero at the inner
    edge, normalised to a peak of 1 (which sits at r = (49/36) r_in).   r (…) in the same units as r_in."""
    shape = r**-0.75 * (1 - torch.sqrt(r_in / r)).clamp_min(0) ** 0.25
    peak = (49 / 36 * r_in) ** -0.75 * (1 - (36 / 49) ** 0.5) ** 0.25
    return shape / peak


def eddington_rate(M, eta):
    """Accretion rate that would give L_Edd: Ṁ_Edd = L_Edd / (η c²)."""
    return eddington_luminosity(M) / (eta * C**2)


def black_hole_luminosity(mdot, M, eta):
    """L = η Ṁ c², capped at L_Edd (above it radiation pressure blows the excess gas away; real super-Eddington
    discs exceed the cap only logarithmically)."""
    return torch.minimum(eta * mdot * C**2, eddington_luminosity(M))


def photon_orbit_radius(spin):
    """Prograde circular photon orbit in the equatorial plane, r_ph = 2 [1 + cos(⅔ arccos(−a))] in GM/c²
    (Bardeen, Press & Teukolsky 1972): 3 = 1.5 r_s for a = 0, → 1 for a → 1."""
    return 2 * (1 + torch.cos(2 / 3 * torch.acos(-spin)))
