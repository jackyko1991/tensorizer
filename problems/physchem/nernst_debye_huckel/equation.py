r"""Physics → tensor translation.

Nernst equation for the Daniell cell Zn | Zn²⁺ || Cu²⁺ | Cu  (Cu²⁺ + Zn → Cu + Zn²⁺, n = 2 electrons):

    E_{tq} = E° − (ln 10 · R T_t / n F) · log₁₀ Q_q,          Q = a(Zn²⁺) / a(Cu²⁺),   a = γ m

Debye–Hückel: an ion is surrounded by a cloud of opposite charge of thickness κ⁻¹, which lowers its activity:

    κ² = 2 N_A e² I / (ε₀ ε_r k T)                      (I = ½ Σ_k m_k z_k², the ionic strength)
    log₁₀ γ = −A z² √I                     (limiting law, I → 0)
    log₁₀ γ = −A z² √I / (1 + B a √I)      (extended law, ions of radius a)

The cloud comes from the linearised Poisson–Boltzmann equation around one ion of charge q and radius a:

    (1/r²) d/dr (r² dφ/dr) = κ² φ     ⇔     u'' = κ² u   with u = r φ

On a radial grid r_ci (c = concentration, i = grid point) this is a linear system for each c,

    A_{cij} u_{cj} = b_{ci}        (j summed)

whose solution is compared with the analytic screened-Coulomb (Yukawa) potential
φ = (q / 4π ε₀ ε_r) · e^{κa}/(1 + κa) · e^{−κr}/r.
"""

import math

import torch

from .tensors import E_CHARGE, EPS0, K_B, N_A, R_GAS, F

LN10 = math.log(10.0)


# ---------------------------------------------------------------- Nernst


def nernst_slope(T: torch.Tensor, n: int) -> torch.Tensor:
    """ln 10 · R T / n F: the EMF change per decade of Q (29.6 mV at 25 °C for n = 2)   (same shape as T)."""
    return LN10 * R_GAS * T / (n * F)


def nernst(E0: float, T: torch.Tensor, log10_Q: torch.Tensor, n: int) -> torch.Tensor:
    """E_tq = E° − s_t log₁₀Q_q   (T, Q).   No index summed: an outer product of slopes and log ratios."""
    return E0 - torch.einsum("t,q->tq", nernst_slope(T, n), log10_Q)


def emf(E0: float, T: float, m_zn, m_cu, log10_gamma_zn, log10_gamma_cu, n: int) -> torch.Tensor:
    """E = E° − s [log₁₀(m_Zn/m_Cu) + log₁₀γ_Zn − log₁₀γ_Cu]: Nernst with activities a = γ m."""
    s = nernst_slope(torch.as_tensor(T), n)
    return E0 - s * (torch.log10(m_zn / m_cu) + log10_gamma_zn - log10_gamma_cu)


# ---------------------------------------------------------------- Debye–Hückel


def permittivity_water(T: torch.Tensor) -> torch.Tensor:
    """ε_r of water (Malmberg & Maryott fit, t in °C): 87.74 − 0.40008 t + 9.398·10⁻⁴ t² − 1.41·10⁻⁶ t³."""
    t = T - 273.15
    return 87.74 - 0.40008 * t + 9.398e-4 * t**2 - 1.41e-6 * t**3


def bjerrum_length(T: torch.Tensor) -> torch.Tensor:
    """l_B = e² / (4π ε₀ ε_r k T): the distance at which two unit charges interact with energy kT (nm)."""
    return E_CHARGE**2 / (4 * math.pi * EPS0 * permittivity_water(T) * K_B * T) * 1e9


def kappa(ionic: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Inverse Debye length κ = √(2 N_A e² I · 1000 / ε₀ ε_r k T) in nm⁻¹.

    I is the ionic strength in mol/kg ≈ mol/L; the factor 1000 turns it into mol per m³.
    """
    k2 = 2 * N_A * E_CHARGE**2 * ionic * 1000 / (EPS0 * permittivity_water(T) * K_B * T)  # m⁻²
    return torch.sqrt(k2) * 1e-9


def debye_A(T: torch.Tensor) -> torch.Tensor:
    """A = l_B (κ/√I) / (2 ln 10): ≈ 0.51 (kg/mol)^½ for water at 25 °C."""
    return bjerrum_length(T) * kappa(torch.ones_like(T), T) / (2 * LN10)


def debye_B(T: torch.Tensor) -> torch.Tensor:
    """B = κ/√I: ≈ 3.29 nm⁻¹ (kg/mol)^½ at 25 °C, so κ⁻¹ ≈ 0.304 nm / √I."""
    return kappa(torch.ones_like(T), T)


def log10_gamma_limiting(z: int, ionic: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Debye–Hückel limiting law, log₁₀ γ = −A z² √I."""
    return -A * z**2 * torch.sqrt(ionic)


def log10_gamma_extended(z: int, ionic: torch.Tensor, A: torch.Tensor, B: torch.Tensor, a: float) -> torch.Tensor:
    """Extended Debye–Hückel law, log₁₀ γ = −A z² √I / (1 + B a √I)."""
    return -A * z**2 * torch.sqrt(ionic) / (1 + B * a * torch.sqrt(ionic))


# ---------------------------------------------------------------- linearised Poisson–Boltzmann


def pb_system(h: torch.Tensor, kap: torch.Tensor, a: float, M: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A_cij and b_ci for u'' = κ_c² u on grid c (potential in units of q/4πε₀ε_r, so Coulomb is 1/r).

    interior rows:  (u_{i−1} − 2u_i + u_{i+1}) / h_c² − κ_c² u_i = 0;   u_M = 0 at the outer edge
    row 0 (r = a):  Gauss's law φ'(a) = −1/a² ⇔ u'(a) − u(a)/a = −1/a, put in with a ghost point:
                    (2u₁ − 2u₀)/h² − (2/(h a)) u₀ − κ² u₀ = −2/(h a)
    h, kap: (C,) → A (C, M, M), b (C, M).
    """
    eye = torch.eye(M, device=h.device, dtype=h.dtype)
    D2 = -2 * eye + torch.diag(torch.ones(M - 1, device=h.device, dtype=h.dtype), 1)
    D2 = D2 + torch.diag(torch.ones(M - 1, device=h.device, dtype=h.dtype), -1)
    D2[0, 1] = 2.0  # ghost point u₋₁ folded back into row 0
    A = torch.einsum("c,ij->cij", 1 / h**2, D2) - torch.einsum("c,ij->cij", kap**2, eye)
    A[:, 0, 0] -= 2 / (h * a)
    b = torch.zeros(len(h), M, device=h.device, dtype=h.dtype)
    b[:, 0] = -2 / (h * a)
    return A, b


def yukawa(r: torch.Tensor, kap: torch.Tensor, a: float) -> torch.Tensor:
    """φ_ci = e^{κ_c a}/(1 + κ_c a) · e^{−κ_c r_ci}/r_ci   (units of q/4πε₀ε_r): the screened Coulomb potential."""
    k = kap[:, None]
    return torch.exp(-k * (r - a)) / ((1 + k * a) * r)


def ln_gamma_from_potential(phi_a: torch.Tensor, a: float, z: int, l_B: torch.Tensor) -> torch.Tensor:
    """ln γ = (z² l_B / 2) (φ(a) − 1/a): the energy of the ion in the potential of its own cloud.

    φ(a) − 1/a is what is left at the ion's surface after removing its own bare Coulomb potential; the factor ½
    is the charging work. With the Yukawa form this gives exactly the extended Debye–Hückel law.
    """
    return 0.5 * z**2 * l_B * (phi_a - 1 / a)


def potential_scale(T: torch.Tensor, z: int) -> torch.Tensor:
    """Volts per unit of φ̃ (nm⁻¹): z e / (4π ε₀ ε_r) · 10⁹."""
    return z * E_CHARGE / (4 * math.pi * EPS0 * permittivity_water(T)) * 1e9
