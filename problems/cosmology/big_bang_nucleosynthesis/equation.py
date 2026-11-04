r"""Physics → tensor translation: the thermal history and the nuclear reaction network.

Abundances are counted per baryon, Y_i = n_i / n_b, so expansion alone does not change them. Every reaction r
turns R_{ri} nuclei of species i into P_{ri} nuclei; with forward and reverse rate coefficients λ⁺_r, λ⁻_r

    dY_i/dt = Σ_r (P_{ri} − R_{ri}) ( λ⁺_r Π_j Y_j^{R_{rj}} − λ⁻_r Π_j Y_j^{P_{rj}} )        (r and j summed)

Baryon number is conserved because Σ_i A_i (P_{ri} − R_{ri}) = 0 for every r. The forward coefficient of a
two-body reaction is λ⁺ = n_b ⟨σv⟩ / (identical-particle factor), the reverse one follows from the Saha equation
(detailed balance):

    λ⁻_r = λ⁺_r · n_b^{−ΔN_r} · Π_i (g_i n_{Q,i})^{R_{ri} − P_{ri}} · e^{−Q_r/T},   n_Q = (m T / 2π)^{3/2}

so photodisintegration (ΔN = 1) does not care about η while fusion grows with n_b ∝ η: that is how η enters.

Clock: radiation domination, H² = (8π/3M_P²)(ρ_γ + ρ_e± + ρ_ν), and entropy conservation a³ T³ S(T) = const with
S = 2 + (7/2) h(m_e/T) (photons + e±). When e± annihilate, S falls from 11/2 to 2 and heats the photons, so
T_ν = T (S / (11/2))^{1/3} → (4/11)^{1/3} T. The integration variable is u = −ln T.
"""

import math

import torch

from .tensors import (
    CHARGE,
    HBAR,
    HBARC,
    M_E,
    M_PLANCK,
    M_U,
    MASS_EXCESS,
    MASS_NUMBER,
    MEV_PER_T9,
    N_AVOGADRO,
    Q_NP,
    SPIN_STATES,
    ZETA3,
    q_values,
    reaclib_table,
    stoichiometry,
)


# ---------------------------------------------------------------- thermal history
def electron_entropy_fraction(T: torch.Tensor, n_y: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    r"""h(z) = (ρ+P)_{e±} / (ρ+P)_{massless e±} and ρ_{e±} (MeV⁴), z = m_e/T. Momentum integral on y = p/T ∈ [0, 40].

    ρ = (4/2π²) T⁴ ∫ y² ε /(e^ε + 1) dy,   P = (4/2π²) T⁴ ∫ y⁴/(3ε) /(e^ε + 1) dy,   ε = √(y² + z²)
    """
    y = torch.linspace(0, 40, n_y, dtype=T.dtype, device=T.device)  # (Y,)
    z = (M_E / T)[:, None]  # (S, 1)
    eps = torch.sqrt(y**2 + z**2)  # (S, Y)
    occ = torch.sigmoid(-eps)  # Fermi–Dirac 1/(e^ε + 1)
    rho = torch.trapezoid(y**2 * eps * occ, y, dim=-1) * 4 / (2 * math.pi**2) * T**4
    P = torch.trapezoid(y**4 / (3 * eps) * occ, y, dim=-1) * 4 / (2 * math.pi**2) * T**4
    h = (rho + P) / (7 * math.pi**2 / 45 * T**4)
    return h, rho


def entropy_dof(T: torch.Tensor) -> torch.Tensor:
    """S(T) = 2 + (7/2) h: photon + e± entropy in units of (2π²/45) T³."""
    return 2 + 3.5 * electron_entropy_fraction(T)[0]


def neutrino_temperature(T: torch.Tensor) -> torch.Tensor:
    """T_ν = T (S/(11/2))^{1/3}: neutrinos decoupled before e± annihilation (instantaneous decoupling)."""
    return T * (entropy_dof(T) / 5.5) ** (1 / 3)


def hubble_rate(T: torch.Tensor, n_nu: float) -> torch.Tensor:
    r"""H = √(8πρ/3) / M_P in s⁻¹, ρ = π²T⁴/15 + ρ_e± + N_ν (7/8)(π²/15) T_ν⁴."""
    rho = math.pi**2 / 15 * T**4 + electron_entropy_fraction(T)[1]
    rho = rho + n_nu * 7 / 8 * math.pi**2 / 15 * neutrino_temperature(T) ** 4
    return torch.sqrt(8 * math.pi * rho / 3) / M_PLANCK / HBAR


def dt_du(T: torch.Tensor, n_nu: float, d: float = 1e-3) -> torch.Tensor:
    r"""dt/du with u = −ln T: (d ln a/du) / H, d ln a/du = 1 + ⅓ d ln S/d ln T (a ∝ 1/(T S^{1/3}))."""
    dlnS = (torch.log(entropy_dof(T * math.exp(d))) - torch.log(entropy_dof(T * math.exp(-d)))) / (2 * d)
    return (1 + dlnS / 3) / hubble_rate(T, n_nu)


def time_of(T: torch.Tensor, n_nu: float, T_hot: float = 100.0) -> torch.Tensor:
    """Cosmic time t(T) in s: t = 1/(2H) at T_hot, then t = ∫ dt/du du (trapezoid on a fine log grid)."""
    u = torch.linspace(-math.log(T_hot), -math.log(float(T.min())) + 1e-9, 3000, dtype=T.dtype, device=T.device)
    Tg = torch.exp(-u)
    rate = dt_du(Tg, n_nu)
    t0 = 0.5 / hubble_rate(Tg[:1], n_nu)
    t_grid = torch.cat([t0, t0 + torch.cumsum(0.5 * (rate[1:] + rate[:-1]) * u.diff(), 0)])
    k = torch.searchsorted(u, -torch.log(T)).clamp(1, len(u) - 1)
    w = (-torch.log(T) - u[k - 1]) / (u[k] - u[k - 1])
    return t_grid[k - 1] * (1 - w) + t_grid[k] * w


def baryon_density(T: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """n_b in cm⁻³, (S, B): η n_γ(today-normalised) = η (2ζ(3)/π²) T³ · S/2 (photons heated later by e± entropy)."""
    n_gamma = 2 * ZETA3 / math.pi**2 * T**3 * entropy_dof(T) / 2 / HBARC**3  # (S,)
    return torch.einsum("s,b->sb", n_gamma, eta)


# ---------------------------------------------------------------- weak n ↔ p rates
def weak_rates(T: torch.Tensor, T_nu: torch.Tensor, tau_n: float, n_w: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
    r"""λ_{n→p}, λ_{p→n} in s⁻¹ (Born approximation, Weinberg 1972; form of Kawano 1992), (S,) each.

    λ_{p→n} = K ∫₁^∞ x√(x²−1) [ (x+q)² / ((1+e^{−xz})(1+e^{(x+q)z_ν})) + (x−q)² / ((1+e^{xz})(1+e^{−(x−q)z_ν})) ] dx
    λ_{n→p} = the same with q → −q.   x = E_e/m_e, q = Q/m_e, z = m_e/T, z_ν = m_e/T_ν, K fixed by λ_{n→p}(T→0) = 1/τ_n.
    Substituting x = 1 + w² removes the square-root edge at x = 1.
    """
    q = Q_NP / M_E
    z, zn = (M_E / T)[:, None], (M_E / T_nu)[:, None]  # (S, 1)
    w_max = torch.sqrt(q + 60 * T / M_E)[:, None]  # covers e^{−60} tails, (S, 1)
    w = torch.linspace(0, 1, n_w, dtype=T.dtype, device=T.device) * w_max  # (S, W)
    x = 1 + w**2
    jac = x * w * torch.sqrt(w**2 + 2) * 2 * w  # x √(x²−1) dx/dw

    def rate(qs: float) -> torch.Tensor:
        a = (x + qs) ** 2 * torch.sigmoid(x * z) * torch.sigmoid(-(x + qs) * zn)
        b = (x - qs) ** 2 * torch.sigmoid(-x * z) * torch.sigmoid((x - qs) * zn)
        return torch.trapezoid(jac * (a + b), w, dim=-1)

    w0 = torch.linspace(0, math.sqrt(q - 1), 20001, dtype=T.dtype, device=T.device)
    x0 = 1 + w0**2
    decay = torch.trapezoid(x0 * (x0 - q) ** 2 * w0 * torch.sqrt(w0**2 + 2) * 2 * w0, w0)  # ∫₁^q x(x−q)²√(x²−1)
    K = 1 / (tau_n * decay)
    return K * rate(-q), K * rate(q)


# ---------------------------------------------------------------- nuclear rates
def reaclib_rates(T: torch.Tensor) -> torch.Tensor:
    r"""N_A⟨σv⟩ (cm³ mol⁻¹ s⁻¹) of every reaction, (S, Rx); column 0 (weak) is 0.

    Σ over fit sets s: exp(a_{sk} φ_k(T9)), φ = (1, T9⁻¹, T9^{−1/3}, T9^{1/3}, T9, T9^{5/3}, ln T9): repeated k summed.
    The fits are used up to T9 = 10 (held constant above, where nuclei are photodissociated anyway).
    """
    a, idx = reaclib_table(T.device)
    T9 = (T / MEV_PER_T9).clamp(max=10.0)
    phi = torch.stack(
        [torch.ones_like(T9), 1 / T9, T9 ** (-1 / 3), T9 ** (1 / 3), T9, T9 ** (5 / 3), torch.log(T9)], -1
    )
    per_set = torch.exp(torch.einsum("sk,tk->ts", a, phi))  # (S_T, sets)
    out = torch.zeros(len(T), int(idx.max()) + 1, dtype=T.dtype, device=T.device)
    return out.index_add(1, idx, per_set)


def nuclear_masses(device: torch.device) -> torch.Tensor:
    """m_i = A_i m_u + Δ_i − Z_i m_e (MeV): nuclear masses from atomic mass excesses, (N,)."""
    return (MASS_NUMBER * M_U + MASS_EXCESS - CHARGE * M_E).to(device)


def reverse_coefficient(T: torch.Tensor, device: torch.device) -> torch.Tensor:
    r"""C_r(T) = Π_i (g_i n_{Q,i})^{R_{ri} − P_{ri}}, n_Q = (m T/2π)^{3/2} in cm⁻³, (S, Rx).

    Repeated i is summed in the logarithm."""
    R, P = stoichiometry(device)
    m = nuclear_masses(device)
    log_gnq = torch.log(SPIN_STATES.to(device)) + 1.5 * torch.log(torch.einsum("s,i->si", T, m) / (2 * math.pi))
    log_gnq = log_gnq - 3 * math.log(HBARC)  # (S, N)
    return torch.exp(torch.einsum("si,ri->sr", log_gnq, R - P))


def rate_coefficients(T: torch.Tensor, eta: torch.Tensor, tau_n: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""λ⁺_{sbr}, λ⁻_{sbr} in s⁻¹ on the temperature grid s for every η_b, (S, B, Rx) each."""
    dev = T.device
    R, P = stoichiometry(dev)
    Q = q_values(dev)
    n_b = baryon_density(T, eta)[:, :, None]  # (S, B, 1)
    n_in = R.sum(1)  # reactants per reaction
    identical = torch.exp(torch.lgamma(R + 1).sum(1))  # Π_i R_ri!: 2 for d + d (identical particles)
    sigv = reaclib_rates(T)[:, None, :] / N_AVOGADRO  # ⟨σv⟩ in cm³/s, (S, 1, Rx)
    lam_f = n_b ** (n_in - 1) * sigv / identical
    dN = n_in - P.sum(1)  # ΔN_r = Σ R − Σ P (1 for radiative captures)
    boltz = torch.exp(-Q / T[:, None])[:, None, :]  # (S, 1, Rx)
    lam_b = lam_f * n_b ** (-dN) * reverse_coefficient(T, dev)[:, None, :] * boltz
    lam_np, lam_pn = weak_rates(T, neutrino_temperature(T), tau_n)
    lam_f[..., 0] = lam_np[:, None]
    lam_b[..., 0] = lam_pn[:, None]
    return lam_f, lam_b


# ---------------------------------------------------------------- network
def monomials(Y: torch.Tensor, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""M_{br} = Π_i Y_{bi}^{E_ri} and its gradient G_{brk} = ∂M_{br}/∂Y_{bk}.   Y: (B, N), E: (Rx, N)."""
    base = Y[:, None, :] ** E  # (B, Rx, N)
    M = base.prod(-1)
    eye = torch.eye(Y.shape[1], dtype=torch.bool, device=Y.device)
    others = torch.where(eye, torch.ones_like(base)[:, :, :, None], base[:, :, None, :]).prod(-1)  # Π_{j≠k}
    G = E * Y[:, None, :] ** (E - 1).clamp(min=0) * others  # (B, Rx, N)
    return M, G


def network(Y: torch.Tensor, lam_f: torch.Tensor, lam_b: torch.Tensor, R: torch.Tensor, P: torch.Tensor):
    r"""dY_{bi}/dt = Σ_r (P−R)_{ri} (λ⁺_{br} M⁺_{br} − λ⁻_{br} M⁻_{br}) and its Jacobian J_{bik}: (B, N), (B, N, N)."""
    Mf, Gf = monomials(Y, R)
    Mb, Gb = monomials(Y, P)
    S = P - R
    f = torch.einsum("ri,br->bi", S, lam_f * Mf - lam_b * Mb)
    J = torch.einsum("ri,brk->bik", S, lam_f[..., None] * Gf - lam_b[..., None] * Gb)
    return f, J


# ---------------------------------------------------------------- observables
def observables(Y: torch.Tensor) -> dict:
    """Y_p = 4 Y_⁴He (helium mass fraction) and number ratios to hydrogen; t → ³He and ⁷Be → ⁷Li decay later."""
    H = Y[..., 1]
    return {
        "Yp": 4 * Y[..., 5],
        "DH": Y[..., 2] / H,
        "He3H": (Y[..., 4] + Y[..., 3]) / H,
        "LiH": (Y[..., 6] + Y[..., 7]) / H,
    }
