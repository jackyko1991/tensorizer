r"""Physics → tensor translation.

A diatomic molecule vibrates (anharmonic oscillator) and rotates (a rotor that stretches a little when it spins).
In wavenumbers (cm⁻¹) its term values are

    G(v)   = ω_e (v + ½) − ω_e x_e (v + ½)²
    F_v(J) = B_v J(J+1) − D_e J²(J+1)²,        B_v = B_e − α_e (v + ½)

The fundamental band v = 0 → 1 obeys ΔJ = ±1 (no Q branch for HCl), so for isotopologue i and lower level J

    R branch (J → J+1):   ν^R_{iJ} = ν₀_i + F_{1,i}(J+1) − F_{0,i}(J)
    P branch (J → J−1):   ν^P_{iJ} = ν₀_i + F_{1,i}(J−1) − F_{0,i}(J),      ν₀ = G(1) − G(0) = ω_e − 2ω_e x_e

Line strengths follow the Boltzmann population of the lower level times the Hönl–London factor:

    I^R_{iJ} = a_i (J+1) e^{−hc F_{0,i}(J)/kT} / Q_i,       I^P_{iJ} = a_i J e^{−hc F_{0,i}(J)/kT} / Q_i

and the spectrum is the sum of a line shape over every line ℓ = (i, branch, J):

    S_{tg} = Σ_ℓ I_{tℓ} g_{ℓg}            (ℓ summed; t = temperature frame, g = wavenumber grid point)
"""

import torch

HC_K = 1.438777  # second radiation constant hc/k in cm K


def vib_term(v: float, we: torch.Tensor, wexe: torch.Tensor) -> torch.Tensor:
    """G(v) = ω_e(v+½) − ω_e x_e(v+½)²   (I,) cm⁻¹."""
    return we * (v + 0.5) - wexe * (v + 0.5) ** 2


def band_origin(we: torch.Tensor, wexe: torch.Tensor) -> torch.Tensor:
    """ν₀ = G(1) − G(0) = ω_e − 2 ω_e x_e   (I,)."""
    return vib_term(1, we, wexe) - vib_term(0, we, wexe)


def rot_constant(v: float, Be: torch.Tensor, ae: torch.Tensor) -> torch.Tensor:
    """B_v = B_e − α_e(v + ½): a vibrating bond is longer on average, so it rotates more slowly   (I,)."""
    return Be - ae * (v + 0.5)


def rot_term(J: torch.Tensor, B: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """F_iJ = B_i J(J+1) − D_i J²(J+1)².   J: (J,), B, D: (I,) → (I, J).   No index summed: outer products."""
    x = J * (J + 1)
    return torch.einsum("i,j->ij", B, x) - torch.einsum("i,j->ij", D, x**2)


def branches(J: torch.Tensor, c: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Line positions ν^R_{iJ}, ν^P_{iJ}   (I, J) cm⁻¹, for every isotopologue i and lower level J at once.

    P(0) does not exist (J = −1); its position is meaningless but its strength below is exactly zero.
    """
    nu0 = band_origin(c["we"], c["wexe"])[:, None]  # (I, 1)
    B0, B1 = rot_constant(0, c["Be"], c["ae"]), rot_constant(1, c["Be"], c["ae"])
    F0 = rot_term(J, B0, c["De"])  # lower level, v = 0
    R = nu0 + rot_term(J + 1, B1, c["De"]) - F0
    P = nu0 + rot_term(J - 1, B1, c["De"]) - F0
    return R, P


def populations(J: torch.Tensor, B0: torch.Tensor, D: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Boltzmann fraction of molecules in level J: p_tiJ = (2J+1) e^{−hcF₀(J)/kT_t} / Q_ti   (T, I, J)."""
    F0 = rot_term(J, B0, D)  # (I, J)
    w = (2 * J + 1) * torch.exp(-HC_K * F0[None] / T[:, None, None])
    return w / w.sum(-1, keepdim=True)  # Q = Σ_J (2J+1) e^{−…}: the rotational partition function


def intensities(J: torch.Tensor, p: torch.Tensor, abundance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """I^R_tiJ = a_i p_tiJ (J+1)/(2J+1),   I^P_tiJ = a_i p_tiJ J/(2J+1)   (T, I, J) each."""
    base = torch.einsum("i,tij->tij", abundance, p) / (2 * J + 1)
    return base * (J + 1), base * J


def lorentzian(nu_lines: torch.Tensor, grid: torch.Tensor, fwhm: float) -> torch.Tensor:
    """g_ℓg = (γ/π) / ((ν̃_g − ν̃_ℓ)² + γ²), γ = FWHM/2: the shape of one line   (L, G)."""
    gamma = 0.5 * fwhm
    return (gamma / torch.pi) / ((grid[None, :] - nu_lines[:, None]) ** 2 + gamma**2)


def spectrum(I_lines: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """S_tg = Σ_ℓ I_tℓ g_ℓg   (T, G).   Repeated ℓ ⇒ summed over every line."""
    return torch.einsum("tl,lg->tg", I_lines, g)


def combination_differences(R: torch.Tensor, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differences that share an upper (or a lower) level, so the other state cancels.  J = 1 … J_max − 1.

    Δ₂F″(J) = R(J−1) − P(J+1) = F₀(J+1) − F₀(J−1)    (same upper level J: depends only on v = 0)
    Δ₂F′(J) = R(J)   − P(J)   = F₁(J+1) − F₁(J−1)    (same lower level J: depends only on v = 1)
    Both equal (4J+2) [B_v − 2 D (J² + J + 1)].       R, P: (I, J) → (I, J−2) each.
    """
    lower = R[:, :-2] - P[:, 2:]
    upper = R[:, 1:-1] - P[:, 1:-1]
    return lower, upper


def morse_depth(we: torch.Tensor, wexe: torch.Tensor) -> torch.Tensor:
    """Well depth of the equivalent Morse potential, D_e = ω_e² / (4 ω_e x_e)   (cm⁻¹)."""
    return we**2 / (4 * wexe)
