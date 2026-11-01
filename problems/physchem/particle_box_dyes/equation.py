r"""Physics → tensor translation.

Free-electron model: the π electrons of a conjugated chain move freely along it, as particles in a
one-dimensional box of length L. For molecule b and level n

    E_{bn} = h² n² / (8 m_e L_b²)

N π electrons fill the lowest N/2 levels two by two (Pauli), so the HOMO is n = N/2 and the LUMO n = N/2 + 1.
The longest-wavelength absorption lifts one electron across that gap:

    ΔE_b = E_{b, N/2+1} − E_{b, N/2} = h² (N_b + 1) / (8 m_e L_b²),      λ_b = h c / ΔE_b

Numerically the same box is a matrix: on M interior grid points

    H_{bij} = (ħ²/2m_e) (2 δ_{ij} − δ_{i,j+1} − δ_{i+1,j}) / dx_b²  +  V_{bi} δ_{ij}
    H_{bij} ψ_{bjn} = E_{bn} ψ_{bin}          (j summed)

 Units: eV and nm.
"""

import math

import torch

# SI constants → eV and nm
H_PLANCK = 6.62607015e-34  # J s
M_E = 9.1093837015e-31  # kg
EV = 1.602176634e-19  # J
H2_8M = H_PLANCK**2 / (8 * M_E) / EV * 1e18  # h²/(8 m_e) = 0.376 eV nm²
HBAR2_2M = H2_8M / math.pi**2  # ħ²/(2 m_e) = h²/(8 π² m_e) = 0.0381 eV nm²
HC = 1239.84198  # h c in eV nm


def energy_levels(n: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """E_bn = h² n_n² / (8 m L_b²).   n: (n,), L: (B,) → (B, n) eV.   No index summed: an outer product."""
    return H2_8M * torch.einsum("n,b->bn", n**2, 1.0 / L**2)


def homo_lumo_gap(n_electrons: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """ΔE_b = h² (N_b + 1) / (8 m L_b²)   (B,) eV — the difference of (N/2+1)² and (N/2)²."""
    return H2_8M * (n_electrons + 1) / L**2


def wavelength(delta_E: torch.Tensor) -> torch.Tensor:
    """λ = h c / ΔE   (nm)."""
    return HC / delta_E


def alternation_potential(x: torch.Tensor, V0: float, bond: float, end: float) -> torch.Tensor:
    """V_bi = −(V0/2) cos(π (x_bi − x₀)/l): low on double bonds, high on single bonds   (B, M) eV.

    x₀ = (end + ½) l is the middle of the first bond. Its period, 2l, is one double + one single bond.
    V0 = 0 is the plain box; a polyene's bond alternation is better described with V0 ≈ 1–2 eV.
    """
    x0 = (end + 0.5) * bond
    return -0.5 * V0 * torch.cos(math.pi * (x - x0) / bond)


def hamiltonian(dx: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """H_bij = (ħ²/2m)(2δ_ij − δ_i,j+1 − δ_i+1,j)/dx_b² + V_bi δ_ij.   dx: (B,), V: (B, M) → (B, M, M)."""
    M = V.shape[1]
    eye = torch.eye(M, device=V.device, dtype=V.dtype)
    second_diff = 2 * eye - torch.diag(torch.ones(M - 1, device=V.device, dtype=V.dtype), 1)
    second_diff = second_diff - torch.diag(torch.ones(M - 1, device=V.device, dtype=V.dtype), -1)  # −dx² d²/dx²
    kinetic = HBAR2_2M * torch.einsum("b,ij->bij", 1.0 / dx**2, second_diff)
    return kinetic + torch.diag_embed(V)


def absorption_band(lam: torch.Tensor, lam_max: torch.Tensor, width: float = 40.0) -> torch.Tensor:
    """A_bw = exp(−½ ((λ_w − λmax_b)/σ)²): a Gaussian absorption band of each dye   (B, W)."""
    return torch.exp(-0.5 * ((lam[None, :] - lam_max[:, None]) / width) ** 2)


def wavelength_rgb(lam: torch.Tensor) -> torch.Tensor:
    """An approximate RGB colour (W, 3) for each visible wavelength (piecewise-linear rainbow, dimmed at the ends)."""
    r = torch.where(lam < 440, (440 - lam) / 60, torch.where(lam < 510, 0.0 * lam, (lam - 510) / 70)).clamp(0, 1)
    g = torch.where(lam < 490, (lam - 440) / 50, torch.where(lam < 580, 1.0 + 0.0 * lam, (645 - lam) / 65)).clamp(0, 1)
    b = torch.where(lam < 490, 1.0 + 0.0 * lam, (510 - lam) / 20).clamp(0, 1)
    fade = torch.where(
        lam < 420, 0.3 + 0.7 * (lam - 380) / 40, torch.where(lam > 700, 0.3 + 0.7 * (780 - lam) / 80, 1.0)
    )
    return torch.stack([r, g, b], dim=-1) * fade.clamp(0, 1)[:, None]


def perceived_colour(A: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Colour of the light that is NOT absorbed:  c_bk = Σ_w (1 − A_bw) rgb_wk / Σ_w rgb_wk   (B, 3).

    Repeated w ⇒ summed over the visible spectrum; white light minus the absorbed band gives the dye's colour.
    """
    transmitted = torch.einsum("bw,wk->bk", 1.0 - A, rgb)
    return (transmitted / rgb.sum(0)).clamp(0, 1)
