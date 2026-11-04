r"""Fit the low multipoles of the map, find their axes, and test the alignment against isotropic random skies.

The a_{ℓm} come from a weighted least-squares fit of all harmonics up to ℓ_fit (one lstsq), which on a full sky
equals the usual quadrature a_{ℓm} = Σ_p w_p T_p Y^*_{ℓm}(\hat n_p) but also removes aliasing from the neighbouring ℓ.
The Monte Carlo p-value is the share of isotropic skies whose two axes are at least as aligned as the observed ones.
"""

import torch

from . import equation, tensors


def fit_alm(dirs: torch.Tensor, T: torch.Tensor, w: torch.Tensor, lmax: int) -> dict[int, torch.Tensor]:
    r"""Least squares min Σ_p w_p (T_p − Σ r_{ℓm} Y^{real}_{ℓm,p})² → {ℓ: complex a_{ℓm} (2ℓ+1,)} for ℓ ≤ lmax."""
    Y = equation.real_harmonics(dirs, lmax)  # (P, (lmax+1)²)
    sw = w.sqrt()
    r = torch.linalg.lstsq(Y * sw[:, None], (T * sw)[:, None]).solution[:, 0]  # ((lmax+1)²,)
    return {ell: equation.real_to_complex(r[ell**2 : (ell + 1) ** 2], ell) for ell in range(lmax + 1)}


def axis_of(a: torch.Tensor, ell: int):
    """Preferred axis (…, 3), its planarity t = max ⟨(n·L)²⟩/ℓ² ∈ [0, 1] and the gap to the 2nd eigenvalue."""
    M = equation.dispersion_tensor(a, ell)
    vals, vecs = torch.linalg.eigh(M)
    return vecs[..., -1], vals[..., -1] / ell**2, vals[..., -1] - vals[..., -2]


def alignment_of(alm: dict[int, torch.Tensor]) -> dict:
    """Axes of the quadrupole and octopole, the angle between them and |cos|."""
    n2, t2, g2 = axis_of(alm[2], 2)
    n3, t3, g3 = axis_of(alm[3], 3)
    cos = equation.alignment(n2, n3).item()
    return {
        "axis2": n2,
        "axis3": n3,
        "cos": cos,
        "angle": torch.rad2deg(torch.acos(torch.tensor(min(cos, 1.0)))).item(),
        "t2": t2.item(),
        "t3": t3.item(),
        "gap2": g2.item(),
        "gap3": g3.item(),
    }


def monte_carlo(n_skies: int, generator: torch.Generator) -> torch.Tensor:
    """|n̂_2 · n̂_3| for n_skies isotropic Gaussian skies at once: (K,)."""
    a2 = tensors.isotropic_coefficients(n_skies, 2, generator)  # (K, 5)
    a3 = tensors.isotropic_coefficients(n_skies, 3, generator)  # (K, 7)
    return equation.alignment(axis_of(a2, 2)[0], axis_of(a3, 3)[0])


def p_value(cos_obs: float, cos_sims: torch.Tensor) -> float:
    """Share of simulated skies at least as aligned as observed (for isotropy this tends to 1 − |cos|)."""
    return (cos_sims >= cos_obs).double().mean().item()


def power(alm: dict[int, torch.Tensor]) -> dict[int, float]:
    r"""Band power D_ℓ = ℓ(ℓ+1) C_ℓ / 2π in μK², with C_ℓ = Σ_m |a_{ℓm}|² / (2ℓ+1)."""
    return {ell: ell * (ell + 1) * (a.abs() ** 2).mean().item() / (2 * torch.pi) for ell, a in alm.items()}
