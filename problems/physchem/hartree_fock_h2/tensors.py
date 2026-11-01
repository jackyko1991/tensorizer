"""Build the concrete tensors: the molecule, its STO-3G basis and the batch of bond lengths.

Atomic units throughout: lengths in bohr (a₀ = 0.529 Å), energies in hartree (Eₕ = 27.2 eV).
"""

import math

import torch

F64 = torch.float64

# STO-3G fit of a Slater 1s function with exponent ζ = 1 (the values in Szabo & Ostlund's Appendix B program).
# For another ζ the Gaussian exponents scale as α → α ζ²; the contraction coefficients d stay the same.
STO3G_ALPHA = (0.109818, 0.405771, 2.22766)
STO3G_D = (0.444635, 0.535328, 0.154329)

# name: (nuclear charges Z_A, Slater exponents ζ_A of the 1s function on each atom, electrons on each separated atom)
MOLECULES = {
    "H2": ((1.0, 1.0), (1.24, 1.24), (1, 1)),  # H₂ → H + H
    "HeH+": ((2.0, 1.0), (2.0925, 1.24), (2, 0)),  # HeH⁺ → He + H⁺
}
N_ELECTRONS = 2


def molecule(name: str, device: torch.device):
    """Z (A,) nuclear charges, zeta (M,) one 1s basis function per atom (M = A = 2), electrons per separated atom."""
    Z, zeta, n_atom = MOLECULES[name]
    return torch.tensor(Z, dtype=F64, device=device), torch.tensor(zeta, dtype=F64, device=device), n_atom


def sto3g(zeta: torch.Tensor):
    """Contracted Gaussians φ_μ(r) = Σ_p c_μp exp(-α_μp |r - A_μ|²).

    alpha (M, K) exponents α_μp = α_p ζ_μ², coef (M, K) = d_p (2α_μp/π)^{3/4}, the primitive normalisation
    folded into the contraction coefficient. K = 3 primitives per function.
    """
    a1 = torch.tensor(STO3G_ALPHA, dtype=F64, device=zeta.device)
    d = torch.tensor(STO3G_D, dtype=F64, device=zeta.device)
    alpha = a1[None, :] * zeta[:, None] ** 2
    coef = d[None, :] * (2 * alpha / math.pi) ** 0.75
    return alpha, coef


def bond_lengths(r_min: float, r_max: float, n: int, device: torch.device) -> torch.Tensor:
    """R_b, shape (B,): the bond lengths of the scan, in bohr."""
    return torch.linspace(r_min, r_max, n, dtype=F64, device=device)


def geometry(R: torch.Tensor) -> torch.Tensor:
    """Nuclear positions (B, A, 3): atom 0 at the origin, atom 1 at (0, 0, R_b). The basis sits on the nuclei."""
    pos = torch.zeros(len(R), 2, 3, dtype=F64, device=R.device)
    pos[:, 1, 2] = R
    return pos


def axis_points(z_min: float, z_max: float, n: int, device: torch.device) -> torch.Tensor:
    """Points (Z, 3) on the bond axis, for drawing the orbitals."""
    pts = torch.zeros(n, 3, dtype=F64, device=device)
    pts[:, 2] = torch.linspace(z_min, z_max, n, dtype=F64, device=device)
    return pts
