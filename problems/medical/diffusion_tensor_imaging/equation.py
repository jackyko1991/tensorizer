r"""Physics → tensor translation.

Stejskal–Tanner: a diffusion-weighting gradient along the unit vector \hat g with strength b attenuates the
signal of voxel v by the diffusion coefficient along \hat g, a quadratic form of the diffusion tensor:

    S_{vg} = S_{0v} \exp(-b\, \hat g_{gi} D_{vij} \hat g_{gj})          (i, j summed; batched over v and g)

Taking logs makes it linear in the 6 unique components d_{vm} of the symmetric tensor:

    \ln(S_{vg}/S_{0v}) = -b\, B_{gm} d_{vm}        (m summed)  →  least squares for every voxel at once

The eigen-decomposition D_{vij} = \sum_a \lambda_{va} e_{vai} e_{vaj} gives the fibre direction (largest λ) and

    MD = \lambda_a / 3 = D_{ii} / 3,      FA = \sqrt{3/2}\, \lVert \lambda - MD \rVert / \lVert \lambda \rVert
"""

import math

import torch


def signal(D: torch.Tensor, g: torch.Tensor, b: float, S0: float = 1.0) -> torch.Tensor:
    r"""S_{vg} = S_0 \exp(-b \hat g_{gi} D_{vij} \hat g_{gj}).   D: (V, 3, 3), g: (G, 3) → (V, G)"""
    adc = torch.einsum("gi,vij,gj->vg", g, D, g)  # i, j summed: apparent diffusion coefficient along each ĝ
    return S0 * torch.exp(-b * adc)


def fit_tensor(S: torch.Tensor, S0: torch.Tensor, B: torch.Tensor, b: float) -> torch.Tensor:
    r"""Least squares for -b B_{gm} d_{vm} = \ln(S_{vg}/S_{0v}), all voxels in one call.   → D (V, 3, 3)"""
    y = torch.log(S.clamp(min=1e-6) / S0.clamp(min=1e-6)[:, None])  # (V, G)
    d = torch.linalg.lstsq(-b * B, y.T).solution.T  # (G, 6) \ (G, V) → (6, V) → (V, 6)
    dxx, dyy, dzz, dxy, dxz, dyz = d.unbind(-1)
    rows = [torch.stack(r, -1) for r in ((dxx, dxy, dxz), (dxy, dyy, dyz), (dxz, dyz, dzz))]
    return torch.stack(rows, dim=-2)  # (V, 3, 3) symmetric


def invariants(D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Eigenvalues λ (V, 3) ascending, principal eigenvector e_1 (V, 3), MD (V,) and FA (V,)."""
    lam, vec = torch.linalg.eigh(D)  # symmetric → real eigenvalues, orthonormal eigenvectors (columns)
    md = torch.einsum("vii->v", D) / 3  # trace / 3 = mean of the eigenvalues
    dev = lam - md[:, None]
    fa = math.sqrt(1.5) * dev.norm(dim=-1) / lam.norm(dim=-1).clamp(min=1e-12)
    return lam, vec[..., -1], md, fa
