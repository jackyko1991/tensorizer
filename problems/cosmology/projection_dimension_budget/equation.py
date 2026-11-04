"""Dimension budget -> tensors: Gauss's law in 3 + n dimensions, the ADD relation, the experimental exclusions.

Shapes in comments. Every bound becomes a boolean mask over the (n, R) grid, broadcast from n (N, 1) and R (1, R);
the summed index is the image (lattice) index k in the force of a mass on R^3 x T^n.
"""

import math

import torch

from . import tensors

# ---------- gravity in 3 + n space dimensions ----------


def sphere_area(k: int) -> float:
    """Area of the unit sphere S^k in R^(k+1): 2 pi^((k+1)/2) / Gamma((k+1)/2)."""
    return 2 * math.pi ** ((k + 1) / 2) / math.gamma((k + 1) / 2)


def potential(x: torch.Tensor, n: int) -> torch.Tensor:
    """phi = -G_D M / ((n+1) |x|^(n+1)) in 3 + n space dimensions (G_D M = 1): its gradient is r_hat / r^(n+2)."""
    return -1 / ((n + 1) * x.norm(dim=-1) ** (n + 1))


def gauss_flux(n: int, radii: torch.Tensor, points: int, generator) -> torch.Tensor:
    """(R,) flux of grad phi through spheres S^(n+2) of radius r: |S^(n+2)| r^(n+2) <grad phi . r_hat> (autograd)."""
    u = torch.randn(points, 3 + n, generator=generator, dtype=torch.float64)
    u = u / u.norm(dim=1, keepdim=True)  # (P, 3+n) random outward normals
    x = (radii[:, None, None] * u).requires_grad_(True)  # (R, P, 3+n)
    (grad,) = torch.autograd.grad(potential(x, n).sum(), x)
    radial = torch.einsum("rpi,pi->rp", grad, u)  # repeated i (coordinate) => summed: the outward component
    return sphere_area(n + 2) * radii ** (n + 2) * radial.mean(1)


def image_force(r: torch.Tensor, n: int, K: int) -> torch.Tensor:
    """(R,) pull at distance r along our 3 dimensions from a mass on R^3 x T^n (circles of radius R = 1):
    the mass and its images at 2 pi k, k in [-K, K]^n, each pulling with 1 / dist^(n+2) times r / dist."""
    axis = torch.arange(-K, K + 1, dtype=torch.float64, device=r.device)
    k = torch.cartesian_prod(*([axis] * n)).reshape(-1, n)  # ((2K+1)^n, n)
    rho2 = (2 * math.pi) ** 2 * (k**2).sum(1)  # (I,) squared distance of image I in the compact directions
    return (r[:, None] / (r[:, None] ** 2 + rho2) ** ((n + 3) / 2)).sum(1)  # summed over images


def g4_over_gd(n: int) -> float:
    """G_4 / G_D = |S^(n+2)| / (4 pi (2 pi R)^n): Gauss's law through S^2 x T^n far from the mass (R = 1)."""
    return sphere_area(n + 2) / (4 * math.pi * (2 * math.pi) ** n)


def torus_yukawa_alpha(n: torch.Tensor) -> torch.Tensor:
    """Strength of the first Kaluza-Klein graviton level on an n-torus: 2n modes x 4/3 = 8n/3 (PDG: 16/3 for n = 2)."""
    return 8 * n / 3


# ---------- ADD: M_Pl^2 = 8 pi R^n M_D^(n+2) ----------


def add_md_tev(n: torch.Tensor, R_m: torch.Tensor) -> torch.Tensor:
    """Fundamental scale M_D (TeV) for n dimensions of radius R (metres), GRW / PDG convention."""
    R = torch.as_tensor(R_m, dtype=torch.float64) / tensors.HBARC_M  # GeV^-1
    log_m = (2 * math.log(tensors.M_PL_GEV) - math.log(8 * math.pi) - n * torch.log(R)) / (n + 2)
    return torch.exp(log_m) / 1e3


def add_radius_m(n: torch.Tensor, M_tev: torch.Tensor) -> torch.Tensor:
    """Radius R (metres) that n dimensions need for M_Pl^2 = 8 pi R^n M_D^(n+2) (broadcast over n and M_D)."""
    M = torch.as_tensor(M_tev, dtype=torch.float64) * 1e3
    log_r = (2 * math.log(tensors.M_PL_GEV) - math.log(8 * math.pi) - (n + 2) * torch.log(M)) / n
    return torch.exp(log_r) * tensors.HBARC_M


# ---------- the experimental bounds as masks on the (n, R) grid ----------


def _limit(n: torch.Tensor, row: dict) -> torch.Tensor:
    """Largest allowed R (metres) that one cited bound sets for each n (inf where it says nothing)."""
    inf = torch.full_like(n, math.inf)
    if row["kind"] == "lambda_max":  # any n >= 1: alpha = 8n/3 >= 1, lambda = R
        return torch.where(n >= 1, torch.full_like(n, row["value"]), inf)
    on = n == row["n"]
    if row["kind"] == "R_max":
        return torch.where(on, torch.full_like(n, row["value"]), inf)
    R = add_radius_m(torch.where(on, n, torch.ones_like(n)), torch.tensor(row["value"]))  # M_D > M_lim <=> R < R(M_lim)
    return torch.where(on, R, inf)


def exclusion_masks(n: torch.Tensor, R: torch.Tensor, rows: list[dict]) -> dict:
    """{bound name: (N, R) mask of excluded points}; n (N, 1) and R (1, R) broadcast."""
    masks = {}
    for row in rows:  # a handful of cited numbers; each one is applied to the whole grid at once
        m = R > _limit(n.to(torch.float64), row)
        masks[row["bound"]] = masks.get(row["bound"], torch.zeros_like(m)) | m
    return masks


def excluded(n: torch.Tensor, R: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    """(N, R) excluded by any bound."""
    m = exclusion_masks(n, R, rows)
    return torch.stack(torch.broadcast_tensors(*m.values())).any(0)


def r_max(n: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    """(N,) largest radius compatible with every bound (inf when nothing applies, e.g. n = 0)."""
    return torch.stack([_limit(n.to(torch.float64), row) for row in rows]).amin(0)


# ---------- the quantum side (step 1) ----------


def race_dims(d: int, K: int) -> int:
    """Projection race with the Born rule: d observed dimensions plus one hidden complex dimension per outcome."""
    return d + K


def hilbert_log2_dim(N: torch.Tensor) -> torch.Tensor:
    """log2 of the dimension N entangled two-level systems need: D >= 2^N."""
    return N.clone()


def spatial_reading_extra_dims(N: torch.Tensor) -> torch.Tensor:
    """If every Hilbert dimension were a spatial direction: 2^N complex = 2^(N+1) real coordinates, minus our 3."""
    return 2 ** (N + 1) - 3
