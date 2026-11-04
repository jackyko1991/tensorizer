"""Horizons for a batch of models at once: every row m of the (M, A) tensors is one universe."""

import math

import torch

from .equation import (
    comoving_hubble_radius,
    horizon_angle,
    hubble,
    hubble_with_inflation,
    particle_horizon,
    patch_count,
    recombination_index,
    speed_of_light,
    vsl_extra_horizon,
)
from .tensors import C0, STEP_WIDTH


def horizons(x: torch.Tensor, model: str, c_ratio=None, efolds=None, a_star: float = 1e-28) -> dict:
    """x = ln a grid (A,). model: "lcdm" (one row), "vsl" (one row per c_ratio), "inflation" (one per efolds).

    Returns c (M, A), H (M, A), the comoving Hubble radius r_H (M, A) and the particle horizon χ_p (M, A), in Mpc.
    """
    a = torch.exp(x)
    if model == "vsl":
        c = speed_of_light(a, c_ratio, a_star)
        H = hubble(a)[None, :].expand_as(c)
    elif model == "inflation":
        H = hubble_with_inflation(a, efolds, a_star)
        c = torch.full_like(H, C0)
    else:
        H = hubble(a)[None, :]
        c = torch.full_like(H, C0)
    r_H = comoving_hubble_radius(a, c, H)
    return {"c": c, "H": H, "hubble_radius": r_H, "chi": particle_horizon(x, r_H)}


def sky_statistics(x: torch.Tensor, h: dict) -> dict:
    """How much of the last-scattering sphere one causal patch covers, for every row.   → each (M,)"""
    k = recombination_index(x)
    chi_rec = h["chi"][:, k]
    distance = h["chi"][:, -1] - chi_rec  # comoving distance to last scattering (c = c₀ after recombination)
    theta = horizon_angle(chi_rec, distance)
    return {"chi_rec": chi_rec, "distance": distance, "theta": theta, "patches": patch_count(theta),
            "solved": chi_rec >= distance}  # fmt: skip


def _gap(x: torch.Tensor) -> tuple[float, float]:
    """(D − χ_p(a_rec), D) for standard ΛCDM: how much horizon is missing."""
    s = sky_statistics(x, horizons(x, "lcdm"))
    return float(s["distance"][0] - s["chi_rec"][0]), float(s["distance"][0])


def vsl_minimum_ratio(x: torch.Tensor, a_star: float) -> float:
    """Smallest c_early/c₀ for which the antipodal past light cones of the CMB overlap (closed-form extra horizon)."""
    missing, _ = _gap(x)
    per_unit = float(vsl_extra_horizon(torch.tensor([2.0], dtype=x.dtype), a_star, STEP_WIDTH))  # (r − 1) = 1
    return 1 + missing / per_unit


def inflation_minimum_efolds(x: torch.Tensor, a_star: float) -> float:
    """Smallest N with c₀/(a_* H_*) (e^N − 1) ≥ missing horizon."""
    missing, _ = _gap(x)
    h_star = float(hubble(torch.tensor(a_star, dtype=x.dtype)))
    return math.log(1 + missing * a_star * h_star / C0)
