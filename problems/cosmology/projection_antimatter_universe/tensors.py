"""Concrete tensors: the cited numbers, worldlines in a space of `dims` dimensions, and the photon-energy grid.

A worldline is a path X(s) = (t, x_1, ..., x_{dims-1}); only its time coordinate decides how a slice reads it.
For the CPT mirror the time coordinate is conformal time eta on a circle: eta = 0 is the Big Bang, 0 < eta < 1 our
side, -1 < eta < 0 the mirror, and eta = +-1 the future boundaries (glued together: an extra assumption).
"""

import csv
import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data"


def cited() -> dict:
    """data/antimatter_criteria.csv: quantity -> value (see the file header for every source)."""
    with open(DATA / "antimatter_criteria.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(r for r in f if not r.startswith("#")))
    return {r["quantity"]: float(r["value"]) for r in rows}


def energies(device=None) -> torch.Tensor:
    """(E,) photon energies today, 0.1 MeV to 1 TeV, log spaced."""
    return torch.logspace(-1, 6, 140, dtype=torch.float64, device=device)


def closed_loops(B: int, S: int, modes: int, dims: int, generator) -> torch.Tensor:
    """(B, S, dims) random closed Fourier worldlines X_bsi = sum_k (a_bik cos ks + c_bik sin ks) / k, each scaled so
    that its time coordinate stays inside (-0.95, 0.95)."""
    s = 2 * math.pi * torch.arange(S, dtype=torch.float64) / S
    k = torch.arange(1, modes + 1, dtype=torch.float64)
    coef = torch.randn(B, dims, modes, 2, generator=generator, dtype=torch.float64) / k[:, None]
    basis = torch.stack([torch.cos(k[:, None] * s), torch.sin(k[:, None] * s)], -1)  # (K, S, 2)
    X = torch.einsum("bikc,ksc->bsi", coef, basis)  # repeated k (harmonic) and c (cos/sin) => summed
    return 0.95 * X / X[..., :1].abs().amax(1, keepdim=True)


def winding_conformal_loop(w: torch.Tensor, S: int, wiggle: float, generator) -> torch.Tensor:
    """(W, S + 1) unwrapped conformal time of loops that wind w times round the eta circle of length 2:
    eta(s) = -1 + 2 w s / 2 pi + wiggle * (periodic zig-zag); the last sample is the first plus 2 w."""
    s = 2 * math.pi * torch.arange(S + 1, dtype=torch.float64) / S
    k = torch.arange(1, 6, dtype=torch.float64)
    coef = torch.randn(len(w), 5, 2, generator=generator, dtype=torch.float64) / k[:, None]
    basis = torch.stack([torch.cos(k[:, None] * s), torch.sin(k[:, None] * s)], -1)
    zig = torch.einsum("wkc,ksc->ws", coef, basis)  # periodic, so the loop closes on the circle
    return -1.0 + w.double()[:, None] * 2.0 * s / (2 * math.pi) + wiggle * zig
