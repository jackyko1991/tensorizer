"""Concrete tensors: batches of worldlines in a space of `dims` dimensions, and the cited data.

A worldline is a path X(s) = (t, x_1, ..., x_{dims-1}) with a parameter s in [0, 2 pi]; t is the observer's time,
the other coordinates are space and any hidden dimensions. Smooth worldlines are random Fourier series (each
coordinate a sum of `modes` harmonics with amplitudes k^-roughness), random ones are Brownian paths. A closed
worldline returns to its start; an open one runs from one time to another.
"""

import csv
import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data"


def parameter(S: int, closed: bool, device) -> torch.Tensor:
    """(S,) s values: 2 pi j / S for a closed loop (the last sample joins the first), linspace(0, 2 pi) if open."""
    if closed:
        return 2 * math.pi * torch.arange(S, dtype=torch.float64, device=device) / S
    return torch.linspace(0, 2 * math.pi, S, dtype=torch.float64, device=device)


def mode_sigmas(modes: int, roughness: float, device) -> torch.Tensor:
    """(K,) amplitude of harmonic k = 1..K, sigma_k ~ k^-roughness, normalised so the variance sum sigma_k^2 = 1."""
    k = torch.arange(1, modes + 1, dtype=torch.float64, device=device)
    sig = k ** (-roughness)
    return sig / sig.norm()


def smooth_worldlines(B, S, modes, roughness, dims, closed, generator, device, drift: float = 1.5) -> torch.Tensor:
    """(B, S, dims) random Fourier worldlines X_bsi = sum_k sigma_k (a_bik cos ks + c_bik sin ks).

    Open worldlines also get a time drift +-drift (s / pi - 1), so they start and end at different times.
    """
    s = parameter(S, closed, device)  # (S,)
    sig = mode_sigmas(modes, roughness, device)  # (K,)
    k = torch.arange(1, modes + 1, dtype=torch.float64, device=device)
    coef = torch.randn(B, dims, modes, 2, generator=generator, dtype=torch.float64).to(device) * sig[:, None]
    basis = torch.stack([torch.cos(k[:, None] * s), torch.sin(k[:, None] * s)], -1)  # (K, S, 2)
    X = torch.einsum("bikc,ksc->bsi", coef, basis)  # repeated k (harmonic) and c (cos/sin) => summed
    if not closed:
        sign = torch.randint(0, 2, (B,), generator=generator).to(device) * 2 - 1
        X[..., 0] += drift * sign[:, None] * (s / math.pi - 1)
    return X


def random_worldlines(B, S, dims, closed, generator, device) -> torch.Tensor:
    """(B, S, dims) Brownian worldlines; a closed one is a Brownian bridge, scaled so that the mean variance is 1."""
    steps = torch.randn(B, S, dims, generator=generator, dtype=torch.float64).to(device) / math.sqrt(S)
    W = torch.cumsum(steps, 1)  # (B, S, dims)
    if closed:
        frac = torch.arange(1, S + 1, dtype=torch.float64, device=device) / S
        W = W - frac[None, :, None] * W[:, -1:, :]  # pin the end to the start
        return W * math.sqrt(6.0)  # bridge variance s (1 - s) averages 1/6
    return W * math.sqrt(2.0)  # walk variance s averages 1/2


def worldlines(kind, B, S, modes, roughness, dims, closed, generator, device) -> torch.Tensor:
    """kind 'smooth' (Fourier) or 'random' (Brownian)."""
    if kind == "smooth":
        return smooth_worldlines(B, S, modes, roughness, dims, closed, generator, device)
    return random_worldlines(B, S, dims, closed, generator, device)


def winding_worldlines(w: torch.Tensor, S: int, period: float, wiggle: float, generator, device) -> torch.Tensor:
    """(W, S + 1) unwrapped time of loops that wind w times round a compact time of length `period` (extension).

    t(s) = w period s / 2 pi + wiggle * (periodic Fourier zig-zag); the last sample is the first plus w period,
    so the loop closes on the time circle.
    """
    s = 2 * math.pi * torch.arange(S + 1, dtype=torch.float64, device=device) / S  # includes s = 2 pi
    k = torch.arange(1, 7, dtype=torch.float64, device=device)
    coef = torch.randn(len(w), 6, 2, generator=generator, dtype=torch.float64).to(device) / k[:, None]
    zig = torch.einsum("wkc,ksc->ws", coef, torch.stack([torch.cos(k[:, None] * s), torch.sin(k[:, None] * s)], -1))
    return w.to(device)[:, None] * period * s / (2 * math.pi) + wiggle * zig


def _rows(name: str) -> list[dict]:
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def positron_fraction(device=None) -> dict:
    """AMS-02 (PRL 122, 101101, 2019) positron fraction: E (GeV), bin edges, fraction, stat and syst errors (E,)."""
    rows = _rows("ams02_positron_fraction.csv")
    col = {k: torch.tensor([float(r[k]) for r in rows], dtype=torch.float64, device=device) for k in rows[0]}
    col["sigma"] = torch.sqrt(col["stat"] ** 2 + col["syst"] ** 2)
    return {"E": col["E_GeV"], "E_lo": col["E_lo_GeV"], "E_hi": col["E_hi_GeV"], "fraction": col["fraction"],
            "sigma": col["sigma"]}  # fmt: skip


def cited() -> dict:
    """Cited numbers (data/matter_antimatter.csv): quantity -> value."""
    return {r["quantity"]: float(r["value"]) for r in _rows("matter_antimatter.csv")}
