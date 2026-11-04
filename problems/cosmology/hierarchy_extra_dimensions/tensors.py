"""Concrete tensors: constants, mode vectors of the torus and their shell counts, the measured limits.

Units in the potential: compact radius R = 1 (circumference 2 pi R), G_D M = 1. Particle physics in GeV.
"""

import csv
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data"
M_PL_GEV = 1.220890e19  # Planck mass sqrt(hbar c / G) (CODATA 2018)
M_PL_REDUCED_GEV = 2.435e18  # reduced Planck mass M_Pl / sqrt(8 pi)
HBARC_M = 1.973269804e-16  # hbar c in GeV m: 1 GeV^-1 = 1.97e-16 m
G_FERMI = 1.1663788e-5  # Fermi constant G_F / (hbar c)^3 in GeV^-2 (CODATA 2018)
V_HIGGS = 246.22  # electroweak scale (Higgs vacuum value) in GeV


def mode_vectors(n: int, K: int, device) -> torch.Tensor:
    """(N, n) every Kaluza-Klein mode vector k in the cube [-K, K]^n (N = (2K+1)^n)."""
    axis = torch.arange(-K, K + 1, dtype=torch.float64, device=device)
    return torch.cartesian_prod(*([axis] * n)).reshape(-1, n)


def shell_counts(n: int, m_max: int, device) -> torch.Tensor:
    """(m_max+1,) r_n(m) = number of k in Z^n with |k|^2 = m: the n-th power of the theta series 1 + 2q + 2q^4 + ...

    Polynomials multiply by convolution, done here with an FFT: theta(q)^n without listing (2K+1)^n vectors.
    """
    j = torch.arange(0, int(m_max**0.5) + 1, device=device)
    theta = torch.zeros(m_max + 1, dtype=torch.float64, device=device)
    theta[j**2] = 2.0
    theta[0] = 1.0
    size = 1 << (n * (m_max + 1)).bit_length()
    power = torch.fft.irfft(torch.fft.rfft(theta, size) ** n, size)[: m_max + 1]
    return torch.round(power)


def torsion_limits() -> list[dict]:
    """Cited torsion-balance limits: |alpha| = 1 excluded above lambda (95% CL)."""
    with open(DATA / "torsion_balance_limits.csv", encoding="utf-8") as f:
        return [row for row in csv.DictReader(r for r in f if not r.startswith("#"))]


def lhc_limits() -> tuple[torch.Tensor, torch.Tensor]:
    """ATLAS 2021 monojet lower limits on M_D (TeV) for n = 2..6 extra dimensions."""
    with open(DATA / "lhc_add_limits.csv", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(r for r in f if not r.startswith("#"))]
    n = torch.tensor([float(r["n"]) for r in rows], dtype=torch.float64)
    m = torch.tensor([float(r["M_D_min_TeV"]) for r in rows], dtype=torch.float64)
    return n, m
