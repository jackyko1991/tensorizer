"""Concrete tensors: constants, the (n, R) grid and the cited bounds (data/gravity_bounds.csv).

Particle physics in GeV, lengths in metres; 1 GeV^-1 = hbar c / GeV = 1.97e-16 m.
"""

import csv
import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data"
M_PL_GEV = 1.220890e19  # Planck mass sqrt(hbar c / G) (CODATA 2018)
HBARC_M = 1.973269804e-16  # hbar c in GeV m
L_PLANCK_M = 1.616255e-35  # Planck length (CODATA 2018)


def grid(n_max: int, R_points: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """n = 0..n_max extra dimensions (N,) and radii R (R,) from the Planck length to 1 m, log spaced."""
    n = torch.arange(0, n_max + 1, dtype=torch.float64, device=device)
    R = torch.logspace(math.log10(L_PLANCK_M), 0, R_points, dtype=torch.float64, device=device)
    return n, R


def bounds() -> list[dict]:
    """Cited bounds: bound, source, year, n (None = any n >= 1), kind (lambda_max | R_max | M_D_min), value."""
    with open(DATA / "gravity_bounds.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(r for r in f if not r.startswith("#")))
    for r in rows:
        r["n"] = None if r["n"] == "any" else float(r["n"])
        r["value"] = float(r["value"])
    return rows


STAGES = ["torsion", "lhc", "sn1987a", "neutron_star"]  # the order in which the map fills in
LABELS = {"torsion": "torsion balances (Kapner 2007, Tan 2016, Lee 2020)", "lhc": "LHC monojet (ATLAS 2021)",
          "sn1987a": "SN1987A cooling (Hanhart 2001)",
          "neutron_star": "neutron-star heating (Hannestad & Raffelt 2003)"}  # fmt: skip
