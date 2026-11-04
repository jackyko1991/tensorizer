"""Concrete tensors: the GW170817 distance samples, the electromagnetic distances, the EHT numbers, Pantheon+.

data/gw170817_distance.csv: 8078 posterior samples of the luminosity distance (GWTC-1, low-spin prior, GWOSC,
CC BY 4.0). The analysis used a d^2 (uniform in volume) prior, so each sample gets the weight 1/d^2 to turn the
posterior into the likelihood of d_GW. data/em_distances.csv and data/eht_shadows.csv hold cited published numbers;
data/pantheon_plus.csv is a copy of the supernova table of cosmology/cosmological_constant_fit.
"""

import csv
from pathlib import Path

import torch

from . import equation

DATA = Path(__file__).parent / "data"
F64 = torch.float64


def _rows(name: str) -> list[dict]:
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def load_gw_samples(device=None) -> dict:
    """d (S,) posterior samples in Mpc, w (S,) = 1/d^2 (prior divided out), h the kernel width (Scott's rule)."""
    d = torch.tensor([float(r["d_L_Mpc"]) for r in _rows("gw170817_distance.csv")], dtype=F64, device=device)
    w = d**-2
    return {"d": d, "w": w, "h": equation.scott_bandwidth(d, w), "label": "GW170817 (GWTC-1)"}


def em_distance(key: str = "sbf") -> dict:
    """The electromagnetic distance to NGC 4993: sbf (Cantiello 2018), shoes or planck (Hubble flow, Pardo 2018)."""
    for r in _rows("em_distances.csv"):
        if r["key"] == key:
            return {"d": float(r["d_Mpc"]), "sigma": float(r["sigma_Mpc"]), "label": r["method"], "key": key}
    raise ValueError(f"unknown EM distance {key!r}")


def load_eht() -> list[dict]:
    """EHT rows (M87*, Sgr A* with Keck and VLTI priors) with numbers as floats (None where empty)."""
    out = []
    for r in _rows("eht_shadows.csv"):
        out.append({k: (float(v) if v and k not in ("key", "object", "prior") else v or None) for k, v in r.items()})
    return out


def load_supernovae(device=None):
    """z, μ, σ_μ (N,) from the bundled Pantheon+ table, sorted by redshift."""
    rows = _rows("pantheon_plus.csv")
    t = torch.tensor([[float(r["z"]), float(r["mu"]), float(r["sigma_mu"])] for r in rows], dtype=F64, device=device)
    t = t[t[:, 0].argsort()]
    return t[:, 0], t[:, 1], t[:, 2]


def synthetic_gw(d_em: float, D: torch.Tensor, frac_err: float, n: int, generator: torch.Generator) -> dict:
    """Fake GW 'posterior' (flat prior, weights 1) centred on d_GW(d_em, D) with a fractional error frac_err."""
    d_gw = float(equation.gw_distance(torch.tensor(d_em, dtype=F64), D))
    d = d_gw * (1 + frac_err * torch.randn(n, generator=generator, dtype=F64))
    w = torch.ones_like(d)
    return {"d": d, "w": w, "h": equation.scott_bandwidth(d, w), "label": "synthetic"}
