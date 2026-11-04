"""Build the concrete tensors: the SPARC rotation curves padded into (galaxy, radius) arrays, and synthetic galaxies.

Units: kpc, km/s, solar masses; accelerations in (km/s)²/kpc (1 (km/s)²/kpc = 3.24 × 10⁻¹⁴ m/s²).
"""

from pathlib import Path

import torch

from . import equation
from .equation import G_NEWTON  # noqa: F401  kpc (km/s)² / M_sun

F64 = torch.float64
MS2_PER_KMS2_KPC = 3.24078e-14  # 1 (km/s)²/kpc in m/s²
A0_DEFAULT = 1.2e-10 / MS2_PER_KMS2_KPC  # Milgrom's a₀ = 1.2 × 10⁻¹⁰ m/s² ≈ 3700 (km/s)²/kpc
UPS_DISK, UPS_BULGE = 0.5, 0.7  # mass-to-light ratios at 3.6 μm expected from stellar populations (Lelli+ 2016)
DATA = Path(__file__).parent / "data" / "sparc_rotation_curves.csv"
FIELDS = ["R", "V", "err", "gas", "disk", "bul"]


def load_sparc(device: torch.device, path: Path = DATA) -> dict:
    """Read the CSV (plain Python; # lines are comments) and pad every galaxy to the longest curve.

    Returns tensors of shape (G, Rmax) for R, V, err, gas, disk, bul, a boolean mask of real points, the names and
    the distances. Padding uses R = 1, err = 1 and zero speeds, and is switched off by the mask.
    """
    rows: dict[str, list[list[float]]] = {}
    dist: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("galaxy") or not line.strip():
                continue
            name, d, *vals = line.strip().split(",")
            rows.setdefault(name, []).append([float(v) for v in vals])
            dist[name] = float(d)
    names = list(rows)
    n = max(len(r) for r in rows.values())
    out = {k: torch.zeros(len(names), n, dtype=F64) for k in FIELDS}
    out["R"] += 1.0
    out["err"] += 1.0
    mask = torch.zeros(len(names), n, dtype=torch.bool)
    for g, name in enumerate(names):
        block = torch.tensor(rows[name], dtype=F64)  # (N_g, 6)
        for k, field in enumerate(FIELDS):
            out[field][g, : len(block)] = block[:, k]
        mask[g, : len(block)] = True
    data = {k: v.to(device) for k, v in out.items()}
    data.update(mask=mask.to(device), names=names, distance=[dist[n] for n in names])
    data["has_bulge"] = (data["bul"].abs().amax(1) > 0).to(F64)
    return data


def synthetic_galaxies(device: torch.device, noise: float = 0.0, seed: int = 0) -> tuple[dict, dict]:
    """Three mock galaxies: exponential stellar disk (Υ_d = 1 template) plus gas plus an NFW halo, known parameters."""
    truth = {
        "ups_d": torch.tensor([0.5, 0.65, 0.4], dtype=F64),
        "V_s": torch.tensor([160.0, 90.0, 240.0], dtype=F64),
        "r_s": torch.tensor([12.0, 5.0, 20.0], dtype=F64),
    }
    M_d = torch.tensor([4e10, 5e9, 1.2e11], dtype=F64)[:, None]
    Rd = torch.tensor([3.0, 1.5, 5.0], dtype=F64)[:, None]
    R = torch.linspace(0.3, 1.0, 30, dtype=F64)[None, :] * 10 * Rd  # out to 10 R_d, (3, 30)
    disk = equation.exponential_disk_velocity(R, M_d, Rd)
    gas = 0.5 * equation.exponential_disk_velocity(R, M_d, 3 * Rd)
    V2 = (
        gas**2
        + truth["ups_d"][:, None] * disk**2
        + equation.nfw_velocity(R, truth["V_s"][:, None], truth["r_s"][:, None]) ** 2
    )
    V = V2.sqrt()
    err = 0.03 * V + 2.0
    gen = torch.Generator().manual_seed(seed)
    V = V + noise * err * torch.randn(V.shape, generator=gen, dtype=F64)
    data = {"R": R, "V": V, "err": err, "gas": gas, "disk": disk, "bul": torch.zeros_like(R)}
    data = {k: v.to(device) for k, v in data.items()}
    data.update(mask=torch.ones_like(R, dtype=torch.bool, device=device), names=["mock A", "mock B", "mock C"],
                has_bulge=torch.zeros(3, dtype=F64, device=device), prior_dex=None)  # fmt: skip
    return data, {k: v.to(device) for k, v in truth.items()}
