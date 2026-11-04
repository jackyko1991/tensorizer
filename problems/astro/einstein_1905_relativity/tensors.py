"""Build the concrete tensors: clock speeds, nuclear masses and the Frisch & Smith (1963) muon counts."""

import csv
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "frisch_smith_1963_muons.csv"
C = 299.792458  # m/μs
M_ELECTRON = 0.51099895069  # MeV (CODATA 2022)
M_MUON = 105.6583755  # MeV
TAU_MUON = 2.1969811  # μs, muon lifetime at rest (PDG)

# Atomic masses in u (AME2020, Wang et al., Chinese Phys. C 45, 030003 (2021)): name → (Z, N, M)
NUCLIDES = {
    "H-2": (1, 1, 2.014101777844),
    "He-4": (2, 2, 4.002603254130),
    "Li-7": (3, 4, 7.016003434),
    "C-12": (6, 6, 12.0),
    "O-16": (8, 8, 15.994914619257),
    "Fe-56": (26, 30, 55.934936326),
    "U-238": (92, 146, 238.050786936),
}


def clock_speeds(beta_max: float, n: int) -> torch.Tensor:
    """Speeds (B,) of the light clocks, from a clock at rest up to beta_max."""
    return torch.linspace(0.0, beta_max, n, dtype=torch.float64)


def nuclear_masses(names: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proton numbers, neutron numbers and atomic masses (u) of the named nuclides: three (K,) tensors."""
    Z, N, M = zip(*(NUCLIDES[n] for n in names))
    return (
        torch.tensor(Z, dtype=torch.float64),
        torch.tensor(N, dtype=torch.float64),
        torch.tensor(M, dtype=torch.float64),
    )


def frisch_smith() -> dict:
    """Muon rates on Mt Washington and at sea level (Frisch & Smith 1963), with the published speed bands."""
    with DATA.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))

    def col(name):
        return torch.tensor([float(r[name]) for r in rows], dtype=torch.float64)

    return {
        "station": [r["station"] for r in rows],
        "altitude": col("altitude_m"),  # (2,) m
        "rate": col("rate_per_hour"),  # (2,)
        "err": col("rate_err_per_hour"),  # (2,)
        "beta_min": col("beta_min"),
        "beta_max": col("beta_max"),
    }


def atmosphere(z: torch.Tensor, rho0: float = 1.225e-3, H: float = 8400.0) -> torch.Tensor:
    """Air column (g/cm²) above sea level up to height z (m), isothermal atmosphere ρ = ρ0 e^{−z/H} (g/cm³)."""
    return rho0 * H * 100 * (1 - torch.exp(-z / H))
