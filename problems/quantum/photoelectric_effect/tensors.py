"""Build the concrete tensors: frequencies, random electron depths, photon counts and Millikan's 1916 data."""

import csv
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "millikan_1916_sodium.csv"
C_LIGHT = 299792458.0  # m/s
BAND = 3.0  # eV: depth range of the occupied electron states below the top (sodium's band is about 3 eV wide)


def frequencies(nu_min: float, nu_max: float, n: int) -> torch.Tensor:
    """Light frequencies (F,) in units of 10^14 Hz."""
    return torch.linspace(nu_min, nu_max, n, dtype=torch.float64)


def electron_depths(n_freq: int, n_photons: int, width: float = BAND, seed: int = 0, n_int: int | None = None):
    """Depth ε ≥ 0 of the electron that absorbs each photon, uniform on [0, width]: (F, N), or (I, F, N) with n_int."""
    g = torch.Generator().manual_seed(seed)
    shape = (n_freq, n_photons) if n_int is None else (n_int, n_freq, n_photons)
    return width * torch.rand(shape, generator=g, dtype=torch.float64)


def photon_counts(intensities: torch.Tensor, n_photons: int, seed: int = 0) -> torch.Tensor:
    """Photons absorbed during one exposure at each relative intensity: Poisson(n_photons · intensity), shape (I,)."""
    g = torch.Generator().manual_seed(seed + 1)
    return torch.poisson(n_photons * intensities.double(), generator=g).long()


def retarding_voltages(v_min: float, v_max: float, n: int = 121) -> torch.Tensor:
    """Collector voltages (V,) in volts: negative accelerates, positive retards the electrons."""
    return torch.linspace(v_min, v_max, n, dtype=torch.float64)


def millikan_sodium() -> tuple[torch.Tensor, torch.Tensor]:
    """Millikan (1916), sodium: frequencies ν = c/λ (M,) in 10^14 Hz and stopping potentials (M,) in volts."""
    with DATA.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    lam = torch.tensor([float(r["wavelength_nm"]) for r in rows], dtype=torch.float64) * 1e-9
    V = torch.tensor([float(r["stopping_potential_V"]) for r in rows], dtype=torch.float64)
    return C_LIGHT / lam / 1e14, V
