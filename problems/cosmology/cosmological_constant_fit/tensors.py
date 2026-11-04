"""Build the concrete tensors: load the supernova table, the redshift grid and the grid of candidate universes.

data/pantheon_plus.csv holds 1466 Type Ia supernovae from the Pantheon+ compilation (Scolnic et al. 2022, Brout et
al. 2022): name, redshift z (CMB frame, corrected for peculiar velocities), distance modulus mu (SH0ES-calibrated
absolute magnitude) and its diagonal error sigma_mu. Only z > 0.01 is kept (slower galaxies are dominated by their
own motions), the Cepheid-calibrator hosts are dropped and a supernova seen by several surveys appears once (the
row with the smallest error). Everything is float64: χ² differences of a few units on top of χ² ≈ 1400 need it.
"""

from pathlib import Path

import torch

from . import equation

DATA = Path(__file__).parent / "data" / "pantheon_plus.csv"
DTYPE = torch.float64
COLUMNS = {  # accepted header spellings (case-insensitive) for each column the fit needs
    "z": ("z", "zhd", "zcmb", "redshift"),
    "mu": ("mu", "mu_sh0es", "distance_modulus", "dm"),
    "sigma": ("sigma_mu", "mu_err", "dmu", "sigma", "mu_sh0es_err_diag", "error"),
}


def load_distance_moduli(path: Path = DATA, device: torch.device | None = None, z_min: float = 0.0):
    """Read a supernova CSV and return z, μ, σ_μ as float64 tensors of shape (N,), sorted by redshift.

    Works for any comma-separated file with a header row that names a redshift, a distance-modulus and an
    error column (see COLUMNS for the accepted names, any order, extra columns ignored, lines starting with #
    skipped), so an uploaded table can replace the bundled Pantheon+ one. Rows with z <= z_min are dropped.
    """
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and ln[0] != "#"]
    header = [h.strip().lower() for h in lines[0].split(",")]
    index = {}
    for key, names in COLUMNS.items():
        found = [header.index(n) for n in names if n in header]
        if not found:
            raise ValueError(f"no {key} column in {path}: expected one of {names}, got {header}")
        index[key] = found[0]
    rows = [ln.split(",") for ln in lines[1:]]
    table = torch.tensor([[float(r[index[k]]) for k in ("z", "mu", "sigma")] for r in rows], dtype=DTYPE)
    table = table[table[:, 0] > z_min]
    table = table[table[:, 0].argsort()].to(device)  # (N, 3)
    return table[:, 0], table[:, 1], table[:, 2]


def redshift_nodes(z_max: float, n: int, device: torch.device | None = None) -> torch.Tensor:
    """Uniform grid 0 … z_max (n nodes) on which the distance integral is accumulated: (n,)."""
    return torch.linspace(0.0, z_max, n, dtype=DTYPE, device=device)


def nodes_for(z: torch.Tensor, n: int = 1000) -> torch.Tensor:
    """Grid that reaches just past the farthest supernova."""
    return redshift_nodes(1.01 * float(z.max()), n, z.device)


def parameter_grid(n_m: int, n_l: int, range_m, range_l, device: torch.device | None = None):
    """Candidate universes: Ω_m values (A,) and Ω_Λ values (B,); the grid itself is their outer product."""
    Om = torch.linspace(*range_m, n_m, dtype=DTYPE, device=device)
    OL = torch.linspace(*range_l, n_l, dtype=DTYPE, device=device)
    return Om, OL


def synthetic_moduli(z, sigma, H0, Om, OL, noise: bool = True, generator: torch.Generator | None = None):
    """Distance moduli a universe (H0, Ω_m, Ω_Λ) would produce at the redshifts z, plus Gaussian errors σ."""
    mu = equation.distance_modulus(equation.luminosity_distance(z, nodes_for(z), H0, Om, OL))
    if noise:
        mu = mu + sigma * torch.randn(len(z), dtype=z.dtype, generator=generator).to(z.device)
    return mu
