"""Build the concrete tensors: tachyon signals, observer frames, the neutrino-speed table and a noisy field.

data/neutrino_speed.csv holds published values of (v − c)/c: the 2011 OPERA anomaly (arXiv:1109.4897v1), the
corrected OPERA result (v4, JHEP 10 (2012) 093), ICARUS (Phys. Lett. B 713 (2012) 17) and the SN1987A bound.
"""

from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "neutrino_speed.csv"
DTYPE = torch.float64


def signal_speeds(u: float, device=None) -> torch.Tensor:
    """A fan of tachyon speeds around the chosen one (units of c): (U,)."""
    return torch.tensor([1.25, 2.0, u, 6.0, 25.0], dtype=DTYPE, device=device).unique()


def frames(beta_max: float, n: int, device=None) -> torch.Tensor:
    """Observer velocities 0 … β_max swept by the animation: (F,)."""
    return torch.linspace(0.0, beta_max, n, dtype=DTYPE, device=device)


def emission_and_reception(u: torch.Tensor, distance: float = 1.0) -> torch.Tensor:
    """Reception events (t = d/u, x = d) of signals sent from the origin: (U, 2) as (t, x)."""
    return torch.stack([distance / u, torch.full_like(u, distance)], dim=-1)


def noise_field(n: int, amplitude: float, seed: int = 0, device=None) -> torch.Tensor:
    """Tiny Gaussian fluctuations around the false vacuum φ = 0 on n lattice sites: (N,)."""
    gen = torch.Generator().manual_seed(seed)
    return (amplitude * torch.randn(n, generator=gen, dtype=DTYPE)).to(device)


def load_measurements(path: Path = DATA) -> dict:
    """{experiment: {year, energy_gev, dv_over_c, err_minus, err_plus, kind}} from the bundled CSV."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and ln[0] != "#"]
    header = lines[0].split(",")
    out = {}
    for ln in lines[1:]:
        row = dict(zip(header, ln.split(",")))
        name = row.pop("experiment")
        out[name] = {k: (v if k == "kind" else float(v)) for k, v in row.items()}
    return out
