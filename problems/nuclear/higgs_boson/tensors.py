"""Build the concrete tensors: the diphoton mass histogram, the background design matrix, toys and constants.

data/atlas_opendata_13tev_myy.csv is the m_γγ spectrum of 10 fb⁻¹ of real 13 TeV ATLAS collisions (2016) from the
ATLAS Open Data 2020 release (CERN Open Data record 15006, doi:10.7483/OPENDATA.ATLAS.B5BJ.3SGS, CC0), selected as in
the ATLAS H → γγ outreach analysis, in 0.5 GeV bins from 100 to 160 GeV. The column sm_signal is the Standard Model
H(125) → γγ expectation from the matching ATLAS simulation with the same cuts.
"""

import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "atlas_opendata_13tev_myy.csv"
DTYPE = torch.float64
G_F = 1.1663788e-5  # Fermi constant, GeV⁻² (PDG 2024)
V_EW = (math.sqrt(2) * G_F) ** -0.5  # electroweak vacuum expectation value v ≈ 246.22 GeV
M_H_PDG = (125.20, 0.11)  # PDG 2024 world average Higgs mass, GeV
LUMI = 10.0  # fb⁻¹


def load_histogram(path: Path = DATA, m_min: float = 105.0, m_max: float = 160.0, device=None) -> dict:
    """Read the binned spectrum and keep the bins inside [m_min, m_max].

    Returns edges (N+1,), centers (N,), widths (N,), counts (N,) and sm_signal (N,) as float64 tensors. Any CSV with
    columns m_low, m_high, events (and optionally sm_signal) works; lines starting with # are comments.
    """
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and ln[0] != "#"]
    header = [h.strip() for h in lines[0].split(",")]
    rows = torch.tensor([[float(v) for v in ln.split(",")] for ln in lines[1:]], dtype=DTYPE)
    col = {name: rows[:, header.index(name)] for name in header}
    keep = (col["m_low"] >= m_min - 1e-9) & (col["m_high"] <= m_max + 1e-9)
    lo, hi = col["m_low"][keep], col["m_high"][keep]
    signal = col.get("sm_signal", torch.zeros_like(col["events"]))[keep]
    out = {
        "edges": torch.cat([lo, hi[-1:]]),
        "centers": (lo + hi) / 2,
        "widths": hi - lo,
        "counts": col["events"][keep],
        "sm_signal": signal,
    }
    return {k: v.to(device) for k, v in out.items()}


def design(centers: torch.Tensor, m_min: float, m_max: float, degree: int = 3) -> torch.Tensor:
    """X_{nk} = x_n^k with x = (m − m_min)/(m_max − m_min) ∈ [0, 1]: the background is exp(X θ). → (N, K)"""
    x = (centers - m_min) / (m_max - m_min)
    return x[:, None] ** torch.arange(degree + 1, dtype=centers.dtype, device=centers.device)


def mass_grid(m_lo: float, m_hi: float, step: float, device=None) -> torch.Tensor:
    """Hypothesised Higgs masses for the scan: (M,)."""
    return torch.arange(m_lo, m_hi + 1e-9, step, dtype=DTYPE, device=device)


def toys(expected: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """n Poisson pseudo-experiments drawn from the expected counts (N,): (n, N), fixed seed."""
    gen = torch.Generator().manual_seed(seed)
    return torch.poisson(expected.cpu().expand(n, -1).contiguous(), generator=gen).to(expected.device)


def thinned(counts: torch.Tensor, fractions: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """The histogram as it fills up: nested binomial subsamples holding a fraction f_j of the events. → (J, N)"""
    gen = torch.Generator().manual_seed(seed)
    counts = counts.cpu()
    out, have, prev = [], torch.zeros_like(counts), 0.0
    for f in fractions.tolist():
        p = (f - prev) / max(1.0 - prev, 1e-12)
        rest = counts - have
        have = have + torch.binomial(rest, torch.full_like(rest, min(max(p, 0.0), 1.0)), generator=gen)
        out.append(have.clone())
        prev = f
    return torch.stack(out)


def potential_parameters(m_h: float, v: float) -> tuple[float, float]:
    """μ² and λ of V = −μ²|φ|² + λ|φ|⁴ that give vacuum v and Higgs mass m_H:  μ² = m_H²/2, λ = m_H²/(2v²)."""
    return m_h**2 / 2, m_h**2 / (2 * v**2)
