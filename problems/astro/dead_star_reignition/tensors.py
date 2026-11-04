"""Build the concrete tensors: compositions, ignition temperatures, the batch of accreting remnants and the grids.

All in cgs. A "case" is one accreting remnant (a white dwarf or a neutron star) with its mass, radius, accretion
rate, composition and heating; every case is one entry of the same (B,) tensors, so the whole survey is one batch.
"""

import math

import torch

from . import equation as eq

F64 = torch.float64

# central temperatures for burning to matter on Gyr timescales (order of magnitude)
T_IGNITE = {"D": 3e5, "H": 3e6}

COMPOSITIONS = {"solar": (0.70, 0.28, 0.02), "helium": (0.0, 0.98, 0.02)}  # (X, Y, Z) mass fractions

NS_MASS, NS_RADIUS = 1.4 * eq.M_SUN, 1.0e6  # 1.4 M☉, 10 km
CRUST_HEAT = 0.15 * eq.MEV / eq.M_U  # deep crustal heating that flows up, erg per accreted gram


def composition(name: str, like: torch.Tensor):
    """(X, Y, Z) as tensors shaped like `like`."""
    return tuple(torch.full_like(like, v) for v in COMPOSITIONS[name])


def radius_grid(n: int = 3000) -> torch.Tensor:
    """Trial radii for a contracting brown dwarf, 10³ km … 3×10⁶ km: (n,)."""
    return torch.logspace(8, 11.5, n, dtype=F64)


def column_grid(n: int = 360) -> torch.Tensor:
    """Column depths y of the accreted layer, 10² … 10¹³ g/cm²: (n,)."""
    return torch.logspace(2, 13, n, dtype=F64)


def shell_cases(wd_masses, wd_rates, ns_rates, m_ch: float, core_temp: float, ns_comps=("solar",)) -> dict:
    """The batch: every (WD mass × accretion rate) then every (NS composition × rate). Rates in M☉/yr.

    Returns a dict of (B,) tensors: kind (0 = WD, 1 = NS), M, R, Mdot (g/s), T_floor, Q_b, f_comp, f_H, X, Y, Z.
    """
    wm, wr = torch.meshgrid(torch.tensor(wd_masses, dtype=F64), torch.tensor(wd_rates, dtype=F64), indexing="ij")
    wm, wr = wm.flatten(), wr.flatten()
    nw = len(wm)
    rates, X, Y, Z = [], [], [], []
    for name in ns_comps:
        rates += list(ns_rates)
        for lst, v in zip((X, Y, Z), COMPOSITIONS[name]):
            lst += [v] * len(ns_rates)
    nr = torch.tensor(rates, dtype=F64)
    nn = len(nr)
    cat = torch.cat
    solar = COMPOSITIONS["solar"]
    return {
        "kind": cat([torch.zeros(nw, dtype=torch.long), torch.ones(nn, dtype=torch.long)]),
        "M": cat([wm * eq.M_SUN, torch.full((nn,), NS_MASS, dtype=F64)]),
        "R": cat([eq.white_dwarf_radius(wm * eq.M_SUN, m_ch), torch.full((nn,), NS_RADIUS, dtype=F64)]),
        "Mdot": cat([wr, nr]) * eq.M_SUN / eq.YEAR,
        # WD: isothermal core floor + compressional heating; NS: crustal heating + steady hot-CNO burning
        "T_floor": cat([torch.full((nw,), core_temp, dtype=F64), torch.full((nn,), 1e7, dtype=F64)]),
        "Q_b": cat([torch.zeros(nw, dtype=F64), torch.full((nn,), CRUST_HEAT, dtype=F64)]),
        "f_comp": cat([torch.ones(nw, dtype=F64), torch.zeros(nn, dtype=F64)]),
        "f_H": cat([torch.zeros(nw, dtype=F64), (torch.tensor(X, dtype=F64) > 0).double()]),
        "X": cat([torch.full((nw,), solar[0], dtype=F64), torch.tensor(X, dtype=F64)]),
        "Y": cat([torch.full((nw,), solar[1], dtype=F64), torch.tensor(Y, dtype=F64)]),
        "Z": cat([torch.full((nw,), solar[2], dtype=F64), torch.tensor(Z, dtype=F64)]),
    }


def lane_emden_start(n: torch.Tensor, xi0: float) -> torch.Tensor:
    """Series start θ ≈ 1 − ξ²/6 + nξ⁴/120 (the equation is singular at ξ = 0): (B, 2)."""
    return torch.stack([1 - xi0**2 / 6 + n * xi0**4 / 120, -xi0 / 3 + n * xi0**3 / 30], dim=1)


def log_ramp(lo: float, hi: float, n: int) -> torch.Tensor:
    """n values log-spaced from lo to hi."""
    return torch.logspace(math.log10(lo), math.log10(hi), n, dtype=F64)
