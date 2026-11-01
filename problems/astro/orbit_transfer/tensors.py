"""Build the concrete tensors: central body, initial state, burn schedule and the grid of radius ratios.

Units: kilometres and seconds, so μ is in km³/s² and every Δv comes out in km/s. Radii are entered in the
body's own unit (Earth radii around the Earth, astronomical units around the Sun). Everything is float64:
the Sun's μ ≈ 1.3e11 km³/s² and radii ≈ 1e8 km would lose too many digits in float32.
"""

import torch

from . import equation

DTYPE = torch.float64

BODIES = {
    #        μ (km³/s²)           length unit (km)      unit label   time unit (s)
    "earth": {"mu": 398600.4418, "unit_km": 6378.137, "unit": "R⊕", "t_unit": 3600.0, "t_label": "h"},
    "sun": {"mu": 1.32712440018e11, "unit_km": 1.495978707e8, "unit": "AU", "t_unit": 86400.0, "t_label": "d"},
}
MARS_AU = 1.523679  # semi-major axis of Mars's orbit (Earth's is 1 AU by definition)


def radii_km(params: dict, body: dict) -> tuple[float, float, float]:
    """r1, r2 and the bi-elliptic intermediate radius r_b = (r_b/r2) · max(r1, r2), all in km."""
    r1 = params["r1"] * body["unit_km"]
    r2 = params["r2"] * body["unit_km"]
    rb = params["rb_ratio"] * max(r1, r2)
    return r1, r2, rb


def spacecraft_kinds(transfer: str) -> list[str]:
    """Which transfer each spacecraft in the batch flies."""
    return ["hohmann", "bi-elliptic"] if transfer == "both" else [transfer.replace("_", "-")]


def initial_state(r1: float, n: int, mu: float, device: torch.device) -> torch.Tensor:
    r"""y0 of shape (B, 2, 2): every spacecraft on the circle r1 at (r1, 0), moving in +y at v_c = \sqrt{μ / r1}."""
    zero = torch.zeros(n, dtype=DTYPE, device=device)
    position = torch.stack([zero + r1, zero], dim=-1)  # (B, 2)
    velocity = torch.stack([zero, zero + (mu / r1) ** 0.5], dim=-1)  # (B, 2)
    return torch.stack([position, velocity], dim=1)


def burn_schedule(kinds: list[str], r1: float, r2: float, rb: float, mu: float, device: torch.device):
    """Burn times t_bk and signed Δv_bk (km/s), both (B, 3), k = burn number.

    Every first burn is at t = 0 (on the +x axis). The Hohmann transfer pads its missing third burn with Δv = 0.
    """
    R1, R2, RB = (torch.tensor(r, dtype=DTYPE) for r in (r1, r2, rb))
    times, dvs = [], []
    for kind in kinds:
        if kind == "hohmann":
            dv1, dv2, t = equation.hohmann(R1, R2, mu)
            times.append([0.0, t, t])
            dvs.append([dv1, dv2, 0.0])
        else:
            dv1, dv2, dv3, t = equation.bi_elliptic(R1, R2, RB, mu)
            t1 = equation.half_period((R1 + RB) / 2, mu)
            times.append([0.0, t1, t])
            dvs.append([dv1, dv2, dv3])
    return _table(times, device), _table(dvs, device)


def _table(rows: list[list], device: torch.device) -> torch.Tensor:
    return torch.tensor([[float(v) for v in row] for row in rows], dtype=DTYPE, device=device)


def ratio_grid(n: int = 400) -> torch.Tensor:
    """Radius ratios R = r2/r1 from 1 to 100, log-spaced, (N,)."""
    return torch.logspace(0, 2, n, dtype=DTYPE)
