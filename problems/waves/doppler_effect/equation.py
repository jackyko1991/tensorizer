r"""Physics → tensor translation.

A point source moving along x emits one wavefront at each time t_e (e = 0 … E−1) from the point s_{ei}.
In still air every front is a circle that grows at the speed of sound:

    r_e(t) = c (t - t_e)        for t ≥ t_e

Front e reaches listener o (position o_{oi}) at

    t_{oe} = t_e + \frac{1}{c}\sqrt{(o_{oi} - s_{ei})(o_{oi} - s_{ei})}        (i summed)

and the listener hears as many fronts per second as arrive: f'_o = (E − 1) / (t_{o,last} − t_{o,first}).
On the line of motion this gives f' = f c / (c ∓ v_s). For v_s > c the fronts touch a cone with sin μ = c / v_s.
"""

import math

import torch


def front_radii(t: float, t_e: torch.Tensor, c: float) -> torch.Tensor:
    """r_e = c (t − t_e), clipped at 0 for fronts not emitted yet.   t_e: (E,) → (E,)"""
    return torch.clamp(c * (t - t_e), min=0.0)


def arrival_times(observers: torch.Tensor, s: torch.Tensor, t_e: torch.Tensor, c: float) -> torch.Tensor:
    r"""t_{oe} = t_e + |o_o − s_e| / c.   observers: (O, 2), s: (E, 2), t_e: (E,) → (O, E)"""
    d = observers[:, None, :] - s[None, :, :]  # (O, E, 2)
    dist = torch.sqrt(torch.einsum("oei,oei->oe", d, d))  # i summed
    return t_e[None, :] + dist / c


def heard_frequency(arrivals: torch.Tensor) -> torch.Tensor:
    """f'_o = (number of fronts − 1) / (time between first and last arrival).   (O, E) → (O,)"""
    n = arrivals.shape[1]
    span = arrivals.max(dim=1).values - arrivals.min(dim=1).values
    return (n - 1) / span


def doppler_formula(f: float, c: float, v_s: float) -> tuple[float, float]:
    """Textbook (ahead, behind) frequencies f c / (c − v_s) and f c / (c + v_s); |…| once v_s > c."""
    ahead = math.inf if abs(c - v_s) < 1e-9 else f * c / abs(c - v_s)
    return ahead, f * c / (c + v_s)


def cone_half_angle(s: torch.Tensor, radii: torch.Tensor, source_now: torch.Tensor) -> torch.Tensor:
    """Half-angle of the lines from the source that touch the fronts: sin μ_e = r_e / |source − s_e|.

    s: (E, 2), radii: (E,), source_now: (2,). For v_s > c every front gives the same μ: the Mach cone.
    """
    d = source_now[None, :] - s  # (E, 2)
    dist = torch.sqrt(torch.einsum("ei,ei->e", d, d))
    ok = (radii > 0) & (dist > radii)  # the source is outside the front
    return torch.asin(radii[ok] / dist[ok])
