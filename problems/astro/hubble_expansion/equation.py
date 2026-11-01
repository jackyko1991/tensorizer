r"""Physics → tensor translation.

In an expanding universe galaxies keep fixed comoving coordinates q_{gi} (a grid that never changes) and
their physical positions grow with the scale factor a(t):

    x_{gi}(t) = a(t)\, q_{gi}

Differentiating in time gives every galaxy's velocity:

    v_{gi} = \dot a\, q_{gi} = H\, x_{gi},        H(t) = \dot a / a      (the Hubble rate)

Seen from any galaxy o (subtract its position and velocity), the same law holds:

    v_{gi} - v_{oi} = H\,(x_{gi} - x_{oi})

so every observer thinks they are at the centre of the expansion. Light also gets stretched with a(t):
a photon emitted at time t_e and received at t_o arrives redshifted by

    1 + z = \lambda_o / \lambda_e = a(t_o) / a(t_e).

On its way the photon covers comoving distance dχ = c\,dt / a(t).
"""

import math

import torch

C = 306.6  # speed of light in Mpc/Gyr
KMS_PER_MPC_GYR = 977.8  # 1 Mpc/Gyr = 977.8 km/s


def scale_factor(t: torch.Tensor, model: str, H0: float) -> torch.Tensor:
    r"""a(t), normalised to a = 1 today (t = t_0).   H0 in 1/Gyr.

    matter:     a = (t / t_0)^{2/3}   with t_0 = 2 / (3 H_0) the age of the universe
    de Sitter:  a = e^{H_0 (t - t_0)} with t_0 = 1 / H_0 (no Big Bang: just a label for "today")
    """
    if model == "matter":
        return (t / today(model, H0)) ** (2 / 3)
    return torch.exp(H0 * (t - today(model, H0)))


def time_of(a: float, model: str, H0: float) -> float:
    """Inverse of scale_factor: the time at which the universe had scale factor a."""
    t0 = today(model, H0)
    if model == "matter":
        return t0 * a**1.5
    return t0 + math.log(a) / H0


def today(model: str, H0: float) -> float:
    return 2 / (3 * H0) if model == "matter" else 1 / H0


def hubble_rate(t: torch.Tensor, model: str, H0: float) -> torch.Tensor:
    r"""H = \dot a / a: 2 / (3t) for matter, the constant H_0 for de Sitter."""
    if model == "matter":
        return 2 / (3 * t)
    return torch.full_like(t, H0)


def positions(q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r"""x_{tgi} = a_t q_{gi}: physical positions at every time.   q: (G, 2), a: (T,) → (T, G, 2)"""
    return torch.einsum("t,gi->tgi", a, q)


def velocities(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    r"""v_{tgi} = H_t x_{tgi} (Hubble flow).   x: (T, G, 2), H: (T,) → (T, G, 2)"""
    return torch.einsum("t,tgi->tgi", H, x)


def relative_to(field: torch.Tensor, o: int) -> torch.Tensor:
    """Positions or velocities as seen from galaxy o (subtract its own).   (T, G, 2) → (T, G, 2)"""
    return field - field[:, o : o + 1, :]


def photon_comoving_distance(t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r"""χ(t) = ∫ c\,dt' / a(t') from the first time step to t (trapezoid rule), (T,)."""
    integrand = C / a
    steps = 0.5 * (integrand[1:] + integrand[:-1]) * t.diff()
    return torch.cat([torch.zeros_like(t[:1]), steps.cumsum(0)])


def redshift(a_obs: torch.Tensor, a_emit: torch.Tensor) -> torch.Tensor:
    """z from 1 + z = a_obs / a_emit."""
    return a_obs / a_emit - 1
