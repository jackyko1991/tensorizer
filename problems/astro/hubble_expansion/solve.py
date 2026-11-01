"""Evaluate the expansion on the whole time axis at once (no time stepping needed: a(t) is known)."""

import torch

from .equation import (
    hubble_rate,
    photon_comoving_distance,
    positions,
    redshift,
    relative_to,
    scale_factor,
    velocities,
)


def solve(q: torch.Tensor, t: torch.Tensor, model: str, H0: float, observer: int) -> dict:
    """Everything the plot needs, as tensors over time t (T,) and galaxies g (G,).

    A photon leaves an extra galaxy (the "emitter") at the first time step and reaches the observer
    exactly at the last one; the emitter's comoving distance is the photon's total comoving path.
    """
    a = scale_factor(t, model, H0)  # (T,)
    H = hubble_rate(t, model, H0)  # (T,)
    x = positions(q, a)  # (T, G, 2)
    v = velocities(x, H)  # (T, G, 2)

    chi = photon_comoving_distance(t, a)  # (T,) comoving distance covered by the photon
    q_obs = q[observer]
    q_emit = q_obs + torch.stack([chi[-1], torch.zeros_like(chi[-1])])  # emitter due "east" of the observer
    q_photon = q_emit - torch.einsum("t,i->ti", chi, torch.tensor([1.0, 0.0], dtype=q.dtype, device=q.device))
    return {
        "t": t,
        "a": a,
        "H": H,
        "x": x,
        "v": v,
        "x_rel": relative_to(x, observer),  # (T, G, 2) seen from the observer
        "v_rel": relative_to(v, observer),
        "emitter": torch.einsum("t,i->ti", a, q_emit),  # (T, 2)
        "photon": torch.einsum("t,ti->ti", a, q_photon),  # (T, 2)
        "z": redshift(a, a[0]),  # (T,) redshift the photon would show if it arrived at time t
    }


def fitted_hubble_rate(x_rel: torch.Tensor, v_rel: torch.Tensor) -> torch.Tensor:
    r"""Least-squares slope of v = H d at each time: H = Σ v_{gi} x_{gi} / Σ x_{gi} x_{gi}.   → (T,)"""
    return torch.einsum("tgi,tgi->t", v_rel, x_rel) / torch.einsum("tgi,tgi->t", x_rel, x_rel)
