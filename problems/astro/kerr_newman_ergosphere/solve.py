"""Evaluate every horizon, ergosurface, orbit and frame-dragging quantity for a whole batch of holes at once."""

import torch

from . import equation


def survey(M: torch.Tensor, a: torch.Tensor, Q: torch.Tensor) -> dict:
    """The batched grid (any shape): horizons, naked mask, surface gravity, equatorial ergosphere radius."""
    r_plus, r_minus = equation.horizons(M, a, Q)
    return {
        "r_plus": r_plus,
        "r_minus": r_minus,
        "naked": equation.is_naked(M, a, Q),
        "kappa": equation.surface_gravity(M, a, Q),
        "ergo_equator": equation.ergosphere(torch.full_like(M, torch.pi / 2), M, a, Q),
    }


def sweep(M: torch.Tensor, a: torch.Tensor, Q: torch.Tensor, theta: torch.Tensor) -> dict:
    """Everything the animation needs along the sweep. M, a, Q: (F,) frames; theta: (T,).

    Returns (F,) radii, (F, T) ergosurfaces, equatorial orbits, horizon spin Ω_H and the L = 0 spiral (F, R).
    """
    out = survey(M, a, Q)
    th, Mb, ab, Qb = theta[None, :], M[:, None], a[:, None], Q[:, None]  # broadcast (F, 1) × (1, T) → (F, T)
    out["ergo_outer"] = equation.ergosphere(th, Mb, ab, Qb)
    out["ergo_inner"] = equation.ergosphere(th, Mb, ab, Qb, inner=True)
    out.update(equation.equatorial_orbits(M, a, Q, n_r=8001))
    out["omega_H"] = a / (out["r_plus"] ** 2 + a**2)  # frame dragging at the horizon (rigid rotation)
    holes = ~out["naked"]
    out["spiral_r"], out["spiral_phi"] = equation.zero_angular_momentum_spiral(M, a, Q)
    out["spiral_phi"][~holes] = float("nan")
    # angular velocities dφ/dt of the circular orbits (for the moving particles), at their ISCO / photon radii
    for name in ("pro", "retro"):
        for kind in ("isco", "photon"):
            r = out[f"{kind}_{name}"]
            orb = equation.circular_orbits(torch.nan_to_num(r, nan=10.0)[:, None], Mb, ab, Qb)[name]
            out[f"Omega_{kind}_{name}"] = orb["Omega"][:, 0]
    return out
