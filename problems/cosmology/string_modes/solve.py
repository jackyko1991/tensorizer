"""Evaluate the string on the whole worldsheet at once, the level degeneracies and the circle spectrum.

No time stepping: the light-cone solution is known in closed form, so every (τ, σ) point is one mode sum.
"""

import torch

from .equation import (
    circle_mass_squared,
    closed_mass_squared,
    derivatives,
    embedding,
    level_degeneracies,
    level_from_field,
    open_mass_squared,
    oscillator_level,
)


def worldsheet(tau, sigma, alpha, alpha_t, kind: str) -> dict:
    """X (T, S, 3) plus the level measured from the field energy at every τ (T,)."""
    X = embedding(tau, sigma, alpha, alpha_t, kind)
    X_dot, X_prime = derivatives(tau, sigma, alpha, alpha_t, kind)
    return {"X": X, "level_field": level_from_field(X_dot, X_prime, sigma, kind),
            "N": oscillator_level(alpha), "N_t": oscillator_level(alpha_t)}  # fmt: skip


def spectrum(kind: str, n_max: int) -> dict:
    """Mass levels α′M² and their degeneracies for N = 0 … n_max (closed strings: d_N² from both sides)."""
    N = torch.arange(n_max + 1, dtype=torch.float64)
    d = level_degeneracies(n_max)
    if kind == "closed":
        return {"N": N, "M2": closed_mass_squared(N), "degeneracy": d**2}
    return {"N": N, "M2": open_mass_squared(N), "degeneracy": d}


def circle_spectrum(radii: torch.Tensor, k_max: int, n_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All level-matched closed-string states with |n|, |w| ≤ k_max and N, Ñ ≤ n_max, for every radius at once.

    Broadcast grid (R, n, w, N, Ñ) → keep the entries with N − Ñ = n w → M² (B, S) and multiplicity d_N d_Ñ (S,).
    """
    k = torch.arange(-k_max, k_max + 1, dtype=torch.float64, device=radii.device)
    lvl = torch.arange(n_max + 1, dtype=torch.float64, device=radii.device)
    n, w, N, N_t = torch.meshgrid(k, k, lvl, lvl, indexing="ij")  # each (K, K, L, L)
    matched = N - N_t == n * w  # level matching
    d = level_degeneracies(n_max).to(radii.device)
    weight = (d[:, None] * d[None, :]).expand_as(N)[matched]  # (S,)
    M2 = circle_mass_squared(radii[:, None], n[matched], w[matched], N[matched], N_t[matched])  # (B, S)
    return M2, weight


# a few low-lying states (n, w, N, Ñ) to draw as curves M²(R): all obey N − Ñ = n w
NAMED_STATES = {
    "tachyon (0,0,0,0)": (0, 0, 0, 0),
    "graviton etc. (0,0,1,1)": (0, 0, 1, 1),
    "momentum n=1": (1, 0, 1, 1),
    "winding w=1": (0, 1, 1, 1),
    "n=w=1, N=1, Ñ=0": (1, 1, 1, 0),
}


def duality_curves(radii: torch.Tensor) -> torch.Tensor:
    """α′M²(R) of the named states, (B, 5): one broadcast over radii and states."""
    s = torch.tensor(list(NAMED_STATES.values()), dtype=torch.float64, device=radii.device)  # (5, 4)
    return circle_mass_squared(radii[:, None], s[:, 0], s[:, 1], s[:, 2], s[:, 3])
