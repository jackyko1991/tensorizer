"""Sweep the boost speed: transform F^{μν} for every β at once and read off E', B' and the invariants."""

import torch

from . import tensors
from .equation import electric, invariant, magnetic, pseudo_invariant, transform


def solve(F: torch.Tensor, beta_max: float, n_steps: int = 100) -> dict[str, torch.Tensor]:
    """Return a dict of tensors along the sweep β = 0 … beta_max (S = n_steps values).

    beta (S,), F (S, 4, 4), E (S, 3), B (S, 3), FF (S,) = F_{μν}F^{μν}, FFdual (S,) = F_{μν}F̃^{μν}
    """
    beta = torch.linspace(0.0, beta_max, n_steps, dtype=tensors.DTYPE)  # (S,)
    Lam = tensors.boost(beta)  # (S, 4, 4)
    F_boosted = transform(Lam, F)  # (S, 4, 4)  all S boosts in one einsum

    eta, eps3, eps4 = tensors.metric(), tensors.levi_civita(3), tensors.levi_civita(4)
    return {
        "beta": beta,
        "F": F_boosted,
        "E": electric(F_boosted),
        "B": magnetic(F_boosted, eps3),
        "FF": invariant(F_boosted, eta),
        "FFdual": pseudo_invariant(F_boosted, eta, eps4),
    }
