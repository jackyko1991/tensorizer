"""Evolve every kaon world on the whole time grid at once and read off the observables."""

import torch

from . import equation


def pipi_rate(h: torch.Tensor, v: torch.Tensor, eta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Decay rate to π⁺π⁻ (in units of Γ_S(ππ)) for a beam that was pure K⁰ or pure K̄⁰ at t = 0.

    h: (B, 2, 2), v: (B, 2, 2), eta: (B,), t: (T,) → (B, T, 2), last axis = starting K⁰, K̄⁰.
    """
    a = equation.pipi_amplitude(v, eta)  # (B, 2)
    u = equation.evolution(h, t)  # (B, T, 2, 2)
    return torch.einsum("bx,btxy->bty", a, u).abs() ** 2  # x summed: project the state onto ππ


def simulate(dm: float, gamma_s: float, gamma_l: float, eps: torch.Tensor, delta: torch.Tensor, t: torch.Tensor):
    """All observables for a batch of (ε, δ) worlds.  eps, delta: (B,) → dict of (B, T, ...) tensors."""
    h = equation.hamiltonian(dm, gamma_s, gamma_l, eps, delta)  # (B, 2, 2)
    v = equation.eigenvectors(eps, delta)
    p = equation.probabilities(equation.evolution(h, t))  # (B, T, final, initial)
    return {
        "h": h,
        "p": p,
        "a_t": equation.kabir_asymmetry(p),
        "a_cpt": equation.cpt_asymmetry(p),
        "pipi": pipi_rate(h, v, eps, t),  # η₊₋ = ε (direct CP violation ε' neglected)
    }
