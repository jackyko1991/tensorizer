r"""Physics → tensor translation.

The Heitler model of an electromagnetic shower: a high-energy electron radiates a photon (bremsstrahlung),
a photon converts into an electron–positron pair, and each time the energy is shared equally:

    e^± → e^± + γ,        γ → e^+ + e^-,        E_{child} = E_{parent} / 2.

Depth t is measured in radiation lengths X_0. In the simplest version every particle splits after one X_0,
so after t steps there are 2^t particles of energy E_0 / 2^t. Multiplication stops once the daughters
would fall below the critical energy E_c (where ionisation losses beat radiation), giving

    N_{max} ≈ E_0 / E_c,        t_{max} ≈ \ln(E_0/E_c) / \ln 2.

On tensors, particles p carry energy E_p and type s_p (1 = e^±, 0 = γ); a split replaces each parent by two
children with energies E_p/2 and types (1, 1 − s_p). Energy is conserved at every depth:

    Σ_p E_p(t) + E_{dep}(t) = E_0.

Stochastic variant: an e^± interacts with probability 1 − e^{−Δt/λ} per step Δt with λ = X_0, a photon with
λ = 9/7 X_0 (its mean free path for pair production).
"""

import math

import torch

PHOTON, ELECTRON = 0, 1


def interaction_probability(kind: torch.Tensor, dt: float) -> torch.Tensor:
    r"""p_p = 1 − e^{−Δt/λ_p},  λ = 1 X_0 for e^±, 9/7 X_0 for γ.   kind: (P,) → (P,)"""
    mean_free_path = torch.where(kind == ELECTRON, 1.0, 9.0 / 7.0)
    return 1.0 - torch.exp(-dt / mean_free_path)


def can_split(energy: torch.Tensor, e_crit: float) -> torch.Tensor:
    r"""A particle multiplies only while its daughters stay above the critical energy: E_p/2 ≥ E_c."""
    return energy / 2 >= e_crit


def children(energy: torch.Tensor, kind: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Two daughters per parent: energies (E/2, E/2), types (e^±, 1 − s): e → e + γ, γ → e + e.

    energy, kind: (P,) → energies (2P,), kinds (2P,)  [first all first daughters, then all second ones]
    """
    half = energy / 2
    first = torch.full_like(kind, ELECTRON)
    second = 1 - kind
    return torch.cat([half, half]), torch.cat([first, second])


def heitler_n_max(e0: float, e_crit: float) -> float:
    r"""N_{max} ≈ E_0 / E_c."""
    return e0 / e_crit


def heitler_t_max(e0: float, e_crit: float) -> float:
    r"""t_{max} ≈ \ln(E_0/E_c) / \ln 2  (in X_0)."""
    return math.log(e0 / e_crit) / math.log(2.0)
