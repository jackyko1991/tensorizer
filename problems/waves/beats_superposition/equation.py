r"""Physics → tensor translation.

Superposition of S travelling sinusoids,

    y(x, t) = \sum_s A_s \cos(k_s x - \omega_s t),

in index form with positions x_j and times t_n:

    \theta_{snj} = k_s x_j - \omega_s t_n,        y_{nj} = A_s \cos\theta_{snj}     (s is summed)

The envelope is the length of the sum of phasors, \mathcal E_{nj} = | A_s e^{i\theta_{snj}} |  (s summed).
For two waves the carrier moves at v_p = \bar\omega / \bar k and the envelope at v_g = \Delta\omega / \Delta k.
"""

import torch


def phase(k: torch.Tensor, omega: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""θ_{snj} = k_s x_j − ω_s t_n by broadcasting.   k, omega: (S,), x: (J,), t: (N,) → (S, N, J)"""
    return k[:, None, None] * x[None, None, :] - omega[:, None, None] * t[None, :, None]


def superpose(A: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    r"""y_{nj} = A_s cos θ_{snj}   — s summed.   A: (S,), theta: (S, N, J) → (N, J)"""
    return torch.einsum("s,snj->nj", A, torch.cos(theta))


def envelope(A: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    r"""\mathcal E_{nj} = | A_s e^{iθ_{snj}} |   — s summed.   → (N, J)"""
    phasors = torch.einsum("s,snj->nj", A.to(torch.complex64), torch.exp(1j * theta))
    return phasors.abs()


def phase_velocity(k: torch.Tensor, omega: torch.Tensor) -> float:
    """v_p = mean ω / mean k (the carrier)."""
    return (omega.mean() / k.mean()).item()


def group_velocity(k: torch.Tensor, omega: torch.Tensor) -> float:
    """v_g = Δω / Δk between the outermost waves (the envelope)."""
    return ((omega[-1] - omega[0]) / (k[-1] - k[0])).item()
