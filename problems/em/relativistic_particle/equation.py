r"""Physics → tensor translation.

The covariant Lorentz force (c = 1, metric η = diag(+1, −1, −1, −1)) for the four-velocity U^μ = γ(1, v):

    \frac{dU^\mu}{d\tau} = \frac{q}{m} F^{\mu}{}_{\nu} U^\nu,      \frac{dX^\mu}{d\tau} = U^\mu

τ is the particle's own (proper) time. The field tensor has two upper indices,

    F^{0i} = -E_i,    F^{i0} = E_i,    F^{ij} = -\varepsilon_{ijk} B_k ,

and to contract with U^ν its second index is lowered with the metric:

    F^{\mu}{}_{\nu} = F^{\mu\alpha} \eta_{\alpha\nu}      (α summed)

For a batch of particles p the force is one einsum('p,mn,pn->pm'): ν is repeated, so it is summed.
Written out, the space part is γ(E + v × B) and the time part E·v, the rate of work done on the particle.

The length of the four-velocity is fixed:  U^\mu U_\mu = \eta_{\mu\nu} U^\mu U^\nu = γ²(1 − v²) = 1.
"""

import torch


def field_tensor(E: torch.Tensor, B: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""F^{μν} of shape (4, 4) from E_i, B_k (3,) and the Levi-Civita symbol ε_{ijk} (3, 3, 3)."""
    F = torch.zeros(4, 4, dtype=E.dtype)
    F[0, 1:] = -E  # F^{0i} = -E_i
    F[1:, 0] = E  # F^{i0} = +E_i
    F[1:, 1:] = -torch.einsum("ijk,k->ij", eps, B)  # F^{ij} = -ε_ijk B_k
    return F


def lower_second(F: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""F^μ_ν = F^{μα} η_{αν}  (α summed).   (4, 4) → (4, 4)"""
    return torch.einsum("ma,an->mn", F, eta)


def four_force(U: torch.Tensor, qm: torch.Tensor, F_mixed: torch.Tensor) -> torch.Tensor:
    r"""dU^μ/dτ = (q/m)_p F^μ_ν U^ν_p  (ν summed).   U: (P, 4), qm: (P,) → (P, 4)"""
    return torch.einsum("p,mn,pn->pm", qm, F_mixed, U)


def rhs(y: torch.Tensor, qm: torch.Tensor, F_mixed: torch.Tensor) -> torch.Tensor:
    """d/dτ of the state y (P, 2, 4): y[:, 0] = X^μ (position), y[:, 1] = U^μ (four-velocity)."""
    X_dot = y[:, 1]
    U_dot = four_force(y[:, 1], qm, F_mixed)
    return torch.stack([X_dot, U_dot], dim=1)


def minkowski_norm(U: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""U^μ U_μ = η_{μν} U^μ U^ν  (both summed).   (..., 4) → (...)"""
    return torch.einsum("...m,mn,...n->...", U, eta, U)
