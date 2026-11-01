r"""Physics → tensor translation.

E and B are the six independent components of the antisymmetric field tensor (c = 1,
metric η = diag(+1, −1, −1, −1)):

    F^{0i} = -E_i,     F^{i0} = E_i,     F^{ij} = -\varepsilon_{ijk} B_k

             ⎡  0   -E_x  -E_y  -E_z ⎤
    F^{μν} = ⎢ E_x    0   -B_z   B_y ⎥
             ⎢ E_y   B_z    0   -B_x ⎥
             ⎣ E_z  -B_y   B_x    0  ⎦

A tensor with two upper indices transforms with one Λ per index:

    F'^{\mu\nu} = \Lambda^{\mu}{}_{\alpha} \Lambda^{\nu}{}_{\beta} F^{\alpha\beta}

α and β are repeated, so both are summed — einsum('ma,nb,ab->mn'), i.e. Λ F Λᵀ.
Contracting all indices gives numbers every observer agrees on (Lorentz invariants):

    F_{\mu\nu} F^{\mu\nu} = 2 (B^2 - E^2),     F_{\mu\nu} \tilde F^{\mu\nu} = -4\, \mathbf E \cdot \mathbf B

with indices lowered by the metric, F_{\mu\nu} = \eta_{\mu\alpha} \eta_{\nu\beta} F^{\alpha\beta},
and the dual tensor \tilde F^{\mu\nu} = ½ \varepsilon^{\mu\nu\rho\sigma} F_{\rho\sigma}.
"""

import torch


def field_tensor(E: torch.Tensor, B: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""F^{μν} of shape (4, 4) from E_i, B_k (3,) and the 3D Levi-Civita ε_{ijk} (3, 3, 3)."""
    F = torch.zeros(4, 4, dtype=E.dtype)
    F[0, 1:] = -E  # F^{0i} = -E_i
    F[1:, 0] = E  # F^{i0} = +E_i   (antisymmetric)
    F[1:, 1:] = -torch.einsum("ijk,k->ij", eps, B)  # F^{ij} = -ε_ijk B_k
    return F


def electric(F: torch.Tensor) -> torch.Tensor:
    """E_i = F^{i0}.   (..., 4, 4) → (..., 3)"""
    return F[..., 1:, 0]


def magnetic(F: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""B_i = -½ ε_{ijk} F^{jk}.   (..., 4, 4) → (..., 3)"""
    return -0.5 * torch.einsum("ijk,...jk->...i", eps, F[..., 1:, 1:])


def transform(Lam: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    r"""F'^{μν} = Λ^μ_α Λ^ν_β F^{αβ} for a batch of S boosts.   Lam: (S, 4, 4), F: (4, 4) → (S, 4, 4)"""
    return torch.einsum("sma,snb,ab->smn", Lam, Lam, F)


def lower(F: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""F_{μν} = η_{μα} η_{νβ} F^{αβ}.   (..., 4, 4) → (..., 4, 4)"""
    return torch.einsum("ma,nb,...ab->...mn", eta, eta, F)


def invariant(F: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""F_{μν} F^{μν} = 2(B² − E²): every index is summed, a scalar per boost.   → (...)"""
    return torch.einsum("...mn,...mn->...", lower(F, eta), F)


def pseudo_invariant(F: torch.Tensor, eta: torch.Tensor, eps4: torch.Tensor) -> torch.Tensor:
    r"""F_{μν} \tilde F^{μν} = −4 E·B with the dual \tilde F^{μν} = ½ ε^{μνρσ} F_{ρσ}.   → (...)"""
    F_low = lower(F, eta)
    dual = 0.5 * torch.einsum("mnrs,...rs->...mn", eps4, F_low)
    return torch.einsum("...mn,...mn->...", F_low, dual)
