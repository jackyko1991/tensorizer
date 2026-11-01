r"""Physics → tensor translation.

Maxwell's curl equations in a medium with relative permittivity ε_r (units with c = ε0 = μ0 = 1):

    \partial_t \mathbf H = -\nabla \times \mathbf E,     \varepsilon_r \partial_t \mathbf E = \nabla \times \mathbf H

The curl is the Levi-Civita symbol contracted with the derivative and the field:

    \partial_t H_i = -\varepsilon_{ijk} \partial_j E_k
    \varepsilon_r \partial_t E_i = \varepsilon_{ijk} \partial_j H_k

j and k are repeated, so both are summed. We store the derivative table D_{jk} = ∂_j F_k as a tensor of
shape (3, 3, Nx, Ny) and contract it with ε_{ijk}: einsum('ijk,jk...->i...').

The problem is 2D (nothing depends on z, so ∂_z = 0). With E = (0, 0, E_z) and H = (H_x, H_y, 0) — the TM mode —
the curls only ever produce those same components, so the TM mode stays TM.

Yee scheme: E and H live half a cell apart in space and half a step apart in time. ∂_j E_k uses forward
differences (E_{n+1} − E_n), ∂_j H_k backward differences (H_n − H_{n−1}), which makes every derivative a
centred one. The update is explicit and only stable under the Courant condition

    c \, \Delta t \le \Delta x / \sqrt{d}        (d = number of space dimensions)
"""

import math

import torch


def forward_diff(F: torch.Tensor, h: float) -> torch.Tensor:
    """D_{jk} = ∂_j F_k by forward differences.   F: (3, Nx, Ny) → (3, 3, Nx, Ny); row j = z stays 0 (∂_z = 0)."""
    D = torch.zeros(3, *F.shape, dtype=F.dtype, device=F.device)
    D[0, :, :-1, :] = (F[:, 1:, :] - F[:, :-1, :]) / h  # ∂_x
    D[1, :, :, :-1] = (F[:, :, 1:] - F[:, :, :-1]) / h  # ∂_y
    return D


def backward_diff(F: torch.Tensor, h: float) -> torch.Tensor:
    """D_{jk} = ∂_j F_k by backward differences.   (3, Nx, Ny) → (3, 3, Nx, Ny)"""
    D = torch.zeros(3, *F.shape, dtype=F.dtype, device=F.device)
    D[0, :, 1:, :] = (F[:, 1:, :] - F[:, :-1, :]) / h
    D[1, :, :, 1:] = (F[:, :, 1:] - F[:, :, :-1]) / h
    return D


def curl(eps: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    r"""(∇×F)_i = ε_{ijk} ∂_j F_k   — j and k summed.   eps: (3, 3, 3), D: (3, 3, Nx, Ny) → (3, Nx, Ny)"""
    return torch.einsum("ijk,jk...->i...", eps, D)


def courant_time_step(h: float, courant: float, dim: int, c: float = 1.0) -> float:
    """Δt = S · Δx / (c √d) for a Courant number S. S > 1 would blow up, so it is refused."""
    if not 0 < courant <= 1:
        raise ValueError(f"Courant number must be in (0, 1], got {courant}: the scheme would be unstable")
    return courant * h / (c * math.sqrt(dim))


def yee_step(E, H, eps, eps_r, damping, dt: float, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One leap-frog step: H from curl E, then E from the new curl H.

    E, H: (3, Nx, Ny);  eps_r, damping: (Nx, Ny) — damping is 1 inside and drops below 1 in the absorbing border.
    """
    H = damping * (H - dt * curl(eps, forward_diff(E, h)))  # ∂_t H_i = −ε_ijk ∂_j E_k
    E = damping * (E + dt / eps_r * curl(eps, backward_diff(H, h)))  # ε_r ∂_t E_i = ε_ijk ∂_j H_k
    return E, H
