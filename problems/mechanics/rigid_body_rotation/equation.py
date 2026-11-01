r"""Physics → tensor translation.

A rigid body spinning freely (no torque) keeps its angular momentum \mathbf L = \mathsf I \boldsymbol\omega fixed
in space. Seen from the body's own axes, \mathbf L turns, and Euler's equations follow:

    \mathsf I \dot{\boldsymbol\omega} + \boldsymbol\omega \times (\mathsf I \boldsymbol\omega) = 0 .

In index notation (body-frame components i, j, k, l ∈ {1, 2, 3}, repeated indices summed):

    I_{ij} \dot\omega_j + \varepsilon_{ijk} \, \omega_j \, I_{kl} \omega_l = 0

The cross product is the Levi-Civita tensor ε_ijk contracted with two vectors — one einsum, 'ijk,j,kl,l->i'.
The orientation R_ij (body → world) changes as

    \dot R_{ij} = R_{ik} \, \varepsilon_{klj} \, \omega_l          (\dot R = R [\omega]_\times)

Conserved: the kinetic energy  T = ½ ω_i I_ij ω_j  and  |L|² = (I_ij ω_j)(I_ik ω_k),
and in the world frame the whole vector  L^{world}_i = R_ij I_jk ω_k.
"""

import torch


def levi_civita(dtype: torch.dtype = torch.float64, device: torch.device | None = None) -> torch.Tensor:
    """ε_ijk of shape (3, 3, 3): +1 for even permutations of (0, 1, 2), −1 for odd ones, 0 otherwise."""
    eps = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[i, j, k], eps[i, k, j] = 1.0, -1.0
    return eps


def gyroscopic_torque(omega: torch.Tensor, inertia: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""-ε_{ijk} ω_j I_{kl} ω_l  (j, k, l summed): the 'torque' that turns ω in the body frame.   → (3,)"""
    return -torch.einsum("ijk,j,kl,l->i", eps, omega, inertia, omega)


def rhs(y: torch.Tensor, inertia: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """dy/dt for the state y of shape (4, 3): y[:3] = orientation R (3, 3), y[3] = body-frame ω (3,)."""
    R, omega = y[:3], y[3]
    omega_dot = torch.linalg.solve(inertia, gyroscopic_torque(omega, inertia, eps))  # I_ij ω̇_j = torque_i
    R_dot = torch.einsum("ik,klj,l->ij", R, eps, omega)  # Ṙ = R [ω]×
    return torch.cat([R_dot, omega_dot[None]], dim=0)


def kinetic_energy(omega: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    r"""T = ½ ω_i I_{ij} ω_j.   omega: (..., 3) → (...)"""
    return 0.5 * torch.einsum("...i,ij,...j->...", omega, inertia, omega)


def angular_momentum(omega: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    r"""Body-frame L_i = I_{ij} ω_j.   (..., 3) → (..., 3)"""
    return torch.einsum("ij,...j->...i", inertia, omega)


def world_angular_momentum(R: torch.Tensor, omega: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
    r"""L^{world}_i = R_{ij} I_{jk} ω_k — a fixed vector when there is no torque.   (..., 3) → (..., 3)"""
    return torch.einsum("...ij,...j->...i", R, angular_momentum(omega, inertia))
