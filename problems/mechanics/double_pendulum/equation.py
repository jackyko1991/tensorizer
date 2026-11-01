r"""Physics → tensor translation.

Two point masses m_1, m_2 on light rods L_1, L_2. With the angles q = (θ_1, θ_2) as generalised
coordinates, the Euler–Lagrange equations take the form

    \mathsf M(\mathbf q) \, \ddot{\mathbf q} = \mathbf f(\mathbf q, \dot{\mathbf q}),

or in index notation, for a batch b of pendulums and coordinates i, j ∈ {1, 2}:

    M_{bij} \, \ddot q_{bj} = f_{bi}          (repeated j ⇒ summed)

with the 2×2 mass matrix (Δ = θ_1 − θ_2)

    M = [[ (m_1 + m_2) L_1^2,        m_2 L_1 L_2 \cos Δ ],
         [ m_2 L_1 L_2 \cos Δ,       m_2 L_2^2          ]]

and the generalised forces (centrifugal + gravity)

    f_1 = -m_2 L_1 L_2 \sin Δ \, \dot θ_2^2 - (m_1 + m_2) g L_1 \sin θ_1
    f_2 =  m_2 L_1 L_2 \sin Δ \, \dot θ_1^2 - m_2 g L_2 \sin θ_2

Each step we build M_{bij} and f_{bi} and let torch.linalg.solve find \ddot q_{bj}: no hand inversion.
The energy E_b = ½ \dot q_{bi} M_{bij} \dot q_{bj} + V_b (i and j summed) is conserved.
"""

import torch


def mass_matrix(q: torch.Tensor, m1: float, m2: float, L1: float, L2: float) -> torch.Tensor:
    """M_bij for angles q of shape (B, 2) → (B, 2, 2)."""
    coupling = m2 * L1 * L2 * torch.cos(q[:, 0] - q[:, 1])  # (B,)
    M = torch.empty(q.shape[0], 2, 2, dtype=q.dtype, device=q.device)
    M[:, 0, 0] = (m1 + m2) * L1**2
    M[:, 0, 1] = coupling
    M[:, 1, 0] = coupling
    M[:, 1, 1] = m2 * L2**2
    return M


def forces(q: torch.Tensor, q_dot: torch.Tensor, m1: float, m2: float, L1: float, L2: float, g: float) -> torch.Tensor:
    """f_bi (centrifugal + gravity) for q, q_dot of shape (B, 2) → (B, 2)."""
    s = m2 * L1 * L2 * torch.sin(q[:, 0] - q[:, 1])  # (B,)
    f1 = -s * q_dot[:, 1] ** 2 - (m1 + m2) * g * L1 * torch.sin(q[:, 0])
    f2 = s * q_dot[:, 0] ** 2 - m2 * g * L2 * torch.sin(q[:, 1])
    return torch.stack([f1, f2], dim=-1)


def rhs(y: torch.Tensor, m1: float, m2: float, L1: float, L2: float, g: float) -> torch.Tensor:
    r"""dy/dt for y of shape (B, 2, 2): y[:, 0] = q, y[:, 1] = \dot q.  Solves M_{bij} \ddot q_{bj} = f_{bi}."""
    q, q_dot = y[:, 0], y[:, 1]
    M = mass_matrix(q, m1, m2, L1, L2)  # (B, 2, 2)
    f = forces(q, q_dot, m1, m2, L1, L2, g)  # (B, 2)
    q_ddot = torch.linalg.solve(M, f)  # (B, 2)
    return torch.stack([q_dot, q_ddot], dim=1)


def energy(y: torch.Tensor, m1: float, m2: float, L1: float, L2: float, g: float) -> torch.Tensor:
    r"""E_b = ½ \dot q_{bi} M_{bij} \dot q_{bj} + V_b.   y: (..., B, 2, 2) → (..., B)"""
    q, q_dot = y[..., 0, :], y[..., 1, :]
    M = mass_matrix(q.reshape(-1, 2), m1, m2, L1, L2).reshape(*q.shape, 2)  # (..., B, 2, 2)
    kinetic = 0.5 * torch.einsum("...i,...ij,...j->...", q_dot, M, q_dot)
    potential = -(m1 + m2) * g * L1 * torch.cos(q[..., 0]) - m2 * g * L2 * torch.cos(q[..., 1])
    return kinetic + potential


def small_angle_modes(m1: float, m2: float, L1: float, L2: float, g: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Normal modes near the bottom: M^0_{ij} \ddot q_j = -K_{ij} q_j  ⇒  (M^0)^{-1}_{ik} K_{kj} e_j = ω^2 e_i.

    Returns ω (2,) sorted ascending and the mode shapes e (2, 2), mode n = column n.
    """
    M0 = mass_matrix(torch.zeros(1, 2, dtype=torch.float64), m1, m2, L1, L2)[0]  # (2, 2) at q = 0
    K = torch.diag(torch.tensor([(m1 + m2) * g * L1, m2 * g * L2], dtype=torch.float64))  # (2, 2)
    omega_sq, e = torch.linalg.eig(torch.linalg.solve(M0, K))  # not symmetric ⇒ general eig
    omega_sq, e = omega_sq.real, e.real
    order = torch.argsort(omega_sq)
    return torch.sqrt(omega_sq[order]), e[:, order]
