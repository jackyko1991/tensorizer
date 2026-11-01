r"""Physics → tensor translation.

A path is a list of positions q_n = q(nΔt), n = 0 … N, with fixed ends q_0, q_N and Δt = T/N. The action

    S = \sum_{n=0}^{N-1} \Big[ \tfrac12 m \Big(\frac{q_{n+1} - q_n}{\Delta t}\Big)^2 - V(q_n) \Big] \Delta t

is the time integral of L = T − V with a forward difference for the velocity and the potential at the left end
of each step. Written with the difference matrix D_{nk} (D_{n,n} = −1, D_{n,n+1} = +1) the velocity is
v_{bn} = D_{nk} q_{bk} / Δt (repeated k ⇒ summed; b labels a path in the batch).

Setting ∂S/∂q_n = 0 for an interior point gives the discrete Euler–Lagrange equation

    \frac{\partial S}{\partial q_n} = -\Delta t \Big[ m\frac{q_{n+1} - 2q_n + q_{n-1}}{\Delta t^2} + V'(q_n) \Big] = 0,

which is exactly the Verlet scheme for m\ddot q = −V'(q). q_n appears only in its own term and its two
neighbours', so the Hessian ∂²S/∂q_n∂q_m is tridiagonal:  2m/Δt − V''Δt on the diagonal, −m/Δt next to it.
"""

from typing import Callable

import torch

Potential = Callable[[torch.Tensor], torch.Tensor]  # V(q) elementwise


def gravity(m: float, g: float) -> tuple[Potential, Potential]:
    """Ball thrown straight up, q = height:  V = m g q,  V' = m g."""
    return (lambda q: m * g * q), (lambda q: m * g * torch.ones_like(q))


def oscillator(m: float, omega: float) -> tuple[Potential, Potential]:
    """Mass on a spring:  V = ½ m ω² q²,  V' = m ω² q."""
    return (lambda q: 0.5 * m * omega**2 * q**2), (lambda q: m * omega**2 * q)


def difference_matrix(n_steps: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """D of shape (N, N+1): (D q)_n = q_{n+1} − q_n."""
    eye = torch.eye(n_steps + 1, dtype=dtype, device=device)
    return eye[1:] - eye[:-1]


def action(q: torch.Tensor, D: torch.Tensor, dt: float, m: float, V: Potential) -> torch.Tensor:
    r"""S_b = Σ_n [½ m v_{bn} v_{bn} − V(q_{bn})] Δt  with  v_{bn} = D_{nk} q_{bk} / Δt.   (B, N+1) → (B,)"""
    v = torch.einsum("nk,bk->bn", D, q) / dt  # (B, N) velocity on each step, k summed
    kinetic = 0.5 * m * torch.einsum("bn,bn->b", v, v)  # n summed
    potential = V(q[:, :-1]).sum(dim=1)  # left end of every step, n = 0 … N−1
    return (kinetic - potential) * dt


def euler_lagrange_residual(q: torch.Tensor, dt: float, m: float, dV: Potential) -> torch.Tensor:
    """r_n = m (q_{n+1} − 2q_n + q_{n−1})/Δt² + V'(q_n) at the interior points.   (B, N+1) → (B, N−1)

    Zero for the true discrete path; ∂S/∂q_n = −Δt r_n.
    """
    accel = (q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]) / dt**2
    return m * accel + dV(q[:, 1:-1])


def exact_path(t: torch.Tensor, q_start: float, q_end: float, system: str, g: float, omega: float) -> torch.Tensor:
    """The classical path through both end points.   (N+1,) → (N+1,)

    gravity:     q(t) = q_0 + (q_N − q_0) t/T + ½ g t (T − t)                    (a parabola)
    oscillator:  q(t) = [q_0 sin ω(T − t) + q_N sin ωt] / sin ωT                   (a sine)
    """
    T = t[-1]
    if system == "gravity":
        return q_start + (q_end - q_start) * t / T + 0.5 * g * t * (T - t)
    return (q_start * torch.sin(omega * (T - t)) + q_end * torch.sin(omega * t)) / torch.sin(omega * T)
