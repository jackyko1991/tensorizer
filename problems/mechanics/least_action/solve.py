"""Minimise the discrete action over the interior points: gradient descent with momentum, or damped Newton.

Only q_1 … q_{N−1} are free; the end points are glued back on before every evaluation of S.
"""

import torch

from problems._common.progress import steps

from .equation import Potential, action


def with_ends(x: torch.Tensor, q_start: torch.Tensor, q_end: torch.Tensor) -> torch.Tensor:
    """Interior points (B, N−1) → whole paths (B, N+1) with the fixed ends attached."""
    return torch.cat([q_start.expand(x.shape[0], 1), x, q_end.expand(x.shape[0], 1)], dim=1)


def hessian(q: torch.Tensor, D: torch.Tensor, dt: float, m: float, V: Potential) -> torch.Tensor:
    """∂²S_b/∂q_{bn}∂q_{bm} for every path by autograd: (B, N+1) → (B, N−1, N−1).

    The paths are independent, so the Hessian of ΣS_b is block diagonal in b: einsum 'bnbm->bnm' keeps the
    diagonal blocks (no index summed, b is just read twice).
    """
    ends = q[:, :1], q[:, -1:]

    def total(x):
        return action(torch.cat([ends[0], x, ends[1]], dim=1), D, dt, m, V).sum()

    full = torch.autograd.functional.hessian(total, q[:, 1:-1].detach(), vectorize=True)  # (B, N−1, B, N−1)
    return torch.einsum("bnbm->bnm", full)


def relax(
    q0: torch.Tensor, D: torch.Tensor, dt: float, m: float, V: Potential, n_iter: int, method: str
) -> dict[str, torch.Tensor]:
    """Push every trial path towards ∂S/∂q_n = 0. Returns the history of paths (K+1, B, N+1),
    actions (K+1, B) and gradients ∂S/∂q_n (K+1, B, N−1).

    gradient: x ← x − η ∂S/∂x + momentum, with η = 0.8 Δt/m (just below the stable limit set by the
              largest Hessian eigenvalue ≈ 4m/Δt). Works when S has a minimum (short T).
    newton:   x ← x − α H⁻¹ ∂S/∂x with α = 0.2. Finds any stationary point, also a saddle (long T).
    """
    q_start, q_end = q0[:, :1], q0[:, -1:]
    x = q0[:, 1:-1].clone().requires_grad_(True)  # (B, N−1) the free interior points
    optimiser = torch.optim.SGD([x], lr=0.8 * dt / m, momentum=0.9)
    paths, actions, grads = [], [], []
    for _ in steps(n_iter + 1, f"{method} iterations"):
        q = with_ends(x, q_start, q_end)  # (B, N+1)
        S = action(q, D, dt, m, V)  # (B,)
        (dS,) = torch.autograd.grad(S.sum(), x)  # (B, N−1): path b's own gradient (paths are independent)
        paths.append(q.detach().clone())
        actions.append(S.detach())
        grads.append(dS)
        if method == "newton":
            H = hessian(q.detach(), D, dt, m, V)  # (B, N−1, N−1) tridiagonal
            with torch.no_grad():
                x -= 0.2 * torch.linalg.solve(H, dS)
        else:
            optimiser.zero_grad()
            x.grad = dS
            optimiser.step()
    # n_iter + 1 states are stored: the start and the result of each of the n_iter updates
    return {"paths": torch.stack(paths), "S": torch.stack(actions), "grad": torch.stack(grads)}
