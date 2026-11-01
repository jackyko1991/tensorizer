"""Levenberg–Marquardt on the reprojection error, with the Jacobian from autograd (torch.func.jacrev) and the
normal equations solved either densely or by the Schur complement (eliminating the points first)."""

import torch

from problems._common.progress import steps

from .equation import cost, normal_equations, residuals
from .tensors import unpack


def residual_function(full_fixed, free, n_cams, observed, focal, centre):
    """r(θ): the flat residual vector (2CP,) as a function of the free unknowns θ only.

    The gauge-fixed numbers stay at their values in full_fixed; E is the 0/1 matrix that places θ into the full
    vector, full = full_fixed (free entries zeroed) + E θ.
    """
    E = torch.eye(len(free), dtype=full_fixed.dtype, device=full_fixed.device)[:, free]  # (N_full, n_free)
    base = torch.where(free.to(full_fixed.device), 0.0, full_fixed)

    def fn(theta: torch.Tensor) -> torch.Tensor:
        full = base + torch.einsum("Nn,n->N", E, theta)
        omega, t, X = unpack(full, n_cams)
        return residuals(omega, t, X, observed, focal, centre).reshape(-1)

    return fn


def jacobian(fn, theta: torch.Tensor) -> torch.Tensor:
    """J_{mn} = ∂r_m / ∂θ_n by reverse-mode autograd, vectorised over the rows m: (2CP, n_free)."""
    return torch.func.jacrev(fn)(theta)


def dense_step(H: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Solve H Δ = -g for all unknowns at once (cost grows like n³)."""
    return torch.linalg.solve(H, -g)


def schur_step(H: torch.Tensor, g: torch.Tensor, n_cam: int) -> torch.Tensor:
    """Solve H Δ = -g using the block structure H = [[U, W], [Wᵀ, V]] (cameras, points).

    Each residual involves one camera and ONE point, so V is block diagonal: P separate 3 × 3 blocks, inverted
    independently. Eliminating the points leaves the small reduced camera system
        S Δ_c = -g_c + W V⁻¹ g_p,   S = U - W V⁻¹ Wᵀ,
    then each point's update follows on its own: Δ_p = V_p⁻¹ (-g_p - Wᵀ Δ_c).
    """
    U, W = H[:n_cam, :n_cam], H[:n_cam, n_cam:]  # (n_c, n_c), (n_c, 3P)
    n_pts = (H.shape[0] - n_cam) // 3
    V_blocks = H[n_cam:, n_cam:].reshape(n_pts, 3, n_pts, 3).diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # (P, 3, 3)
    V_inv = torch.linalg.inv(V_blocks)  # P small inverses instead of one 3P × 3P
    Wp = W.reshape(n_cam, n_pts, 3)  # (n_c, P, 3)
    g_c, g_p = g[:n_cam], g[n_cam:].reshape(n_pts, 3)
    S = U - torch.einsum("apj,pjk,bpk->ab", Wp, V_inv, Wp)  # p, j, k summed
    rhs = -g_c + torch.einsum("apj,pjk,pk->a", Wp, V_inv, g_p)
    d_c = torch.linalg.solve(S, rhs)
    d_p = torch.einsum("pjk,pk->pj", V_inv, -g_p - torch.einsum("apk,a->pk", Wp, d_c))
    return torch.cat([d_c, d_p.reshape(-1)])


def levenberg_marquardt(fn, theta: torch.Tensor, n_cam: int, iterations: int, method: str, lam: float = 1e-3):
    """Minimise ½|r(θ)|². Each iteration: J by autograd, H = JᵀJ, g = Jᵀr, try (H + λ diag H) Δ = -g; accept the
    step and shrink λ if the cost drops, otherwise grow λ and retry (at most 10 tries).

    Returns θ after every iteration (entry 0 = start), the costs, the damping used, and the last H.
    """
    history = {"theta": [theta.clone()], "cost": [cost(fn(theta)).item()], "lam": []}
    H = None
    for _ in steps(iterations, "LM iterations"):
        r = fn(theta)
        J = jacobian(fn, theta)  # (m, n)
        H, g = normal_equations(J, r)
        current = cost(r).item()
        for _try in range(10):
            A = H + lam * torch.diag(torch.diagonal(H))  # Marquardt's scaling: damp each unknown by its own curvature
            delta = schur_step(A, g, n_cam) if method == "schur" else dense_step(A, g)
            trial = cost(fn(theta + delta)).item()
            if trial < current:
                theta, lam = theta + delta, max(lam / 3, 1e-9)
                break
            lam *= 4
        history["theta"].append(theta.clone())
        history["cost"].append(cost(fn(theta)).item())
        history["lam"].append(lam)
    return theta, history, H
