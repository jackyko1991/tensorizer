"""The search: a batch of candidate profiles, one per trial exponent λ, trained as one tensor.

Profiles: Gauss–Newton (Levenberg–Marquardt) on the expansion coefficients, the second-order optimiser the 2025 paper
uses to reach high precision; every member of the batch takes its own step in one batched least-squares solve.
Exponent: the best residual at fixed λ, the funnel, has a cusp at every admissible λ, so plain gradient steps on λ
stall next to it. A batch scan over a grid of λ finds the funnels, and a golden-section search, with the profile
re-trained at every trial λ, pins each one down. For CCF a final joint Gauss–Newton step trains profile and λ
together, with the Jacobian (including ∂R/∂λ) from autograd.
"""

import torch

from problems._common.progress import steps

from . import equation, tensors


def gauss_newton(residual, C: torch.Tensor, lam: torch.Tensor, iters: int, keep_path: bool = False) -> tuple:
    """Levenberg–Marquardt on every row of C (B, K) at fixed λ (B,). residual(C, λ, need_jac) → R (B, M) and
    J = ∂R/∂C (B, M, K) (None when need_jac is False).
    Each step solves min ‖J d + R‖² + μ‖D d‖² (D² = diag JᵀJ) by least squares; μ shrinks after a successful step
    and grows (and the step is retried once) after a failed one, per member. Returns C, ‖R‖ (B,) and the ‖R‖
    history (iters + 1, B); with keep_path also the coefficients after every step (iters + 1, B, K)."""
    B, K = C.shape
    mu = torch.full((B,), 1e-3, dtype=C.dtype, device=C.device)
    R, J = residual(C, lam, True)
    hist, path = [R.norm(dim=1)], [C]
    for _ in range(iters):
        D = torch.einsum("bmk,bmk->bk", J, J)  # diag JᵀJ, repeated m summed
        rhs = torch.cat([-R, torch.zeros_like(C)], 1)[..., None]
        todo = torch.ones(B, dtype=torch.bool, device=C.device)
        for _ in range(2):  # one retry with a larger μ for the members whose step failed
            A = torch.cat([J, torch.diag_embed((mu[:, None] * D + 1e-14).sqrt())], 1)  # (B, M + K, K)
            d = torch.linalg.lstsq(A, rhs).solution[..., 0]
            R_new, _ = residual(C + d, lam, False)
            better = todo & (R_new.norm(dim=1) < R.norm(dim=1))
            C = torch.where(better[:, None], C + d, C)
            mu = torch.where(better, (mu / 3).clamp_min(1e-14), torch.where(todo, mu * 4, mu))
            todo = todo & ~better
            if not todo.any():
                break
        R, J = residual(C, lam, True)
        hist.append(R.norm(dim=1))
        if keep_path:
            path.append(C)
    if keep_path:
        return C, hist[-1], torch.stack(hist), torch.stack(path)
    return C, hist[-1], torch.stack(hist)


def golden_section(residual, lo: torch.Tensor, hi: torch.Tensor, C: torch.Tensor, n_iter: int, n_fit: int, label: str):
    """Pin down λ in each bracket [lo, hi] (B,) by golden-section search on the funnel: at every trial λ the profiles
    are re-trained for n_fit Gauss–Newton steps (warm start). Every member has its own bracket; all are updated at
    once with torch.where. Returns the best λ, its coefficients, ‖R‖ and the λ history (n_iter + 1, B)."""
    phi = (5**0.5 - 1) / 2
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    C1, f1, _ = gauss_newton(residual, C, x1, n_fit)
    C2, f2, _ = gauss_newton(residual, C, x2, n_fit)
    hist = [torch.where(f1 < f2, x1, x2)]
    for _ in steps(n_iter, label):
        left = f1 < f2  # the minimum lies in [lo, x2]: x1 becomes the new x2
        lo, hi = torch.where(left, lo, x1), torch.where(left, x2, hi)
        x_new = torch.where(left, hi - phi * (hi - lo), lo + phi * (hi - lo))
        x_keep, f_keep = torch.where(left, x1, x2), torch.where(left, f1, f2)
        C_keep = torch.where(left[:, None], C1, C2)
        C_new, f_new, _ = gauss_newton(residual, C_keep, x_new, n_fit)
        x1, x2 = torch.where(left, x_new, x_keep), torch.where(left, x_keep, x_new)
        f1, f2 = torch.where(left, f_new, f_keep), torch.where(left, f_keep, f_new)
        C1, C2 = torch.where(left[:, None], C_new, C_keep), torch.where(left[:, None], C_keep, C_new)
        hist.append(torch.where(f1 < f2, x1, x2))
    best = f1 < f2
    return torch.where(best, x1, x2), torch.where(best[:, None], C1, C2), torch.where(best, f1, f2), torch.stack(hist)


def local_minima(res: torch.Tensor) -> torch.Tensor:
    """Indices of the interior local minima of a funnel res (B,), deepest first."""
    i = torch.nonzero((res[1:-1] < res[:-2]) & (res[1:-1] < res[2:])).flatten() + 1
    return i[torch.argsort(res[i])]


# ---------------------------------------------------------------- Burgers
N_MODES, N_COL = 64, 220


def burgers_residual_fn(level: int, device: torch.device):
    r = tensors.chebyshev_points(N_COL, device)
    return lambda C, lam, need_jac=True: equation.burgers_batch(C, lam, r, level, need_jac)


def burgers_funnel_history(lams: torch.Tensor, level: int, iters: int = 40) -> tuple:
    """Train one profile per λ of the grid (B,) at fixed λ, all from P = −r (c = 0): the coefficients after every
    Gauss–Newton step (iters + 1, B, K) and the funnel after every step (iters + 1, B)."""
    C0 = torch.zeros(len(lams), N_MODES, dtype=lams.dtype, device=lams.device)
    _, _, hist, path = gauss_newton(burgers_residual_fn(level, lams.device), C0, lams, iters, keep_path=True)
    return path, hist


def burgers_funnel(lams: torch.Tensor, level: int, iters: int = 40) -> torch.Tensor:
    """Best residual ‖R‖ at each fixed λ (B,): the funnel."""
    return burgers_funnel_history(lams, level, iters)[1][-1]


def burgers_search(lam0: torch.Tensor, level: int, width: float = 0.03, n_golden: int = 24, n_fit: int = 20) -> dict:
    """Pin down λ near each starting value lam0 (B,): golden-section search on the funnel in [λ0 − w, λ0 + w], the
    profile re-trained (warm start) at every trial λ. level 0: no constraint; level 1: c₃ = 0; level 2: c₃ = c₅ = 0."""
    fn = burgers_residual_fn(level, lam0.device)
    C, _, _ = gauss_newton(fn, torch.zeros(len(lam0), N_MODES, dtype=lam0.dtype, device=lam0.device), lam0, 40)
    lam, C, res, hist = golden_section(fn, lam0 - width, lam0 + width, C, n_golden, n_fit, f"λ search, level {level}")
    C, res, _ = gauss_newton(fn, C, lam, 20)  # converge the profile at the final λ
    return {"lam": lam, "coef": C, "res": res, "lam_hist": hist}


def project_profile(fn, lam: float, r: torch.Tensor, n_modes: int) -> torch.Tensor:
    """Least-squares coefficients c with y P(r) ≈ fn(y) at the points r (P = −r + Σ c_k b_k): (K,)."""
    lam_t = torch.tensor(lam, dtype=r.dtype, device=r.device)
    y = tensors.L_BURGERS * (r ** (-2 * (1 + lam_t)) - 1).sqrt()
    phi, _ = tensors.burgers_features(r, 1 / (2 * (1 + lam_t)), n_modes // tensors.N_FAMILIES)
    return torch.linalg.lstsq(phi, (fn(y) / y + r)[:, None]).solution[:, 0]


# ---------------------------------------------------------------- CCF
CCF_J, CCF_M, CCF_COL = 3, 10, 200


def ccf_residual_fn(device: torch.device):
    y = tensors.ccf_points(CCF_COL, device)
    return lambda A, lam, need_jac=True: equation.ccf_batch(A, lam, y, CCF_J, CCF_M, need_jac)


def ccf_funnel_history(lams: torch.Tensor, iters: int = 40) -> tuple:
    """CCF profiles trained at each fixed λ of the grid (B,): coefficients (B, K), funnel history (iters + 1, B)."""
    A0 = tensors.ccf_initial_coefficients(lams, CCF_J, CCF_M)
    A, _, hist = gauss_newton(ccf_residual_fn(lams.device), A0, lams, iters)
    return A, hist


def joint_polish(A: torch.Tensor, lam: torch.Tensor, iters: int) -> tuple:
    """Gauss–Newton on (a, λ) together for every member; the Jacobian, including the column ∂R/∂λ (λ enters the
    exponents s = jβ + m of the basis), comes from autograd (vmap of jacrev)."""
    y = tensors.ccf_points(CCF_COL, A.device)

    def f(p):  # one member: p = (a₁ … a_K, λ) → R (N + 1,)
        return equation.ccf_residual(p[:-1], p[-1], y, CCF_J, CCF_M)

    jac = torch.func.vmap(torch.func.jacrev(f))

    def residual(P, _, need_jac=True):
        return torch.func.vmap(f)(P), jac(P) if need_jac else None

    P, res, hist = gauss_newton(residual, torch.cat([A, lam[:, None]], 1), lam, iters)
    return P[:, :-1], P[:, -1], res, hist


def ccf_search(
    lam0: torch.Tensor, width: float = 0.06, n_golden: int = 16, n_fit: int = 40, polish: int = 8, warm: int = 60
) -> dict:
    """Golden-section search for λ around lam0 (B,), then a joint Gauss–Newton polish of profile and λ together."""
    fn = ccf_residual_fn(lam0.device)
    A, _, warm_hist = gauss_newton(fn, tensors.ccf_initial_coefficients(lam0, CCF_J, CCF_M), lam0, warm)
    lam, A, res, lam_hist = golden_section(fn, lam0 - width, lam0 + width, A, n_golden, n_fit, "CCF λ search")
    A, res, fit_hist = gauss_newton(fn, A, lam, 3 * n_fit)  # converge the profile at the found λ
    A, lam, res, polish_hist = joint_polish(A, lam, polish)
    y = tensors.ccf_points(CCF_COL, lam0.device)
    omega = torch.stack([equation.ccf_fields(a, lv, y, CCF_J, CCF_M)["omega"] for a, lv in zip(A, lam)], 1)  # (N, B)
    return {
        "lam": lam,
        "coef": A,
        "res": res,
        "lam_history": torch.cat([lam_hist, lam[None]]),
        "residual_history": torch.cat([warm_hist, fit_hist, polish_hist]),
        "omega": omega,
        "y": y,
    }
