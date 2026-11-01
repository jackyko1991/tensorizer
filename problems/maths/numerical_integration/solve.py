"""Apply the three rules for every N to every integrand, and fit the convergence order of each error curve."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import gauss_legendre, integral, simpson, trapezoid

METHODS = ["trapezoid", "Simpson", "Gauss–Legendre"]
FLOOR = 1e-15  # below this the error is float64 round-off, not the rule


def rule(method: int, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights of one rule for one N."""
    if method == 0:
        return trapezoid(n, device)
    if method == 1:
        return simpson(n, device)
    return gauss_legendre(tensors.jacobi_matrix(n, device))


def all_rules(ns: list[int], device: torch.device) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """rules[m][i] = (nodes, weights) of method m with N = ns[i]. Built once, reused by the plot."""
    return [[rule(m, n, device) for n in ns] for m in steps(len(METHODS), "rules")]


def errors(rules: list, kink: float, alpha: float) -> torch.Tensor:
    """|I_rule − I_exact| for every method, N and integrand: (3 methods, len(ns), 3 integrands)."""
    device = rules[0][0][0].device
    exact = tensors.exact(kink, alpha).to(device)
    err = torch.empty(len(rules), len(rules[0]), len(exact), dtype=torch.float64, device=device)
    for m, per_n in enumerate(rules):
        for i, (x, w) in enumerate(per_n):
            err[m, i] = (integral(w, tensors.integrands(x, kink, alpha)) - exact).abs()
    return err.clamp(min=FLOOR / 10)


def orders(ns: list[int], err: torch.Tensor) -> torch.Tensor:
    """Fitted p in error ∝ N^(−p) for every method and integrand, from the points with N ≥ 16 above round-off.

    err (3, M, 3) → (3, 3); NaN when fewer than 3 points are usable (Gauss on the smooth integrand hits round-off).
    """
    n = torch.tensor(ns, dtype=torch.float64, device=err.device)
    logn = torch.log(n)
    p = torch.full(err.shape[::2], float("nan"), dtype=torch.float64, device=err.device)
    for m in range(err.shape[0]):
        for f in range(err.shape[2]):
            keep = (n >= 16) & (err[m, :, f] > 1e3 * FLOOR)
            if keep.sum() >= 3:
                x, y = logn[keep] - logn[keep].mean(), torch.log(err[m, keep, f])
                p[m, f] = -(x @ (y - y.mean())) / (x @ x)
    return p


def exponential_rate(ns: list[int], err: torch.Tensor) -> float:
    """ρ in error ∝ ρ^(−N), fitted to the Gauss–Legendre error on the smooth integrand above round-off."""
    n = torch.tensor(ns, dtype=torch.float64)
    e = err[2, :, 0].cpu()
    keep = e > 1e3 * FLOOR
    x, y = n[keep] - n[keep].mean(), torch.log(e[keep])
    return float(torch.exp(-(x @ (y - y.mean())) / (x @ x)))
