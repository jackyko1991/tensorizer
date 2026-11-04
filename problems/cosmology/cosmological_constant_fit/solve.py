r"""Two ways to fit: a batched χ² over a whole grid of universes, and gradient descent with autograd.

On the grid, H_0 enters only as a constant shift of every μ (μ = 5 log10(d_L H_0) + 25 − 5 log10 H_0), so its best
value is found analytically for every cell: the inverse-variance mean of the residuals. Autograd instead moves
(H_0, Ω_m, Ω_Λ) together downhill with Adam and polishes with L-BFGS; the Hessian of χ² at the minimum gives the
error bars (covariance = 2 H^{-1}).
"""

import math

import torch

from problems._common.progress import steps as progress_steps

from . import equation, tensors


def chi2_of(p: torch.Tensor, z, mu, sigma, nodes=None) -> torch.Tensor:
    """χ² of one universe p = (H0, Ω_m, Ω_Λ), differentiable in p."""
    nodes = tensors.nodes_for(z) if nodes is None else nodes
    d_L = equation.luminosity_distance(z, nodes, p[0], p[1], p[2])  # (N,)
    return equation.chi_squared(equation.distance_modulus(d_L), mu, sigma)


def chi2_grid(z, mu, sigma, Om, OL, chunk: int = 8):
    """χ² (A, B) of every universe on the Ω_m × Ω_Λ grid, minimised over H0, and that best H0 (A, B).

    Rows of the grid are processed `chunk` at a time to keep the (rows, B, N) residual tensor small.
    """
    nodes = tensors.nodes_for(z)
    w = sigma**-2  # (N,) inverse variances
    chi2_rows, h0_rows = [], []
    for start in range(0, len(Om), chunk):
        om = Om[start : start + chunk, None, None]  # (a, 1, 1)
        ol = OL[None, :, None]  # (1, B, 1)
        d1 = equation.luminosity_distance(z, nodes, torch.ones_like(om), om, ol)  # (a, B, N), H0 = 1
        r = mu - equation.distance_modulus(d1)  # residuals before the H0 shift
        shift = torch.einsum("abi,i->ab", r, w) / w.sum()  # best −5 log10 H0, the weighted mean residual
        chi2 = torch.einsum("abi,abi,i->ab", r - shift[..., None], r - shift[..., None], w)
        ok = equation.valid(nodes, om, ol) & (d1 > 0).all(-1)  # (a, B); closed models can wrap past the antipode
        chi2_rows.append(torch.where(ok, chi2, torch.full_like(chi2, math.inf)))
        h0_rows.append(10 ** (-shift / 5))
    return torch.cat(chi2_rows), torch.cat(h0_rows)


def fit(z, mu, sigma, start=(70.0, 1.0, 0.0), steps: int = 400, lr: float = 0.02, flat: bool = False) -> dict:
    """Adam from `start` (default: the matter-only Einstein–de Sitter universe), then L-BFGS to the minimum.

    The optimiser works on (H0/100, Ω_m, Ω_Λ) so all three move on the same scale. With flat=True, Ω_Λ = 1 − Ω_m.
    Returns the path (S, 3), χ² along it (S,), the best parameters (3,) and their 1σ errors (3,).
    """
    nodes = tensors.nodes_for(z)
    q = torch.tensor([start[0] / 100, start[1], start[2]], dtype=z.dtype, device=z.device, requires_grad=True)

    def params(q):
        OL = 1 - q[1] if flat else q[2]
        return torch.stack([100 * q[0], q[1], OL])

    def loss(q):
        return chi2_of(params(q), z, mu, sigma, nodes)

    opt = torch.optim.Adam([q], lr=lr)
    path, chi2 = [], []
    for _ in progress_steps(steps, "Adam"):
        opt.zero_grad()
        value = loss(q)
        path.append(params(q).detach().clone())
        chi2.append(value.item())
        value.backward()
        opt.step()

    lbfgs = torch.optim.LBFGS([q], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        value = loss(q)
        value.backward()
        return value

    lbfgs.step(closure)
    best = params(q).detach()
    path.append(best.clone())
    chi2.append(loss(q).item())

    free = [0, 1] if flat else [0, 1, 2]  # parameters the Hessian is taken over
    p_best = best[free].clone()

    def chi2_free(p):
        full = torch.stack([p[0], p[1], 1 - p[1]]) if flat else p
        return chi2_of(full, z, mu, sigma, nodes)

    hess = torch.autograd.functional.hessian(chi2_free, p_best)
    errors = torch.linalg.inv(hess).diagonal().mul(2).sqrt()  # covariance = 2 H⁻¹ for χ² = −2 ln L
    if flat:
        errors = torch.cat([errors, errors[1:]])  # σ(Ω_Λ) = σ(Ω_m) on the flat line
    return {"path": torch.stack(path), "chi2": torch.tensor(chi2), "best": best, "errors": errors}


def significance(chi2: torch.Tensor, Om: torch.Tensor, OL: torch.Tensor) -> dict:
    r"""How strongly the grid rejects Ω_Λ ≤ 0 and q_0 ≥ 0 (a decelerating universe).

    nσ = \sqrt{Δχ²} with Δχ² = min χ² inside the rejected region − global min (profile likelihood, one constraint).
    p_accel is the share of the likelihood e^{−χ²/2} (flat prior on the grid) with q_0 < 0.
    """
    om, ol = Om[:, None], OL[None, :]  # broadcast to the (A, B) grid
    q0 = equation.deceleration(om, ol).expand_as(chi2)
    best = chi2.min()
    i, j = divmod(int(chi2.argmin()), chi2.shape[1])
    no_lambda = chi2[(ol <= 0).expand_as(chi2)].min()
    decel = chi2[q0 >= 0].min()
    like = torch.exp(-(chi2 - best) / 2)  # 0 where χ² = inf
    return {
        "chi2_min": best.item(),
        "best_Om": Om[i].item(),
        "best_OL": OL[j].item(),
        "best_q0": q0[i, j].item(),
        "sigma_lambda": (no_lambda - best).clamp_min(0).sqrt().item(),
        "sigma_accel": (decel - best).clamp_min(0).sqrt().item(),
        "p_accel": (like[q0 < 0].sum() / like.sum()).item(),
    }
