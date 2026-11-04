r"""Fits: batched Fisher scoring for the mass scan and the toys, and autograd (Adam + L-BFGS) for the free-mass fit.

Fisher scoring (Newton's method with the expected Hessian) for p = (θ_k, n_s):
    J_{bnj} = ∂ν_{bn}/∂p_j = (w_n B_{bn} X_{nk},  S_{bn}),
    g_{bj} = J_{bnj} (1 − k_{bn}/ν_{bn}),     I_{bjl} = J_{bnj} J_{bnl} / ν_{bn}      (n summed),
    p ← p − I⁻¹ g,
run for every pseudo-experiment and every hypothesised mass at once.
"""

import torch

from problems._common.progress import steps as progress_steps

from . import equation


def _start(counts: torch.Tensor, x: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """Least-squares fit of ln(counts/width) by the polynomial: a good first guess for θ. → (B, K)"""
    target = torch.log((counts + 0.5) / widths)  # (B, N)
    return torch.linalg.lstsq(x, target.T).solution.T


def fisher_fit(counts, x, widths, shape=None, iterations: int = 12):
    """Maximum-likelihood θ (and signal yield n_s if a shape is given) for a batch of histograms.

    counts: (B, N), x: (N, K), widths: (N,), shape: (B, N) or None (background only).
    Returns θ (B, K), n_s (B,) and −ln L (B,).
    """
    theta = _start(counts, x, widths)
    n_s = torch.zeros(counts.shape[0], dtype=counts.dtype, device=counts.device)
    sig = torch.zeros_like(counts) if shape is None else shape.expand_as(counts)
    for _ in range(iterations):
        b = widths * equation.background(theta, x)  # (B, N)
        nu = b + n_s[:, None] * sig
        jac = torch.cat([b[..., None] * x, sig[..., None]], dim=-1)  # (B, N, K+1)
        if shape is None:
            jac = jac[..., :-1]
        g = torch.einsum("bnj,bn->bj", jac, 1 - counts / nu)
        info = torch.einsum("bnj,bnl->bjl", jac / nu[..., None], jac)  # I = Jᵀ diag(1/ν) J
        step = torch.linalg.solve(info, g)  # (B, K+1)
        theta = theta - step[:, : x.shape[1]]
        if shape is not None:
            n_s = n_s - step[:, -1]
    nu = widths * equation.background(theta, x) + n_s[:, None] * sig
    return theta, n_s, equation.nll(nu, counts)


def q0_scan(counts, x, edges, widths, masses, sigma: float):
    """Discovery statistic q₀ at every hypothesised mass for every histogram.  counts: (B, N) → (B, M)"""
    n_b, n_m = counts.shape[0], masses.shape[0]
    _, _, nll_b = fisher_fit(counts, x, widths)  # (B,) background only
    shapes = equation.signal_shape(edges, masses, sigma)  # (M, N)
    rep = counts[:, None, :].expand(n_b, n_m, -1).reshape(n_b * n_m, -1)  # (B·M, N)
    sig = shapes[None].expand(n_b, -1, -1).reshape(n_b * n_m, -1)
    _, n_s, nll_sb = fisher_fit(rep, x, widths, sig)
    q0 = 2 * (nll_b[:, None] - nll_sb.reshape(n_b, n_m))
    return torch.where(n_s.reshape(n_b, n_m) > 0, q0.clamp_min(0), torch.zeros_like(q0))


def q0_known_background(counts, b, s, iterations: int = 30) -> torch.Tensor:
    """q₀ when the background b (N,) is known: profile only the signal strength μ of ν = b + μ s (Newton in μ)."""
    mu = torch.zeros((), dtype=counts.dtype)
    for _ in range(iterations):
        nu = b + mu * s
        grad = (s * (1 - counts / nu)).sum()
        hess = (counts * s**2 / nu**2).sum()
        mu = mu - grad / hess
    q0 = 2 * (equation.nll(b, counts) - equation.nll(b + mu * s, counts))
    return q0 if mu > 0 else torch.zeros_like(q0)


def fit_free_mass(counts, x, edges, widths, sigma: float, m_start: float, n_sm: float, adam_steps: int = 150):
    """Fit θ, signal strength μ and m_H to one histogram with autograd; errors from the Hessian of −ln L.

    Adam (the path is kept for the animation) then L-BFGS polishes. n_sm is the Standard Model signal expectation,
    so μ = n_s / n_sm. Returns floats m_h, m_err, mu, mu_err, z (local, √q₀), theta (K,) and the path
    (S, K + 2) of (m_H, μ, θ).
    """
    counts = counts.reshape(1, -1)
    theta0, _, nll_b = fisher_fit(counts, x, widths)
    k = x.shape[1]
    # free parameters: θ (K), μ, (m_H − 125)/2  →  all of order one for Adam
    p = torch.cat([theta0[0], counts.new_tensor([0.5, (m_start - 125.0) / 2])]).requires_grad_(True)

    def loss(p):
        m_h = 125.0 + 2 * p[k + 1]
        shape = equation.signal_shape(edges, m_h, sigma)
        nu = equation.expected(p[:k], p[k] * n_sm, x, widths, shape)
        return equation.nll(nu, counts[0])

    opt = torch.optim.Adam([p], lr=0.03)
    path = []
    for _ in progress_steps(adam_steps, "Adam"):
        opt.zero_grad()
        value = loss(p)
        value.backward()
        opt.step()
        path.append(torch.cat([p.detach()[k + 1 :] * 2 + 125.0, p.detach()[: k + 1].roll(1)]))
    lbfgs = torch.optim.LBFGS([p], lr=1.0, max_iter=300, tolerance_grad=1e-10, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        value = loss(p)
        value.backward()
        return value

    lbfgs.step(closure)
    best = p.detach().clone()
    path.append(torch.cat([best[k + 1 :] * 2 + 125.0, best[: k + 1].roll(1)]))
    cov = torch.linalg.inv(torch.autograd.functional.hessian(loss, best))  # −ln L: covariance = H⁻¹
    q0 = 2 * (nll_b[0] - loss(best)).detach().clamp_min(0)
    return {
        "m_h": 125.0 + 2 * best[k + 1].item(),
        "m_err": 2 * cov[k + 1, k + 1].sqrt().item(),
        "mu": best[k].item(),
        "mu_err": cov[k, k].sqrt().item(),
        "z": equation.significance(q0).item(),
        "theta": best[:k],
        "path": torch.stack(path),  # (S, K + 2) rows (m_H, μ, θ_0 … θ_{K−1})
    }


def fit_resolution(edges, sm_signal, m_start: float = 125.0) -> tuple[float, float]:
    """Mass and width of a Gaussian fitted (least squares, autograd) to the simulated signal histogram."""
    total = sm_signal.sum()
    p = sm_signal.new_tensor([m_start, 1.5]).requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        model = total * equation.signal_shape(edges, p[0], p[1].abs())
        value = ((model - sm_signal) ** 2).sum()
        value.backward()
        return value

    opt.step(closure)
    return p[0].item(), p[1].abs().item()


def upcrossings(q0: torch.Tensor, level: float) -> torch.Tensor:
    """Number of times each scan q₀(m) crosses `level` upwards.  q0: (B, M) → (B,)"""
    above = q0 > level
    return (above[:, 1:] & ~above[:, :-1]).sum(-1)


def global_p_value(q0_toys: torch.Tensor, q0_obs: float, level: float = 0.5) -> dict:
    """Look-elsewhere effect from background-only toy scans.

    Direct estimate: the share of toys whose largest q₀ anywhere beats the observed one. Gross–Vitells (2010):
    p_global ≈ p_local + ⟨N_up(u₀)⟩ e^{−(q₀ − u₀)/2}, with the mean number of upcrossings counted in the toys.
    """
    q_max = q0_toys.max(-1).values
    n_up = upcrossings(q0_toys, level).double().mean()
    q = torch.tensor(q0_obs, dtype=torch.float64)
    p_local = equation.p_value(q)
    p_gv = p_local + n_up * torch.exp(-(q - level) / 2)
    return {
        "p_local": p_local.item(),
        "p_global_toys": (q_max >= q0_obs).double().mean().item(),
        "p_global_gv": p_gv.clamp(max=1).item(),
        "n_up": n_up.item(),
        "q_max": q_max,
    }
