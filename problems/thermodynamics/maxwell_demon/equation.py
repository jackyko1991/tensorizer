"""Physics → tensors: the potential, overdamped Langevin with a Metropolis correction, work, information.

Every function acts elementwise on a batch $b$ of independent particles (no sums over particles except the averages
the averages). Shapes in comments; B = batch.
"""

import math

import torch

from . import tensors


def wall(u: torch.Tensor) -> torch.Tensor:
    """Soft wall energy κ/2 · relu(u)² for a penetration u (any shape)."""
    return 0.5 * tensors.KAPPA * torch.relu(u) ** 2


def potential(x, p, side, tilt: float = 0.0):
    """V(x; p, s) = box walls + partition at p felt from side s (−1 left, +1 right, 0 = no partition) + tilt·x.

    x, p, side: (B,) or broadcastable. A particle on the left (s = −1) is pushed back when x > p;
    on the right when x < p.
    """
    return wall(-x) + wall(x - 2 * tensors.A) + wall(side * (p - x)) + tilt * x


def force(x, p, side, tilt: float = 0.0):
    """F = −∂V/∂x (B,)."""
    k = tensors.KAPPA
    return k * torch.relu(-x) - k * torch.relu(x - 2 * tensors.A) + k * side * torch.relu(side * (p - x)) - tilt


def log_kernel(y, x, f_x, dt: float):
    """log of the Euler–Maruyama (Onsager–Machlup) transition density x → y, without the constant:
    −(y−x−F dt)²/4D dt."""
    return -((y - x - f_x * dt) ** 2) / (4 * dt)


def mala_step(x, v_x, f_x, p, side, dt, noise, uniform, tilt=0.0):
    """One overdamped Langevin step x → x + F dt + √(2 dt) ξ, accepted with the Metropolis–Hastings probability so that
    e^{−V/kT} is exactly stationary. Returns the new x, V, F and the path log ratio ln P[x→x'] − ln P[x'→x] (B,)."""
    y = x + f_x * dt + math.sqrt(2 * dt) * noise
    v_y, f_y = potential(y, p, side, tilt), force(y, p, side, tilt)
    log_q_f, log_q_b = log_kernel(y, x, f_x, dt), log_kernel(x, y, f_y, dt)  # proposal densities x→y and y→x
    log_r = (v_x - v_y) + log_q_b - log_q_f  # Metropolis–Hastings log ratio
    accept = torch.log(uniform) < log_r
    # path weight of this step forward over reversed: P[x→y]/P[y→x] = q(y|x) min(1, r) / q(x|y) min(1, 1/r)
    log_path = log_q_f + log_r.clamp(max=0) - log_q_b - (-log_r).clamp(max=0)
    x_new = torch.where(accept, y, x)
    return x_new, torch.where(accept, v_y, v_x), torch.where(accept, f_y, f_x), torch.where(accept, log_path, 0 * x)


def partition_function(p, side, grid, tilt: float = 0.0):
    """Z(p, s) = ∫ e^{−V(x; p, s)} dx by the trapezoid rule; p, side: (P,), grid (X,) → (P,)."""
    v = potential(grid[None, :], p[:, None], side[:, None], tilt)  # (P, X)
    return torch.trapezoid(torch.exp(-v), grid, dim=1)


def mutual_information(eps):
    """I = ln 2 + ε ln ε + (1−ε) ln(1−ε): what a measurement with error rate ε tells about a fair bit (nats)."""
    xlogx = lambda q: torch.where(q > 0, q * torch.log(q.clamp_min(1e-300)), 0 * q)  # noqa: E731
    return math.log(2.0) + xlogx(eps) + xlogx(1 - eps)


def trajectory_information(correct, eps):
    """Stochastic mutual information I = ln P(m|s)/P(m) for each run: ln 2(1−ε) if the demon was right,
    else ln 2ε (B,)."""
    return torch.where(correct, torch.log(2 * (1 - eps)), torch.log((2 * eps).clamp_min(1e-300)))


def crooks_line(w_f, minus_w_r, bins: int = 24, min_count: int = 40):
    """Least-squares line through ln P_F(W) − ln P_R(−W) on the bins both histograms populate → (slope, intercept)."""
    lo, hi = max(w_f.min(), minus_w_r.min()).item(), min(w_f.max(), minus_w_r.max()).item()
    edges = torch.linspace(lo, hi, bins + 1, dtype=w_f.dtype)
    h_f = torch.histc(w_f, bins, lo, hi) / len(w_f)
    h_r = torch.histc(minus_w_r, bins, lo, hi) / len(minus_w_r)
    keep = (h_f * len(w_f) >= min_count) & (h_r * len(minus_w_r) >= min_count)
    centres = 0.5 * (edges[1:] + edges[:-1])[keep]
    y = torch.log(h_f[keep] / h_r[keep])
    design = torch.stack([centres, torch.ones_like(centres)], 1)
    coef = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    return coef[0].item(), coef[1].item()
