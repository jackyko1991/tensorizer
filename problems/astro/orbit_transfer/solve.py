"""Fly the whole batch with RK4, burning at the scheduled apsides, and compare transfer costs over radius ratios."""

import math

import torch

from problems._common.integrators import integrate

from . import equation


def fly(y0: torch.Tensor, mu: float, burn_t: torch.Tensor, burn_dv: torch.Tensor, t_end: float, dt: float):
    """Coast between burns with RK4 and apply each impulsive burn exactly on time.

    y0: (B, 2, 2), burn_t / burn_dv: (B, K). The run is split at every burn time of any spacecraft; each piece uses a
    step close to dt that ends exactly on the next burn. Returns times (T,), trajectory (T, B, 2, 2) and the burn
    positions (B, K, 2). At a burn time the trajectory stores the state just after the burn.
    """
    events = sorted({float(t) for t in burn_t.flatten()} | {t_end})
    burn_pos = torch.zeros(*burn_t.shape, 2, dtype=y0.dtype, device=y0.device)
    times, pieces, y = [], [], y0
    for t0, t1 in zip(events[:-1], events[1:]):
        now = (burn_t - t0).abs() < 1e-9 * max(1.0, t_end)  # (B, K)  which burns happen at t0
        burn_pos[now] = y[:, 0].unsqueeze(1).expand_as(burn_pos)[now]
        y = equation.apply_burn(y, (burn_dv * now).sum(dim=1))  # the padded Hohmann burn adds Δv = 0
        n = max(1, math.ceil((t1 - t0) / dt))
        piece = integrate(lambda s, t: equation.rhs(s, mu), y, (t1 - t0) / n, n)  # (n + 1, B, 2, 2)
        pieces.append(piece[:-1])
        times.append(t0 + torch.arange(n, dtype=y0.dtype, device=y0.device) * (t1 - t0) / n)
        y = piece[-1]
    pieces.append(y[None])
    times.append(torch.tensor([t_end], dtype=y0.dtype, device=y0.device))
    return torch.cat(times), torch.cat(pieces), burn_pos


def total_dv_curves(R: torch.Tensor, rb_ratio: float) -> dict[str, torch.Tensor]:
    """Total Δv / v_c1 for r1 = 1, μ = 1 over the radius ratios R (N,): Hohmann, bi-elliptic with r_b = rb_ratio · r2,
    and the bi-parabolic limit r_b → ∞."""
    one = torch.ones_like(R)
    dv1, dv2, _ = equation.hohmann(one, R, 1.0)
    b1, b2, b3, _ = equation.bi_elliptic(one, R, rb_ratio * R, 1.0)
    return {
        "hohmann": dv1.abs() + dv2.abs(),
        "bi-elliptic": b1.abs() + b2.abs() + b3.abs(),
        "bi-parabolic": equation.bi_parabolic(one, R, 1.0),
    }


def bisect(f, lo: float, hi: float, n_iter: int = 60) -> float:
    """Root of a scalar function that changes sign between lo and hi."""
    f_lo = f(lo)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crossover_any_rb() -> float:
    """r2/r1 above which a far enough r_b makes the bi-elliptic transfer cheaper: Hohmann = bi-parabolic (≈ 11.94)."""

    def gap(R: float) -> float:
        one, r2 = torch.tensor(1.0, dtype=torch.float64), torch.tensor(R, dtype=torch.float64)
        dv1, dv2, _ = equation.hohmann(one, r2, 1.0)
        return float(dv1 + dv2 - equation.bi_parabolic(one, r2, 1.0))

    return bisect(gap, 2.0, 50.0)


def crossover_every_rb() -> float:
    """r2/r1 above which every r_b > r2 beats Hohmann (≈ 15.58).

    At r_b = r2 the bi-elliptic transfer is the Hohmann transfer; it becomes cheaper for every r_b once raising r_b
    a little lowers the cost, i.e. once d(Δv_total)/d r_b < 0 at r_b = r2. Autograd gives that derivative.
    """

    def slope(R: float) -> float:
        r2 = torch.tensor(R, dtype=torch.float64)
        rb = r2.clone().requires_grad_(True)
        dv1, dv2, dv3, _ = equation.bi_elliptic(torch.tensor(1.0, dtype=torch.float64), r2, rb, 1.0)
        (grad,) = torch.autograd.grad(dv1 + dv2 - dv3, rb)  # Δv₃ < 0 is a braking burn, so its size is -Δv₃
        return float(grad)

    return bisect(slope, 2.0, 50.0)
