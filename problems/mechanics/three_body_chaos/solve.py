"""Adaptive Runge–Kutta (Dormand–Prince 5(4)): the step shrinks by itself during close encounters.

Each step computes a 5th- and a 4th-order answer from the same 7 stages; their difference estimates the
error. Steps that are too inaccurate are retried with a smaller dt, and quiet stretches get bigger steps.
"""

import math

import torch

from problems._common.progress import bar

from .equation import energy, rhs

# Butcher tableau of Dormand–Prince 5(4): stage s uses y + dt · A[s, r] k_r (sum over earlier stages r)
A = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],  # = the 5th-order weights
    ],
    dtype=torch.float64,
)
B5 = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=torch.float64)
B4 = torch.tensor([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40], dtype=torch.float64)


def dopri_step(f, y: torch.Tensor, k1: torch.Tensor, dt: float, A=A, B5=B5, B4=B4):
    """One Dormand–Prince trial step. Returns (y_new, k at y_new, error estimate), all shaped like y."""
    k = [k1]  # the stage slopes k_s, each shaped like y
    for s in range(1, 7):
        k.append(f(y + dt * torch.tensordot(A[s, :s], torch.stack(k), dims=1)))  # sum over earlier stages r
    k = torch.stack(k)  # (7, ...)
    y_new = y + dt * torch.tensordot(B5, k, dims=1)
    err = dt * torch.tensordot(B5 - B4, k, dims=1)  # 5th − 4th order answer
    return y_new, k[6], err  # k[6] was evaluated at y_new: reused as the next k1 ("first same as last")


def solve(
    m: torch.Tensor, x0: torch.Tensor, v0: torch.Tensor, G: float, t_max: float, tol: float, dt_out: float = 0.05
):
    """Return times (T,), positions (T, K, N, 2), velocities (T, K, N, 2) every dt_out, and the accepted step count.

    All runs share one step (the error is the largest over every run), so they stay on the same clock.
    Steps are trimmed so that they land exactly on each output time.
    """
    n_out = int(round(t_max / dt_out))
    y = torch.stack([x0, v0])  # (2, K, N, 2)
    f = lambda y: rhs(y, m, G)  # noqa: E731
    tableau = A.to(y.device), B5.to(y.device), B4.to(y.device)
    k1 = f(y)
    ys, t, dt, n_steps = [y], 0.0, 1e-3, 0
    with bar(n_out, "output times") as progress:
        for j in range(n_out):
            t_next = (j + 1) * dt_out
            while t < t_next - 1e-12:
                h = min(dt, t_next - t)
                y_new, k_new, err = dopri_step(f, y, k1, h, *tableau)
                ratio = (err.abs().max() / (tol * (1 + y.abs().max()))).item()  # ≤ 1 means accurate enough
                if ratio <= 1:  # accept
                    y, k1, t = y_new, k_new, t + h
                    n_steps += 1
                dt_new = h * min(5.0, max(0.2, 0.9 * ratio ** (-1 / 5))) if ratio > 0 else 5 * h  # error ∝ dt^5
                trimmed = h < dt and ratio <= 1  # a short step to hit t_next says nothing about the right dt
                dt = max(dt, dt_new) if trimmed else dt_new
            ys.append(y)
            progress.update(1)
    traj = torch.stack(ys)  # (T, 2, K, N, 2)
    times = torch.arange(n_out + 1, dtype=x0.dtype, device=x0.device) * dt_out
    return times, traj[:, 0], traj[:, 1], n_steps


def lyapunov_fit(times: torch.Tensor, dist: torch.Tensor, high: float = 0.1):
    r"""Fit ln|Δ| = c + λ t by least squares, from t = 0 until |Δ| exceeds `high` for good.

    |Δ| spikes at every close encounter (the twin passes a moment earlier or later), so the window ends
    only once the runs have separated for good: from then on |Δ| never drops below `high` again.
    Returns (λ, t_start, t_end, c); λ is nan if the runs never separate (e.g. δ = 0).
    """
    stays_above = dist.flip(0).cummin(0).values.flip(0) > high  # (T,) True from the point of no return on
    if dist[0] <= 0 or not stays_above.any():
        return math.nan, math.nan, math.nan, math.nan
    i1 = stays_above.nonzero()[0].item()
    t, y = times[: i1 + 1], dist[: i1 + 1].log()
    X = torch.stack([torch.ones_like(t), t], dim=1)  # (n, 2) least squares for [c, λ]
    c, lam = torch.linalg.lstsq(X, y[:, None]).solution[:, 0].tolist()
    return lam, t[0].item(), t[-1].item(), c


def ejection(xs: torch.Tensor, vs: torch.Tensor, m: torch.Tensor, G: float, r_min: float = 5.0):
    """First output step at which one body has escaped from the other two, as (step, body) or (None, None).

    Escape: the body is farther than r_min from the centre of mass of the other two, and its two-body energy
    relative to that pair (treated as one mass) is positive, so it never comes back.  xs, vs: (T, N, 2)
    """
    for a in range(len(m)):
        others = [b for b in range(len(m)) if b != a]
        M = m[others].sum()
        x_pair = torch.einsum("b,tbi->ti", m[others], xs[:, others]) / M  # (T, 2) centre of mass of the pair
        v_pair = torch.einsum("b,tbi->ti", m[others], vs[:, others]) / M
        r = (xs[:, a] - x_pair).norm(dim=-1)  # (T,)
        u = (vs[:, a] - v_pair).norm(dim=-1)
        e_rel = 0.5 * u**2 - G * (m[a] + M) / r  # energy per reduced mass
        gone = ((r > r_min) & (e_rel > 0)).nonzero()
        if len(gone):
            return gone[0].item(), a
    return None, None


def energy_error(xs: torch.Tensor, vs: torch.Tensor, m: torch.Tensor, G: float) -> torch.Tensor:
    """|E(t) − E(0)| / |E(0)| for every run.   (T, K, N, 2) → (T, K)"""
    e = energy(xs, vs, m, G)
    return (e - e[0]).abs() / e[0].abs()
