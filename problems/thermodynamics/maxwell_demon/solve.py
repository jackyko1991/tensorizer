"""Solvers: equilibrium sampling, driven Langevin runs, the Szilard engine with a demon, Landauer erasure, Crooks."""

import math

import torch

from problems._common.progress import steps

from . import equation, tensors


def sample_equilibrium(n: int, p: float, side: float, gen, device, tilt: float = 0.0) -> torch.Tensor:
    """n positions drawn from e^{-V(x; p, s)}/Z by inverting the cumulative distribution on the grid (n,)."""
    grid = tensors.x_grid(device)
    rho = torch.exp(-equation.potential(grid, torch.tensor(p), torch.tensor(float(side)), tilt))
    cdf = torch.cumsum(rho, 0)
    cdf = cdf / cdf[-1]
    u = torch.rand(n, generator=gen, dtype=tensors.DTYPE).to(device)
    i = torch.searchsorted(cdf, u).clamp(1, len(grid) - 1)
    w = (u - cdf[i - 1]) / (cdf[i] - cdf[i - 1]).clamp_min(1e-300)
    return grid[i - 1] + w * (grid[i] - grid[i - 1])


def ramp(t: float, p0, p1, tau):
    """Linear protocol p(t) = p0 + (p1 - p0) min(t/tau, 1), per particle (B,)."""
    return p0 + (p1 - p0) * torch.clamp(t / tau, max=1.0)


def run_protocol(x, side, p0, p1, tau, t_total, gen, tilt=0.0, record=()):
    """Drive the partition (or piston) from p0 to p1 in time tau (all (B,)) and integrate overdamped Langevin.

    Work is the energy change at fixed x when p moves: W += V(x, p_{k+1}) - V(x, p_k). The path log ratio sums
    ln P[x_k -> x_{k+1}] - ln P[x_{k+1} -> x_k] over the steps (Euler proposal + Metropolis).
    record: step numbers to store.
    """
    dt = tensors.DT
    n_steps = max(1, math.ceil(t_total / dt - 1e-9))  # the protocol always reaches its end
    p = p0.clone()
    v = equation.potential(x, p, side, tilt)
    W = torch.zeros_like(x)
    log_path = torch.zeros_like(x)
    frames = {}
    record = set(record)
    if 0 in record:
        frames[0] = (x.clone(), p.clone(), W.clone())
    for k in steps(n_steps, "Langevin steps"):
        p_new = ramp((k + 1) * dt, p0, p1, tau)
        v_new = equation.potential(x, p_new, side, tilt)
        W += v_new - v  # work done on the particle by moving the wall
        f = equation.force(x, p_new, side, tilt)
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
        uni = torch.rand(x.shape, generator=gen, dtype=x.dtype).to(x.device)
        x, v, f, lr = equation.mala_step(x, v_new, f, p_new, side, dt, noise, uni, tilt)
        log_path += lr
        p = p_new
        if k + 1 in record:
            frames[k + 1] = (x.clone(), p.clone(), W.clone())
    return {"x": x, "p": p, "W": W, "log_path": log_path, "frames": frames, "n_steps": n_steps}


def partition_z(p, side, device) -> torch.Tensor:
    """Z(p, s) for one side or a batch of partition positions (P,)."""
    grid = tensors.x_grid(device)
    p = torch.as_tensor(p, dtype=tensors.DTYPE, device=device).reshape(-1)
    side = torch.as_tensor(side, dtype=tensors.DTYPE, device=device).reshape(-1).expand_as(p)
    return equation.partition_function(p, side, grid)


def _random_side(n: int, gen, device) -> torch.Tensor:
    return torch.where(torch.rand(n, generator=gen, dtype=tensors.DTYPE) < 0.5, -1.0, 1.0).to(device, tensors.DTYPE)


def szilard(n_per: int, eps, tau: float, gen, device, t_total=None, record=()):
    """Szilard engine with a noisy demon for every error rate eps_e at once (E groups of n_per engines).

    1. Equilibrium with the partition at x = a; the particle's true side s = -1 (left) or +1 (right), prob. 1/2.
    2. The demon reads m = s, wrong with probability eps.
    3. Feedback: the partition slides so that the side the demon believes in gets 2a(1 - eps) of the box (optimal).
    4. The partition is removed (its residual spring energy is returned, W_remove).
    """
    a = tensors.A
    eps = eps.to(device)
    E = len(eps)
    eps_b = eps.repeat_interleave(n_per)  # (B,)
    B = E * n_per
    side = _random_side(B, gen, device)
    x = torch.where(side < 0, sample_equilibrium(B, a, -1.0, gen, device), sample_equilibrium(B, a, 1.0, gen, device))
    wrong = torch.rand(B, generator=gen, dtype=tensors.DTYPE).to(device) < eps_b
    m = torch.where(wrong, -side, side)  # the demon's reading
    p0 = torch.full((B,), a, dtype=tensors.DTYPE, device=device)
    p1 = torch.where(m < 0, 2 * a * (1 - eps_b), 2 * a * eps_b)  # final partition position
    run = run_protocol(x, side, p0, p1, torch.full_like(p0, tau), t_total or tau, gen, record=record)
    W_remove = -equation.wall(side * (run["p"] - run["x"]))  # pulling the partition out returns its spring energy
    z0 = partition_z(a, -1.0, device) + partition_z(a, 1.0, device)  # Z of the partitioned box at the start
    p_fin = torch.cat([2 * a * (1 - eps), 2 * a * eps])  # (2E,) final positions for m = L, R
    z1 = (partition_z(p_fin, -1.0, device) + partition_z(p_fin, 1.0, device)).view(2, E)
    dF_groups = -torch.log(z1 / z0)  # (2, E) Delta F_m = -ln Z(p_f(m)) / Z(a)
    group = torch.arange(E, device=device).repeat_interleave(n_per)
    dF_m = torch.where(m < 0, dF_groups[0, group], dF_groups[1, group])
    info = equation.trajectory_information(~wrong, eps_b)
    return {"W_on": run["W"], "W_remove": W_remove, "dF_m": dF_m, "info": info, "side": side, "m": m,
            "wrong": wrong, "p1": p1, "frames": run["frames"], "eps": eps, "n_per": n_per}  # fmt: skip


def erasure(n_per: int, taus, gen, device, record=()):
    """Landauer erasure of a bit stored as the side of a partitioned box, for every protocol time tau_t at once.

    Start: equilibrium with the partition, side unknown (1 bit). Remove the partition, push a piston from x = 2a to a
    in time tau: the particle ends on the left (bit reset to 0). Delta F = ln[Z(left)+Z(right)] - ln Z(left) = ln 2.
    """
    a = tensors.A
    taus = taus.to(device)
    B = len(taus) * n_per
    tau_b = taus.repeat_interleave(n_per)
    side = _random_side(B, gen, device)
    x = torch.where(side < 0, sample_equilibrium(B, a, -1.0, gen, device), sample_equilibrium(B, a, 1.0, gen, device))
    W_remove = -equation.wall(side * (a - x))  # remove the partition
    piston = torch.full((B,), -1.0, dtype=tensors.DTYPE, device=device)  # the piston pushes from the right
    p0 = torch.full((B,), 2 * a, dtype=tensors.DTYPE, device=device)
    p1 = torch.full((B,), a, dtype=tensors.DTYPE, device=device)
    run = run_protocol(x, piston, p0, p1, tau_b, taus.max().item(), gen, record=record)
    z_left, z_right = partition_z(a, -1.0, device), partition_z(a, 1.0, device)
    dF = (torch.log(z_left + z_right) - torch.log(z_left)).item()
    success = (run["x"] < a + 4 / tensors.KAPPA**0.5).to(tensors.DTYPE).view(len(taus), -1).mean(1)
    return {"W": W_remove + run["W"], "dF": dF, "success": success, "side": side, "taus": taus,
            "frames": run["frames"], "n_per": n_per}  # fmt: skip


def crooks(n: int, tau: float, gen, device, record=()):
    """Forward: compress the empty box with a piston 2a -> a in time tau. Reverse: expand a -> 2a from equilibrium.

    Also returns sigma = ln P_F[x]/P_R[x~]: the log ratio of the transition probabilities along the path plus
    ln rho_eq(x_0) - ln rho_eq(x_N).
    """
    a = tensors.A
    s = torch.full((n,), -1.0, dtype=tensors.DTYPE, device=device)
    big, small = (torch.full((n,), v, dtype=tensors.DTYPE, device=device) for v in (2 * a, a))
    t = torch.full((n,), tau, dtype=tensors.DTYPE, device=device)
    z_big, z_small = partition_z(2 * a, -1.0, device), partition_z(a, -1.0, device)
    dF = (torch.log(z_big) - torch.log(z_small)).item()

    x0 = sample_equilibrium(n, 2 * a, -1.0, gen, device)
    fwd = run_protocol(x0, s, big, small, t, tau, gen, record=record)
    ln_rho0 = -equation.potential(x0, big, s) - torch.log(z_big)
    ln_rho_n = -equation.potential(fwd["x"], small, s) - torch.log(z_small)
    sigma = fwd["log_path"] + ln_rho0 - ln_rho_n

    x0r = sample_equilibrium(n, a, -1.0, gen, device)
    rev = run_protocol(x0r, s, small, big, t, tau, gen, record=record)
    return {"W_forward": fwd["W"], "W_reverse": rev["W"], "dF": dF, "sigma_path": sigma,
            "frames_forward": fwd["frames"], "frames_reverse": rev["frames"]}  # fmt: skip
