"""Rate curves on a broadcast grid, and Langevin dynamics with Landau–Zener surface hopping for every ΔG° at once."""

import math

import torch

from problems._common.progress import steps

from .equation import crossing_point, landau_zener, marcus_rate, solvent_force


def rate_grid(dG: torch.Tensor, lam: torch.Tensor, kT: torch.Tensor, H_ab: float) -> torch.Tensor:
    """k_ltg for every reorganisation energy, temperature and driving force.   (L, T, G)"""
    return marcus_rate(dG[None, None, :], lam[:, None, None], kT[None, :, None], H_ab)


def hopping_dynamics(
    x: torch.Tensor,
    v: torch.Tensor,
    dG: torch.Tensor,
    lam: float,
    kT: float,
    H_ab: float,
    omega: float,
    gamma: float,
    dt: float,
    n_steps: int,
    snapshots: list[int],
    seed: int = 1,
) -> dict:
    """Underdamped Langevin motion of the solvent coordinate on the current diabatic surface (BAOAB, mass-weighted).

    x, v (D, N): one row per driving force ΔG°_d (dG (D,)). Each time a trajectory passes x* it switches surface
    with the Landau–Zener probability. The forward rate is (number of R → P hops) / (time spent on R).
    Returns hop counts (D,), time on R (D,) in ps, mean P_LZ per passage (D,), and snapshots (S, D, N) of x and state.
    """
    gen = torch.Generator().manual_seed(seed)
    x_star = crossing_point(lam, dG)[:, None]  # (D, 1)
    state = torch.zeros_like(x)  # 0 = reactant (electron on the donor), 1 = product
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt((1 - c1**2) * kT / (2 * lam)) * omega  # thermal velocity kick
    hops = torch.zeros(len(dG), device=x.device)
    t_react = torch.zeros(len(dG), device=x.device)
    p_sum = torch.zeros(len(dG), device=x.device)
    n_pass = torch.zeros(len(dG), device=x.device)
    snap_x, snap_s = [], []
    for step in steps(n_steps, "Langevin steps"):
        x_old = x
        t_react += dt * (1 - state).sum(1)
        v = v + 0.5 * dt * solvent_force(x, state, omega)  # B
        x = x + 0.5 * dt * v  # A
        v = c1 * v + c2 * torch.randn(v.shape, generator=gen).to(v.device)  # O
        x = x + 0.5 * dt * v  # A
        # passage through the crossing during this step → Landau–Zener hop
        crossed = (x_old - x_star) * (x - x_star) < 0
        p = landau_zener((x - x_old).abs() / dt, lam, H_ab)
        hop = crossed & (torch.rand(x.shape, generator=gen).to(x.device) < p)
        hops += (hop & (state == 0)).sum(1)
        p_sum += (p * crossed).sum(1)
        n_pass += crossed.sum(1)
        state = torch.where(hop, 1 - state, state)
        v = v + 0.5 * dt * solvent_force(x, state, omega)  # B, on the (possibly new) surface
        if step + 1 in snapshots:
            snap_x.append(x.clone())
            snap_s.append(state.clone())
    return {
        "hops": hops,
        "t_react": t_react,
        "p_lz": p_sum / n_pass.clamp_min(1),
        "snap_x": torch.stack(snap_x),
        "snap_state": torch.stack(snap_s),
    }


def simulated_rates(result: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """k = hops / time on R with a Poisson error bar √hops / time.   (D,), (D,)"""
    k = result["hops"] / result["t_react"]
    return k, result["hops"].sqrt() / result["t_react"]


def snapshot_steps(n_steps: int, n: int) -> list[int]:
    """Up to n steps spread geometrically from step 10 to the last: early times show the hopping start."""
    return sorted({round(10 * (n_steps / 10) ** (k / (n - 1))) for k in range(n)})
