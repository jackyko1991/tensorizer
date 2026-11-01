"""Main loop: monomers → population balance (step + living growth) → kinetic Monte Carlo → compare → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

SEED = 0


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    p_max = params["p_max"]
    n0 = params["n_monomers"] // (10 if fast else 1)
    L = tensors.chain_length_cap(p_max)
    x = tensors.lengths(L, dev)  # (L,)
    idx, mask = tensors.shift_index(L, dev)  # (L, L)

    # --- population balance: both mechanisms in one (2, L) tensor, clock s = X_n - 1 ---
    t0 = time.perf_counter()
    s_max = p_max / (1 - p_max)  # Carothers: X_n - 1 = p / (1 - p)
    s, traj = solve.population_balance(tensors.initial_state(L, dev), idx, mask, s_max, 0.25 if fast else 0.1)
    t_ode = time.perf_counter() - t0

    # frames evenly spaced in conversion p (not in s, which races at the end)
    p_frames = torch.linspace(0, p_max, 90, dtype=torch.float64, device=dev)
    picks = torch.searchsorted(s, (p_frames / (1 - p_frames)).clamp_max(s[-1])).clamp_max(len(s) - 1).unique()
    s_k, P_k = s[picks], traj[picks]  # (K,), (K, 2, L)
    p_k = s_k / (1 + s_k)  # conversion of the step-growth row at each frame

    # --- kinetic Monte Carlo of step growth at the same conversions ---
    t0 = time.perf_counter()
    events = [int(round(p.item() * n0)) for p in p_k]
    kmc = solve.kinetic_monte_carlo(n0, events, L, SEED).to(dev)  # (K, L) chains per initial monomer
    t_kmc = time.perf_counter() - t0
    p_kmc = torch.tensor(events, dtype=torch.float64, device=dev) / n0

    xn, _, disp = equation.averages(P_k, x)  # (K, 2) each
    xn_mc, _, disp_mc = equation.averages(kmc, x)  # (K,)
    flory = equation.flory_number(x, p_k[:, None])  # (K, L) chains per initial monomer is (1-p) n(x)
    flory_err = (P_k[:, 0] - (1 - p_k[:, None]) * flory).abs().max().item()
    poisson = equation.poisson(x, s_k[:, None])  # (K, L)
    poisson_err = (P_k[:, 1] - poisson).abs().max().item()
    m_end = equation.moments(traj[-1], x)  # (2, 3)

    print(f"A–B step growth and living chain growth, chain lengths x = 1 … {L}, "
          f"RK4 over {len(s) - 1} steps in {t_ode:.1f} s; KMC with {n0:,} monomers in {t_kmc:.1f} s")  # fmt: skip
    print("   p      X_n: ODE  Carothers   KMC     Đ step: ODE   1 + p    KMC     Đ living: ODE  1+ν/(1+ν)²  1+1/X_n")
    shown = sorted({int(torch.argmin((p_k - q).abs())) for q in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, p_max) if q <= p_max})
    for k in shown:
        p, nu = p_k[k].item(), s_k[k].item()
        step = f"{xn[k, 0].item():8.2f} {equation.carothers(p):9.2f} {xn_mc[k].item():8.2f}"
        disp_step = f"{disp[k, 0].item():10.3f} {1 + p:8.3f} {disp_mc[k].item():7.3f}"
        disp_living = f"{disp[k, 1].item():12.4f} {1 + nu / (1 + nu) ** 2:10.4f} {1 + 1 / xn[k, 1].item():9.4f}"
        print(f"  {p:.3f}   {step}   {disp_step}   {disp_living}")
    print(f"population balance vs exact solutions: max |P - N₀(1-p)²p^(x-1)| = {flory_err:.1e},   "
          f"max |P - Poisson| = {poisson_err:.1e}")  # fmt: skip
    print(f"conserved: monomer units Σ x P_x = {m_end[0, 1].item():.6f} (step growth, per monomer), "
          f"living chains Σ P_x = {m_end[1, 0].item():.6f} (per initiator)")  # fmt: skip

    return plot.animate(
        x, p_k, s_k, P_k, poisson, kmc, p_kmc, xn, disp, xn_mc, disp_mc, out_dir / "polymerisation_mwd.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
