"""Main loop: Marcus rate curves for a batch of λ and T → surface-hopping simulation → compare."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import activation_energy, crossing_point, marcus_rate


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lam, T, H_ab = params["lam"], params["T"], params["H_ab"]
    kT = tensors.KB * T
    dG = tensors.driving_forces(-3 * lam, lam, 241, dev)  # (G,)
    lams = tensors.reorganisation_energies(lam, dev)  # (L,)
    temps = tensors.temperatures(T, dev)  # (T,)
    k = solve.rate_grid(dG, lams, tensors.KB * temps, H_ab)  # (L, T, G)
    print(f"Marcus rate on a grid of {len(lams)} λ × {len(temps)} T × {len(dG)} ΔG° = {tuple(k.shape)}")
    for i, lm in enumerate(lams.tolist()):
        g = int(k[i, 1].argmax())
        print(f"  λ = {lm:.2f} eV: fastest at ΔG° = {dG[g].item():+.3f} eV (expected -λ), k_max = {k[i, 1, g]:.3g} /ps")

    # stochastic check: Langevin motion of the solvent + Landau–Zener hops, one row per driving force
    omega, gamma = params["omega"], params["friction"]
    dt = 0.1 / omega  # ten steps per radian of solvent motion
    n_steps = 300 if fast_mode() else params["steps"]
    n_traj = 400 if fast_mode() else 2000
    dG_sim = tensors.simulated_driving_forces(lam, 9, dev)  # (D,)
    x, v = tensors.thermal_ensemble(len(dG_sim), n_traj, lam, kT, omega, dev)
    snaps = solve.snapshot_steps(n_steps, 6)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # tensors of 10⁴ numbers: starting threads every step costs more than it saves
    try:
        res = solve.hopping_dynamics(x, v, dG_sim, lam, kT, H_ab, omega, gamma, dt, n_steps, snaps)
    finally:
        torch.set_num_threads(threads)
    k_sim, k_err = solve.simulated_rates(res)
    k_th = marcus_rate(dG_sim, torch.tensor(lam), torch.tensor(kT), H_ab)
    print(
        f"\nsurface hopping: {len(dG_sim)} ΔG° × {n_traj} trajectories, {n_steps} steps × dt = {dt * 1e3:.2f} fs "
        f"= {n_steps * dt:.1f} ps, k_BT = {kT * 1e3:.1f} meV"
    )
    print("   ΔG° (eV)   x*      ΔG‡ (eV)   hops   P_LZ    k_sim (1/ps)        k_Marcus (1/ps)")
    for d in range(len(dG_sim)):
        g0 = dG_sim[d].item()
        print(
            f"   {g0:+.3f}   {crossing_point(lam, g0):+.3f}   {activation_energy(lam, g0):.3f}     "
            f"{int(res['hops'][d]):5d}  {res['p_lz'][d]:.3f}   {k_sim[d]:.3g} ± {k_err[d]:.2g}      {k_th[d]:.3g}"
        )
    ok = (res["hops"] > 0) & (k_th > 0)
    ratio = (k_sim[ok] / k_th[ok]).log10().abs().max().item()
    print(f"largest |log10(k_sim / k_Marcus)| = {ratio:.2f}  (Landau–Zener hops give the golden rule when P_LZ ≪ 1)")
    return plot.animate(dG, lams, temps, k, dG_sim, k_sim, k_err, res, snaps, dt, lam, T, out_dir / "marcus.gif")


if __name__ == "__main__":
    cli(run, __file__)
