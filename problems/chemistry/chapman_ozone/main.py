"""Main loop: altitudes → rate coefficients → backward-Euler network S·r(c) from sunrise → steady ozone layer → GIF."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import ATOMS, SPECIES, S, chapman_approx, steady_state


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    n_levels = 10 if fast else params["levels"]
    mu = math.cos(math.radians(params["sza"]))
    z = tensors.altitudes(n_levels, dev)  # (A,)
    times = tensors.time_grid(10.0 ** params["log10_t_end"], 30 if fast else 300)  # (T,)
    print(f"Chapman network at {n_levels} altitudes, 15–60 km, sun at {params['sza']:.0f}° from the zenith")
    print("S (rows " + ", ".join(SPECIES) + "; columns reactions 1–4):")
    print(S.numpy())
    print(f"oxygen atoms per species a = {ATOMS.tolist()},  a·S = {(ATOMS @ S).tolist()}  (every reaction conserves O)")

    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # 3 × 3 solves per level: one thread is far steadier than many on a busy CPU
    try:
        c, (j1, j3, k2, k4, M) = solve.solve(z, times, mu)  # (T, A, 3)
    finally:
        torch.set_num_threads(threads)
    atoms = c @ ATOMS.to(c)  # (T, A) oxygen atoms per cm³
    print(
        f"{len(times) - 1} backward-Euler steps to t = {times[-1]:.0e} s; max relative change of O atoms "
        f"{((atoms - atoms[0]) / atoms[0]).abs().max():.1e}"
    )

    O2 = c[-1, :, 1]
    O_ss, O3_ss = steady_state(j1, j3, k2, k4, M, O2)
    approx = chapman_approx(j1, j3, k2, k4, M, O2)
    fast_tau, slow_tau = solve.lifetimes(c[-1], j1, j3, k2, k4, M)
    print(" z/km   j1/s⁻¹   j3/s⁻¹   [O3] final   exact SS   √-formula   [O]/[O3]   τ_fast/s   τ_Ox/s")
    for a in range(0, n_levels, max(1, n_levels // 9)):
        print(
            f"{z[a]:5.1f}  {j1[a]:8.1e} {j3[a]:8.1e}  {c[-1, a, 2]:10.3e} {O3_ss[a]:10.3e} {approx[a]:10.3e}"
            f"   {c[-1, a, 0] / c[-1, a, 2]:8.1e}  {fast_tau[a]:8.1e}  {slow_tau[a]:8.1e}"
        )
    done = slow_tau < times[-1] / 30  # levels that have had many odd-oxygen lifetimes to settle
    err = ((c[-1, done, 2] - O3_ss[done]) / O3_ss[done]).abs().max()
    print(f"levels with τ_Ox < t_end / 30: final [O3] matches the exact steady state to {err:.1e}")
    peak = int(c[-1, :, 2].argmax())
    column = (c[-1, :, 2].sum() * (z[1] - z[0]) * 1e5 / 2.687e16).item()
    print(
        f"ozone peak {c[-1, peak, 2]:.2e} cm⁻³ at {z[peak]:.1f} km; column above 15 km {column:.0f} DU "
        f"(observed: peak ≈ 5e12 cm⁻³ near 20–25 km, total ≈ 300 DU)"
    )
    return plot.animate(times, z, c, O3_ss, fast_tau, slow_tau, out_dir / "chapman_ozone.gif")


if __name__ == "__main__":
    cli(run, __file__)
