"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import critical_coupling, local_order, ott_antonsen_r


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    n = 300 if fast else params["n_oscillators"]
    width, k_max = params["width"], params["k_max"]
    n_k = 8 if fast else params["n_couplings"]
    t_relax = 10.0 if fast else params["t_relax"]
    dt = 0.1

    k = tensors.couplings(k_max, n_k, dev)  # (B, 1) one population per coupling
    sweeps = {}
    for dist in ("lorentzian", "gaussian"):
        omega = tensors.frequencies(dist, n, width, dev, params["seed"]).float()  # (N,)
        _, r = solve.sweep(omega, k, t_relax, dt, params["seed"] + 1)  # r (B, T + 1)
        r_late = r[:, r.shape[1] // 2 :].mean(1)  # time average after the transient
        k_c = critical_coupling(dist, width)
        sweeps[dist] = (r_late.cpu(), k_c)
        below = r_late[k[:, 0] < 0.8 * k_c]
        print(f"{dist}: N = {n}, width {width}, K_c = 2/(π g(0)) = {k_c:.3f}")
        print(f"  below 0.8 K_c: mean r = {below.mean():.3f}  (finite-size level 1/√N = {1 / math.sqrt(n):.3f})")
        if dist == "lorentzian":
            exact = ott_antonsen_r(k[:, 0].double(), width).float()
            above = k[:, 0] > 1.3 * k_c
            err = (r_late[above] - exact[above]).abs().max()
            print(f"  above 1.3 K_c: max |r − √(1 − K_c/K)| = {err:.3f}  (Ott–Antonsen, exact for N → ∞)")
            for kk, rr, ee in list(zip(k[:, 0].tolist(), r_late.tolist(), exact.tolist()))[:: max(1, n_k // 6)]:
                print(f"    K = {kk:5.2f}: r = {rr:.3f}   Ott–Antonsen {ee:.3f}")

    dist = params["distribution"]
    omega = tensors.frequencies(dist, n, width, dev, params["seed"] + 2).float()
    ramp_time = 40.0 if fast else params["ramp_time"]
    k_t, thetas, r_t = solve.ramp(omega, k_max, ramp_time, dt, 10 if fast else 60, params["seed"] + 3)
    print(f"slow ramp of K from 0 to {k_max} over t = {ramp_time:.0f} ({dist}): final r = {r_t[-1]:.3f}")

    x, theta_ring = solve.chimera(n=128 if fast else 256, t_max=30.0 if fast else 300.0, dt=0.05, seed=params["seed"])
    loc = local_order(theta_ring, 5)
    print(
        f"chimera ring (identical oscillators, A = 0.995, α = π/2 − 0.18): local order ranges from "
        f"{loc.min():.2f} (drifting part) to {loc.max():.2f} (locked part)"
    )
    return plot.animate(
        omega.cpu(),
        k_t,
        thetas.cpu(),
        r_t.cpu(),
        k[:, 0].cpu(),
        sweeps,
        width,
        dist,
        (x, theta_ring),
        out_dir / "kuramoto.gif",
    )


if __name__ == "__main__":
    cli(run, __file__)
