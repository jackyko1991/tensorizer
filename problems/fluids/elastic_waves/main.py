"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import stable_time_step, wave_speeds

N, H, BORDER, RHO = 181, 1.0, 20, 1.0  # grid cells per side, cell size, absorbing border width, density
F0 = 0.08  # peak frequency of the source wavelet


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lam, mu = params["lam"], params["mu"]
    c_p, c_s = wave_speeds(lam, mu, RHO)
    dt = stable_time_step(H, c_p, c_s, params["courant"])
    u, u_prev = tensors.at_rest(N, dev)  # (2, N, N) each
    force = tensors.point_force(N, params["direction"], 1.5, dev)  # (2, N, N)
    damping = tensors.sponge(N, BORDER, 0.08, dev)  # (N, N)

    result = solve.simulate(u, u_prev, force, damping, H, dt, lam, mu, RHO, F0, params["t_max"])

    t0 = 1.2 / F0  # the wavelet peaks at t0
    times = result["times"]
    print(f"c_P = sqrt((λ+2μ)/ρ) = {c_p:.3f},  c_S = sqrt(μ/ρ) = {c_s:.3f},  c_P/c_S = {c_p / c_s:.3f}")
    print(f"dt = {dt:.3f} (Courant number {params['courant']}),  {len(times) - 1} leapfrog steps")
    if params["direction"] == 0.0:
        a, b = int(len(times) * 0.6), len(times) - 1
        dtt = (times[b] - times[a]).item()
        cp_m = (solve.front_radius(result["div"][b], "x") - solve.front_radius(result["div"][a], "x")) * H / dtt
        cs_m = (solve.front_radius(result["curl"][b], "y") - solve.front_radius(result["curl"][a], "y")) * H / dtt
        print(f"measured front speeds: P {cp_m:.2f} (along the force), S {cs_m:.2f} (across it)")
    return plot.animate(result, c_p, c_s, t0, BORDER, params["direction"], out_dir / "elastic_waves.gif")


if __name__ == "__main__":
    cli(run, __file__)
