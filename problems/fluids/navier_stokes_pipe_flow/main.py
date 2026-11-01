"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import flow_rate, hagen_poiseuille, wall_shear_stress

N, RHO = 40, 1000.0  # radial grid intervals, density of water (kg/m³)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    R = params["R"] / 1000  # pipe radius in m
    nu = params["nu"] * 1e-6  # kinematic viscosity in m²/s
    G, mu = params["G"], RHO * nu
    tau = R**2 / nu  # viscous diffusion time across the pipe (s)

    r, dr = tensors.grid(N, R, dev)  # (N+1,) from axis to wall
    L = tensors.radial_laplacian(r, dr)  # (N, N)
    times, u = solve.simulate(tensors.at_rest(N, dev), L, G, RHO, nu, dr, params["t_max"] * tau)  # (T,), (T, N+1)
    u_s = hagen_poiseuille(r, G, mu, R)  # (N+1,)
    releases, C = solve.dye_lines(times, u, release_every=0.02 * tau)

    Q, Q_s = flow_rate(u[-1], r), math.pi * G * R**4 / (8 * mu)
    print(
        f"R²/ν = {tau:.1f} s,  {len(times) - 1} RK4 steps,  "
        f"Reynolds number 2R·u_mean/ν = {2 * Q_s / (math.pi * R * nu):.1f}"
    )
    print(f"centreline speed: {u[-1, 0] * 1000:.3f} mm/s  (Hagen–Poiseuille G R²/4μ = {u_s[0] * 1000:.3f} mm/s)")
    print(f"flow rate: {Q * 1e9:.3f} mm³/s  (Hagen–Poiseuille π G R⁴/8μ = {Q_s * 1e9:.3f} mm³/s)")
    print(
        f"wall shear stress: {wall_shear_stress(u[-1], dr, mu) * 1000:.4f} mPa  "
        f"(steady G R/2 = {G * R / 2 * 1000:.4f} mPa)"
    )
    k63 = int((u[:, 0] > 0.63 * u_s[0]).int().argmax())
    print(f"centreline reaches 63 % of its final speed at t = {times[k63] / tau:.3f} R²/ν")
    return plot.animate(times, r, u, C, releases, u_s, R, tau, out_dir / "navier_stokes_pipe_flow.gif")


if __name__ == "__main__":
    cli(run, __file__)
