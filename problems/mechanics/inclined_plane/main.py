"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    mu_s, mu_k = params["mu_static"], max(0.0, min(params["mu_kinetic"], params["mu_static"]))  # μ_k ≤ μ_s
    g = tensors.gravity(params["g"], dev)
    theta = tensors.ramp_angles(params["n_ramps"], params["min_angle"], params["max_angle"])
    s, n = tensors.unit_vectors(theta, dev)

    a = equation.acceleration(g, s, n, mu_s, mu_k)  # (B, 2)
    sol = solve.solve(a, s, params["ramp_length"], params["t_max"])

    print(f"block stays at rest when θ < arctan(μ_s) = {math.degrees(math.atan(mu_s)):.1f}°")
    for deg, a_par in zip(theta.rad2deg().tolist(), sol["a_par"].tolist()):
        state = "slides" if a_par > 1e-6 else "at rest"
        print(f"angle {deg:5.1f}°  a along slope {a_par:6.3f} m/s²  ({state})")
    return plot.animate(sol, theta, params["ramp_length"], out_dir / "inclined_plane.gif")


if __name__ == "__main__":
    cli(run, __file__)
