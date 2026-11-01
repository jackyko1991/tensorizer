"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lam, c, h = params["wavelength"], tensors.C, tensors.H
    X, Y = tensors.grid(dev)
    K = tensors.stencil(dev)
    gamma = tensors.damping(dev)
    blob, s1, s2 = tensors.sources(X, Y, params["separation"])
    dt = equation.time_step(h, c, params["courant"])

    times, u, intensity = solve.solve(blob, gamma, K, lam, c, h, dt, params["t_max"])

    D = equation.path_difference(X, Y, s1, s2)
    print(f"time step dt = {dt:.3f}  (Courant number {c * dt / h:.2f} <= 1/sqrt(2))")
    print(f"sources {params['separation']:g} cells apart = {params['separation'] / lam:.2f} wavelengths")
    far = Y > tensors.SOURCE_Y + 3 * lam  # away from the near field of the sources
    I_node, I_anti = solve.nodal_contrast(intensity[-1], D, lam, (gamma == 0) & far)
    print(f"mean intensity on predicted nodal lines {I_node:.3g}, on antinodal lines {I_anti:.3g}")
    return plot.animate(times, u, intensity, D, s1, s2, lam, tensors.BORDER, out_dir / "ripple_tank.gif")


if __name__ == "__main__":
    cli(run, __file__)
