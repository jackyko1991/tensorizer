"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors

LENGTH = 80.0  # box side
X0, X_SCREEN, V0 = -15.0, 18.0, 40.0  # packet start, detector line, barrier height


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    k0 = params["k0"]
    X, Y, KX, KY, dx = tensors.grid(params["n_grid"], LENGTH, dev)
    dA = dx * dx
    if params["barrier"] == "disk":
        V = tensors.disk(X, Y, V0, params["slit_sep"])
    else:
        V = tensors.double_slit(X, Y, V0, params["slit_sep"], params["slit_width"])
    M = tensors.absorbing_mask(X, Y, LENGTH, width=8.0)
    psi0 = tensors.gaussian_packet(X, Y, X0, sx=3.0, sy=5.0, k0=k0, dA=dA)

    column = int(((X[:, 0] - X_SCREEN).abs()).argmin())  # grid index of the detector line
    t_max = (X_SCREEN - X0 + 12.0) / k0  # long enough for the packet to pass the screen
    result = solve.solve(psi0, V, KX, KY, dA, t_max, M=M, screen_column=column)

    print(f"packet energy E = {k0**2 / 2:.3f}, barrier V0 = {V0}, wavelength = {2 * math.pi / k0:.3f}")
    if params["barrier"] == "double_slit":
        D, d = X_SCREEN, params["slit_sep"]
        print(f"expected fringe spacing ~ lambda D / d = {2 * math.pi / k0 * D / d:.2f}")
    print(f"norm: start {result['norm'][0]:.6f}   end {result['norm'][-1]:.6f}  (absorbed at the edges)")
    print(f"probability through the detector line: {result['screen'][-1].sum() * dx:.4f}")
    return plot.animate(X, Y, V, result, X_SCREEN, out_dir / "scattering_2d.gif")


if __name__ == "__main__":
    cli(run, __file__)
