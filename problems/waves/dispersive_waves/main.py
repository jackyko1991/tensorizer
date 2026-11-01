"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    g, sigma = tensors.G, params["width"]
    k0 = 2 * math.pi / params["wavelength"]
    x, k, dx = tensors.grid(2048, dev)
    a0 = tensors.wave_packet(x, tensors.X0, sigma, k0)

    times, a = solve.solve(a0, k, g, params["t_max"])
    xbar = solve.centroid(x, a)

    v_p, v_g = equation.phase_velocity(k0, g), equation.group_velocity(k0, g)
    print(f"phase velocity: theory {v_p:.3f} m/s, measured at the packet centre {solve.crest_speed(x, a, times):.3f} m/s")
    print(f"group velocity: theory {v_g:.3f} m/s, envelope centre moved at {solve.envelope_speed(times, xbar):.3f} m/s")
    E = equation.wave_energy(a)
    print(f"relative change of sum |a|^2: {((E - E[0]).abs().max() / E[0]).item():.1e}")
    return plot.animate(times, x, a, xbar, k0, v_p, v_g, tensors.X0, sigma, out_dir / "dispersive_waves.gif")


if __name__ == "__main__":
    cli(run, __file__)
