"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import closest_approach, rutherford_angle, rutherford_cross_section, scattering_angle


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    energy, m = params["energy"], tensors.ALPHA_MASS
    k = tensors.coupling(params["z_target"])
    d = k / energy  # head-on distance of closest approach, fm
    start, b_max = 30 * d, params["b_max"] * d

    b_show = tensors.display_impact_parameters(15, min(3.0, params["b_max"]) * d)
    b_beam = tensors.beam_impact_parameters(params["n_beam"], b_max)
    y0 = tensors.initial_state(torch.cat([b_show, b_beam]), energy, k, start, dev)  # (S + B, 2, 2)

    v_inf = math.sqrt(2 * energy / m)
    n_steps = params["n_steps"]
    dt = 2.2 * start / v_inf / n_steps  # long enough for every alpha to fly back out to ~30 d
    paths, y_end, detect = solve.simulate(y0, k, m, dt, n_steps, len(b_show), r_detect=5 * d)

    theta = scattering_angle(y_end[len(b_show) :, 1])  # (B,) final angles of the beam
    detect = detect[len(b_show) :]
    theta_grid = torch.linspace(rutherford_angle(torch.tensor(b_max), energy, k).item(), math.pi, 200)
    flux = params["n_beam"] / (math.pi * b_max**2)  # alphas per fm² of beam
    expected = flux * rutherford_cross_section(theta_grid, energy, k)  # dN/dΩ per sr

    print(f"k = {k:.1f} MeV fm   head-on closest approach d = k/E = {d:.1f} fm")
    for b in [0.5 * d, d, 2 * d]:
        b_t = torch.tensor([b], dtype=torch.float64)
        print(
            f"b = {b:6.1f} fm   θ = {math.degrees(rutherford_angle(b_t, energy, k).item()):6.1f}°"
            f"   r_min = {closest_approach(b_t, energy, k).item():6.1f} fm"
        )
    print(f"fraction scattered beyond 90°: {(theta > math.pi / 2).double().mean().item():.4f}")
    first = int((start - 5.5 * d) / (v_inf * dt))  # the drawn alphas enter the view
    last = min(n_steps, first + int(3 * 5.5 * d / (v_inf * dt)))
    return plot.animate(paths, first, last, theta, detect, theta_grid, expected, d, out_dir / "rutherford.gif")


if __name__ == "__main__":
    cli(run, __file__)
