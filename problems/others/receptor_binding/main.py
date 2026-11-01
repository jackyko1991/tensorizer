"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    names = tensors.variant_names()
    k_on, k_off = tensors.rate_constants(dev)  # (B,), (B,)
    L = tensors.ligand_concentrations(params["log10_L"], dev)  # (C,) nM, last entry = sensorgram
    depletion = params["depletion"] == "on"
    R_T = params["R_T"]
    y0 = tensors.initial_state(R_T, len(names), len(L), dev)  # (2, B, C)

    times, traj = solve.solve(y0, L, k_on, k_off, params["t_assoc"], params["t_dissoc"], depletion)
    theta = traj[:, 1] / R_T  # (T, B, C) fraction of receptors bound

    K_D = equation.dissociation_constant(k_on, k_off)  # (B,) nM
    L_fine = torch.logspace(-2, 4, 200, device=dev)
    if depletion:
        theta_eq = equation.occupancy_with_depletion(L_fine, K_D, R_T)  # (B, 200)
    else:
        theta_eq = equation.occupancy(L_fine, K_D)

    t_half = equation.half_life(k_off)
    i_wash = int(round(params["t_assoc"] / (times[1] - times[0]).item()))
    print(f"sensorgram ligand L = {L[-1].item():.3g} nM   ·   depletion {params['depletion']}")
    for b, name in enumerate(names):
        print(
            f"{name:<19} K_D = {K_D[b].item():9.3g} nM   t½ = {t_half[b].item():8.3g} s   "
            f"θ at wash = {theta[i_wash, b, -1].item():.3f}   θ at end = {theta[-1, b, -1].item():.3f}"
        )
    return plot.animate(times, theta, L, L_fine, theta_eq, names, params["t_assoc"], out_dir / "receptor_binding.gif")


if __name__ == "__main__":
    cli(run, __file__)
