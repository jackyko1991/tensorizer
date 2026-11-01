"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import invariant_mass, metric, rest_frame_momentum


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    title, big_m, m1, m2, name1, name2 = tensors.DECAYS[params["decay"]]
    eta = metric(dev)  # (4, 4)
    parent = tensors.parent(big_m, params["p_parent"], dev)  # (4,)
    theta = tensors.decay_angles(90, dev)  # (B,)
    p_star = rest_frame_momentum(big_m, m1, m2)
    rest1, rest2 = tensors.rest_frame_daughters(theta, p_star, big_m, m1, m2)  # (B, 4) each

    lab1, lab2 = solve.to_lab(parent, rest1, rest2)  # (B, 4) each
    m_inv = solve.reconstruct(lab1, lab2, eta)  # (B,)

    gamma, beta = parent[0].item() / big_m, parent[3].item() / parent[0].item()
    print(f"{title}:  M = {big_m:.4f} GeV, p = {params['p_parent']:.2f} GeV  →  γ = {gamma:.3f}, β = {beta:.4f}")
    print(f"rest-frame momentum p* = {p_star:.4f} GeV")
    mass1, mass2 = invariant_mass(lab1, eta).mean().item(), invariant_mass(lab2, eta).mean().item()
    print(f"masses after the boost: {name1} {mass1:.5f}  {name2} {mass2:.5f} GeV")
    print(f"reconstructed parent mass: {m_inv.min().item():.6f} … {m_inv.max().item():.6f} GeV")
    print(f"max |p1 + p2 − P| = {(lab1 + lab2 - parent).abs().max().item():.1e} GeV")
    for k in (0, 45, 89):
        angle = math.degrees(theta[k].item())
        print(f"θ* = {angle:5.1f}°   lab energies {lab1[k, 0].item():.3f} + {lab2[k, 0].item():.3f} GeV")
    return plot.animate(
        theta, rest1, rest2, lab1, lab2, parent, m_inv, (title, name1, name2), out_dir / "four_momentum.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
