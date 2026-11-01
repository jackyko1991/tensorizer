"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import numpy as np
import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import pmns


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    angles = [math.radians(params[k]) for k in ("theta12", "theta13", "theta23", "delta_cp")]
    u = pmns(*angles)  # (3, 3) complex
    m2 = tensors.squared_masses(params["dm21"] * 1e-5, params["dm31"] * 1e-3, dev)  # (3,) eV²
    dist = tensors.baselines(10.0, params["l_max"], 2500, dev)  # (K,) km
    alpha = tensors.FLAVOURS.index(params["flavour"])

    p = solve.oscillate(u, m2, dist, params["energy"], alpha)  # (K, 3)

    beam_l, beam_y, beam_u = tensors.beam(400, 10.0, params["l_max"])
    index = torch.from_numpy(np.searchsorted(dist.cpu().numpy(), beam_l.numpy()).clip(max=len(dist) - 1))
    flavour = solve.detected_flavour(p.cpu()[index], beam_u)  # (n,)

    print("PMNS |U|:")
    for row, name in zip(u.abs().tolist(), ["e  ", "mu ", "tau"]):
        print(f"  {name} " + "  ".join(f"{x:.3f}" for x in row))
    print(f"unitarity: max |U U† − 1| = {(u @ u.conj().T - torch.eye(3)).abs().max().item():.1e}")
    print(f"max |Σ_β P − 1| over all baselines = {(p.sum(1) - 1).abs().max().item():.1e}")
    for target in (300.0, 1300.0, 12700.0):
        k = int(torch.argmin((dist - target).abs()))
        probs = "  ".join(f"P({params['flavour']}→{f}) = {p[k, b].item():.3f}" for b, f in enumerate(tensors.FLAVOURS))
        print(f"L = {dist[k].item():7.0f} km   {probs}")
    return plot.animate(
        dist, p, beam_l, beam_y, flavour, alpha, params["energy"], out_dir / "neutrino_oscillations.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
