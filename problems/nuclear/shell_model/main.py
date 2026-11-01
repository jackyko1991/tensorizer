"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    r, h = tensors.radial_grid(params["n_grid"], 18.0, dev)  # (N,) fm
    d2 = tensors.second_difference(len(r), h, dev)  # (N, N)
    ell, j = tensors.channels(8, dev)  # (K,) each
    r0, a = 1.27, params["diffuseness"]
    radius = r0 * params["mass_number"] ** (1 / 3)

    strengths = torch.linspace(0.0, params["v_so"], 40, dtype=torch.float64)  # (S,)
    energies = solve.sweep(r, d2, ell, j, params["v0"], strengths, radius, a, r0)  # (S, K, n_radial)

    for name, e in (("without spin–orbit", energies[0]), (f"V_so = {params['v_so']:.0f} MeV", energies[-1])):
        scheme = solve.level_scheme(e, ell, j)
        _, magic = solve.largest_gaps(scheme, 7)
        print(f"{name}: shell gaps after {sorted(magic.tolist())} nucleons")
    scheme = solve.level_scheme(energies[-1], ell, j)
    print("level      E (MeV)   2j+1   filled")
    for e, k, n, filled in zip(*(scheme[key].tolist() for key in ("energy", "channel", "radial", "filled"))):
        lk, jk = int(ell[k].item()), j[k].item()
        print(f"{tensors.label(n + 1, lk, jk):8s} {e:9.2f}   {int(2 * jk + 1):4d}   {int(filled):6d}")
    return plot.animate(strengths, energies, ell, j, out_dir / "shell_model.gif")


if __name__ == "__main__":
    cli(run, __file__)
