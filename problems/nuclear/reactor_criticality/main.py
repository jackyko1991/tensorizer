"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import analytic_k, critical_width, loss_operator, production_operator


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    width, n = params["width"], params["n_grid"]
    d, sa, nsf = params["diffusion"], params["sigma_a"], params["nu_sigma_f"]
    x, h = tensors.grid(width, n, dev)  # (N,) cm
    m = loss_operator(tensors.laplacian(n, h, dev), d, sa)  # (N, N)
    f = production_operator(n, nsf, dev)  # (N, N)

    ks, fluxes = solve.power_iteration(m, f, tensors.initial_guess(x, width), params["n_iter"])
    k_direct = solve.k_eigen(m, f).item()
    k_exact = analytic_k(width, d, sa, nsf)

    l_crit = critical_width(d, sa, nsf)
    widths = torch.linspace(0.4 * width, 2.0 * width, 40, dtype=torch.float64, device=dev)  # (S,)
    k_widths = solve.k_versus_width(widths, 100, d, sa, nsf)  # (S,)

    print(f"k∞ = νΣf/Σa = {nsf / sa:.4f}   critical width L_c = {l_crit:.2f} cm")
    print(f"slab L = {width:.1f} cm:  power iteration k = {ks[-1].item():.6f}  ({params['n_iter']} iterations)")
    print(f"                eigen solve      k = {k_direct:.6f}")
    print(f"                analytic νΣf/(Σa + D(π/L)²) = {k_exact:.6f}")
    state = "supercritical" if k_exact > 1 else "subcritical" if k_exact < 1 else "critical"
    print(f"→ the slab is {state}")
    return plot.animate(x, width, ks, fluxes, k_exact, widths, k_widths, l_crit, out_dir / "reactor_criticality.gif")


if __name__ == "__main__":
    cli(run, __file__)
