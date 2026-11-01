"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve

T1, PHI = 1.0, math.pi / 2  # nearest-neighbour coupling (sets the units) and the Haldane phase
WIDTH, HEIGHT = 40.0, 24.0  # flake size in units of the resonator spacing


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    t2, M, R = params["t2"], params["mass"], params["defect_radius"]
    M_trivial = 2 * 3 * math.sqrt(3) * t2 * math.sin(PHI)  # twice the critical mass: a trivial gap of similar size

    # 1. topology of the infinite lattice: Chern number of the lower band (FHS on a k-grid)
    topo = solve.chern_number(T1, t2, PHI, M, 48, dev)
    triv = solve.chern_number(T1, t2, PHI, M_trivial, 48, dev)
    print(f"critical mass 3√3 t2 sin φ = {3 * math.sqrt(3) * t2:.3f}")
    for name, mass, r in [("lattice 1", M, topo), ("lattice 2", M_trivial, triv)]:
        C = round(r["C"], 4) + 0.0  # + 0.0 turns -0.0 into 0.0
        print(f"{name} (M = {mass:.2f}): Chern number C = {C:+.4f}, bulk gap {r['gap']:.3f}")
    print(f"∫Ω d²k / 2π = {topo['omega'].sum().item() * topo['cell'] / (2 * math.pi):+.4f}")

    # 2. zigzag ribbon: edge states inside the gap, and the direction they travel on the bottom edge
    bands = solve.ribbon_bands(24, T1, t2, PHI, M, 161, dev)
    bands_triv = solve.ribbon_bands(24, T1, t2, PHI, M_trivial, 161, dev)
    window = 0.3 * max(topo["gap"], 0.05)  # frequencies well inside the bulk gap
    v, k_c = solve.bottom_edge_mode(bands, window)
    print(f"bottom-edge state: k_c·√3 = {k_c * math.sqrt(3):.3f}, group velocity dE/dk = {v:+.3f}")

    # 3. finite flake with a notch cut into the bottom edge: the same edge light in both lattices
    n_times = 30 if fast_mode() else 160
    flake = solve.edge_transport(
        T1, t2, PHI, (M, M_trivial), params["disorder"], R, v, k_c, window, WIDTH, HEIGHT, n_times, dev
    )
    pos, intensity, times, transmitted = flake["pos"], flake["intensity"], flake["times"], flake["transmitted"]
    print(f"{len(pos)} resonators, the edge packet keeps {flake['kept']:.0%} of the input spot (in-gap part)")
    print(
        f"transmitted past the notch at t_max: lattice 1 {transmitted[0, -1]:.2f},  lattice 2 {transmitted[1, -1]:.2f}"
    )
    return plot.animate(
        pos, intensity, times, transmitted, topo, triv["C"], bands, bands_triv, M, M_trivial, flake["notch"], R,
        out_dir / "topological_photonics.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
