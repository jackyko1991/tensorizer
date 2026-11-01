"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import analytic_height, drain_time

G = 9.81


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    h0, D = params["h0"], params["tank_diameter"]
    A = tensors.circle_area(D)  # tank cross-section (m²)
    d = tensors.hole_diameters(params["n_tanks"], params["d_min"] / 100, params["d_max"] / 100, dev)  # (B,) in m
    a = tensors.circle_area(d)  # (B,) hole areas
    h_start = tensors.initial_levels(len(d), h0, dev)  # (B,)

    T = drain_time(a, A, h0, G)  # (B,) exact drain times
    times, levels = solve.solve(h_start, a, A, G, t_max=1.05 * T.max().item())
    exact = analytic_height(times, a, A, h0, G)  # (T, B)

    T_num = solve.numerical_drain_time(times, levels)
    for db, Tb, Tn in zip(d.tolist(), T.tolist(), T_num.tolist()):
        print(f"hole {100 * db:4.1f} cm   drain time {Tn:7.1f} s   (exact (A/a) sqrt(2 h0/g) = {Tb:7.1f} s)")
    print(f"largest deviation from the exact h(t): {(levels - exact).abs().max():.2e} m")
    return plot.animate(times, levels, exact, d, D, h0, G, out_dir / "torricelli_draining.gif")


if __name__ == "__main__":
    cli(run, __file__)
