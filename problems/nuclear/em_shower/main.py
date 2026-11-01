"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import heitler_n_max, heitler_t_max


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    e0, e_crit = params["e0"] * 1000.0, params["e_crit"]  # MeV
    n_max, t_max = heitler_n_max(e0, e_crit), heitler_t_max(e0, e_crit)
    steps_per_x0 = 20
    depths = tensors.depth_axis(t_max + 8.0, steps_per_x0, dev)  # (T,) in X0

    stochastic = params["mode"] == "stochastic"
    result = solve.shower(tensors.primary(e0, dev), depths, steps_per_x0, e_crit, stochastic)
    other = solve.shower(tensors.primary(e0, dev), depths, steps_per_x0, e_crit, stochastic=not stochastic)
    reference = other["counts"].sum(dim=1)  # (T,) the other splitting mode, for comparison

    total = result["counts"].sum(dim=1)
    k = int(total.argmax())
    balance = (result["in_flight"] + result["deposited"] - e0).abs().max().item()
    print(f"E0 = {e0:.0f} MeV, Ec = {e_crit:.1f} MeV  ({params['mode']} splitting)")
    print(f"Heitler estimate: N_max ≈ E0/Ec = {n_max:.0f}   t_max ≈ ln(E0/Ec)/ln2 = {t_max:.2f} X0")
    print(f"simulated:        N_max = {total[k].item()}   at depth t = {depths[k].item():.2f} X0")
    print(f"particles created: {len(result['tracks'])}   max |Σ E_in_flight + E_deposited − E0| = {balance:.1e} MeV")
    other_label = "fixed length (split every X₀)" if stochastic else "stochastic length"
    return plot.animate(depths, result, reference, other_label, n_max, t_max, out_dir / "em_shower.gif")


if __name__ == "__main__":
    cli(run, __file__)
