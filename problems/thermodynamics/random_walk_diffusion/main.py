"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    D, n_steps = params["D"], params["n_steps"]
    t = tensors.times(params["t_max"], n_steps, dev)  # (T,)
    dt = (t[1] - t[0]).item()
    kicks = tensors.random_kicks(n_steps, params["n_walkers"], D, dt, dev)  # (S, N, 2)

    r = solve.solve(t, kicks)
    var = r["x"][-1].var(dim=0)  # (2,)  spread of the final cloud along x and y
    print(f"fitted D = {r['D_fit']:.3f}   true D = {D:.3f}")
    print(f"final variance per axis {var.tolist()}   theory 2Dt = {2 * D * t[-1]:.2f}")
    return plot.animate(t, r["x"], r["msd"], D, r["D_fit"], out_dir / "random_walk_diffusion.gif")


if __name__ == "__main__":
    cli(run, __file__)
