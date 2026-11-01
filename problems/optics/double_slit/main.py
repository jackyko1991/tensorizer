"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lam, d, dist = params["wavelength"], params["slit_sep"], params["screen_dist"]
    setup = tensors.setup(d, params["slit_width"], dist, dev)

    times, frames, intensity = solve.solve(setup, lam, params["t_max"])

    print(f"Courant number c dt/h = {equation.courant(1.0, 0.5, 1.0):.2f} (stable below {equation.CFL_LIMIT:.3f})")
    print(f"small-angle fringe spacing lambda L / d = {lam * dist / d:.1f} cells")
    fringes = solve.fringe_positions(lam, d, setup.screen_x - tensors.BARRIER_X)
    return plot.animate(times, frames, intensity, setup, fringes, out_dir / "double_slit.gif")


if __name__ == "__main__":
    cli(run, __file__)
