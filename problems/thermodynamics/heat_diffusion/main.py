"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    alpha = params["alpha"] * 1e-6  # mm²/s → m²/s
    X, Y, dx = tensors.grid(params["n_grid"], dev)
    T0 = tensors.hot_spots(X, Y, params["n_spots"])  # (1, 1, n, n)
    rho_c = tensors.heat_capacity(alpha)
    boundary = params["boundary"]

    times, frames = solve.solve(
        T0, tensors.K, rho_c, dx, params["t_max"], params["r"], boundary, tensors.T_EDGE
    )  # (F,), (F, n, n)
    flux = solve.flux_snapshots(frames, tensors.K, dx, boundary, tensors.T_EDGE)  # (F, 2, n, n)

    heat = solve.heat_content(frames, rho_c, dx, tensors.T_EDGE)  # (F,)
    print(f"alpha = {alpha:.2e} m^2/s, dx = {1000 * dx:.2f} mm, stability number alpha dt/dx^2 = {params['r']}")
    print(f"max T: {frames[0].max():.1f} °C -> {frames[-1].max():.1f} °C")
    print(f"heat content changed by {100 * (heat[-1] / heat[0] - 1):+.2f} % ({boundary} edges)")
    t_range = (tensors.T_EDGE, tensors.T_HOT)
    return plot.animate(times, frames, flux, heat, X, Y, t_range, out_dir / "heat_diffusion.gif")


if __name__ == "__main__":
    cli(run, __file__)
