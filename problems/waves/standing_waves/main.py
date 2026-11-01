"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    L, c = params["L"], params["c"]
    x, dx = tensors.grid(L, params["n_points"], dev)
    D = tensors.stencil(len(x) - 2, dx, dev)
    u0 = tensors.initial_shapes(x, L, params["n_harmonics"], params["pluck_at"])

    t_max = params["periods"] * 2 * L / c  # fundamental period 2L/c
    times, traj, dt = solve.solve(u0, D, c, dx, t_max)

    freqs = equation.mode_frequencies(D, c)[: params["n_harmonics"]].tolist()
    for n, f in enumerate(freqs, start=1):
        print(f"harmonic {n}:  f = {f:.4f} Hz   (theory n c / 2L = {n * c / (2 * L):.4f} Hz)")
    E = equation.energy(traj, dx, dt, c)  # (T-2, B)
    drift = ((E - E[0]).abs().max(0).values / E[0]).max().item()
    print(f"largest relative energy change: {drift:.2e}")
    return plot.animate(times, x, traj, freqs, out_dir / "standing_waves.gif")


if __name__ == "__main__":
    cli(run, __file__)
