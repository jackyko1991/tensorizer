"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import energy, mean_position, norm


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    g, x0, dt = params["g"], params["x0"], params["dt"]
    x0s = [x0, 2 * x0 / 3, x0 / 3]  # three packets released from different heights, evolved as columns
    x, dx = tensors.grid(params["n_grid"], 2 * (x0 + 5), dev)
    V = tensors.potential(x, g)
    H = tensors.hamiltonian(x, dx, V)
    psi0 = tensors.packets(x, x0s)  # (N, B)

    times, psi = solve.crank_nicolson(H, psi0, dt, params["t_max"])  # (T,), (T, N, B)
    t_euler, n_euler = solve.euler_norms(H, psi0, dx, dt, params["t_max"])  # (S,), (S, B)

    norms, energies = norm(psi, dx), energy(psi, H, dx)  # (T, B) each
    print(f"grid N = {len(x)}, dx = {dx:.4f}, dt = {dt}, {round(params['t_max'] / dt)} Crank–Nicolson steps")
    for b, start in enumerate(x0s):
        print(
            f"packet x0 = {start:4.2f}: <H> = {energies[0, b].item():8.4f} -> {energies[-1, b].item():8.4f}   "
            f"|norm - 1| max = {(norms[:, b] - 1).abs().max().item():.1e}"
        )
    print(f"explicit Euler, same dt: norm = {n_euler[-1].max().item():.1e} after {len(t_euler) - 1} steps")
    err = (psi[-1] - solve.exact(H, psi0, times[-1].item())).abs().max().item()
    print(f"max |psi_CN - exp(-iHt) psi0| at t_max = {err:.2e}   (shrinks like dt^2)")
    x_mean = mean_position(psi, x, dx)  # (T, B)
    return plot.animate(
        x, V, times, psi, x_mean, norms, energies, t_euler, n_euler, x0s, dt, out_dir / "schrodinger_crank_nicolson.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
