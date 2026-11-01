"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import mass, slope

N = 512  # grid points


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    nu, A, U0 = params["nu"], params["amplitude"], params["offset"]
    x, dx = tensors.grid(N, dev)  # (N,)
    D, L = tensors.derivative_matrices(N, dx, dev)  # (N, N) each
    u0 = tensors.initial_wave(x, A, U0)  # (N,)

    dt = solve.time_step(dx, abs(U0) + A, nu)
    times, u = solve.simulate(u0, D, L, nu, dt, params["t_max"])  # (T,), (T, N)
    steepness = (-slope(u, D)).max(dim=-1).values  # (T,) steepest downhill slope max(-u_x)

    t_star = 1 / A
    k_form = int((steepness > 0.5 * steepness.max()).int().argmax())
    m = mass(u, dx)
    exact = solve.cole_hopf(x, times[-1].item(), nu, A, U0)
    print(f"{N} points, dt = {dt:.2e}, {len(times) - 1} RK4 steps")
    print(
        f"inviscid breaking time t* = 1/max(-u0') = {t_star:.2f};  "
        f"slope reaches half its peak at t = {times[k_form]:.2f}"
    )
    print(f"mass ∫u dx: start {m[0]:.6f}, end {m[-1]:.6f}")
    print(f"max |u - u_Cole-Hopf| at t = {times[-1]:.2f}: {(u[-1] - exact).abs().max():.2e}")
    return plot.animate(
        x, times, u, lambda t: solve.cole_hopf(x, t, nu, A, U0), steepness, t_star, out_dir / "burgers_shock.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
