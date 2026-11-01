"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    dt = 0.005
    x, k, dx = tensors.grid(512, dev)  # k_max² Δt / 2 < π avoids the split-step grid instability
    half_linear = tensors.half_linear_step(k, dt)
    psi0 = tensors.initial_wave(x, params["mode"], params["noise"], params["eta"])

    times, psi = solve.solve(psi0, half_linear, dt, params["t_max"])

    N, H = equation.norm(psi, dx), equation.hamiltonian(psi, k, dx)
    print(f"norm        N = {N[0]:.6f}, largest relative change {((N - N[0]).abs().max() / N[0]):.1e}")
    print(f"Hamiltonian H = {H[0]:.6f}, largest change {((H - H[0]).abs().max()):.1e}")
    peak = psi.abs().max(dim=1).values
    when = int(peak.argmax())
    print(f"highest peak |psi| = {peak[when]:.3f} at t = {times[when]:.2f}  (Peregrine breather: 3)")
    return plot.animate(times, x, psi, out_dir / "nonlinear_schrodinger.gif")


if __name__ == "__main__":
    cli(run, __file__)
