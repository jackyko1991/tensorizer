"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    T, gamma, k, m = params["temperature"], params["gamma"], params["k"], params["m"]
    x0, v0 = tensors.initial_state(params["n_particles"], dev)  # (B,), (B,)
    dt = tensors.time_step(gamma, k, m)

    times, x, v = solve.solve(x0, v0, gamma, k, m, T, params["t_max"], dt, tensors.generator(dev))  # (S,), (S, B)
    x_det = equation.damped_oscillator(times, tensors.X0, gamma, k, m)  # (S,) noise-free curve

    x2 = equation.ensemble_mean_square(x)  # (S,) ⟨x²⟩(t)
    v2 = equation.ensemble_mean_square(v)  # (S,) ⟨v²⟩(t)
    x2_eq, v2_eq = equation.equipartition(T, k, m)
    late = times > 0.5 * times[-1]  # second half of the run: (close to) equilibrium
    print(
        f"B = {x.shape[1]} particles, Δt = {dt:.4g}, noise kick per step σ_v = "
        f"{equation.noise_amplitude(gamma, T, m, dt):.4g}"
    )
    print(f"<x²> late = {x2[late].mean().item():.4f}   (k_B T / k = {x2_eq:.4f})")
    print(f"<v²> late = {v2[late].mean().item():.4f}   (k_B T / m = {v2_eq:.4f})")
    print(f"<x> at end = {x[-1].mean().item():+.4f}   (noise-free x(t_max) = {x_det[-1].item():+.4f})")
    return plot.animate(times, x, x_det, x2, v2, x2_eq, v2_eq, out_dir / "langevin_noise.gif")


if __name__ == "__main__":
    cli(run, __file__)
