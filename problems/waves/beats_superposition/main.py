"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    f1, f2 = sorted((params["f1"], params["f2"]))
    if f2 - f1 < 1e-6:
        f2 = f1 + 0.05  # identical frequencies would give no beats at all
    omega = tensors.angular_frequencies(f1, f2, params["n_waves"], dev)
    k = tensors.wavenumbers(omega, params["dispersion"])
    A = tensors.amplitudes(params["n_waves"], dev)

    v_p, v_g = equation.phase_velocity(k, omega), equation.group_velocity(k, omega)
    x_max = max(10.0, v_p * params["t_max"])  # wide enough to follow the crest rider
    x, t = tensors.grids(x_max, params["t_max"], dev)
    result = solve.solve(A, k, omega, x, t)

    dt = (t[1] - t[0]).item()
    print(f"phase velocity v_p = {v_p:.3f} m/s,  group velocity v_g = {v_g:.3f} m/s,  ratio {v_g / v_p:.3f}")
    print(f"beat frequency measured {solve.beat_frequency(result['trace'], dt, f1):.3f} Hz   (|f1 - f2| = {f2 - f1:.3f} Hz)")
    return plot.animate(x, t, result, v_p, v_g, out_dir / "beats_superposition.gif")


if __name__ == "__main__":
    cli(run, __file__)
