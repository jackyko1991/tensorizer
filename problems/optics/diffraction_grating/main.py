"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    wavelength = params["wavelength"] * 1e-3  # nm → µm, so every length is in µm
    d, a = params["d"], params["a"]
    sin_theta = tensors.sin_theta_grid()
    counts = tensors.slit_counts(params["n_max"])

    patterns = solve.solve(counts, sin_theta, wavelength, d, a)
    frames = solve.frames(counts)
    orders = solve.principal_maxima(wavelength, d, sin_theta.max().item())

    print("principal maxima d sin θ = mλ at sin θ =", ", ".join(f"{s:+.3f}" for s in orders))
    first_zero = wavelength / (counts[-1] * d)
    print(f"first zero next to the central peak: sin θ = λ/(N d) = {first_zero:.4f} for N = {counts[-1]}")
    env = equation.envelope(sin_theta, a, wavelength)
    path = out_dir / "diffraction_grating.gif"
    return plot.animate(sin_theta, patterns, env, counts, frames, orders, wavelength, d, path)


if __name__ == "__main__":
    cli(run, __file__)
