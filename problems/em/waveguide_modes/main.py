"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve


def run(params: dict, out_dir: Path) -> Path:
    a, b = params["a"], params["b"]  # cm   dense eigenproblems: CPU float64

    found = solve.solve(a, b, params["n_grid"], params["n_modes"])

    frequencies = [solve.cutoff_frequency_GHz(mode["k_c"]) for mode in found]
    for mode, f in zip(found, frequencies):
        error = abs(mode["k_c"] - mode["k_exact"]) / mode["k_exact"]
        print(f"{mode['kind']}{mode['mn']}  k_c = {mode['k_c']:.4f} /cm  (exact {mode['k_exact']:.4f}, "
              f"error {error:.1e})   f_c = {f:.2f} GHz")  # fmt: skip
    return plot.animate(found, frequencies, a, b, out_dir / "waveguide_modes.gif")


if __name__ == "__main__":
    cli(run, __file__)
