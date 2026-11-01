"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    M0 = 10 ** params["log10_mass"]
    names, masses = tensors.example_masses(M0, dev)

    T = equation.hawking_temperature(masses)  # (B,) one temperature per mass, all at once
    t_life = equation.lifetime(masses)
    print(f"{'black hole':>26}  {'mass (kg)':>10}  {'T_H (K)':>10}  {'λ_max (m)':>10}  {'lifetime (s)':>12}")
    for name, M, T_b, lam_b, t_b in zip(names, masses, T, equation.wien_peak(T), t_life):
        print(f"{name:>26}  {M:10.2e}  {T_b:10.2e}  {lam_b:10.2e}  {t_b:12.2e}")

    times, history = solve.evaporate(masses, fraction=params["burn_fraction"])  # (S, B) each
    numeric = times[-1] / t_life
    print(f"integrated lifetime / formula 5120πG²M³/(ħc⁴): {numeric.min():.4f} … {numeric.max():.4f}")

    h = history[:, -1]  # the animated hole
    T_h = equation.hawking_temperature(h)
    lam = tensors.wavelengths(0.03 * float(equation.wien_peak(T_h[-1])), 30 * float(equation.wien_peak(T_h[0])),
                              300, dev)  # fmt: skip
    return plot.animate(times[:, -1], h, T_h, equation.power(h), lam, float(t_life[-1]),
                        out_dir / "hawking_radiation.gif")  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
