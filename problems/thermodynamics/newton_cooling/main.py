"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, T_env = params["n_cups"], params["T_env"]
    k = tensors.cooling_constants(n, params["k_min"], params["k_max"], dev)  # (B,)
    T0 = tensors.start_temperatures(n, dev)  # (B,)

    times, temps = solve.solve(T0, k, T_env, params["t_max"])  # (T,), (T, B)
    exact = equation.exact_temperature(times, T0, k, T_env)  # (T, B)

    for b, t_half in enumerate(equation.half_life(k).tolist()):
        print(f"cup {b + 1}: k = {k[b]:.3f} /min   t½ = {t_half:5.1f} min   T(end) = {temps[-1, b]:5.1f} °C")
    print(f"largest difference RK4 vs exact: {(temps - exact).abs().max():.2e} °C")
    return plot.animate(times, temps, exact, k, T_env, out_dir / "newton_cooling.gif")


if __name__ == "__main__":
    cli(run, __file__)
