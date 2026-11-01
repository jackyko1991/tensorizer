"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    n = params["n"]
    a, b = tensors.GASES[params["gas"]]
    T_c, V_c, p_c = equation.critical_point(n, a, b)
    T = tensors.temperatures(T_c, params["T_high"], params["T_low"], params["n_temps"])  # (T,)
    V = tensors.volumes(V_c)  # (V,)

    r = solve.solve(T, V, T_c, V_c, n, a, b)
    print(f"critical point: T_c = {T_c:.1f} K   V_c = {1e6 * V_c:.1f} cm³   p_c = {1e-6 * p_c:.3f} MPa")
    for t in [len(T) // 2, 3 * len(T) // 4, len(T) - 1]:
        if T[t] < T_c:
            print(
                f"T = {T[t]:6.1f} K   p* = {1e-6 * r['p_star'][t]:.3f} MPa"
                f"   liquid V = {1e6 * r['V_l'][t]:6.1f} cm³   gas V = {1e6 * r['V_g'][t]:7.1f} cm³"
            )
    return plot.animate(T, V, r, (T_c, V_c, p_c), out_dir / "van_der_waals.gif")


if __name__ == "__main__":
    cli(run, __file__)
