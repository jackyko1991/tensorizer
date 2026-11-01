"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    names = tensors.enzyme_names()
    k1, k_1, k2 = tensors.rate_constants(dev)  # (B,) each
    S0 = tensors.substrate_concentrations(dev)  # (C,) µM
    E0 = params["E0"]
    y0 = tensors.initial_state(E0, S0, len(names))  # (4, B, C)

    K_m = equation.michaelis_constant(k1, k_1, k2)  # (B,)
    V_max = equation.max_velocity(k2, E0)  # (B,)
    times, full, S_qssa = solve.solve(y0, k1, k_1, k2, V_max, K_m, params["t_end"])

    S_v, v0 = solve.initial_rate(full, k2)  # (B, C) measured initial velocities and the S they were measured at
    K_fit, V_fit = equation.lineweaver_burk(S_v, v0)

    enzyme_total = full[:, 0] + full[:, 2]  # E + ES
    substrate_total = full[:, 1] + full[:, 2] + full[:, 3]  # S + ES + P
    P_qssa = S0 - S_qssa  # product of the reduced model
    gap = ((full[:, 3] - P_qssa).abs().amax(dim=(0, 2)) / S0.max()).tolist()  # (B,) worst gap, fraction of S0
    print(f"E0 = {E0:g} µM, S0 = {S0[0]:g} … {S0[-1]:g} µM, {len(times) - 1} RK4 steps of {times[1]:.4f} s")
    print(
        f"conservation: max |E + ES − E0| = {(enzyme_total - E0).abs().max():.1e} µM, "
        f"max |S + ES + P − S0| = {(substrate_total - S0).abs().max():.1e} µM"
    )
    for b, name in enumerate(names):
        print(
            f"enzyme {name}: K_m = {K_m[b]:5.2f} µM (Lineweaver–Burk {K_fit[b]:5.2f})   "
            f"V_max = {V_max[b]:.3f} µM/s (fit {V_fit[b]:.3f})   "
            f"full vs QSSA product: max gap {100 * gap[b]:.2f} % of S0"
        )
    e = names.index(params["enzyme"])
    return plot.animate(
        times, full, P_qssa, S0, S_v, v0, K_m, V_max, K_fit, V_fit, names, e, out_dir / "michaelis_menten.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
