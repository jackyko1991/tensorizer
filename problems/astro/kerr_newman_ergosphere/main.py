"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # float64 closed forms and a fine radius grid: the CPU is the right device here.
    M, a, Q = tensors.parameter_grid(n=params["grid"], masses=(1.0, 2.0, 5.0))  # (3, n, n): one batch
    g = solve.survey(M, a, Q)
    frac = g["naked"].double().mean().item()
    print(f"batched grid: {M.numel()} holes (M in {{1, 2, 5}}, a/M and Q/M in [0, 1.3]); "
          f"{100 * frac:.1f}% are naked (a² + Q² > M²)")  # fmt: skip
    scale = (g["r_plus"][1] / g["r_plus"][0]).nan_to_num(2.0)
    print(f"r₊ scales with M: r₊(2M)/r₊(M) in [{scale.min():.6f}, {scale.max():.6f}]")

    chi, Mf, af, Qf = tensors.sweep(params["angle"], params["chi_max"], params["frames"])
    theta = tensors.polar_angles()
    sw = solve.sweep(Mf, af, Qf, theta)

    keys = (
        "r_plus",
        "r_minus",
        "ergo_equator",
        "isco_pro",
        "isco_retro",
        "photon_pro",
        "photon_retro",
        "omega_H",
        "kappa",
    )
    print(" ".join(f"{h:>6}" for h in ("χ", "a", "Q", "r₊", "r₋", "r_E eq", "ISCO+", "ISCO−", "γ+", "γ−", "Ω_H", "κ")))
    for k in sorted({0, len(chi) // 4, len(chi) // 2, int((chi <= 1).sum()) - 1, len(chi) - 1}):
        row = [chi[k], af[k], Qf[k]] + [sw[key][k] for key in keys]
        print(" ".join(f"{float(v):6.3f}" for v in row) + ("   naked" if sw["naked"][k] else ""))
    one = Mf[:1]
    print(f"checks: Schwarzschild ISCO = {float(sw['isco_pro'][0]):.4f} M (exact 6), photon sphere "
          f"{float(sw['photon_pro'][0]):.4f} M (exact 3); extremal Kerr prograde ISCO (BPT) = "
          f"{float(equation.kerr_isco(one, one)):.3f} M")  # fmt: skip

    grid = {"r_plus": g["r_plus"][0].numpy(), "naked": g["naked"][0].numpy(), "top": a[0, -1, 0].item()}
    return plot.animate(chi, af, Qf, theta, sw, grid, params["angle"], out_dir / "kerr_newman.gif")


if __name__ == "__main__":
    cli(run, __file__)
