"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.plotting import frame_indices
from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    a_star = 10 ** params["log10_a_star"]
    x = tensors.log_a_grid(min(a_star * math.exp(-params["efolds"] - 3), 1e-40), params["n_grid"], dev)
    n_frames = len(frame_indices(60, 60))
    c_ratio = 10 ** tensors.sweep(params["log10_c_ratio"], n_frames, dev)  # (F,) VSL switched on gradually
    efolds = tensors.sweep(params["efolds"], n_frames, dev)  # (F,) inflation switched on gradually

    std = solve.horizons(x, "lcdm")  # (1, A)
    vsl = solve.horizons(x, "vsl", c_ratio=c_ratio, a_star=a_star)  # (F, A): one universe per frame
    infl = solve.horizons(x, "inflation", efolds=efolds, a_star=a_star)  # (F, A)
    s_std, s_vsl, s_inf = (solve.sky_statistics(x, h) for h in (std, vsl, infl))

    D = float(s_std["distance"][0])
    print(f"Planck 2018 ΛCDM: particle horizon today χ_p(1) = {float(std['chi'][0, -1]) / 1e3:.2f} Gpc, "
          f"distance to last scattering D = {D / 1e3:.2f} Gpc")  # fmt: skip
    print(f"horizon at recombination χ_p(a_rec) = {float(s_std['chi_rec'][0]):.0f} Mpc → θ = "
          f"{math.degrees(float(s_std['theta'][0])):.2f}°, ≈ {float(s_std['patches'][0]):.0f} "
          "causally disconnected patches")  # fmt: skip
    r_min = solve.vsl_minimum_ratio(x, a_star)
    n_min = solve.inflation_minimum_efolds(x, a_star)
    print(f"transition at a_* = 1e{params['log10_a_star']:.0f}: needs c_early/c₀ ≥ {r_min:.2e} (VSL) "
          f"or N ≥ {n_min:.1f} e-folds (inflation)")  # fmt: skip
    for name, s, val in (("VSL", s_vsl, f"c_early/c₀ = {float(c_ratio[-1]):.1e}"),
                         ("inflation", s_inf, f"N = {float(efolds[-1]):.0f}")):  # fmt: skip
        print(
            f"{name:9s} {val}: χ_p(a_rec)/D = {float(s['chi_rec'][-1] / s['distance'][-1]):.3g}  "
            f"→ horizon problem {'solved' if bool(s['solved'][-1]) else 'NOT solved'}"
        )
    print("observational bounds (VSL here only changes c before a_*, far earlier than any of them probe):")
    for name, b in equation.BOUNDS.items():
        print(f"  {name:38s} {b['value'][0]:+.1e} … {b['value'][1]:+.1e} {b['unit']:14s} ({b['ref']})")

    return plot.animate(x, std, vsl, infl, (s_std, s_vsl, s_inf), c_ratio, efolds, a_star,
                        out_dir / "variable_speed_of_light.gif")  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
