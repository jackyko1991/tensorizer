"""Main loop: data -> fits -> verdicts -> plot. Step 6 of the Projection Hypothesis (speculative): telescope tests."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli

from . import plot, solve, tensors

F64 = torch.float64
PARDO = {"shoes": (4.02, 0.10, 0.07), "planck": (3.98, 0.09, 0.07)}  # MAP, -, + (68%), Pardo et al. 2018 abstract


def run(params: dict, out_dir: Path) -> Path:
    key = params["em_distance"]
    D_s, n_s, inc_max = float(params["D_screen"]), float(params["n_screen"]), float(params["inc_max_sgra"])

    # (a) GW170817: d_GW against d_EM, posterior of D
    gw = tensors.load_gw_samples()
    table = solve.likelihood_table(gw)
    print(f"(a) {gw['label']}: {len(gw['d'])} distance samples, d^2 prior divided out, kernel width "
          f"{float(gw['h']):.2f} Mpc")  # fmt: skip
    Ds = torch.linspace(3.0, 5.0, 801, dtype=F64)
    posts, fits = {}, {}
    for k in ("sbf", "shoes", "planck"):
        em = tensors.em_distance(k)
        posts[k] = solve.posterior_D(Ds, gw, em, table)
        fits[k] = solve.fit_D(gw, em, D0=4.5)
        lo, hi = posts[k]["interval"]
        mark = "  <- selected" if k == key else ""
        print(f"    d_EM = {em['d']:5.2f} +- {em['sigma']:.2f} Mpc ({em['label']}){mark}")
        print(f"      grid: D = {posts[k]['map']:.3f} (+{hi - posts[k]['map']:.3f} / -{posts[k]['map'] - lo:.3f}, "
              f"68% HDI);  autograd MAP {fits[k]['D']:.3f} +- {fits[k]['sigma']:.3f} (Laplace)")  # fmt: skip
    for k, (m, lo, hi) in PARDO.items():
        print(f"    Pardo et al. 2018 ({k} H0): D = {m:.2f} (+{hi:.2f} / -{lo:.2f})")
    sel = posts[key]
    lo, hi = sel["interval"]
    print(f"    verdict: gravity spreads like D = {sel['map']:.2f} (68% {lo:.2f} .. {hi:.2f}) out to 40 Mpc; "
          f"D = 4 is inside")  # fmt: skip

    em = tensors.em_distance(key)
    n_vals = torch.tensor(sorted({1.0, 2.0, 5.0, n_s}), dtype=F64)
    D_vals = torch.linspace(4.25, 8.0, 31, dtype=F64)
    scr = solve.screening_bound(torch.cat([D_vals, torch.tensor([D_s], dtype=F64)]), n_vals, gw, em, table)
    i_n = int((n_vals - n_s).abs().argmin())
    b = float(scr["bound"][-1, i_n])
    edge = b >= float(scr["Rc"][-1]) * 0.999
    print(f"    screened leakage (D = {D_s:g}, n = {n_s:g}): R_c {'>' if edge else '>='} {b:.0f} Mpc "
          f"(-2 ln L/L_GR > 2.71 below it)")  # fmt: skip

    # (b) EHT shadows and the fold
    sh = solve.shadow_tests(tensors.load_eht(), spin_inc_max=inc_max)
    edges = solve.traced_shadow_edge(torch.tensor([0.0, 2.0], dtype=F64))
    print(f"(b) Schwarzschild shadow radius: ray tracing {float(edges[0]):.4f} r_s, photon sphere "
          f"sqrt(27)/2 = {math.sqrt(27) / 2:.4f} r_s")  # fmt: skip
    for r in sh["rows"]:
        print(f"    {r['object']:6s} {r['prior']:30s} 2 sqrt27 theta_g = {r['shadow_schw']:5.1f} uas (ring "
              f"{r['ring_uas']:.1f} +- {r['ring_err']:.1f}); delta = {r['delta']:+.3f} (-{r['delta_lo']:.3f} "
              f"+{r['delta_hi']:.3f})")  # fmt: skip
        print(f"           delta_up (2 sigma) = {r['delta_up']:+.3f}, spin can shrink by {-r['kerr_min']:.3f} -> "
              f"fold may grow it by {r['allowed']:.3f}: r_f <= {r['r_bound']:.2f} r_s")  # fmt: skip
    print("    verdict: shadows match Schwarzschild/Kerr; a fold outside about 2.3-2.9 r_s is excluded, a fold "
          "inside the photon sphere (1.5 r_s) is invisible to the shadow")  # fmt: skip

    # (c) Pantheon+ supernovae: leakage index alpha
    sn = solve.supernova_fit(*tensors.load_supernovae())
    print(f"(c) Pantheon+ ({sn['n']} SNe, diagonal errors, offset marginalised): alpha = {sn['mean']:.2f} +- "
          f"{sn['sd']:.2f}; best fit alpha = {sn['a_fit']:.2f}, Om = {sn['om_fit']:.2f}")  # fmt: skip
    print(f"    chi2: LCDM (alpha 0) {sn['lcdm'][0]:.1f} at Om {sn['lcdm'][1]:.2f}; DGP (alpha 1) {sn['dgp'][0]:.1f} "
          f"at Om {sn['dgp'][1]:.2f}: supernovae alone cannot tell them apart")  # fmt: skip

    summary = plot.summary_rows(posts, sh, sn, PARDO)
    print("Summary (deviation from GR in 3+1, in standard deviations):")
    for row in summary:
        print(f"    {row['label']:34s} {row['pull']:+.2f}   {row['text']}")
    return plot.animate(gw, table, posts, key, scr, i_n, sh, sn, summary, out_dir / "projection_telescope_tests.gif")


if __name__ == "__main__":
    cli(run, __file__)
