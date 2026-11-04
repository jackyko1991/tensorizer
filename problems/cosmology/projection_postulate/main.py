"""Main loop: parameters -> tensors -> solve -> plot. A speculative hypothesis: its mathematics is checked here."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(1905)
    D_max, d, K = int(params["D_max"]), int(params["d"]), int(params["outcomes"])
    m_max, trials = int(params["m_max"]), int(params["trials"])
    if not 1 <= d < D_max:
        raise ValueError("need 1 <= d < D_max")

    # 1. P1: a uniform state in C^D seen through d coordinates
    Ds = tensors.dimension_ladder(d, D_max)
    con = solve.concentration(Ds, d, trials, gen, dev)
    print(f"P1  ||P Psi||^2 ~ Beta(d, D - d), d = {d}, {trials} states per D (KS 1% critical value "
          f"{1.63 / math.sqrt(trials):.4f})")  # fmt: skip
    print("      D     mean (d/D)            sample mean     var ratio   KS        KS (3D real shadow)")
    for i, D in enumerate(Ds.tolist()):
        print(f"  {D:6d}   {con['mean'][i]:.6f}   {con['q_mean'][i]:.6f}   "
              f"{con['q_var'][i] / con['var'][i]:9.3f}   {con['ks'][i]:.4f}    {con['ks3'][i]:.4f}")  # fmt: skip
    print(f"    normalised 3D shadow: height KS vs uniform [-1, 1] = {con['ks_height']:.4f} (uniform on the sphere)")

    # 2. P2 as a projection race
    rc = solve.race(K, m_max, trials, gen, dev)
    print(f"P2  projection race, |c_k|^2 = {[round(x, 4) for x in rc['w'].tolist()]}, {trials} events per m")
    print("   m   P(k) quadrature (complex hidden dims)      Monte Carlo                  TV complex   TV real")
    for i in range(m_max):
        pq = " ".join(f"{x:.4f}" for x in rc["P_c"][i].tolist())
        pm = " ".join(f"{x:.4f}" for x in rc["f_c"][i].tolist())
        print(f"  {i + 1:2d}   {pq:40s}   {pm:28s}   {rc['tv_c'][i]:.5f}      {rc['tv_r'][i]:.5f}")
    print(f"    m = 1 complex is the Born rule (TV {rc['tv_c'][0]:.1e}); largest MC pull {rc['max_pull']:.2f} sigma; "
          f"one REAL hidden dimension gives the arcsine law (TV {rc['tv_r'][0]:.4f})")  # fmt: skip

    # 3. P5: how many dimensions
    dm = solve.dimensions(d, K, dev)
    print(f"P5  Born race: D >= d + K m = {d} + {K} x 1 = {dm['D_born']}")
    for N, lg in zip(dm["table_N"].tolist(), dm["table_log10_D"].tolist()):
        print(f"    {N:9.3g} entangled two-level systems: D >= 2^N = 10^{lg:.4g}")
    print("    the same Hilbert space standard quantum mechanics already uses: so far a reinterpretation")

    # 4. data: three-path interference
    sb = solve.sorkin_bound(m_max, dev)
    print(f"Data  race model in a triple slit: kappa ~ s (a - 1), s = {min(sb['slopes']):.3f} .. "
          f"{max(sb['slopes']):.3f} over 9 binnings (conservative s = {sb['s_min']:.3f})")  # fmt: skip
    for r, lim, al in zip(sb["rows"], sb["limits"], sb["allowed"]):
        print(f"    {r['source']:16s} {r['year']}  {r['regime']:24s} |kappa| <= {lim:.1e}  ->  |a - 1| <= {al:.1e}")
    print("    integer m (complex):", " ".join(f"{m}:{k:+.3g}" for m, k in
                                              zip(range(1, m_max + 1), sb["kappa_complex"].tolist())))  # fmt: skip
    print("    integer m (real)   :", " ".join(f"{m}:{k:+.3g}" for m, k in
                                              zip(range(1, m_max + 1), sb["kappa_real"].tolist())))  # fmt: skip
    print(f"    verdict: only m = 1 complex (= 2 real) hidden dimension per outcome survives; tightest band "
          f"|a - 1| <= {min(sb['allowed']):.1e}")  # fmt: skip

    return plot.animate(con, rc, dm, sb, d, out_dir / "projection_postulate.gif")


if __name__ == "__main__":
    cli(run, __file__)
