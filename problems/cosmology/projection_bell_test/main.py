"""Main loop: parameters -> tensors -> solve -> plot. Step 5 of a speculative hypothesis: Bell tests of the race."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(1964)
    D, m_max = int(params["D"]), int(params["m_max"])
    B, trials = int(params["settings"]), int(params["trials"])
    if D < 8:
        raise ValueError("need D >= 8: 4 visible coordinates and one hidden one per joint outcome")

    co = solve.correlations(m_max, D, trials, gen, dev)
    print(f"Known: singlet E(a, b) = -cos(theta); CHSH S <= 2 (local), S <= 2 sqrt 2 = {solve.TSIRELSON:.4f} (quantum)")
    print(f"(i)  joint race on the shared Psi in C^{D} (one event in the full space), m = 1:")
    print(f"     closed form E = -cos(theta) to {(co['E_joint'][0] - co['E_qm']).abs().max():.1e}; Monte Carlo "
          f"({trials} events per angle) max |E + cos| = {co['mc_err']:.4f} "
          f"(~{2 / math.sqrt(trials):.4f} expected)")  # fmt: skip
    print(f"     max S along the CHSH family = {co['S_joint'][0].max():.4f}")

    rc = solve.random_chsh(B, D, gen, dev)
    print(f"     {B} random settings: max S = {rc['S_joint'].max():.4f} (never above 2 sqrt 2)")
    print("(ii) local races (each party sees only its own hidden energies and a hidden direction n set at the source):")
    for mm, S, So in zip(solve.LOCAL_M, rc["S_local"], rc["S_local_opt"]):
        print(f"     m = {mm:>3}: max S over random settings {S.max():.4f}, at the quantum-optimal settings {So:.4f}")
    print("     Bell's theorem: S <= 2 for every local model; m -> oo is Bell's own 1964 model E = -(1 - 2 theta/pi)")

    bb = solve.beyond_born(m_max, gen, dev)
    i1 = (bb["m"] - 1).abs().argmin()
    print(f"Beyond Born: joint race S(m) at the optimal settings: m = 1 -> {bb['S'][i1]:.4f}, m = {m_max} -> "
          f"{bb['S'][-1]:.4f} (Tsirelson broken for m > 1); dS/dm = {bb['dS_dm']:.3f} at m = 1")  # fmt: skip
    print(f"     partly entangled state: Alice's marginal moves by up to {bb['signal'][-1]:.3f} when Bob turns his "
          f"setting (m = {m_max}): faster-than-light signalling unless m = 1")  # fmt: skip
    for k, (dm, dS, ds, shape) in bb["bands"].items():
        print(f"     triple-slit band ({k}) |m - 1| <= {dm:.1e}: |dS| <= {dS:.1e}, signalling <= {ds:.1e}, "
              f"E(theta) shape residual <= {shape:.1e}")  # fmt: skip

    rows = solve.data_table()
    print("Data: loophole-free Bell tests (cited numbers, data/bell_tests.csv)")
    for r in rows:
        if isinstance(r["S"], float):
            print(f"     {r['experiment']:26s} {r['year']}  S = {r['S']:.4f} +- {r['sigma']:.4f}: "
                  f"{r['above_local']:5.1f} sigma above 2, "
                  f"{r['below_tsirelson']:5.1f} sigma below 2 sqrt 2")  # fmt: skip
        else:
            print(
                f"     {r['experiment']:26s} {r['year']}  {r['inequality']}: p = {r['p_value']} against local realism"
            )
    storz = rows[-1]
    print(f"Verdict: (ii) is ruled out (Storz 2023 alone is {storz['above_local']:.0f} sigma above 2). (i) agrees "
          "with every test, but only because with m = 1 it IS quantum mechanics. The deviation step 1 still allows "
          f"(|dS| <= {bb['bands']['single photons'][1]:.0e}) is larger than sigma_S = {storz['sigma']} but is "
          f"absorbed by the unknown visibility (S / 2 sqrt 2 = {storz['visibility']:.3f}); only the shape of E(theta) "
          f"(<= {bb['bands']['single photons'][3]:.0e}) or signalling (<= {bb['bands']['single photons'][2]:.0e}) "
          "could show it, and m != 1 breaks relativistic causality.")  # fmt: skip

    return plot.animate(co, rc, bb, rows, D, out_dir / "projection_bell_test.gif")


if __name__ == "__main__":
    cli(run, __file__)
