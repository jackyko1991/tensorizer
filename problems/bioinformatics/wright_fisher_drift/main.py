"""Main loop: replicate populations → Wright–Fisher drift → heterozygosity, fixation, Hardy–Weinberg → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

N_SHOWN = 60  # populations whose genotypes are drawn in the Hardy–Weinberg panel


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator(device=dev).manual_seed(params["seed"])
    N, s, G = params["N"], params["s"], params["generations"]
    R = 200 if fast_mode() else params["replicates"]
    n_fix = 2000 if fast_mode() else params["fix_replicates"]

    # 1) drift (with optional selection) in R populations that all start at p0
    p = solve.simulate(tensors.initial_frequencies(R, params["p0"], dev), N, s, G, gen)  # (R, G+1)
    t = torch.arange(G + 1, device=dev, dtype=torch.float64)
    H_sim = equation.heterozygosity(p).mean(dim=0)  # (G+1,) average over replicates
    H_theory = equation.expected_heterozygosity(H_sim[0].item(), N, t)  # (G+1,) neutral prediction

    # 2) fixation of a single new mutant (p0 = 1/2N) for several selection strengths
    s_grid = tensors.selection_grid(N, s, dev)  # (S,)
    u_sim = solve.fixation(tensors.new_mutants(len(s_grid), n_fix, N, dev), N, s_grid, gen, max_gen=40 * N)  # (S,)
    u_theory = equation.kimura_fixation(1 / (2 * N), N, s_grid)  # (S,)

    # 3) Hardy–Weinberg: genotypes of N individuals made from random gametes, in a few populations
    G_obs = solve.sample_genotypes(p[:N_SHOWN], N, gen)  # (K, G+1, 3) in every generation
    G_hw = equation.hardy_weinberg(p[:N_SHOWN, 0])  # (K, 3) expected at t = 0

    fixed, lost = (p[:, -1] == 1).double().mean().item(), (p[:, -1] == 0).double().mean().item()
    print(f"N = {N} diploids (2N = {2 * N} copies), s = {s:g}, {R} populations from p0 = {params['p0']:g}")
    print(f"Hardy–Weinberg at t = 0: AA, Aa, aa observed {_fmt(G_obs[:, 0].mean(0))}  expected {_fmt(G_hw.mean(0))}")
    for tt in sorted({G // 4, G // 2, G}):
        print(f"t = {tt:4d}: mean H = {H_sim[tt]:.4f}   H0 (1 - 1/2N)^t = {H_theory[tt]:.4f}")
    print(f"after {G} generations: fixed {fixed:.1%}, lost {lost:.1%}, still segregating {1 - fixed - lost:.1%}")
    print(f"single new mutant, {n_fix} populations per s:   1/2N = {1 / (2 * N):.4f}")
    for k in range(len(s_grid)):
        print(f"  4Ns = {4 * N * s_grid[k]:6.2f}   fixed {u_sim[k]:.4f}   Kimura {u_theory[k]:.4f}")
    return plot.animate(
        p, G_obs, H_sim, H_theory, s_grid, u_sim, u_theory, N, s, n_fix, out_dir / "wright_fisher_drift.gif"
    )


def _fmt(x: torch.Tensor) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in x.tolist()) + ")"


if __name__ == "__main__":
    cli(run, __file__)
