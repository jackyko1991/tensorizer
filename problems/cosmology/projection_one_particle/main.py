"""Main loop: parameters -> worldlines -> slices -> counts -> data. A speculative hypothesis, tested here."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(1940)
    B, S, modes = int(params["worldlines"]), int(params["samples"]), int(params["modes"])
    rough, dims, L = float(params["roughness"]), int(params["dims"]), int(params["slices"])
    if dims < 2:
        raise ValueError("a worldline needs time plus at least one other dimension (dims >= 2)")
    levels = torch.linspace(-3, 3, L, dtype=torch.float64, device=dev)

    # 1. a batch of closed loops in `dims` dimensions, cut by L time slices
    bt = solve.batch(B, S, modes, rough, dims, levels, gen, dev)
    mid = L // 2
    print(f"P3  {B} closed worldlines in D = {dims} dimensions, {modes} harmonics, roughness {rough}, {L} slices")
    print(f"    slice t = 0: mean particles {bt['up'][:, mid].double().mean():.3f}, antiparticles "
          f"{bt['down'][:, mid].double().mean():.3f}; mean crossings {bt['mean_N'][mid]:.3f} "
          f"(Rice formula {bt['rice'][mid]:.3f} +- {bt['sem_N'][mid]:.3f})")  # fmt: skip
    print(f"    largest |net charge| on any slice of any closed loop: {bt['max_abs_Q']} (a closed loop is neutral)")
    print(f"    crossings = 2 (#pair creations below - #annihilations below) on every slice: {bt['morse_ok']}")

    # 2. the observed asymmetry
    ob = solve.observations(dev)
    print(
        f"Data  AMS-02 2019: positron fraction {ob['fraction'].min():.3f} .. {ob['fraction'].max():.3f} over "
        f"{ob['E'].min():.2f} .. {ob['E'].max():.0f} GeV -> asymmetry 1 - 2f = {ob['A_min']:.3f} .. {ob['A_max']:.3f}"
    )
    print(f"      Planck 2018: eta = {ob['eta']:.3e}; observable universe: N_gamma = {ob['N_gamma']:.2e}, "
          f"N_B = eta N_gamma = {ob['N_B']:.2e}")  # fmt: skip
    print(f"      a single worldline gives |Q| <= 1: |Q|/N <= {1 / ob['N_gamma']:.1e} at N = N_gamma, "
          f"versus eta = {ob['eta']:.1e}; net charge short by a factor {ob['N_B']:.1e}")  # fmt: skip

    # 3. survey of the open parameters: can any configuration reach the observed asymmetry?
    sv = solve.survey(min(B, 256), S, dims, levels, ob["A_min"], gen, dev)
    tot = sum(r["slices"] for r in sv)
    hit = sum(r["reproduce"] for r in sv)
    print(f"Survey {len(sv)} configurations (smooth/random, closed/open, harmonics, roughness), {tot} slices with "
          f"N >= 2: {hit} reach |Q|/N >= {ob['A_min']:.3f}")  # fmt: skip
    for r in sv[:2] + [x for x in sv if x["modes"] in (2, 32) and x["roughness"] == 1.0]:
        tag = "closed" if r["closed"] else "open  "
        print(f"    {r['kind']:6s} {tag} K = {r['modes']:2d}  max N = {r['max_N']:3d}  max |Q| = {r['max_abs_Q']}  "
              f"max |Q|/N (N >= 2) = {r['max_A']:.3f}")  # fmt: skip
    verdict = "ruled out in its simplest form" if hit == 0 else "not ruled out"
    print(f"    verdict: {verdict} (fraction of configurations that reproduce the data: {hit}/{tot})")

    # 4. labelled extensions
    wd = solve.winding_demo(dev, gen)
    print("Extension A (open worldline): |Q| <= 1, still no asymmetry beyond one particle")
    print(f"Extension B (biased projection): antiparticle visibility r = f/(1-f) = {ob['r'].min():.3f} .. "
          f"{ob['r'].max():.3f} from AMS (energy dependent), < {ob['cited']['antihelium_to_helium']:.1e} from BESS "
          f"antihelium: hidden antimatter = visible baryons = {ob['hidden_over_dm']:.2f} of dark matter")  # fmt: skip
    print(f"Extension C (compact time, winding w): net charge per slice {wd['Q'][:, 0].tolist()} for w = "
          f"{wd['w'].long().tolist()}; matching the data needs w ~ {ob['N_B']:.1e}, put in by hand")  # fmt: skip

    rep = solve.representative(S, min(modes, 6), 0.7, dims, levels, dev)
    return plot.animate(rep, bt, ob, levels, dims, out_dir / "projection_one_particle.gif")


if __name__ == "__main__":
    cli(run, __file__)
