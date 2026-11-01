"""Main loop: noise model → batched Monte Carlo shots → decoder → threshold and Λ → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import binomial_logical_error


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    model = params["model"]
    ds = tensors.distances(5 if fast else params["d_max"])
    ps = tensors.error_rates(5 if fast else params["n_p"], params["p_min"], params["p_max"])
    shots = 1000 if fast else params["shots"]
    rounds = "1 round, perfect checks" if model == "code_capacity" else "d rounds, checks misread with q = p·ratio"
    print(f"bit-flip repetition code, d = {ds}, {shots} shots per point, {model} noise ({rounds})")

    result = solve.sweep(ds, ps, params["q_ratio"], model, shots, seed=0, device=dev)
    pl = result["p_logical"]

    print("\n   p     " + "".join(f"  P_L(d={d})" for d in ds))
    for b, p in enumerate(ps.tolist()):
        print(f"{p:7.4f}  " + "".join(f"  {pl[a, b]:.3e}  " for a in range(len(ds))))
    if model == "code_capacity":
        worst = max(abs(pl[a] - binomial_logical_error(ps, d)).max().item() for a, d in enumerate(ds))
        print(f"majority vote vs exact binomial tail: max |difference| = {worst:.1e} (statistical error ~1/√shots)")

    crossings = [solve.crossing(ps, pl[a], pl[a + 1]) for a in range(len(ds) - 1)]
    for d, pc in zip(ds, crossings):
        print(
            f"curves d = {d} and d = {d + 2} cross at p ≈ {pc:.3f}"
            if math.isfinite(pc)
            else f"d = {d}/{d + 2}: no crossing in range"
        )
    finite = [c for c in crossings if math.isfinite(c)]
    p_th = sum(finite) / len(finite) if finite else float("nan")
    if model == "code_capacity" and not finite:
        print("code capacity: the majority vote threshold is exactly p = 0.5 (raise p_max to see it)")

    lam = solve.suppression(result["eps"], pl, shots)
    for k, d in enumerate(ds[:-1]):
        good = [(p, lam[k, b].item()) for b, p in enumerate(ps.tolist()) if lam[k, b] == lam[k, b]]
        good = [(p, v) for p, v in good if not math.isfinite(p_th) or p < p_th]
        if good:
            text = ", ".join(f"{v:.2f} at p = {p:.3f}" for p, v in good[:3])
            print(f"Λ_{d}/{d + 2} = ε_{d}/ε_{d + 2} (per round): {text}")
    print("below threshold Λ > 1: every +2 in distance divides the logical error rate by Λ")
    return plot.animate(ds, ps, result, lam, p_th, model, out_dir / "error_correction_threshold.gif")


if __name__ == "__main__":
    cli(run, __file__)
