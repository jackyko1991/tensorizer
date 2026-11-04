"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.plotting import frame_indices
from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    kind, n_a, n_b = params["string"], params["mode_a"], params["mode_b"]
    alpha, alpha_t = (a.to(dev) for a in tensors.oscillators(n_a, n_b, kind))
    sigma = tensors.sigma_grid(params["n_points"], kind, dev)
    steps = frame_indices(72, 72)
    tau = tensors.tau_grid(72, dev)[steps]  # one worldsheet time per frame

    sheet = solve.worldsheet(tau, sigma, alpha, alpha_t, kind)  # X: (F, S, 3)
    level = round(sheet["N"])
    print(f"{kind} string: modes {n_a} and {n_b}, oscillator sum N = {sheet['N']:.3f}"
          + (f", Ñ = {sheet['N_t']:.3f} (level matched)" if kind == "closed" else ""))  # fmt: skip
    lf = sheet["level_field"]
    print(f"level from the field energy (1/4πα′)∫(Ẋ²+X′²)dσ: {lf.min():.6f} … {lf.max():.6f} over τ (conserved)")

    spec = solve.spectrum(kind, params["n_levels"])
    print("level N:        " + "  ".join(f"{int(n):>9d}" for n in spec["N"]))
    print("α′M²:           " + "  ".join(f"{m:>9.0f}" for m in spec["M2"]))
    print("degeneracy:     " + "  ".join(f"{d:>9.4g}" for d in spec["degeneracy"]))
    m2_state = float(spec["M2"][level]) if level < len(spec["M2"]) else float("nan")
    print(f"this classical configuration sits at level N = {level}: α′M² = {m2_state:.0f} (quantum, intercept a = 1)")

    radii = tensors.radii(1 / params["R_max"], params["R_max"], len(steps), dev)
    M2, _ = solve.circle_spectrum(torch.cat([radii, 1 / radii]), k_max=6, n_max=3)  # (2F, S) all states, all radii
    sorted_M2 = M2.sort(dim=1).values
    gap = (sorted_M2[: len(radii)] - sorted_M2[len(radii) :]).abs().max(dim=1).values  # (F,)
    print(f"T-duality R ↔ α′/R on {M2.shape[1]} states × {len(radii)} radii: max |ΔM²| = {gap.max():.2e}")

    curves = (solve.duality_curves(radii), solve.duality_curves(1 / radii))
    level = min(level, params["n_levels"])
    return plot.animate(sheet, kind, spec, level, radii, curves, list(solve.NAMED_STATES), gap,
                        out_dir / "string_modes.gif")  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
