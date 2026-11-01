"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    names = tensors.species_names()
    w, depth = params["w"], params["depth"]
    D = tensors.diffusion_coefficients(dev)  # (B,) µm²/s
    x, h = tensors.grid(w, dev)  # (N,) µm
    mask = tensors.bleach_mask(x, w)  # (N, N)
    weights = tensors.stencil(dev)  # (5,)
    c0 = tensors.initial_state(mask, depth, len(names))  # (B, N, N), just after the bleach

    n_frames = 5 if fast_mode() else 90
    times, F_spot, snap_times, snaps = solve.solve(c0, D, weights, h, mask, params["t_end"], n_frames)

    F_inf = c0.mean(dim=(1, 2))  # (B,) the dye is conserved, so the patch ends up uniform at its mean
    f = equation.normalized_recovery(F_spot, F_spot[0], F_inf)  # (T, B)
    t_half = solve.half_time(times, f)
    t_half_theory = equation.soumpasis_half_time(w, D)
    D_fit = solve.fit_diffusion(times, f, w, t_half)
    tau = equation.diffusion_time(w, D)

    drift = (snaps[-1].mean(dim=(1, 2)) - F_inf).abs().max()
    print(f"{len(x)}×{len(x)} grid, h = {h:.3f} µm, bleach radius w = {w:g} µm, {len(times) - 1} RK4 steps")
    print(f"total fluorescence conserved to {drift:.1e} (no-flux edges)")
    for b, name in enumerate(names):
        print(
            f"{name:<17} D = {D[b]:.3g} µm²/s   τ_D = w²/4D = {tau[b]:6.3f} s   "
            f"t½ = {t_half[b]:6.3f} s (Soumpasis 0.224 w²/D = {t_half_theory[b]:6.3f} s)   D fit = {D_fit[b]:.3g}"
        )
    return plot.animate(
        times, f, snap_times, snaps, x, w, depth, D, D_fit, t_half, names, out_dir / "membrane_diffusion_frap.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
