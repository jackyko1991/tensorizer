"""Main loop: batch of f → stability of the steady state (Hopf points) → stiff integration → 2D spiral → GIF."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import jacobian, steady_state, trace


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    eps, q, dt = params["eps"], tensors.Q, tensors.DT
    t_max = 2.0 if fast else params["t_max"]
    n_steps = round(t_max / dt)
    f = tensors.f_values(12 if fast else 55, dev)  # (B,)
    f_sel = params["f"]
    print(f"Oregonator, ε = {eps}, q = {q} (approximate), Δt = {dt}, {n_steps} Patankar–Euler steps, {len(f)} f values")

    # --- linear stability: where does the steady state start to oscillate? ---
    f_fine = torch.linspace(0.05, 4.0, 4000, dtype=torch.float64)
    hopf = solve.hopf_points(f_fine, eps, q)
    print("Hopf bifurcations (tr J = 0) at f = " + ", ".join(f"{h:.3f}" for h in hopf.tolist()))
    fs = torch.tensor([f_sel], dtype=torch.float64)
    xs = steady_state(fs, q)
    J = jacobian(xs, xs, fs, eps, q)[0]
    lam = torch.linalg.eigvals(J)
    print(
        f"f = {f_sel}: steady state x* = z* = {xs.item():.5f}, tr J = {trace(J[None]).item():+.2f}, "
        f"det J = {torch.linalg.det(J).item():.2f}, eigenvalues {lam[0].item():.3f}, {lam[1].item():.3f}"
    )

    # --- the batch of well-mixed reactions, plus the selected f ---
    f_all = torch.cat([f, torch.tensor([f_sel], dtype=torch.float64, device=dev)])
    x0, z0 = tensors.initial_state(f_all)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # thousands of tiny steps: one thread is far steadier than many on a busy CPU
    try:
        t, x, z = solve.integrate(x0, z0, f_all, eps, q, dt, n_steps)
        spiral = None
        if params["spiral"] == "on":
            n = 48 if fast else params["grid"]
            sx, sz = tensors.broken_wave(n, f_sel, dev)
            spiral = solve.spiral(sx, sz, f_sel, eps, q, dt, tensors.H, n_steps)
    finally:
        torch.set_num_threads(threads)
    lo, hi = solve.amplitude(x)
    osc = hi / lo > 1.5
    if osc[:-1].any():
        fo = f[osc[:-1]]
        print(f"simulation: x oscillates for f in [{fo.min():.3f}, {fo.max():.3f}] (grid step {f[1] - f[0]:.3f})")
    else:
        print("simulation: no oscillations at this ε")
    T = solve.period(t, x[:, -1])
    print(f"f = {f_sel}: x swings between {lo[-1]:.2e} and {hi[-1]:.3f}" + (f", period {T:.2f}" if T else ""))

    # 2D medium (integrated above): a broken wave curls into a spiral
    if spiral is not None:
        n = spiral.shape[-1]
        print(f"spiral: {n}×{n} grid, h = {tensors.H}, D_x = 1, z ∈ [{spiral[-1].min():.3f}, {spiral[-1].max():.3f}]")
    return plot.animate(t, x, z, f_all, lo, hi, hopf, eps, q, spiral, out_dir / "belousov_zhabotinsky.gif")


if __name__ == "__main__":
    cli(run, __file__)
