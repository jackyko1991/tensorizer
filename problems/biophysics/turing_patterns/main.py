"""Main loop: (F, k) pairs → linear stability (dispersion relation) → batched simulation → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import growth_rate, jacobian, uniform_states


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    Du = tensors.DU
    Dv = params["ratio"] * Du
    n = 48 if fast else params["grid"]
    n_steps = 300 if fast else params["steps"]
    names = [name for name, _, _ in tensors.PAIRS]
    F, k = tensors.parameters(dev)  # (B, 1, 1) each
    print(f"Gray–Scott on a {n}×{n} periodic grid, D_u = {Du}, D_v = {Dv:.3f}, {len(names)} (F, k) pairs at once")

    # --- linear stability of the uniform states: the dispersion relation λ(q) ---
    q = torch.linspace(0, 1.2, 241, dtype=torch.float64)
    curves = {}  # name → list of (state, λ(q))
    for name, f, kk in tensors.PAIRS:
        curves[name] = []
        states = uniform_states(f, kk)
        for u, v in states[1:]:
            lam = growth_rate(jacobian(u, v, f, kk), Du, Dv, q)
            curves[name].append(((u, v), lam))
        print(f"{name:8s} F = {f:.4f}, k = {kk:.4f}: " + _verdict(states, curves[name], q))

    y0 = tensors.initial_state(n, Du, Dv, dev)  # (2, B, n, n)
    at, frames, amp = solve.solve(y0, F, k, Du, Dv, n_steps)
    wl = solve.dominant_wavelength(frames[-1])
    for b, name in enumerate(names):
        print(f"{name:8s} after {n_steps} steps: pattern wavelength ≈ {wl[b].item():.1f} cells")

    grid_F = torch.linspace(0.02, 0.09, 12 if fast else 71)
    grid_k = torch.linspace(0.054, 0.068, 12 if fast else 71)
    classes = solve.stability_map(grid_F, grid_k, Du, Dv)
    return plot.animate(
        at, frames, amp, tensors.PAIRS, curves, q, grid_F, grid_k, classes, out_dir / "turing_patterns.gif"
    )


def _verdict(states: list, curves: list, q: torch.Tensor) -> str:
    if len(states) == 1:
        return "only u = 1, v = 0 (stable for every q): patterns need a finite seed"
    parts = []
    for (u, v), lam in curves:
        i = int(lam.argmax())
        kind = "Turing-unstable" if lam[0] < 0 < lam[i] else ("unstable at q = 0" if lam[0] > 0 else "stable")
        text = f"(u, v) = ({u:.3f}, {v:.3f}) {kind}"
        if lam[0] < 0 < lam[i]:
            text += f", fastest q = {q[i]:.3f} (λ = {lam[i]:.4f}/step, wavelength {2 * math.pi / q[i]:.1f} cells)"
        parts.append(text)
    return "; ".join(parts)


if __name__ == "__main__":
    cli(run, __file__)
