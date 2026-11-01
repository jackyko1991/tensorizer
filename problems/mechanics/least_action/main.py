"""Main loop: parameters → discrete action → wrong trial paths → minimise S → compare with the true path → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    m, g, omega = 1.0, 9.81, params["omega"]
    system, T, N = params["system"], params["T"], params["n_steps"]
    q_start, q_end = params["q_start"], params["q_end"]
    V, dV = equation.gravity(m, g) if system == "gravity" else equation.oscillator(m, omega)
    dt = T / N

    t = tensors.time_grid(T, N, dev)  # (N+1,)
    D = equation.difference_matrix(N, device=dev)  # (N, N+1)
    q0 = tensors.trial_paths(t, q_start, q_end, params["n_paths"], params["wiggle"])  # (B, N+1)
    exact = equation.exact_path(t, q_start, q_end, system, g, omega)  # (N+1,)
    S_exact = equation.action(exact[None], D, dt, m, V)[0]

    # Is the true path a minimum? Look at the smallest eigenvalue of the (tridiagonal) Hessian.
    H = solve.hessian(q0[:1], D, dt, m, V)[0]  # (N−1, N−1)
    lowest = torch.linalg.eigvalsh(H)[0].item()
    band = H.triu(2).abs().max().item()
    print(
        f"Hessian {tuple(H.shape)}: diagonal {H[0, 0]:.3f}, off-diagonal {H[0, 1]:.3f}, "
        f"beyond the tridiagonal band {band:.1e}; smallest eigenvalue {lowest:.4f}"
    )
    method = params["method"]
    if lowest <= 0:
        print(f"T = {T} s is longer than half a period π/ω = {math.pi / omega:.3f} s: the true path is a saddle of S,")
        print("not a minimum, so gradient descent would run away. Switching to Newton steps (stationary action).")
        method = "newton"

    hist = solve.relax(q0, D, dt, m, V, params["n_iter"], method)
    q_final = hist["paths"][-1]  # (B, N+1)
    resid = equation.euler_lagrange_residual(q_final, dt, m, dV)  # (B, N−1)
    print(f"{method}: {params['n_iter']} iterations, {q0.shape[0]} trial paths in one tensor")
    print(
        f"action of the true path S = {S_exact:.6f} J·s, after relaxing S_b = "
        + ", ".join(f"{s:.6f}" for s in hist["S"][-1].tolist())
    )
    print(
        f"max |q_b − q_exact| = {(q_final - exact).abs().max():.2e} m,  "
        f"max |∂S/∂q_n| = {hist['grad'][-1].abs().max():.2e},  "
        f"max |discrete Euler–Lagrange residual| = {resid.abs().max():.2e}"
    )
    return plot.animate(t, hist, exact, S_exact, out_dir / "least_action.gif")


if __name__ == "__main__":
    cli(run, __file__)
