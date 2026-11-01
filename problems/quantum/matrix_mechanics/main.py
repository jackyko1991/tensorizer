"""Main loop: parameters → matrices → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import commutator, dagger, hamiltonian, number_hamiltonian


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    N, alpha, lam = params["n_states"], params["alpha"], params["lam"]
    a = tensors.lowering(N, dev)  # (N, N)
    X, P = tensors.position(a), tensors.momentum(a)  # (N, N) each
    H = hamiltonian(X, P, lam)  # (N, N)
    C = commutator(X, P)  # (N, N) ≈ i·I

    E = solve.energies(H)
    E0 = solve.energies(hamiltonian(X, P))  # λ = 0 ladder, for the console table
    exact_H = number_hamiltonian(a)
    print(f"N = {N} states.  H Hermitian: {torch.allclose(H, dagger(H))}")
    print(
        f"lambda = 0:  max |1/2(P^2 + X^2) - (a^dag a + 1/2)| over the top-left {N - 1}x{N - 1} block = "
        f"{(hamiltonian(X, P) - exact_H)[:-1, :-1].abs().max().item():.1e}"
    )
    print(f"[X, P] diagonal / i: {C.diagonal().imag[:3].tolist()} ... last = {C[-1, -1].imag.item():.1f} (= 1 - N)")
    n = torch.arange(N, dtype=torch.float64, device=dev)
    pert = solve.first_order_energies(n, lam)
    for k in range(4):
        print(
            f"E_{k} = {E[k].item():9.5f}   harmonic {E0[k].item():.5f} (n + 1/2 = {k + 0.5})   "
            f"1st-order PT {pert[k].item():9.5f}"
        )
    print(
        f"truncation artefact: last diagonal entry of 1/2(P^2 + X^2) = {hamiltonian(X, P).real[-1, -1].item():.1f}"
        f" = (N - 1)/2 instead of N - 1/2 = {exact_H.real[-1, -1].item():.1f}"
    )

    times = torch.linspace(0.0, params["periods"] * 2 * math.pi, 241, dtype=torch.float64, device=dev)
    U = solve.propagators(H, times)  # (T, N, N)
    c = tensors.coherent_state(complex(alpha, 0.0), N, dev)  # (N,)
    ops = {"X": X, "P": P}
    heis, moving = solve.heisenberg_picture(U, ops, c)
    schr = solve.schrodinger_picture(U, ops, c)
    gap = max((heis[k] - schr[k]).abs().max().item() for k in ops)
    classical = math.sqrt(2) * alpha * torch.cos(times)
    print(f"max |<X>_Heisenberg - <X>_Schrodinger| = {gap:.1e}   (the two pictures agree)")
    print(f"max |<X> - sqrt(2) Re(alpha e^-it)| = {(heis['X'] - classical).abs().max().item():.1e}")
    return plot.animate(times, moving["X"], C, heis, schr, alpha, lam, out_dir / "matrix_mechanics.gif")


if __name__ == "__main__":
    cli(run, __file__)
