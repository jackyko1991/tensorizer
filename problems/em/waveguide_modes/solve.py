"""Solve both eigenproblems, merge the TE and TM modes and sort them by cutoff."""

import torch

from . import tensors
from .equation import exact_cutoff, laplacian, modes, transverse_E


def solve(a: float, b: float, n_b: int, n_modes: int) -> list[dict]:
    """Return the n_modes lowest modes as dicts with keys kind, mn, k_c, k_exact, psi (Nx, Ny), E_t (2, Nx, Ny).

    psi is normalised so that Σ_m ψ_m² = 1 (eigh returns orthonormal eigenvectors).
    """
    nx, ny = tensors.grid_size(a, b, n_b)
    found = []
    for kind, boundary in [("TM", "dirichlet"), ("TE", "neumann")]:
        _, _, hx, hy = tensors.grid(a, b, nx, ny, kind)
        L = laplacian(nx, ny, hx, hy, boundary)  # (Nx·Ny, Nx·Ny)
        k_c, psi = modes(L, n_modes, skip_zero=(kind == "TE"))  # (K,), (Nx·Ny, K)
        for k, column in zip(k_c.tolist(), psi.T):
            pattern = column.reshape(nx, ny)  # back from ψ_m to ψ_{ab}
            # name the mode after the exact (m, n) with the nearest cutoff
            mn = min(tensors.candidate_labels(kind), key=lambda p: abs(exact_cutoff(*p, a, b) - k))
            E_t = transverse_E(pattern, hx, hy, kind)
            found.append(
                {"kind": kind, "mn": mn, "k_c": k, "k_exact": exact_cutoff(*mn, a, b), "psi": pattern, "E_t": E_t}
            )
    return sorted(found, key=lambda mode: mode["k_c"])[:n_modes]


def cutoff_frequency_GHz(k_c: float) -> float:
    """f_c = c k_c / 2π with k_c in 1/cm and c = 3 × 10¹⁰ cm/s, in GHz."""
    return 3e10 * k_c / (2 * torch.pi) / 1e9
