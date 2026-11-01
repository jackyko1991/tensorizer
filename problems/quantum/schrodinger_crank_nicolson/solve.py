"""March the packets with Crank–Nicolson (LU factorised once); compare with explicit Euler and the exact e^{−iHt}."""

import torch

from problems._common.progress import steps

from .equation import cayley_matrices, cn_step, cn_step_solve, euler_step, norm


def crank_nicolson(
    H: torch.Tensor, psi0: torch.Tensor, dt: float, t_max: float, n_saved: int = 240, use_lu: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), ψ (T, N, B) complex), keeping about n_saved snapshots.

    A and B never change, so A is LU-factorised once (O(N³)); every step is then a matrix–vector product
    and two triangular solves (O(N²) per packet). use_lu=False calls torch.linalg.solve every step instead.
    """
    n_steps = max(1, int(round(t_max / dt)))
    save_every = max(1, n_steps // n_saved)
    A, B = cayley_matrices(H, dt)  # (N, N) each
    lu, pivots = torch.linalg.lu_factor(A)

    psi, snapshots, times = psi0, [psi0], [0.0]
    for step in steps(n_steps, "Crank–Nicolson steps"):
        psi = cn_step(lu, pivots, B, psi) if use_lu else cn_step_solve(A, B, psi)
        if (step + 1) % save_every == 0:
            snapshots.append(psi)
            times.append((step + 1) * dt)
    return torch.tensor(times, dtype=torch.float64), torch.stack(snapshots)


def euler_norms(
    H: torch.Tensor, psi0: torch.Tensor, dx: float, dt: float, t_max: float, cap: float = 1e12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit Euler with the same Δt; return (times (S,), norm (S, B)), stopping once the norm passes cap."""
    n_steps = max(1, int(round(t_max / dt)))
    psi, norms, times = psi0, [norm(psi0, dx)], [0.0]
    for step in range(n_steps):
        psi = euler_step(H, psi, dt)
        norms.append(norm(psi, dx))
        times.append((step + 1) * dt)
        if norms[-1].max() > cap:
            break
    return torch.tensor(times, dtype=torch.float64), torch.stack(norms)


def exact(H: torch.Tensor, psi0: torch.Tensor, t: float) -> torch.Tensor:
    """Reference ψ(t) = e^{−iHt} ψ0 = φ e^{−iEt} φᵀ ψ0 from torch.linalg.eigh. (N, B) → (N, B)"""
    E, phi = torch.linalg.eigh(H)  # (N,), (N, N)
    phi = phi.to(psi0.dtype)
    c = torch.einsum("xn,xb->nb", phi, psi0)  # components in the eigenbasis (phi is real orthogonal)
    return torch.einsum("xn,nb->xb", phi, torch.exp(-1j * E * t)[:, None] * c)
