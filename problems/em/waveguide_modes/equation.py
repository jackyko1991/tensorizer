r"""Physics → tensor translation.

Inside a hollow metal pipe of cross-section a × b, a wave travelling along z has the form
ψ(x, y) e^{i(βz − ωt)}. Maxwell's equations reduce to a 2D Helmholtz eigenproblem for ψ:

    (\partial_i \partial_i + k_c^2)\, \psi = 0,      i ∈ {x, y} summed,      β^2 = (ω/c)^2 - k_c^2

    TM modes: ψ = E_z, which must vanish on the metal walls       (Dirichlet:  ψ = 0)
    TE modes: ψ = H_z, whose normal derivative vanishes there      (Neumann:    ∂_n ψ = 0)

Below the cutoff frequency f_c = c k_c / 2π the mode cannot propagate (β is imaginary).

On an Nx × Ny grid the 1D second derivative is a tridiagonal matrix D. Flatten ψ_{ab} (a = x, b = y)
into ψ_m with m = a·Ny + b; then ∂_x² acts on the first index and ∂_y² on the second, a Kronecker sum:

    L = D_x ⊗ I_y + I_x ⊗ D_y,         L_{nm} ψ_m = -k_c^2 ψ_n       (m summed)

So the modes are the eigenvectors of the symmetric matrix −L and k_c² its eigenvalues: one call to
torch.linalg.eigh. The exact answer for a rectangle is k_c = π √((m/a)² + (n/b)²).
"""

import math

import torch


def second_difference(n: int, h: float, boundary: str) -> torch.Tensor:
    """D of shape (N, N): (ψ_{a−1} − 2ψ_a + ψ_{a+1}) / h² with the walls built in.

    dirichlet: grid nodes strictly inside the walls; the missing neighbours are ψ = 0, so they drop out.
    neumann:   grid cells centred half a cell from the walls; the missing neighbour mirrors ψ_a (zero slope),
               so the corner entries become −1 instead of −2.
    """
    D = torch.diag(-2.0 * torch.ones(n, dtype=torch.float64))
    D += torch.diag(torch.ones(n - 1, dtype=torch.float64), 1) + torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)
    if boundary == "neumann":
        D[0, 0] = D[-1, -1] = -1.0
    return D / h**2


def laplacian(nx: int, ny: int, hx: float, hy: float, boundary: str) -> torch.Tensor:
    """L = D_x ⊗ I_y + I_x ⊗ D_y of shape (Nx·Ny, Nx·Ny)."""
    Ix = torch.eye(nx, dtype=torch.float64)
    Iy = torch.eye(ny, dtype=torch.float64)
    return torch.kron(second_difference(nx, hx, boundary), Iy) + torch.kron(Ix, second_difference(ny, hy, boundary))


def modes(L: torch.Tensor, count: int, skip_zero: bool) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Solve −L_{nm} ψ_m = k_c² ψ_n.   Returns (k_c (K,), ψ (Nx·Ny, K)), lowest k_c first, orthonormal columns.

    skip_zero drops the constant Neumann solution (k_c = 0): a constant H_z carries no wave.
    """
    k2, psi = torch.linalg.eigh(-L)  # eigenvalues in ascending order
    start = 1 if skip_zero else 0
    return torch.sqrt(k2[start : start + count].clamp(min=0)), psi[:, start : start + count]


def exact_cutoff(m: int, n: int, a: float, b: float) -> float:
    """k_c = π √((m/a)² + (n/b)²) for the (m, n) mode of an a × b rectangle."""
    return math.pi * math.sqrt((m / a) ** 2 + (n / b) ** 2)


def transverse_E(psi: torch.Tensor, hx: float, hy: float, kind: str) -> torch.Tensor:
    """Direction of the transverse electric field, (..., Nx, Ny) → (..., 2, Nx, Ny).

    TM: E_t ∝ ∇E_z = (∂_x ψ, ∂_y ψ).   TE: E_t ∝ ẑ × ∇H_z = (−∂_y ψ, ∂_x ψ).
    """
    dx, dy = torch.gradient(psi, spacing=(hx, hy), dim=(-2, -1))
    return torch.stack([dx, dy], dim=-3) if kind == "TM" else torch.stack([-dy, dx], dim=-3)
