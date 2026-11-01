r"""Physics → tensor translation.

Gauss's law with E = −∇φ gives Poisson's equation for the potential,

    \nabla^2 \phi = -\rho / \varepsilon_0,

in index notation (i is repeated, so it is summed over x and y):

    \partial_i \partial_i \phi = -\rho / \varepsilon_0,      E_i = -\partial_i \phi

On an N × N grid with spacing h the 1D second derivative is a tridiagonal matrix

    D_{ab} φ_b = (φ_{a-1} - 2 φ_a + φ_{a+1}) / h²

Flatten the grid φ_{ab} (row a = y, column b = x) into one vector φ_m with m = a·N + b.
Then ∂_y² acts on the row index and ∂_x² on the column index, which is a Kronecker product:

    L = D ⊗ I + I ⊗ D,        L_{nm} φ_m = -ρ_n / ε_0

The repeated m is summed: one big matrix–vector product, solved for φ in one call.
"""

import torch


def second_difference(n: int, h: float, dtype=torch.float64) -> torch.Tensor:
    """D of shape (N, N): -2 on the diagonal, 1 on both neighbours, divided by h².

    Missing neighbours at the ends are the grounded walls (φ = 0), so they simply drop out.
    """
    main = -2.0 * torch.ones(n, dtype=dtype)
    side = torch.ones(n - 1, dtype=dtype)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / h**2


def laplacian(n: int, h: float) -> torch.Tensor:
    """L = D ⊗ I + I ⊗ D of shape (N², N²): ∂_y² on rows plus ∂_x² on columns."""
    D = second_difference(n, h)
    I = torch.eye(n, dtype=D.dtype)  # noqa: E741
    return torch.kron(D, I) + torch.kron(I, D)


def apply_laplacian(L: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    r"""(∂_i∂_i φ)_n = L_{nm} φ_m.   phi: (N, N) → (N, N)"""
    n = phi.shape[0]
    return torch.einsum("nm,m->n", L, phi.reshape(-1)).reshape(n, n)


def electric_field(phi: torch.Tensor, h: float) -> torch.Tensor:
    """E_i = -∂_i φ with central differences; the grounded walls (φ = 0) are padded on.  → (2, N, N): (E_x, E_y)"""
    padded = torch.nn.functional.pad(phi, (1, 1, 1, 1))  # (N+2, N+2) walls at φ = 0
    E_x = -(padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * h)  # column index b is x
    E_y = -(padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * h)  # row index a is y
    return torch.stack([E_x, E_y])
