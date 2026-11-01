"""Build the concrete tensors: radial grid, second-difference matrix and the list of (l, j) channels."""

import torch

L_LETTERS = "spdfghijk"


def radial_grid(n: int, r_max: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """Interior points r_i = h … r_max − h (u = 0 at r = 0 and r = r_max): (N,), spacing h."""
    h = r_max / (n + 1)
    return torch.arange(1, n + 1, dtype=torch.float64, device=device) * h, h


def second_difference(n: int, h: float, device: torch.device) -> torch.Tensor:
    """D_ij u_j ≈ u''(r_i): the tridiagonal (1, −2, 1)/h² matrix, shape (N, N)."""
    main = torch.full((n,), -2.0, dtype=torch.float64, device=device)
    off = torch.ones(n - 1, dtype=torch.float64, device=device)
    return (torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)) / h**2


def channels(l_max: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (l, j) with j = l ± ½ (only j = ½ for l = 0): two tensors of shape (K,)."""
    ls, js = [], []
    for ell in range(l_max + 1):
        for j in [ell + 0.5, ell - 0.5] if ell > 0 else [0.5]:
            ls.append(float(ell))
            js.append(j)
    as_tensor = {"dtype": torch.float64, "device": device}
    return torch.tensor(ls, **as_tensor), torch.tensor(js, **as_tensor)


def label(n: int, ell: int, j: float) -> str:
    """Spectroscopic name such as 1g9/2 (n counts levels with the same l and j from 1)."""
    return f"{n}{L_LETTERS[ell]}{int(2 * j)}/2"
