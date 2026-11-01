"""Build the concrete tensors: the (Z, N) grid, the coefficients and a few measured nuclei."""

import torch


def grid(z_max: int, n_max: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Z as a column (Z, 1) and N as a row (1, N), so any formula in Z and N broadcasts to (Z, N)."""
    z = torch.arange(1, z_max + 1, dtype=torch.float64, device=device)[:, None]
    n = torch.arange(0, n_max + 1, dtype=torch.float64, device=device)[None, :]
    return z, n


def coefficients(params: dict) -> dict[str, float]:
    """The five liquid-drop coefficients in MeV."""
    return {key: params[key] for key in ("a_V", "a_S", "a_C", "a_A", "a_P")}


# (name, Z, A, measured B/A in MeV) — from the atomic mass evaluation, for comparison only
MEASURED = [
    ("²H", 1, 2, 1.112),
    ("⁴He", 2, 4, 7.074),
    ("¹²C", 6, 12, 7.680),
    ("¹⁶O", 8, 16, 7.976),
    ("⁵⁶Fe", 26, 56, 8.790),
    ("⁶²Ni", 28, 62, 8.795),
    ("¹²⁰Sn", 50, 120, 8.505),
    ("²⁰⁸Pb", 82, 208, 7.868),
    ("²³⁸U", 92, 238, 7.570),
]
