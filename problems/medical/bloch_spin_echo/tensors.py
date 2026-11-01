"""Build the concrete tensors: Levi-Civita symbol, relaxation matrix, off-resonance spread and the pulse list."""

import math

import torch


def levi_civita(device: torch.device) -> torch.Tensor:
    """ε_{ijk}: +1 for even permutations of (x, y, z), −1 for odd ones, 0 otherwise. Shape (3, 3, 3)."""
    eps = torch.zeros(3, 3, 3, dtype=torch.float64, device=device)
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[i, j, k] = 1.0
        eps[j, i, k] = -1.0
    return eps


def relaxation(T1: float, T2: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """R_{ij} = diag(1/T2, 1/T2, 1/T1) (3, 3) and the equilibrium M^0_j = (0, 0, 1) (3,). Times in ms."""
    R = torch.diag(torch.tensor([1 / T2, 1 / T2, 1 / T1], dtype=torch.float64, device=device))
    M0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)
    return R, M0


def off_resonance(n: int, t2_prime: float, kind: str, device: torch.device) -> torch.Tensor:
    """Δω_n (rad/ms) for n isochromats, spread by field inhomogeneity with width 1/T2'. Shape (N,).

    The spins sit at evenly spaced quantiles u_n of the line shape (deterministic, no random numbers):
    Lorentzian Δω = tan(π(u − ½))/T2' gives an exponential FID e^{−t/T2'}; Gaussian gives e^{−t²/2T2'²}.
    """
    u = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n  # (N,) in (0, 1)
    width = 1.0 / t2_prime
    if kind == "gaussian":
        return width * math.sqrt(2.0) * torch.erfinv(2 * u - 1)
    u = 0.01 + 0.98 * u  # clip the far Lorentzian tails
    return width * torch.tan(math.pi * (u - 0.5))


def equilibrium_state(n: int, device: torch.device) -> torch.Tensor:
    """M_{ni} = (0, 0, 1) for every isochromat: all magnetisation along the main field. Shape (N, 3)."""
    M = torch.zeros(n, 3, dtype=torch.float64, device=device)
    M[:, 2] = 1.0
    return M


def pulse_list(TE: float, n_echoes: int, duration: float) -> list[tuple[float, float, float, float]]:
    """(start, length, flip angle, phase) of every RF pulse, times in ms.

    A 90° pulse about x at t = 0, then refocusing 180° pulses about y (CPMG) centred at TE/2, 3TE/2, ...
    measured from the centre of the 90° pulse, so echo k forms at k·TE after it.
    """
    pulses = [(0.0, duration, math.pi / 2, 0.0)]
    t90 = duration / 2  # centre of the excitation
    for k in range(n_echoes):
        centre = t90 + (k + 0.5) * TE
        pulses.append((centre - duration / 2, duration, math.pi, math.pi / 2))
    return pulses
