"""Build the concrete tensors: worldsheet grids and the oscillator amplitudes α_n, α̃_n of a chosen state.

Units: α′ = 1 (string length √α′ = 1). Only three of the 24 transverse directions are excited, so the string can be
drawn in 3D; the other 21 sit at zero.
"""

import math

import torch

ALPHA_PRIME = 1.0  # Regge slope α′ (sets the string tension T = 1/(2πα′))
TRANSVERSE = 24  # D − 2 transverse directions in the critical dimension D = 26


def sigma_grid(n: int, kind: str, device: torch.device) -> torch.Tensor:
    """Points along the string: [0, 2π) for a closed loop (periodic), [0, π] for an open string.   (S,)"""
    if kind == "closed":
        return torch.arange(n, dtype=torch.float64, device=device) * (2 * math.pi / n)
    return torch.linspace(0, math.pi, n, dtype=torch.float64, device=device)


def tau_grid(n: int, device: torch.device) -> torch.Tensor:
    """One full period of worldsheet time, τ ∈ [0, 2π].   (T,)"""
    return torch.linspace(0, 2 * math.pi, n, dtype=torch.float64, device=device)


def oscillators(n_a: int, n_b: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Amplitudes (α_{nd}, α̃_{nd}), each (max(n_a, n_b), 3) complex.

    closed: right movers in mode n_a, circularly polarised in the (1, 2) plane, and left movers in mode n_b in the
            (2, 3) plane. Each side gets level N = lcm(n_a, n_b), i.e. N / n quanta of mode n, so N = Ñ
            (level matching).
    open:   one quantum in mode n_a (|α|² = n_a) plus one in mode n_b (|α|² = n_b): N = n_a + n_b. α̃ = 0.
    """
    modes = max(n_a, n_b)
    alpha = torch.zeros(modes, 3, dtype=torch.complex128)
    alpha_t = torch.zeros(modes, 3, dtype=torch.complex128)
    plane_a = torch.tensor([1, 1j, 0], dtype=torch.complex128) / math.sqrt(2)  # unit circular polarisation
    plane_b = torch.tensor([0, 1, 1j], dtype=torch.complex128) / math.sqrt(2)
    if kind == "closed":
        level = math.lcm(n_a, n_b)
        alpha[n_a - 1] = math.sqrt(level) * plane_a
        alpha_t[n_b - 1] = math.sqrt(level) * plane_b
    else:
        alpha[n_a - 1] += math.sqrt(n_a) * plane_a
        alpha[n_b - 1] += math.sqrt(n_b) * plane_b
    return alpha, alpha_t


def radii(r_min: float, r_max: float, n: int, device: torch.device) -> torch.Tensor:
    """Log-spaced compactification radii R (units of √α′), symmetric about the self-dual radius when r_min = 1/r_max."""
    return torch.logspace(math.log10(r_min), math.log10(r_max), n, dtype=torch.float64, device=device)
