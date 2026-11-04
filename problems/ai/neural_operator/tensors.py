"""Build the concrete tensors: periodic grids, random initial conditions and the FNO weights."""

import math

import torch

LENGTH = 2 * math.pi  # periodic domain [0, 2π)


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """N equally spaced points x on [0, 2π) (float64) and the spacing Δx."""
    dx = LENGTH / n
    return torch.arange(n, device=device, dtype=torch.float64) * dx, dx


def random_modes(n_samples: int, n_modes: int, seed: int, device: torch.device) -> torch.Tensor:
    """Random Fourier coefficients c_{bk} = (a_bk, b_bk) for k = 1 … n_modes: shape (B, n_modes, 2).

    Amplitudes fall off like 1/k so the initial conditions are smooth; the mean (k = 0) is zero.
    """
    gen = torch.Generator().manual_seed(seed)
    k = torch.arange(1, n_modes + 1, dtype=torch.float64)
    coeffs = torch.randn(n_samples, n_modes, 2, generator=gen, dtype=torch.float64) * 0.6 / k[:, None]
    return coeffs.to(device)


def initial_conditions(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """u_0(x) = Σ_k a_k cos kx + b_k sin kx, sampled on the grid x: (B, K, 2), (N,) → (B, N).

    The same coefficients evaluated on a finer grid give the SAME functions at higher resolution.
    """
    k = torch.arange(1, coeffs.shape[1] + 1, device=x.device, dtype=x.dtype)
    kx = k[:, None] * x[None, :]  # (K, N)
    return torch.einsum("bk,kn->bn", coeffs[..., 0], torch.cos(kx)) + torch.einsum(
        "bk,kn->bn", coeffs[..., 1], torch.sin(kx)
    )


def init_fno(width: int, n_modes: int, n_layers: int, seed: int, device: torch.device) -> dict:
    """FNO weights (all trainable):

    P (C, 2), p (C,)                        lift: [u_0, x] → C channels
    R[l] (C, C, m) complex, W[l] (C, C), b[l] (C,)   Fourier layer l = 0 … L-1
    Q1 (H, C), q1 (H,), Q2 (H,), q2 ()      projection: C → H = 2C → 1
    """
    gen = torch.Generator().manual_seed(seed)

    def dense(*shape):  # uniform ±1/√fan_in, like torch.nn.Linear
        fan_in = shape[-1] if len(shape) > 1 else width
        return (torch.rand(*shape, generator=gen) * 2 - 1) / math.sqrt(fan_in)

    def spectral():  # small random complex weights, as in the original FNO paper
        scale = 1.0 / (width * width)
        return scale * torch.complex(
            torch.rand(width, width, n_modes, generator=gen), torch.rand(width, width, n_modes, generator=gen)
        )

    hidden = 2 * width
    w = {
        "P": dense(width, 2),
        "p": dense(width, 2)[:, 0],
        "R": [spectral() for _ in range(n_layers)],
        "W": [dense(width, width) for _ in range(n_layers)],
        "b": [dense(width, width)[:, 0] for _ in range(n_layers)],
        "Q1": dense(hidden, width),
        "q1": dense(hidden, width)[:, 0],
        "Q2": dense(1, hidden)[0],
        "q2": torch.zeros(()),
    }
    return {
        key: [t.to(device).requires_grad_(True) for t in val]
        if isinstance(val, list)
        else val.to(device).requires_grad_(True)
        for key, val in w.items()
    }


def parameters(w: dict) -> list[torch.Tensor]:
    """Flat list of every trainable tensor (for the optimiser)."""
    return [t for val in w.values() for t in (val if isinstance(val, list) else [val])]


def count(w: dict) -> int:
    """Number of real trainable numbers (a complex weight counts twice)."""
    return sum(t.numel() * (2 if t.is_complex() else 1) for t in parameters(w))
