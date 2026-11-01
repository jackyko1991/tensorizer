"""Build the concrete tensors: the (F, k) parameter pairs and the starting concentrations on the grid.

Grid units: cell size h = 1 and time step 1, with D_u = 0.2 and by default D_v = D_u / 2, the values of
Pearson's 1993 classification of Gray–Scott patterns (D_u / h² ≈ 0.2 there too).
"""

import torch

from .equation import growth_rate, jacobian, uniform_states

DU = 0.2  # substrate diffusion, cells² per step

# name, feed rate F, kill rate k
PAIRS = [
    ("spots", 0.0367, 0.0649),
    ("stripes", 0.0545, 0.062),
    ("maze", 0.029, 0.057),
    ("Turing", 0.060, 0.062),
]


def parameters(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """F_b and k_b, shape (B, 1, 1) each, so they broadcast over the (n, n) grid."""
    F = torch.tensor([f for _, f, _ in PAIRS], device=device)[:, None, None]
    k = torch.tensor([kk for _, _, kk in PAIRS], device=device)[:, None, None]
    return F, k


def turing_state(F: float, k: float, Du: float, Dv: float) -> tuple[float, float] | None:
    """The uniform state that is stable at q = 0 but unstable for some wavelength, if there is one."""
    q = torch.linspace(0, 1.5, 301)
    for u, v in uniform_states(F, k)[1:]:
        lam = growth_rate(jacobian(u, v, F, k), Du, Dv, q)
        if lam[0] < 0 < lam.max():
            return u, v
    return None


def initial_state(n: int, Du: float, Dv: float, device: torch.device, seed: int = 0) -> torch.Tensor:
    """y0 of shape (2, B, n, n).

    Where a Turing-unstable uniform state exists, start there with 1 % noise: the pattern has to grow out of
    nothing but the instability. Elsewhere the only stable uniform state is u = 1, v = 0, which small noise
    cannot destabilise, so a few square patches of v are dropped in to seed the pattern (a finite-amplitude start).
    """
    gen = torch.Generator().manual_seed(seed)
    u = torch.ones(len(PAIRS), n, n)
    v = torch.zeros(len(PAIRS), n, n)
    for b, (_, F, k) in enumerate(PAIRS):
        state = turing_state(F, k, Du, Dv)
        if state is not None:
            u[b], v[b] = state
            continue
        for _ in range(6):
            i, j = torch.randint(0, n - n // 12, (2,), generator=gen).tolist()
            u[b, i : i + n // 12, j : j + n // 12] = 0.5
            v[b, i : i + n // 12, j : j + n // 12] = 0.25
    y = torch.stack([u, v])
    y = y * (1 + 0.01 * torch.randn(y.shape, generator=gen))
    return y.to(device)
